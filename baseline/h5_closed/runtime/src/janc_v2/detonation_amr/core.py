"""GPU-resident JANC WENO/chemistry with bounded JAX-AMR block topology.

Reusable detonation AMR integration; laboratory non-AMR solvers are unchanged.
Conserved restriction, face-flux synchronization and time-interpolated parent
ghosts are explicit here. Scientific acceptance still requires convergence.
One Engine per fresh process: upstream thermochemistry/AMR use module globals.
"""
from dataclasses import asdict, dataclass
from functools import partial
import json
from pathlib import Path

import cantera as ct
import jax
import jax.numpy as jnp
import numpy as np
from jaxamr import amr
from janc_v2.model import thermo_model as thermo, reaction_model as reaction
from janc_v2.solver_2D import aux_func, flux

SOLVER_REVISION = '20261022-shared-parent-face-reflux-v2'


@dataclass
class Config:
    nx: int = 4096
    ny: int = 128
    lx: float = 1.0
    ly: float = .02
    levels: int = 3
    patch_x: int = 64
    patch_y: int = 32
    capacities: tuple = (64, 128, 256)
    block_batch: int = 4
    chem_cells: int = 2048
    cfl: float = .20
    dt_max: float = 2e-8
    amr_every: int = 4
    p0: float = 50662.5
    t0: float = 300.
    phi: float = 1.
    hot_m: float = .002
    hot_t: float = 3500.
    hot_p: float = 7599375.
    density_jump: float = .04
    pressure_jump: float = .04
    temperature_jump: float = .03
    species_jump: float = .02
    buffer: int = 4

    def validate(self):
        if self.nx % self.patch_x or self.ny % self.patch_y:
            raise ValueError('Base grid must be divisible by patch dimensions')
        if min(self.patch_x, self.patch_y) < 6 or self.levels < 0:
            raise ValueError('WENO5 needs patches at least 6 cells wide')
        if len(self.capacities) != self.levels or any(x < 2 for x in self.capacities):
            raise ValueError('One positive fixed capacity per fine level, including a sentinel')
        if self.block_batch < 1 or self.chem_cells < 1 or not 0 < self.cfl <= .25:
            raise ValueError('Invalid batching/CFL')
        if self.amr_every < 1 or self.buffer < 3 or self.dt_max <= 0:
            raise ValueError('Invalid time/refinement settings')

    def memory_estimate_gib(self):
        # Budget, NOT an XLA allocation guarantee. Leaves room for compilation.
        cells = self.nx*self.ny + sum(self.capacities)*4*self.patch_x*self.patch_y
        return 16 * cells * 14 * 8 / 2**30 + 2.


def pressure(s):
    return s[0] * thermo.get_R(s[4:-2]/s[0:1])[0] * s[-1]


def refresh(U, aux):
    return jnp.concatenate((U, aux_func.update_aux(U, aux)), axis=0)


def physical_ghost(s):
    """Three reflecting slip walls; right zero-gradient outflow. x is axis 1."""
    g = jnp.pad(s, ((0,0),(3,3),(3,3)), mode='edge')
    left = s[:, :3, :][:, ::-1, :].at[1].multiply(-1)
    bottom = s[:, :, :3][:, :, ::-1].at[2].multiply(-1)
    top = s[:, :, -3:][:, :, ::-1].at[2].multiply(-1)
    g = g.at[:, :3, 3:-3].set(left)
    g = g.at[:, 3:-3, :3].set(bottom)
    return g.at[:, 3:-3, -3:].set(top)


def tiled_chemistry(s, dt, cells):
    """Exact same point-local source formula, never a CPU callback or T cutoff."""
    shape = s.shape
    flat = s.reshape(shape[0], -1)
    pad = (-flat.shape[1]) % cells
    blocks = jnp.pad(flat, ((0,0),(0,pad)), mode='edge').reshape(shape[0], -1, cells)
    blocks = blocks.transpose(1,0,2)[..., None]

    def one(b):
        U, aux = b[:-2], b[-2:]
        return refresh(U + reaction.reaction_source_terms(U, aux, dt, None), aux)

    out = jax.lax.map(one, blocks).squeeze(-1).transpose(1,0,2).reshape(shape[0], -1)
    return out[:, :flat.shape[1]].reshape(shape)


def interface_flux(g, alpha):
    """Same WENO5-JS/LF reconstruction; one wave bound per level, not per block.

    A shared LF bound makes a shared block face have the same numerical flux
    from either side; that is needed for conservation across block boundaries.
    """
    U, aux = g[:-2], g[-2:]
    rho, u, v, Y, p, a = aux_func.U_to_prim(U, aux)
    F = jnp.concatenate((rho*u, rho*u*u+p, rho*u*v, u*(U[3:4]+p), rho*u*Y))
    G = jnp.concatenate((rho*v, rho*u*v, rho*v*v+p, v*(U[3:4]+p), rho*v*Y))
    rx, ry = flux.reconstruction_L_x_dict['WENO5_JS'], flux.reconstruction_L_y_dict['WENO5_JS']
    sx, sy = flux.reconstruction_R_x_dict['WENO5_JS'], flux.reconstruction_R_y_dict['WENO5_JS']
    return (rx(.5*(F+alpha[0]*U)) + sx(.5*(F-alpha[0]*U)),
            ry(.5*(G+alpha[1]*U)) + sy(.5*(G-alpha[1]*U)))


class BaseEngine:
    def __init__(self, config, mechanism):
        config.validate()
        self.c = config
        thermo.set_thermo({'thermo_model':'nasa7', 'mechanism_directory':str(mechanism)}, None, '2D')
        reaction.set_reaction({'is_detailed_chemistry':True, 'mechanism_directory':str(mechanism)}, None, '2D')
        gas = ct.Solution(str(mechanism))
        if gas.species_names != ['H2','O2','OH','O','H','H2O','HO2','H2O2','N2']:
            raise ValueError('Unexpected mechanism/species ordering')
        gas.TP = config.t0, config.p0
        gas.set_equivalence_ratio(config.phi, 'H2:1', 'O2:1,N2:3.76')
        self.Y0 = gas.Y[:-1].copy()
        self.s0 = self.uniform(config.t0, config.p0, 1, 1)[:,0,0]
        self.nblock = [[1,1]] + ([[config.nx//config.patch_x, config.ny//config.patch_y]]
                       + [[2,2]]*(config.levels-1) if config.levels else [])
        amr.set_amr({'base_grid':{'Lx':config.lx,'Ly':config.ly,'Nx':config.nx,'Ny':config.ny},
                     'n_block':self.nblock, 'template_node_num':3, 'buffer_num':config.buffer,
                     'refinement_tolerance':{}})
        self.dx = [config.lx/config.nx/2**l for l in range(config.levels+1)]
        self.dy = [config.ly/config.ny/2**l for l in range(config.levels+1)]
        self.stage = jax.jit(self._stage)
        self.chem = jax.jit(self._chem)
        self.ghost = jax.jit(self._ghost, static_argnums=0)
        self.restrict = jax.jit(self._restrict)
        self.sync_flux = jax.jit(self._sync_flux)
        self.metrics = jax.jit(self._metrics)
        self.probes = jax.jit(self._probes)
        self.cycle = jax.jit(self._cycle)  # old state retained for step rejection
        self.mask = jax.jit(self._mask, static_argnums=0)
        self._fine = jax.jit(self._build_fine, static_argnums=0)

    def uniform(self, T, p, nx, ny):
        Y = jnp.broadcast_to(jnp.asarray(self.Y0)[:,None,None], (8,nx,ny))
        T = jnp.broadcast_to(jnp.asarray(T), (1,nx,ny))
        p = jnp.broadcast_to(jnp.asarray(p), (1,nx,ny))
        _, gamma, h, R, _ = thermo.get_thermo(T, Y)
        rho = p/(R*T)
        return jnp.concatenate((rho, jnp.zeros_like(rho), jnp.zeros_like(rho), rho*h-p, rho*Y, gamma, T))

    def initial(self):
        c = self.c
        hot = ((jnp.arange(c.nx)+.5)*self.dx[0] < c.hot_m)[None,:,None]
        base = self.uniform(jnp.where(hot,c.hot_t,c.t0), jnp.where(hot,c.hot_p,c.p0), c.nx,c.ny)[None]
        states = [base]
        infos = [dict(number=jnp.array(1), index=jnp.array([[0,0,0]]),
                      glob_index=jnp.array([[0,0]]), neighbor_index=jnp.full((1,4),-1))]
        for l in range(1,c.levels+1):
            data, info = self.regrid_level(l, states, infos)
            states.append(data)
            infos.append(info)
        return tuple(states), tuple(infos)

    def _ghost(self, level, s, info, parent_g=None):
        if level == 0:
            return physical_ghost(s[0])[None]
        px, py = self.c.patch_x, self.c.patch_y
        nx, ny = s.shape[-2:]
        index = info['index']
        ix = jnp.arange(-3,nx+3)//2
        iy = jnp.arange(-3,ny+3)//2

        def one(i):
            b,r,c = jnp.maximum(index[i],0)
            g = parent_g[b][:, (r*px+ix+3)[:,None], (c*py+iy+3)[None,:]]
            g = g.at[:,3:-3,3:-3].set(s[i])
            n = info['neighbor_index'][i]
            safe = jnp.maximum(n,0)
            g = g.at[:,:3,3:-3].set(jnp.where(n[0]>=0, s[safe[0],:,-3:,:], g[:,:3,3:-3]))
            g = g.at[:,-3:,3:-3].set(jnp.where(n[1]>=0, s[safe[1],:,:3,:], g[:,-3:,3:-3]))
            g = g.at[:,3:-3,:3].set(jnp.where(n[2]>=0, s[safe[2],:,:,-3:], g[:,3:-3,:3]))
            g = g.at[:,3:-3,-3:].set(jnp.where(n[3]>=0, s[safe[3],:,:,:3], g[:,3:-3,-3:]))
            coord = jnp.zeros(2, dtype=jnp.int64)
            total = np.ones(2,dtype=int)
            for k in range(level+1):
                coord = coord*jnp.asarray(self.nblock[k]) + info['glob_index'][i,2*k:2*k+2]
                total *= self.nblock[k]
            wall = physical_ghost(s[i])
            g = g.at[:,:3,3:-3].set(jnp.where(coord[0]==0,wall[:,:3,3:-3],g[:,:3,3:-3]))
            g = g.at[:,-3:,3:-3].set(jnp.where(coord[0]==total[0]-1,wall[:,-3:,3:-3],g[:,-3:,3:-3]))
            g = g.at[:,3:-3,:3].set(jnp.where(coord[1]==0,wall[:,3:-3,:3],g[:,3:-3,:3]))
            return g.at[:,3:-3,-3:].set(jnp.where(coord[1]==total[1]-1,wall[:,3:-3,-3:],g[:,3:-3,-3:]))

        return jax.vmap(one)(jnp.arange(s.shape[0]))

    def _stage(self, g, old, info, dx, dy, dt, w0, w1):
        batch = min(self.c.block_batch, old.shape[0])
        count, nv, nx, ny = old.shape
        pad = (-count) % batch
        gp = jnp.pad(g, ((0,pad),(0,0),(0,0),(0,0)), mode='edge')
        op = jnp.pad(old, ((0,pad),(0,0),(0,0),(0,0)), mode='edge')
        valid = (jnp.arange(count) < info['number'])[:,None,None]
        # Include ghost states as well as interiors in the LF wave bound.
        def speed(a):
            rho,u,v,Y,p,cs = aux_func.U_to_prim(a[:-2],a[-2:])
            return jnp.max(jnp.abs(u)+cs), jnp.max(jnp.abs(v)+cs)
        ax, ay = jax.vmap(speed)(g)
        alpha = jnp.array([jnp.max(jnp.where(valid[:,0,0],ax,0)),jnp.max(jnp.where(valid[:,0,0],ay,0))])
        fx = jnp.zeros((count+pad,nv-2,nx+1,ny))
        fy = jnp.zeros((count+pad,nv-2,nx,ny+1))

        def body(k, carry):
            out, fx, fy = carry
            start = k*batch
            gb = jax.lax.dynamic_slice_in_dim(gp,start,batch)
            ob = jax.lax.dynamic_slice_in_dim(op,start,batch)
            def one(gg, oo):
                F,G = interface_flux(gg,alpha)
                rhs = -(F[:,1:]-F[:,:-1])/dx - (G[:,:,1:]-G[:,:,:-1])/dy
                U = w0*oo[:-2] + w1*(gg[:-2,3:-3,3:-3]+dt*rhs)
                return refresh(U,gg[-2:,3:-3,3:-3]), F, G
            sb, fb, hb = jax.vmap(one)(gb,ob)
            return (jax.lax.dynamic_update_slice_in_dim(out,sb,start,0),
                    jax.lax.dynamic_update_slice_in_dim(fx,fb,start,0),
                    jax.lax.dynamic_update_slice_in_dim(fy,hb,start,0))

        out, fx, fy = jax.lax.fori_loop(0,(info['number']+batch-1)//batch,body,(op,fx,fy))
        return out[:count], fx[:count], fy[:count]

    def _chem(self, data, info, dt):
        def body(i,out):
            return out.at[i].set(tiled_chemistry(data[i],dt,self.c.chem_cells))
        return jax.lax.fori_loop(0,info['number'],body,data)

    def _restrict(self, coarse, fine, info):
        px,py = self.c.patch_x,self.c.patch_y
        mean = fine[:,:-2].reshape(fine.shape[0],12,px,2,py,2).mean(axis=(3,5))
        def body(i,out):
            b,r,c = info['index'][i]
            return jax.lax.dynamic_update_slice(out,mean[i][None],(b,0,r*px,c*py))
        U = jax.lax.fori_loop(0,info['number'],body,coarse[:,:-2])
        return jax.vmap(refresh)(U,coarse[:,-2:])

    def _sync_flux(self, coarse, fluxes, fine_fluxes, info, parent_info, dx, dy):
        """Reflux with time-integrated fluxes, then caller restricts covered cells.

        Replace coarse faces by area-averaged fine faces, including patch edges.
        A parent-block boundary has TWO stored copies of the same face. Update
        both, including when the neighboring parent has no child at that face.
        Returned composite fluxes also feed the next coarser level's register.
        """
        F,G = fluxes
        ff,gg = fine_fluxes
        px,py = self.c.patch_x,self.c.patch_y
        ff = ff[:,:,::2,:].reshape(ff.shape[0],12,px+1,py,2).mean(axis=-1)
        gg = gg[:,:,:,::2].reshape(gg.shape[0],12,px,2,py+1).mean(axis=3)
        def body(i,fg):
            a,b = fg
            p,r,c = info['index'][i]
            a = jax.lax.dynamic_update_slice(a,ff[i][None],(p,0,r*px,c*py))
            b = jax.lax.dynamic_update_slice(b,gg[i][None],(p,0,r*px,c*py))
            n = parent_info['neighbor_index'][p]
            nx,ny = coarse.shape[-2:]
            # Each update writes a shared flux, not an extra additive correction.
            # If both sides are refined their numerical face fluxes agree; this
            # also avoids double-counting when the second child is visited.
            a = jax.lax.cond((r==0)&(n[0]>=0),
                lambda a: jax.lax.dynamic_update_slice(a,ff[i][None,:,0:1,:],
                                                       (jnp.maximum(n[0],0),0,nx,c*py)),lambda a:a,a)
            a = jax.lax.cond(((r+1)*px==nx)&(n[1]>=0),
                lambda a: jax.lax.dynamic_update_slice(a,ff[i][None,:,-1:,:],
                                                       (jnp.maximum(n[1],0),0,0,c*py)),lambda a:a,a)
            b = jax.lax.cond((c==0)&(n[2]>=0),
                lambda b: jax.lax.dynamic_update_slice(b,gg[i][None,:,:,0:1],
                                                       (jnp.maximum(n[2],0),0,r*px,ny)),lambda b:b,b)
            b = jax.lax.cond(((c+1)*py==ny)&(n[3]>=0),
                lambda b: jax.lax.dynamic_update_slice(b,gg[i][None,:,:,-1:],
                                                       (jnp.maximum(n[3],0),0,r*px,0)),lambda b:b,b)
            return a,b
        Fn,Gn = jax.lax.fori_loop(0,info['number'],body,(F,G))
        dF,dG = Fn-F,Gn-G
        U = coarse[:,:-2] - (dF[:,:,1:]-dF[:,:,:-1])/dx - (dG[:,:,:,1:]-dG[:,:,:,:-1])/dy
        return jax.vmap(refresh)(U,coarse[:,-2:]),(Fn,Gn)

    def _cycle(self, states, infos, dt):
        return self._cycle_with_flux(states, infos, dt)[0]

    def _cycle_with_flux(self, states, infos, dt):
        """Accepted-step candidate and RK/subcycle-integrated composite fluxes.

        The existing cycle API still returns states only. Exposing the already
        accumulated refluxed faces permits open-domain conservation accounting;
        no stage, chemistry or reflux formula is changed.
        """
        def evolve(level, data, h, parent_at=None):
            old = data[level]
            current = old
            F = G = None
            for w0,w1,tau,weight in ((0.,1.,0.,1/6),(.75,.25,1.,1/6),(1/3,2/3,.5,2/3)):
                pg = parent_at(tau) if parent_at is not None else None
                ghost = self._ghost(level,current,infos[level],pg)
                current, f,g = self._stage(ghost,old,infos[level],self.dx[level],self.dy[level],h,w0,w1)
                F = h*weight*f if F is None else F+h*weight*f
                G = h*weight*g if G is None else G+h*weight*g
            new = self._chem(current,infos[level],h)
            data = data[:level]+(new,)+data[level+1:]
            if level < self.c.levels:
                def parent_interp(f):
                    U = (1-f)*old[:,:-2]+f*new[:,:-2]
                    a = (1-f)*old[:,-2:]+f*new[:,-2:]
                    interp = jax.vmap(refresh)(U,a)
                    return self._ghost(level,interp,infos[level],parent_at(f) if parent_at else None)
                def half(k,carry):
                    d,fs,gs = carry
                    d,fg = evolve(level+1,d,h/2,lambda q:parent_interp((k+q)/2))
                    return d,fs+fg[0],gs+fg[1]
                child = data[level+1]
                fx = jnp.zeros((child.shape[0],12,child.shape[2]+1,child.shape[3]))
                gy = jnp.zeros((child.shape[0],12,child.shape[2],child.shape[3]+1))
                data,fx,gy = jax.lax.fori_loop(0,2,half,(data,fx,gy))
                new, (F,G) = self._sync_flux(new,(F,G),(fx,gy),infos[level+1],infos[level],self.dx[level],self.dy[level])
                new = self._restrict(new,data[level+1],infos[level+1])
                data = data[:level]+(new,)+data[level+1:]
            return data,(F,G)
        return evolve(0,states,dt)

    def _metrics(self, states, infos):
        dt = jnp.asarray(self.c.dt_max)
        good = jnp.array(True)
        rho_min,Tmin,ymin = jnp.inf,jnp.inf,jnp.inf
        pmax,Tmax,thermo_error = 0.,0.,0.
        for l,s in enumerate(states):
            active = (jnp.arange(s.shape[0])<infos[l]['number'])[:,None,None]
            rho = s[:,0]
            Y = s[:,4:-2]/rho[:,None]
            p = jax.vmap(pressure)(s)
            cs = jnp.sqrt(s[:,-2]*p/rho)
            rate = (jnp.abs(s[:,1]/rho)+cs)/self.dx[l] + (jnp.abs(s[:,2]/rho)+cs)/self.dy[l]
            dt = jnp.minimum(dt,2**l*self.c.cfl/jnp.maximum(jnp.max(jnp.where(active,rate,0)),1.))
            good &= jnp.all(jnp.where(active[:,None],jnp.isfinite(s),True))
            good &= jnp.all(jnp.where(active,(rho>0)&(p>0)&(s[:,-1]>100)&(s[:,-2]>1),True))
            good &= jnp.all(jnp.where(active[:,None],Y>=-1e-8,True))
            good &= jnp.all(jnp.where(active,jnp.sum(Y,axis=1)<=1+1e-8,True))
            def residual(a):
                r=a[0:1]; y=a[4:-2]/r; T=a[-1:]
                _,_,h,R,_=thermo.get_thermo(T,y)
                e=a[3:4]/r-.5*((a[1:2]/r)**2+(a[2:3]/r)**2)
                scale=jnp.maximum(jnp.maximum(jnp.abs(e),jnp.abs(h)),R*T)
                return (jnp.abs(e-(h-R*T))/jnp.maximum(scale,1.))[0]
            err=jax.vmap(residual)(s)
            local_error=jnp.max(jnp.where(active,err,0.))
            thermo_error=jnp.maximum(thermo_error,local_error)
            good &= local_error<1e-6
            rho_min = jnp.minimum(rho_min,jnp.min(jnp.where(active,rho,jnp.inf)))
            Tmin = jnp.minimum(Tmin,jnp.min(jnp.where(active,s[:,-1],jnp.inf)))
            ymin = jnp.minimum(ymin,jnp.min(jnp.where(active[:,None],Y,jnp.inf)))
            pmax = jnp.maximum(pmax,jnp.max(jnp.where(active,p,0)))
            Tmax = jnp.maximum(Tmax,jnp.max(jnp.where(active,s[:,-1],0)))
        base=states[0][0]
        mass=jnp.sum(base[0])*self.dx[0]*self.dy[0]
        energy=jnp.sum(base[3])*self.dx[0]*self.dy[0]
        return jnp.array([dt,good,rho_min,Tmin,ymin,pmax,Tmax,thermo_error,mass,energy])

    def _probes(self, states, infos):
        base = states[0][0]
        p = pressure(base)
        mean_p = p.mean(axis=1)
        above = mean_p>2*self.c.p0
        idx = jnp.max(jnp.where(above,jnp.arange(self.c.nx),-1))
        i = jnp.clip(idx,0,self.c.nx-2)
        frac = (mean_p[i]-2*self.c.p0)/jnp.maximum(mean_p[i]-mean_p[i+1],1e-30)
        front = jnp.where(idx>=0,(i+.5+jnp.clip(frac,0,1))*self.dx[0],0.)
        # L0 is conservative composite after restriction, not independent coarse data.
        totals = jnp.sum(base[:-2],axis=(1,2))*self.dx[0]*self.dy[0]
        return front, totals, mean_p, base[-1].mean(axis=1), base[4].mean(axis=1)/base[0].mean(axis=1)

    def _mask(self, level, parent, info, ghost):
        c = self.c
        def one(s):
            rho = s[0]
            fields = (rho,pressure(s),s[-1],s[4]/rho)
            scales = (jnp.maximum(jnp.abs(rho),.01),jnp.maximum(jnp.abs(fields[1]),c.p0),
                      jnp.maximum(jnp.abs(s[-1]),c.t0), jnp.full_like(rho,self.Y0[0]))
            mask = jnp.zeros_like(rho,dtype=bool)
            for f,scale,tol in zip(fields,scales,(c.density_jump,c.pressure_jump,c.temperature_jump,c.species_jump)):
                # One-sided jumps do not miss a discontinuity on a patch boundary.
                a = jnp.abs(jnp.diff(f,axis=0))
                b = jnp.abs(jnp.diff(f,axis=1))
                jump = jnp.maximum(jnp.pad(a,((1,0),(0,0))),jnp.pad(a,((0,1),(0,0))))
                jump += jnp.maximum(jnp.pad(b,((0,0),(1,0))),jnp.pad(b,((0,0),(0,1))))
                mask |= jump/scale > tol
            mask = jax.lax.reduce_window(mask,False,jax.lax.bitwise_or,(2*c.buffer+1,)*2,(1,1),'SAME')
            return mask[3:-3,3:-3]
        m = jax.vmap(one)(ghost)
        active = (jnp.arange(parent.shape[0])<info['number'])[:,None,None]
        m = jnp.where(active,m,False)
        return amr.get_refinement_block_mask(level,m.astype(jnp.float64))

    def _build_fine(self, level, parent, info):
        raw = amr.get_refinement_block_data(level,parent,info)
        valid = (jnp.arange(raw.shape[0])<info['number'])[:,None,None,None]
        return jnp.where(valid,raw,self.s0[None,:,None,None])

    def regrid_level(self, level, states, infos, old=None, old_info=None):
        ghost = None
        for l in range(level):
            ghost = self.ghost(l,states[l],infos[l],ghost)
        mask = self.mask(level,states[level-1],infos[level-1],ghost)
        # Exchange only small block metadata, never physical field arrays.
        n = int(jnp.sum(mask))
        cap = self.c.capacities[level-1]
        if n >= cap:
            raise MemoryError(f'L{level} requires {n}+sentinel blocks, capacity={cap}. No refinement silently discarded.')
        info = amr.get_refinement_block_info(infos[level-1],mask,cap)
        new = self._fine(level,states[level-1],info)
        if old is not None:
            keys = np.asarray(info['glob_index'])[:n]
            previous = np.asarray(old_info['glob_index'])[:int(old_info['number'])]
            lookup = {tuple(key):i for i,key in enumerate(previous)}
            pairs = [(j,lookup[tuple(key)]) for j,key in enumerate(keys) if tuple(key) in lookup]
            if pairs:
                dst,src = np.array(pairs).T
                new = new.at[dst].set(old[src])
        return new,info

    def regrid(self, states, infos):
        new_states,new_infos = [states[0]],[infos[0]]
        for l in range(1,self.c.levels+1):
            s,i = self.regrid_level(l,new_states,new_infos,states[l],infos[l])
            new_states.append(s); new_infos.append(i)
        # Recompute thermodynamics from conserved restriction, never average T/gamma.
        for l in range(self.c.levels,0,-1):
            new_states[l-1] = self.restrict(new_states[l-1],new_states[l],new_infos[l])
        return tuple(new_states),tuple(new_infos)

    def save(self, path, states, infos, t, step, extra=None):
        path = Path(path)
        payload = {'metadata':np.frombuffer(json.dumps({'config':asdict(self.c),'t':t,'step':step,
                    'extra':extra or {}}).encode(),dtype=np.uint8)}
        for l,(s,i) in enumerate(zip(states,infos)):
            payload[f'L{l}_state'] = np.asarray(s)
            for k,v in i.items(): payload[f'L{l}_{k}'] = np.asarray(v)
        # Periodic snapshots remain separate; the temporary file makes each atomic.
        temp = path.with_suffix('.writing.npz')
        np.savez_compressed(temp,**payload)
        temp.replace(path)

    def load(self,path):
        with np.load(path,allow_pickle=False) as z:
            meta = json.loads(z['metadata'].tobytes())
            if json.dumps(meta['config'],sort_keys=True) != json.dumps(asdict(self.c),sort_keys=True):
                raise ValueError('Resume configuration differs; do not mix meshes/mechanisms')
            states,infos = [],[]
            for l in range(self.c.levels+1):
                states.append(jnp.asarray(z[f'L{l}_state']))
                infos.append({k:jnp.asarray(z[f'L{l}_{k}']) for k in ('number','index','glob_index','neighbor_index')})
        return tuple(states),tuple(infos),meta


def window_speed(history, span=.05, min_x=.10):
    """Least-squares position/time slope over a downstream spatial window.

    It is a diagnostic, not a CJ formation detector; overdrive is not a pass.
    """
    if len(history)<4 or history[-1]['xfront_m'] < min_x+span:
        return None
    right = history[-1]['xfront_m']
    rows = [r for r in history if right-span<=r['xfront_m']<=right]
    if len(rows)<4 or rows[-1]['xfront_m']-rows[0]['xfront_m']<.8*span:
        return None
    t = np.array([r['t_s'] for r in rows]); x=np.array([r['xfront_m'] for r in rows])
    t = t-t.mean()
    if np.dot(t,t)<=0: return None
    speed = float(np.dot(t,x-x.mean())/np.dot(t,t))
    residual = float(np.sqrt(np.mean((x-x.mean()-speed*t)**2)))
    return {'speed_m_s':speed,'fit_rms_m':residual,'span_m':float(x[-1]-x[0]),'samples':len(rows)}
