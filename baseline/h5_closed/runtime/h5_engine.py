"""H5 with fixed impermeable left end, open right end and slip y walls."""
import numpy as np
import cantera as ct
import jax
import jax.numpy as jnp
from jaxamr import amr
from janc_v2.detonation_amr.predevelopment import PredevelopmentEngine,block_coordinates,assert_composite_restricted
from janc_v2.detonation_amr.core import refresh,pressure,aux_func,interface_flux
from composition_repair import make_repair
from factored_chemistry import make_factored_matrix
from janc_v2.model import reaction_model as reaction
from janc_v2.model.thermo_model import fill_Y


class H5Engine(PredevelopmentEngine):
    def __init__(self,c,mechanism,physics,amr_policy):
        self.physics=physics;self.amr_policy=amr_policy
        super().__init__(c,mechanism)
        self.checkpoint_identity['boundary']='x_left_fixed_velocity_mirror_right_zero_gradient_y_slip'
        gas=ct.Solution(str(mechanism));gas.TPX=c.t0,c.p0,physics['mole_ratio']
        self.Y0=gas.Y[:-1].copy();self.s0=self.uniform(c.t0,c.p0,1,1)[:,0,0]
        hot=self.uniform(physics['hot_T_K'],physics['hot_p_Pa'],1,1)[:,0,0];u=physics['hot_u_m_s']
        self.hot=hot.at[1].set(hot[0]*u).at[3].add(.5*hot[0]*u*u)
        hydrogen=[gas.n_atoms(k,'H')*gas.atomic_weight('H')/gas.molecular_weights[k] for k in range(8)]
        self.repair=make_repair(hydrogen,amr_policy['composition_repair_max_mass_fraction'])
        matrix,_=make_factored_matrix()
        def point_update(s,dt):
            U=self.repair(s[:12]);rho=U[:1];Y=fill_Y(U[4:]/rho)
            raw=matrix(s[-1:],rho*Y/reaction.Mex,dt)
            # A common limiter preserves the implicit chemical increment's
            # element constraints; unlike independent per-species clipping.
            dy=jnp.concatenate((raw,-jnp.sum(raw,axis=0,keepdims=True)))
            available=jnp.maximum(rho*Y,0)
            ratio=jnp.where(dy<0,jnp.minimum(1.,available/jnp.maximum(-dy,1e-300)),1.)
            factor=jnp.min(ratio,axis=0,keepdims=True);factor=jnp.where(factor<1,factor*(1-1e-12),factor)
            return refresh(U+jnp.concatenate((jnp.zeros_like(U[:4]),raw*factor)),s[-2:])
        self.point_update=point_update
        self.accept_metrics=jax.jit(self._accept_metrics)

    def _ghost(self,level,s,info,parent_g=None):
        g=super()._ghost(level,s,info,parent_g)
        # Stationary wall at physical x=0: mirror all scalar fields and reverse
        # both velocity components. Interior gas is never reset or re-seeded.
        wall=s[:,:,:3,:][:,:,::-1,:].at[:,1:3].multiply(-1)
        if level==0:
            return g.at[:,:,:3,3:-3].set(wall)
        coord=jnp.zeros(s.shape[0],dtype=jnp.int64)
        for k in range(level+1):coord=coord*self.nblock[k][0]+info['glob_index'][:,2*k]
        on_wall=(coord==0)&(jnp.arange(s.shape[0])<info['number'])
        return g.at[:,:,:3,3:-3].set(jnp.where(on_wall[:,None,None,None],wall,g[:,:,:3,3:-3]))

    def _stage(self,g,old,info,dx,dy,dt,w0,w1):
        from positive_amr import limited_stage
        return limited_stage(self,g,old,info,dx,dy,dt,w0,w1)

    def _sync_flux(self,*args):
        s,fg=super()._sync_flux(*args)
        def body(k,out):
            block=out[k]
            fixed=jax.lax.cond(jnp.any(block[4:12]<0),lambda v:refresh(self.repair(v[:12]),v[-2:]),lambda v:v,block)
            return out.at[k].set(fixed)
        return jax.lax.fori_loop(0,args[4]['number'],body,s),fg

    def _chem(self,data,info,dt):
        count,nv,nx,ny=data.shape;cells=self.c.chem_cells
        flat=data.transpose(1,0,2,3).reshape(nv,-1,1);pad=(-flat.shape[1])%cells
        padded=jnp.pad(flat,((0,0),(0,pad),(0,0)),mode='edge');valid=info['number']*nx*ny
        def body(k,out):
            chunk=jax.lax.dynamic_slice_in_dim(padded,k*cells,cells,axis=1)
            changed=self.point_update(chunk,dt)
            changed=jnp.where((k*cells+jnp.arange(cells)<valid)[None,:,None],changed,chunk)
            return jax.lax.dynamic_update_slice_in_dim(out,changed,k*cells,axis=1)
        out=jax.lax.fori_loop(0,(valid+cells-1)//cells,body,padded)
        return out[:,:flat.shape[1]].reshape(nv,count,nx,ny).transpose(1,0,2,3)

    def _mask(self,level,parent,info,ghost):
        original=super()._mask(level,parent,info,ghost)
        # Per-column leading crossing on parent grid; no forced full-width band
        # if a column's wave has moved beyond this block.
        p=jax.vmap(pressure)(parent);nx,ny=parent.shape[-2:]
        ids=jnp.max(jnp.where(p>2*self.c.p0,jnp.arange(nx)[None,:,None],-1),axis=1)
        local=(jnp.arange(nx)[None,:,None]-.5-ids[:,None,:])*self.dx[level-1]
        band=(local>=-self.amr_policy['protect_behind_m'])&(local<=self.amr_policy['protect_ahead_m'])
        # Only blocks containing the pressure crossing get a protected band;
        # gradients plus the existing global buffer handle adjacent boundaries.
        band&=((ids>=0)&(ids<nx-1))[:,None,:]
        band&=(jnp.arange(parent.shape[0])<info['number'])[:,None,None]
        protected=amr.get_refinement_block_mask(level,band.astype(jnp.float64))
        return jnp.maximum(original,protected)

    def initial_block(self,xedges,yedges):
        nx=len(xedges)-1;ny=len(yedges)-1;f=np.zeros((nx,ny));q=4
        for a in range(q):
            x=xedges[:-1]+(a+.5)/q*np.diff(xedges)
            for b in range(q):
                y=yedges[:-1]+(b+.5)/q*np.diff(yedges)
                hot=np.broadcast_to(x[:,None]<self.physics['hot_length_m'],(nx,ny)).copy()
                for fy in self.physics['hotspot_y_fractions']:
                    hot|=(x[:,None]-self.physics['hot_length_m'])**2+(y[None,:]-fy*self.c.ly)**2<=self.physics['hotspot_radius_m']**2
                f+=hot/(q*q)
        f=jnp.asarray(f);U=self.s0[:12,None,None]+f*(self.hot[:12,None,None]-self.s0[:12,None,None])
        aux=self.s0[-2:,None,None]+f*(self.hot[-2:,None,None]-self.s0[-2:,None,None])
        return refresh(U,aux)

    def initial(self):
        c=self.c;s=self.initial_block(np.arange(c.nx+1)*self.dx[0],np.arange(c.ny+1)*self.dy[0])
        states=[s[None]];infos=[dict(number=jnp.array(1),index=jnp.array([[0,0,0]]),glob_index=jnp.array([[0,0]]),neighbor_index=jnp.full((1,4),-1))]
        for level in range(1,c.levels+1):
            fine,info=self.regrid_level(level,states,infos);nx,ny=fine.shape[-2:]
            coords=block_coordinates(info,level)
            for start in range(0,len(coords),16):
                tiles=[self.initial_block((bx*nx+np.arange(nx+1))*self.dx[level],(by*ny+np.arange(ny+1))*self.dy[level]) for bx,by in coords[start:start+16]]
                fine=fine.at[start:start+len(tiles)].set(jnp.stack(tiles))
            states.append(fine);infos.append(info)
        for level in range(c.levels,0,-1):states[level-1]=self.restrict(states[level-1],states[level],infos[level])
        assert_composite_restricted(self,states,infos)
        return tuple(states),tuple(infos)

    def _accept_metrics(self,states,infos):
        m=self._metrics(states,infos);minimum=jnp.inf
        for s,i in zip(states,infos):
            Y=s[:,4:12]/s[:,:1];N=1-jnp.sum(Y,axis=1,keepdims=True)
            active=(jnp.arange(s.shape[0])<i['number'])[:,None,None,None]
            minimum=jnp.minimum(minimum,jnp.min(jnp.where(active,jnp.concatenate((Y,N),axis=1),jnp.inf)))
        return m.at[1].set(m[1].astype(bool)&(minimum>=-1e-13)).at[4].set(minimum)
