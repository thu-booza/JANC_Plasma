"""Conservative planar ZND startup, not a mature cellular detonation seed."""
import hashlib
import json
from pathlib import Path
import cantera as ct
import numpy as np
import jax.numpy as jnp
from janc_v2.model import thermo_model as thermo
from .predevelopment import PredevelopmentEngine, block_coordinates, assert_composite_restricted


class ZNDSeed:
    def __init__(self,engine,mechanism,reference_directory):
        if not isinstance(engine,PredevelopmentEngine):raise TypeError('ZND startup requires an explicit open-tail engine')
        self.engine=engine;root=Path(reference_directory)
        refpath=root/'reference.json';profile=root/'CJ_ZND_tail.npz'
        ref=json.loads(refpath.read_text(encoding='utf-8'))
        digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        if digest(mechanism)!=ref['mechanism_sha256']:raise ValueError('ZND/mechanism mismatch')
        initial=ref['initial']
        for name,key in (('p0','P_Pa'),('t0','T_K'),('phi','phi')):
            if not np.isclose(getattr(engine.c,name),initial[key],rtol=1e-12,atol=0):raise ValueError('ZND/initial state mismatch: '+name)
        self.speed=float(ref['cj_speed_refined_m_s'])
        with np.load(profile,allow_pickle=False) as a:
            z=a['distance_m'];rho=a['rho'];T=a['T_K'];Y=a['Y'].T;w=a['U_m_s']
        if (len(z)<2 or z[0]!=0 or np.any(np.diff(z)<=0) or Y.shape!=(len(z),9)
                or not all(np.isfinite(a).all() for a in (z,rho,T,Y,w))):raise ValueError('Invalid ZND profile')
        if min(rho.min(),T.min(),w.min())<=0 or Y.min()<-1e-12:raise ValueError('Unphysical ZND profile')
        np.testing.assert_allclose(Y.sum(axis=1),1,rtol=0,atol=1e-12)
        gas=ct.Solution(str(mechanism));g=ct.SolutionArray(gas,len(z));g.TDY=T,rho,Y
        u=self.speed-w
        self.values=np.column_stack((rho,rho*u,np.zeros_like(rho),rho*(g.int_energy_mass+.5*u*u),rho[:,None]*Y[:,:8]))
        self.z=z;self.T=T;self.fresh=np.asarray(engine.s0[:12])
        self.slope=np.diff(self.values,axis=0)/np.diff(z)[:,None]
        self.integral=np.vstack((np.zeros(12),np.cumsum(.5*(self.values[1:]+self.values[:-1])*np.diff(z)[:,None],axis=0)))
        self.manifest=dict(kind='planar_ZND_startup_not_mature_cellular',reference_sha256=digest(refpath),
            profile_sha256=digest(profile),mechanism_sha256=digest(mechanism),speed_m_s=self.speed,
            maximum_tail_m=float(z[-1]),velocity_conversion='laboratory u = D_CJ - stored shock-relative U_m_s',
            fresh_state='Native engine.s0, also used for window replenishment',
            postshock_state='Cantera reference conserved state; native EOS recovers cell-average T/gamma')

    def primitive(self,distance):
        d=np.asarray(distance,dtype=float)
        if not np.isfinite(d).all() or np.any(d>self.z[-1]):raise ValueError('ZND tail extrapolation forbidden')
        clipped=np.maximum(d,0);k=np.minimum(np.searchsorted(self.z,clipped,side='right')-1,len(self.z)-2)
        dz=clipped-self.z[k]
        v=self.integral[k]+self.values[k]*dz[...,None]+.5*self.slope[k]*dz[...,None]**2
        return np.where((d<0)[...,None],d[...,None]*self.fresh,v)

    def block(self,edges,ny,shock):
        edges=np.asarray(edges)
        if np.any(np.diff(edges)<=0) or not np.isfinite(shock):raise ValueError('Invalid seed coordinates')
        f=self.primitive(shock-edges)
        U=((f[:-1]-f[1:])/np.diff(edges)[:,None]).T
        upstream=edges[:-1]>=shock
        U[:,upstream]=self.fresh[:,None]
        rho=U[0:1];Y=U[4:]/rho
        if rho.min()<=0 or Y.min()<-1e-12 or Y.sum(axis=0).max()>1+1e-12:raise ValueError('Invalid mapped species/density')
        internal=U[3:4]/rho-.5*((U[1:2]/rho)**2+(U[2:3]/rho)**2)
        center=.5*(edges[:-1]+edges[1:]);distance=shock-center
        guess=np.where(distance<0,self.engine.c.t0,np.interp(np.maximum(distance,0),self.z,self.T))
        aux=thermo.get_T(jnp.asarray(internal[...,None]),jnp.asarray(Y[...,None]),jnp.asarray(guess[None,:,None]))
        state=jnp.concatenate((jnp.asarray(U[...,None]),aux),axis=0)
        # Avoid a different upstream thermodynamic roundoff path from replenishment.
        state=state.at[:,upstream,0].set(self.engine.s0[:,None])
        return jnp.broadcast_to(state,(14,len(edges)-1,ny))

    def initial(self,shock):
        e=self.engine;c=e.c
        if e.amplitude!=0:raise ValueError('This initializer is planar; it does not apply the hot-spot perturbation')
        if not 0<shock<c.lx or shock>self.z[-1]:raise ValueError('Invalid initial shock/tail position')
        base=self.block(np.arange(c.nx+1)*e.dx[0],c.ny,shock)[None]
        states=[base];infos=[dict(number=jnp.array(1),index=jnp.array([[0,0,0]]),
            glob_index=jnp.array([[0,0]]),neighbor_index=jnp.full((1,4),-1))]
        for l in range(1,c.levels+1):
            fine,info=e.regrid_level(l,states,infos)
            # Seed the retained resolution directly; never interpolate a smeared coarse front.
            for i,(bx,by) in enumerate(block_coordinates(info,l)):
                nx,ny=fine.shape[-2:];edges=(bx*nx+np.arange(nx+1))*e.dx[l]
                fine=fine.at[i].set(self.block(edges,ny,shock))
            states.append(fine);infos.append(info)
        for l in range(c.levels,0,-1):states[l-1]=e.restrict(states[l-1],states[l],infos[l])
        assert_composite_restricted(e,states,infos)
        exact=(self.primitive(shock)-self.primitive(shock-c.lx))*c.ly
        total=np.asarray(states[0][0,:12]).sum(axis=(1,2))*e.dx[0]*e.dy[0]
        error=float(np.max(abs(total-exact)/np.maximum(abs(exact),1)))
        if error>1e-11:raise ArithmeticError('Initial AMR mapping inventory mismatch')
        return tuple(states),tuple(infos),dict(self.manifest,shock_local_m=float(shock),initial_inventory_scaled_error=error)
