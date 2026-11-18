"""AMR tagging repair: valid face sensors, then global same-level tag exchange.

No physical state, flux, chemistry or time integrator is modified. Buffering
uses Boolean tags of actual active cells, including diagonal neighbours. No
fluid corner ghosts are read. Missing same-level blocks provide no tags:
refinement remains within existing parent support (as in the original tree).
"""
import jax
import jax.numpy as jnp
import numpy as np
from jaxamr import amr
from .core import BaseEngine, Config, pressure
from .seed import perturbation_average

MASK_REVISION = '20261028-valid-face-global-tag-buffer-v1'


def buffer_tags(tags, coords, active, total, radius):
    """Square Chebyshev dilation in physical same-level cell coordinates.

Only an integer block lookup is dense over the domain. Field/tag work scales
with allocated blocks, not with the full finest-resolution physical domain.
Physical exterior and absent same-level blocks are False, not duplicated tags.
"""
    count,nx,ny=tags.shape
    total_x,total_y=total
    size=total_x*total_y
    key=coords[:,0]*total_y+coords[:,1]
    lookup=jnp.full((size+1,),-1,dtype=jnp.int32)
    lookup=lookup.at[jnp.where(active,key,size)].set(jnp.where(active,jnp.arange(count,dtype=jnp.int32),-1))
    ix=jnp.arange(-radius,nx+radius)
    iy=jnp.arange(-radius,ny+radius)
    def one(coord):
        bx=coord[0]+ix//nx;by=coord[1]+iy//ny
        inside=(bx[:,None]>=0)&(bx[:,None]<total_x)&(by[None,:]>=0)&(by[None,:]<total_y)
        slot=lookup[jnp.clip(bx[:,None]*total_y+by[None,:],0,size)]
        halo=tags[jnp.maximum(slot,0),(ix%nx)[:,None],(iy%ny)[None,:]]
        halo=halo&inside&(slot>=0)
        return jax.lax.reduce_window(halo,False,jax.lax.bitwise_or,
                                    (2*radius+1,)*2,(1,1),'VALID')
    return jax.vmap(one)(coords)&active[:,None,None]


class RepairedMaskMixin:
    def _mask(self,level,parent,info,ghost):
        c=self.c
        def one(s):
            rho=s[0];fields=(rho,pressure(s),s[-1],s[4]/rho)
            mid=lambda a:a[3:-3,3:-3]
            scales=(jnp.maximum(jnp.abs(mid(rho)),.01),
                    jnp.maximum(jnp.abs(mid(fields[1])),c.p0),
                    jnp.maximum(jnp.abs(mid(s[-1])),c.t0),
                    jnp.full_like(mid(rho),self.Y0[0]))
            tag=jnp.zeros_like(mid(rho),dtype=bool)
            for f,scale,tol in zip(fields,scales,(c.density_jump,c.pressure_jump,c.temperature_jump,c.species_jump)):
                center=mid(f)
                dx=jnp.maximum(jnp.abs(center-f[2:-4,3:-3]),jnp.abs(f[4:-2,3:-3]-center))
                dy=jnp.maximum(jnp.abs(center-f[3:-3,2:-4]),jnp.abs(f[3:-3,4:-2]-center))
                tag|=(dx+dy)/scale>tol
            return tag
        active=jnp.arange(parent.shape[0])<info['number']
        tags=jax.vmap(one)(ghost)&active[:,None,None]
        coords=jnp.zeros((parent.shape[0],2),dtype=jnp.int64)
        total=np.ones(2,dtype=int)
        for k in range(level):
            coords=coords*jnp.asarray(self.nblock[k])+info['glob_index'][:,2*k:2*k+2]
            total*=self.nblock[k]
        tags=buffer_tags(tags,coords,active,tuple(int(x) for x in total),c.buffer)
        return amr.get_refinement_block_mask(level,tags.astype(jnp.float64))


class Engine(RepairedMaskMixin, BaseEngine):
    def __init__(self,config,mechanism,amplitude=0.):
        if not 0<=amplitude<=1e-3:raise ValueError('Diagnostic amplitude outside [0,1e-3]')
        self.amplitude=amplitude
        super().__init__(config,mechanism)

    def initial(self):
        if self.amplitude==0.:return super().initial()
        c=self.c
        hot_cells=int(np.sum((np.arange(c.nx)+.5)*self.dx[0]<c.hot_m))
        edge=hot_cells*self.dx[0]
        x=(np.arange(c.nx)+.5)*self.dx[0];y=(np.arange(c.ny)+.5)*self.dy[0]
        hot=(x<edge)[None,:,None]
        base=self.uniform(jnp.where(hot,c.hot_t,c.t0),jnp.where(hot,c.hot_p,c.p0),c.nx,c.ny)
        g=perturbation_average(x,y,self.dx[0],self.dy[0],edge,c.ly)
        base=base.at[:-2].multiply(jnp.asarray(1+self.amplitude*g)[None])
        states=[base[None]]
        infos=[dict(number=jnp.array(1),index=jnp.array([[0,0,0]]),glob_index=jnp.array([[0,0]]),neighbor_index=jnp.full((1,4),-1))]
        for level in range(1,c.levels+1):
            s,i=self.regrid_level(level,states,infos);states.append(s);infos.append(i)
        return tuple(states),tuple(infos)
