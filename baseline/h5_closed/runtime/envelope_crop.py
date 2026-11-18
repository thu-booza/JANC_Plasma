"""Read-only crop based on the transverse front envelope, then leaf validation."""
import numpy as np
import jax.numpy as jnp
import cantera as ct
from janc_v2.detonation_amr.core import pressure
from janc_v2.detonation_amr.strip import make_crop,leading_shock_details


def envelope_bounds(front_min,front_max,length,guard,behind=.0018,ahead=.001):
    if not np.isfinite([front_min,front_max]).all() or not 0<front_min<=front_max<length:
        raise ValueError('Leading front missing or outside physical domain')
    rear=front_min-behind
    if rear<=0:raise ValueError('Required reaction band extends outside the physical domain')
    return max(0.,rear-guard),min(length,front_max+ahead+guard)


class EnvelopeCrop:
    def __init__(self,engine,mechanism):
        self.engine=engine;self.cache={};self.last={}
        self.M=ct.Solution(str(mechanism)).molecular_weights

    def __call__(self,states,infos,unused_mean_front):
        e=self.engine;c=e.c;dx=e.dx[-1]
        p=np.asarray(pressure(states[0][0]));n,m=p.shape;cols=np.arange(m)
        ids=np.max(np.where(p>2*c.p0,np.arange(n)[:,None],-1),axis=0)
        if np.any((ids<0)|(ids>=n-1)):raise ValueError('Leading threshold missing or at physical domain edge')
        den=p[ids+1,cols]-p[ids,cols]
        frac=np.divide(2*c.p0-p[ids,cols],den,out=np.zeros(m),where=den!=0)
        front=(ids+.5+np.clip(frac,0,1))*e.dx[0]
        left,right=envelope_bounds(float(front.min()),float(front.max()),c.lx,max(8*e.dx[0],.0002))
        for retry in range(4):
            # Quantized shape limits recompilation while placement follows the rear-most column.
            width=min(c.lx,np.ceil((right-left)/.0005)*.0005)
            cells=min(c.nx*2**c.levels,int(np.ceil(width/dx))+1)
            if cells not in self.cache:self.cache[cells]=make_crop(e,behind=0.,ahead=(cells-1)*dx)
            x,u,cov=map(np.asarray,self.cache[cells](states,infos,jnp.asarray(left)))
            if not np.isfinite(u).all() or np.any(cov<0):raise ValueError('Missing AMR data in diagnostic crop')
            rho=u[0];Y=np.concatenate((u[4:12]/rho,1-(u[4:12]/rho).sum(axis=0,keepdims=True)))
            if Y.min() < -1e-8:raise ValueError('Invalid diagnostic composition')
            Y=np.maximum(Y,0);Y/=Y.sum(axis=0,keepdims=True)
            pp=rho*u[-1]*ct.gas_constant*np.sum(Y/self.M[:,None,None],axis=0)
            shock,invalid,cross,quality=leading_shock_details(x,pp,cov,e.dx[0],c.p0)
            rear=shock-.0018
            if not quality['geometry_invalid'].any() and np.all(rear>=x[0]):
                self.last=dict(revision='transverse_envelope_leaf_checked_v1',attempts=retry+1,
                               x_min_m=float(x[0]),x_max_m=float(x[-1]),columns=int(len(shock)),
                               minimum_rear_margin_m=float(np.min(rear-x[0])),width_cells=cells)
                return x,u,cov
            if x[0]<=dx and rear.min()<x[0]:raise ValueError('Reaction search band reaches physical left boundary')
            if x[-1]>=c.lx-dx and quality['geometry_invalid'].any():raise ValueError('Leading-front geometry cannot be resolved inside physical domain')
            left=max(0.,min(left-.0005,float(rear.min())-.0002));right=min(c.lx,right+.0005)
        raise ValueError('Adaptive diagnostic crop could not verify complete front/reaction band')
