"""Shared-face positivity limiter for block AMR RK stages.

Low-order LLF supplies nonnegative species inventory at the existing CFL.
One coefficient per face blends ALL conserved flux components. Adjacent blocks
exchange cell ratios, making the two stored copies of a shared face identical.
Reflux and parent-time interpolation remain native and retain health checks.
"""
import jax
import jax.numpy as jnp
from janc_v2.detonation_amr.core import interface_flux,aux_func,refresh


def species(U):
    return jnp.concatenate((U[4:12],U[:1]-jnp.sum(U[4:12],axis=0,keepdims=True)),axis=0)


def face_coefficients(ratio,ax,ay,neighbor):
    """ratio[block,species,x,y]; ax/ay are species antidiffusive inventories."""
    count=ratio.shape[0]
    ids=jnp.arange(count)
    def one(i):
        r=ratio[i];n=neighbor[i]
        left=jnp.where(n[0]>=0,ratio[jnp.maximum(n[0],0),:,-1,:],1.)
        right=jnp.where(n[1]>=0,ratio[jnp.maximum(n[1],0),:,0,:],1.)
        bottom=jnp.where(n[2]>=0,ratio[jnp.maximum(n[2],0),:,:,-1],1.)
        top=jnp.where(n[3]>=0,ratio[jnp.maximum(n[3],0),:,:,0],1.)
        rl=jnp.concatenate((left[:,None,:],r),axis=1)
        rr=jnp.concatenate((r,right[:,None,:]),axis=1)
        rb=jnp.concatenate((bottom[:,:,None],r),axis=2)
        rt=jnp.concatenate((r,top[:,:,None]),axis=2)
        tx=jnp.min(jnp.where(ax[i]>0,rl,jnp.where(ax[i]<0,rr,1.)),axis=0)
        ty=jnp.min(jnp.where(ay[i]>0,rb,jnp.where(ay[i]<0,rt,1.)),axis=0)
        return tx,ty
    return jax.vmap(one)(ids)


def limited_stage(engine,g,old,info,dx,dy,dt,w0,w1,details=False):
    count,nv,nx,ny=old.shape;batch=min(engine.c.block_batch,count);pad=(-count)%batch
    gp=jnp.pad(g,((0,pad),(0,0),(0,0),(0,0)),mode='edge');op=jnp.pad(old,((0,pad),(0,0),(0,0),(0,0)),mode='edge')
    def speed(a):
        rho,u,v,Y,p,cs=aux_func.U_to_prim(a[:12],a[-2:])
        return jnp.max(jnp.abs(u)+cs),jnp.max(jnp.abs(v)+cs)
    ax_,ay_=jax.vmap(speed)(g);active=jnp.arange(count)<info['number']
    alpha=jnp.array([jnp.max(jnp.where(active,ax_,0)),jnp.max(jnp.where(active,ay_,0))])
    F0=jnp.zeros((count+pad,12,nx+1,ny));G0=jnp.zeros((count+pad,12,nx,ny+1))
    def evaluate(gg,oo):
        U=gg[:12];rho,u,v,Y,p,cs=aux_func.U_to_prim(U,gg[-2:]);F,G=interface_flux(gg,alpha)
        fx=jnp.concatenate((rho*u,rho*u*u+p,rho*u*v,u*(U[3:4]+p),rho*u*Y))
        fy=jnp.concatenate((rho*v,rho*u*v,rho*v*v+p,v*(U[3:4]+p),rho*v*Y))
        f=.5*(fx[:,2:-3,3:-3]+fx[:,3:-2,3:-3]-alpha[0]*(U[:,3:-2,3:-3]-U[:,2:-3,3:-3]))
        q=.5*(fy[:,3:-3,2:-3]+fy[:,3:-3,3:-2]-alpha[1]*(U[:,3:-3,3:-2]-U[:,3:-3,2:-3]))
        rhs=-(f[:,1:]-f[:,:-1])/dx-(q[:,:,1:]-q[:,:,:-1])/dy
        low=w0*oo[:12]+w1*(U[:,3:-3,3:-3]+dt*rhs)
        ax=dt*w1/dx*species(F-f);ay=dt*w1/dy*species(G-q)
        negative=jnp.minimum(ax[:,:-1],0)+jnp.minimum(-ax[:,1:],0)+jnp.minimum(ay[:,:,:-1],0)+jnp.minimum(-ay[:,:,1:],0)
        stock=species(low)
        ratio=jnp.where(negative<0,jnp.minimum(1.,jnp.maximum(stock,0)/jnp.maximum(-negative,1e-300)),1.)
        ratio=jnp.where(ratio<1.,ratio*(1-1e-12),ratio)
        # If LLF itself loses positivity, reject rather than projecting large errors.
        valid=jnp.all(stock>=-1e-14*low[:1],axis=0)&(low[0]>0)&jnp.all(jnp.isfinite(low),axis=0)
        ratio=jnp.where(valid[None],ratio,jnp.nan)
        return F,G,f,q,ratio
    ratio0=jnp.ones((count+pad,9,nx,ny))
    def collect(k,carry):
        start=k*batch;gg=jax.lax.dynamic_slice_in_dim(gp,start,batch);oo=jax.lax.dynamic_slice_in_dim(op,start,batch)
        parts=jax.vmap(evaluate)(gg,oo)
        return tuple(jax.lax.dynamic_update_slice_in_dim(a,b,start,axis=0) for a,b in zip(carry,parts))
    F,G,f,q,ratio=jax.lax.fori_loop(0,(info['number']+batch-1)//batch,collect,(F0,G0,F0,G0,ratio0))
    F,G,f,q,ratio=[a[:count] for a in (F,G,f,q,ratio)]
    ax=dt*w1/dx*jax.vmap(species)(F-f);ay=dt*w1/dy*jax.vmap(species)(G-q)
    tx,ty=face_coefficients(ratio,ax,ay,info['neighbor_index'])
    limitedF=f+tx[:,None]*(F-f);limitedG=q+ty[:,None]*(G-q)
    fp=jnp.pad(limitedF,((0,pad),(0,0),(0,0),(0,0)));qp=jnp.pad(limitedG,((0,pad),(0,0),(0,0),(0,0)))
    def finish(k,out):
        start=k*batch
        gg=jax.lax.dynamic_slice_in_dim(gp,start,batch);oo=jax.lax.dynamic_slice_in_dim(op,start,batch)
        ff=jax.lax.dynamic_slice_in_dim(fp,start,batch);qq=jax.lax.dynamic_slice_in_dim(qp,start,batch)
        rhs=-(ff[:,:,1:]-ff[:,:,:-1])/dx-(qq[:,:,:,1:]-qq[:,:,:,:-1])/dy
        U=w0*oo[:,:12]+w1*(gg[:,:12,3:-3,3:-3]+dt*rhs)
        outblock=jax.vmap(refresh)(U,gg[:,-2:,3:-3,3:-3])
        return jax.lax.dynamic_update_slice_in_dim(out,outblock,start,axis=0)
    out=jax.lax.fori_loop(0,(info['number']+batch-1)//batch,finish,op)[:count]
    if details:
        return out,limitedF,limitedG,dict(ratio=ratio,theta_x=tx,theta_y=ty,unlimited_F=F,unlimited_G=G)
    return out,limitedF,limitedG
