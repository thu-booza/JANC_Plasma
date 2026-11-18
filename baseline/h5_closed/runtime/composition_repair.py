"""Local constrained correction of trace negative reactive species.

Keeps density, momenta, total energy and H/O/N inventories. This is a declared
numerical extension, not a claim of bitwise equivalence with native JANC.
"""
import jax
import jax.numpy as jnp


def make_repair(hydrogen_mass_fractions,maximum_change=1e-6):
    h=jnp.asarray(hydrogen_mass_fractions)[:,None,None];o=1-h
    def project(U):
        rho=U[0:1];q=U[4:12]
        target_h=jnp.sum(h*q,axis=0);target_o=jnp.sum(o*q,axis=0)
        needs=jnp.any(q<0,axis=0)
        w=jnp.maximum(jnp.abs(q),jnp.maximum(rho,1e-30)*1e-10)
        # Bound-constrained weighted projection, clamping one violating variable
        # per iteration. If the small active-set solve fails, reject the step.
        free=jnp.ones_like(q,dtype=bool)
        def solve(free):
            wf=w*free;start=q*free
            a=jnp.sum(wf*h*h,axis=0);b=jnp.sum(wf*h*o,axis=0);c=jnp.sum(wf*o*o,axis=0)
            rh=target_h-jnp.sum(start*h,axis=0);ro=target_o-jnp.sum(start*o,axis=0)
            det=a*c-b*b;safe=jnp.maximum(det,1e-300)
            lh=(c*rh-b*ro)/safe;lo=(a*ro-b*rh)/safe
            z=start+wf*(h*lh+o*lo)
            return z
        def body(i,free):
            z=solve(free);index=jnp.argmin(jnp.where(free,z,jnp.inf),axis=0)
            negative=jnp.min(z,axis=0)<0
            return free&~((jnp.arange(8)[:,None,None]==index[None])&negative[None])
        free=jax.lax.fori_loop(0,8,body,free)
        z=solve(free);z=jnp.where(needs[None],z,q)
        err_h=jnp.abs(jnp.sum(h*z,axis=0)-target_h);err_o=jnp.abs(jnp.sum(o*z,axis=0)-target_o)
        valid=(jnp.min(z,axis=0)>=-1e-14*rho[0])&(jnp.max(jnp.abs(z-q),axis=0)<=maximum_change*rho[0])
        valid&=(err_h+err_o<=1e-13*rho[0])&(target_h>=-1e-14*rho[0])&(target_o>=-1e-14*rho[0])
        # Never change species that were already nonnegative.
        z=jnp.where(valid[None],z,jnp.nan)
        return U.at[4:12].set(z)
    def repair(U):
        return jax.lax.cond(jnp.any(U[4:12]<0),project,lambda value:value,U)
    return repair
