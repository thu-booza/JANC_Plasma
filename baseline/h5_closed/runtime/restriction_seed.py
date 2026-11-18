"""Refresh covered parents using fine-temperature means as inversion guesses.

Only the conserved fine-cell mean defines the final parent state. Gamma and T
are recomputed from that conserved state; averaged T is NOT the returned T.
Uncovered cells retain their original guesses, including failure sentinels.
"""
import jax
import jax.numpy as jnp
from h5_engine import H5Engine
from janc_v2.detonation_amr.core import refresh

class RestrictionSeedEngine(H5Engine):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.checkpoint_identity['restriction_aux_guess']='fine_temperature_seed_for_covered_parents_v1'

    def _restrict(self,coarse,fine,info):
        px,py=self.c.patch_x,self.c.patch_y
        mean=fine[:,:12].reshape(fine.shape[0],12,px,2,py,2).mean(axis=(3,5))
        temperature=fine[:,-1:].reshape(fine.shape[0],1,px,2,py,2).mean(axis=(3,5))
        def body(i,carry):
            U,T=carry;b,r,c=info['index'][i]
            U=jax.lax.dynamic_update_slice(U,mean[i][None],(b,0,r*px,c*py))
            T=jax.lax.dynamic_update_slice(T,temperature[i][None],(b,0,r*px,c*py))
            return U,T
        U,T=jax.lax.fori_loop(0,info['number'],body,(coarse[:,:12],coarse[:,-1:]))
        aux=jnp.concatenate((coarse[:,-2:-1],T),axis=1)
        return jax.vmap(refresh)(U,aux)
