"""Opt-in algebraic factorization of the existing JANC 2D chemistry formula.

No rate law, clipping, thermodynamics, time step or implicit solve is changed.
The native implementation remains the reference, including its derivative
convention; this module is not a new mechanism/parser or a physical correction.
Create once AFTER set_reaction, in a fresh process per mechanism.
"""
import jax
import jax.numpy as jnp
import numpy as np
from janc_v2.model import reaction_model as r
from janc_v2.preprocess import nondim


def make_factored_matrix():
    if r.dimensions!='2D':raise ValueError('Only the existing 2D convention is supported')
    p=r.ReactionParams;nr=int(r.nr);reactions=int(p['num_of_reactions'])
    vf=jnp.asarray(np.asarray(p['vf']).reshape(reactions,nr))
    vb=jnp.asarray(np.asarray(p['vb']).reshape(reactions,nr))
    eff=jnp.asarray(np.asarray(p['third_body_coeffs']).reshape(reactions,-1))
    M=jnp.asarray(r.species_M);net=vb-vf
    stoich_mass=net*M[:nr][None,:]
    A=p['A'][:,None];B=p['B'][:,None];Ea=p['Ea/Ru'][:,None]
    third=p['is_third_body'][:,None];vsum=p['vsum'][:,None]
    def rates_and_jacobian(T,X):
        shape=T.shape[1:];Tflat=T.reshape(1,-1);x=X.reshape(X.shape[0],-1)
        # Match the reference: third-body sum uses unclipped X; logarithms use
        # the clipped X. Do not silently alter this order at trace species.
        body=(eff@x)*third+(1-third)
        x=jnp.maximum(x,1e-50);logx=jnp.log(x[:nr])
        arr=A*Tflat**B*jnp.exp(-Ea/Tflat)
        forward=arr*jnp.exp(vf@logx)
        reverse=arr/(jnp.exp(net@r.get_gibbs(Tflat[0]))*(101325/nondim.P0/Tflat)**vsum)*jnp.exp(vb@logx)
        diff=forward-reverse
        S=stoich_mass.T@(body*diff)
        # [reaction, derivative-species, cell], instead of carrying both
        # species axes through every reaction/thermodynamic calculation.
        partial=diff[:,None,:]*(eff[:,:nr]/M[:nr][None,:])[:,:,None]
        partial+=body[:,None,:]*(vf[:,:,None]*forward[:,None,:]-vb[:,:,None]*reverse[:,None,:])/(M[:nr,None]*x[:nr])[None,:,:]
        J=jnp.einsum('rk,rnc->knc',stoich_mass,partial,precision=jax.lax.Precision.HIGHEST)
        return S.reshape((nr,)+shape),J.reshape((nr,nr)+shape)
    def matrix(T,X,dt):
        S,J=rates_and_jacobian(T,X)
        b=jnp.moveaxis(S,0,-1)[...,None];D=jnp.moveaxis(J,(0,1),(-2,-1))
        solved=jnp.linalg.solve(jnp.eye(nr)/dt-D,b)[...,0]
        return jnp.moveaxis(solved,-1,0)
    return matrix,rates_and_jacobian


def make_update():
    matrix,rates=make_factored_matrix()
    from janc_v2.model.thermo_model import fill_Y
    from janc_v2.detonation_amr.core import refresh
    def source(U,aux,dt):
        rho=U[:1];Y=fill_Y(U[4:]/rho)
        change=matrix(aux[1:2],rho*Y/r.Mex,dt)/rho
        change=jnp.clip(change,min=-Y[:-1],max=1-Y[:-1])
        return jnp.concatenate((jnp.zeros_like(U[:4]),rho*change))
    def update(state,dt):
        return refresh(state[:12]+source(state[:12],state[-2:],dt),state[-2:])
    return update,source,matrix,rates
