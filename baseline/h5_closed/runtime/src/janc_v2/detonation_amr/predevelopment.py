"""Opt-in predevelopment boundary and block-aligned AMR translation.

Does not change the fixed-tube Engine. Boundary is zero normal gradient at
both x ends, with the existing reflecting slip walls in y. It is NOT claimed
to be a nonreflecting characteristic boundary; tail-length tests are required.
"""
import numpy as np
import hashlib
from pathlib import Path
import jax
import jax.numpy as jnp
from jaxamr import amr
from .solver import Engine
from .window import WindowFrame, translate_uniform


def block_coordinates(info, level):
    """Global same-level block indices, excluding capacity padding."""
    n=int(info['number']);g=np.asarray(info['glob_index'])[:n]
    if g.shape!=(n,2*(level+1)) or np.any(g<0):raise ValueError('Invalid AMR global index')
    if np.any(g[:,:2]):raise ValueError('Expected one base block')
    if level==0:return np.zeros((n,2),dtype=int)
    c=g[:,2:4].copy()
    for k in range(2,level+1):
        bits=g[:,2*k:2*k+2]
        if np.any(bits>1):raise ValueError('Invalid child index')
        c=2*c+bits
    if len({tuple(v) for v in c})!=n:raise ValueError('Duplicate active AMR blocks')
    return c


class PredevelopmentEngine(Engine):
    """Separate engine: never attach a reflecting wall to the moving tail."""
    def __init__(self,config,mechanism,amplitude=0.):
        super().__init__(config,mechanism,amplitude=amplitude)
        self.cycle_budget=jax.jit(self._cycle_budget)
        sources={}
        for prefix,root in (('janc_v2',Path(__file__).resolve().parents[1]),('jaxamr',Path(amr.__file__).resolve().parent)):
            for p in sorted(root.rglob('*.py')):
                sources[prefix+'/'+p.relative_to(root).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        self.checkpoint_identity=dict(schema='predevelopment_window_checkpoint_v1',
            mechanism_sha256=hashlib.sha256(Path(mechanism).read_bytes()).hexdigest(),
            source_sha256=sources,boundary='x_zero_gradient_y_slip',
            jax_version=jax.__version__,x64=bool(jax.config.jax_enable_x64))

    def _check_window_metadata(self,extra,t,step):
        if 'window' not in extra:raise ValueError('Predevelopment checkpoint requires window coordinates')
        frame=WindowFrame.from_dict(extra['window'])
        if (frame.nx,frame.ny,frame.dx,frame.dy)!=(self.c.nx,self.c.ny,self.dx[0],self.dy[0]):
            raise ValueError('Checkpoint window/config mismatch')
        if frame.translations:
            receipt=extra.get('translation',{})
            if (receipt.get('frame_after')!=frame.to_dict() or receipt.get('step',step+1)>step
                    or receipt.get('physical_time_s',t+1)>t):
                raise ValueError('Missing or inconsistent last translation receipt')

    def save(self,path,states,infos,t,step,extra=None):
        extra=dict(extra or {})
        self._check_window_metadata(extra,t,step)
        if 'predevelopment_identity' in extra and extra['predevelopment_identity']!=self.checkpoint_identity:
            raise ValueError('Cannot overwrite checkpoint identity')
        assert_composite_restricted(self,states,infos)
        extra['predevelopment_identity']=self.checkpoint_identity
        return super().save(path,states,infos,t,step,extra)

    def load(self,path):
        states,infos,meta=super().load(path)
        extra=meta['extra']
        if extra.get('predevelopment_identity')!=self.checkpoint_identity:
            raise ValueError('Predevelopment source/mechanism/boundary/runtime identity mismatch')
        self._check_window_metadata(extra,meta['t'],meta['step'])
        assert_composite_restricted(self,states,infos)
        return states,infos,meta

    def _cycle_budget(self,states,infos,dt):
        states,(fx,fy)=self._cycle_with_flux(states,infos,dt)
        # Fluxes already include RK weights, fine substeps, and reflux.
        # Positive sign means transport INTO the window, per unit depth.
        transfer=(jnp.sum(fx[0,:,0,:]-fx[0,:,-1,:],axis=-1)*self.dy[0]
                  +jnp.sum(fy[0,:,:,0]-fy[0,:,:,-1],axis=-1)*self.dx[0])
        return states,transfer

    def _ghost(self,level,s,info,parent_g=None):
        g=super()._ghost(level,s,info,parent_g)
        if level==0:
            return g.at[0,:,:3,3:-3].set(jnp.broadcast_to(s[0,:,:1,:],(s.shape[1],3,s.shape[-1])))
        coord=jnp.zeros(s.shape[0],dtype=jnp.int64)
        for k in range(level+1):coord=coord*self.nblock[k][0]+info['glob_index'][:,2*k]
        left=jnp.broadcast_to(s[:,:,:1,:],(s.shape[0],s.shape[1],3,s.shape[-1]))
        active=jnp.arange(s.shape[0])<info['number']
        return g.at[:,:,:3,3:-3].set(jnp.where(((coord==0)&active)[:,None,None,None],left,g[:,:,:3,3:-3]))


def assert_composite_restricted(engine,states,infos):
    px,py=engine.c.patch_x,engine.c.patch_y
    for l in range(1,len(states)):
        coarse=np.asarray(states[l-1]);fine=np.asarray(states[l]);info=infos[l]
        for i,(b,r,c) in enumerate(np.asarray(info['index'])[:int(info['number'])]):
            if min(b,r,c)<0 or b>=int(infos[l-1]['number']):raise ValueError('Invalid parent')
            mean=fine[i,:12].reshape(12,px,2,py,2).mean(axis=(2,4))
            old=coarse[b,:12,r*px:(r+1)*px,c*py:(c+1)*py]
            if old.shape!=mean.shape or np.max(abs(mean-old)/np.maximum(abs(mean),1))>1e-11:
                raise ValueError('Hierarchy must be conservatively restricted at an accepted coarse step')


def translate_hierarchy(engine,states,infos,frame,shift_cells,**policy):
    """Translate accepted hierarchy without derefining retained blocks.

Host reference implementation: block data are copied at translation events.
Only full base-patch-aligned shifts are supported. Regridding is a separate
operation AFTER this transaction; it must not hide translation discrepancies.
"""
    c=engine.c
    if not isinstance(engine,PredevelopmentEngine):raise ValueError('Use the dedicated open-tail engine')
    if (frame.nx,frame.ny,frame.dx,frame.dy)!=(c.nx,c.ny,engine.dx[0],engine.dy[0]):raise ValueError('Frame/config mismatch')
    if len(states)!=c.levels+1 or len(infos)!=len(states):raise ValueError('Missing hierarchy level')
    if not isinstance(shift_cells,(int,np.integer)) or isinstance(shift_cells,bool) or shift_cells%c.patch_x:
        raise ValueError('AMR translation must align with base patch_x')
    assert_composite_restricted(engine,states,infos)
    base,newframe,receipt=translate_uniform(np.asarray(states[0][0]),np.asarray(engine.s0),frame,shift_cells,**policy)
    newstates=[jnp.asarray(base[None])];newinfos=[infos[0]]
    previous_lookup={(0,0):0};counts=[]
    for l in range(1,c.levels+1):
        oldcoords=block_coordinates(infos[l],l)
        shifted=oldcoords.copy();shifted[:,0]-=(shift_cells//c.patch_x)*2**(l-1)
        keep=np.flatnonzero(shifted[:,0]>=0)
        maxcoord=np.array([c.nx//c.patch_x,c.ny//c.patch_y])*2**(l-1)
        if np.any(shifted[keep]>=maxcoord):raise ValueError('Out-of-domain block')
        entries=[]
        for oldi in keep:
            coord=shifted[oldi];parent=(0,0) if l==1 else tuple(coord//2)
            if parent not in previous_lookup:raise ValueError('Retained child lost parent')
            rc=coord if l==1 else coord%2
            entries.append((previous_lookup[parent],int(rc[0]),int(rc[1]),int(oldi),tuple(coord)))
        entries.sort(key=lambda e:e[:3]);n=len(entries);cap=states[l].shape[0]
        if n>=cap:raise MemoryError('No capacity sentinel available')
        old=np.asarray(states[l]);new=np.broadcast_to(np.asarray(engine.s0)[None,:,None,None],old.shape).copy()
        index=np.full((cap,3),-1,dtype=np.int64);glob=np.full((cap,2*(l+1)),-1,dtype=np.int64)
        neighbor=np.full((cap,4),-1,dtype=np.int64);lookup={e[4]:i for i,e in enumerate(entries)}
        for i,(b,r,k,oldi,coord) in enumerate(entries):
            new[i]=old[oldi];index[i]=(b,r,k)
            glob[i]=np.r_[np.asarray(newinfos[l-1]['glob_index'])[b],r,k]
            x,y=coord
            neighbor[i]=[lookup.get(v,-1) for v in ((x-1,y),(x+1,y),(x,y-1),(x,y+1))]
        info=dict(number=jnp.array(n),index=jnp.asarray(index),glob_index=jnp.asarray(glob),neighbor_index=jnp.asarray(neighbor))
        newstates.append(jnp.asarray(new));newinfos.append(info);previous_lookup=lookup
        counts.append(dict(level=l,before=int(infos[l]['number']),after=n,discarded=int(infos[l]['number'])-n,
                           retained_blocks_copied_exactly=True))
    assert_composite_restricted(engine,newstates,newinfos)
    receipt.update(kind='block_aligned_amr_translation_v1',levels=counts,
        implementation='Host block copy; numerical stages unchanged; no regrid during translation')
    return tuple(newstates),tuple(newinfos),newframe,receipt
