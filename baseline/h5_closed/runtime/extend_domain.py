"""Append a cold downstream domain without translating the closed left wall.

Host-side geometry transaction only. The caller must explicitly migrate the
runtime contract and account receipt['added_conserved'] in its budget. This
module does not alter initial inventory, boundary history, chemistry or time.
"""
from copy import deepcopy
import hashlib
import numpy as np


def _coordinates(info,level):
    n=int(info['number']);g=np.asarray(info['glob_index'])[:n]
    if g.shape!=(n,2*(level+1)) or np.any(g<0) or np.any(g[:,:2]):
        raise ValueError('Invalid global block indices')
    c=g[:,2:4].copy() if level else np.zeros((n,2),dtype=np.int64)
    for k in range(2,level+1):
        bits=g[:,2*k:2*k+2]
        if np.any(bits>1):raise ValueError('Invalid child coordinate')
        c=2*c+bits
    if len(set(map(tuple,c)))!=n:raise ValueError('Duplicate active block')
    return c


def validate_hierarchy(states,infos,config):
    if len(states)!=config['levels']+1 or len(infos)!=len(states):
        raise ValueError('Incomplete hierarchy')
    px,py=config['patch_x'],config['patch_y']
    for level,(s,info) in enumerate(zip(states,infos)):
        n=int(info['number']);capacity=1 if level==0 else config['capacities'][level-1]
        shape=(capacity,14,config['nx'],config['ny']) if level==0 else (capacity,14,2*px,2*py)
        if s.shape!=shape or s.dtype!=np.float64 or n<1 or n>capacity or (level and n>=capacity):
            raise ValueError('Invalid state shape, dtype, active count or missing sentinel')
        for key,width in [('index',3),('glob_index',2*(level+1)),('neighbor_index',4)]:
            a=np.asarray(info[key])
            if a.shape!=(capacity,width) or not np.issubdtype(a.dtype,np.integer):
                raise ValueError('Invalid topology shape/dtype: '+key)
            if level and np.any((a[n:,-2:] if key=='glob_index' else a[n:])!=-1):
                raise ValueError('Nonempty topology padding')
        coords=_coordinates(info,level)
        lookup={tuple(c):k for k,c in enumerate(coords)}
        if level==0:
            if n!=1 or np.any(info['index']!=0) or np.any(info['neighbor_index']!=-1):
                raise ValueError('Invalid base topology')
        else:
            bounds=np.array([config['nx']//px,config['ny']//py])*2**(level-1)
            if np.any(coords>=bounds):raise ValueError('Block outside physical domain')
            for k,(b,r,c) in enumerate(np.asarray(info['index'])[:n]):
                if min(b,r,c)<0 or b>=int(infos[level-1]['number']):raise ValueError('Invalid parent index')
                parent=states[level-1][b]
                if (r+1)*px>parent.shape[-2] or (c+1)*py>parent.shape[-1]:raise ValueError('Parent tile out of bounds')
                expected=np.r_[infos[level-1]['glob_index'][b],r,c]
                if not np.array_equal(info['glob_index'][k],expected):raise ValueError('Parent/global index mismatch')
                x,y=coords[k]
                nb=[lookup.get(v,-1) for v in ((x-1,y),(x+1,y),(x,y-1),(x,y+1))]
                if not np.array_equal(info['neighbor_index'][k],nb):raise ValueError('Neighbor mismatch')
                restricted=s[k,:12].reshape(12,px,2,py,2).mean(axis=(2,4))
                old=parent[:12,r*px:(r+1)*px,c*py:(c+1)*py]
                if np.max(abs(restricted-old)/np.maximum(abs(restricted),1))>1e-11:
                    raise ValueError('Parent not conservatively restricted')
        # Inspect active blocks in batches to avoid full-hierarchy temporary arrays.
        for start in range(0,n,8):
            a=s[start:min(start+8,n)]
            if not np.isfinite(a).all() or np.min(a[:,0])<=0 or np.min(a[:,-1])<=100:
                raise ValueError('Unhealthy active state')
            Y=a[:,4:12]/a[:,:1]
            if Y.min()<-1e-13 or (1-Y.sum(axis=1)).min()<-1e-13:
                raise ValueError('Invalid mass fractions')


def array_digest(a):
    h=hashlib.sha256()
    # Works on the noncontiguous original region of an extended base array.
    for block in a:
        h.update(np.ascontiguousarray(block).view(np.uint8))
    return h.hexdigest()


def append_downstream(states,infos,config,fresh,*,new_nx,new_capacities=None,tail_cells=128,tail_rtol=1e-10):
    """Return new host arrays/config and an explicit inventory receipt.

    Only exactly fresh downstream material may be appended. An already affected
    old outlet or refined block in the guarded tail is rejected. Original active
    arrays and topology are copied exactly; coordinates retain their meaning
    because dx, dy, patch dimensions and all parent-child indices are unchanged.
    """
    validate_hierarchy(states,infos,config)
    if isinstance(new_nx,bool) or not isinstance(new_nx,(int,np.integer)) or new_nx<=config['nx'] or new_nx%config['patch_x']:
        raise ValueError('New extent must increase by whole base patches')
    fresh=np.asarray(fresh)
    if fresh.shape!=(14,) or fresh.dtype!=np.float64 or not np.isfinite(fresh).all() or fresh[0]<=0 or fresh[-1]<=100:
        raise ValueError('Invalid fresh gas state')
    if np.any(fresh[1:3]!=0):raise ValueError('Fresh downstream gas must be stationary')
    if not 6<=tail_cells<=config['nx'] or tail_rtol<=0:raise ValueError('Invalid downstream guard')
    tail=states[0][0,:,-tail_cells:,:]
    scale=np.maximum(abs(fresh),1.)
    departure=float(np.max(abs(tail-fresh[:,None,None])/scale[:,None,None]))
    if departure>tail_rtol:raise ValueError('Old outlet is not fresh; extending now changes an affected boundary')
    for level,info in enumerate(infos[1:],1):
        end=(_coordinates(info,level)[:,0]+1)*2*config['patch_x']/2**level
        if np.any(end>config['nx']-tail_cells):raise ValueError('Refined block intersects downstream guard')
    caps=tuple(config['capacities']) if new_capacities is None else tuple(new_capacities)
    if len(caps)!=config['levels'] or any(isinstance(a,bool) or not isinstance(a,(int,np.integer)) or a<b for a,b in zip(caps,config['capacities'])):
        raise ValueError('Capacities may only be preserved or increased')
    dx=config['lx']/config['nx'];dy=config['ly']/config['ny']
    new=deepcopy(config);new.update(nx=int(new_nx),lx=float(new_nx*dx),capacities=list(map(int,caps)))
    if new['lx']/new['nx']!=dx:raise ValueError('Extension would change the grid spacing')
    base=np.broadcast_to(fresh[None,:,None,None],(1,14,new_nx,config['ny'])).copy()
    base[:,:,:config['nx'],:]=states[0]
    out=[base];outinfo=[{k:np.array(v,copy=True) for k,v in infos[0].items()}]
    proofs=[dict(level=0,active_blocks=1,source_sha256=array_digest(states[0]),retained_sha256=array_digest(base[:,:,:config['nx'],:]))]
    for level in range(1,len(states)):
        s=states[level];info=infos[level];n=int(info['number']);cap=caps[level-1]
        a=np.broadcast_to(fresh[None,:,None,None],(cap,*s.shape[1:])).copy();a[:n]=s[:n]
        ni={'number':np.array(info['number'],copy=True)}
        for key in ('index','glob_index','neighbor_index'):
            ni[key]=np.full((cap,info[key].shape[1]),-1,dtype=info[key].dtype);ni[key][:n]=info[key][:n]
        out.append(a);outinfo.append(ni)
        proofs.append(dict(level=level,active_blocks=n,source_sha256=array_digest(s[:n]),retained_sha256=array_digest(a[:n])))
    validate_hierarchy(out,outinfo,new)
    if any(p['source_sha256']!=p['retained_sha256'] for p in proofs):raise AssertionError('Retained field changed')
    added=fresh[:12]*((new_nx-config['nx'])*config['ny']*dx*dy)
    old_total=states[0][0,:12].sum(axis=(1,2))*dx*dy
    new_total=out[0][0,:12].sum(axis=(1,2))*dx*dy
    residual=new_total-old_total-added
    # The momentum scale here is only a floating-point identity check, not the
    # physical conservation normalization used in the time integration runner.
    denom=np.maximum(np.maximum(abs(old_total),abs(added)),1.)
    if np.max(abs(residual)/denom)>1e-11:raise AssertionError('Added-domain inventory mismatch')
    receipt=dict(schema='fixed_left_downstream_extension_v1',old_config=deepcopy(config),new_config=new,
                 old_right_m=config['lx'],new_right_m=new['lx'],fixed_left_m=0.,dx_m=dx,dy_m=dy,
                 fresh_state=fresh.tolist(),guard_cells=tail_cells,guard_relative_departure=departure,
                 added_conserved=added.tolist(),inventory_roundoff=residual.tolist(),retained_fields=proofs,
                 active_topology_unchanged=True,initial_inventory_must_remain_unchanged=True,
                 note='Geometry transaction only; no ignition, translation, time advance, regrid or physical-convergence claim.')
    return tuple(out),tuple(outinfo),new,receipt
