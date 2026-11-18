"""Runtime contract and immutable domain-inventory ledger for closed H5."""
from pathlib import Path
import json,hashlib
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4194304),b''):h.update(block)
    return h.hexdigest()

def contract_for(root,plan):
    import jax,cantera
    root=Path(root)
    return dict(plan=plan,versions=dict(jax=jax.__version__,numpy=np.__version__,cantera=cantera.__version__),
        source={p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*.py'))},
        mechanism_sha256=sha(root/'mechanism.yaml'),reference_sha256=sha(root/'reference.json'))

def projection(mechanism,initial,cj):
    import cantera as ct
    gas=ct.Solution(str(mechanism));P=np.zeros((7,12));P[:4,:4]=np.eye(4)
    for j,n in enumerate(('H','O','N')):
        w=np.array([gas.n_atoms(k,n)*gas.atomic_weight(n)/gas.molecular_weights[k] for k in range(9)])
        P[4+j,0]=w[8];P[4+j,4:]=w[:8]-w[8]
    scale=np.maximum(abs(P@initial),1e-12);scale[1:3]=initial[0]*cj
    return P,scale

def canonical_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def seal_event(event,previous_hash):
    event=dict(event,previous_hash=previous_hash)
    return dict(event,event_hash=canonical_digest(event))

def verify_ledger(extra,config,t,step):
    ledger=extra.get('domain_ledger');receipt=extra.get('long_run_receipt')
    if not receipt or receipt.get('schema')!='closed_h5_long_run_v1':raise ValueError('Missing long-run provenance')
    if ledger is None or not ledger:raise ValueError('Missing domain ledger')
    if receipt['original_initial']!=extra['initial']:raise ValueError('Original inventory was modified')
    if extra['events'] or np.any(extra['remap']) or extra['window']['origin_cells'] or extra['window']['translations']:
        raise ValueError('Closed-end physical frame must never translate')
    previous='';added=np.zeros(12);last_t=-1.;last_step=-1;last_config=None
    for event in ledger:
        check=dict(event);claimed=check.pop('event_hash')
        if event['previous_hash']!=previous or canonical_digest(check)!=claimed:raise ValueError('Domain ledger chain mismatch')
        if not last_t<=event['t_s']<=t or not last_step<=event['step']<=step:raise ValueError('Domain event clock mismatch')
        old,new=event['old_config'],event['new_config']
        if last_config is not None and old!=last_config:raise ValueError('Domain configuration discontinuity')
        for key in old:
            if key not in ('nx','lx','capacities') and old[key]!=new[key]:raise ValueError('Unapproved numerical/physical change')
        if new['nx']<old['nx'] or new['lx']/new['nx']!=old['lx']/old['nx'] or any(a<b for a,b in zip(new['capacities'],old['capacities'])):
            raise ValueError('Domain/mesh/capacity regression')
        inventory=np.array(event['added_conserved'],dtype=float)
        expected=np.array(event['fresh_state'][:12])*(new['nx']-old['nx'])*old['ny']*(old['lx']/old['nx'])*(old['ly']/old['ny'])
        if inventory.shape!=(12,) or not np.isfinite(inventory).all() or not np.allclose(inventory,expected,rtol=2e-14,atol=1e-20):
            raise ValueError('Incorrect appended-domain inventory')
        added+=inventory;previous=claimed;last_t=event['t_s'];last_step=event['step'];last_config=new
    if json.dumps(last_config,sort_keys=True)!=json.dumps(config,sort_keys=True):raise ValueError('Ledger/current config mismatch')
    if not np.array_equal(added,np.array(extra['domain_added'])):raise ValueError('Domain inventory total mismatch')
    return added
