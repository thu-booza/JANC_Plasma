"""Explicit migration into a separate stage; never reset historic inventory."""
from pathlib import Path
from dataclasses import asdict
import argparse,copy,json,shutil
import numpy as np
from verified_store import resolve,commit,sha,inside,atomic_json
from long_contract import contract_for,projection,seal_event,verify_ledger
from extend_domain import append_downstream,validate_hierarchy,array_digest


def migrate(source,out,runtime,plan):
    import jax
    jax.config.update('jax_enable_x64',True)
    from restriction_seed import RestrictionSeedEngine
    from janc_v2.detonation_amr.core import Config
    source,out,runtime=(Path(p).resolve() for p in (source,out,runtime))
    if source.is_relative_to(out) or out.is_relative_to(source):raise ValueError('Use a separate output stage')
    if out.exists():raise FileExistsError('Migration output exists; do not overwrite')
    cp,m,manifest=resolve(source);ex=m['extra'];old_cfg=m['config'];old_plan=ex['contract']['plan']
    expected=json.loads((runtime/'long_source.json').read_text())
    if ex.get('long_run_receipt'):
        if ex['contract']!=contract_for(runtime,old_plan):raise ValueError('Existing branch runtime changed')
        verify_ledger(ex,old_cfg,m['t'],m['step'])
        if ex['status'] not in ('paused','needs_extension','needs_capacity','running'):raise ValueError('Stage status cannot migrate')
    elif manifest['checkpoint_sha256']!=expected['checkpoint_sha256'] or m['step']!=expected['step'] or m['t']!=expected['t_s']:
        raise ValueError('Not the explicitly audited original checkpoint')
    if plan['physics']!=old_plan['physics'] or plan['amr']!=old_plan['amr']:raise ValueError('Physical model/refinement policy change prohibited')
    cfg=Config(**plan['config']);cfg.validate();new_cfg=asdict(cfg)
    for key,value in old_cfg.items():
        if key not in ('nx','lx','capacities') and value!=new_cfg[key]:raise ValueError('Unexpected numerical change: '+key)
    if cfg.lx/cfg.nx!=old_cfg['lx']/old_cfg['nx'] or cfg.nx<old_cfg['nx'] or any(a<b for a,b in zip(cfg.capacities,old_cfg['capacities'])):
        raise ValueError('Spacing/domain/capacity mismatch')
    reference=json.loads((runtime/'reference.json').read_text())
    if sha(runtime/'mechanism.yaml')!=ex['contract']['mechanism_sha256'] or sha(runtime/'reference.json')!=ex['contract']['reference_sha256']:
        raise ValueError('Mechanism/reference changed')
    e=RestrictionSeedEngine(cfg,runtime/'mechanism.yaml',plan['physics'],plan['amr'])
    with np.load(cp,allow_pickle=False) as z:
        states=tuple(z[f'L{l}_state'] for l in range(cfg.levels+1))
        infos=tuple({k:z[f'L{l}_{k}'] for k in ('number','index','glob_index','neighbor_index')} for l in range(cfg.levels+1))
    validate_hierarchy(states,infos,old_cfg);fresh=np.asarray(e.s0)
    before=states[0][0,:12].sum(axis=(1,2))*e.dx[0]*e.dy[0]
    if cfg.nx>old_cfg['nx']:
        states,infos,checked_cfg,event=append_downstream(states,infos,old_cfg,fresh,new_nx=cfg.nx,new_capacities=cfg.capacities)
        assert json.dumps(checked_cfg,sort_keys=True)==json.dumps(new_cfg,sort_keys=True)
    else:
        changed=[];changed_info=[];proof=[]
        for l,(s,i) in enumerate(zip(states,infos)):
            n=int(i['number']);cap=1 if l==0 else cfg.capacities[l-1]
            a=np.broadcast_to(fresh[None,:,None,None],(cap,*s.shape[1:])).copy();a[:n]=s[:n]
            ni={'number':i['number'].copy()}
            for k in ('index','glob_index','neighbor_index'):
                ni[k]=np.full((cap,i[k].shape[1]),-1,dtype=i[k].dtype);ni[k][:n]=i[k][:n]
            changed.append(a);changed_info.append(ni);proof.append(dict(level=l,source_sha256=array_digest(s[:n]),retained_sha256=array_digest(a[:n])))
        states,infos=tuple(changed),tuple(changed_info)
        event=dict(schema='fixed_left_capacity_growth_v1',old_config=old_cfg,new_config=new_cfg,added_conserved=[0.]*12,fresh_state=fresh.tolist(),retained_fields=proof)
    validate_hierarchy(states,infos,new_cfg)
    event.update(step=m['step'],t_s=m['t'],source_checkpoint_sha256=manifest['checkpoint_sha256'])
    nx=copy.deepcopy(ex);ledger=nx.get('domain_ledger',[]);event=seal_event(event,ledger[-1]['event_hash'] if ledger else '');ledger.append(event)
    added=sum((np.array(v['added_conserved']) for v in ledger),np.zeros(12))
    old_added=np.array(ex.get('domain_added',[0.]*12));initial=np.array(ex['initial']);P,scale=projection(runtime/'mechanism.yaml',initial,reference['cj_speed_m_s'])
    after=states[0][0,:12].sum(axis=(1,2))*e.dx[0]*e.dy[0]
    old_res=P@(before-initial-np.array(ex['boundary'])-np.array(ex['remap'])-old_added)/scale
    new_res=P@(after-initial-np.array(ex['boundary'])-np.array(ex['remap'])-added)/scale
    if np.max(abs(new_res-old_res))>1e-10:raise ValueError('Migration changed normalized conservation residual')
    if np.max(abs(new_res))>plan['run']['budget_limit']:raise ValueError('Source/migrated budget exceeds limit')
    nx.update(status='paused',contract=contract_for(runtime,plan),domain_added=added.tolist(),domain_ledger=ledger,
              predevelopment_identity=e.checkpoint_identity,maximum_budget=max(ex['maximum_budget'],float(np.max(abs(new_res)))))
    nx['window'].update(nx=cfg.nx,ny=cfg.ny,dx=e.dx[0],dy=e.dy[0])
    nx['long_run_receipt']=ex.get('long_run_receipt') or dict(schema='closed_h5_long_run_v1',source_checkpoint_sha256=manifest['checkpoint_sha256'],source_t_s=m['t'],source_step=m['step'],original_initial=ex['initial'],original_contract=ex['contract'],method_change='covered-parent thermodynamic inversion uses finite fine-temperature seed')
    nx.setdefault('stability_confirmation',{})
    verify_ledger(nx,new_cfg,m['t'],m['step'])
    health=np.asarray(e.accept_metrics(tuple(__import__('jax').numpy.asarray(s) for s in states),tuple({k:__import__('jax').numpy.asarray(v) for k,v in i.items()} for i in infos)))
    if not health[1]:raise ValueError('Migrated state unhealthy')
    out.mkdir(parents=True);atomic_json(out/'migration_in_progress.json',dict(source=str(source),step=m['step'],t_s=m['t']))
    # Immutable old diagnostics may be absent in an export; record absence, never invent frames.
    missing=[]
    for row in ex['history']:
        src=inside(source,row['file']);dst=inside(out,row['file'])
        if src.is_file() and not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        elif not src.is_file():missing.append(row['file'])
    if (source/'latest_fields.npz').is_file():shutil.copy2(source/'latest_fields.npz',out/'latest_fields.npz')
    temp=out/'migration_state.npz';e.save(temp,states,infos,m['t'],m['step'],nx)
    summary=dict(status='paused',reason='Verified long-run stage migration; no physical time advanced',step=m['step'],t_s=m['t'],wall_s=nx['wall_s'],maximum_budget=nx['maximum_budget'],budget_scaled=new_res.tolist(),rejected_steps=nx['rejected_steps'],physical_convergence_pass=False,blocks=[int(i['number']) for i in infos],translations=0,domain_added=added.tolist())
    result=commit(temp,out,summary,nx['history'],nx['contract'])
    atomic_json(out/'migration_receipt.json',dict(event_hash=event['event_hash'],old_budget=old_res.tolist(),new_budget=new_res.tolist(),missing_optional_frames=missing,initial_preserved=nx['initial']==ex['initial'],time_preserved=True,source_checkpoint_sha256=manifest['checkpoint_sha256']))
    atomic_json(out/'case.json',plan);temp.unlink();(out/'migration_in_progress.json').unlink()
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for key in ('source','out','runtime','case'):ap.add_argument('--'+key,type=Path,required=True)
    args=ap.parse_args();print(json.dumps(migrate(args.source,args.out,args.runtime,json.loads(args.case.read_text())),indent=2))
