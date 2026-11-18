"""Diagnostic predevelopment runner. Completion is NOT convergence or CJ PASS.

Use one fresh process per configuration. Explicit diagnostic translation permits
tail-length tests before physical isolation is established; it records that fact.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time
import uuid
import numpy as np


def write_json(path,data):
    tmp=path.with_suffix('.writing.json');tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    for key in ('config','experiment','mechanism','reference','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--mirror',type=Path);p.add_argument('--resume',type=Path)
    p.add_argument('--device',choices=('cpu','gpu'),default='cpu')
    p.add_argument('--t-end',type=float,required=True);p.add_argument('--wall-limit-s',type=float,default=1800)
    p.add_argument('--save-every-s',type=float,default=60)
    a=p.parse_args()
    import jax
    jax.config.update('jax_enable_x64',True);jax.config.update('jax_platforms','cuda' if a.device == 'gpu' else a.device)
    import jax.numpy as jnp
    import cantera as ct
    from .core import Config
    from .predevelopment import PredevelopmentEngine,translate_hierarchy
    from .window import WindowFrame
    from .znd_seed import ZNDSeed
    from .strip import FrontRecorder
    if not np.isfinite(a.t_end) or a.t_end<=0 or min(a.wall_limit_s,a.save_every_s)<=0:raise ValueError('Invalid limits')
    cfg=Config(**json.loads(a.config.read_text()));experiment=json.loads(a.experiment.read_text())
    expected={'shock_initial_m','shift_at_s','shift_cells','minimum_tail_m','ahead_margin_m','diagnostic_interval_s'}
    if set(experiment)!=expected:raise ValueError('Experiment schema mismatch')
    if not 0<cfg.lx<=.3:raise ValueError('Predevelopment window must be <=30 cm')
    if experiment['shift_at_s'] is not None and (not np.isfinite(experiment['shift_at_s']) or experiment['shift_at_s']<0):raise ValueError('Invalid shift time')
    if experiment['diagnostic_interval_s']<=0:raise ValueError('Invalid diagnostic interval')
    # No phase directory overwrite or silent restart. Resume uses checkpoint metadata.
    if not a.resume and (a.out.exists() or (a.mirror and a.mirror.exists())):raise FileExistsError('Use a new case directory or an explicit checkpoint')
    e=PredevelopmentEngine(cfg,a.mechanism);seed=ZNDSeed(e,a.mechanism,a.reference)
    contract=dict(config=asdict(cfg),experiment=experiment,seed=seed.manifest,
                  cantera_version=ct.__version__,numpy_version=np.__version__)
    # Normalize tuples as JSON lists before checkpoint comparison.
    contract=json.loads(json.dumps(contract))
    frame=WindowFrame(cfg.nx,cfg.ny,e.dx[0],e.dy[0]);history=[];events=[]
    boundary=np.zeros(12);remap=np.zeros(12);regrid_sum=np.zeros(12);rejected=0
    t=0.;step=0;last_translation=None;wall_before=0.
    if a.resume:
        states,infos,m=e.load(a.resume);extra=m['extra']
        if extra['run_contract']!=contract:raise ValueError('Resume experiment/reference/version mismatch')
        if extra['status']=='failed':raise ValueError('Failed checkpoint needs diagnosis, not continuation')
        t=m['t'];step=m['step'];frame=WindowFrame.from_dict(extra['window']);history=extra['history'];events=extra['events']
        boundary=np.array(extra['boundary_transfer']);remap=np.array(extra['remap_transfer']);regrid_sum=np.array(extra['regrid_change'])
        initial=np.array(extra['initial_inventory']);last_translation=extra.get('translation');wall_before=extra['wall_s'];rejected=extra['rejected_steps']
        if t>=a.t_end:print('Checkpoint already at requested time; no integration performed.',flush=True);return
    else:
        states,infos,seed_init=seed.initial(experiment['shock_initial_m'])
        initial=np.asarray(e.probes(states,infos)[1])
    for root in (a.out,a.mirror):
        if root:root.mkdir(parents=True,exist_ok=True)
    write_json(a.out/'manifest.json',dict(contract=contract,identity=e.checkpoint_identity,device=[str(d) for d in jax.devices()],
        physics_note='Planar ZND startup; open-tail diagnostic. No mature 2D seed or convergence claim.'))
    segment='segments/'+uuid.uuid4().hex
    for root in (a.out,a.mirror):
        if root:(root/segment).mkdir(parents=True)
    rec=FrontRecorder(e,a.mechanism,a.out/segment,a.mirror/segment if a.mirror else None,
                      interval=experiment['diagnostic_interval_s'],start_x=0,raw_interval=experiment['diagnostic_interval_s'])
    gas=ct.Solution(str(a.mechanism));projection=np.zeros((7,12));projection[:4,:4]=np.eye(4)
    for j,name in enumerate(('H','O','N')):
        weight=np.array([gas.n_atoms(k,name)*gas.atomic_weight(name)/gas.molecular_weights[k] for k in range(9)])
        projection[4+j,0]=weight[8];projection[4+j,4:]=weight[:8]-weight[8]
    labels=['mass','x_momentum','y_momentum','total_energy','element_H','element_O','element_N']
    start=time.perf_counter();last_save=start;stop=[False];status='running';reason=''
    signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    maximum_budget_error=0.;maximum_regrid_error=0.
    if a.resume:
        maximum_budget_error=extra['maximum_budget_error'];maximum_regrid_error=extra['maximum_regrid_error']
    def budget():
        total=np.asarray(e.probes(states,infos)[1]);residual=projection@(total-initial-boundary-remap)
        scale=np.maximum(np.abs(projection@initial),1e-12)
        # Momentum may initially be zero: scale by mass * CJ velocity instead.
        scale[1:3]=max(abs(initial[0])*seed.speed,1e-12)
        return total,residual,float(np.max(abs(residual)/scale))
    def observe(force=False):
        nonlocal maximum_budget_error
        before=len(rec.rows);rec.observe(states,infos,t,step,force=force)
        total,residual,error=budget();maximum_budget_error=max(maximum_budget_error,error)
        if len(rec.rows)>before:
            row=dict(rec.rows[-1]);row.update(file=segment+'/front_diagnostics/'+row['file'],window_origin_m=frame.origin_cells*frame.dx,
                origin_cells=frame.origin_cells,shock_mean_lab_m=float(frame.lab_x(row['shock_mean_x_m'])),
                budget_residual=residual.tolist(),budget_scaled_error=error,blocks=[int(i['number']) for i in infos])
            history.append(row);print(json.dumps(row),flush=True)
        if error>1e-7:raise FloatingPointError('Open-window mass/momentum/energy/element budget failed')
    def save():
        nonlocal last_save
        _,residual,error=budget();wall=wall_before+time.perf_counter()-start
        extra=dict(window=frame.to_dict(),run_contract=contract,status=status,history=history,events=events,
            boundary_transfer=boundary.tolist(),remap_transfer=remap.tolist(),regrid_change=regrid_sum.tolist(),
            initial_inventory=initial.tolist(),wall_s=wall,rejected_steps=rejected,
            maximum_budget_error=maximum_budget_error,maximum_regrid_error=maximum_regrid_error)
        if last_translation is not None:extra['translation']=last_translation
        e.save(a.out/'checkpoint_latest.npz',states,infos,t,step,extra=extra)
        summary=dict(status=status,reason=reason,t_s=t,step=step,wall_s=wall,window=frame.to_dict(),
            translations=len(events),rejected_steps=rejected,budget_order=labels,budget_residual=residual.tolist(),
            maximum_budget_error=maximum_budget_error,maximum_regrid_error=maximum_regrid_error,
            cj_reference_m_s=seed.speed,detonation_convergence_validated=False,tail_isolation_validated=False)
        write_json(a.out/'summary.json',summary);write_json(a.out/'history.json',history);write_json(a.out/'events.json',events)
        if a.mirror:
            for name in ('checkpoint_latest.npz','summary.json','history.json','events.json','manifest.json'):
                dest=a.mirror/name;tmp=dest.with_name(dest.name+'.partial');shutil.copy2(a.out/name,tmp);tmp.replace(dest)
        last_save=time.perf_counter()
    try:
        health=np.asarray(e.metrics(states,infos))
        if not bool(health[1]):raise FloatingPointError('Unhealthy initial/resumed state: '+str(health))
        # On resume preserve old observations; the new segment cannot overwrite them.
        if not history or history[-1]['step']!=step:observe(force=True)
        else:rec.next_t=t+experiment['diagnostic_interval_s']
        save()
        while t<a.t_end:
            if stop[0]:status='interrupted';reason='Accepted checkpoint saved';break
            if time.perf_counter()-start>a.wall_limit_s:status='paused';reason='Wall budget reached';break
            shift_at=experiment['shift_at_s']
            if shift_at is not None and not events and t>=shift_at:
                # Trial operation explicitly records that physical tail isolation is unvalidated.
                front=float(e.probes(states,infos)[0])
                states,infos,frame,last_translation=translate_hierarchy(e,states,infos,frame,experiment['shift_cells'],
                    shock_range_local=(front,front),minimum_tail_m=experiment['minimum_tail_m'],ahead_margin_m=experiment['ahead_margin_m'],
                    tail_isolation_verified=False,diagnostic_unvalidated_tail=True,physical_time_s=t,step=step)
                remap+=np.array(last_translation['added'])-np.array(last_translation['removed']);events.append(last_translation)
                save()
            health=np.asarray(e.metrics(states,infos));dt=min(float(health[0]),a.t_end-t)
            if shift_at is not None and not events and t<shift_at:dt=min(dt,shift_at-t)
            for attempt in range(9):
                candidate,transfer=e.cycle_budget(states,infos,jnp.array(dt));check=np.asarray(e.metrics(candidate,infos))
                if bool(check[1]):break
                dt*=.5;rejected+=1
            else:raise FloatingPointError('Repeated rejected steps; last health: '+str(check))
            states=candidate;boundary+=np.asarray(transfer);t+=dt;step+=1
            if step%cfg.amr_every==0:
                before=np.asarray(e.probes(states,infos)[1]);states,infos=e.regrid(states,infos)
                after=np.asarray(e.probes(states,infos)[1]);change=after-before;regrid_sum+=change
                err=float(np.max(abs(change)/np.maximum(abs(before),1e-12)));maximum_regrid_error=max(maximum_regrid_error,err)
                if err>1e-7:raise FloatingPointError('Regrid inventory changed')
            observe()
            if time.perf_counter()-last_save>=a.save_every_s:save()
        else:status='completed_time';reason='Diagnostic physical-time limit, not a convergence PASS'
        observe(force=True)
    except Exception:
        import traceback
        status='failed';reason=traceback.format_exc();raise
    finally:
        save();print(json.dumps(json.loads((a.out/'summary.json').read_text())),flush=True)


if __name__=='__main__':main()
