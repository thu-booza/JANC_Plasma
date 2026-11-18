"""One continuous smooth-tube experiment: initialize once, save and resume."""
import argparse,hashlib,json,os,shutil,signal,time,uuid,traceback,csv
from pathlib import Path
import numpy as np


def write_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.writing')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


from long_statistics import screen,update_confirmation
from long_contract import contract_for,projection,verify_ledger


def main():
    parser=argparse.ArgumentParser()
    for name in ('root','out','mirror','resume','case'):
        parser.add_argument('--'+name,type=Path,required=name in ('root','out'))
    parser.add_argument('--device',choices=['cpu','gpu'],default='gpu')
    parser.add_argument('--target-us',type=float)
    parser.add_argument('--session-seconds',type=float)
    parser.add_argument('--max-steps',type=int,help='Local validation only; count accepted steps this session')
    args=parser.parse_args();started=time.monotonic();stop=[False]
    signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    import jax
    jax.config.update('jax_enable_x64',True);jax.config.update('jax_platforms','cuda' if args.device=='gpu' else 'cpu')
    import jax.numpy as jnp
    import cantera as ct
    from restriction_seed import RestrictionSeedEngine as H5Engine
    from step_control import StepControl
    from verified_store import commit,resolve
    from h5_diagnostics import H5Recorder
    from janc_v2.detonation_amr.predevelopment import translate_hierarchy
    from janc_v2.detonation_amr.core import Config,pressure
    from janc_v2.detonation_amr.window import WindowFrame
    root=args.root.resolve();plan=json.loads((args.case or root/'case.json').read_text());policy=plan['run']
    ref=json.loads((root/'reference.json').read_text());mech=root/'mechanism.yaml'
    if digest(mech)!=ref['mechanism_sha256']:raise ValueError('Reference/mechanism mismatch')
    if (plan['physics']['mole_ratio'],plan['config']['t0'],plan['config']['p0'])!=(ref['mixture'],ref['T_K'],ref['P_Pa']):
        raise ValueError('CJ reference is for a different initial mixture, T or p')
    cfg=Config(**plan['config']);cfg.validate()
    if plan['physics']['hot_u_m_s']!=0. or plan['run']['window_translation']:
        raise ValueError('This fixed closed-end case requires stationary initialization and no window translation')
    target=float('inf') if args.target_us is None else args.target_us*1e-6
    session_limit=policy['session_wall_s'] if args.session_seconds is None else args.session_seconds
    if target<=0 or session_limit<0:raise ValueError('Invalid time budget')
    if args.mirror and args.out.resolve()==args.mirror.resolve():raise ValueError('Local and mirror paths must differ')
    for path in (args.out,args.mirror):
        if path:path.mkdir(parents=True,exist_ok=True)
    if not args.resume and any((p/'latest.json').exists() for p in (args.out,args.mirror) if p):
        raise FileExistsError('Existing checkpoint: use resume, do not ignite again')
    contract=contract_for(root,plan)
    print('Building JANC engine; first compilation and initial checkpoint may take minutes.',flush=True)
    e=H5Engine(cfg,mech,plan['physics'],plan['amr'])
    frame=WindowFrame(cfg.nx,cfg.ny,e.dx[0],e.dy[0]);t=0.;step=0;history=[];events=[];wall_before=0.;rejected=0
    boundary=np.zeros(12);remap=np.zeros(12);chemistry=np.zeros(12);flow_defect=np.zeros(12);maximum_budget=0.
    next_diagnostic=0.;next_picture=0.;status='running';reason='';last_save=time.monotonic();session_steps=0
    if args.resume:
        states,infos,m=e.load(args.resume);ex=m['extra']
        if ex['contract']!=contract:raise ValueError('Restore identity mismatch; use the same runtime and case')
        if ex['status'] in ('needs_extension','needs_capacity','statistical_stability'):raise ValueError('Supervisor must handle this stage status before resuming')
        if ex['status']=='failed':raise ValueError('Failed state retained for diagnosis; not automatically resumed')
        t=m['t'];step=m['step'];frame=WindowFrame.from_dict(ex['window']);history=ex['history'];events=ex['events']
        initial=np.array(ex['initial']);boundary=np.array(ex['boundary']);remap=np.array(ex['remap'])
        chemistry=np.array(ex['chemistry']);flow_defect=np.array(ex['flow_defect']);wall_before=ex['wall_s']
        rejected=ex['rejected_steps'];maximum_budget=ex['maximum_budget'];next_diagnostic=ex['next_diagnostic'];next_picture=ex['next_picture']
        if events and (events[-1]['frame_after']!=frame.to_dict() or events[-1]['step']>step):raise ValueError('Invalid window ledger')
        counted=sum((np.array(ev['added'])-np.array(ev['removed']) for ev in events),np.zeros(12))
        if not np.allclose(counted,remap,rtol=1e-12,atol=1e-20):raise ValueError('Shift inventory mismatch')
        print(f'Resumed physical time {t*1e6:.6f} us at step {step}, {len(events)} shifts.',flush=True)
    else:
        raise ValueError('This entry only resumes a verified recovery branch; no new ignition')
    domain_added=verify_ledger(ex,m['config'],t,step)
    domain_ledger=ex['domain_ledger'];long_receipt=ex['long_run_receipt'];confirmation=ex.get('stability_confirmation',{})
    stage_request=None
    recovery=ex.get('recovery_receipt')
    if not recovery or recovery.get('schema')!='closed_crop_recovery_v1':raise ValueError('Missing recovery record')
    receipt=ex.get('branch_receipt')
    if not receipt or receipt.get('schema')!='h5_closed_stationary_v1':raise ValueError('Missing explicit method-migration receipt')
    controller=StepControl.restore(ex['step_control'])
    performance=ex['performance']
    perf=performance['categories_s']
    def add_time(name,elapsed):perf[name]=perf.get(name,0.)+elapsed
    add_time('session_initialization',time.monotonic()-started)
    P,scale=projection(mech,initial,ref['cj_speed_m_s'])
    last_regrid_step=ex.get('last_regrid_step',-1) if args.resume else -1
    segment='segments/'+uuid.uuid4().hex
    for path in (args.out,args.mirror):
        if path:(path/segment).mkdir(parents=True)
    recorder=H5Recorder(e,mech,args.out/segment,args.mirror/segment if args.mirror else None,
                        interval=policy['diagnostic_every_s'],start_x=0.,raw_interval=policy['picture_every_s'])
    recorder.next_t=next_diagnostic;recorder.next_raw=next_picture
    front_extent=jax.jit(lambda base:jnp.max(jnp.where(jnp.any(pressure(base)>2*cfg.p0,axis=1),jnp.arange(cfg.nx),-1))*e.dx[0])
    health=np.asarray(e.accept_metrics(states,infos))
    trial_path=args.out/segment/'trials.csv'
    trial_stream=trial_path.open('x',encoding='utf-8',newline='',buffering=1)
    writer=csv.writer(trial_stream)
    writer.writerow(['accepted_step_before','t_before_s','attempt','cfl_dt_s','candidate_dt_s','accepted','reason','cycle_wall_s','health_wall_s','blocks_L0','blocks_L1','blocks_L2'])
    last_progress=time.monotonic()
    def budget(total):
        nonlocal maximum_budget
        residual=P@(total-initial-boundary-remap-domain_added)/scale
        maximum_budget=max(maximum_budget,float(np.max(abs(residual))))
        return residual
    def push(name):
        if args.mirror:
            dst=args.mirror/name;dst.parent.mkdir(parents=True,exist_ok=True);temp=dst.with_suffix(dst.suffix+'.partial')
            shutil.copy2(args.out/name,temp);temp.replace(dst)
    def observe(force=False):
        nonlocal next_diagnostic,next_picture,confirmation
        tick=time.monotonic()
        row=recorder.sample(states,infos,t,step,frame.origin_cells*frame.dx,force=force)
        if row is None:return
        total=np.asarray(e.probes(states,infos)[1]);r=budget(total)
        row['file']=segment+'/front_diagnostics/'+row['file']
        row.update(budget_scaled=r.tolist(),blocks=[int(i['number']) for i in infos],
                   allocated_cells=sum(int(s.shape[0])*s.shape[-2]*s.shape[-1] for s in states),
                   chemistry_scaled=None,flow_defect_scaled=None,
                   budget_attribution_note='Composite total budget; no unsupported decomposition across covered parent chemistry.')
        history.append(row);confirmation=update_confirmation(confirmation,screen(history,ref['cj_speed_m_s'],policy),t,step,policy);
        next_diagnostic=recorder.next_t;next_picture=recorder.next_raw
        if row['raw_leaf_strip']:
            shutil.copy2(args.out/row['file'],args.out/'latest_fields.npz');push('latest_fields.npz')
        print(f't={t*1e6:.6f} us step={step} x_lab={row["front_lab_m"]*1000:.3f} mm '
              f'pmax={row["local_pmax_Pa"]/101325:.2f} atm gap_um={None if row["reaction_gap_m"] is None else row["reaction_gap_m"]*1e6} '
              f'blocks={row["blocks"]} budget={np.max(abs(r)):.3e}',flush=True)
        add_time('diagnostics_and_picture_io',time.monotonic()-tick)
        if np.max(abs(r))>policy['budget_limit']:raise FloatingPointError('Cumulative conservation limit exceeded; state preserved')
        if row['shock_geometry_invalid_fraction']>0 or row['reaction_band_truncated_fraction']>0:
            raise ValueError('Leading shock or reaction strip genuinely truncated; state retained')
    def save():
        nonlocal last_save
        tick=time.monotonic()
        trial_stream.flush()
        push(segment+'/trials.csv')
        total=np.asarray(e.probes(states,infos)[1]);residual=budget(total)
        ex=dict(contract=contract,status=status,window=frame.to_dict(),initial=initial.tolist(),boundary=boundary.tolist(),remap=remap.tolist(),
                chemistry=chemistry.tolist(),flow_defect=flow_defect.tolist(),history=history,events=events,
                wall_s=wall_before+time.monotonic()-started,rejected_steps=rejected,maximum_budget=maximum_budget,
                next_diagnostic=next_diagnostic,next_picture=next_picture,last_regrid_step=last_regrid_step,branch_receipt=receipt,recovery_receipt=recovery,step_control=controller.to_dict(),performance=performance,domain_added=domain_added.tolist(),domain_ledger=domain_ledger,long_run_receipt=long_receipt,stability_confirmation=confirmation,stage_request=stage_request)
        if events:ex['translation']=events[-1]
        verify_ledger(ex,m['config'],t,step)
        e.save(args.out/'checkpoint_latest.npz',states,infos,t,step,ex)
        add_time('checkpoint_and_mirror_io',time.monotonic()-tick)
        memory=jax.devices()[0].memory_stats() or {}
        summary=dict(status=status,reason=reason,stability_confirmation=confirmation,stage_request=stage_request,domain_added=domain_added.tolist(),step_control=controller.to_dict(),performance=performance,source_step=receipt['source_step'],source_t_s=receipt['source_t_s'],t_s=t,step=step,translations=len(events),wall_s=ex['wall_s'],
                     maximum_budget=maximum_budget,budget_scaled=residual.tolist(),rejected_steps=rejected,
                     cj_speed_m_s=ref['cj_speed_m_s'],statistics=screen(history,ref['cj_speed_m_s'],policy),
                     memory_stats={k:int(v) for k,v in memory.items() if isinstance(v,(int,np.integer))},
                     physical_convergence_pass=False,mesh_spacing_m=[e.dx[0],e.dy[0]],finest_spacing_m=[e.dx[-1],e.dy[-1]],blocks=[int(i['number']) for i in infos],case_kind=plan['kind'],
                     initial_mass_relative_boundary_input=float(boundary[0]/initial[0]),initial_energy_relative_boundary_input=float(boundary[3]/initial[3]))
        manifest=commit(args.out/'checkpoint_latest.npz',args.out,summary,history,contract)
        if args.mirror:manifest=commit(args.out/'checkpoint_latest.npz',args.mirror,summary,history,contract)
        print(f'CHECKPOINT VERIFIED step={step} t_us={t*1e6:.9f} file={manifest["checkpoint"]}',flush=True)
        last_save=time.monotonic()
    try:
        if not health[1]:raise FloatingPointError('Initial/restored state fails physical health checks')
        observe(force=True);save()
        while t<target:
            if confirmation.get('confirmed',False):
                status='statistical_stability';reason='Three contiguous windows passed and remained passed during confirmation; mesh/time convergence not certified';break
            if stop[0] or (session_limit>0 and time.monotonic()-started>=session_limit):
                status='paused';reason='Session interrupted or wall budget reached; resume this checkpoint';break
            if args.max_steps is not None and session_steps>=args.max_steps:
                status='paused';reason='Local validation step budget';break
            if step%cfg.amr_every==0 and step!=last_regrid_step:
                before=np.asarray(e.probes(states,infos)[1]);tick=time.monotonic()
                try:trial,trial_info=e.regrid(states,infos)
                except MemoryError as error:
                    import re
                    match=re.fullmatch(r'L(\d+) requires (\d+)\+sentinel blocks, capacity=(\d+)\. No refinement silently discarded\.',str(error))
                    if match is None:raise
                    stage_request=dict(kind='capacity',level=int(match[1]),required_blocks=int(match[2]))
                    status='needs_capacity';reason=str(error);break
                after=np.asarray(e.probes(trial,trial_info)[1])
                if np.max(abs(P@(after-before)/scale))>1e-10:raise FloatingPointError('Regrid changed composite inventory')
                h=np.asarray(e.accept_metrics(trial,trial_info))
                if not h[1]:raise FloatingPointError('Unhealthy regridded hierarchy')
                states,infos=trial,trial_info;health=h;last_regrid_step=step
                add_time('regrid_and_health',time.monotonic()-tick)
            if step%policy['front_check_every']==0:
                # Cheap coarse transverse column crossing; diagnostic accuracy uses leaf strip.
                front=float(front_extent(states[0][0]))
                if front>cfg.lx-policy['ahead_margin_m']:raise ValueError('Fresh-gas margin exhausted')
                if front>=policy['stop_front_m']:
                    stage_request=dict(kind='domain',new_nx=cfg.nx+policy['domain_growth_cells'])
                    status='needs_extension';reason='Fixed left wall retained; extend fresh downstream domain in a new process';break
            cfl=float(health[0]);remaining=(target-t) if np.isfinite(target) else cfg.dt_max
            dt=controller.propose(cfl,remaining)
            for attempt in range(12):
                tick=time.monotonic();candidate,b=e.cycle_budget(states,infos,jnp.asarray(dt))
                jax.block_until_ready((candidate,b));cycle_wall=time.monotonic()-tick
                tick=time.monotonic();metrics=np.asarray(e.accept_metrics(candidate,infos));health_wall=time.monotonic()-tick
                okay=bool(metrics[1]);reason_code='accepted'
                if not okay:
                    if not np.isfinite(metrics).all():reason_code='nonfinite_candidate_or_rejection_sentinel'
                    elif metrics[4]<-1e-13:reason_code='species_positivity'
                    elif metrics[2]<=0 or metrics[3]<=100:reason_code='density_or_temperature'
                    elif metrics[7]>=1e-6:reason_code='thermodynamic_residual'
                    else:reason_code='other_health_check'
                writer.writerow([step,t,attempt,cfl,dt,int(okay),reason_code,cycle_wall,health_wall,*[int(i['number']) for i in infos]])
                performance['trial_count']+=1
                add_time('accepted_cycles' if okay else 'rejected_cycles',cycle_wall)
                add_time('candidate_health_checks',health_wall)
                if okay:break
                rejected+=1;performance['rejected_branch_trials']+=1
                if attempt==0:print(f'Retrying step {step+1}: dt={dt:.3e} s, reason={reason_code}',flush=True)
                dt=controller.reject(dt)
            else:raise FloatingPointError('Repeated unhealthy candidate steps; preserving last accepted state')
            controller.accept(dt,endpoint_limited=(np.isfinite(target) and dt==remaining and attempt==0))
            states=candidate;boundary+=np.asarray(b)
            health=metrics;t+=dt;step+=1;session_steps+=1
            performance['accepted_branch_steps']+=1;performance['accepted_dt_sum_s']+=dt
            if time.monotonic()-last_progress>=30:
                print(f'Progress t={t*1e6:.6f} us step={step} dt={dt*1e12:.2f} ps '
                      f'branch_steps={performance["accepted_branch_steps"]} branch_rejects={performance["rejected_branch_trials"]} '
                      f'last_cycle={cycle_wall:.3f} s',flush=True)
                last_progress=time.monotonic()
            observe()
            if time.monotonic()-last_save>=policy['save_every_s']:save()
        else:status='validation_target';reason='Explicit diagnostic time limit reached; not production completion'
        observe(force=True)
    except BaseException:
        status='failed';reason=traceback.format_exc();raise
    finally:
        save();trial_stream.close();print(json.dumps(dict(status=status,t_us=t*1e6,step=step,translations=len(events))),flush=True)

if __name__=='__main__':main()
