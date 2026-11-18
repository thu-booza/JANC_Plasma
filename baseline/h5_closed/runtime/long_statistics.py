"""Time-window statistics. Invalid observations are never filtered into a pass."""
import numpy as np

FIELDS=('mean_profile_pmax_Pa','local_pmax_Pa','v_rms_m_s','front_span_m','reaction_gap_m')

def screen(history,cj,policy):
    unique={r['step']:r for r in history};rows=sorted(unique.values(),key=lambda r:r['t_s'])
    out=dict(status='developing_or_insufficient_history',windows=[],checks={},physical_convergence_pass=False)
    if not rows:return out
    end=rows[-1]['t_s'];width=policy['statistics_window_s'];cadence=policy['diagnostic_every_s']
    for right in (end-2*width,end-width,end):
        left=right-width
        if left<policy['statistics_skip_until_s']-1e-14:continue
        part=[r for r in rows if left-1e-14<=r['t_s']<=right+1e-14]
        if len(part)<7:continue
        times=np.array([r['t_s'] for r in part]);gaps=np.diff(np.r_[left,times,right])
        if np.max(gaps)>1.5*cadence or times[-1]-times[0]<.8*width:continue
        valid=all(r.get('valid_front',False) and r.get('reaction_gap_valid_fraction',0)>=.95 and
                  r.get('finest_fraction_active_reaction',0)>=.95 and
                  r.get('shock_geometry_invalid_fraction',1)==0 and r.get('reaction_band_truncated_fraction',1)==0 and
                  max(map(abs,r.get('budget_scaled',[float('inf')])))<=policy['budget_limit'] and
                  all(r.get(k) is not None and np.isfinite(r[k]) for k in FIELDS) for r in part)
        if not valid:
            out['windows'].append(dict(start_s=left,end_s=right,samples=len(part),valid=False));continue
        tx=times-times.mean();x=np.array([r['front_lab_m'] for r in part]);speed=float(np.dot(tx,x-x.mean())/np.dot(tx,tx))
        w=dict(start_s=left,end_s=right,samples=len(part),valid=True,speed_m_s=speed,cj_relative=speed/cj-1)
        for field in FIELDS:
            vals=np.array([r[field] for r in part]);w[field]=dict(mean=float(vals.mean()),std=float(vals.std()))
        out['windows'].append(w)
    if len(out['windows'])!=3 or not all(w['valid'] for w in out['windows']):return out
    velocities=np.array([w['speed_m_s'] for w in out['windows']]);checks=out['checks']
    checks['speed_within_2percent_cj']=bool(np.max(abs(velocities/cj-1))<=.02)
    checks['window_speed_change_within_1percent']=bool(np.max(abs(np.diff(velocities)/velocities[:-1]))<=.01)
    for field in ('mean_profile_pmax_Pa','local_pmax_Pa','reaction_gap_m'):
        vals=np.array([w[field]['mean'] for w in out['windows']]);checks[field+'_window_change_within_5percent']=bool(np.all(vals>0) and np.max(abs(np.diff(vals)/vals[:-1]))<=.05)
    if all(checks.values()):out['status']='statistical_screen_passed'
    return out

def update_confirmation(state,result,t,step,policy):
    state=dict(state or {});last=state.get('last_step',-1)
    if step<=last:return state
    if result['status']!='statistical_screen_passed':return dict(last_step=step,first_pass_t_s=None,confirmed=False)
    first=state.get('first_pass_t_s')
    if first is None:first=t
    return dict(last_step=step,first_pass_t_s=first,confirmed=bool(t-first>=policy['statistics_confirmation_s']-1e-14))
