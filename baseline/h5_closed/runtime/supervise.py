"""Run until statistical confirmation, saving/resuming and growing stages.

A new process is required for each geometry/capacity because JANC/JAX-AMR
configuration uses module globals. An explicit target is validation only.
"""
from pathlib import Path
import argparse,copy,json,os,signal,subprocess,sys,time,uuid
from verified_store import resolve,atomic_json,inside,sha


def next_plan(plan,request):
    plan=copy.deepcopy(plan);cfg=plan['config']
    if request['kind']=='domain':
        dx=cfg['lx']/cfg['nx'];cfg['nx']=int(request['new_nx']);cfg['lx']=cfg['nx']*dx
        plan['run']['stop_front_m']=cfg['lx']-.005
    elif request['kind']=='capacity':
        level=request['level'];required=request['required_blocks'];old=cfg['capacities'][level-1]
        cfg['capacities'][level-1]=max(2*(old-1)+1,((required+63)//64)*64+1)
    else:raise ValueError('Unknown stage request')
    return plan


def main():
    ap=argparse.ArgumentParser()
    for key in ('root','work','store','source'):ap.add_argument('--'+key,type=Path,required=key!='source')
    ap.add_argument('--device',choices=('gpu','cpu'),default='gpu')
    ap.add_argument('--max-steps',type=int,help='Validation only, not used by production notebook')
    ap.add_argument('--initial-case',type=Path,help='Validation alternative; production uses runtime case')
    a=ap.parse_args();root=a.root.resolve();work=a.work.resolve();store=a.store.resolve()
    if work==store or work.is_relative_to(store) or store.is_relative_to(work):raise ValueError('Work and durable store must be independent')
    for p in (work,store):p.mkdir(parents=True,exist_ok=True)
    run_id=uuid.uuid4().hex;lock=store/'supervisor_lock.json'
    boot=Path('/proc/sys/kernel/random/boot_id');boot=boot.read_text().strip() if boot.exists() else 'windows'
    owner=dict(pid=os.getpid(),boot=boot,run_id=run_id)
    # A stale lock from a destroyed Colab runtime must be acknowledged in the
    # notebook, which can inspect the currently attached runtime. Never race it.
    with lock.open('x',encoding='utf-8') as f:json.dump(owner,f)
    current_file=store/'current.json';stopped=[False];child=[None]
    def interrupt(*_):
        stopped[0]=True
        if child[0] is not None and child[0].poll() is None:child[0].send_signal(signal.SIGINT)
    signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(root/'src'),str(root)]+([env['PYTHONPATH']] if env.get('PYTHONPATH') else []));env['JAX_ENABLE_X64']='true';env['PYTHONIOENCODING']='utf-8'
    def run(cmd,device,label):
        ce=env.copy();ce['JAX_PLATFORMS']='cuda' if device=='gpu' else 'cpu'
        logpath=store/(label+'_'+uuid.uuid4().hex[:8]+'.log');print('LOG',logpath,flush=True)
        with logpath.open('w',encoding='utf-8',buffering=1) as log:
            child[0]=subprocess.Popen(cmd,env=ce,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in child[0].stdout:log.write(line);print(line,end='',flush=True)
            code=child[0].wait();child[0]=None
        if code:raise RuntimeError(f'{label} exited {code}; inspect {logpath}')
    try:
        while not stopped[0]:
            if not current_file.exists():
                if a.source is None:raise ValueError('First run requires the verified source RUN directory')
                source=a.source.resolve();plan=json.loads((a.initial_case or root/'case.json').read_text());request=dict(kind='initial')
            else:
                pointer=json.loads(current_file.read_text());stage=inside(store,pointer['stage']);cp,m,manifest=resolve(stage)
                status=m['extra']['status'];plan=m['extra']['contract']['plan']
                if status in ('statistical_stability','failed','validation_target'):
                    print(json.dumps(dict(status=status,t_us=m['t']*1e6,step=m['step'],stage=str(stage))),flush=True)
                    if status=='failed':raise RuntimeError('Saved stage failed; diagnostics retained')
                    break
                if status in ('needs_extension','needs_capacity'):
                    request=m['extra']['stage_request'];source=stage;plan=next_plan(plan,request)
                elif status in ('paused','running'):
                    if a.device=='gpu':
                        # Advisory budget must be checked before launching; this is not
                        # a promise about XLA allocation or future active block count.
                        cap=plan['config'];cells=cap['nx']*cap['ny']+sum(cap['capacities'])*4*cap['patch_x']*cap['patch_y']
                        estimate=(16*cells*14*8/2**30+2)
                        memory=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).splitlines()[0])/1024
                        if estimate>.85*memory:raise MemoryError(f'Planned stage memory budget {estimate:.1f} GiB exceeds available allocation allowance {.85*memory:.1f} GiB; verified state retained')
                    cmd=[sys.executable,'-u',str(root/'run_h5.py'),'--root',str(root),'--out',str(work/stage.name),'--mirror',str(stage),'--resume',str(cp),'--case',str(stage/'case.json'),'--device',a.device]
                    if a.max_steps is not None:cmd+=['--max-steps',str(a.max_steps)]
                    run(cmd,a.device,'solve')
                    _,saved,_=resolve(stage)
                    if saved['extra']['status']=='paused':break
                    continue
                else:raise ValueError('Unknown stage status: '+str(status))
            name='stages/stage_'+uuid.uuid4().hex;stage=inside(store,name)
            config=work/(stage.name+'.case.json');atomic_json(config,plan)
            atomic_json(store/'pending_migration.json',dict(source=str(source),stage=name,request=request,case=plan))
            run([sys.executable,'-u',str(root/'migrate_stage.py'),'--source',str(source),'--out',str(stage),'--runtime',str(root),'--case',str(config)],'cpu','migration')
            _,m,manifest=resolve(stage)
            atomic_json(current_file,dict(schema='closed_h5_long_run_pointer_v1',stage=name,initial_commit_sha256=manifest['checkpoint_sha256'],step_at_activation=m['step'],t_s_at_activation=m['t']))
            (store/'pending_migration.json').unlink()
    finally:
        if child[0] is not None and child[0].poll() is None:
            child[0].send_signal(signal.SIGINT);child[0].wait()
        if lock.exists() and json.loads(lock.read_text()).get('run_id')==run_id:lock.unlink()

if __name__=='__main__':main()
