"""Branch exactly from the verified older checkpoint; never splice future history."""
from pathlib import Path
import argparse,copy,io,json,shutil,zipfile
import numpy as np
from verified_store import sha,metadata,commit


def migrate(source,out,runtime):
    source,out,runtime=map(lambda p:Path(p).resolve(),(source,out,runtime))
    expected=json.loads((runtime/'recovery_source.json').read_text())
    checkpoint=source/'RUN/checkpoint_latest.npz';bundle=source/'runtime_fc32b8e28d4f.zip'
    if out.is_relative_to(source) or source.is_relative_to(out):raise ValueError('Use an independent recovery directory')
    if out.exists():raise FileExistsError('Recovery directory exists; resolve its commit or inspect incomplete migration')
    if sha(checkpoint)!=expected['checkpoint_sha256'] or sha(bundle)!=expected['runtime_sha256']:raise ValueError('Not the verified older complete checkpoint/runtime')
    m=metadata(checkpoint);ex=m['extra']
    if m['step']!=expected['source_step'] or m['t']!=expected['source_t_s']:raise ValueError('Source clock mismatch')
    with zipfile.ZipFile(checkpoint) as z:
        if z.testzip() is not None:raise ValueError('Source checkpoint CRC failed')
    with np.load(checkpoint,allow_pickle=False) as z:
        for l in range(m['config']['levels']+1):
            a=z[f'L{l}_state'][:int(z[f'L{l}_number'])];Y=a[:,4:12]/a[:,:1]
            if not np.isfinite(a).all() or a[:,0].min()<=0 or Y.min()<-1e-13 or (1-Y.sum(axis=1)).min()<-1e-13:raise ValueError('Unhealthy source field')
    if json.loads((runtime/'case.json').read_text())!=ex['contract']['plan']:raise ValueError('Physical case changed')
    if ex['maximum_budget']>ex['contract']['plan']['run']['budget_limit']:raise ValueError('Source budget already exceeds limit')
    allowed={'run_h5.py','h5_diagnostics.py'}
    with zipfile.ZipFile(bundle) as z:
        for name,digest in ex['contract']['source'].items():
            if __import__('hashlib').sha256(z.read(name)).hexdigest()!=digest:raise ValueError('Original contract mismatch')
        for name in z.namelist():
            if name not in allowed and (runtime/name).read_bytes()!=z.read(name):raise ValueError('Unexpected solver/physics change: '+name)
    report=json.loads((source/'RUN/summary.json').read_text());outside_history=json.loads((source/'RUN/history.json').read_text())
    if outside_history[:len(ex['history'])]!=ex['history']:raise ValueError('Older checkpoint history is not a prefix of exported history')
    new=copy.deepcopy(m);nx=new['extra']
    new_contract=copy.deepcopy(ex['contract']);new_contract['source']={p.relative_to(runtime).as_posix():sha(p) for p in sorted(runtime.rglob('*.py'))}
    missing=[]
    for row in ex['history']:
        path=(source/'RUN'/row['file']).resolve()
        if not path.is_relative_to((source/'RUN').resolve()):raise ValueError('Unsafe history path')
        if not path.is_file():missing.append(row['file'])
    receipt=dict(schema='closed_crop_recovery_v1',source_step=m['step'],source_t_s=m['t'],source_checkpoint_sha256=expected['checkpoint_sha256'],
        old_contract=ex['contract'],new_contract=new_contract,observed_later_report_step=report['step'],observed_later_report_t_s=report['t_s'],
        rollback_physical_duration_s=report['t_s']-m['t'],retained_history_rows=len(ex['history']),excluded_later_history_rows=len(outside_history)-len(ex['history']),
        missing_optional_historical_frames=missing,arrays_unchanged=True,physical_case_unchanged=True,
        note='Recovery starts at the actual stored clock; later reports are retained as evidence, not added to live history or used to relabel the saved field.')
    nx.update(status='paused',contract=new_contract,recovery_receipt=receipt,wall_s=max(ex['wall_s'],expected['minimum_wall_s'],report['wall_s']))
    nx['next_diagnostic']=m['t'];nx['next_picture']=m['t']
    out.mkdir(parents=True)
    (out/'migration_in_progress.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    (out/'evidence').mkdir()
    for name in ('summary.json','history.json','contract.json'):shutil.copy2(source/'RUN'/name,out/'evidence'/('later_export_'+name))
    for row in ex['history']:
        src=source/'RUN'/row['file'];dst=out/row['file']
        if src.is_file() and not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    temp=out/'migration_checkpoint.npz'
    with zipfile.ZipFile(checkpoint) as old,zipfile.ZipFile(temp,'x',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for info in old.infolist():
            if info.filename=='metadata.npy':
                buf=io.BytesIO();np.save(buf,np.frombuffer(json.dumps(new).encode(),dtype=np.uint8),allow_pickle=False);z.writestr(info.filename,buf.getvalue())
            else:
                with old.open(info) as src,z.open(info.filename,'w',force_zip64=True) as dst:shutil.copyfileobj(src,dst,4194304)
    summary=dict(status='paused',reason='Explicit recovery from older verified checkpoint; later history excluded',step=m['step'],t_s=m['t'],wall_s=nx['wall_s'],
                 maximum_budget=nx['maximum_budget'],rejected_steps=nx['rejected_steps'],translations=len(nx['events']),physical_convergence_pass=False,
                 recovery_receipt='recovery_receipt.json')
    manifest=commit(temp,out,summary,nx['history'],new_contract)
    (out/'recovery_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    temp.unlink();(out/'migration_in_progress.json').unlink()
    return manifest

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('source','out','runtime'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();print(json.dumps(migrate(a.source,a.out,a.runtime),indent=2))
