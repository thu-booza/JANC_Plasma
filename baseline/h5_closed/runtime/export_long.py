"""Export the current long-run commit and its reproducible runtime."""
from pathlib import Path
import json,zipfile
from verified_store import resolve,inside,sha


def export_long(store,destination):
    store=Path(store).resolve();destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    if (store/'supervisor_lock.json').exists():raise RuntimeError('Stop the supervisor and verify its exit before exporting')
    pointer=json.loads((store/'current.json').read_text());stage=inside(store,pointer['stage']);cp,m,manifest=resolve(stage)
    paths=[store/'current.json',stage/'latest.json']+[inside(stage,manifest[k]) for k in ('checkpoint','summary','history','contract')]
    paths+=[p for p in store.iterdir() if p.is_file() and p.suffix in ('.log','.json','.zip')]
    paths+=[p for p in stage.iterdir() if p.is_file() and p.suffix in ('.log','.json','.csv')]
    paths+=list(stage.glob('segments/**/*.csv'))
    missing=[]
    for row in m['extra']['history']:
        p=inside(stage,row['file'])
        if p.is_file():paths.append(p)
        else:missing.append(row['file'])
    if (stage/'latest_fields.npz').exists():paths.append(stage/'latest_fields.npz')
    with zipfile.ZipFile(destination,'x',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for p in sorted(set(paths)):
            if not p.is_relative_to(store) or '.partial' in p.name or '.writing' in p.name:raise ValueError('Unsafe export member')
            z.write(p,store.name+'/'+p.relative_to(store).as_posix())
        z.writestr(store.name+'/export_receipt.json',json.dumps(dict(step=m['step'],t_s=m['t'],checkpoint_sha256=manifest['checkpoint_sha256'],missing_optional_historical_frames=missing,previous_commits_included=False,physical_convergence_pass=False),indent=2))
    if (store/'supervisor_lock.json').exists() or json.loads((store/'current.json').read_text())!=pointer or json.loads((stage/'latest.json').read_text())!=manifest or sha(cp)!=manifest['checkpoint_sha256']:
        raise RuntimeError('Run changed during export; archive is invalid')
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None:raise IOError('Export CRC failed')
    return destination
