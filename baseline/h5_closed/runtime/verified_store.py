"""Verified immutable checkpoint commits; latest.json is the only authority."""
from pathlib import Path
import hashlib,json,os,shutil,uuid,zipfile
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()


def metadata(path):
    with np.load(path,allow_pickle=False) as z:
        m=json.loads(z['metadata'].tobytes())
        for l in range(m['config']['levels']+1):
            for key in ('state','number','index','glob_index','neighbor_index'):
                if f'L{l}_{key}' not in z:raise ValueError('Incomplete hierarchy checkpoint')
    return m


def atomic_json(path,value):
    path=Path(path);temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    with temp.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    temp.replace(path)
    if json.loads(path.read_text(encoding='utf-8'))!=value:raise IOError('JSON readback failed: '+str(path))


def inside(root,name):
    path=(Path(root)/name).resolve()
    if not path.is_relative_to(Path(root).resolve()) or path==Path(root).resolve():raise ValueError('Invalid snapshot path')
    return path


def resolve(root,check_hash=True):
    root=Path(root);manifest=json.loads((root/'latest.json').read_text(encoding='utf-8'))
    if manifest.get('schema')!='verified_h5_commit_v1':raise ValueError('Wrong commit schema')
    cp=inside(root,manifest['checkpoint']);m=metadata(cp)
    if m['step']!=manifest['step'] or m['t']!=manifest['t_s'] or m['extra']['status']!=manifest['status']:
        raise ValueError('Checkpoint time/status disagrees with commit manifest')
    if check_hash and sha(cp)!=manifest['checkpoint_sha256']:raise IOError('Checkpoint SHA256 disagrees with manifest')
    summary=json.loads(inside(root,manifest['summary']).read_text(encoding='utf-8'))
    history=json.loads(inside(root,manifest['history']).read_text(encoding='utf-8'))
    contract=json.loads(inside(root,manifest['contract']).read_text(encoding='utf-8'))
    if summary['step']!=m['step'] or summary['t_s']!=m['t'] or summary['status']!=m['extra']['status']:raise ValueError('Snapshot summary mismatch')
    if history!=m['extra']['history'] or contract!=m['extra']['contract']:raise ValueError('Snapshot history/contract mismatch')
    return cp,m,manifest


def commit(checkpoint,root,summary,history,contract):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);checkpoint=Path(checkpoint)
    m=metadata(checkpoint)
    if (m['step'],m['t'],m['extra']['status'])!=(summary['step'],summary['t_s'],summary['status']):raise ValueError('Local checkpoint/summary mismatch')
    if m['extra']['history']!=history or m['extra']['contract']!=contract:raise ValueError('Local history/contract mismatch')
    prior=None
    if (root/'latest.json').exists():
        _,previous,prior=resolve(root,check_hash=False)
        if previous['step']>m['step'] or previous['t']>m['t']:raise ValueError('Refuse to overwrite a newer commit')
    stem=f'checkpoints/state_{m["step"]:09d}_{uuid.uuid4().hex}'
    target=inside(root,stem+'.npz');target.parent.mkdir(exist_ok=True)
    temporary=target.with_suffix('.npz.partial')
    source_hash=sha(checkpoint)
    with checkpoint.open('rb') as src,temporary.open('xb') as dst:
        shutil.copyfileobj(src,dst,4194304);dst.flush();os.fsync(dst.fileno())
    if sha(temporary)!=source_hash:raise IOError('Checkpoint transfer checksum mismatch; prior committed state preserved')
    if metadata(temporary)!=m:raise ValueError('Copied checkpoint metadata differs')
    temporary.replace(target)
    if sha(target)!=source_hash:raise IOError('Renamed checkpoint checksum mismatch')
    names={k:stem+'.'+k+'.json' for k in ('summary','history','contract')}
    for key,value in [('summary',summary),('history',history),('contract',contract)]:atomic_json(inside(root,names[key]),value)
    manifest=dict(schema='verified_h5_commit_v1',step=m['step'],t_s=m['t'],status=m['extra']['status'],checkpoint=stem+'.npz',checkpoint_sha256=source_hash,**names)
    if prior:
        manifest['previous']={k:prior[k] for k in ('checkpoint','checkpoint_sha256','summary','history','contract','step','t_s','status')}
    # Update authority only after every snapshot member passes readback.
    atomic_json(root/'latest.json',manifest);resolve(root)
    # Convenient browsing copies are NOT authoritative for resume/export.
    for key,value in [('summary',summary),('history',history),('contract',contract)]:atomic_json(root/(key+'.json'),value)
    # Keep two verified checkpoints. Retire only files named by an older owned
    # manifest, never wildcard-delete original experiments or uncommitted data.
    if prior and prior.get('previous'):
        for key in ('checkpoint','summary','history','contract'):
            name=prior['previous'][key]
            path=inside(root,name)
            if path.parent==root.resolve()/'checkpoints' and path.name.startswith('state_') and path.exists():path.unlink()
    return manifest


def export_committed(root,destination):
    root=Path(root);destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    cp,m,manifest=resolve(root)
    # Include one authoritative commit plus referenced diagnostics, runtime and
    # logs. Historical missing optional frames remain explicitly listed.
    include=[root/'latest.json']+[inside(root,manifest[k]) for k in ('checkpoint','summary','history','contract')]
    include+=[p for p in root.iterdir() if p.is_file() and p.name!='latest.json' and p.suffix in ('.json','.log','.csv')]
    include+=list(root.glob('segments/**/*.log'))+list(root.glob('segments/**/*.csv'))+list(root.glob('segments/**/index.json'))
    missing=[]
    for row in m['extra']['history']:
        p=inside(root,row['file'])
        if p.is_file():include.append(p)
        else:missing.append(row['file'])
    if (root/'latest_fields.npz').is_file():include.append(root/'latest_fields.npz')
    app=root.parent
    extras=[p for p in app.iterdir() if p.is_file() and p.suffix in ('.json','.log','.zip')]
    with zipfile.ZipFile(destination,'x',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for p in sorted(set(include+extras)):
            if '.partial' in p.name or '.writing' in p.name:continue
            z.write(p,app.name+'/'+p.relative_to(app).as_posix())
        z.writestr(app.name+'/export_receipt.json',json.dumps(dict(step=m['step'],t_s=m['t'],checkpoint_sha256=manifest['checkpoint_sha256'],missing_historical_frames=missing,
            previous_commit_included=False,note='Only latest verified checkpoint exported; previous pointer is historical evidence.'),indent=2))
    # A solver writing during export invalidates the archive even if no lock was present.
    if json.loads((root/'latest.json').read_text())!=manifest or sha(cp)!=manifest['checkpoint_sha256']:
        raise IOError('Commit changed during export; this ZIP must not be used')
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None:raise IOError('Export CRC mismatch')
    return destination
