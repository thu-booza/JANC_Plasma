"""Inspect Git candidates without staging or sending anything to a remote."""
from pathlib import Path
import subprocess,json,re
root=Path(__file__).resolve().parents[1]
raw=subprocess.check_output(['git','-C',str(root),'ls-files','--cached','--others','--exclude-standard','-z'])
paths=sorted(set(s.decode('utf-8') for s in raw.split(b'\0') if s));problems=[];total=0
for name in paths:
 p=(root/name).resolve()
 if not p.is_relative_to(root) or not p.is_file():problems.append(dict(path=name,reason='invalid file path'));continue
 size=p.stat().st_size;total+=size
 if size>5*1024**2:problems.append(dict(path=name,reason='file exceeds 5 MiB code-repository limit'))
 if p.suffix.lower() in ('.npz','.npy','.zip','.h5','.hdf5','.pem','.key','.pub','.exe','.dll','.keras','.whl'):
  problems.append(dict(path=name,reason='data/binary/key file must stay outside Git'))
 if size<5*1024**2:
  text=p.read_text(encoding='utf-8-sig',errors='replace')
  # Exact credential headers, not generic words such as password in code.
  if re.search(r'-----BEGIN (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----',text):problems.append(dict(path=name,reason='private-key material'))
print(json.dumps(dict(files=len(paths),bytes=total,problems=problems,push_performed=False,scope='size/type/private-key-header guard; not an exhaustive secret scanner'),indent=2))
raise SystemExit(1 if problems else 0)
