import argparse,hashlib,json
from pathlib import Path

def main():
 root=Path(__file__).resolve().parents[1]
 manifest=json.loads((root/'baseline/h5_closed/provenance.json').read_text())
 runtime=root/'baseline/h5_closed/runtime'
 actual={p.relative_to(runtime).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.yaml') and '__pycache__' not in p.parts}
 if actual!=manifest['files']:raise SystemExit('Frozen baseline source mismatch')
 print(json.dumps(dict(baseline_files=len(actual),source_hashes_match=True,production_ready=False)))
if __name__=='__main__':main()
