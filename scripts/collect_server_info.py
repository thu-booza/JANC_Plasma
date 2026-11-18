"""Read-only Ubuntu compute-node inventory. No installs, network, or GPU compute.
Run: python3 collect_server_info.py > server_info.json
Use --data-path /your/data/path to inspect the intended result filesystem.
"""
import argparse,datetime,importlib.metadata,json,os,platform,shutil,subprocess,sys
from pathlib import Path

def command(argv):
    if shutil.which(argv[0]) is None:return dict(available=False)
    try:
        p=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,errors='replace',timeout=20,check=False)
        return dict(available=True,returncode=p.returncode,stdout=p.stdout[:20000],stderr=p.stderr[:3000])
    except (OSError,subprocess.TimeoutExpired) as e:return dict(available=True,error=str(e))

def collect(data_path):
    path=Path(data_path).expanduser().resolve()
    if not path.is_dir():raise ValueError('Data path must be an existing directory; no directories will be created')
    packages={}
    for name in ('jax','jaxlib','jax-cuda12-plugin','jax-cuda13-plugin','numpy','cantera','scipy','h5py'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    usage=shutil.disk_usage(path)
    report=dict(schema='janc_readonly_server_inventory_v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        platform=platform.system(),architecture=platform.machine(),kernel=platform.release(),python_version=sys.version.split()[0],
        cpu_logical_count=os.cpu_count(),package_versions=packages,
        tools={name:bool(shutil.which(name)) for name in ('git','rsync','tmux','screen','conda','docker','apptainer','sbatch','srun','nvidia-smi','nvcc')},
        data_path=str(path),disk_bytes=dict(total=usage.total,free=usage.free),
        note='No hostname/IP/username, environment dump, SSH keys, tokens or process listing requested. No JAX/GPU computation was initialized.')
    release=Path('/etc/os-release')
    if release.is_file():report['os_release']=release.read_text()
    for key,cmd in [('gpu', ['nvidia-smi','--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap','--format=csv,noheader,nounits']),
                    ('gpu_topology',['nvidia-smi','topo','-m']),('cpu',['lscpu']),('memory',['free','-h']),('filesystem',['df','-hT',str(path)]),('cuda_toolkit',['nvcc','--version'])]:
        report[key]=command(cmd)
    if report['gpu'].get('returncode',0)!=0:
        report['gpu_fallback']=command(['nvidia-smi','--query-gpu=index,name,memory.total,memory.free,driver_version','--format=csv,noheader,nounits'])
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data-path',default='.')
    args=ap.parse_args();print(json.dumps(collect(args.data_path),ensure_ascii=False,indent=2))
