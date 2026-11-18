"""Small environment smoke test only; does not advance a physical experiment."""
import argparse,datetime,json,os,platform,sys,time
from pathlib import Path


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 import numpy as np
 import jax
 jax.config.update('jax_enable_x64',True)
 import jax.numpy as jnp
 import cantera as ct
 devices=jax.devices()
 if len(devices)!=2 or any(d.platform!='gpu' for d in devices):raise RuntimeError('Expected two GPUs; no silent CPU fallback')
 @jax.jit
 def kernel(x,b):
  return jnp.linalg.solve(x,b),x@x.T
 mat=np.eye(16)*4+np.ones((16,16))*.01;b=np.arange(16,dtype=float)
 results=[]
 for device in devices:
  start=time.monotonic()
  x=jax.device_put(jnp.asarray(mat,dtype=jnp.float64),device);rhs=jax.device_put(jnp.asarray(b,dtype=jnp.float64),device)
  solution,product=kernel(x,rhs);jax.block_until_ready((solution,product))
  if solution.dtype!=jnp.float64:raise RuntimeError('Float64 disabled')
  residual=float(np.max(abs(mat@np.asarray(solution)-b)))
  if residual>1e-11 or not np.allclose(np.asarray(product),mat@mat.T,rtol=1e-13,atol=1e-13):raise RuntimeError('GPU numerical check failed')
  results.append(dict(id=device.id,kind=device.device_kind,platform=device.platform,dtype=str(solution.dtype),linear_solve_residual=residual,compile_and_run_seconds=time.monotonic()-start))
 root=Path(__file__).resolve().parents[1];gas=ct.Solution(str(root/'baseline/h5_closed/runtime/mechanism.yaml'));gas.TPX=300.,101325.,'H2:2,O2:1,N2:1.5'
 if (gas.n_species,gas.n_reactions)!=(9,19):raise RuntimeError('Mechanism differs')
 report=dict(schema='v100_environment_smoke_v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=platform.python_version(),jax=jax.__version__,jaxlib=__import__('jaxlib').__version__,numpy=np.__version__,cantera=ct.__version__,devices=results,mechanism=dict(species=gas.n_species,reactions=gas.n_reactions,rho_kg_m3=gas.density),passed=True,scope='Independent small float64 operations on both GPUs and mechanism loading; not multi-GPU AMR or production memory/speed validation.',production_ready=False)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
