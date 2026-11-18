"""Formal repaired AMR entry. One process/engine, accepted-state recovery only.

python -m janc_v2.detonation_amr.run --config ... --mechanism ... --out ...
No legacy diagnostic checkpoint imports: start a new formal experiment once.
Then extend its physical end time without reinitializing conservation/history.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import time
import traceback
import uuid

REVISION = '20261029-formal-mask-front-recovery-v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'mechanism', 'out', 'mirror', 'resume'):
        p.add_argument('--'+key, type=Path, required=key in ('config', 'mechanism', 'out'))
    p.add_argument('--device', choices=('cpu', 'gpu'), default='gpu')
    p.add_argument('--t-end', type=float, default=20e-6)
    p.add_argument('--stop-x', type=float, default=.28)
    p.add_argument('--amplitude', type=float, default=1e-5)
    p.add_argument('--wall-limit-s', type=float, default=2400.)
    p.add_argument('--save-every-s', type=float, default=120.)
    p.add_argument('--step-budget', type=int, default=0)
    p.add_argument('--front-interval-s', type=float, default=.5e-6)
    p.add_argument('--raw-interval-s', type=float, default=2e-6)
    p.add_argument('--front-start-x', type=float, default=.005)
    p.add_argument('--stations', default='.03,.05,.10,.15,.20,.25,.27')
    args = p.parse_args()
    if min(args.t_end, args.wall_limit_s, args.save_every_s, args.front_interval_s) <= 0 or args.step_budget < 0 or args.raw_interval_s < 0:
        p.error('Invalid integration/diagnostic limits')
    if args.device == 'cpu': os.environ['JAX_PLATFORMS'] = 'cpu'
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import numpy as np
    from .solver import Engine, Config, MASK_REVISION
    from .diagnostics import Diagnostics
    from jaxamr import amr
    from janc_v2.model import thermo_model, reaction_model
    from janc_v2.solver_2D import flux, aux_func
    if args.device == 'gpu' and not any(d.platform == 'gpu' for d in jax.devices()):
        raise RuntimeError('GPU unavailable; not silently falling back to CPU')
    cfg = Config(**json.loads(args.config.read_text(encoding='utf-8'))); cfg.validate()
    stations = [float(x) for x in args.stations.split(',')]
    if not stations or not all(0 < x < cfg.lx for x in stations) or not 0 < args.stop_x < cfg.lx:
        p.error('Pressure stations and observation limit must be inside domain')
    if not 0 <= args.front_start_x < args.stop_x: p.error('Invalid front start')
    controls = dict(amplitude=args.amplitude, stop_x_m=args.stop_x, stations_m=stations,
                    front_interval_s=args.front_interval_s, raw_interval_s=args.raw_interval_s,
                    front_start_x_m=args.front_start_x,
                    dt_protocol='CFL; endpoint<=1.05*CFL; nine retries; no diagnostic time clipping')
    root = Path(__file__).parent
    source = {f.name:digest(f) for f in sorted(root.glob('*.py'))}
    upstream = {m.__name__:digest(m.__file__) for m in (amr, thermo_model, reaction_model, flux, aux_func)}
    # Include transitive numerical Python sources, not just their import facade.
    for label, folder in (('model', Path(thermo_model.__file__).parent),
                          ('solver_2D', Path(flux.__file__).parent), ('jaxamr', Path(amr.__file__).parent)):
        for f in sorted(folder.rglob('*.py')):
            upstream[label+'/'+f.relative_to(folder).as_posix()] = digest(f)
    runtime = {n:importlib.metadata.version(n) for n in ('jax', 'jaxlib', 'numpy', 'cantera')}
    runtime['devices'] = [getattr(d, 'device_kind', str(d)) for d in jax.devices()]
    manifest = json.loads(json.dumps(dict(revision=REVISION, mask_revision=MASK_REVISION,
        config=asdict(cfg), controls=controls, source=source, upstream=upstream,
        mechanism=digest(args.mechanism), runtime=runtime)))
    engine = Engine(cfg, args.mechanism, args.amplitude)
    states = infos = None
    t = 0.; step = 0; regrid_done = -1; balance = None; history = []; dts = []; regrids = []
    rejected = 0; max_drift = np.zeros(2); max_regrid = np.zeros(2); cumulative_wall = 0.
    restored = None; run_id = uuid.uuid4().hex
    # Everything below is checked before any write to an existing experiment.
    if args.resume:
        states, infos, meta = engine.load(args.resume)
        extra = meta['extra']
        if extra.get('manifest') != manifest:
            raise ValueError('Checkpoint source/runtime/config mismatch. Legacy diagnostics are not formal checkpoints.')
        if extra['status'] not in ('running', 'paused', 'interrupted', 'completed_time'):
            raise ValueError('Failed or downstream-terminal run cannot be blindly resumed')
        t, step = meta['t'], meta['step']
        if args.t_end <= t:
            raise ValueError('Requested physical time is already reached; keep results, or extend t-end')
        run_id = extra['run_id']
        for folder in (args.out, args.mirror):
            if folder and folder.exists():
                latest = folder/'checkpoint_latest.npz'
                if latest.exists() and digest(latest) != digest(args.resume):
                    with np.load(latest, allow_pickle=False) as z:
                        previous = json.loads(z['metadata'].tobytes())
                    old_extra = previous['extra']
                    # A disconnect during mirror copy may leave one destination
                    # behind. Only roll forward along the same accepted time grid.
                    if (old_extra.get('run_id') != run_id or previous['step'] >= step
                            or old_extra.get('manifest') != manifest
                            or old_extra.get('dts') != extra['dts'][:previous['step']]
                            or old_extra.get('balance') != extra['balance']):
                        raise ValueError('Not the newest compatible checkpoint. Refuse rollback/divergent overwrite.')
                m = folder/'manifest.json'
                if m.exists() and json.loads(m.read_text()) != manifest:
                    raise ValueError('Existing directory belongs to another experiment')
        balance = np.asarray(extra['balance'])
        history, dts, regrids = extra['history'], extra['dts'], extra['regrids']
        regrid_done, rejected = extra['regrid_done'], extra['rejected']
        max_drift, max_regrid = np.asarray(extra['max_drift']), np.asarray(extra['max_regrid'])
        cumulative_wall = extra['cumulative_wall']; restored = extra['diagnostics']
        if len(dts) != step or not np.isclose(sum(dts), t, rtol=1e-12, atol=1e-18):
            raise ValueError('Corrupt checkpoint time grid')
        for row in restored['frames']:
            # Missing /content is expected; Drive alone is sufficient.
            if not any(folder and (folder/row['file']).is_file() for folder in (args.out, args.mirror)):
                raise ValueError('Missing committed diagnostic frame: '+row['file'])
    else:
        for folder in (args.out, args.mirror):
            if folder and folder.exists(): raise FileExistsError('Existing results are never overwritten; resume explicitly')
    for folder in (args.out, args.mirror):
        if folder: folder.mkdir(parents=True, exist_ok=bool(args.resume))
    jax.config.update('jax_compilation_cache_dir', str(args.out.parent/'jax_cache'))

    def push(name):
        if args.mirror:
            dest = args.mirror/name; dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name+'.partial')
            shutil.copy2(args.out/name, tmp); tmp.replace(dest)

    def write(name, data):
        path = args.out/name; path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name+'.partial')
        tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8'); tmp.replace(path); push(name)

    write('manifest.json', manifest)
    session = uuid.uuid4().hex
    write('sessions/'+session+'.json', dict(run_id=run_id, requested_t_end_s=args.t_end, resumed_t_s=t,
        resumed_step=step, checkpoint_sha256=digest(args.resume) if args.resume else None,
        wall_limit_s=args.wall_limit_s, save_every_s=args.save_every_s,
        note='Checkpoint is authoritative; segment frames beyond its cursor are uncommitted evidence, not resumed history.'))
    diag = Diagnostics(engine, args.mechanism, args.out, args.mirror, controls, restored)
    start = time.perf_counter(); last_save = start; start_step = step; compile_s = 0.
    status = 'running'; reason = ''; stop_requested = False; health = None
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, 'SIGTERM'): signal.signal(signal.SIGTERM, request_stop)

    def checkpoint(final=False):
        if states is None or balance is None: return
        extra = dict(manifest=manifest, run_id=run_id, status=status, history=history, dts=dts,
            regrids=regrids, regrid_done=regrid_done, balance=balance.tolist(), rejected=rejected,
            max_drift=max_drift.tolist(), max_regrid=max_regrid.tolist(), diagnostics=diag.state(),
            cumulative_wall=cumulative_wall+time.perf_counter()-start)
        # The mirror is replaced only after a complete npz is written/copied.
        engine.save(args.out/'checkpoint_latest.npz', states, infos, t, step, extra)
        push('checkpoint_latest.npz')
        if final:
            shutil.copy2(args.out/'checkpoint_latest.npz', args.out/'checkpoint_final.npz'); push('checkpoint_final.npz')
            milestone = f'milestones/checkpoint_{step:08d}.npz'
            (args.out/'milestones').mkdir(exist_ok=True)
            shutil.copy2(args.out/'checkpoint_latest.npz', args.out/milestone); push(milestone)

    def report():
        write('history.json', history); write('schedule.json', dict(dt_s=dts, rejected_steps=rejected))
        write('regrids.json', regrids)
        write('front_index.json', dict(frames=diag.frames, note='Only checkpoint-owned rows are restored; raw strips include coverage.'))
        write('pressure_history.json', dict(stations_m=stations, y_order=['lower_wall_cell','center_cell','upper_wall_cell'], samples=diag.sensors))
        try: mem = jax.devices()[0].memory_stats()
        except Exception: mem = None
        write('summary.json', dict(status=status, reason=reason, t_s=t, step=step, revision=REVISION,
            mask_revision=MASK_REVISION, compile_s=compile_s,
            total_wall_s=cumulative_wall+time.perf_counter()-start, rejected_steps=rejected,
            max_mass_energy_relative_change=max_drift.tolist(), max_regrid_relative_change=max_regrid.tolist(),
            diagnostics=diag.report(), jax_allocator_memory_stats_bytes=mem,
            accuracy_note='Execution/repair validation is not stable detonation or CJ/grid/time convergence PASS.'))

    def sample():
        diag.observe(states, infos, t, step)
        front = float(engine.probes(states, infos)[0])
        if not history or (step%20 == 0 and history[-1]['step'] != step):
            row = dict(t_s=t, step=step, coarse_pressure_front_m=front, pmax_Pa=float(health[5]),
                Tmax_K=float(health[6]), blocks=[int(i['number']) for i in infos],
                relative_mass_energy_change=((health[8:10]-balance)/np.maximum(abs(balance),1e-30)).tolist())
            history.append(row); print(json.dumps(row), flush=True)
        return front

    try:
        if states is None: states, infos = engine.initial()
        health = np.asarray(engine.metrics(states, infos))
        if not np.isfinite(health).all() or not bool(health[1]): raise FloatingPointError('Invalid initial/resumed state')
        if balance is None: balance = health[8:10].copy()
        sample(); checkpoint(); report()
        tic = time.perf_counter()
        compiled = engine.cycle.lower(states, infos, jnp.array(float(health[0]))).compile()
        compile_s = time.perf_counter()-tic
        while t < args.t_end-8*np.spacing(args.t_end):
            if stop_requested:
                status='interrupted'; reason='Saved at complete accepted-operation boundary'; break
            if (args.step_budget and step-start_step >= args.step_budget) or time.perf_counter()-start >= args.wall_limit_s:
                status='paused'; reason='Session budget reached; rerun this stage to resume'; break
            if step > 0 and step%cfg.amr_every == 0 and regrid_done != step:
                candidate, ni = engine.regrid(states, infos)
                nh = np.asarray(engine.metrics(candidate, ni))
                drift = np.abs((nh[8:10]-health[8:10])/np.maximum(abs(balance), 1e-30))
                if not np.isfinite(nh).all() or not bool(nh[1]) or drift.max() > 1e-10:
                    raise FloatingPointError('Regrid health/conservation failure')
                regrids.append(dict(step=step, t_s=t, before=[int(i['number']) for i in infos],
                                    after=[int(i['number']) for i in ni], relative_change=drift.tolist()))
                states, infos, health = candidate, ni, nh; regrid_done = step
                max_regrid = np.maximum(max_regrid, drift)
            dt = min(float(health[0]), args.t_end-t)
            for attempt in range(9):
                trial = compiled(states, infos, jnp.array(dt)); check = np.asarray(engine.metrics(trial, infos))
                if np.isfinite(check).all() and bool(check[1]) and dt <= 1.05*check[0]: break
                rejected += 1; dt *= .5
            else: raise FloatingPointError('Nine rejected trials; last accepted state preserved')
            if dt < 1e-15: raise FloatingPointError('Time step below 1 fs')
            drift = np.abs((check[8:10]-balance)/np.maximum(abs(balance), 1e-30))
            if drift.max() > 1e-6: raise FloatingPointError('Cumulative mass/energy drift >1e-6 before observation limit')
            states = trial; health = check; t += dt; step += 1; dts.append(dt)
            max_drift = np.maximum(max_drift, drift)
            if args.t_end-t <= 8*np.spacing(args.t_end): t = args.t_end
            front = sample()
            if front >= args.stop_x:
                status='completed_observation'; reason='Reached downstream observation limit; not a CJ PASS'; break
            if time.perf_counter()-last_save >= args.save_every_s:
                checkpoint(); report(); last_save = time.perf_counter()
        if status == 'running':
            status='completed_time'; reason='Physical time target reached; may extend this formal run without resetting history'
    except Exception:
        status='failed'; reason=traceback.format_exc(); print(reason, flush=True)
    finally:
        checkpoint(final=status.startswith('completed'))
        report()
    print(json.dumps(dict(status=status,t_s=t,step=step,reason=reason)), flush=True)
    if status == 'failed': raise SystemExit(2)


if __name__ == '__main__': main()
