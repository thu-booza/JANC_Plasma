"""Read-only observations, with checkpoint-owned cursors and segment files."""
from pathlib import Path
import time
import uuid
import numpy as np
import jax
import jax.numpy as jnp
from .core import pressure
from .strip import FrontRecorder


def speed_fit(rows, span=.02):
    """Shock-gradient speed, not the coarse pressure-threshold speed or CJ PASS."""
    rows = [r for r in rows if r.get('shock_invalid_fraction', r['peak_boundary_fraction']) < .05]
    if len(rows) < 4:
        return None
    right = rows[-1]['shock_mean_x_m']
    rows = [r for r in rows if right-span <= r['shock_mean_x_m'] <= right]
    if len(rows) < 4 or rows[-1]['shock_mean_x_m']-rows[0]['shock_mean_x_m'] < .8*span:
        return None
    t = np.array([r['t_s'] for r in rows]); x = np.array([r['shock_mean_x_m'] for r in rows])
    t -= t.mean()
    if np.dot(t, t) <= 0:
        return None
    d = np.dot(t, x-x.mean())/np.dot(t, t)
    return dict(speed_m_s=float(d), span_m=float(x[-1]-x[0]), samples=len(rows),
                fit_rms_m=float(np.sqrt(np.mean((x-x.mean()-d*t)**2))))


def sensor_sampler(engine, stations):
    """Finest covering cell at physical wall-adjacent/center sample points.

    These are cell pressures, NOT a simulated transducer transfer function.
    Three y points per x; physical boundaries use the interior wall cell.
    """
    c = engine.c
    xx = jnp.repeat(jnp.asarray(stations), 3)
    yy = jnp.tile(jnp.array([0., c.ly/2, np.nextafter(c.ly, 0.)]), len(stations))

    def sample(states, infos):
        result = jnp.full(xx.shape, jnp.nan)
        levels = jnp.full(xx.shape, -1, dtype=jnp.int32)
        oxs = jnp.zeros((1, 2), dtype=jnp.int64)
        for level, s in enumerate(states):
            if level:
                idx = jnp.maximum(infos[level]['index'], 0)
                oxs = 2*(oxs[idx[:, 0]]+idx[:, 1:]*jnp.array([c.patch_x, c.patch_y]))
            gx = jnp.floor(xx/engine.dx[level]).astype(jnp.int64)
            gy = jnp.clip(jnp.floor(yy/engine.dy[level]).astype(jnp.int64), 0, c.ny*2**level-1)
            def insert(b, values):
                out, cov = values
                ix, iy = gx-oxs[b, 0], gy-oxs[b, 1]
                inside = (ix >= 0)&(ix < s.shape[-2])&(iy >= 0)&(iy < s.shape[-1])
                cells = s[b][:, jnp.clip(ix, 0, s.shape[-2]-1), jnp.clip(iy, 0, s.shape[-1]-1)]
                return jnp.where(inside, pressure(cells[..., None])[:, 0], out), jnp.where(inside, level, cov)
            result, levels = jax.lax.fori_loop(0, infos[level]['number'], insert, (result, levels))
        return result.reshape(-1, 3), levels.reshape(-1, 3)
    return jax.jit(sample)


class Diagnostics:
    def __init__(self, engine, mechanism, out, mirror, controls, restored=None):
        self.out, self.mirror = Path(out), Path(mirror) if mirror else None
        self.controls = controls
        self.frames = list(restored['frames']) if restored else []
        self.sensors = list(restored['sensors']) if restored else []
        self.wall_s = restored['wall_s'] if restored else 0.
        self.segment = 'segments/'+uuid.uuid4().hex
        for root in (self.out, self.mirror):
            if root: (root/self.segment).mkdir(parents=True, exist_ok=False)
        self.rec = FrontRecorder(engine, mechanism, self.out/self.segment,
                    self.mirror/self.segment if self.mirror else None,
                    interval=controls['front_interval_s'], raw_interval=controls['raw_interval_s'],
                    start_x=controls['front_start_x_m'])
        if restored:
            self.rec.next_t = restored['next_front_t']
            self.rec.next_raw = restored['next_raw_t']
            self.rec.last_t = restored['last_front_t']
        self.sample = sensor_sampler(engine, controls['stations_m'])

    def observe(self, states, infos, t, step):
        tic = time.perf_counter()
        if not self.sensors or self.sensors[-1]['step'] != step:
            p, levels = map(np.asarray, self.sample(states, infos))
            if not np.isfinite(p).all() or np.any(levels < 0):
                raise ValueError('Incomplete/nonfinite station pressure')
            self.sensors.append(dict(t_s=t, step=step, pressure_Pa=p.tolist(), levels=levels.tolist()))
        n = len(self.rec.rows)
        self.rec.observe(states, infos, t, step)
        if len(self.rec.rows) > n:
            row = dict(self.rec.rows[-1])
            row['file'] = self.segment+'/front_diagnostics/'+row['file']
            self.frames.append(row)
        self.wall_s += time.perf_counter()-tic

    def state(self):
        # Finite initial cursors make strict JSON round trips portable.
        finite = lambda x: float(x) if np.isfinite(x) else -1.
        return dict(frames=self.frames, sensors=self.sensors, wall_s=self.wall_s,
                    next_front_t=finite(self.rec.next_t), next_raw_t=finite(self.rec.next_raw),
                    last_front_t=self.rec.last_t)

    def report(self):
        p = np.asarray([r['pressure_Pa'] for r in self.sensors])
        times = np.asarray([r['t_s'] for r in self.sensors])
        peaks = []
        if len(p):
            for k, x in enumerate(self.controls['stations_m']):
                indices = p[:, k, :].argmax(axis=0)
                peaks.append(dict(x_m=x, pressure_Pa=p[indices, k, np.arange(3)].tolist(),
                                  peak_t_s=times[indices].tolist()))
        return dict(front_frames=len(self.frames), pressure_samples=len(self.sensors),
                    pressure_sample_max_dt_s=float(np.diff(times).max()) if len(times)>1 else None,
                    station_peak_lower_center_upper=peaks,
                    shock_speed_20mm=speed_fit(self.frames), shock_speed_50mm=speed_fit(self.frames, .05),
                    last_front=self.frames[-1] if self.frames else None, diagnostic_wall_s=self.wall_s,
                    note='Read-only cell-pressure samples; no sensor response or automatic CJ/stability PASS. Coarse cells repeated in strips are not additional resolution.')
