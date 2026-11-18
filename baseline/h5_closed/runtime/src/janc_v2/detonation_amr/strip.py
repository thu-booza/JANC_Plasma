"""Read-only moving-front observations. Never feeds back into the integrator.

Only a narrow leaf-cell strip leaves the device. Fine-grid display repeats
coarse cells exactly; coverage is saved so this is not mistaken for resolution.
"""
import json
from pathlib import Path
import shutil
import time
import cantera as ct
import jax
import jax.numpy as jnp
import numpy as np


def peak_positions(x, f, mask, *, allow_outside_mask=False):
    """Locate a local peak; interpolation failure is not a crop-boundary test.

    A search mask selects candidate centres. For shock localization its real
    neighbours may lie outside that mask, but must form a concave local peak.
    Invalid interpolation falls back to the native cell centre and stays flagged.
    """
    k = np.argmax(np.where(mask, f, -np.inf), axis=0)
    cols = np.arange(f.shape[1]); km = np.maximum(k-1, 0); kp = np.minimum(k+1, len(x)-1)
    fm, fc, fp = f[km, cols], f[k, cols], f[kp, cols]
    den = fm-2*fc+fp
    good = (k > 0)&(k < len(x)-1)&mask.any(axis=0)&np.isfinite(fm+fc+fp)
    good &= (den < 0)&(fc >= fm)&(fc >= fp)
    if not allow_outside_mask:
        good &= mask[km, cols]&mask[kp, cols]
    delta = np.divide(.5*(fm-fp), den, out=np.zeros_like(den), where=good)
    good &= np.isfinite(delta)&(np.abs(delta) <= .5+1e-12)
    return x[k]+np.where(good, np.clip(delta, -.5, .5), 0.)*(x[1]-x[0]), ~good


def leading_shock_details(x, p, coverage, dx0, p0):
    """Separate missing/truncated shocks from subcell interpolation quality."""
    above = p > 2*p0
    k = np.max(np.where(above, np.arange(len(x))[:, None], -1), axis=0)
    cols = np.arange(p.shape[1]); left = np.clip(k, 0, len(x)-2)
    den = p[left+1, cols]-p[left, cols]
    frac = np.divide(2*p0-p[left, cols], den, out=np.zeros_like(den), where=den!=0)
    anchor = x[left]+np.clip(frac, 0, 1)*(x[1]-x[0])
    radius = 3*dx0/2.**coverage[left, cols]
    mask = np.abs(x[:, None]-anchor) <= radius
    gradient = -np.gradient(p, x, axis=0)
    peak = np.argmax(np.where(mask, gradient, -np.inf), axis=0)
    shock, interpolation_invalid = peak_positions(x, gradient, mask, allow_outside_mask=True)
    causes = dict(crossing_missing=k < 0, crossing_at_crop_end=k == len(x)-1,
                  peak_at_crop_edge=(peak == 0)|(peak == len(x)-1),
                  no_positive_compressive_gradient=np.max(np.where(mask, gradient, -np.inf), axis=0) <= 0)
    geometry_invalid = np.logical_or.reduce(list(causes.values()))
    return shock, geometry_invalid|interpolation_invalid, anchor, dict(
        geometry_invalid=geometry_invalid, interpolation_invalid=interpolation_invalid, **causes)


def leading_shock_positions(x, p, coverage, dx0, p0):
    """Compatibility API: all low-quality positions remain invalid for statistics."""
    return leading_shock_details(x, p, coverage, dx0, p0)[:3]


def make_crop(engine, behind=.002, ahead=.0008):
    c = engine.c; dx = engine.dx[-1]; dy = engine.dy[-1]
    width = int(np.ceil((behind+ahead)/dx))+1
    total_x = c.nx*2**c.levels; total_y = c.ny*2**c.levels
    if width > total_x: raise ValueError('Moving strip is wider than the domain')

    def crop(states, infos, front):
        first = jnp.clip(jnp.floor((front-behind)/dx).astype(jnp.int64), 0, total_x-width)
        xs = first+jnp.arange(width)
        u = jnp.full((14, width, total_y), jnp.nan)
        coverage = jnp.full((width, total_y), -1, dtype=jnp.int8)
        origin = jnp.zeros((1, 2), dtype=jnp.int64)
        for level, s in enumerate(states):
            scale = 2**(c.levels-level)
            if level:
                idx = jnp.maximum(infos[level]['index'], 0)
                origin = origin[idx[:, 0]] + idx[:, 1:]*jnp.array([c.patch_x, c.patch_y])*2*scale
            ny = s.shape[-1]*scale
            def body(i, carry):
                out, cov = carry; ox, oy = origin[i]
                active = (xs >= ox)&(xs < ox+s.shape[-2]*scale)
                def insert(carry):
                    out, cov = carry
                    ix = jnp.clip((xs-ox)//scale, 0, s.shape[-2]-1)
                    tile = jnp.repeat(s[i][:, ix, :], scale, axis=2)
                    old = jax.lax.dynamic_slice(out, (0, 0, oy), (14, width, ny))
                    oldcov = jax.lax.dynamic_slice(cov, (0, oy), (width, ny))
                    out = jax.lax.dynamic_update_slice(out, jnp.where(active[None, :, None], tile, old), (0, 0, oy))
                    cov = jax.lax.dynamic_update_slice(cov, jnp.where(active[:, None], level, oldcov).astype(jnp.int8), (0, oy))
                    return out, cov
                return jax.lax.cond(jnp.any(active), insert, lambda a: a, (out, cov))
            u, coverage = jax.lax.fori_loop(0, infos[level]['number'], body, (u, coverage))
        return (xs+.5)*dx, u, coverage
    return jax.jit(crop)


class FrontRecorder:
    def __init__(self, engine, mechanism, out, mirror=None, interval=2e-7, start_x=.005, raw_interval=2e-6):
        if interval <= 0 or raw_interval < 0 or not 0 <= start_x < engine.c.lx:
            raise ValueError('Invalid front diagnostic cadence/range')
        self.engine = engine; self.interval = interval; self.start_x = start_x
        self.raw_interval = raw_interval; self.next_t = -np.inf; self.next_raw = -np.inf
        self.last_t = None; self.rows = []; self.wall_s = 0.
        self.out = Path(out)/'front_diagnostics'; self.out.mkdir(exist_ok=False)
        self.mirror = Path(mirror)/'front_diagnostics' if mirror else None
        if self.mirror: self.mirror.mkdir(exist_ok=False)
        self.crop = make_crop(engine); self.gas = ct.Solution(str(mechanism))
        self.gas.TP = engine.c.t0, engine.c.p0
        self.gas.set_equivalence_ratio(engine.c.phi, 'H2:1', 'O2:1,N2:3.76')
        self.initial_h2 = self.gas.Y[0]

    def push(self, path):
        if self.mirror:
            dest = self.mirror/path.name; partial = dest.with_name(dest.name+'.partial')
            shutil.copy2(path, partial); partial.replace(dest)

    def write_index(self):
        data = dict(requested_interval_s=self.interval, raw_interval_s=self.raw_interval,
                    total_diagnostic_wall_s=self.wall_s, frames=self.rows,
                    note='Actual accepted-step times; no added integration steps. Instantaneous Cantera heat release, not discrete JANC source. No automatic stable-detonation PASS.')
        path = self.out/'index.json'; tmp = self.out/'index.partial.json'
        tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8'); tmp.replace(path); self.push(path)

    def observe(self, states, infos, t, step, force=False):
        if not force and (t == self.last_t or t < self.next_t): return
        tic = time.perf_counter()
        front = float(self.engine.probes(states, infos)[0])
        self.next_t = t+self.interval
        if front < self.start_x: return
        x, u, coverage = map(np.asarray, self.crop(states, infos, jnp.array(front)))
        if not np.isfinite(u).all() or (coverage < 0).any(): raise ValueError('Incomplete diagnostic crop')
        rho, temp = u[0], u[-1]
        Y = np.concatenate((u[4:12]/rho, 1-np.sum(u[4:12]/rho, axis=0, keepdims=True)))
        if Y.min() < -1e-8: raise ValueError('Invalid species in diagnostic crop')
        Y = np.maximum(Y, 0); Y /= Y.sum(axis=0, keepdims=True)
        shape = rho.shape; p = np.empty(rho.size); q = p.copy(); sigma = p.copy()
        flat_y = Y.reshape(9, -1).T
        for start in range(0, rho.size, 16384):
            stop = min(start+16384, rho.size); gas = ct.SolutionArray(self.gas, stop-start)
            gas.TDY = temp.ravel()[start:stop], rho.ravel()[start:stop], flat_y[start:stop]
            p[start:stop] = gas.P; q[start:stop] = gas.heat_release_rate
            dydt = gas.net_production_rates*self.gas.molecular_weights/gas.density[:, None]
            h = gas.partial_molar_enthalpies/self.gas.molecular_weights
            sigma[start:stop] = np.sum((gas.mean_molecular_weight[:, None]/self.gas.molecular_weights-h/(gas.cp_mass*gas.T)[:, None])*dydt, axis=1)
        p = p.reshape(shape); q = q.reshape(shape); sigma = sigma.reshape(shape)
        if not all(np.isfinite(a).all() for a in (p, q, sigma)): raise ValueError('Nonfinite diagnostic rates')
        shock, flag_s, crossing, shock_quality = leading_shock_details(x, p, coverage, self.engine.dx[0], self.engine.c.p0)
        reaction_mask = (x[:, None] <= shock[None, :])&(x[:, None] >= shock[None, :]-.0018)
        xp, flag_q = peak_positions(x, q, reaction_mask)
        xt, flag_t = peak_positions(x, sigma, reaction_mask)
        y = (np.arange(shape[1])+.5)*self.engine.dy[-1]
        band = (x[:, None] <= shock)&(x[:, None] >= shock-.0018)
        arrays = dict(x_m=x, y_m=y, shock_x_m=shock, heat_peak_x_m=xp, thermicity_peak_x_m=xt,
                      coverage=coverage, p_mean_Pa=p.mean(axis=1), T_mean_K=temp.mean(axis=1),
                      Y_H2_density_weighted_mean=(rho*Y[0]).mean(axis=1)/rho.mean(axis=1),
                      heat_release_mean_W_m3=q.mean(axis=1), thermicity_mean_s_1=sigma.mean(axis=1),
                      shock_or_peak_boundary_flag=flag_s|flag_q|flag_t,
                      shock_invalid=flag_s, leading_pressure_crossing_x_m=crossing,
                      shock_geometry_invalid=shock_quality['geometry_invalid'],
                      shock_interpolation_invalid=shock_quality['interpolation_invalid'])
        raw = self.raw_interval > 0 and (force or t >= self.next_raw)
        # Display samples repeat coarse cells. Count native-cell equivalents,
        # never label those repeated pixels as independent resolved cells.
        native_dx = self.engine.dx[0]/2.**coverage
        gap_band = (x[:, None] <= shock)&(x[:, None] >= xp)
        cell_count = np.sum(np.where(gap_band, self.engine.dx[-1]/native_dx, 0.), axis=0)
        positive_q = np.where(reaction_mask, np.maximum(q, 0.), 0.)
        qmax = positive_q.max(axis=0)
        invalid_q = (qmax <= 0.)|flag_q|(xp >= shock)
        half_band = reaction_mask&(positive_q >= .5*qmax[None])&(qmax[None] > 0.)
        half_cells = np.sum(np.where(half_band, self.engine.dx[-1]/native_dx, 0.), axis=0)
        arrays.update(shock_to_heat_peak_m=shock-xp,
                      shock_to_heat_native_cell_equivalents=cell_count,
                      heat_half_max_native_cell_equivalents=half_cells,
                      reaction_peak_invalid=invalid_q,
                      wall_bottom_p_Pa=p[:, 0], wall_top_p_Pa=p[:, -1],
                      center_p_Pa=p[:, shape[1]//2])
        if raw:
            arrays.update(rho=rho, T=temp, Y_first8=Y[:8], v=u[2]/rho, u=u[1]/rho,
                          p_Pa=p, heat_release_W_m3=q, thermicity_s_1=sigma)
            self.next_raw = t+self.raw_interval
        meta = dict(t_s=t, step=step, raw_leaf_strip=raw, base_front_x_m=front)
        arrays['metadata'] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
        path = self.out/f'front_{step:08d}.npz'
        if path.exists():path=self.out/f'front_{step:08d}_repeat_{len(self.rows):04d}.npz'
        if path.exists(): raise FileExistsError(path)
        tmp = path.with_suffix('.writing.npz'); np.savez_compressed(tmp, **arrays); tmp.replace(path); self.push(path)
        self.last_t = t
        self.rows.append(dict(meta, file=path.name, shock_mean_x_m=float(shock.mean()),
             shock_span_mm=float(np.ptp(shock)*1000), heat_gap_mean_mm=float((shock-xp).mean()*1000),
             thermicity_gap_mean_mm=float((shock-xt).mean()*1000), peak_boundary_fraction=float(np.mean(flag_s|flag_q|flag_t)),
             shock_invalid_fraction=float(np.mean(flag_s)),
             shock_geometry_invalid_fraction=float(np.mean(shock_quality['geometry_invalid'])),
             shock_interpolation_invalid_fraction=float(np.mean(shock_quality['interpolation_invalid'])),
             diagnostic_revision='shock_quality_v2',
             finest_fraction_reaction_band=float(np.mean(coverage[band] == self.engine.c.levels)),
             v_rms_front_m_s=float(np.sqrt(np.mean((u[2]/rho)[band]**2))),
             pressure_strip_max_Pa=float(p.max()), pressure_mean_profile_max_Pa=float(p.mean(axis=1).max()),
             wall_bottom_strip_max_Pa=float(p[:, 0].max()), wall_top_strip_max_Pa=float(p[:, -1].max()),
             reaction_peak_invalid_fraction=float(invalid_q.mean()),
             shock_to_heat_native_cells_median=float(np.median(cell_count)),
             heat_half_max_native_cells_median=float(np.median(half_cells)),
             reaction_band_truncated_fraction=float(np.mean(shock-.0018 < x[0])),
             temperature_outside_mechanism_fraction=float(np.mean((temp < self.gas.min_temp)|(temp > self.gas.max_temp)))))
        self.wall_s += time.perf_counter()-tic; self.write_index()
