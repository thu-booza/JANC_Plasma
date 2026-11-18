"""AMR-front diagnostics with explicit reaction marker and coverage quality."""
from pathlib import Path
import numpy as np
from janc_v2.detonation_amr.strip import FrontRecorder,make_crop
from envelope_crop import EnvelopeCrop


class H5Recorder(FrontRecorder):
    def __init__(self,engine,mechanism,out,mirror=None,**kwargs):
        super().__init__(engine,mechanism,out,mirror,**kwargs)
        self.crop=EnvelopeCrop(engine,mechanism)
        self.gas.TPX=engine.c.t0,engine.c.p0,engine.physics['mole_ratio']
        self.initial_h2=self.gas.Y[0]

    def sample(self,states,infos,t,step,origin,force=False):
        n=len(self.rows);self.observe(states,infos,t,step,force=force)
        if len(self.rows)==n:return None
        r=dict(self.rows[-1]);r['crop_verification']=dict(self.crop.last);p=self.out/r['file']
        with np.load(p,allow_pickle=False) as f:
            crossing=f['leading_pressure_crossing_x_m'];peak=f['heat_peak_x_m']
            geometry=f['shock_geometry_invalid'];qbad=f['reaction_peak_invalid']
            valid=(~geometry)&(~qbad)&(crossing>peak)
            gaps=crossing[valid]-peak[valid]
            roi=(f['x_m'][:,None]>=np.minimum(crossing,peak)[None]-.0001)&(f['x_m'][:,None]<=crossing[None]+.00005)&valid[None]
            fine_fraction=float(np.mean(f['coverage'][roi]==self.engine.c.levels)) if roi.any() else 0.
            r.update(front_lab_m=float(crossing.mean()+origin),front_local_min_m=float(crossing.min()),front_local_max_m=float(crossing.max()),
                     front_span_m=float(np.ptp(crossing)),valid_front=bool(not geometry.any()),
                     shock_marker='leading 2*p0 pressure crossing; gradient-peak interpolation retained separately',
                     reaction_marker='instantaneous Cantera heat-release maximum within 1.8mm behind shock',
                     reaction_gap_m=float(np.median(gaps)) if len(gaps) else None,
                     reaction_gap_valid_fraction=float(valid.mean()),finest_fraction_active_reaction=fine_fraction,
                     reaction_halfmax_cells_median=float(np.median(f['heat_half_max_native_cell_equivalents'][valid])) if valid.any() else None,
                     local_pmax_Pa=r['pressure_strip_max_Pa'],mean_profile_pmax_Pa=r['pressure_mean_profile_max_Pa'],
                     v_rms_m_s=r['v_rms_front_m_s'],window_origin_m=origin)
        return r
