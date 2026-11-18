"""Opt-in reference operations for a translating predevelopment window.

No solver entry imports this module yet. Uniform *interior* state only; an AMR
hierarchy must never be passed or flattened here. Translation is an integer
index copy, not a Galilean transform: velocities and physical time do not change.
Tail isolation is a caller-verified physical prerequisite, not inferred here.
"""
from dataclasses import asdict, dataclass, replace
import math
import numbers
import numpy as np


@dataclass(frozen=True)
class WindowFrame:
    nx: int
    ny: int
    dx: float
    dy: float
    origin_cells: int = 0
    translations: int = 0

    def __post_init__(self):
        for name in ('nx', 'ny', 'origin_cells', 'translations'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,numbers.Integral):
                raise ValueError(f'{name} must be an integer')
        if min(self.nx,self.ny)<1 or min(self.origin_cells,self.translations)<0:
            raise ValueError('Invalid extent or translation counters')
        if not all(math.isfinite(v) and v>0 for v in (self.dx,self.dy)):
            raise ValueError('Cell sizes must be finite and positive')

    def lab_x(self, local_x):
        return self.origin_cells*self.dx+np.asarray(local_x)

    def to_dict(self):return asdict(self)

    @classmethod
    def from_dict(cls,value):
        if set(value)!=set(cls.__dataclass_fields__):raise ValueError('Frame schema mismatch')
        return cls(**value)


def translate_uniform(state,fresh,frame,shift_cells,*,shock_range_local,
                      minimum_tail_m,ahead_margin_m,tail_isolation_verified,
                      physical_time_s,step,diagnostic_unvalidated_tail=False):
    """Return copied state, new frame and mass/momentum/energy/species receipt.

The caller must persist state, frame and receipt as ONE accepted checkpoint.
This operation cannot certify tail isolation, outflow BCs or mature detonation.
It deliberately rejects AMR arrays, nonintegral movement and unsupported layout.
"""
    if not isinstance(diagnostic_unvalidated_tail,bool):raise ValueError('Diagnostic flag must be Boolean')
    if tail_isolation_verified is not True and not diagnostic_unvalidated_tail:
        raise ValueError('Tail/boundary isolation must be verified before translation')
    if isinstance(shift_cells,bool) or not isinstance(shift_cells,numbers.Integral) or not 0<shift_cells<frame.nx:
        raise ValueError('Shift must be a positive integer smaller than nx')
    if isinstance(step,bool) or not isinstance(step,numbers.Integral) or step<0:
        raise ValueError('Invalid accepted step')
    if not math.isfinite(physical_time_s) or physical_time_s<0:
        raise ValueError('Invalid physical time')
    if not all(math.isfinite(v) and v>0 for v in (minimum_tail_m,ahead_margin_m)):
        raise ValueError('Positive tail and ahead margins required')
    front=np.asarray(shock_range_local,dtype=float)
    if front.shape!=(2,) or not np.isfinite(front).all() or front[0]>front[1]:
        raise ValueError('Provide finite minimum/maximum front coordinates')
    length=frame.nx*frame.dx;offset=shift_cells*frame.dx
    if front[0]-offset<minimum_tail_m or front[1]>length-ahead_margin_m:
        raise ValueError('Translation would violate retained-tail or upstream margin')
    a=np.asarray(state)
    if a.shape!=(14,frame.nx,frame.ny) or a.dtype.kind!='f' or not np.isfinite(a).all():
        raise ValueError('Expected finite 14-field uniform interior state, no ghosts/AMR')
    f=np.asarray(fresh,dtype=a.dtype)
    if f.shape==(14,):f=np.broadcast_to(f[:,None],(14,frame.ny))
    if f.shape!=(14,frame.ny) or not np.isfinite(f).all():
        raise ValueError('Fresh state must be 14 fields, optionally varying across y')
    # A state-health check is still required by the caller after its physical BCs.
    for b in (a,f):
        if np.min(b[0])<=0 or np.min(b[-1])<=0 or np.min(b[-2])<=1:
            raise ValueError('Unphysical density/temperature/gamma')
    new=np.empty_like(a)
    new[:,:-shift_cells]=a[:,shift_cells:]
    new[:,-shift_cells:]=f[:,None,:]
    cell_volume=frame.dx*frame.dy
    total=lambda b:np.sum(b[:12],axis=(1,2),dtype=np.float64)*cell_volume
    before=total(a);after=total(new)
    removed=total(a[:,:shift_cells]);added=total(new[:,-shift_cells:])
    expected=before-removed+added;residual=after-expected
    scale=np.maximum.reduce([abs(before),abs(after),abs(removed)+abs(added),np.ones(12)])
    if np.max(abs(residual)/scale)>1e-11:
        raise ArithmeticError('Translation inventory does not close')
    frame_new=replace(frame,origin_cells=frame.origin_cells+int(shift_cells),translations=frame.translations+1)
    receipt=dict(kind='integer_window_translation_v1',step=int(step),physical_time_s=float(physical_time_s),
        tail_isolation_verified=tail_isolation_verified is True,diagnostic_unvalidated_tail=diagnostic_unvalidated_tail,
        shift_cells=int(shift_cells),shift_m=float(offset),frame_before=frame.to_dict(),frame_after=frame_new.to_dict(),
        conserved_order=['rho','rho_u','rho_v','rho_E','rho_H2','rho_O2','rho_OH','rho_O','rho_H','rho_H2O','rho_HO2','rho_H2O2'],
        before=before.tolist(),after=after.tolist(),removed=removed.tolist(),added=added.tolist(),
        residual=residual.tolist(),scaled_residual_max=float(np.max(abs(residual)/scale)),
        shock_lab_before=frame.lab_x(front).tolist(),shock_lab_after=frame_new.lab_x(front-offset).tolist(),
        note='Per-unit-depth inventories; boundary-flow/source budgets remain separate. Tail isolation asserted by caller, not proven by remap.')
    return new,frame_new,receipt
