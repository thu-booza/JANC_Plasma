"""Persistent accepted-step controller. CFL remains an upper bound, not a reset."""
from dataclasses import dataclass,asdict
import math

@dataclass
class StepControl:
    last_accepted_dt: float | None = None
    next_dt: float | None = None
    successes: int = 0
    accepted: int = 0
    rejected: int = 0
    growth_interval: int = 8
    growth_factor: float = 1.2

    def propose(self,cfl,remaining):
        if not math.isfinite(cfl) or cfl<=0 or not math.isfinite(remaining) or remaining<=0:
            raise ValueError('Finite positive CFL and remaining time required')
        return min(cfl,remaining,self.next_dt if self.next_dt is not None else cfl)

    def reject(self,dt):
        if not math.isfinite(dt) or dt<=0:raise ValueError('Invalid rejected dt')
        self.next_dt=dt*.5;self.successes=0;self.rejected+=1
        return self.next_dt

    def accept(self,dt,endpoint_limited=False):
        if not math.isfinite(dt) or dt<=0:raise ValueError('Invalid accepted dt')
        self.last_accepted_dt=dt;self.accepted+=1
        # A tiny final-time remainder should not throttle a later extended run.
        if endpoint_limited:return
        self.successes+=1;self.next_dt=dt
        if self.successes>=self.growth_interval:
            self.next_dt=dt*self.growth_factor;self.successes=0

    def to_dict(self):return asdict(self)

    @classmethod
    def restore(cls,value):
        obj=cls(**value)
        if obj.growth_interval<1 or not 1<obj.growth_factor<=2:raise ValueError('Invalid growth policy')
        if obj.successes<0 or min(obj.accepted,obj.rejected)<0:raise ValueError('Invalid controller counters')
        for x in (obj.next_dt,obj.last_accepted_dt):
            if x is not None and (not math.isfinite(x) or x<=0):raise ValueError('Invalid persisted dt')
        return obj
