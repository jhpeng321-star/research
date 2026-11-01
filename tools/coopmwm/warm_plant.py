"""Capture genuine sensed warmup history without changing a plant action/RNG."""
from .warm_history import ObservableHistory


def observed_backend(base):
    class ObservableWarmBackend(base):
        def __init__(self,*args,**kw):
            self.causal_history=ObservableHistory()
            super().__init__(*args,**kw)

        def observe(self,env):
            super().observe(env)
            now=float(env.now)
            # Match the registered1s control cadence, retaining actual rows only.
            if now<240 or not now.is_integer():return
            if self.causal_history.frames and now<=self.causal_history.frames[-1]['time_s']:return
            from tools.coopfree.backend import observations
            self.causal_history.push(now,observations(self))
    return ObservableWarmBackend
