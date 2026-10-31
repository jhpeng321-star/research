"""Expected-delay argmin and v7 hand-over extended with wall lifecycle hooks."""
import math
import statistics

from tools.c1switch.core import Handover as V7Handover

from .design import CONTROLLERS, K


def means(samples):
    if set(samples) != set(CONTROLLERS):
        raise ValueError('six-controller library required')
    if any(len(v) != K or any(not math.isfinite(x) or x < 0 for x in v)
           for v in samples.values()):
        raise ValueError('K8 finite nonnegative full-team delay samples required')
    return {c: statistics.fmean(samples[c]) for c in CONTROLLERS}


def choose(costs, incumbent='native'):
    if set(costs) != set(CONTROLLERS) or any(not math.isfinite(v) or v < 0
                                           for v in costs.values()):
        raise ValueError('six finite nonnegative expected delays required')
    # Exact argmin. No dwell, hysteresis, switch penalty, or behavioural cutoff.
    best = min(costs.values())
    ties = [c for c in CONTROLLERS if costs[c] == best]
    return incumbent if incumbent in ties else ties[0]


class Handover(V7Handover):
    """Wall controller implements start/stop; its assembly law stays in S-WALLSW.

    Existing five-way memory/ownership reset is executed by the frozen v7 method.
    A wall-to-wall selection preserves assembly and release state across decisions.
    """
    def __init__(self, factory):
        super().__init__(factory, controllers=CONTROLLERS)
