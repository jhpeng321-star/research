"""Eight-arm episode and live-prefix full-rerun measurement protocol (CPU core)."""
import random

from .core import choose
from .design import CONTROLLERS, measurement_indices
from .rerun import replay
from .storage import canonical

ARMS = ('wm', 'bb', 'random') + CONTROLLERS


def measure(factory_for, schedule, live_prefix):
    """Five fresh processes/backends; compare to the actual policy state, not each other.

    The host factory must isolate an active live SUMO connection from each rerun,
    e.g. by executing replay in a child process. No native repeat is billed in main.
    """
    return {c:replay(factory_for(c), schedule, c, expected_prefix=live_prefix)
            for c in CONTROLLERS}


def episode(backend, arm, world, *, predictor=None, measurement=None, receipt=None):
    """SUMO is paused during both prediction and exact measurement.

    Measurement costs/results are evaluation-only, never provided to the predictor;
    picks are committed to a receipt before measurement can produce an outcome.
    """
    if arm not in ARMS:
        raise ValueError('unregistered episode arm')
    if arm in ('wm', 'bb') and (predictor is None or measurement is None):
        raise ValueError('switching requires inference and live-prefix measurement')
    schedule, decisions = [], []
    measured = measurement_indices(world['seed']) if arm in ('wm', 'bb') else []
    incumbent = 'native'
    start = backend.metrics()
    for j in range(10):
        # Complete observable state is copied at the boundary, no backend reference
        # reaches the predictor. Exact fingerprints/cell identities remain evaluation.
        payload = backend.belief.payload()
        prefix = backend.fingerprint()
        if arm in ('wm', 'bb'):
            costs = predictor.predict(payload, f'{arm}_s{world["seed"]}_j{j}')
            picked = choose(costs, incumbent)
        elif arm == 'random':
            costs = None
            picked = random.Random(f'0:{world["seed"]}:{j}').choice(CONTROLLERS)
        else:
            costs, picked = None, arm
        row = dict(j=j, t_s=300+60*j, pick=picked, incumbent=incumbent,
                   predicted_costs=costs, belief_sha256=canonical(payload), prefix=prefix)
        if receipt is not None:
            receipt('pick', row)
        # No simulator tick or controller handover occurs before exact branching.
        # Branch truth cannot influence this already-written decision.
        if j in measured:
            row['branches'] = measurement(list(schedule), prefix)
            if set(row['branches']) != set(CONTROLLERS):
                raise ValueError('incomplete five-way measurement')
            if canonical(backend.fingerprint()) != canonical(prefix):
                raise ValueError('measurement changed live episode state')
            if receipt is not None:
                receipt('measurement', row)
        backend.select(picked)
        backend.advance(60)
        incumbent = picked
        schedule.append(picked)
        decisions.append(row)
    final = backend.metrics()
    return dict(arm=arm, world=world, schedule=schedule, decisions=decisions,
                J600={k:final[k]-start[k] for k in final},
                switching_frequency=sum(a != b for a,b in zip(['native']+schedule, schedule)))
