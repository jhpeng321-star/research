"""Receding-horizon runner. Truth and fingerprints never reach any picker."""
import copy
import random
import time

from tools.c1switch.storage import canonical

from .core import choose
from .design import ARMS, CADENCE, CONTROLLERS, MODEL_ARMS, PICKERS, TIMES, measurement_indices
from .rerun import action_valid


def episode(backend, arm, world, *, predictor=None, selector=None, receipt=None):
    if arm not in ARMS+('wm_phys',):
        raise ValueError('unregistered arm')
    if arm in MODEL_ARMS and predictor is None:
        raise ValueError('missing model predictor')
    if arm in ('fleet_ppo', 'cav_mappo') and selector is None:
        raise ValueError('missing frozen RL picker')
    schedule, decisions, branches = [], [], []
    incumbent = 'native'
    start = backend.metrics()
    measured = measurement_indices(world['seed']) if arm in PICKERS+('wm_phys',) else []
    for j, now in enumerate(TIMES):
        prefix = backend.fingerprint()
        if prefix['t_s'] != now:
            raise ValueError('decision clock mismatch')
        payload = backend.inference_payload()
        began = time.monotonic()
        costs = None
        if arm in MODEL_ARMS:
            # Predictor obtains ONLY anonymous observables and causal command state.
            costs = predictor.predict(copy.deepcopy(payload), f'{arm}_s{world["seed"]}_j{j}')
            picked = choose(costs, incumbent if isinstance(incumbent, str) else 'native')
        elif arm in ('fleet_ppo', 'cav_mappo'):
            picked = selector.select(copy.deepcopy(payload))
            if arm == 'fleet_ppo' and not isinstance(picked, str):
                raise ValueError('fleet picker must return one shared controller')
            if arm == 'cav_mappo':
                own = {r['vehicle_id'] for r in payload['history'][-1]['self_reports']}
                if not isinstance(picked, dict) or set(picked) != own:
                    raise ValueError('per-CAV picker must cover exactly live self-reporting CAVs')
        elif arm == 'random':
            picked = random.Random(f'S-C1SWITCH2:random:{world["seed"]}:{j}').choice(CONTROLLERS)
        else:
            picked = arm
        if not action_valid(picked):
            raise ValueError('invalid library action')
        if canonical(prefix) != canonical(backend.fingerprint()):
            raise ValueError('picker modified live simulator')
        row = dict(j=j, t_s=now, pick=copy.deepcopy(picked), incumbent=copy.deepcopy(incumbent),
                   predicted_costs=costs, belief_sha256=canonical(payload), prefix=prefix,
                   picker_wall_s=time.monotonic()-began)
        if receipt is not None:
            receipt('pick', row)  # Durable before intervention or branch outcomes.
        if j in measured:
            branches.append(dict(j=j, schedule=copy.deepcopy(schedule), prefix=prefix,
                                 chosen=copy.deepcopy(picked)))
        backend.select(picked)
        backend.advance(CADENCE)  # Only 20s of the imagined 60s is ever executed.
        incumbent = copy.deepcopy(picked)
        schedule.append(copy.deepcopy(picked))
        decisions.append(row)
    if backend.fingerprint()['t_s'] != 900:
        raise ValueError('episode did not end at 900s')
    end = backend.metrics()
    j600 = {k: end[k]-start[k] for k in end}
    # Secondary J900 is 900 CONTROL seconds, not warmup plus J600. No additional
    # picker decisions: retain the t=880 controller/assembly through t=1200.
    backend.advance(300)
    if backend.fingerprint()['t_s'] != 1200:
        raise ValueError('J900 tail did not end at 1200s')
    long_end = backend.metrics()
    return dict(arm=arm, world=world, schedule=schedule, decisions=decisions,
                measurement_specs=branches, J600=j600,
                J900={k: long_end[k]-start[k] for k in long_end},
                switch_count=sum(a != b for a, b in zip(['native']+schedule, schedule)),
                wall_decisions=sum(a == 'wall' if isinstance(a, str) else
                                   any(c == 'wall' for c in a.values()) for a in schedule))
