"""v7 full-prefix branching with 20s replay and a single exact 60s future."""
from tools.c1switch.storage import canonical

from .design import CADENCE, CONTROLLERS, HORIZON, TIMES


def action_valid(action):
    return (isinstance(action, str) and action in CONTROLLERS) or (
        isinstance(action, dict) and all(isinstance(k, str) and v in CONTROLLERS
                                        for k, v in action.items()))


def replay(factory, schedule, alternative, *, expected_prefix):
    if not action_valid(alternative) or len(schedule) >= len(TIMES):
        raise ValueError('unregistered branch')
    if any(not action_valid(c) for c in schedule):
        raise ValueError('invalid replay schedule')
    backend = factory()  # Fresh SUMO starts at zero, native warmup to 300.
    try:
        for action in schedule:
            backend.select(action)
            backend.advance(CADENCE)
        prefix = backend.fingerprint()
        if canonical(prefix) != canonical(expected_prefix):
            raise ValueError('full-rerun trunk bitwise mismatch before alternative')
        base = backend.metrics()
        backend.select(alternative)
        backend.advance(HORIZON)
        return dict(prefix=prefix, **{'60': {k: v-base[k] for k, v in backend.metrics().items()}},
                    end=backend.fingerprint(), actual_sim_s=backend.actual)
    finally:
        backend.close()
