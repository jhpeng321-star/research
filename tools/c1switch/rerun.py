"""Full-prefix counterfactuals; deliberately no save/load API or state reuse."""
from .design import CONTROLLERS
from .storage import canonical


def replay(factory, schedule, alternative, *, expected_prefix=None, horizon=90):
    """Factory starts native t0..300; replay completed 60s decisions exactly.

    Prefix equality is checked BEFORE selecting the alternative or simulating any
    counterfactual tick. Controller memory is rebuilt by execution, not snapshots.
    """
    if alternative not in CONTROLLERS or horizon not in (60, 90):
        raise ValueError('unregistered alternative/horizon')
    if len(schedule) > 9 or any(c not in CONTROLLERS for c in schedule):
        raise ValueError('invalid prior controller schedule')
    backend = factory()
    try:
        for name in schedule:
            backend.select(name)
            backend.advance(60)
        prefix = backend.fingerprint()
        if expected_prefix is not None and canonical(prefix) != canonical(expected_prefix):
            raise ValueError('full-rerun trunk bitwise mismatch before alternative')
        base = backend.metrics()
        belief = backend.belief.payload()
        backend.select(alternative)
        backend.advance(60)
        sixty = {k:v-base[k] for k,v in backend.metrics().items()}
        if horizon == 90:
            backend.advance(30)
        end = backend.metrics()
        return dict(prefix=prefix, belief=belief, **{'60':sixty},
                    **{str(horizon):{k:v-base[k] for k,v in end.items()}},
                    end=backend.fingerprint(), actual_sim_s=backend.actual)
    finally:
        backend.close()


def verify_native_repeat(reference, repeat):
    # actual_sim_s is cost metadata; every scientific bit must match.
    keys = ('prefix', 'belief', '60', '90', 'end')
    if canonical({k:reference[k] for k in keys}) != canonical({k:repeat[k] for k in keys}):
        raise ValueError('native full-rerun continuation bitwise mismatch')


def candidates(factory_for, schedule):
    """Native repeat gate precedes every nonnative continuation, including first."""
    rows = {}
    rows['native'] = replay(factory_for('native'), schedule, 'native')
    rows['native_repeat'] = replay(factory_for('native_repeat'), schedule, 'native',
                                   expected_prefix=rows['native']['prefix'])
    verify_native_repeat(rows['native'], rows['native_repeat'])
    for name in CONTROLLERS[1:]:
        rows[name] = replay(factory_for(name), schedule, name,
                            expected_prefix=rows['native']['prefix'])
    return rows
