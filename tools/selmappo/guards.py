"""Episode-scoped guards, including frozen modules' captured import aliases."""
from contextlib import contextmanager

from .protocol import guard_world


@contextmanager
def episode_guards(world, variant, *, evaluation=False, smoke=False, panel=()):
    guard_world(world, variant, evaluation=evaluation, smoke=smoke, panel=panel)
    from traffic_shepherd import seeds, scenarios
    from traffic_shepherd.c4 import seeds as c4seeds, env
    targets = [(c4seeds, 'check_case'), (env, 'check_case'),
               (seeds, 'require_development_case'), (scenarios, 'require_development_case')]
    originals = [(module, key, getattr(module, key)) for module, key in targets]

    def check(seed, demand_vph):
        if type(seed) is not int or seed != world['seed'] or demand_vph != world['demand']:
            raise ValueError('case differs from exact current episode')
        guard_world(world, variant, evaluation=evaluation, smoke=smoke, panel=panel)

    def source(seed, loading, role):
        check(seed, loading)
        if role != 'exposed':
            raise ValueError('current episode must be exposed TRAIN/evaluation')

    try:
        for module, key in targets:
            setattr(module, key, check if key == 'check_case' else source)
        yield
    finally:
        for module, key, original in originals:
            setattr(module, key, original)
