"""Exact 4,680 sim-s mechanical gate; never invoked by CPU tests or polling."""
from tools.c1switch.storage import canonical, write

from .design import CONTROLLERS, canary_world
from .runner import admitted, backend


def run(root, binding):
    if admitted(root, phase='canary') != binding:
        raise ValueError('canary runtime binding mismatch')
    world = canary_world()
    def make(label):
        return backend(root, binding, 'canary', world, 'canary__'+label)
    native = []
    # 2 x (300+60): prove native rerun repeat before ANY nonnative command.
    for label in ('native_reference', 'native_repeat'):
        env = make(label)
        try:
            prefix = env.fingerprint()
            env.select('native')
            env.advance(60)
            native.append(dict(prefix=prefix, end=env.fingerprint(), metrics=env.metrics()))
        finally:
            env.close()
    if canonical(native[0]) != canonical(native[1]):
        raise ValueError('native full-rerun canary failed')
    # 6 x (300+60): each alternative starts fresh, verifies before intervention.
    branches = {}
    for controller in CONTROLLERS:
        env = make('branch_'+controller)
        try:
            if canonical(env.fingerprint()) != canonical(native[0]['prefix']):
                raise ValueError('canary candidate native prefix mismatch')
            env.select(controller)
            env.advance(60)
            branches[controller] = env.metrics()
        finally:
            env.close()
    # 2 x 900: repeated nonnative trunks exercise all 6, starts/stops and memory.
    episodes = []
    for label in ('six_way_reference', 'six_way_repeat'):
        env = make(label)
        try:
            prefixes = []
            for j in range(30):
                prefixes.append(env.fingerprint())
                env.select(CONTROLLERS[j % 6])
                env.advance(20)
            episodes.append(dict(prefixes=prefixes, end=env.fingerprint(), metrics=env.metrics()))
        finally:
            env.close()
    if canonical(episodes[0]) != canonical(episodes[1]):
        raise ValueError('six-way full-rerun repeat failed')
    result = dict(native_repeat='PASS', six_way_repeat='PASS', planned_sim_s=4680,
                  native=native, branches=branches, episodes=episodes)
    write(root/'CANARY.json', result)
    return result
