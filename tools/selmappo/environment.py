"""Thirty-decision controller-selection environment over a tick backend.

The backend alone owns controllers/SUMO. Observations and interval reward take
separate paths; no metric/truth object can enter the feature builder.
"""
import math
from numbers import Integral

from .observations import build
from .protocol import CADENCE, LIBRARY, TIMES


def action_name(action):
    if isinstance(action, bool) or not isinstance(action, Integral) or not 0 <= action < len(LIBRARY):
        raise ValueError(f'action must be an integer controller index in [0,{len(LIBRARY)})')
    return LIBRARY[int(action)]


def window_reward(intervals):
    road = source = 0.
    for interval in intervals:
        a = float(interval['speed_deficit_vehicle_seconds'])
        b = float(interval['source_wait_vehicle_seconds'])
        if not all(math.isfinite(x) and x >= 0 for x in (a, b)):
            raise ValueError('invalid full-team reward')
        road += a
        source += b
    return -(road+source), dict(road_delay_vs=road, source_wait_vs=source, team_delay_vs=road+source)


class SelectorEnv:
    def __init__(self, backend, variant):
        if variant not in ('fleet', 'per_cav') or backend.now != 300:
            raise ValueError('selector starts at native warmup t300')
        self.backend, self.variant = backend, variant
        self.index, self.incumbents = 0, {}
        self.observation = self.observe()

    def observe(self):
        return build(self.backend.observable(), variant=self.variant, incumbents=self.incumbents)

    def step(self, actions):
        if self.index >= len(TIMES) or self.backend.now != TIMES[self.index]:
            raise ValueError('terminal or off-cadence decision')
        # Validate ALL actions before releasing any ownership.
        if set(actions) != set(self.observation.keys):
            raise ValueError('actions must match decision-time CAVs exactly')
        names = {key: action_name(action) for key, action in actions.items()}
        if self.variant == 'fleet':
            self.backend.select_fleet(names['fleet'])
        else:
            self.backend.select_per_cav(names)
        self.incumbents = names
        intervals = []
        for _ in range(CADENCE):
            interval = self.backend.tick()
            if any(interval.get(k, 0) for k in ('collision_involvements', 'starting_teleports',
                                                'dropped_vehicle_steps')):
                raise ValueError('safety/slot fault: stop dispatch')
            intervals.append(interval)
        reward, metrics = window_reward(intervals)
        self.index += 1
        if self.backend.now != 300+CADENCE*self.index:
            raise ValueError('window clock mismatch')
        if hasattr(self.backend, 'current_names'):
            self.incumbents = self.backend.current_names()
        self.observation = self.observe()
        if self.variant == 'per_cav':
            self.incumbents = {k: self.incumbents.get(k, 'native') for k in self.observation.keys}
        for source, target in (('ttc_below_3_vehicle_seconds', 'ttc_below_3_vs'),
                               ('braking_vehicle_seconds', 'hard_brake_vs'),
                               ('collision_involvements', 'collisions'),
                               ('starting_teleports', 'teleports')):
            metrics[target] = math.fsum(float(row.get(source, 0)) for row in intervals)
        return self.observation, reward, self.index == len(TIMES), metrics


class PerCavHandover:
    """Disjoint contexts, each implementing the original C1SWITCH hand-over.

    A context's select releases only its own command owner and resets controller
    state on change. Entries have no inherited per-CAV action before a decision.
    """
    def __init__(self, factory):
        self.factory, self.contexts = factory, {}
        self.switches = 0

    def sync(self, live, *, retired=()):
        live = set(live)
        for key in list(self.contexts):
            if key not in live or key in retired:
                self.contexts.pop(key).close(live=key in live)
        for key in sorted(live - set(self.contexts)):
            self.contexts[key] = self.factory(key)

    def select(self, names):
        if not set(names) <= set(self.contexts) or any(c not in LIBRARY for c in names.values()):
            raise ValueError('invalid per-CAV hand-over')
        for key, name in names.items():
            old = self.contexts[key].name
            self.contexts[key].select(name)
            self.switches += int(old != name)
