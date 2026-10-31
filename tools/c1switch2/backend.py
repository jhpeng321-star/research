"""Frozen v7 SUMO backend overlay. No edits to, or mutation of, v7 modules.

Only dependency injection (seed guard, six-way handover/factory), wall lane audit,
and full-episode evaluation accounting differ. All original plant code is reused.
"""
import contextlib
import importlib.util
from pathlib import Path

from tools.c1switch import sumo as v7
from tools.c1switch.main_backend import MainBackend

from .core import Handover
from .design import guard
from .observable import anonymous_payload


def frozen_overlay(wall):
    # A PRIVATE module namespace avoids changing another imported runner's globals.
    name = 'tools.c1switch._c1switch2_sumo_overlay'
    spec = importlib.util.spec_from_file_location(name, Path(v7.__file__))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.guard, module.Handover = guard, Handover
    module.Controller = lambda name, inputs: wall.create_sumo() if name == 'wall' else v7.Controller(name, inputs)
    return module


class Backend(MainBackend):
    def __init__(self, root, world, ledger, stage, ident, inputs, *, wall):
        self.wall = wall
        self.warmup_metrics = None
        frozen_overlay(wall).Backend.__init__(self, root, world, ledger, stage, ident, inputs)
        # Horizon is only env termination metadata; scenario demand already spans
        # the unchanged 2400s fixture. Extend before any controlled physics tick.
        if self.env.config.episode_seconds < 1231:
            raise ValueError('frozen fixture cannot support J900 control tail')
        self.env.horizon_s, self.env.end_s = 931., 1231.
        if self.warmup_metrics is None:
            raise ValueError('missing full 300s native warmup metrics')

    def observe(self, env):
        super().observe(env)
        if env.now == 300 and self.warmup_metrics is None:
            fields = dict(team_delay_vs=None, road_delay_vs='speed_deficit_vehicle_seconds',
                          source_wait_vs='source_wait_vehicle_seconds',
                          collisions='collision_involvements', teleports='starting_teleports',
                          ttc_below_3_vs='ttc_below_3_vehicle_seconds', hard_brake='braking_vehicle_seconds')
            self.warmup_metrics = {k: float(env.totals.get(v, 0)) for k, v in fields.items() if v}
            self.warmup_metrics['team_delay_vs'] = (self.warmup_metrics['road_delay_vs']+
                                                  self.warmup_metrics['source_wait_vs'])

    def inference_payload(self):
        if self.handover.name == 'wall':
            payload = self.belief.payload()
            payload['controller_state'] = dict(incumbent='wall', now_s=float(self.env.now),
                pi={}, commands={}, wall=self.wall.capture_memory(self.handover.controller, self.env))
        else:
            payload = super().inference_payload()
            payload['controller_state']['wall'] = None
        return anonymous_payload(payload, self.wall.validate_memory)

    def advance(self, seconds):
        if self.handover.name != 'wall':
            return super().advance(seconds)
        # Preserve the exact v7 one-second command and two-half-second plant ticks.
        self.reserve(seconds)
        for _ in range(seconds):
            command, kw = self.handover.controller.command(self.env)
            with contextlib.nullcontext():
                _, info = self.env.step(command, **kw)
            interval = info['interval']
            road, source = interval['speed_deficit_vehicle_seconds'], interval['source_wait_vehicle_seconds']
            self.totals['team_delay_vs'] += road+source
            self.totals['road_delay_vs'] += road
            self.totals['source_wait_vs'] += source
            for key, field in [('collisions', 'collision_involvements'), ('teleports', 'starting_teleports'),
                               ('ttc_below_3_vs', 'ttc_below_3_vehicle_seconds'), ('hard_brake', 'braking_vehicle_seconds')]:
                self.totals[key] += interval.get(field, 0.)
            self.observe(self.env)
            self.wall.validate_lane_commands(info['lane_commands'])
            if info['done'] or any(interval.get(k, 0) for k in
                    ('collision_involvements', 'starting_teleports', 'dropped_vehicle_steps')):
                raise ValueError('early termination/safety/slot anomaly')

    def episode_metrics(self):
        return {k: self.warmup_metrics[k]+v for k, v in self.metrics().items()}

    def fingerprint(self):
        from tools.c1switch.storage import canonical
        row = super().fingerprint()
        row['controller_state_sha256'] = canonical(self.inference_payload()['controller_state'])
        return row

    def close(self):
        try:
            if getattr(self, 'env', None) is not None and self.handover.name == 'wall':
                self.handover.controller.stop(now_s=self.env.now)
        finally:
            super().close()
