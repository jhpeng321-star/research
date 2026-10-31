"""Approval-gated C1SWITCH library adapter. No simulator import at module load.

Mixed actions use disjoint instances of the *same* command-owner and controller
code. CTRL0 still takes the frozen residual/half-second execution path; it is
never approximated by a one-second acceleration or native release.
"""
import contextlib
import hashlib
import importlib
import inspect
import time
from types import SimpleNamespace

from .environment import PerCavHandover
from .integration import load as load_integration
from .protocol import approval_line, guard_world, safe_path
from .guards import episode_guards


def library(root):
    import tools
    root = safe_path(root)
    tools.__path__.append(str(root/'tools'))
    return importlib.import_module('tools.c1switch.sumo')


class CavContext:
    def __init__(self, key, backend):
        from traffic_shepherd.ownership import CommandOwnership
        self.key, self.backend = key, backend
        env, lib = backend.env, backend.lib
        self.owner = CommandOwnership(env.api, {key}, env._circles,
                                      cav_speed_mode=env.ownership.cav_speed_mode)
        self.view = SimpleNamespace(ownership=self.owner, api=env.api, _circles=env._circles,
                                    pi_session=None, _direct_pi=False)
        from tools.c1switch.core import Handover
        self.handover = Handover(lambda name: lib.Controller(name, backend.inputs))
        self.pi = None

    @property
    def name(self):
        return self.handover.name

    def select(self, name):
        if name != self.name:
            self.handover.select(name, self.view)
            self.pi = None
            if name == 'pi':
                from traffic_shepherd.controller_factory import create_pi_session
                self.pi = create_pi_session(self.backend.env.config, {self.key}, desired_speed_mps=31.29)
                self.owner.controller = self.pi.adapter

    def step(self, frames, *, now_s, live_ids):
        env, lib = self.backend.env, self.backend.lib
        own_frames = {self.key: frames[self.key]} if self.key in frames else {}
        if self.pi:
            return self.pi.step(self.owner, own_frames, now_s=now_s, live_ids={self.key})
        wall, defer = {}, False
        if self.name in ('ctrl0', 'mappo'):
            command, _ = self.handover.controller.command(env)
            if self.name == 'ctrl0':
                from wms.control import nativeinit
                with lib.zero_scope():
                    wall = nativeinit.sumo_residual_commands(env.api, {self.key: 0.},
                                                            command.accel_min, command.accel_max)
                from wms.control.execution import enabled
                defer = enabled()
            else:
                values = env._accel_by_vehicle(command)
                if self.key in values:
                    wall = {self.key: values[self.key]}
        self.owner.wall_accel_max_mps2 = env.direct_accel_max_mps2 if wall else None
        kwargs = {'defer_wall': defer} if 'defer_wall' in inspect.signature(self.owner.step).parameters else {}
        return self.owner.step(own_frames, now_s=now_s, live_ids={self.key}, wall_commands=wall,
                               native_ids={self.key} if self.name == 'native' else (), **kwargs)

    def close(self, *, live):
        self.owner.release_all(live_ids={self.key} if live else ())


class MixedFleet:
    def __init__(self, backend):
        self.backend = backend
        self.handover = PerCavHandover(lambda key: CavContext(key, backend))

    def step(self, owner, frames, *, now_s, live_ids, disappeared_ids=(), reappeared_ids=()):
        live = set(live_ids) & set(self.backend.env.cav_ids)
        self.handover.sync(live, retired=set(disappeared_ids) | set(reappeared_ids))
        records = []
        for key in sorted(live):
            context = self.handover.contexts[key]
            records.extend(context.step(frames, now_s=now_s, live_ids=live))
            if context.pi:
                context.pi.commit_evidence()
        return records

    def execute_residual(self, residuals, lo, hi, *, now_s, live_ids, dt=.5):
        commands = {}
        for key, context in self.handover.contexts.items():
            if context.name == 'ctrl0' and key in residuals:
                commands.update(context.owner.execute_residual({key: residuals[key]}, lo, hi,
                                                               now_s=now_s, live_ids=live_ids, dt=dt))
        return commands


class Backend:
    def __init__(self, *, world, variant, registration, c1_root, episode_root, budget,
                 integration, design_sha, evaluation=False, smoke=False, panel=(), mixed_factory=None):
        # Must run before setup(), controller loading, scenario creation or SUMO.
        if not approval_line(registration):
            raise PermissionError('MAIN approval missing; SUMO forbidden')
        guard_world(world, variant, evaluation=evaluation, smoke=smoke, panel=panel)
        receipt = load_integration(integration, registration, c1_root, design_sha, variant=variant)
        self.lib = library(c1_root)
        self.inputs = self.lib.setup()  # Frozen source/controller hashes + nice19/CUDA guard.
        wall_library = importlib.import_module(receipt['wall_module'])
        self.wall_library = wall_library
        import numpy as np
        import random
        import torch
        self.np = np
        random.seed(world['seed'])
        np.random.seed(world['seed'])
        # Controller RNG matches the paired executor; selector sampling uses its
        # own generator in the training supervisor.
        torch.manual_seed(world['seed'])
        from tools.c1switch.core import Belief
        from traffic_shepherd.c4.env import CavSumoEnv
        self.belief, self.digest, self.digest_time = Belief(), hashlib.sha256(), None
        self.timing = dict(controller_wall_s=0., environment_wall_s=0., observable_wall_s=0.)
        self.variant, self.actual, self.budget = variant, 0., budget
        self.env, self.mixed = None, None
        self.handover = wall_library.make_handover(self.inputs)
        budget.reserve(world, variant, evaluation=evaluation, smoke=smoke, panel=panel)
        # Both definitions AND pre-imported env/scenario aliases must be scoped.
        self.guard_scope = episode_guards(world, variant, evaluation=evaluation, smoke=smoke, panel=panel)
        self.guard_scope.__enter__()
        import traci
        self.traci, self.original_step = traci, traci.simulationStep
        self.allowance = 900.
        def charged_step(*args, **kwargs):
            if self.actual+.5 > self.allowance:
                raise ValueError('unreserved simulation step')
            before = float(traci.simulation.getTime())
            result = self.original_step(*args, **kwargs)
            self.actual += float(traci.simulation.getTime())-before
            return result
        traci.simulationStep = charged_step
        try:
            self.env = CavSumoEnv(k_slots=192, controller='direct', ttc=True, physics_log=False,
                                  scenario_root=safe_path(episode_root)/'scenarios', population='textbook')
            self.env.reset(seed=world['seed'], demand_vph=world['demand'],
                           penetration=world['penetration'], warmup_s=300, horizon_s=600,
                           controller='direct', warmup_hook=lambda env: self.lib.Backend.observe(self, env))
            if variant == 'per_cav':
                self.mixed = (mixed_factory or MixedFleet)(self)
                self.env.fleet = self.mixed
                self.mixed.handover.sync(self.self_ids())
                self.original_convert = self.env._accel_by_vehicle
                def finite_convert(value):
                    import math
                    return {k: v for k, v in self.original_convert(value).items() if math.isfinite(v)}
                self.env._accel_by_vehicle = finite_convert
                self.env.ownership.execute_residual = self.mixed.execute_residual
        except BaseException:
            self.close()
            raise

    @property
    def now(self):
        return self.env.now

    def observable(self):
        # C1SWITCH reads CAV self telemetry and rsu_detections only. No ledger counters.
        begin = time.monotonic()
        if not self.belief.history or self.belief.history[-1]['now_s'] != self.now:
            self.lib.Backend.observe(self, self.env)
        result = self.belief.payload()
        self.timing['observable_wall_s'] += time.monotonic()-begin
        return result

    def self_ids(self):
        return [r['vehicle_id'] for r in self.observable()['history'][-1]['self_reports']]

    def select_fleet(self, name):
        self.handover.select(name, self.env)

    def select_per_cav(self, names):
        self.mixed.handover.sync(self.self_ids())
        self.mixed.handover.select(names)

    def tick(self):
        began = time.monotonic()
        command, kwargs = None, {}
        scope = contextlib.nullcontext()
        if self.mixed:
            selected = {key for key, c in self.mixed.handover.contexts.items() if c.name == 'ctrl0'}
            if selected:
                import numpy as np
                # Retain the real typed CTRL0 object, including frozen bounds.
                context = self.mixed.handover.contexts[next(iter(selected))]
                template, _ = context.handover.controller.command(self.env)
                residual = np.full_like(template.residual, float('nan'))
                for key in selected:
                    slot = self.env.builder.pool.slot_of(key)
                    if slot is not None:
                        residual[..., slot] = 0.
                # dataclasses.replace preserves any future typed-command fields.
                from dataclasses import replace
                command = replace(template, residual=residual)
                scope = self.lib.zero_scope()
        elif self.handover.controller:
            command, kwargs = self.handover.controller.command(self.env)
            if self.handover.name == 'ctrl0':
                scope = self.lib.zero_scope()
        self.timing['controller_wall_s'] += time.monotonic()-began
        began = time.monotonic()
        with scope:
            _, info = self.env.step(command, **kwargs)
        self.timing['environment_wall_s'] += time.monotonic()-began
        if any(info['lane_commands'].values()) or (info['done'] and self.now != 900):
            raise ValueError('lane call or premature termination')
        began = time.monotonic()
        self.lib.Backend.observe(self, self.env)
        self.timing['observable_wall_s'] += time.monotonic()-began
        return info['interval']

    def current_names(self):
        if self.mixed:
            return {key: c.name for key, c in self.mixed.handover.contexts.items()}
        return {'fleet': self.handover.name}

    def close(self):
        try:
            if self.env is not None and self.handover.name == 'wall':
                self.handover.controller.stop(now_s=self.env.now)
            if self.mixed:
                for key, context in self.mixed.handover.contexts.items():
                    context.close(live=key in set(self.env.api.vehicle.getIDList()))
            if self.env is not None:
                self.env.close()
        finally:
            self.traci.simulationStep = self.original_step
            self.guard_scope.__exit__(None, None, None)
            self.budget.finish(self.actual)
