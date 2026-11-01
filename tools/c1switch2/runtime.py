"""Runtime factories: frozen fleet backend and peer-owned grouped per-CAV execution."""
import copy
import importlib
from types import SimpleNamespace

from tools.c1switch import sumo as v7
from tools.c1switch.storage import canonical

from .backend import Backend
from .design import CONTROLLERS
from .observable import anonymous_payload
from .selectors import package
from .wall import Wall, optional


class PerCavBackend(Backend):
    def __init__(self, *args, shared, mixed_factory, **kwargs):
        self.mixed, self._terminal_time = None, None
        super().__init__(*args, **kwargs)
        self.shared = shared
        self.lib = SimpleNamespace(Controller=v7.Controller, zero_scope=v7.zero_scope, Backend=Backend)
        self.timing = dict(controller_wall_s=0., environment_wall_s=0., observable_wall_s=0.)
        self.mixed = mixed_factory(self)
        self.env.fleet = self.mixed
        self.mixed.handover.sync(self.self_ids())
        original = self.env._accel_by_vehicle
        self.original_convert = original
        import math
        self.env._accel_by_vehicle = lambda value: {k: v for k, v in original(value).items() if math.isfinite(v)}
        # The frozen runtime may implement residual recomposition internally
        # without an ownership hook. Training installs this instance hook even
        # when absent; remember that exact prior state for fleet-oracle handback.
        self.residual_existed = hasattr(self.env.ownership, 'execute_residual')
        self.original_residual = getattr(self.env.ownership, 'execute_residual', None)
        self.env.ownership.execute_residual = self.mixed.execute_residual

    def observable(self):
        return self.belief.payload()

    def self_ids(self):
        return {r['vehicle_id'] for r in self.observable()['history'][-1]['self_reports']}

    def current_names(self):
        if self.mixed is None:
            raise ValueError('joint context features requested after fleet handover')
        contexts = self.mixed.handover.contexts
        return {key: contexts[key].name if key in contexts else 'native' for key in self.self_ids()}

    def inference_payload(self):
        if self.mixed is None:
            return super().inference_payload()
        payload = self.observable()
        # Per-CAV policy keeps its own incumbent map, just as during training.
        # No mixed controller internals or evaluation-only memory reach any picker.
        payload['controller_state'] = dict(incumbent='native', now_s=self.env.now,
                                           pi={}, commands={}, wall=None)
        return anonymous_payload(payload, self.wall.validate_memory)

    def retire_mixed(self):
        if self.mixed is None:
            return
        live = set(self.env.api.vehicle.getIDList())
        self.mixed.stop_group(live=live)
        for key, context in self.mixed.handover.contexts.items():
            context.close(live=key in live)
        self.mixed, self.env.fleet = None, None
        if self.residual_existed:
            self.env.ownership.execute_residual = self.original_residual
        else:
            del self.env.ownership.execute_residual
        self.env._accel_by_vehicle = self.original_convert

    def select(self, action):
        if isinstance(action, str):
            if action not in CONTROLLERS:
                raise ValueError('unregistered fleet alternative')
            # Six oracle alternatives are the SAME fleet controllers as every arm,
            # not six "everyone consents" approximations of a per-CAV controller.
            self.retire_mixed()
            return super().select(action)
        if self.mixed is None or set(action) != self.self_ids():
            raise ValueError('joint action requires active grouped backend and exact self roster')
        self.mixed.handover.sync(self.self_ids())
        self.mixed.handover.select(action)

    def advance(self, seconds):
        if self.mixed is None:
            return super().advance(seconds)
        self.reserve(seconds)
        for _ in range(seconds):
            interval = self.shared.Backend.tick(self)
            road, source = interval['speed_deficit_vehicle_seconds'], interval['source_wait_vehicle_seconds']
            self.totals['team_delay_vs'] += road+source
            self.totals['road_delay_vs'] += road
            self.totals['source_wait_vs'] += source
            for key, field in [('collisions', 'collision_involvements'), ('teleports', 'starting_teleports'),
                               ('ttc_below_3_vs', 'ttc_below_3_vehicle_seconds'), ('hard_brake', 'braking_vehicle_seconds')]:
                self.totals[key] += interval.get(field, 0.)
            if self._terminal_time is not None or any(interval.get(k, 0) for k in
                    ('collision_involvements', 'starting_teleports', 'dropped_vehicle_steps')):
                raise ValueError('early termination/safety/slot anomaly')

    def fingerprint(self):
        row = super().fingerprint()
        if self.mixed is not None:
            contexts = {}
            for key, context in self.mixed.handover.contexts.items():
                state = dict(name=context.name, switches=context.handover.switches)
                if context.pi is not None:
                    state['pi'] = copy.deepcopy(context.pi.adapter.pi._state)
                controller = context.handover.controller
                if controller is not None and context.name == 'mappo':
                    mem = controller.exe.memory
                    state['commands'] = {k: None if v is None else v.tolist() for k, v in vars(mem).items()}
                contexts[key] = state
            wall = self.mixed.wall
            law = None if wall is None else wall.law
            group = None
            if law is not None:
                exe = law.executor
                group = dict(members=law.members, cooldown_s=optional(law.cooldown_until),
                    started_s=wall.started_at_s, last_s=wall.last_time_s,
                    phase=None if exe is None else exe.state.phase.tolist(),
                    speed=None if exe is None else exe.speed.tolist(),
                    assembly=None if exe is None else [optional(v) for v in exe.state.assembly_since],
                    hold=None if exe is None else [optional(v) for v in exe.state.assembled_at])
            row['joint_memory_sha256'] = canonical(dict(contexts=contexts, wall=group))
        return row

    def close(self):
        try:
            if getattr(self, 'env', None) is not None:
                self.retire_mixed()
        finally:
            super().close()


class Runtime:
    def __init__(self, binding):
        self.binding = binding
        # This factory is imported/constructed only after runner.admitted().
        self.inputs = v7.setup()

    def create_backend(self, path, world, ledger, stage, ident, *, arm):
        kwargs = dict(wall=Wall())
        cls = Backend
        if arm == 'cav_mappo':
            peers = self.binding['execution_peers']
            shared_name = package(peers['shared'], {'runtime.py', 'observations.py', 'environment.py',
                                                   'integration.py', 'protocol.py', 'guards.py'})
            mixed_name = package(peers['per_cav'], {'mixed.py'})
            shared = importlib.import_module(shared_name+'.runtime')
            factory = importlib.import_module(mixed_name+'.mixed').make_mixed_fleet(shared)
            kwargs.update(shared=shared, mixed_factory=factory)
            cls = PerCavBackend
        return cls(path, world, ledger, stage, ident, self.inputs, **kwargs)


def create(binding):
    return Runtime(binding)
