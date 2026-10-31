"""Real SUMO backend. Import only inside the existing gpuR CPU environment."""
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

from .core import Belief, Handover
from .design import FORBIDDEN, guard, sha
from .storage import canonical

RUNTIME = Path('/root/autodl-tmp/c4/speedhead20261030/e3final')
WRAPPERS = Path('/root/wmdata2')
MAPPO = Path('/root/autodl-tmp/c4/speedhead20261030/mappospeed_final169.pt')
SOURCE_SHA = 'b3c031adb6bc2b5c2bca160bea308f83ba5567b97c1cd0e853eddcd8367a8193'


def setup():
    import tools
    tools.__path__.append(str(WRAPPERS/'tools'))
    sys.path[:0] = [str(RUNTIME/'runtime/src'), str(RUNTIME/'runtime'),
                   str(WRAPPERS/'tools/speedhead'), str(WRAPPERS/'tools/ctrlarm')]
    # Resolve the frozen application's traffic_shepherd, not this checkout's sources.
    import importlib.metadata
    import shutil
    import torch
    torch.set_num_threads(1)
    if os.getpriority(os.PRIO_PROCESS, 0) != 19 or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('nice19, CPU-only environment required')
    if importlib.metadata.version('eclipse-sumo') != '1.27.1' or not shutil.which('sumo'):
        raise ValueError('SUMO 1.27.1 required on PATH before allocation')
    manifest = RUNTIME/'runtime/SOURCE_MANIFEST.json'
    if sha(manifest) != SOURCE_SHA:
        raise ValueError('runtime manifest mismatch')
    entries = json.loads(manifest.read_text())
    if len(entries) != 543:
        raise ValueError('runtime file count')
    for name, expected in entries.items():
        if not FORBIDDEN.search(name) and sha(RUNTIME/'runtime'/name) != expected:
            raise ValueError('runtime hash mismatch')
    expected = {
        WRAPPERS/'tools/ctrlarm/zero.py':'24095a1092eafb4c894804aa8142478a802eaea046142ec777e5f24f49f73596',
        WRAPPERS/'tools/speedhead/observer.py':'b22af4623b5fe7fbb9a8920b8e8417c3591efae66a52bae7b72f4d899b4a7874',
        MAPPO:'d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca',
        RUNTIME/'inputs/policy_latest.pt':'64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c',
    }
    if any(sha(p) != value for p, value in expected.items()):
        raise ValueError('controller binding mismatch')
    # Reuse the current saved-state/RNG engine, including loaderTime fixes, with
    # the frozen E3 plant. Source hash is bound by the immutable deployment receipt.
    import importlib.util
    import traffic_shepherd.c4 as package
    root = Path('/root/c1switch20261031')
    engine = root/'pytools/engine/branching.py'
    staged = json.loads((root/'STAGED.json').read_text())
    if sha(engine) != staged['engine/branching.py']:
        raise ValueError('branching engine changed')
    spec = importlib.util.spec_from_file_location('traffic_shepherd.c4.branching', engine)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    package.branching = module
    return {name:str(p) for name, p in [('mappo', MAPPO), ('ctrl0', RUNTIME/'inputs/policy_latest.pt')]}


class Controller:
    """Same direct native-lane policy/fallback contract used by the audit."""
    def __init__(self, name, inputs):
        self.name, self.exe, self.previous = name, None, None
        if name in ('mappo', 'ctrl0'):
            from wms.control.policy import load_policy_fn
            self.exe = load_policy_fn(inputs[name], device='cpu')
            self.exe.reset()
            if name == 'ctrl0':
                from zero import ZeroResidual
                self.exe = ZeroResidual(self.exe)

    def command(self, env):
        if self.name in ('native', 'pi', 'circles'):
            return None, dict(fallback={'native':'native', 'pi':'pi', 'circles':'damping'}[self.name])
        if self.name == 'ctrl0':
            return self.exe(env.state()), {}
        import numpy as np
        import torch
        ids = np.asarray(env.frame.vid_index)
        keep = np.zeros(len(ids), bool) if self.previous is None else (ids == self.previous) & (ids >= 0)
        self.previous = ids.copy()
        for obj in (self.exe.memory, self.exe.dwell):
            if obj.present is not None:
                obj.present = obj.present & torch.as_tensor(keep, device=obj.present.device)[None]
        out, lanes = self.exe(env.state(), with_lanes=True)
        if lanes is not None:
            raise ValueError('speed-only MAPPO attempted lanes')
        return out, dict(lanes=None)


@contextlib.contextmanager
def zero_scope():
    from zero import forced_base_commands
    from wms.control import nativeinit
    original = nativeinit.sumo_residual_commands
    def forced(api, residuals, lo, hi, **kw):
        return forced_base_commands(api, residuals, lo, hi, dt=kw.get('dt', .5))
    nativeinit.sumo_residual_commands = forced
    try:
        yield
    finally:
        nativeinit.sumo_residual_commands = original


class Backend:
    def __init__(self, root, world, ledger, stage, ident, inputs):
        import numpy as np
        from traffic_shepherd.c4 import branching as B
        from traffic_shepherd.c4.env import CavSumoEnv
        self.B, self.np = B, np
        self.root, self.world = root, world
        self.ledger, self.stage, self.ident, self.inputs = ledger, stage, ident, inputs
        self.actual, self.allowed, self.serial, self.env = 0., 0., 0, None
        self.belief = Belief()
        self.handover = Handover(lambda name:Controller(name, self.inputs))
        self.totals = {k:0. for k in ('team_delay_vs', 'road_delay_vs', 'source_wait_vs',
                                      'collisions', 'teleports', 'ttc_below_3_vs', 'hard_brake')}
        self.digest = hashlib.sha256()
        guard(world)
        # Count actual physics steps, including partial reset/advance failures.
        import traci
        self.traci, self.original_step = traci, traci.simulationStep
        def step(*args, **kwargs):
            if self.actual+.5 > self.allowed:
                raise ValueError('physics step without precharge')
            before = float(traci.simulation.getTime())
            result = self.original_step(*args, **kwargs)
            self.actual += float(traci.simulation.getTime())-before
            return result
        traci.simulationStep = step
        try:
            self.reserve(300)
            self.env = CavSumoEnv(k_slots=192, controller='direct', ttc=True,
                                  extra_sumo_cli=B.SAVE_STATE_CLI, scenario_root=root/'scenarios',
                                  physics_log=False, population='textbook')
            self.env.reset(seed=world['seed'], demand_vph=world['demand'],
                           penetration=world['penetration'], warmup_s=300, horizon_s=631,
                           controller='direct', warmup_hook=lambda e:self.observe(e))
            self.observe(self.env)
        except BaseException:
            from .storage import write
            write(root/'initialization_failure.json', dict(actual_sim_s=self.actual, allowed_sim_s=self.allowed))
            self.close()
            raise

    def reserve(self, seconds):
        self.ledger.charge(f'{self.ident}_{self.serial:04d}', seconds, self.stage)
        self.serial += 1
        self.allowed += seconds

    def observe(self, env):
        from traffic_shepherd.c4.state import RSU_MAINLINE_S_M, RSU_RAMP_S_M, rsu_detections
        f, ids = env.frame, env.builder.vehicle_ids
        # The frame is accessed only to obtain CAV self telemetry; HDV rows come from RSU.
        own = [dict(vehicle_id=ids[int(f.vid_index[i])], timestamp_s=float(f.time_s),
                    s_m=float(f.s[i]), lane=int(f.lane_carry[i] if f.junction[i] else f.lane[i]),
                    v_mps=float(f.v[i]), a_mps2=float(f.a[i]))
               for i in self.np.flatnonzero(f.present & f.is_cav)]
        coverage = [dict(lane=l, s0_m=z[0], s1_m=z[1], timestamp_s=env.now)
                    for l, z in ((0, RSU_MAINLINE_S_M), (1, RSU_MAINLINE_S_M), (2, RSU_RAMP_S_M))]
        return self.belief.observe(env.now, self_reports=own,
                                   detections=rsu_detections(f, ids), coverage=coverage)

    def select(self, name):
        self.handover.select(name, self.env)

    def advance(self, seconds):
        self.reserve(seconds)
        for _ in range(seconds):
            controller = self.handover.controller
            cmd, kw = controller.command(self.env) if controller else (None, {})
            with zero_scope() if self.handover.name == 'ctrl0' else contextlib.nullcontext():
                _, info = self.env.step(cmd, **kw)
            interval = info['interval']
            road, source = interval['speed_deficit_vehicle_seconds'], interval['source_wait_vehicle_seconds']
            self.totals['team_delay_vs'] += road+source
            self.totals['road_delay_vs'] += road
            self.totals['source_wait_vs'] += source
            for k, field in [('collisions', 'collision_involvements'), ('teleports', 'starting_teleports'),
                             ('ttc_below_3_vs', 'ttc_below_3_vehicle_seconds'), ('hard_brake', 'braking_vehicle_seconds')]:
                self.totals[k] += interval.get(field, 0.)
            self.observe(self.env)
            raw = self.env.history.raw(3)
            # Evaluation-only exact tape: never present in inference payload.
            self.digest.update(canonical({k:raw[k][-2:].tolist() for k in ('kin', 'mask', 'vid_index')}).encode())
            if any(info['lane_commands'].values()):
                raise ValueError('unexpected controller lane command')
            if info['done'] or any(interval.get(k, 0) for k in
                                    ('collision_involvements', 'starting_teleports', 'dropped_vehicle_steps')):
                raise ValueError('early termination/safety/slot anomaly')

    def save(self, label):
        path = self.root/f'state_{self.serial:04d}_{label}.xml'
        self.B.save_state(self.env, path)
        return (path, self.B.snapshot_env(self.env),
                copy.deepcopy((self.handover.name, self.handover.controller, self.handover.switches,
                               self.belief, self.totals)), self.digest.copy())

    def load(self, state):
        path, snap, python, digest = state
        self.B.load_state(self.env, path, snap)
        (self.handover.name, self.handover.controller, self.handover.switches,
         self.belief, self.totals) = copy.deepcopy(python)
        self.digest = digest.copy()
        self.B.prepare_branch_mode(self.env)

    def metrics(self):
        return dict(self.totals)

    def close(self):
        if self.env is not None:
            self.env.close()
        self.traci.simulationStep = self.original_step
