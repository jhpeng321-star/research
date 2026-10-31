"""CPU-only production routing gate, including the frozen control-tick body.

One read-only SSH source capture was made on 2026-11-01. No test connects to
SSH, imports TraCI, loads a checkpoint, or constructs a simulator environment.
On cpuQ, read the frozen source directly through C1SWITCH_RUNTIME; locally,
reconstruct that exact source from the captured residual blocks. Verify its
complete hash before compiling privately. Production mixed contexts, owners,
PI/CIRCLES and handover are real; policy inference and TraCI are synthetic.
The primary CTRL0 contract is C1's forced-base execution at both halfsteps.
"""
import ast
import contextlib
import hashlib
import importlib
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from tools.selmappo2.mixed import make_mixed_fleet
from traffic_shepherd.config import load_config
from traffic_shepherd.damping import CirclesController
from traffic_shepherd.ownership import CommandOwnership
from traffic_shepherd import ownership
from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame


ROOT = Path(__file__).resolve().parents[1]
SHARED = (ROOT / "tools/selmappo" if (ROOT / "tools/selmappo/runtime.py").is_file()
          else Path("E:/agent-work/c4-selmappo/tools/selmappo"))
C1 = (ROOT / "tools/c1switch" if (ROOT / "tools/c1switch/core.py").is_file()
      else Path("E:/agent-work/c4-c1switch2/tools/c1switch"))
ZERO = (Path(os.environ["C1SWITCH_WRAPPERS"]) / "tools/ctrlarm/zero.py"
        if "C1SWITCH_WRAPPERS" in os.environ
        else Path("E:/agent-work/c4-ctrlarm/tools/ctrlarm/zero.py"))
FROZEN_ENV_SHA = "59aecd0bff2c0218ea33450c0726b06bb3497e6fe029a3e3797ee5151b505040"
FROZEN_OWNER_SHA = "05693a7439be998fb0b5764d54b2e2371d6ebae59ac05d6b0fc47687c6edd1fd"
FROZEN_NATIVEINIT_SHA = "f95df10586cfe519c0c6a61eefc5a79f661f3e4a034961748d51d6023ae1cc20"


def extract(path, names):
    source = path.read_text(encoding="utf-8")
    return extract_source(source, names)


def extract_source(source, names):
    tree = ast.parse(source)
    return "\n\n".join("".join("@" + ast.get_source_segment(source, decorator) + "\n"
                               for decorator in node.decorator_list) + ast.get_source_segment(source, node)
                        for node in ast.walk(tree)
                        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names)


# Verbatim cpuQ nativeinit.py:56..64 and 95..119, captured in the single SSH read.
@dataclass(frozen=True)
class NativeResidualCommand:
    residual: np.ndarray
    accel_min: float
    accel_max: float
    native_residual = True

    def __array__(self, *args, **kwargs):
        raise TypeError("residual is not absolute acceleration; use the native-aware SUMO executor")


NATIVE_EXECUTOR = '''
def sumo_residual_commands(api, residuals, lo, hi, *, dt=0.5):
    """Prospective absolute commands for the observable collection action ledger."""
    composed = {}
    for vid, residual in residuals.items():
        if not np.isfinite(residual):
            raise ValueError("nonfinite residual for a designated CAV")
        if residual == 0.0:
            continue
        baseline = sumo_native_accel(api, vid, dt)
        upper = min(float(hi), float(api.vehicle.getAccel(vid)))
        composed[vid] = float(np.clip(baseline + residual, min(lo, baseline), max(upper, baseline)))
    return composed

def execute_sumo_residual(api, residuals, lo, hi, *, dt=0.5):
    """Every physics tick: release zeros; recompute native then add held residual.

    The caller owns only designated CAV ids. Native lanes are never taken over.
    SUMO speedMode27 remains the final safety executor.
    """
    composed = sumo_residual_commands(api, residuals, lo, hi, dt=dt)
    for vid, command in composed.items():
        api.vehicle.setSpeed(vid, -1)  # cancel last tick's external speed influence
        api.vehicle.setAcceleration(vid, command, dt)
    return composed
'''


def frozen_env_source():
    """Read the frozen source on cpuQ; reconstruct it only for local checks."""
    if "C1SWITCH_RUNTIME" in os.environ:
        path = Path(os.environ["C1SWITCH_RUNTIME"]) / "runtime/traffic_shepherd/c4/env.py"
        source = path.read_text(encoding="utf-8")
        assert hashlib.sha256(source.encode()).hexdigest() == FROZEN_ENV_SHA
        return source
    source = (ROOT / "traffic_shepherd/c4/env.py").read_text(encoding="utf-8")
    before = '        api = self.api\n        now = self.now'
    after = '''        api = self.api
        residual_command = accel_by_slot if getattr(accel_by_slot, "native_residual", False) else None
        residuals = None
        if residual_command is not None:
            if self.mode != "direct" or lanes is not None or fallback != "native":
                raise CavEnvError("native residual requires direct mode, native fallback and native lanes")
            residuals = self._accel_by_vehicle(residual_command.residual)
            # Zero residuals remain ordinary native ownership. Nonzero commands are
            # also recorded as direct actions, then recomposed each physics tick below.
            accel_by_slot = None
        now = self.now'''
    assert source.count(before) == 1
    source = source.replace(before, after)
    before = '            requested = self._accel_by_vehicle(accel_by_slot)'
    after = '''            if residuals is not None:
                from wms.control.nativeinit import sumo_residual_commands
                requested = sumo_residual_commands(api, {v: a for v, a in residuals.items() if v in live_cavs},
                                                   residual_command.accel_min, residual_command.accel_max)
            else:
                requested = self._accel_by_vehicle(accel_by_slot)'''
    assert source.count(before) == 1
    source = source.replace(before, after)
    before = '        for substep in range(2):\n            api.simulationStep()'
    after = '''        for substep in range(2):
            if residuals is not None and in_window:
                from wms.control.nativeinit import execute_sumo_residual
                live_now = set(api.vehicle.getIDList()) & self.cav_ids
                execute_sumo_residual(api, {v: a for v, a in residuals.items() if v in live_now},
                                      residual_command.accel_min, residual_command.accel_max)
            api.simulationStep()'''
    assert source.count(before) == 1
    source = source.replace(before, after)
    # Equality to the SHA returned by cpuQ proves this is the actual frozen
    # source, rather than a lookalike replacement for the environment loop.
    assert hashlib.sha256(source.encode()).hexdigest() == FROZEN_ENV_SHA
    return source


@pytest.fixture
def sources(monkeypatch):
    assert hashlib.sha256(Path(ownership.__file__).read_text(encoding="utf-8").encode()).hexdigest() == FROZEN_OWNER_SHA
    package = ModuleType("_selmappo2_runtime_gate_shared")
    package.__path__ = [str(SHARED)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    runtime = importlib.import_module(package.__name__ + ".runtime")
    native = ModuleType("wms.control.nativeinit")
    native.np = np
    native.NativeResidualCommand = NativeResidualCommand
    native.sumo_native_accel = lambda api, key, dt=.5: api.base(key)
    executor = NATIVE_EXECUTOR
    if "C1SWITCH_RUNTIME" in os.environ:
        path = Path(os.environ["C1SWITCH_RUNTIME"]) / "runtime/src/wms/control/nativeinit.py"
        source = path.read_text(encoding="utf-8")
        assert hashlib.sha256(source.encode()).hexdigest() == FROZEN_NATIVEINIT_SHA
        executor = extract_source(source, {"sumo_residual_commands", "execute_sumo_residual"})
    exec(compile(executor, "<cpuQ-nativeinit-capture>", "exec"), native.__dict__)
    for name in ("wms", "wms.control"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["wms.control"].nativeinit = native
    monkeypatch.setitem(sys.modules, "wms.control.nativeinit", native)

    core = ModuleType("tools.c1switch.core")
    core.CONTROLLERS = ("native", "ctrl0", "mappo", "pi", "circles")
    exec(compile(extract(C1 / "core.py", {"Handover"}), str(C1 / "core.py"), "exec"), core.__dict__)
    package_c1 = ModuleType("tools.c1switch")
    package_c1.__path__ = []
    monkeypatch.setitem(sys.modules, "tools.c1switch", package_c1)
    monkeypatch.setitem(sys.modules, "tools.c1switch.core", core)

    zero = ModuleType("_selmappo2_gate_zero")
    zero.np = np
    exec(compile(extract(ZERO, {"forced_base_commands"}), str(ZERO), "exec"), zero.__dict__)
    monkeypatch.setitem(sys.modules, "zero", zero)
    lib = ModuleType("_selmappo2_gate_controller")
    lib.contextlib = contextlib
    exec(compile(extract(C1 / "sumo.py", {"Controller", "zero_scope"}), str(C1 / "sumo.py"), "exec"), lib.__dict__)

    class Policy:
        residual_on_native = True
        memory = NS(present=None)
        dwell = NS(present=None)

        def reset(self):
            pass

        def __call__(self, state, **kwargs):
            if kwargs.get("with_lanes"):
                return np.full((1, len(state)), .25), None
            return NativeResidualCommand(np.ones((1, len(state))), -3., 2.6)

    # Real Controller/ZeroResidual construction, synthetic checkpoint inference only.
    zero.NativeResidualCommand = NativeResidualCommand
    exec(compile(extract(ZERO, {"ZeroResidual"}), str(ZERO), "exec"), zero.__dict__)
    policy = ModuleType("wms.control.policy")
    policy.load_policy_fn = lambda *args, **kwargs: Policy()
    monkeypatch.setitem(sys.modules, "wms.control.policy", policy)
    # The real Controller imports torch for its memory-mask helper. No tensor
    # inference occurs here; the fake checkpoint returns CPU numpy arrays.
    torch = ModuleType("torch")
    torch.is_tensor = lambda value: False
    monkeypatch.setitem(sys.modules, "torch", torch)

    namespace = dict(np=np, math=math, Counter=Counter, CavEnvError=RuntimeError,
                     action_entry=lambda *args: args)
    source = frozen_env_source()
    methods = extract_source(source, {"_control_tick", "_accel_by_vehicle", "now"})
    exec(compile(methods, "<cpuQ-frozen-env-control-tick>", "exec"), namespace)
    yield NS(runtime=runtime, native=native, lib=lib, methods=namespace)
    for name in tuple(sys.modules):
        if name.startswith(package.__name__ + "."):
            del sys.modules[name]


class FakeEnv:
    def __init__(self, sources):
        self.now, self.k = 300., 6
        self.cav_ids = {"a", "b", "c", "d", "e", "f"}
        self.ids = sorted(self.cav_ids)
        self.calls, self.physics, self.residual_calls = [], [], []
        vehicle = NS(getIDList=lambda: self.ids + ["hdv"], getSpeedMode=lambda key: 31,
                     slowDown=lambda *args: self.write("slowDown", *args),
                     setSpeed=lambda *args: self.write("setSpeed", *args),
                     setAcceleration=lambda *args: self.write("setAcceleration", *args))
        self.api = NS(vehicle=vehicle, simulation=NS(getArrivedIDList=lambda: ()),
                      simulationStep=self.advance, base=lambda key: .4 if self.now == 300. else -.2)
        self.config = load_config()
        self._circles = CirclesController(self.config)
        self.ownership = CommandOwnership(self.api, self.cav_ids, self._circles)
        self.direct_accel_max_mps2 = 2.6
        self.builder = NS(vehicle_ids=self.ids,
                          pool=NS(slot_of=lambda key: self.ids.index(key) if key in self.ids else None))
        self.frame = NS(present=np.ones(self.k, bool), is_cav=np.ones(self.k, bool),
                        vid_index=np.arange(self.k))
        self.sensor = NS(bind=lambda table: None, capture=self.capture)
        self.slow_sensor = None
        self._table, self._lc_original = {}, {}
        self._previous_live, self._retired_cavs = set(self.ids), set()
        self.warmup_s, self.mode = 300., "direct"
        self.pi_session, self._direct_pi, self.recorder = None, False, None
        self.command_counts, self.fallback_counts = Counter(), Counter()
        self.lane_command_counts, self.totals, self.window_totals = Counter(), Counter(), Counter()
        self._after_physics = lambda *args: {}
        self._lane_control = lambda *args: None
        self.convert = sources.methods["_accel_by_vehicle"].__get__(self)
        self._accel_by_vehicle = lambda values: {key: value for key, value in self.convert(values).items()
                                               if math.isfinite(value)}
        self._control_tick = sources.methods["_control_tick"].__get__(self)

    def state(self):
        return self.ids

    def write(self, kind, key, *args):
        assert key in self.cav_ids, "write to nonconsenting HDV"
        self.calls.append((self.now, kind, key, *args))

    def advance(self):
        self.physics.append((self.now, list(self.calls)))
        self.now += .5

    def capture(self, key, *, now_s):
        return SensorFrame(key, now_s, 20., 0., 31.29, 0, distance_to_merge_m=1000.,
                           front=NeighborMeasurement(30., 0.), front_coverage_m=100., rear_coverage_m=100.)


def backend(sources):
    class CpuBackend(sources.runtime.Backend):
        """Use real Backend.tick/now without its simulator-initializing constructor."""
        def __init__(self):
            self.env = FakeEnv(sources)
            self.lib = sources.lib
            self.inputs = {"ctrl0": "fake-ctrl0", "mappo": "fake-mappo"}
            self._terminal_time = None
            self.timing = dict(controller_wall_s=0., environment_wall_s=0., observable_wall_s=0.)
            self.mixed = self.env.fleet = make_mixed_fleet(sources.runtime)(self)
            self.mixed.handover.sync(self.env.ids)
    return CpuBackend()


def test_production_ctrl0_context_selects_native_without_second_policy_call(sources):
    b = backend(sources)
    context = b.env.fleet.handover.contexts["b"]
    context.select("ctrl0")
    context.handover.controller.command = lambda env: pytest.fail("CTRL0 context reran policy inference")
    records = context.step({"b": b.env.capture("b", now_s=300.)}, now_s=300., live_ids=b.env.ids)
    assert len(records) == 1
    assert records[0].vehicle_id == "b" and records[0].owner == "native"
    assert records[0].wall_request_mps2 is None
    assert b.env.calls == []


def test_original_owner_has_neither_invented_api(sources):
    assert not hasattr(CommandOwnership, "execute_residual")
    assert "defer_wall" not in __import__("inspect").signature(CommandOwnership.step).parameters


def test_production_mappo_absolute_output_only_commands_selected_cav(sources):
    b = backend(sources)
    c = b.env.fleet.handover.contexts["c"]
    c.select("mappo")
    records = c.step({"c": b.env.capture("c", now_s=300.)}, now_s=300., live_ids=b.env.ids)
    assert len(records) == 1
    assert records[0].vehicle_id == "c" and records[0].wall_request_mps2 == .25
    assert b.env.calls == [(300., "slowDown", "c", 20.25, 1.)]


@pytest.mark.parametrize("forced", [True, False], ids=["production-C1CTRL0-forced-base", "raw-zero-explicit-contrast"])
def test_production_contexts_frozen_control_tick_and_both_halfsteps(sources, forced):
    b = backend(sources)
    env = b.env
    names = dict(a="native", b="ctrl0", c="mappo", d="pi", e="circles", f="native")
    env.fleet.handover.select(names)
    original_execute = sources.native.execute_sumo_residual

    def trace(api, residuals, lo, hi, **kwargs):
        env.residual_calls.append((env.now, dict(residuals), lo, hi))
        return original_execute(api, residuals, lo, hi, **kwargs)
    sources.native.execute_sumo_residual = trace  # private capture, never peer module
    if not forced:
        sources.lib.zero_scope = contextlib.nullcontext
    sources.lib.Backend = NS(observe=lambda *args: None)

    def step(command, **kwargs):
        assert env._accel_by_vehicle(command.residual) == {"b": 0.}
        interval = env._control_tick(command, **kwargs)
        return None, dict(done=False, lane_commands={}, interval=interval)
    env.step = step
    assert b.tick() == {}
    assert env.now == 301.
    assert [t for t, _ in env.physics] == [300., 300.5]
    assert env.residual_calls == [(300., {"b": 0.}, -3., 2.6), (300.5, {"b": 0.}, -3., 2.6)]
    assert {key for _, kind, key, *_ in env.calls if kind == "slowDown"} == {"c", "d", "e"}
    acceleration = [call for call in env.calls if call[1] == "setAcceleration"]
    if forced:
        assert acceleration == [(300., "setAcceleration", "b", .4, .5),
                                (300.5, "setAcceleration", "b", -.2, .5)]
        assert [call for call in env.calls if call[1] == "setSpeed"] == [
            (300., "setSpeed", "b", -1), (300.5, "setSpeed", "b", -1)]
    else:
        assert not acceleration
        assert all(call[2] != "b" for call in env.calls)
    assert all(call[2] not in {"a", "f", "hdv"} for call in env.calls)
    print("CPU_TRACE", "forced_base" if forced else "raw_zero", env.calls, env.residual_calls)


def test_ctrl0_must_not_fall_through_to_circles_when_zero_skips(sources):
    b = backend(sources)
    c = b.env.fleet.handover.contexts["b"]
    c.select("circles")
    c.step({"b": b.env.capture("b", now_s=300.)}, now_s=300., live_ids=b.env.ids)
    assert c.owner._owned == {"b": "damping"}
    b.env.now = 301.
    c.select("ctrl0")
    assert b.env.calls[-1] == (301., "setSpeed", "b", -1)
    c.step({"b": b.env.capture("b", now_s=301.)}, now_s=301., live_ids=b.env.ids)
    assert c.owner._last["b"].owner == "native"
    assert len([call for call in b.env.calls if call[0] == 301.]) == 1


def test_production_handover_refreshes_view_clock(sources):
    b = backend(sources)
    c = b.env.fleet.handover.contexts["a"]
    c.select("circles")
    stops = []
    c.handover.controller.stop = lambda **kwargs: stops.append(kwargs)
    b.env.now = 320.
    c.select("native")
    assert stops == [{"now_s": 320.}]
    assert c.view.now == 320.
    b.env.now = 321.
    c.select("native")
    assert c.view.now == 321.


def test_nonselected_nan_slots_require_existing_finite_filter(sources):
    b = backend(sources)
    env = b.env
    env.fleet.handover.select({"b": "ctrl0"})
    residual = np.full((1, 6), np.nan)
    residual[0, 1] = 0.
    env._accel_by_vehicle = env.convert
    with pytest.raises(ValueError, match="nonfinite residual"):
        env._control_tick(NativeResidualCommand(residual, -3., 2.6))
    assert env.physics == []


def test_mixed_raw_zero_matches_original_frozen_native_release_trace(sources):
    mixed = backend(sources).env
    original = FakeEnv(sources)
    original.fleet = None
    mixed.fleet.handover.select({"b": "ctrl0"})
    residual = np.full((1, 6), np.nan)
    residual[0, 1] = 0.
    command = NativeResidualCommand(residual, -3., 2.6)
    mixed._control_tick(command)
    original._control_tick(command)
    assert mixed.calls == original.calls == []
    assert mixed.physics == original.physics == [(300., []), (300.5, [])]
    assert mixed.command_counts == original.command_counts == Counter(native=6)


def test_production_terminal_wall_cleanup_uses_cached_clock_without_writes(sources):
    b = backend(sources)
    fleet = b.env.fleet
    env = b.env
    # Populate both real owners so a stale live mask would produce writes.
    for key in ("a", "b"):
        context = fleet.handover.contexts[key]
        context.select("circles")
        context.step({key: env.capture(key, now_s=300.)}, now_s=300., live_ids=env.ids)
        assert context.owner._owned == {key: "damping"}
    env.calls.clear()
    stopped = []
    fleet.members, fleet.live = {"a", "b"}, set(b.env.ids)
    fleet.wall = NS(stop=lambda **kwargs: stopped.append(kwargs))
    # The actual Backend subclass inherits its cached now property. The frozen
    # env clock getter would raise if cleanup queried this closed connection.
    terminal = type("ClosedEnv", (), {"now": sources.methods["now"], "api": None})()
    b.env, b._terminal_time = terminal, 900.
    assert b.now == 900.
    for context in fleet.handover.contexts.values():
        context.close(live=False)
    assert stopped == [{"now_s": 900.}]
    assert env.calls == []
    assert fleet.wall is None and fleet.members == set() and fleet.requests == {}
    assert all(not c.owner._owned for c in fleet.handover.contexts.values())
