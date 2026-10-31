"""CPU contracts: real frozen WALL + command owner, synthetic observable receipts."""
import importlib
import inspect
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import pytest

from tools.selmappo2.mixed import make_mixed_fleet
from traffic_shepherd.config import load_config
from traffic_shepherd.damping import CirclesController
from traffic_shepherd.ownership import CommandOwnership
from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame


SHARED = Path("E:/agent-work/c4-selmappo/tools/selmappo")
WALL = Path("E:/agent-work/c4-wallsw/traffic_shepherd/c4")


@pytest.fixture(scope="module")
def sources():
    """Private source imports, without patching any peer's modules or loaders."""
    if not (SHARED / "runtime.py").is_file() or not (WALL / "switchable_wall.py").is_file():
        pytest.skip("explicit peer source trees unavailable")
    names = ("_selmappo2_mixed_shared", "_selmappo2_mixed_c4")
    for name, root in zip(names, (SHARED, WALL)):
        package = ModuleType(name)
        package.__path__ = [str(root)]
        package.ROOT = root.parents[1]
        sys.modules[name] = package
    runtime = importlib.import_module(f"{names[0]}.runtime")
    wall = importlib.import_module(f"{names[1]}.switchable_wall")
    yield NS(runtime=runtime, wall=wall)
    for key in tuple(sys.modules):
        if any(key == name or key.startswith(name + ".") for name in names):
            del sys.modules[key]


class FakeHandover:
    def __init__(self, context):
        self.context = context
        self.name, self.controller, self.switches = "native", None, 0

    def select(self, name, view):
        assert name != "wall", "per-CAV WALL must remain a sentinel"
        if name != self.name:
            view.ownership.release_all(live_ids={self.context.key})
            view.ownership.controller = self.context.backend.env._circles
            self.name, self.controller = name, NS(name=name, memory=object())
            self.switches += 1


class PeerContext:
    """Synthetic peer outputs; adapter must delegate non-WALL calls unchanged.

    The real shared MixedFleet loop/residual dispatcher and real CommandOwnership
    remain in use. Learned policies/PI are spies, not checkpoint loads.
    """
    def __init__(self, key, backend):
        self.key, self.backend = key, backend
        env = backend.env
        self.owner = CommandOwnership(env.api, {key}, env._circles)
        self.view = NS(ownership=self.owner, pi_session=None, _direct_pi=False)
        self.handover = FakeHandover(self)
        self.pi, self.steps, self.residual_calls = None, [], []

        def residual(*args, **kwargs):
            self.residual_calls.append((args, kwargs))
            return {key: .125}
        self.owner.execute_residual = residual

    @property
    def name(self):
        return self.handover.name

    def select(self, name):
        self.handover.select(name, self.view)

    def step(self, frames, *, now_s, live_ids):
        self.steps.append((self.name, now_s, set(live_ids)))
        return self.owner.step({self.key: frames[self.key]} if self.key in frames else {},
                               now_s=now_s, live_ids={self.key},
                               native_ids={self.key} if self.name == "native" else ())

    def close(self, *, live):
        self.owner.release_all(live_ids={self.key} if live else ())


def receipt(key, lane, position, speed=20., now=300.):
    return dict(vehicle_id=key, timestamp_s=now, s_m=position, lane=lane,
                v_mps=speed, a_mps2=0.)


class PublicBackend:
    def __init__(self):
        self.calls, self.created = [], []
        self.rows = {r["vehicle_id"]: r for r in (
            receipt("a", 0, -1000), receipt("b", 1, -1050),
            receipt("c", 2, -1600), receipt("d", 0, -1500),
            receipt("e", 1, -1501), receipt("f", 2, -1800))}
        self.live, self.detections = set(self.rows), []
        env = NS(now=300., k=16, cav_ids=set(self.rows), direct_accel_max_mps2=2.6)

        def write(kind, *args):
            caller = inspect.currentframe().f_back.f_back
            assert caller.f_code.co_filename.endswith("ownership.py"), "independent writer"
            self.calls.append((env.now, kind, *args))

        env.api = NS(vehicle=NS(
            getIDList=lambda: sorted(self.live | {"hdv"}), getSpeedMode=lambda key: 31,
            slowDown=lambda *args: write("slowDown", *args),
            setSpeed=lambda *args: write("setSpeed", *args)))
        env._circles = CirclesController(load_config())
        self.env = env

    def observable(self):
        return {"history": [dict(now_s=self.env.now, self_reports=list(self.rows.values()),
                                 detections=self.detections)]}

    def advance(self, delta=1.):
        self.env.now += delta
        for row in list(self.rows.values()) + self.detections:
            row["timestamp_s"] = self.env.now

    def frames(self):
        return {key: SensorFrame(key, self.env.now, row["v_mps"], 0., 31.29,
                                row["lane"], distance_to_merge_m=-row["s_m"],
                                front=NeighborMeasurement(30., 0.),
                                front_coverage_m=100., rear_coverage_m=100.)
                for key, row in self.rows.items() if key in self.live}


@pytest.fixture
def setup(sources):
    backend = PublicBackend()
    shared = NS(CavContext=PeerContext, MixedFleet=sources.runtime.MixedFleet,
                PerCavHandover=sources.runtime.PerCavHandover)

    def new_wall():
        wall = sources.wall.SwitchableWallController()
        backend.created.append(wall)
        return wall

    fleet = make_mixed_fleet(shared, wall_factory=new_wall)(backend)
    fleet.handover.sync(backend.live)
    return backend, fleet


def tick(backend, fleet, **lifecycle):
    return fleet.step(None, backend.frames(), now_s=backend.env.now,
                      live_ids=backend.live | {"hdv"}, **lifecycle)


def select(fleet, **names):
    fleet.handover.select(names)


def assert_single_writer(backend):
    counts = Counter((time, key) for time, kind, key, *_ in backend.calls if kind == "slowDown")
    assert counts and max(counts.values()) == 1
    assert all(call[2] in backend.env.cav_ids for call in backend.calls)


def test_real_law_consent_single_writer_and_nonmember_circles(setup, sources):
    backend, fleet = setup
    select(fleet, a="wall", b="wall", c="wall", d="native", e="mappo", f="pi")
    rows = {r.vehicle_id: r for r in tick(backend, fleet)}
    assert len(backend.created) == 1
    assert type(fleet.wall.law) is sources.wall.WallLaw
    assert fleet.members == {"a", "b"}  # d/e are a closer pair but did not consent
    assert {key for key, r in rows.items() if r.wall_request_mps2 is not None} == {"a", "b"}
    assert rows["c"].owner == "damping"
    assert rows["d"].owner == "native"
    for key in ("d", "e", "f"):
        context = fleet.handover.contexts[key]
        assert context.steps == [(context.name, backend.env.now, backend.live)]
    for key in ("a", "b", "c"):
        context = fleet.handover.contexts[key]
        assert not context.steps  # WALL alone takes the adapter's owner path
        assert context.handover.controller is None
        assert context.owner.controller is backend.env._circles
        assert context.owner.wall_accel_max_mps2 is None
    assert_single_writer(backend)


def test_no_cross_consent_pair_and_no_wall_until_selected(setup):
    backend, fleet = setup
    tick(backend, fleet)
    assert not backend.created
    backend.advance()
    select(fleet, a="wall")
    rows = tick(backend, fleet)
    assert fleet.members == set()
    assert all(r.wall_request_mps2 is None for r in rows)
    backend.advance()
    select(fleet, b="wall")
    tick(backend, fleet)
    assert len(backend.created) == 1
    assert fleet.members == {"a", "b"}


def test_unchanged_selection_preserves_assembly_hold_release_and_cooldown(setup):
    backend, fleet = setup
    select(fleet, a="wall", b="wall")
    tick(backend, fleet)
    wall, law, executor = fleet.wall, fleet.wall.law, fleet.wall.law.executor
    contexts = dict(fleet.handover.contexts)
    backend.advance()
    select(fleet, a="wall", b="wall")
    backend.rows["a"].update(s_m=-900., v_mps=17.)
    backend.rows["b"].update(s_m=-903., v_mps=17.)
    tick(backend, fleet)
    assert fleet.wall is wall and wall.law is law and law.executor is executor
    assert executor.state.phase[0] == 1  # actual frozen HOLD
    assert executor.state.assembly_since[0] == 300.
    assert fleet.handover.contexts == contexts
    backend.advance()
    backend.rows["a"]["s_m"], backend.rows["b"]["s_m"] = 0., -3.
    tick(backend, fleet)
    assert law.executor is None and not fleet.members
    assert law.cooldown_until == backend.env.now + law.COOLDOWN_S
    backend.advance(law.COOLDOWN_S - 1.)
    backend.rows["a"].update(s_m=-1000., v_mps=20.)
    backend.rows["b"].update(s_m=-1050., v_mps=20.)
    select(fleet, a="wall", b="wall")
    tick(backend, fleet)
    assert fleet.wall is wall and wall.law is law and law.executor is None
    backend.advance()
    tick(backend, fleet)
    assert law.n_walls == 2
    assert sum(e["event"] == "merge_release" for e in wall.events) == 1
    assert_single_writer(backend)


def test_withdrawal_releases_both_before_dispatch_without_touching_peer_memory(setup):
    backend, fleet = setup
    select(fleet, a="wall", b="wall", c="wall", e="mappo")
    tick(backend, fleet)
    wall = fleet.wall
    peer = fleet.handover.contexts["e"].handover.controller
    backend.advance()
    start = len(backend.calls)
    select(fleet, a="native")
    assert not wall.running and wall.law is None and not fleet.requests
    assert [(kind, key) for _, kind, key, *_ in backend.calls[start:]] == [
        ("setSpeed", "a"), ("setSpeed", "b")]
    rows = {r.vehicle_id: r for r in tick(backend, fleet)}
    assert rows["a"].owner == "native"
    assert rows["b"].owner == rows["c"].owner == "damping"
    assert not fleet.members  # no consenting lane-0 replacement
    assert fleet.handover.contexts["e"].handover.controller is peer
    assert_single_writer(backend)


@pytest.mark.parametrize("reuse", [False, True])
def test_departure_or_same_id_reentry_stops_group_and_retires_context(setup, reuse):
    backend, fleet = setup
    select(fleet, a="wall", b="wall")
    tick(backend, fleet)
    wall, old_context = fleet.wall, fleet.handover.contexts["a"]
    backend.advance()
    if not reuse:
        backend.live.remove("a")
        del backend.rows["a"]
    start = len(backend.calls)
    rows = tick(backend, fleet, disappeared_ids={"a"},
                reappeared_ids={"a"} if reuse else ())
    assert not wall.running and wall.law is None
    assert not fleet.members
    assert not any(r.wall_request_mps2 is not None for r in rows)
    assert backend.calls[start][1:3] == (("setSpeed", "a") if reuse else ("setSpeed", "b"))
    if not reuse:
        assert all(call[2] != "a" for call in backend.calls[start:])
        backend.advance()
        backend.live.add("a")
        backend.rows["a"] = receipt("a", 0, -1000., now=backend.env.now)
        tick(backend, fleet, reappeared_ids={"a"})
    assert fleet.handover.contexts["a"] is not old_context
    assert fleet.handover.contexts["a"].name == "native"
    backend.advance()
    select(fleet, a="wall")
    tick(backend, fleet)
    assert fleet.members == {"a", "b"} and fleet.wall is not wall
    assert fleet.wall.law.executor.state.assembly_since[0] == backend.env.now
    assert_single_writer(backend)


def test_nonmember_consent_changes_do_not_replace_active_pair(setup):
    backend, fleet = setup
    select(fleet, a="wall", b="wall", c="wall")
    tick(backend, fleet)
    wall, executor = fleet.wall, fleet.wall.law.executor
    backend.advance()
    select(fleet, c="circles", d="wall", e="wall")
    tick(backend, fleet)
    assert fleet.wall is wall and wall.law.executor is executor
    assert fleet.members == {"a", "b"}  # newly consenting closer pair cannot steal membership
    backend.advance()
    select(fleet, d="native", e="native")
    tick(backend, fleet)
    assert fleet.wall is wall and wall.law.executor is executor


def test_missing_member_self_report_stops_atomically_even_with_rsu_sighting(setup):
    backend, fleet = setup
    select(fleet, a="wall", b="wall")
    tick(backend, fleet)
    wall = fleet.wall
    backend.advance()
    backend.detections = [backend.rows.pop("a")]
    rows = tick(backend, fleet)
    assert not wall.running and not fleet.members
    assert not any(r.wall_request_mps2 is not None for r in rows)
    assert not fleet.requests


def test_rsu_nonmember_rear_risk_reuses_frozen_abort_and_cooldown(setup):
    backend, fleet = setup
    select(fleet, a="wall", b="wall")
    backend.detections = [receipt("hdv", 0, -1055., speed=30.)]
    rows = tick(backend, fleet)
    assert not fleet.members
    assert not any(r.wall_request_mps2 is not None for r in rows)
    law = fleet.wall.law
    assert law.cooldown_until == backend.env.now + law.COOLDOWN_S
    assert any(e["event"] == "safety_abort" and e["reason"].startswith("non_member_rear_end_risk")
               for e in fleet.wall.events)
    assert all(call[2] != "hdv" for call in backend.calls)


@pytest.mark.parametrize("bad", [{"a": "wall", "b": "unknown"}, {"absent": "wall"}])
def test_invalid_selection_has_no_partial_handback(setup, bad):
    backend, fleet = setup
    select(fleet, a="wall", b="wall")
    tick(backend, fleet)
    calls, wall = list(backend.calls), fleet.wall
    with pytest.raises(ValueError, match="hand-over"):
        fleet.handover.select(bad)
    assert backend.calls == calls and fleet.wall is wall and wall.running
    assert fleet.members == {"a", "b"}


def test_ctrl0_residual_dispatch_is_exact_shared_implementation(setup, sources):
    backend, fleet = setup
    select(fleet, a="wall", b="wall", e="ctrl0", f="mappo")
    tick(backend, fleet)
    assert fleet.execute_residual.__func__ is sources.runtime.MixedFleet.execute_residual
    result = fleet.execute_residual({"a": 9., "e": -.5, "f": 8.}, -3., 2.6,
                                    now_s=300.5, live_ids=backend.live, dt=.5)
    assert result == {"e": .125}
    context = fleet.handover.contexts["e"]
    assert context.residual_calls == [(({"e": -.5}, -3., 2.6),
                                       dict(now_s=300.5, live_ids=backend.live, dt=.5))]
    assert all(not c.residual_calls for key, c in fleet.handover.contexts.items() if key != "e")


def test_actual_shared_nonwall_step_remains_disjoint(sources):
    # Only construction/checkpoint outputs are faked. Run the peer's actual
    # non-WALL step rather than a second implementation of MAPPO/native routing.
    class Context(PeerContext):
        select = sources.runtime.CavContext.select
        step = sources.runtime.CavContext.step

    backend = PublicBackend()
    backend.lib = NS()
    backend.env._accel_by_vehicle = lambda value: value
    shared = NS(CavContext=Context, MixedFleet=sources.runtime.MixedFleet,
                PerCavHandover=sources.runtime.PerCavHandover)
    fleet = make_mixed_fleet(shared, wall_factory=sources.wall.SwitchableWallController)(backend)
    fleet.handover.sync(backend.live)
    select(fleet, a="wall", b="wall", c="circles", d="mappo")
    fleet.handover.contexts["d"].handover.controller.command = lambda env: (
        {key: .25 for key in backend.live}, {})
    rows = {r.vehicle_id: r for r in tick(backend, fleet)}
    assert rows["d"].wall_request_mps2 == .25
    assert rows["c"].owner == "damping" and rows["e"].owner == "native"
    assert fleet.members == {"a", "b"}
    assert rows["c"].wall_request_mps2 is None
    assert_single_writer(backend)


def test_unrelated_entrant_and_departure_preserve_group_and_start_native(setup):
    backend, fleet = setup
    select(fleet, a="wall", b="wall")
    tick(backend, fleet)
    wall, executor = fleet.wall, fleet.wall.law.executor
    backend.advance()
    backend.live.add("entrant")
    backend.env.cav_ids.add("entrant")
    backend.rows["entrant"] = receipt("entrant", 0, -1500., now=backend.env.now)
    rows = {r.vehicle_id: r for r in tick(backend, fleet)}
    assert rows["entrant"].owner == "native"
    assert fleet.wall is wall and wall.law.executor is executor
    backend.advance()
    backend.live.remove("entrant")
    del backend.rows["entrant"]
    tick(backend, fleet, disappeared_ids={"entrant"})
    assert fleet.wall is wall and wall.law.executor is executor
    assert "entrant" not in fleet.handover.contexts


def test_public_junction_flag_excludes_formation_without_global_reads(setup):
    backend, fleet = setup
    select(fleet, a="wall", b="wall")
    backend.rows["a"]["junction"] = True
    rows = tick(backend, fleet)
    assert not fleet.members
    assert not any(r.wall_request_mps2 is not None for r in rows)
    backend.advance()
    backend.rows["a"]["junction"] = False
    tick(backend, fleet)
    assert fleet.members == {"a", "b"}


def test_context_close_stops_shared_wall_before_releasing_owners(setup):
    backend, fleet = setup
    select(fleet, a="wall", b="wall")
    tick(backend, fleet)
    wall = fleet.wall
    fleet.handover.contexts["a"].close(live=True)
    assert not wall.running and not fleet.members
    assert not fleet.handover.contexts["b"].owner._owned
