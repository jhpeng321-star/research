"""PL01-B: PlannerDecision -> dispatcher adapter (final mixed_assignment/3)."""
import importlib.util
import sys
import types

import pytest

from traffic_shepherd.planner import (
    WallSpec,
    DISPATCH_V2_SCHEMA,
    DISPATCH_V3_SCHEMA,
    AssignmentPlanV2,
    DispatcherPlanV3,
    DispatcherWallV3,
    PlannerConfig,
    PlannerDecision,
    WallAssignmentError,
    plan_id_for,
    to_dispatcher_v3,
    to_mixed_assignment_v2,
)
from traffic_shepherd.planner.dispatch import to_mixed_assignment_v3_real
from traffic_shepherd.planner.paths import research_dense_root
from tests.planner._helpers import plan, wall

REAL_V3_SOURCE = research_dense_root() / "traffic_shepherd" / "mixed_assignment.py"
REAL_V3_PACKAGE = "_pl01_real_mixed_assignment"


def decision_with_wall():
    hold = wall("w0", ("c0", "c1"), speed=16.0, phase="hold", hold_s=60.0)
    return PlannerDecision(plan=plan({"c2": "damping"}, walls=[hold]),
                           reasons=())


def decision_with_two_walls():
    hold_a = wall("w0", ("c0", "c1"), speed=16.0, phase="hold", hold_s=60.0)
    hold_b = wall("w1", ("c2", "c3"), speed=20.0, phase="hold", hold_s=120.0)
    return PlannerDecision(plan=plan({"c4": "native"}, walls=[hold_a, hold_b]),
                           reasons=())


def plain_decision():
    return PlannerDecision(plan=plan({"c0": "native", "c1": "damping"}),
                           reasons=())


class TestDispatcherV3:
    def test_round_trip_with_wall(self):
        cfg = PlannerConfig()
        out = to_dispatcher_v3(decision_with_wall(), cfg)
        assert isinstance(out, DispatcherPlanV3)
        assert out.schema == DISPATCH_V3_SCHEMA == "mixed_assignment/3"
        assert out.commit_horizon_s == 0.0  # WM-40: dwell 0
        assert out.strict_roster is True
        assert out.plan_id == plan_id_for(decision_with_wall().plan)
        # wall members are (re-)assigned wall-v2; free CAVs keep their action
        assert out.assignments == {"c0": "wall-v2", "c1": "wall-v2",
                                   "c2": "damping"}
        assert len(out.walls) == 1
        dispatched = out.walls[0]
        assert dispatched.wall_id == "w0"
        assert dispatched.members == ("c0", "c1")  # lane-ordered tuple
        assert dispatched.wall_speed_mps == 16.0
        assert dispatched.phase == "hold"
        assert dispatched.hold_s == 60.0 and dispatched.release_at_s is None

    def test_round_trip_without_wall(self):
        out = to_dispatcher_v3(plain_decision())
        assert out.walls == ()
        assert out.assignments == {"c0": "native", "c1": "damping"}
        assert out.commit_horizon_s == 0.0

    def test_release_is_encoded_by_wall_omission(self):
        # a hold wall released after hold: the loop omits the wall and
        # assigns its members native IN THE SAME plan; the adapter emits
        # no wall for it.
        released = PlannerDecision(
            plan=plan({"c0": "native", "c1": "native"}), reasons=())
        out = to_dispatcher_v3(released)
        assert out.walls == ()
        assert out.assignments == {"c0": "native", "c1": "native"}

    def test_release_wall_becomes_omission_with_native_members(self):
        # toward predict() a release candidate keeps the wall as
        # phase="release" with locked members; toward the WM-40 dispatcher
        # the release stays encoded as OMISSION (members -> native)
        released = WallSpec(wall_id="w0", members=("c0", "c1"),
                            speed_mps=16.0, phase="release",
                            release_at_s=0.0, hold_s=None)
        decision = PlannerDecision(plan=plan({}, walls=[released]), reasons=())
        out = to_dispatcher_v3(decision)
        assert out.walls == ()  # omitted from the dispatcher walls
        assert out.assignments == {"c0": "native", "c1": "native"}

    def test_release_at_s_offset_converts_to_absolute_episode_time(self):
        held = WallSpec(wall_id="w0", members=("c0", "c1"), speed_mps=16.0,
                        phase="hold", release_at_s=15.0, hold_s=None)
        decision = PlannerDecision(plan=plan({"c2": "damping"},
                                             walls=[held]), reasons=())
        out = to_dispatcher_v3(decision, t0_s=1000.0)
        assert out.walls[0].release_at_s == 1015.0  # absolute = t0 + offset
        assert out.walls[0].hold_s is None
        passthrough = to_dispatcher_v3(decision)
        assert passthrough.walls[0].release_at_s == 15.0  # no t0: unchanged

    def test_full_map_strict_rejects_unknown_actions(self):
        with pytest.raises(ValueError, match="invalid assignment modes"):
            DispatcherPlanV3(assignments={"c0": "turbo"}, plan_id="p")

    def test_wall_v2_union_must_coincide(self):
        with pytest.raises(ValueError, match="coincide"):
            DispatcherPlanV3(assignments={"c0": "wall-v2"}, plan_id="p")
        plan_out = DispatcherPlanV3(
            assignments={"c0": "wall-v2", "c1": "wall-v2"},
            plan_id="p",
            walls=(DispatcherWallV3(wall_id="w", members=("c0", "c1"),
                                    wall_speed_mps=16.0, phase="hold",
                                    hold_s=60.0),))
        assert plan_out.walls[0].members == ("c0", "c1")

    def test_plan_validation_unique_ids_and_disjoint_members(self):
        with pytest.raises(ValueError, match="unique"):
            DispatcherPlanV3(
                assignments={"c0": "wall-v2", "c1": "wall-v2"}, plan_id="p",
                walls=(DispatcherWallV3("w", ("c0", "c1"), 16.0, "hold",
                                        hold_s=1.0),
                       DispatcherWallV3("w", ("c2", "c3"), 16.0, "hold",
                                        hold_s=1.0)))
        with pytest.raises(ValueError, match="two walls"):
            DispatcherPlanV3(
                assignments={"c0": "wall-v2", "c1": "wall-v2",
                             "c2": "wall-v2", "c3": "wall-v2"},
                plan_id="p",
                walls=(DispatcherWallV3("w0", ("c0", "c1"), 16.0, "hold",
                                        hold_s=1.0),
                       DispatcherWallV3("w1", ("c1", "c3"), 16.0, "hold",
                                        hold_s=1.0)))

    def test_wall_validation_v3_schema(self):
        def build(**over):
            args = dict(wall_id="w", members=("c0", "c1"),
                        wall_speed_mps=16.0, phase="hold", hold_s=60.0)
            args.update(over)
            return DispatcherWallV3(**args)

        with pytest.raises(ValueError, match="at least two"):
            build(members=("c0",))  # a wall spans >= 2 lanes
        with pytest.raises(ValueError, match="two walls|distinct"):
            build(members=("c0", "c0"))
        with pytest.raises(ValueError, match="> 2.0"):
            build(wall_speed_mps=2.0)  # strictly above the dispatcher floor
        with pytest.raises(ValueError, match="> 2.0"):
            build(wall_speed_mps=0.0)
        with pytest.raises(ValueError, match="exactly one"):
            build(hold_s=None, release_at_s=None)
        with pytest.raises(ValueError, match="exactly one"):
            build(hold_s=60.0, release_at_s=15.0)
        with pytest.raises(ValueError, match="phase"):
            build(phase="cruise")
        with pytest.raises(ValueError, match="release_at_s"):
            build(hold_s=None, release_at_s=-1.0)
        assert build().hold_s == 60.0
        assert build(hold_s=None, release_at_s=120.0).release_at_s == 120.0


class TestMixedAssignmentV2Fallback:
    def test_zero_walls_round_trip(self):
        cfg = PlannerConfig(v2_commit_horizon_s=10.0)
        out = to_mixed_assignment_v2(plain_decision(), cfg)
        assert isinstance(out, AssignmentPlanV2)
        assert out.assignments == {"c0": "native", "c1": "damping"}  # FULL map
        assert out.commit_horizon_s == 10.0  # minimal positive from config
        assert out.wall_members == frozenset()
        assert out.wall_candidate is None
        assert out.plan_id == plan_id_for(plain_decision().plan)

    def test_plan_id_deterministic(self):
        assert plan_id_for(plain_decision().plan) == plan_id_for(plain_decision().plan)
        assert plan_id_for(plain_decision().plan) != plan_id_for(decision_with_wall().plan)

    def test_walls_rejected_with_documented_error(self):
        with pytest.raises(WallAssignmentError, match="rejects walls"):
            to_mixed_assignment_v2(decision_with_wall())
        with pytest.raises(WallAssignmentError, match="rejects walls"):
            AssignmentPlanV2(assignments={"c0": "wall-v2"}, plan_id="p",
                             commit_horizon_s=1.0,
                             wall_members=frozenset({"c0"}))

    def test_default_configs_used_when_none_given(self):
        out = to_mixed_assignment_v2(plain_decision())
        assert out.commit_horizon_s == PlannerConfig().v2_commit_horizon_s
        assert to_dispatcher_v3(plain_decision()).commit_horizon_s == 0.0
        assert DISPATCH_V2_SCHEMA == "mixed_assignment/2"


@pytest.fixture(scope="module")
def real_v3_module():
    """The REAL final mixed_assignment/3 module, loaded read-only from
    the research-dense tree via importlib (its package imports resolve inside a
    synthetic package rooted there, so the worktree's v2 module is not
    touched). Skips with the precise reason if unavailable."""
    if not REAL_V3_SOURCE.is_file():
        pytest.skip(f"real mixed_assignment/3 source not found: {REAL_V3_SOURCE}")
    for stale in [k for k in sys.modules
                  if k == REAL_V3_PACKAGE or k.startswith(REAL_V3_PACKAGE + ".")]:
        del sys.modules[stale]
    try:
        spec = importlib.util.spec_from_file_location(
            REAL_V3_PACKAGE, REAL_V3_SOURCE,
            submodule_search_locations=[str(REAL_V3_SOURCE.parent)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[REAL_V3_PACKAGE] = module
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - depends on external tree
        pytest.skip(f"cannot import real mixed_assignment from "
                    f"{REAL_V3_SOURCE}: {exc!r}")
    version = getattr(module, "VERSION", None)
    if version != "mixed_assignment/3":
        pytest.skip(f"real module VERSION {version!r} != 'mixed_assignment/3' "
                    f"at {REAL_V3_SOURCE}")
    return module


class TestMixedAssignmentV3Real:
    @pytest.fixture(autouse=True)
    def _install_real_module(self, real_v3_module, monkeypatch):
        monkeypatch.setitem(sys.modules, "traffic_shepherd.mixed_assignment",
                            real_v3_module)
        self.ma = real_v3_module

    def test_version_mismatch_raises_clear_error(self, monkeypatch):
        stub = types.ModuleType("traffic_shepherd.mixed_assignment")
        stub.VERSION = "mixed_assignment/2"
        monkeypatch.setitem(sys.modules, "traffic_shepherd.mixed_assignment", stub)
        with pytest.raises(WallAssignmentError,
                           match="mixed_assignment/3"):
            to_mixed_assignment_v3_real(plain_decision())

    def test_zero_walls(self):
        out = to_mixed_assignment_v3_real(plain_decision())
        assert isinstance(out, self.ma.AssignmentPlan)
        assert out.strict_roster is True
        assert out.commit_horizon_s == 0.0
        assert out.walls == ()
        assert out.assignments == {"c0": "native", "c1": "damping"}

    def test_one_hold_wall(self):
        out = to_mixed_assignment_v3_real(decision_with_wall())
        assert isinstance(out, self.ma.AssignmentPlan)
        assert len(out.walls) == 1
        real_wall = out.walls[0]
        assert isinstance(real_wall, self.ma.WallAssignment)
        assert real_wall.wall_id == "w0"
        assert real_wall.members == ("c0", "c1")  # lane-ordered, >= 2
        assert real_wall.wall_speed_mps == 16.0  # final field name
        assert real_wall.phase == "hold"
        assert real_wall.hold_s == 60.0 and real_wall.release_at_s is None
        assert out.assignments == {"c0": "wall-v2", "c1": "wall-v2",
                                   "c2": "damping"}

    def test_two_hold_walls(self):
        out = to_mixed_assignment_v3_real(decision_with_two_walls())
        assert [w.wall_id for w in out.walls] == ["w0", "w1"]
        assert {w.members for w in out.walls} == {("c0", "c1"), ("c2", "c3")}
        assert out.assignments["c4"] == "native"
        assert {c for c, mode in out.assignments.items()
                if mode == "wall-v2"} == {"c0", "c1", "c2", "c3"}

    def test_release_after_hold_is_omission_with_native_members(self):
        # after hold the planner releases by OMITTING the wall and assigning
        # its members native in the same plan; the real v3 plan carries no
        # wall for it and no release_at_s/hold_s term anywhere.
        released = PlannerDecision(
            plan=plan({"c0": "native", "c1": "native"}), reasons=())
        out = to_mixed_assignment_v3_real(released)
        assert out.walls == ()
        assert out.assignments == {"c0": "native", "c1": "native"}
