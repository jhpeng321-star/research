"""PL01-B: wall proposer, folding, lifecycle/lock, registry, multi-wall."""
from types import SimpleNamespace

import numpy as np
import pytest

from traffic_shepherd.planner import (
    Planner,
    PlannerConfig,
    WallRegistry,
    WallSpec,
    plan_key,
    propose_walls,
    t0_lane_position,
    wall_id_for,
)
from tests.planner._helpers import make_samples, make_world, plan, state_from_plan, wall
from traffic_shepherd.planner.paths import rd_wt_root
from traffic_shepherd.planner.walls import feasible_wall_members

# PL01-E2: EVERY candidate plan this module's Planner.decide produces must
# pass the kit's validate_plan (contract section 3.3). The kit lives in a
# read-only sibling workspace; tests skip precisely if it is absent.
KIT_ROOT = rd_wt_root() / "wd01b" / "output" / "wd01"
KIT_READY = KIT_ROOT.is_dir()
if KIT_READY:
    import sys as _sys
    if str(KIT_ROOT) not in _sys.path:
        _sys.path.insert(0, str(KIT_ROOT))
    try:
        from predict import contract as _kit
        from predict.validate import validate_plan as _kit_validate_plan
    except ImportError:
        KIT_READY = False


def _kit_entry(entry):
    """Planner (possibly partial) state dict -> strict kit ControllerStateEntry."""
    lifted = dict(entry)
    if lifted.get("locked") and not lifted.get("wall_id"):
        lifted["locked"] = False
    return _kit.ControllerStateEntry(
        action=lifted.get("action", "native"),
        wall_id=lifted.get("wall_id"),
        wall_phase=lifted.get("wall_phase"),
        time_in_phase_s=float(lifted.get("time_in_phase_s") or 0.0),
        wall_speed_mps=lifted.get("wall_speed_mps"),
        locked=bool(lifted.get("locked", False)),
    )


def _lifted_state(controller_state, candidate):
    """The controller state the candidate assumes, as strict kit entries:
    locked entries whose wall the candidate no longer carries are lifted to
    native/unlocked (the planner encodes a predicted safety abort by
    dissolving the wall, which presumes the dispatcher-side abort
    completed), mirroring the loop's own prediction-state sync."""
    walls = {w.wall_id for w in candidate.walls}
    out = {}
    for cav, entry in (controller_state or {}).items():
        entry = dict(entry)
        if entry.get("locked") and entry.get("wall_id") not in walls:
            # §3.2 hygiene: a native CAV carries no wall fields at all
            entry["locked"] = False
            entry["action"] = "native"
            entry["wall_id"] = None
            entry["wall_phase"] = None
            entry["wall_speed_mps"] = None
        out[cav] = _kit_entry(entry)
    return out


def assert_kit_valid_plan(candidate, controller_state):
    if not KIT_READY:
        pytest.skip(f"WD-01 kit not present at {KIT_ROOT}")
    plan_kit = _kit.PlanCandidate(
        plan_version=candidate.plan_version,
        per_cav=dict(candidate.per_cav),
        walls=tuple(
            _kit.WallSpec(
                wall_id=w.wall_id,
                members=tuple(w.members),
                speed_mps=w.speed_mps,
                phase=w.phase,
                release_at_s=w.release_at_s,
                hold_s=w.hold_s,
            )
            for w in candidate.walls
        ),
    )
    _kit_validate_plan(plan_kit, _lifted_state(controller_state, candidate))


class _KitValidatingPlanner(Planner):
    """Planner whose every decide() output is kit-validated."""

    def decide(self, state_samples, now_s, *, controller_state=None):
        decision = super().decide(state_samples, now_s,
                                  controller_state=controller_state)
        assert_kit_valid_plan(decision.plan, controller_state)
        return decision


Planner = _KitValidatingPlanner

CFG = PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0)


def lane_world(cavs, **config_over):
    """World with per-CAV (lane, s0); cavs: {id: (lane, s0)}."""
    world = make_world(tuple(cavs), part_a=10)
    for cav, (lane, s0) in cavs.items():
        world.lane_of[cav] = float(lane)
        world.s0_of[cav] = float(s0)
    return world


def two_lane_setup(k=4, **config_over):
    world = lane_world({"c0": (0.0, 100.0), "c1": (1.0, 102.0)})
    cfg = PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0, **config_over)
    return world, cfg, make_samples(k), wall_id_for(("c0", "c1"), 16.0)


def framed_samples(ids, t0=0.0):
    """Samples whose hybrid tracked layer observes exactly ``ids``."""
    tracked = {vid: (0.0, 0, 20.0) for vid in ids}
    out = []
    for i in range(4):
        out.append(SimpleNamespace(
            hybrid={"tracked": dict(tracked)},
            arrivals=(),
            meta={"sample_id": f"s{i}", "time_s": t0,
                  "merge_observed": True, "coverage": 1.0}))
    return out


NATIVE2 = {"c0": {"action": "native", "locked": False},
           "c1": {"action": "native", "locked": False}}


class TestProposer:
    def test_three_lanes_give_three_members(self):
        t0 = {"a": (0.0, 50.0), "b": (1.0, 51.0), "c": (2.0, 52.0)}
        candidates = propose_walls(t0, ["a", "b", "c"], CFG)
        assert candidates
        best = candidates[0]
        assert best.spec.members == ("a", "b", "c")  # lane-ordered
        assert best.spread_m == pytest.approx(2.0)

    def test_one_cav_per_lane_and_lane_order(self):
        t0 = {"c0": (1.0, 60.0), "c1": (0.0, 58.0), "c2": (1.0, 150.0)}
        candidates = propose_walls(t0, ["c0", "c1", "c2"], CFG)
        member_sets = {c.spec.members for c in candidates}
        assert ("c1", "c0") in member_sets  # lane 0 first
        assert ("c1", "c2") in member_sets  # spread 92 m <= gap 100 m
        assert not any("c0" in m and "c2" in m for m in member_sets)  # same lane

    def test_gap_limit_blocks_far_members(self):
        t0 = {"c0": (0.0, 0.0), "c1": (1.0, 150.0)}
        assert propose_walls(t0, ["c0", "c1"], CFG) == ()
        tight = PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0, wall_max_member_gap_m=200.0)
        assert propose_walls(t0, ["c0", "c1"], tight)

    def test_single_lane_no_proposal(self):
        t0 = {"c0": (0.0, 0.0), "c1": (0.0, 3.0)}
        assert propose_walls(t0, ["c0", "c1"], CFG) == ()

    def test_locked_cavs_excluded(self):
        # only free ids are passed in; locked members never appear
        t0 = {"c0": (0.0, 10.0), "c1": (1.0, 12.0), "c2": (0.0, 400.0)}
        candidates = propose_walls(t0, ["c0", "c1"], CFG)  # c2 locked elsewhere
        assert all(set(c.spec.members) <= {"c0", "c1"} for c in candidates)

    def test_speed_and_hold_grid(self):
        t0 = {"c0": (0.0, 0.0), "c1": (1.0, 1.0)}
        candidates = propose_walls(t0, ["c0", "c1"], CFG)
        speeds = {c.spec.speed_mps for c in candidates}
        assert speeds == {12.0, 16.0, 20.0, 24.0}
        for speed in speeds:
            holds = sorted(c.spec.hold_s for c in candidates
                           if c.spec.speed_mps == speed)
            # every release option encodes as hold_s; the None option is the
            # open hold (no automatic release)
            assert holds == [60.0, 120.0, 240.0, CFG.wall_open_hold_s]
        # v3 encoding: release_at_s is never emitted (never 0.0, never set)
        assert all(c.spec.release_at_s is None for c in candidates)
        # deterministic ordering, stable ids
        keys = [c.order_key for c in candidates]
        assert keys == sorted(keys)
        assert all(c.spec.wall_id == wall_id_for(c.spec.members, c.spec.speed_mps)
                   for c in candidates)

    def test_open_hold_option_and_config_validation(self):
        cfg = PlannerConfig()
        assert cfg.wall_open_hold_s == 3600.0
        world_cfg = PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0,
                                  wall_release_options_s=(None,),
                                  wall_open_hold_s=1800.0)
        t0 = {"c0": (0.0, 0.0), "c1": (1.0, 1.0)}
        candidates = propose_walls(t0, ["c0", "c1"], world_cfg)
        assert {c.spec.hold_s for c in candidates} == {1800.0}
        assert all(c.spec.release_at_s is None for c in candidates)
        # config validation: open hold positive, speeds strictly above 2.0
        with pytest.raises(ValueError, match="wall_open_hold_s"):
            PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0, wall_open_hold_s=0.0)
        with pytest.raises(ValueError, match="wall_speeds_mps"):
            PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0, wall_speeds_mps=(2.0, 16.0))
        with pytest.raises(ValueError, match="wall_speeds_mps"):
            PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0, wall_speeds_mps=(1.0,))


class TestWallAdoption:
    def test_wall_proposed_when_beneficial_and_registered(self):
        world, cfg, samples, wid = two_lane_setup()
        world.wall_delay[wid] = -8.0
        planner = Planner(world, cfg)
        decision = planner.decide(samples, 0.0, controller_state=NATIVE2)
        assert len(decision.plan.walls) == 1
        wall = decision.plan.walls[0]
        assert wall.wall_id == wid and wall.phase == "assemble"
        assert wall.hold_s == 60.0 and wall.release_at_s is None
        assert decision.plan.per_cav == {}  # members locked on dispatch
        record = planner.wall_registry.get(wid)
        assert record is not None
        assert record.phase == "assemble" and record.dispatched_at_s == 0.0
        assert record.assembled_at_s is None
        passed = [r for r in decision.reasons if "wall bar passed" in r.detail]
        assert passed

    def test_wall_not_proposed_without_benefit(self):
        world, cfg, samples, _ = two_lane_setup()
        planner = Planner(world, cfg)
        decision = planner.decide(samples, 0.0, controller_state=NATIVE2)
        assert decision.plan.walls == ()
        assert decision.plan.per_cav == {"c0": "native", "c1": "native"}

    def test_higher_bar_blocks_marginal_wall(self):
        # saving [1.0]*5 + [-0.01]: passes the normal bar, fails wall_agreement
        world, cfg, samples, wid = two_lane_setup(k=6)
        world.wall_delay[wid] = np.array([1.0] * 5 + [-0.01])
        planner = Planner(world, cfg)
        decision = planner.decide(samples, 0.0, controller_state=NATIVE2)
        assert decision.plan.walls == ()
        failed = [r for r in decision.reasons if "wall bar failed" in r.detail]
        assert failed

    def test_higher_bar_margin_k_blocks_small_saving(self):
        # saving [2]*9 + [-2]: median 2, std 1.2 -> passes wall_margin_k = 1
        # (margin 1.2) but fails wall_margin_k = 2 (margin 2.4)
        world, cfg, samples, wid = two_lane_setup(k=10)
        world.wall_delay[wid] = -np.array([2.0] * 9 + [-2.0])
        lenient = Planner(world, PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0, wall_margin_k=1.0))
        assert lenient.decide(samples, 0.0, controller_state=NATIVE2).plan.walls != ()
        strict = Planner(world, PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0, wall_margin_k=2.0))
        decision = strict.decide(samples, 0.0, controller_state=NATIVE2)
        assert decision.plan.walls == ()


class TestAssemblyLock:
    def test_no_benefit_release_during_assemble(self):
        world = lane_world({"c0": (0.0, 100.0), "c1": (1.0, 102.0),
                            "c2": (0.0, 400.0)})
        cfg = PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0)
        wid = wall_id_for(("c0", "c1"), 16.0)
        world.wall_delay[wid] = -5.0  # beneficial at dispatch time
        world.action_delay["c2"] = {"damping": -50.0}
        planner = Planner(world, cfg)
        planner.decide(make_samples(4), 0.0, controller_state={
            "c0": {"action": "native", "locked": False},
            "c1": {"action": "native", "locked": False},
            "c2": {"action": "native", "locked": False}})
        assert len(planner._current_plan.walls) == 1  # dispatched
        world.wall_delay[wid] = 5.0  # the wall now hurts: still locked
        state = {"c0": {"action": "wall", "wall_id": wid,
                        "wall_phase": "assemble", "locked": True,
                        "time_in_phase_s": 1.0},
                 "c1": {"action": "wall", "wall_id": wid,
                        "wall_phase": "assemble", "locked": True,
                        "time_in_phase_s": 1.0},
                 "c2": {"action": "damping", "locked": False}}
        decision = planner.decide(make_samples(4), 1.0, controller_state=state)
        assert len(decision.plan.walls) == 1
        assert decision.plan.walls[0].wall_id == wid
        assert decision.plan.walls[0].phase == "assemble"  # NOT released
        assert decision.plan.per_cav == {"c2": "damping"}
        assert planner.wall_registry.get(wid).phase == "assemble"

    def test_step5_fallback_never_drops_assembling_wall_except_safety(self):
        world = lane_world({"c0": (0.0, 100.0), "c1": (1.0, 102.0),
                            "c2": (0.0, 400.0)})
        cfg = PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0)
        wid = wall_id_for(("c0", "c1"), 16.0)
        world.wall_delay[wid] = -5.0
        world.action_delay["c2"] = {"damping": -3.0}
        combined = plan({"c2": "damping"}, walls=[
            WallSpec(wall_id=wid, members=("c0", "c1"), speed_mps=16.0,
                     phase="assemble", hold_s=60.0)])
        world.safety_script[plan_key(combined)] = {"collision": True}
        planner = Planner(world, cfg)
        decision = planner.decide(make_samples(4), 0.0, controller_state={
            "c0": {"action": "native", "locked": False},
            "c1": {"action": "native", "locked": False},
            "c2": {"action": "native", "locked": False}})
        # the c2 action change was dropped, the assembling wall was kept
        assert decision.plan.per_cav == {"c2": "native"}
        assert len(decision.plan.walls) == 1
        drops = [r for r in decision.reasons if r.action == "drop"]
        assert drops and drops[0].subject == "c2"
        assert not any("safety abort" in (r.detail or "") and r.action == "abort"
                       for r in decision.reasons)

    def test_step5_safety_pass_aborts_wall_when_nothing_else_helps(self):
        # two walls, each safe alone, combined unsafe and NOT fixable by
        # dropping any action change: the fallback's final safety pass aborts
        # the lower-saving wall (the only allowed non-safety-independent drop)
        world = lane_world({"c0": (0.0, 100.0), "c1": (1.0, 102.0),
                            "c2": (0.0, 300.0), "c3": (1.0, 302.0)})
        cfg = PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0, wall_release_options_s=(60.0,))
        wid_a = wall_id_for(("c0", "c1"), 16.0)
        wid_b = wall_id_for(("c2", "c3"), 16.0)
        world.wall_delay[wid_a] = -5.0
        world.wall_delay[wid_b] = -2.0
        spec_a = WallSpec(wall_id=wid_a, members=("c0", "c1"), speed_mps=16.0,
                          phase="assemble", hold_s=60.0)
        spec_b = WallSpec(wall_id=wid_b, members=("c2", "c3"), speed_mps=16.0,
                          phase="assemble", hold_s=60.0)
        world.safety_script[plan_key(plan({}, walls=[spec_a, spec_b]))] = {
            "collision": True}
        planner = Planner(world, cfg)
        decision = planner.decide(make_samples(4), 0.0, controller_state={
            cav: {"action": "native", "locked": False}
            for cav in ("c0", "c1", "c2", "c3")})
        assert [w.wall_id for w in decision.plan.walls] == [wid_a]
        aborts = [r for r in decision.reasons
                  if r.action == "abort" and r.subject == wid_b]
        assert aborts and "safety" in aborts[0].detail


class TestSafetyAbort:
    @pytest.mark.parametrize("signal", ["plan_unsafe", "wall_signal"])
    def test_assembly_ends_only_by_safety(self, signal):
        world, cfg, samples, wid = two_lane_setup()
        planner = Planner(world, cfg)
        world.wall_delay[wid] = -5.0
        planner.decide(samples, 0.0, controller_state=NATIVE2)
        assert planner.wall_registry.get(wid) is not None
        walled = plan({}, walls=[WallSpec(wall_id=wid, members=("c0", "c1"),
                                          speed_mps=16.0, phase="assemble",
                                          hold_s=60.0)])
        if signal == "plan_unsafe":
            world.safety_script[plan_key(walled)] = {"collision": True}
        else:
            world.wall_script[wid] = {"safety_abort": True}
        state = state_from_plan(walled, phase_time=1.0)
        decision = planner.decide(samples, 1.0, controller_state=state)
        assert decision.plan.walls == ()
        assert decision.plan.per_cav == {"c0": "native", "c1": "native"}
        assert planner.wall_registry.get(wid) is None
        aborts = [r for r in decision.reasons if r.action == "abort"]
        assert aborts and aborts[0].subject == wid


class TestHoldReleaseLifecycle:
    def _to_hold(self, world, cfg, samples, wid, planner, delay_after=0.0):
        """Dispatch at t=0 (beneficial); assemble by t=1 (time 10 >= duration).

        ``delay_after`` is the wall's delay effect from t=1 on (the world
        after the wall is up).
        """
        world.wall_delay[wid] = -5.0
        planner.decide(samples, 0.0, controller_state=NATIVE2)
        world.wall_delay[wid] = delay_after
        state = {"c0": {"action": "wall", "wall_id": wid,
                        "wall_phase": "assemble", "locked": True,
                        "time_in_phase_s": 10.0},
                 "c1": {"action": "wall", "wall_id": wid,
                        "wall_phase": "assemble", "locked": True,
                        "time_in_phase_s": 10.0}}
        return planner.decide(samples, 1.0, controller_state=state), state

    def test_assembled_event_flips_to_hold(self):
        world, cfg, samples, wid = two_lane_setup()
        planner = Planner(world, cfg)
        decision, _ = self._to_hold(world, cfg, samples, wid, planner)
        assert decision.plan.walls[0].phase == "hold"
        record = planner.wall_registry.get(wid)
        assert record.phase == "hold" and record.assembled_at_s == 1.0

    def test_release_keeps_wall_as_release_now_with_members(self):
        world, cfg, samples, wid = two_lane_setup()
        planner = Planner(world, cfg)
        self._to_hold(world, cfg, samples, wid, planner)
        world.wall_delay[wid] = 5.0  # hold now costs 5 vs release (native)
        state = {"c0": {"action": "wall", "wall_id": wid,
                        "wall_phase": "hold", "locked": True,
                        "time_in_phase_s": 11.0},
                 "c1": {"action": "wall", "wall_id": wid,
                        "wall_phase": "hold", "locked": True,
                        "time_in_phase_s": 11.0}}
        decision = planner.decide(samples, 2.0, controller_state=state)
        # release encoding toward predict() (§3.3/§5): the wall STAYS in the
        # candidate as phase="release", release_at_s=0.0 (release now, offset
        # from t0) with its locked members -- they may not move into per_cav.
        assert decision.plan.walls[0].phase == "release"
        assert decision.plan.walls[0].release_at_s == 0.0
        assert decision.plan.walls[0].hold_s is None
        assert decision.plan.walls[0].members == ("c0", "c1")
        assert decision.plan.per_cav == {}
        assert planner.wall_registry.get(wid).phase == "release"
        released = [r for r in decision.reasons
                    if r.subject == wid and r.action == "release"]
        assert released and released[0].median_saving_s == pytest.approx(5.0)
        # next tick: the dispatcher reports the handback done (members
        # unlocked) -> the wall dissolves; members return to per_cav native.
        decision = planner.decide(samples, 3.0, controller_state=NATIVE2)
        assert decision.plan.walls == ()
        assert decision.plan.per_cav == {"c0": "native", "c1": "native"}
        assert planner.wall_registry.get(wid) is None

    def test_release_by_worse_than_native_ignores_margin(self):
        # saving [10,10,10,-40]: median 10 > 0 (N1) although the switch rule's
        # margin (std 15) blocks it
        world, cfg, samples, wid = two_lane_setup()
        planner = Planner(world, cfg)
        self._to_hold(world, cfg, samples, wid, planner)
        world.wall_delay[wid] = np.array([10.0, 10.0, 10.0, -40.0])
        state = {"c0": {"action": "wall", "wall_id": wid,
                        "wall_phase": "hold", "locked": True,
                        "time_in_phase_s": 11.0},
                 "c1": {"action": "wall", "wall_id": wid,
                        "wall_phase": "hold", "locked": True,
                        "time_in_phase_s": 11.0}}
        decision = planner.decide(samples, 2.0, controller_state=state)
        # release encoding: the wall stays as phase="release" with its
        # locked members; the registry holds it until the handback
        assert decision.plan.walls[0].phase == "release"
        assert decision.plan.per_cav == {}
        released = [r for r in decision.reasons
                    if r.subject == wid and r.action == "release"]
        assert released and "N1" in released[0].detail
        assert planner.wall_registry.get(wid).phase == "release"

    def test_hold_kept_without_benefit(self):
        world, cfg, samples, wid = two_lane_setup()
        planner = Planner(world, cfg)
        decision, _ = self._to_hold(world, cfg, samples, wid, planner)
        assert decision.plan.walls[0].phase == "hold"
        keeps = [r for r in decision.reasons
                 if r.subject == wid and r.action == "keep"]
        assert keeps and keeps[0].median_saving_s == pytest.approx(0.0)

    def test_controller_state_can_dissolve_wall(self):
        world, cfg, samples, wid = two_lane_setup()
        planner = Planner(world, cfg)
        self._to_hold(world, cfg, samples, wid, planner)
        # the dispatcher reports the wall gone (members unlocked)
        decision = planner.decide(samples, 2.0, controller_state=NATIVE2)
        assert decision.plan.walls == ()
        assert planner.wall_registry.get(wid) is None


class TestFolding:
    def test_affected_cav_folds_into_wall_group(self):
        world = lane_world({"c0": (0.0, 100.0), "c1": (1.0, 102.0),
                            "c2": (0.0, 400.0)})
        cfg = PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0)
        wid = wall_id_for(("c0", "c1"), 16.0)
        world.wall_delay[wid] = -5.0
        world.wall_coupling_pos[(wid, "c2")] = 3.0  # c2 measured-affected
        planner = Planner(world, cfg)
        decision = planner.decide(make_samples(4), 0.0, controller_state={
            **{c: {"action": "native", "locked": False}
               for c in ("c0", "c1", "c2")}})
        assert len(decision.plan.walls) == 1
        folds = [r for r in decision.reasons if "folded: c2" in r.detail]
        assert folds

    def test_cap_drops_weakest_folded_first(self):
        world = lane_world({"c0": (0.0, 100.0), "c1": (1.0, 102.0),
                            "c2": (0.0, 400.0), "c3": (1.0, 420.0)})
        cfg = PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0, group_cap=3)
        wid = wall_id_for(("c0", "c1"), 16.0)
        world.wall_delay[wid] = -5.0
        world.wall_coupling_pos[(wid, "c2")] = 6.0   # stronger interaction
        world.wall_coupling_pos[(wid, "c3")] = 2.5   # weaker interaction
        planner = Planner(world, cfg)
        decision = planner.decide(make_samples(4), 0.0, controller_state={
            **{c: {"action": "native", "locked": False}
               for c in ("c0", "c1", "c2", "c3")}})
        folds = [r for r in decision.reasons if "dropped weakest: c3" in r.detail]
        assert folds and "folded: c2" in folds[0].detail

    def test_members_alone_above_cap_infeasible(self):
        world = lane_world({"a": (0.0, 0.0), "b": (1.0, 1.0), "c": (2.0, 2.0)})
        cfg = PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0, group_cap=2)
        candidates = propose_walls(t0_lane_position(
            _prediction_for(world, cfg), ["a", "b", "c"]), ["a", "b", "c"], cfg)
        assert candidates  # 3-member cross-section exists...
        assert all(len(c.spec.members) > cfg.group_cap for c in candidates)


def _prediction_for(world, cfg):
    samples = make_samples(1)
    everyone = plan({cav: "native" for cav in world.cav_ids})
    return world.predict_many(samples, [everyone], trajectory_s=10)[0]


class TestMultiWall:
    def _four_cav_world(self, cfg):
        world = lane_world({"c0": (0.0, 100.0), "c1": (1.0, 102.0),
                            "c2": (0.0, 300.0), "c3": (1.0, 302.0)})
        wid_a = wall_id_for(("c0", "c1"), 16.0)
        wid_b = wall_id_for(("c2", "c3"), 16.0)
        world.wall_delay[wid_a] = -5.0
        world.wall_delay[wid_b] = -2.0
        state = {cav: {"action": "native", "locked": False}
                 for cav in ("c0", "c1", "c2", "c3")}
        return world, wid_a, wid_b, state

    def test_two_disjoint_walls_chosen_together(self):
        world, wid_a, wid_b, state = self._four_cav_world(PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0))
        planner = Planner(world, PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0))
        decision = planner.decide(make_samples(4), 0.0, controller_state=state)
        assert {w.wall_id for w in decision.plan.walls} == {wid_a, wid_b}
        assert decision.plan.per_cav == {}
        assert set(planner.wall_registry.ids()) == {wid_a, wid_b}

    def test_max_walls_one_keeps_higher_saving(self):
        cfg = PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0, max_walls=1)
        world, wid_a, wid_b, state = self._four_cav_world(cfg)
        planner = Planner(world, cfg)
        decision = planner.decide(make_samples(4), 0.0, controller_state=state)
        assert [w.wall_id for w in decision.plan.walls] == [wid_a]
        vetoes = [r for r in decision.reasons if r.action == "veto"
                  and r.subject == wid_b]
        assert vetoes and "max_walls" in vetoes[0].candidate

    def test_two_lanes_only_cross_sections_are_feasible(self):
        world, wid_a, wid_b, state = self._four_cav_world(PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0))
        samples = make_samples(1)
        current = world.predict_many(
            samples, [plan({c: "native" for c in world.cav_ids})],
            trajectory_s=10)[0]
        candidates = propose_walls(t0_lane_position(current, world.cav_ids),
                                   world.cav_ids, CFG)
        member_sets = {c.spec.members for c in candidates}
        assert ("c0", "c1") in member_sets and ("c2", "c3") in member_sets
        assert ("c0", "c3") not in member_sets  # spread 202 m > 100 m
        assert ("c1", "c2") not in member_sets


class TestRegistry:
    def test_sync_from_plan_adopts_constructor_walls(self):
        wall = WallSpec(wall_id="w0", members=("c1",), speed_mps=16.0,
                        phase="hold", hold_s=60.0)
        registry = WallRegistry()
        registry.sync_from_plan(plan({"c0": "native"}, walls=[wall]))
        assert registry.get("w0").phase == "hold"
        assert registry.get("w0").dispatched_at_s is None

    def test_controller_state_phase_sync_and_dissolution(self):
        registry = WallRegistry()
        registry.register_spec(WallSpec(wall_id="w0", members=("c0", "c1"),
                                        speed_mps=16.0, phase="assemble",
                                        hold_s=60.0), dispatched_at_s=0.0)
        state = {"c0": {"action": "wall", "wall_id": "w0",
                        "wall_phase": "hold", "locked": True,
                        "time_in_phase_s": 12.0},
                 "c1": {"action": "wall", "wall_id": "w0",
                        "wall_phase": "hold", "locked": True,
                        "time_in_phase_s": 12.0}}
        registry.sync_from_controller_state(state)
        assert registry.get("w0").phase == "hold"
        registry.sync_from_controller_state(
            {"c0": {"action": "native", "locked": False},
             "c1": {"action": "native", "locked": False}})
        assert registry.get("w0") is None

    def test_telemetry_wall_phase_vocabulary_mapping(self):
        """Dispatcher telemetry reports assembly|hold|handback; the registry
        stores assemble|hold|release. Both vocabularies are accepted."""
        def record_phase(telemetry_phase):
            registry = WallRegistry()
            registry.register_spec(
                WallSpec(wall_id="w0", members=("c0", "c1"), speed_mps=16.0,
                         phase="assemble", hold_s=60.0), dispatched_at_s=0.0)
            state = {"c0": {"action": "wall", "wall_id": "w0",
                            "wall_phase": telemetry_phase, "locked": True,
                            "time_in_phase_s": 1.0},
                     "c1": {"action": "wall", "wall_id": "w0",
                            "wall_phase": telemetry_phase, "locked": True,
                            "time_in_phase_s": 1.0}}
            registry.sync_from_controller_state(state)
            return registry.get("w0").phase

        # dispatcher vocabulary (mixed_assignment/3 telemetry)
        assert record_phase("assembly") == "assemble"
        assert record_phase("hold") == "hold"
        assert record_phase("handback") == "release"
        # planner vocabulary still accepted (both vocabularies)
        assert record_phase("assemble") == "assemble"
        assert record_phase("release") == "release"

    def test_cells_rule(self):
        from traffic_shepherd.planner import cells_rule
        import numpy as np
        assert cells_rule(np.array([False, True]), "any")
        assert not cells_rule(np.array([False, True]), "majority")
        assert cells_rule(np.array([True, True, True, False]), "majority")
        with pytest.raises(ValueError):
            cells_rule(np.array([True]), "all")


class TestWallDeterminism:
    def test_same_inputs_same_decision(self):
        def run():
            world, cfg, samples, wid = two_lane_setup()
            world.wall_delay[wid] = -8.0
            world.wall_coupling_pos[(wid, "c1")] = 3.0
            planner = Planner(world, cfg)
            return planner.decide(samples, 0.0, controller_state=NATIVE2)

        assert run() == run()


class TestReleaseRule:
    """PL01-B3: the named hold-vs-release rules (PlannerConfig).

    ``wall_release_margin_s``: releases only when the median saving exceeds
    the standard switch rule's margin/agreement bar; unsafe holds always
    force-release regardless of rule.
    """

    def _held(self, wall_delay, **config_over):
        """Dispatch at t=0, assemble by t=1; returns (world, cfg, planner,
        samples, wid, hold_state) with the REAL wall id in the state."""
        world, _, samples, wid = two_lane_setup()
        cfg = PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0, **config_over)
        planner = Planner(world, cfg)
        world.wall_delay[wid] = -5.0
        planner.decide(samples, 0.0, controller_state=NATIVE2)
        assemble = {"c0": {"action": "wall", "wall_id": wid,
                           "wall_phase": "assemble", "locked": True,
                           "time_in_phase_s": 10.0},
                    "c1": {"action": "wall", "wall_id": wid,
                           "wall_phase": "assemble", "locked": True,
                           "time_in_phase_s": 10.0}}
        planner.decide(samples, 1.0, controller_state=assemble)
        world.wall_delay[wid] = wall_delay
        hold = {"c0": {"action": "wall", "wall_id": wid,
                       "wall_phase": "hold", "locked": True,
                       "time_in_phase_s": 11.0},
                "c1": {"action": "wall", "wall_id": wid,
                       "wall_phase": "hold", "locked": True,
                       "time_in_phase_s": 11.0}}
        return world, cfg, planner, samples, wid, hold

    def _release_reasons(self, decision, wid):
        return [r for r in decision.reasons
                if r.subject == wid and r.action == "release"]

    def test_default_rule_releases_small_positive_saving(self):
        world, cfg, planner, samples, wid, hold = self._held(
            wall_delay=2.0, wall_release_rule="median_saving_gt0")
        decision = planner.decide(samples, 2.0, controller_state=hold)
        assert decision.plan.walls[0].phase == "release"
        assert decision.plan.walls[0].release_at_s == 0.0
        assert decision.plan.walls[0].members == ("c0", "c1")
        assert decision.plan.per_cav == {}  # members stay locked to the wall
        released = self._release_reasons(decision, wid)
        assert released and released[0].median_saving_s == pytest.approx(2.0)
        assert "median_saving_gt0 fired" in released[0].detail

    def test_release_margin_blocks_release_below_it(self):
        world, cfg, planner, samples, wid, hold = self._held(
            wall_delay=2.0, wall_release_margin_s=5.0)
        decision = planner.decide(samples, 2.0, controller_state=hold)
        assert decision.plan.walls and decision.plan.walls[0].phase == "hold"
        assert decision.plan.per_cav == {}
        released = self._release_reasons(decision, wid)
        assert not released
        keeps = [r for r in decision.reasons
                 if r.subject == wid and r.action == "keep"]
        assert keeps and "did not fire" in keeps[0].detail
        assert "median_saving_gt0" in keeps[0].detail

    def test_switch_rule_blocks_release_when_margin_or_agreement_fail(self):
        # saving [10, 10, 10, -40]: median 10 > 0 (the default rule would
        # release) but std ~21.65 puts the switch margin above 10 and the
        # agreement (3/4) below 0.80
        world, cfg, planner, samples, wid, hold = self._held(
            wall_delay=np.array([10.0, 10.0, 10.0, -40.0]),
            wall_release_rule="switch_rule")
        decision = planner.decide(samples, 2.0, controller_state=hold)
        assert decision.plan.walls and decision.plan.walls[0].phase == "hold"
        released = self._release_reasons(decision, wid)
        assert not released
        keeps = [r for r in decision.reasons
                 if r.subject == wid and r.action == "keep"]
        assert keeps and "switch_rule did not fire" in keeps[0].detail

    def test_switch_rule_releases_when_it_passes(self):
        # saving [10, 10, 10, 10]: median 10 > margin 0, agreement 4/4
        world, cfg, planner, samples, wid, hold = self._held(
            wall_delay=np.array([10.0, 10.0, 10.0, 10.0]),
            wall_release_rule="switch_rule")
        decision = planner.decide(samples, 2.0, controller_state=hold)
        assert decision.plan.walls[0].phase == "release"
        assert decision.plan.walls[0].release_at_s == 0.0
        assert decision.plan.per_cav == {}  # members stay locked to the wall
        released = self._release_reasons(decision, wid)
        assert released and "switch_rule fired" in released[0].detail

    @pytest.mark.parametrize("rule", ["median_saving_gt0", "switch_rule"])
    def test_unsafe_hold_force_releases_under_both_rules(self, rule):
        world, _, planner, samples, wid, hold = self._held(
            wall_delay=5.0, wall_release_rule=rule)
        hold_plan = plan({}, walls=[wall(wid, ("c0", "c1"), speed=16.0,
                                         phase="hold", hold_s=60.0)])
        world.safety_script[plan_key(hold_plan)] = {"collision": True}
        decision = planner.decide(samples, 2.0, controller_state=hold)
        # force-release under both rules, with the predict() release
        # encoding: the wall is recast as phase="release" (release_at_s=0.0)
        # and stays in the plan with its locked members
        assert decision.plan.walls[0].phase == "release"
        assert decision.plan.walls[0].release_at_s == 0.0
        assert decision.plan.walls[0].members == ("c0", "c1")
        assert decision.plan.per_cav == {}
        assert planner.wall_registry.get(wid).phase == "release"
        released = self._release_reasons(decision, wid)
        assert released and "unsafe" in released[0].detail


# --- PL01-E3: exact-boundary kit validation ---------------------------------

def _kit_plan(candidate):
    """Planner PlanCandidate -> strict kit PlanCandidate (same fields)."""
    return _kit.PlanCandidate(
        plan_version=candidate.plan_version,
        per_cav=dict(candidate.per_cav),
        walls=tuple(
            _kit.WallSpec(
                wall_id=w.wall_id,
                members=tuple(w.members),
                speed_mps=w.speed_mps,
                phase=w.phase,
                release_at_s=w.release_at_s,
                hold_s=w.hold_s,
            )
            for w in candidate.walls
        ),
    )


def assert_kit_valid_plan_exact(candidate, controller_state):
    """Kit validate_plan against the controller_state EXACTLY as the loop
    passed it to this predict_many call -- only contract-defaulted entry
    fields are filled in; nothing is lifted (a post-abort / post-release
    state passed by the loop is what gets validated)."""
    if not KIT_READY:
        pytest.skip(f"WD-01 kit not present at {KIT_ROOT}")
    entries = {cav: _kit_entry(entry)
               for cav, entry in (controller_state or {}).items()}
    _kit_validate_plan(_kit_plan(candidate), entries)


class KitValidatingWorld:
    """PL01-E3: test wrapper around a world model. Validates EVERY plan of
    every predict_many call against the kit's validate_plan, using the EXACT
    controller_state argument of that same call (no lifting), then
    delegates to the wrapped model."""

    def __init__(self, inner):
        self._inner = inner

    def predict_many(self, state_samples, plans, controller_state=None, *,
                     horizon_s=600.0, trajectory_s=None):
        for candidate in plans:
            assert_kit_valid_plan_exact(candidate, controller_state)
        return self._inner.predict_many(
            state_samples, plans, controller_state=controller_state,
            horizon_s=horizon_s, trajectory_s=trajectory_s)

    def predict(self, state_samples, candidate, controller_state=None, **kw):
        return self.predict_many(state_samples, [candidate],
                                 controller_state, **kw)[0]


class TestPredictBoundaryKitValidation:
    """The wall scenarios' decide() calls run on a wrapped world: every plan
    the loop hands to predict_many must pass the kit's validate_plan against
    that call's own controller_state (exact, no lifting)."""

    HELD = {"c0": {"action": "wall", "wall_id": None,
                   "wall_phase": "hold", "locked": True,
                   "time_in_phase_s": 11.0},
            "c1": {"action": "wall", "wall_id": None,
                   "wall_phase": "hold", "locked": True,
                   "time_in_phase_s": 11.0}}

    def _drive_dispatch_to_hold_to_release(self):
        world = two_lane_setup()[0]
        samples = make_samples(4)
        wid = wall_id_for(("c0", "c1"), 16.0)
        wrapped = KitValidatingWorld(world)
        planner = Planner(wrapped, PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0))
        native = {"c0": {"action": "native", "locked": False},
                  "c1": {"action": "native", "locked": False}}
        decisions = []
        world.wall_delay[wid] = -5.0
        decisions.append(planner.decide(samples, 0.0,
                                        controller_state=native))
        assemble = {"c0": {"action": "wall", "wall_id": wid,
                           "wall_phase": "assemble", "locked": True,
                           "time_in_phase_s": 10.0},
                    "c1": {"action": "wall", "wall_id": wid,
                           "wall_phase": "assemble", "locked": True,
                           "time_in_phase_s": 10.0}}
        decisions.append(planner.decide(samples, 1.0,
                                        controller_state=assemble))
        world.wall_delay[wid] = 5.0  # hold now costs -> release fires
        hold = {"c0": {"action": "wall", "wall_id": wid,
                       "wall_phase": "hold", "locked": True,
                       "time_in_phase_s": 11.0},
                "c1": {"action": "wall", "wall_id": wid,
                       "wall_phase": "hold", "locked": True,
                       "time_in_phase_s": 11.0}}
        decisions.append(planner.decide(samples, 2.0, controller_state=hold))
        # handback done: members unlocked -> the wall dissolves
        decisions.append(planner.decide(samples, 3.0,
                                        controller_state=native))
        return decisions

    def test_dispatch_assemble_release_handback_all_valid(self):
        decisions = self._drive_dispatch_to_hold_to_release()
        assert decisions[0].plan.walls[0].phase == "assemble"
        assert decisions[1].plan.walls[0].phase == "hold"
        # release keeps the wall in the candidate (phase release, members)
        assert decisions[2].plan.walls[0].phase == "release"
        assert decisions[2].plan.per_cav == {}
        # after handback the wall is gone and members are native
        assert decisions[3].plan.walls == ()
        assert decisions[3].plan.per_cav == {"c0": "native", "c1": "native"}

    def test_unsafe_hold_force_release_is_boundary_valid(self):
        world = two_lane_setup()[0]
        samples = make_samples(4)
        wid = wall_id_for(("c0", "c1"), 16.0)
        wrapped = KitValidatingWorld(world)
        planner = Planner(wrapped, PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0))
        native = {"c0": {"action": "native", "locked": False},
                  "c1": {"action": "native", "locked": False}}
        world.wall_delay[wid] = -5.0
        planner.decide(samples, 0.0, controller_state=native)
        assemble = {"c0": {"action": "wall", "wall_id": wid,
                           "wall_phase": "assemble", "locked": True,
                           "time_in_phase_s": 10.0},
                    "c1": {"action": "wall", "wall_id": wid,
                           "wall_phase": "assemble", "locked": True,
                           "time_in_phase_s": 10.0}}
        planner.decide(samples, 1.0, controller_state=assemble)
        hold_plan = plan({}, walls=[wall(wid, ("c0", "c1"), speed=16.0,
                                         phase="hold", hold_s=60.0)])
        world.safety_script[plan_key(hold_plan)] = {"collision": True}
        world.wall_delay[wid] = 5.0
        hold = {"c0": {"action": "wall", "wall_id": wid,
                       "wall_phase": "hold", "locked": True,
                       "time_in_phase_s": 11.0},
                "c1": {"action": "wall", "wall_id": wid,
                       "wall_phase": "hold", "locked": True,
                       "time_in_phase_s": 11.0}}
        decision = planner.decide(samples, 2.0, controller_state=hold)
        # force-release keeps the wall as phase="release" with its members
        assert decision.plan.walls[0].phase == "release"
        assert decision.plan.per_cav == {}

    def test_predicted_assembly_abort_boundary_valid(self):
        # a predicted-abort candidate dissolves the wall; the loop passes
        # predict a POST-ABORT controller_state (members unlocked) and that
        # exact state is what the kit validates here.
        world = two_lane_setup()[0]
        samples = make_samples(4)
        wid = wall_id_for(("c0", "c1"), 16.0)
        wrapped = KitValidatingWorld(world)
        planner = Planner(wrapped, PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0))
        native = {"c0": {"action": "native", "locked": False},
                  "c1": {"action": "native", "locked": False}}
        world.wall_delay[wid] = -5.0
        planner.decide(samples, 0.0, controller_state=native)
        assemble = {"c0": {"action": "wall", "wall_id": wid,
                           "wall_phase": "assemble", "locked": True,
                           "time_in_phase_s": 1.0},
                    "c1": {"action": "wall", "wall_id": wid,
                           "wall_phase": "assemble", "locked": True,
                           "time_in_phase_s": 1.0}}
        world.wall_script[wid] = {"safety_abort": True}
        decision = planner.decide(samples, 1.0, controller_state=assemble)
        assert decision.plan.walls == ()  # aborted: members native
        assert decision.plan.per_cav == {"c0": "native", "c1": "native"}
        assert planner.wall_registry.get(wid) is None


class TestMemberLivenessFrameFilter:
    """PL01-E4: propose only LIVE members with a FRESH frame at decision
    time (the dispatcher otherwise aborts with member_frame_missing)."""

    def _framed_samples(self, ids, t0=0.0, frames=None):
        out = []
        for i in range(4):
            sample = SimpleNamespace(
                hybrid={"tracked": {vid: (0.0, 0, 20.0) for vid in ids}},
                arrivals=(),
                meta={"sample_id": f"s{i}", "time_s": t0,
                      "merge_observed": True, "coverage": 1.0})
            if frames is not None:
                sample.frames = dict(frames)
            out.append(sample)
        return out

    def test_not_yet_inserted_cav_excluded(self):
        cfg = PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0)
        state = {"c0": {"action": "native", "locked": False},
                 "c1": {"action": "native", "locked": False},
                 "c2": {"action": "native", "locked": False}}
        samples = self._framed_samples(("c0", "c1"))  # c2 has no frame yet
        eligible, reasons = feasible_wall_members(
            ("c0", "c1", "c2"), state, samples, cfg, t0_s=0.0)
        assert eligible == ("c0", "c1")
        assert any("c2" in r and "no observed frame" in r for r in reasons)

    def test_stale_frame_excluded(self):
        cfg = PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0,
                            wall_member_max_frame_age_s=0.0)
        state = {"c0": {"action": "native", "locked": False},
                 "c1": {"action": "native", "locked": False}}
        samples = self._framed_samples(("c0", "c1"),
                                       frames={"c0": 60.0, "c1": 57.5})
        eligible, reasons = feasible_wall_members(
            ("c0", "c1"), state, samples, cfg, t0_s=60.0)
        assert eligible == ("c0",)
        assert any("c1" in r and "stale frame" in r and "2.5" in r
                   for r in reasons)

    def test_positive_max_age_admits_recent_frame(self):
        cfg = PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0,
                            wall_member_max_frame_age_s=5.0)
        state = {"c0": {"action": "native", "locked": False},
                 "c1": {"action": "native", "locked": False}}
        samples = self._framed_samples(("c0", "c1"),
                                       frames={"c0": 60.0, "c1": 57.5})
        eligible, reasons = feasible_wall_members(
            ("c0", "c1"), state, samples, cfg, t0_s=60.0)
        assert eligible == ("c0", "c1")  # age 2.5 s <= 5 s
        assert not reasons

    def test_all_fresh_pair_proposed_end_to_end(self):
        world = lane_world({"c0": (0.0, 100.0), "c1": (1.0, 102.0)})
        cfg = PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0)
        samples = self._framed_samples(("c0", "c1"), t0=0.0)
        eligible, excluded = feasible_wall_members(
            ("c0", "c1"), {"c0": {"action": "native", "locked": False},
                           "c1": {"action": "native", "locked": False}},
            samples, cfg, t0_s=0.0)
        assert eligible == ("c0", "c1") and not excluded
        world.wall_delay[wall_id_for(("c0", "c1"), 16.0)] = -5.0
        planner = Planner(world, cfg)
        decision = planner.decide(samples, 0.0, controller_state={
            "c0": {"action": "native", "locked": False},
            "c1": {"action": "native", "locked": False}})
        assert len(decision.plan.walls) == 1  # the all-fresh pair proposed

    def test_stale_member_block_proposal_with_named_reason(self):
        world = lane_world({"c0": (0.0, 100.0), "c1": (1.0, 102.0)})
        cfg = PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0,
                            wall_member_max_frame_age_s=0.0)
        samples = self._framed_samples(("c0", "c1"), t0=60.0,
                                       frames={"c0": 60.0, "c1": 55.0})
        planner = Planner(world, cfg)
        state = {"c0": {"action": "native", "locked": False},
                 "c1": {"action": "native", "locked": False}}
        world.wall_delay[wall_id_for(("c0", "c1"), 16.0)] = -5.0
        decision = planner.decide(samples, 60.0, controller_state=state)
        assert decision.plan.walls == ()  # no wall without a fresh member
        named = [r for r in decision.reasons
                 if r.step == "wall" and "feasibility filter" in r.detail]
        assert named and "c1" in named[0].detail and "stale" in named[0].detail

    def test_legacy_samples_without_frame_data_pass_through(self):
        world, cfg, samples, wid = two_lane_setup()
        world.wall_delay[wid] = -5.0
        planner = Planner(world, cfg)
        decision = planner.decide(samples, 0.0,
                                  controller_state={"c0": {"action": "native",
                                                           "locked": False},
                                                    "c1": {"action": "native",
                                                           "locked": False}})
        assert len(decision.plan.walls) == 1  # no frame info: all eligible


class TestWallMergeCapEnforcement:
    """PL-03b (A): chained wall candidates must not merge evaluation groups
    beyond group_cap (root cause of the observed 8-sized groups with
    '0 edges')."""

    def test_chained_wall_candidates_respect_group_cap(self):
        world = lane_world({
            "c0": (0.0, 100.0), "c1": (1.0, 102.0),
            "c2": (0.0, 104.0), "c3": (1.0, 106.0),
            "c4": (0.0, 108.0), "c5": (1.0, 110.0),
            "c6": (0.0, 112.0), "c7": (1.0, 114.0),
        })
        cfg = PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0, group_cap=5)
        samples = framed_samples(tuple(f"c{i}" for i in range(8)))
        state = {f"c{i}": {"action": "native", "locked": False}
                 for i in range(8)}
        planner = Planner(world, cfg)
        decision = planner.decide(samples, 0.0, controller_state=state)
        sizes = [len(g) for g in decision.groups]
        assert all(size <= cfg.group_cap for size in sizes), sizes
        # at least one chaining wall candidate was dropped with a reason
        drops = [r for r in decision.reasons
                 if r.action == "veto" and "group_cap" in (r.detail or "")]
        assert drops
