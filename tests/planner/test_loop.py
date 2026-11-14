"""End-to-end planner decision loop against the mock world model."""
import numpy as np
import pytest

from traffic_shepherd.planner import (
    Planner,
    PlannerConfig,
    WallSpec,
    plan_key,
)
from tests.planner._helpers import make_samples, make_world, plan, state_from_plan


def make(cav_ids=("c0", "c1"), k=4, **config_over):
    world = make_world(cav_ids, part_a=10)
    config = PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0, **config_over)
    return world, config, make_samples(k)


NATIVE_STATE = {
    "c0": {"action": "native", "locked": False},
    "c1": {"action": "native", "locked": False},
}


class TestStep1Individual:
    def test_native_default_when_nothing_beats_native(self):
        world, cfg, samples = make()
        world.action_delay["c0"] = {"damping": 2.0, "pi-external-u": 3.0}
        world.action_delay["c1"] = {"damping": 1.0, "pi-external-u": 0.5}
        planner = Planner(world, cfg)
        decision = planner.decide(samples, 0.0, controller_state=NATIVE_STATE)
        assert decision.plan.per_cav == {"c0": "native", "c1": "native"}
        assert decision.switched is False
        assert any(r.step == "individual" for r in decision.reasons)

    def test_best_action_adopted(self):
        world, cfg, samples = make()
        world.action_delay["c0"] = {"damping": -5.0}
        planner = Planner(world, cfg)
        decision = planner.decide(samples, 0.0, controller_state=NATIVE_STATE)
        assert decision.plan.per_cav == {"c0": "damping", "c1": "native"}
        assert decision.switched is True
        switch = [r for r in decision.reasons if r.action == "switch"]
        assert switch and switch[0].median_saving_s == pytest.approx(5.0)
        assert switch[0].agreement == pytest.approx(1.0)

    def test_tie_break_prefers_native(self):
        world, cfg, samples = make()
        world.action_delay["c0"] = {"damping": -5.0, "pi-external-u": -5.0}
        planner = Planner(world, cfg)
        decision = planner.decide(samples, 0.0, controller_state=NATIVE_STATE)
        # native (0) is worse; damping and pi tie at -5 -> damping (lower rank)
        assert decision.plan.per_cav["c0"] == "damping"

    def test_margin_blocks_the_switch(self):
        world, cfg, samples = make(k=10)
        # saving [2]*9+[-8]: median 2 < margin std 3 -> keep current
        world.action_delay["c0"] = {"damping": np.array([-2.0] * 9 + [8.0])}
        planner = Planner(world, cfg)
        decision = planner.decide(samples, 0.0, controller_state=NATIVE_STATE)
        assert decision.plan.per_cav["c0"] == "native"
        keep = [r for r in decision.reasons
                if r.step == "group" and r.subject == "c0"]
        assert keep and keep[0].margin_s is not None
        assert keep[0].median_saving_s == pytest.approx(2.0)

    def test_margin_passes_when_spread_small(self):
        world, cfg, samples = make(k=10)
        # saving [2]*9+[-2]: median 2 > margin 1.2 -> switch
        world.action_delay["c0"] = {"damping": np.array([-2.0] * 9 + [2.0])}
        planner = Planner(world, cfg)
        decision = planner.decide(samples, 0.0, controller_state=NATIVE_STATE)
        assert decision.plan.per_cav["c0"] == "damping"

    def test_agreement_79_vs_80_percent(self):
        for n_bad, switches in ((21, False), (20, True)):
            world, cfg, samples = make(k=100)
            world.action_delay["c0"] = {
                "damping": np.array([-1.0] * (100 - n_bad) + [0.01] * n_bad)}
            planner = Planner(world, cfg)
            decision = planner.decide(samples, 0.0, controller_state=NATIVE_STATE)
            assert (decision.plan.per_cav["c0"] == "damping") is switches

    def test_unsafe_action_never_picked(self):
        world, cfg, samples = make()
        world.action_delay["c0"] = {"damping": -50.0}
        world.safety_script[plan_key(plan({"c0": "damping", "c1": "native"}))] = \
            {"collision": True}
        planner = Planner(world, cfg)
        decision = planner.decide(samples, 0.0, controller_state=NATIVE_STATE)
        assert decision.plan.per_cav["c0"] == "native"  # vetoed unsafe damping


class TestStep2InteractionAndGroups:
    def test_coupled_pair_groups_and_joint_candidate_wins(self):
        world, cfg, samples = make(k=4)
        world.traj_coupling_pos[("c0", "c1")] = {"damping": 3.0}  # edge
        world.action_delay["c0"] = {"damping": -1.0}
        world.action_delay["c1"] = {"damping": -1.0}
        world.pair_delay[("c0", "c1")] = {("damping", "damping"): -4.0}
        planner = Planner(world, cfg)
        decision = planner.decide(samples, 0.0, controller_state=NATIVE_STATE)
        assert decision.groups == (("c0", "c1"),)
        assert decision.plan.per_cav == {"c0": "damping", "c1": "damping"}
        joint = next(r for r in decision.reasons if "joint" in r.candidate)
        assert joint.action == "switch"
        assert joint.median_saving_s == pytest.approx(6.0)

    def test_flip_one_beats_joint(self):
        world, cfg, samples = make(k=4)
        world.traj_coupling_pos[("c0", "c1")] = {"damping": 3.0}
        # individual picks: both damping (-5, -1); pair term makes joint +2;
        # flipping c1 back to native gives {damping, native} = -5 -> wins.
        world.action_delay["c0"] = {"damping": -5.0}
        world.action_delay["c1"] = {"damping": -1.0}
        world.pair_delay[("c0", "c1")] = {("damping", "damping"): 8.0}
        planner = Planner(world, cfg)
        decision = planner.decide(samples, 0.0, controller_state=NATIVE_STATE)
        assert decision.plan.per_cav == {"c0": "damping", "c1": "native"}
        flip = next(r for r in decision.reasons
                    if r.candidate.startswith("flip"))
        assert flip.action == "switch"
        assert "c1" in flip.candidate

    def test_no_edge_no_group_singletons(self):
        world, cfg, samples = make(k=4)
        world.action_delay["c0"] = {"damping": -5.0}
        world.action_delay["c1"] = {"damping": -4.0}
        planner = Planner(world, cfg)
        decision = planner.decide(samples, 0.0, controller_state=NATIVE_STATE)
        assert decision.groups == (("c0",), ("c1",))
        assert decision.plan.per_cav == {"c0": "damping", "c1": "damping"}

    def test_cap_splits_large_component(self):
        ids = tuple(f"c{i}" for i in range(6))
        world = make_world(ids, part_a=10)
        cfg = PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0, group_cap=2)
        samples = make_samples(2)
        state = {cav: {"action": "native", "locked": False} for cav in ids}
        for i in range(5):  # chain c0-...-c5; only offsets 3,4,5 exceed 2.0
            world.traj_coupling_pos[(f"c{i}", f"c{i + 1}")] = {"damping": 1.0 + i}
        planner = Planner(world, cfg)
        decision = planner.decide(samples, 0.0, controller_state=state)
        assert all(len(g) <= 2 for g in decision.groups)
        # the 4-node component c2..c5 split at its weakest edge (c2-c3)
        assert ("c4", "c5") in decision.groups
        assert ("c2",) in decision.groups and ("c3",) in decision.groups


class TestStep4SwitchAndNativeReturn:
    def test_keep_current_when_no_candidate_passes(self):
        world, cfg, samples = make()
        planner = Planner(world, cfg)
        decision = planner.decide(samples, 0.0, controller_state=NATIVE_STATE)
        assert decision.switched is False
        assert any(r.action == "keep" and r.step == "combine"
                   for r in decision.reasons)

    def test_native_return_when_current_worse_than_native(self):
        world, cfg, samples = make(k=4)
        start = plan({"c0": "damping", "c1": "native"})
        # current delay_vs_native [4,4,4,-10]: median 4 > 0 (worse than
        # native), but median saving vs native (4) < margin (std 6.06), so the
        # plain switch rule fails and the native-return exception fires.
        world.plan_dvn_override[plan_key(start)] = np.array([4.0, 4.0, 4.0, -10.0])
        planner = Planner(world, cfg, current_plan=start)
        decision = planner.decide(samples, 0.0, controller_state={
            "c0": {"action": "damping", "locked": False},
            "c1": {"action": "native", "locked": False}})
        assert decision.plan.per_cav["c0"] == "native"
        returns = [r for r in decision.reasons if r.action == "native_return"]
        assert returns and "worse than native" in returns[0].detail

    def test_native_return_when_current_unsafe(self):
        world, cfg, samples = make(k=2)
        start = plan({"c0": "damping", "c1": "native"})
        world.safety_script[plan_key(start)] = {"collision": True}
        world.action_delay["c0"] = {"damping": np.array([-5.0, 5.0])}  # rule fails
        planner = Planner(world, cfg, current_plan=start)
        decision = planner.decide(samples, 0.0, controller_state={
            "c0": {"action": "damping", "locked": False},
            "c1": {"action": "native", "locked": False}})
        assert decision.plan.per_cav["c0"] == "native"
        returns = [r for r in decision.reasons if r.action == "native_return"]
        assert returns and "unsafe" in returns[0].detail


class TestStep5Combine:
    def test_combined_unsafe_drops_lowest_saving_group(self):
        ids = ("c0", "c1", "c2", "c3")
        world = make_world(ids, part_a=10)
        cfg = PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0)
        samples = make_samples(4)
        world.traj_coupling_pos[("c0", "c1")] = {"damping": 3.0}
        world.traj_coupling_pos[("c2", "c3")] = {"damping": 3.0}
        world.action_delay["c0"] = {"damping": -5.0}
        world.action_delay["c1"] = {"damping": -5.0}
        world.action_delay["c2"] = {"damping": -1.0}
        world.action_delay["c3"] = {"damping": -1.0}
        combined = plan({cav: "damping" for cav in ids})
        world.safety_script[plan_key(combined)] = {"collision": True}
        planner = Planner(world, cfg)
        decision = planner.decide(samples, 0.0, controller_state={
            **{cav: {"action": "native", "locked": False} for cav in ids}})
        # group (c0,c1) saves 10, group (c2,c3) saves 2 -> drop (c2,c3) first
        assert decision.plan.per_cav == {"c0": "damping", "c1": "damping",
                                         "c2": "native", "c3": "native"}
        drops = [r for r in decision.reasons if r.action == "drop"]
        assert drops and drops[0].subject == "c2+c3"

    def test_worst_case_everything_native_when_even_native_unsafe(self):
        world, cfg, samples = make(k=2)
        start = plan({"c0": "damping", "c1": "native"})
        all_native = plan({"c0": "native", "c1": "native"})
        world.safety_script[plan_key(start)] = {"collision": True}
        world.safety_script[plan_key(all_native)] = {"collision": True}
        planner = Planner(world, cfg, current_plan=start)
        decision = planner.decide(samples, 0.0, controller_state={
            "c0": {"action": "damping", "locked": False},
            "c1": {"action": "native", "locked": False}})
        assert decision.plan.per_cav == {"c0": "native", "c1": "native"}
        worst = [r for r in decision.reasons if "worst-case" in r.detail]
        assert worst and worst[0].safety_ok is False

    def test_fallback_disabled_commits_unsafe_combined(self):
        world, cfg, samples = make(combined_plan_fallback=False)
        world.action_delay["c0"] = {"damping": -5.0}
        world.action_delay["c1"] = {"damping": -5.0}
        combined = plan({"c0": "damping", "c1": "damping"})
        world.safety_script[plan_key(combined)] = {"collision": True}
        planner = Planner(world, cfg)
        decision = planner.decide(samples, 0.0, controller_state=NATIVE_STATE)
        assert decision.plan == combined  # committed despite the veto
        assert any("combined plan unsafe" in r.detail or
                   (r.step == "combine" and r.safety_ok is False)
                   for r in decision.reasons)


class TestStep0Abstain:
    def test_abstain_keeps_current_plan(self):
        world, cfg, samples = make(k=4, abstain_min_observed=0.8)
        samples = make_samples(4, observed=3)
        world.action_delay["c0"] = {"damping": -5.0}
        planner = Planner(world, cfg)
        decision = planner.decide(samples, 0.0, controller_state=NATIVE_STATE)
        assert decision.abstain is True and decision.switched is False
        assert decision.plan.per_cav == {"c0": "native", "c1": "native"}
        assert decision.reasons[0].step == "abstain"

    def test_observed_fraction_at_threshold_does_not_abstain(self):
        world, cfg, _ = make(k=4, abstain_min_observed=0.75)
        samples = make_samples(4, observed=3)  # exactly 0.75
        planner = Planner(world, cfg)
        decision = planner.decide(samples, 0.0, controller_state=NATIVE_STATE)
        assert decision.abstain is False

    def test_abstain_exception_replaces_unsafe_current(self):
        world, cfg, _ = make(abstain_min_observed=0.9)
        samples = make_samples(4, observed=1)
        start = plan({"c0": "damping", "c1": "native"})
        world.safety_script[plan_key(start)] = {"collision": True}
        planner = Planner(world, cfg, current_plan=start)
        decision = planner.decide(samples, 0.0, controller_state={
            "c0": {"action": "damping", "locked": False},
            "c1": {"action": "native", "locked": False}})
        assert decision.abstain is True
        assert decision.plan.per_cav == {"c0": "native", "c1": "native"}
        assert any("owner exception" in r.detail for r in decision.reasons)

    def test_no_exception_keeps_unsafe_current_when_observation_poor(self):
        world, cfg, _ = make(abstain_min_observed=0.9,
                             abstain_safety_exception=False)
        samples = make_samples(4, observed=1)
        start = plan({"c0": "damping", "c1": "native"})
        world.safety_script[plan_key(start)] = {"collision": True}
        planner = Planner(world, cfg, current_plan=start)
        decision = planner.decide(samples, 0.0, controller_state={
            "c0": {"action": "damping", "locked": False},
            "c1": {"action": "native", "locked": False}})
        assert decision.abstain is True and decision.switched is False
        assert decision.plan == start


class TestCadence:
    def test_skip_between_ticks_then_full_at_tick(self):
        world, cfg, samples = make()
        world.action_delay["c0"] = {"damping": -5.0}
        planner = Planner(world, cfg)
        first = planner.decide(samples, 0.0, controller_state=NATIVE_STATE)
        assert first.switched and first.cadence_skipped is False
        world.action_delay["c0"] = {"damping": 5.0}  # world turned bad
        skipped = planner.decide(samples, 0.5, controller_state={
            "c0": {"action": "damping", "locked": False},
            "c1": {"action": "native", "locked": False}})
        assert skipped.cadence_skipped is True and skipped.switched is False
        assert skipped.plan.per_cav["c0"] == "damping"  # kept despite being bad
        full = planner.decide(samples, 1.0, controller_state={
            "c0": {"action": "damping", "locked": False},
            "c1": {"action": "native", "locked": False}})
        assert full.cadence_skipped is False
        assert full.plan.per_cav["c0"] == "native"  # re-optimized at the tick

    def test_cadence_safety_check_replaces_unsafe_plan(self):
        world, cfg, samples = make()
        world.action_delay["c0"] = {"damping": -5.0}
        planner = Planner(world, cfg)
        first = planner.decide(samples, 0.0, controller_state=NATIVE_STATE)
        assert first.plan.per_cav["c0"] == "damping"
        damping = plan({"c0": "damping", "c1": "native"})
        world.safety_script[plan_key(damping)] = {"collision": True}
        later = planner.decide(samples, 0.5, controller_state={
            "c0": {"action": "damping", "locked": False},
            "c1": {"action": "native", "locked": False}})
        assert later.cadence_skipped is True
        assert later.plan.per_cav == {"c0": "native", "c1": "native"}
        assert later.switched is True
        assert later.reasons[0].step == "cadence"


class TestLockedWallMembers:
    def _walled_setup(self):
        world, cfg, samples = make()
        wall = WallSpec(wall_id="w0", members=("c1",), speed_mps=16.0,
                        phase="hold", hold_s=60.0)
        start = plan({"c0": "native"}, walls=[wall])
        return world, cfg, samples, wall, start

    def test_free_cav_optimized_wall_untouched(self):
        world, cfg, samples, wall, start = self._walled_setup()
        world.action_delay["c0"] = {"damping": -5.0}
        planner = Planner(world, cfg, current_plan=start)
        decision = planner.decide(samples, 0.0,
                                  controller_state=state_from_plan(start))
        assert decision.plan.per_cav == {"c0": "damping"}
        assert decision.plan.walls == (wall,)
        assert "c1" not in decision.plan.per_cav
        assert decision.groups == (("c0",),)  # locked CAV never grouped

    def test_native_return_leaves_wall(self):
        world, cfg, samples, wall, start = self._walled_setup()
        worse = plan({"c0": "damping"}, walls=[wall])
        world.action_delay["c0"] = {"damping": 4.0}
        planner = Planner(world, cfg, current_plan=worse)
        decision = planner.decide(samples, 0.0,
                                  controller_state=state_from_plan(worse))
        assert decision.plan.per_cav == {"c0": "native"}
        assert decision.plan.walls == (wall,)

    def test_locked_member_requires_explicit_current_plan(self):
        world, cfg, samples = make()
        planner = Planner(world, cfg)
        state = dict(NATIVE_STATE)
        state["c1"] = {"action": "wall", "wall_id": "w0", "locked": True}
        with pytest.raises(ValueError, match="current_plan"):
            planner.decide(samples, 0.0, controller_state=state)

    def test_current_plan_missing_locked_wall_rejected(self):
        world, cfg, samples = make()
        planner = Planner(world, cfg, current_plan=plan({"c0": "native"}))
        state = dict(NATIVE_STATE)
        state["c1"] = {"action": "wall", "wall_id": "w0", "locked": True}
        with pytest.raises(ValueError, match="existing wall"):
            planner.decide(samples, 0.0, controller_state=state)

    def test_controller_state_without_plan_and_roster(self):
        world, cfg, samples = make()
        planner = Planner(world, cfg)
        with pytest.raises(ValueError, match="controller_state or"):
            planner.decide(samples, 0.0)


class TestDeterminismAndRecords:
    def test_same_inputs_same_decision(self):
        def run():
            world, cfg, samples = make(k=4)
            world.traj_coupling_pos[("c0", "c1")] = {"damping": 3.0}
            world.action_delay["c0"] = {"damping": -1.0}
            world.action_delay["c1"] = {"damping": -1.0}
            world.pair_delay[("c0", "c1")] = {("damping", "damping"): -4.0}
            planner = Planner(world, cfg)
            return planner.decide(samples, 0.0, controller_state=NATIVE_STATE)

        assert run() == run()

    def test_reason_records_carry_required_fields(self):
        world, cfg, samples = make(k=4)
        world.traj_coupling_pos[("c0", "c1")] = {"damping": 3.0}
        world.action_delay["c0"] = {"damping": -1.0}
        world.action_delay["c1"] = {"damping": -1.0}
        world.pair_delay[("c0", "c1")] = {("damping", "damping"): -4.0}
        decision = Planner(world, cfg).decide(samples, 0.0,
                                              controller_state=NATIVE_STATE)
        steps = {r.step for r in decision.reasons}
        assert {"individual", "interaction", "group", "combine"} <= steps
        switch = next(r for r in decision.reasons if r.action == "switch")
        assert switch.median_saving_s is not None
        assert switch.margin_s is not None
        assert switch.agreement is not None
        assert switch.safety_ok is True

    def test_empty_samples_rejected(self):
        world, cfg, _ = make()
        with pytest.raises(ValueError):
            Planner(world, cfg).decide([], 0.0, controller_state=NATIVE_STATE)
