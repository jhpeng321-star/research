"""MockWorldModel: contract shapes, additive model, scripting, validation."""
import numpy as np
import pytest

from traffic_shepherd.planner import plan_key
from tests.planner._helpers import make_samples, make_world, plan, wall


class TestContractShapes:
    def test_mk_shapes_and_delay_identity(self):
        world = make_world(cav_ids=("c0", "c1", "c2"), n_members=3)
        samples = make_samples(k=4)
        out = world.predict_many(samples, [plan({"c0": "damping", "c1": "native",
                                                 "c2": "native"})],
                                 horizon_s=600.0, trajectory_s=10)
        pred = out[0]
        for name in ("delay_vs_native_s", "plan_delay_s", "native_delay_s",
                     "delay_part_a_s", "delay_part_b_s", "min_ttc_s",
                     "ttc_lt3_exposure_s", "native_ttc_lt3_exposure_s",
                     "hard_brake_s", "native_hard_brake_s", "native_min_ttc_s"):
            assert getattr(pred, name).shape == (3, 4), name
        assert pred.collision.shape == (3, 4) and pred.collision.dtype == np.bool_
        assert pred.native_collision.dtype == np.bool_
        assert pred.plan_delay_s - pred.native_delay_s == \
            pytest.approx(pred.delay_vs_native_s)
        assert pred.delay_part_a_s + pred.delay_part_b_s == \
            pytest.approx(pred.plan_delay_s)

    def test_trajectory_shape_includes_t0_and_nan_after_exit(self):
        world = make_world(cav_ids=("c0",))
        world.exit_step["c0"] = {"damping": 4}
        samples = make_samples(k=2)
        pred = world.predict_many(samples, [plan({"c0": "damping"})],
                                  trajectory_s=10)[0]
        traj = pred.trajectories_tracked["c0"]
        assert traj.shape == (1, 2, 11, 4)  # T = trajectory_s + 1 incl. t0
        assert traj[0, 0, 0] == pytest.approx((0.0, 0.0, 0.0, 20.0))
        assert np.all(np.isnan(traj[0, 0, 4:]))
        assert np.all(np.isfinite(traj[0, 0, :4]))
        # columns are (t_s, s_m, lane_id, speed_mps)
        assert traj[0, 0, 3, 0] == pytest.approx(3.0)
        assert traj[0, 0, 3, 1] == pytest.approx(60.0)

    def test_output_order_and_predict_delegation(self):
        world = make_world(cav_ids=("c0",))
        samples = make_samples(k=1)
        plans = [plan({"c0": "native"}), plan({"c0": "damping"})]
        out = world.predict_many(samples, plans, trajectory_s=10)
        assert len(out) == 2
        assert out[1].trajectories_tracked["c0"][0, 0, -1, 1] == \
            pytest.approx(out[0].trajectories_tracked["c0"][0, 0, -1, 1])
        again = world.predict(samples, plans[1], None, trajectory_s=10)
        assert np.array_equal(again.delay_vs_native_s, out[1].delay_vs_native_s)

    def test_native_computed_once_per_call_and_shared(self):
        world = make_world(cav_ids=("c0",))
        samples = make_samples(k=2)
        out = world.predict_many(samples, [plan({"c0": "native"}),
                                           plan({"c0": "damping"})],
                                 trajectory_s=10)
        assert out[0].native_delay_s is out[1].native_delay_s
        assert (out[0].native_delay_s == 100.0).all()

    def test_walls_block_and_abort_reason_int8(self):
        world = make_world(cav_ids=("c0", "c1"))
        hold = wall("w0", ("c0", "c1"), phase="hold")
        samples = make_samples(k=1)
        pred = world.predict_many(samples, [plan({}, walls=[hold])],
                                  trajectory_s=10)[0]
        block = pred.walls["w0"]
        assert block.assembled.dtype == np.bool_ and block.assembled.all()
        assert block.safety_abort.dtype == np.bool_ and not block.safety_abort.any()
        assert block.abort_reason.dtype == np.int8
        assert int(block.abort_reason[0, 0]) == 0
        assert np.isfinite(block.assembled_at_s).all()
        assembling = wall("w0", ("c0", "c1"), phase="assemble")
        on_road = {"c0": {"action": "wall", "wall_id": "w0",
                          "wall_phase": "assemble", "locked": True,
                          "time_in_phase_s": 0.0},
                   "c1": {"action": "wall", "wall_id": "w0",
                          "wall_phase": "assemble", "locked": True,
                          "time_in_phase_s": 0.0}}
        pred = world.predict_many(
            samples, [plan({}, walls=[assembling])],
            controller_state=on_road, trajectory_s=10)[0]
        assert not pred.walls["w0"].assembled.any()
        assert np.isnan(pred.walls["w0"].assembled_at_s).all()
        assert pred.meta["native_kept_walls"] == ["w0"]  # N1: locked assemble walls kept
        # a wall candidate that is NOT on the road (absent from
        # controller_state) is not part of the native reference
        pred = world.predict_many(samples, [plan({}, walls=[assembling])],
                                  trajectory_s=10)[0]
        assert pred.meta["native_kept_walls"] == []

    def test_meta_fields(self):
        world = make_world(cav_ids=("c0",))
        samples = make_samples(k=2)
        meta = world.predict_many(samples, [plan({"c0": "native"})],
                                  horizon_s=600.0, trajectory_s=10)[0].meta
        assert meta["contract_version"] == "wd01_predict/1"
        assert meta["sample_ids"] == ["s0", "s1"]
        assert meta["member_ids"] == ["m0"]
        assert meta["horizon_s"] == 600.0
        assert meta["safety_window_s"] == 10
        assert meta["native_unlocked_available"] is False


class TestAdditiveModel:
    def test_action_and_pair_effects_sum(self):
        world = make_world(cav_ids=("c0", "c1"))
        world.action_delay["c0"] = {"damping": -5.0}
        world.action_delay["c1"] = {"damping": 2.0}
        world.pair_delay[("c0", "c1")] = {("damping", "damping"): 3.0}
        samples = make_samples(k=2)
        joint = world.predict_many(samples, [plan({"c0": "damping", "c1": "damping"})],
                                   trajectory_s=10)[0]
        assert joint.delay_vs_native_s == pytest.approx(0.0)  # -5 + 2 + 3
        single = world.predict_many(
            samples, [plan({"c0": "damping", "c1": "native"})],
            trajectory_s=10)[0]
        assert single.delay_vs_native_s == pytest.approx(-5.0)

    def test_scripted_values_broadcast_per_sample(self):
        world = make_world(cav_ids=("c0",))
        world.action_delay["c0"] = {"damping": np.array([-4.0, -1.0, 1.0])}
        samples = make_samples(k=3)
        pred = world.predict_many(samples, [plan({"c0": "damping"})],
                                  trajectory_s=10)[0]
        assert pred.delay_vs_native_s[0].tolist() == pytest.approx([-4.0, -1.0, 1.0])

    def test_plan_delay_override_replaces_additive_terms(self):
        world = make_world(cav_ids=("c0",))
        world.action_delay["c0"] = {"damping": -5.0}
        key = plan_key(plan({"c0": "damping"}))
        world.plan_delay_override[key] = 40.0
        samples = make_samples(k=1)
        pred = world.predict_many(samples, [plan({"c0": "damping"})],
                                  trajectory_s=10)[0]
        assert pred.plan_delay_s == pytest.approx(40.0)
        assert pred.delay_vs_native_s == pytest.approx(-60.0)

    def test_safety_and_native_safety_scripts(self):
        world = make_world(cav_ids=("c0",))
        key = plan_key(plan({"c0": "damping"}))
        world.safety_script[key] = {
            "collision": np.array([False, True]),
            "min_ttc_s": np.array([2.0, 0.5]),
            "ttc_lt3_exposure_s": np.array([1.0, 2.0]),
            "hard_brake_s": np.array([0.5, 0.0]),
        }
        world.native_safety = {
            "collision": False,
            "min_ttc_s": 9.0,
            "ttc_lt3_exposure_s": 0.5,
            "hard_brake_s": 0.25,
        }
        samples = make_samples(k=2)
        pred = world.predict_many(samples, [plan({"c0": "damping"})],
                                  trajectory_s=10)[0]
        assert pred.collision[0].tolist() == [False, True]
        assert pred.min_ttc_s[0].tolist() == pytest.approx([2.0, 0.5])
        assert pred.native_min_ttc_s == pytest.approx(9.0)
        assert pred.ttc_lt3_exposure_s[0].tolist() == pytest.approx([1.0, 2.0])
        assert pred.native_ttc_lt3_exposure_s == pytest.approx(0.5)
        assert pred.hard_brake_s[0].tolist() == pytest.approx([0.5, 0.0])
        assert pred.native_hard_brake_s == pytest.approx(0.25)

    def test_trajectory_offsets_own_and_coupling(self):
        world = make_world(cav_ids=("c0", "c1"))
        world.traj_pos_offset["c0"] = {"damping": -3.0}
        world.traj_speed_offset["c0"] = {"damping": -1.0}
        world.traj_coupling_pos[("c0", "c1")] = {"damping": 5.0}
        samples = make_samples(k=1)
        base = world.predict_many(
            samples, [plan({"c0": "native", "c1": "native"})],
            trajectory_s=10)[0]
        moved = world.predict_many(
            samples, [plan({"c0": "damping", "c1": "native"})],
            trajectory_s=10)[0]
        # own trajectory shifted by -3 in s and -1 in speed
        assert moved.trajectories_tracked["c0"][0, 0, 5, 1] == \
            pytest.approx(base.trajectories_tracked["c0"][0, 0, 5, 1] - 3.0)
        assert moved.trajectories_tracked["c0"][0, 0, 5, 3] == \
            pytest.approx(base.trajectories_tracked["c0"][0, 0, 5, 3] - 1.0)
        # coupled CAV c1 shifted by +5 in s although c1 keeps its action
        assert moved.trajectories_tracked["c1"][0, 0, 5, 1] == \
            pytest.approx(base.trajectories_tracked["c1"][0, 0, 5, 1] + 5.0)

    def test_native_trajectories_are_unaffected(self):
        world = make_world(cav_ids=("c0",))
        world.traj_pos_offset["c0"] = {"damping": -3.0}
        samples = make_samples(k=1)
        pred = world.predict_many(samples, [plan({"c0": "damping"})],
                                  trajectory_s=10)[0]
        plan_traj = pred.trajectories_tracked["c0"][0, 0, :, 1]
        native_traj = pred.native_trajectories_tracked["c0"][0, 0, :, 1]
        assert plan_traj[5] == pytest.approx(native_traj[5] - 3.0)

    def test_wall_script(self):
        world = make_world(cav_ids=("c0", "c1"))
        world.wall_script["w0"] = {"safety_abort": True, "abort_reason": 1}
        samples = make_samples(k=1)
        pred = world.predict_many(
            samples, [plan({}, walls=[wall("w0", ("c0", "c1"),
                                           phase="assemble")])],
            trajectory_s=10)[0]
        block = pred.walls["w0"]
        assert block.safety_abort.all()
        assert int(block.abort_reason[0, 0]) == 1


class TestDeterminism:
    def test_repeated_calls_identical(self):
        world = make_world(cav_ids=("c0", "c1"), n_members=2)
        world.action_delay["c0"] = {"damping": np.array([-4.0, -1.0])}
        world.traj_coupling_pos[("c0", "c1")] = {"damping": 3.0}
        samples = make_samples(k=2)
        plans = [plan({"c0": "damping", "c1": "native"}),
                 plan({"c0": "native", "c1": "damping"})]
        first = world.predict_many(samples, plans, trajectory_s=10)
        second = world.predict_many(samples, plans, trajectory_s=10)
        for a, b in zip(first, second):
            assert np.array_equal(a.delay_vs_native_s, b.delay_vs_native_s)
            for cav in ("c0", "c1"):
                assert np.array_equal(a.trajectories_tracked[cav],
                                      b.trajectories_tracked[cav],
                                      equal_nan=True)


class TestContractValidation:
    def test_incomplete_per_cav_rejected(self):
        world = make_world(cav_ids=("c0", "c1"))
        with pytest.raises(ValueError, match="FULL map"):
            world.predict_many(make_samples(1), [plan({"c0": "native"})],
                               trajectory_s=10)

    def test_wall_member_in_per_cav_rejected(self):
        world = make_world(cav_ids=("c0", "c1"))
        with pytest.raises(ValueError, match="per_cav"):
            world.predict_many(make_samples(1),
                               [plan({"c0": "native", "c1": "damping"},
                                     walls=[wall("w", ("c1",))])],
                               trajectory_s=10)

    def test_locked_cav_in_per_cav_rejected(self):
        world = make_world(cav_ids=("c0", "c1", "c2"))
        state = {"c0": {"action": "wall", "wall_id": "w", "locked": True},
                 "c1": {"action": "wall", "wall_id": "w", "locked": True},
                 "c2": {"action": "wall", "wall_id": "w", "locked": True}}
        # locked c2 named in per_cav instead of being a member of wall "w"
        bad = plan({"c2": "native"}, walls=[wall("w", ("c0", "c1"))])
        with pytest.raises(ValueError, match="must not appear in per_cav"):
            world.predict_many(make_samples(1), [bad],
                               controller_state=state, trajectory_s=10)

    def test_locked_cav_wrong_wall_rejected(self):
        world = make_world(cav_ids=("c0", "c1", "c2", "c3"))
        state = {"c0": {"action": "wall", "wall_id": "w", "locked": True},
                 "c1": {"action": "wall", "wall_id": "w", "locked": True},
                 "c2": {"action": "wall", "wall_id": "w", "locked": True},
                 "c3": {"action": "wall", "wall_id": "w2", "locked": True}}
        # locked CAV c2 points at "w", but the plan places it under "w2"
        bad = plan({}, walls=[wall("w", ("c0", "c1")),
                              wall("w2", ("c2", "c3"))])
        with pytest.raises(ValueError, match="existing wall"):
            world.predict_many(make_samples(1), [bad],
                               controller_state=state, trajectory_s=10)

    def test_trajectory_s_above_part_a_rejected(self):
        world = make_world(cav_ids=("c0",), part_a=10)
        with pytest.raises(ValueError, match="part_a_horizon_s"):
            world.predict_many(make_samples(1), [plan({"c0": "native"})],
                               trajectory_s=11)

    def test_mismatched_sample_times_rejected(self):
        world = make_world(cav_ids=("c0",))
        samples = make_samples(2)
        samples[1].meta["time_s"] = 1.0
        with pytest.raises(ValueError, match="time_s"):
            world.predict_many(samples, [plan({"c0": "native"})], trajectory_s=10)

    def test_empty_samples_rejected_empty_plans_allowed(self):
        world = make_world(cav_ids=("c0",))
        with pytest.raises(ValueError, match="non-empty"):
            world.predict_many([], [plan({"c0": "native"})], trajectory_s=10)
        assert world.predict_many(make_samples(1), [], trajectory_s=10) == []

    def test_sample_meta_fields_required(self):
        world = make_world(cav_ids=("c0",))
        sample = make_samples(1)[0]
        del sample.meta["coverage"]
        with pytest.raises(ValueError, match="coverage"):
            world.predict_many([sample], [plan({"c0": "native"})], trajectory_s=10)


def test_mock_records_calls():
    world = make_world(cav_ids=("c0",))
    world.predict_many(make_samples(3), [plan({"c0": "native"}),
                                         plan({"c0": "damping"})],
                       horizon_s=600.0, trajectory_s=10)
    assert world.calls == [{"n_plans": 2, "n_samples": 3, "n_members": 1,
                            "horizon_s": 600.0, "trajectory_s": 10}]


class TestV11Diagnostics:
    """contract v1.1: the mock reports cell validity and closing-pair flags
    (all-valid with a closing pair by default), scriptable per cell."""

    def test_defaults_are_valid_cells_with_closing_pair(self):
        world = make_world(cav_ids=("c0",))
        pred = world.predict_many(make_samples(2), [plan({"c0": "damping"})],
                                  trajectory_s=10)[0]
        for name in ("cell_valid", "native_cell_valid", "has_closing_pair",
                     "native_has_closing_pair"):
            arr = getattr(pred, name)
            assert arr.shape == (1, 2) and arr.dtype == np.bool_, name
            assert arr.all(), name

    def test_plan_side_scripting_per_cell(self):
        world = make_world(cav_ids=("c0",))
        world.safety_script[plan_key(plan({"c0": "damping"}))] = {
            "cell_valid": np.array([True, False]),
            "has_closing_pair": np.array([False, True]),
        }
        pred = world.predict_many(make_samples(2), [plan({"c0": "damping"})],
                                  trajectory_s=10)[0]
        assert pred.cell_valid[0].tolist() == [True, False]
        assert pred.has_closing_pair[0].tolist() == [False, True]
        assert pred.native_cell_valid.all()  # native side untouched

    def test_native_side_scripting_per_cell(self):
        world = make_world(cav_ids=("c0",))
        world.native_safety = {
            "native_cell_valid": np.array([True, False]),
            "native_has_closing_pair": np.array([False, True]),
        }
        pred = world.predict_many(make_samples(2), [plan({"c0": "native"})],
                                  trajectory_s=10)[0]
        assert pred.native_cell_valid[0].tolist() == [True, False]
        assert pred.native_has_closing_pair[0].tolist() == [False, True]
        assert pred.cell_valid.all()  # plan side untouched
