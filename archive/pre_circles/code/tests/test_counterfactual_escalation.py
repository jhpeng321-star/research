from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from config.constants import PRE_MERGE_EDGE
from policies.counterfactual_escalation import (
    DAMPING_ONLY,
    DAMPING_PLUS_WALL,
    CounterfactualEscalationPolicy,
    TreatmentState,
)
from policies.inflow_control import WallHoldPolicy


class _BasePolicy:
    def __init__(self):
        self.actions = {"cav.in": 2, "cav.out": 1}
        self.calls = []

    def reset(self, env, seed=None):
        self.env = env
        self.seed = seed

    def act(self, obs):
        self.calls.append(tuple(obs))
        return self.actions

    def close(self):
        pass


def _traci_state(*, collision=False, hard_brake=False, include_hdv=False):
    live = ["cav.in", "cav.out"]
    if include_hdv:
        live.append("hdv.lane1")
    roads = {vehicle_id: PRE_MERGE_EDGE for vehicle_id in live}
    lanes = {
        "cav.in": f"{PRE_MERGE_EDGE}_0",
        "cav.out": f"{PRE_MERGE_EDGE}_1",
        "hdv.lane1": f"{PRE_MERGE_EDGE}_1",
    }
    distances = {"cav.in": 300.0, "cav.out": 900.0, "hdv.lane1": 310.0}
    return live, roads, lanes, distances, collision, hard_brake


def _policy(*, include_hdv=False, cooldown_steps=2):
    base = _BasePolicy()
    wall = MagicMock()
    wall_factory = MagicMock(return_value=wall)
    policy = CounterfactualEscalationPolicy(
        base_policy=base,
        wall_factory=wall_factory,
        min_wall_cavs=1,
        min_wall_lanes=1,
        require_all_occupied_lanes=include_hdv,
        cooldown_steps=cooldown_steps,
    )
    policy.reset(SimpleNamespace(simulation_step=1.0), seed=101)
    return policy, base, wall


def _patch_traci(state):
    live, roads, lanes, distances, collision, hard_brake = state
    return (
        patch("traci.vehicle.getIDList", return_value=live),
        patch("traci.vehicle.getRoadID", side_effect=lambda vehicle_id: roads[vehicle_id]),
        patch("traci.vehicle.getLaneID", side_effect=lambda vehicle_id: lanes[vehicle_id]),
        patch(
            "policies.counterfactual_escalation.distance_to_merge",
            side_effect=lambda vehicle_id: distances[vehicle_id],
        ),
        patch(
            "traci.simulation.getCollidingVehiclesIDList",
            return_value=["cav.in"] if collision else [],
        ),
        patch("traci.simulation.getStartingTeleportIDList", return_value=[]),
        patch(
            "traci.vehicle.getAcceleration",
            return_value=-5.0 if hard_brake else 0.0,
        ),
    )


def _act(policy, state, command, obs=None):
    policy.set_command(command)
    patches = _patch_traci(state)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        return policy.act(obs or {"cav.in": None, "cav.out": None})


def test_damping_only_returns_exact_base_action_object_without_wall_write():
    policy, base, wall = _policy()

    actions = _act(policy, _traci_state(), DAMPING_ONLY)

    assert actions is base.actions
    assert base.calls == [("cav.in", "cav.out")]
    wall.act.assert_not_called()
    assert policy.last_audit["wall_override_ids"] == ()


def test_treatment_overrides_only_inband_cav_and_preserves_out_of_band_action():
    policy, base, wall = _policy()

    actions = _act(policy, _traci_state(), DAMPING_PLUS_WALL)

    assert actions == {"cav.out": base.actions["cav.out"]}
    wall.act.assert_called_once()
    assert set(wall.act.call_args.args[0]) == {"cav.in"}
    assert policy.last_audit["wall_override_ids"] == ("cav.in",)
    assert policy.last_audit["out_of_band_damping_ids"] == ("cav.out",)
    assert policy.last_audit["applied_env_actions"] == {"cav.out": base.actions["cav.out"]}


def test_uncovered_occupied_lane_deterministically_vetoes_formation():
    policy, base, wall = _policy(include_hdv=True)

    actions = _act(
        policy,
        _traci_state(include_hdv=True),
        DAMPING_PLUS_WALL,
    )

    assert actions is base.actions
    wall.act.assert_not_called()
    assert policy.last_audit["transition_reason"] == "formation_veto:uncovered_occupied_lane"


def test_newly_occupied_lane_does_not_abort_an_active_frozen_formation():
    policy, _base, wall = _policy(include_hdv=True)
    _act(policy, _traci_state(include_hdv=False), DAMPING_PLUS_WALL)
    wall.reset_mock()

    actions = _act(
        policy,
        _traci_state(include_hdv=True),
        DAMPING_PLUS_WALL,
    )

    assert policy.wall_active is True
    assert policy.treatment_state == TreatmentState.HOLDING
    assert actions == {"cav.out": 1}
    wall.act.assert_called_once()
    assert policy.last_audit["wall_override_ids"] == ("cav.in",)
    assert policy.last_audit["frozen_lane_ids"] == (f"{PRE_MERGE_EDGE}_0",)
    assert policy.last_audit["transition_reason"] == (
        "wall_holding_with_formation_gap:uncovered_occupied_lane"
    )


def test_departed_member_returns_to_damping_and_gets_same_lane_handoff():
    policy, base, wall = _policy()
    _act(policy, _traci_state(), DAMPING_PLUS_WALL)
    wall.reset_mock()

    live = ["cav.in", "cav.out"]
    roads = {vehicle_id: PRE_MERGE_EDGE for vehicle_id in live}
    lanes = {
        "cav.in": f"{PRE_MERGE_EDGE}_0",
        "cav.out": f"{PRE_MERGE_EDGE}_0",
    }
    distances = {"cav.in": 100.0, "cav.out": 300.0}
    state = (live, roads, lanes, distances, False, False)
    actions = _act(policy, state, DAMPING_PLUS_WALL)

    assert actions == {"cav.in": base.actions["cav.in"]}
    wall.act.assert_called_once()
    assert set(wall.act.call_args.args[0]) == {"cav.out"}
    assert policy.last_audit["wall_override_ids"] == ("cav.out",)
    assert policy.last_audit["out_of_band_damping_ids"] == ("cav.in",)
    events = policy.last_audit["member_changes"]
    assert [event["event"] for event in events] == ["removed", "handoff"]
    assert events[0]["reason"] == "left_wall_band"
    assert events[1]["previous_member_id"] == "cav.in"
    assert events[1]["member_id"] == "cav.out"
    wall.restore_tau.assert_called_once_with(("cav.in",))

    released = _act(policy, state, DAMPING_ONLY)
    assert released is base.actions
    assert wall.restore_tau.call_args_list == [
        call(("cav.in",)),
        call(("cav.in", "cav.out")),
    ]


def test_activation_safety_screen_vetoes_fast_closing_follower():
    policy, base, wall = _policy()

    speeds = {"cav.in": 10.0, "hdv.follow": 15.0}
    with (
        patch("traci.vehicle.getSpeed", side_effect=lambda vehicle_id: speeds[vehicle_id]),
        patch("traci.vehicle.getLeader", return_value=None),
        patch("traci.vehicle.getFollower", return_value=("hdv.follow", 10.0)),
    ):
        actions = _act(policy, _traci_state(), DAMPING_PLUS_WALL)

    assert actions is base.actions
    wall.act.assert_not_called()
    assert policy.last_audit["transition_reason"] == "activation_safety_veto:rear_ttc"
    assert policy.last_audit["activation_safety"]["safe"] is False
    assert policy.last_audit["activation_safety"]["min_rear_ttc_s"] == 2.0


def test_release_returns_to_damping_and_never_uses_release_to_sumo():
    policy, base, wall = _policy()
    _act(policy, _traci_state(), DAMPING_PLUS_WALL)

    with patch("policies.traci_control.release_to_sumo") as release:
        actions = _act(policy, _traci_state(), DAMPING_ONLY)

    assert actions is base.actions
    assert policy.wall_active is False
    assert policy.cooldown_remaining == 2
    assert policy.last_audit["transition_reason"] == "pulse_released_to_damping"
    wall.restore_tau.assert_called_once_with(("cav.in",))
    release.assert_not_called()


def test_wall_tau_restore_changes_only_tau_and_keeps_base_speed_control():
    wall = WallHoldPolicy(
        d_lo=200.0,
        d_hi=400.0,
        wall_speed=None,
        wall_tau=1.0,
        tau_ramp_step=0.05,
    )
    wall.reset(SimpleNamespace(simulation_step=0.5), seed=101)

    with (
        patch("policies.inflow_control.configure_cav"),
        patch("traci.vehicle.getIDList", return_value=["cav.in"]),
        patch("traci.vehicle.getTau", return_value=1.0),
        patch("policies.inflow_control.distance_to_merge", return_value=300.0),
        patch("traci.vehicle.setTau") as set_tau,
        patch("traci.vehicle.setSpeed") as set_speed,
        patch("policies.inflow_control.release_to_sumo") as release,
    ):
        wall.act({"cav.in": None})
        restoration = wall.restore_tau(("cav.in",))

    assert [call.args for call in set_tau.call_args_list] == [
        ("cav.in", 1.05),
        ("cav.in", 1.0),
    ]
    set_speed.assert_not_called()
    release.assert_not_called()
    assert restoration == [
        {
            "vehicle_id": "cav.in",
            "original_tau_s": 1.0,
            "removed_wall_tau_offset_s": 0.05,
            "restored": True,
            "error": None,
        }
    ]


def test_safety_abort_latches_request_and_cooldown_blocks_next_rising_edge():
    policy, base, wall = _policy(cooldown_steps=2)
    _act(policy, _traci_state(), DAMPING_PLUS_WALL)
    wall.reset_mock()

    aborted = _act(
        policy,
        _traci_state(hard_brake=True),
        DAMPING_PLUS_WALL,
    )
    assert aborted is base.actions
    assert policy.wall_active is False
    assert policy.last_audit["transition_reason"] == "safety_abort:controlled_cav_hard_brake"
    wall.act.assert_not_called()

    _act(policy, _traci_state(), DAMPING_ONLY)
    vetoed = _act(policy, _traci_state(), DAMPING_PLUS_WALL)
    assert vetoed is base.actions
    assert policy.last_audit["transition_reason"] == "cooldown_veto"
    wall.act.assert_not_called()


def test_internal_wall_trigger_options_are_rejected():
    try:
        CounterfactualEscalationPolicy(base_policy=_BasePolicy(), wall_params={"trigger_h": 0.25})
    except ValueError as exc:
        assert "autonomous option trigger_h" in str(exc)
    else:
        raise AssertionError("health-triggered treatment must be rejected")


def test_compact_geometry_selects_one_headroom_safe_cav_per_lane_and_tracks_members():
    base = _BasePolicy()
    wall = MagicMock()
    wall_factory = MagicMock(return_value=wall)
    policy = CounterfactualEscalationPolicy(
        base_policy=base,
        wall_factory=wall_factory,
        wall_params={"d_lo": 200.0, "d_hi": 800.0},
        geometry_mode="compact_moving",
        planned_treatment_duration_s=30.0,
        max_longitudinal_span_m=50.0,
    )
    policy.reset(SimpleNamespace(simulation_step=0.5), seed=101)
    live = ["cav.0.near", "cav.0.far", "cav.1.far", "cav.out"]
    roads = {vehicle_id: PRE_MERGE_EDGE for vehicle_id in live}
    lanes = {
        "cav.0.near": f"{PRE_MERGE_EDGE}_0",
        "cav.0.far": f"{PRE_MERGE_EDGE}_0",
        "cav.1.far": f"{PRE_MERGE_EDGE}_1",
        "cav.out": f"{PRE_MERGE_EDGE}_1",
    }
    distances = {
        "cav.0.near": 300.0,
        "cav.0.far": 705.0,
        "cav.1.far": 730.0,
        "cav.out": 900.0,
    }
    base.actions = {vehicle_id: 1 for vehicle_id in live}
    state = (live, roads, lanes, distances, False, False)
    with patch("traci.vehicle.getSpeed", return_value=15.0):
        actions = _act(
            policy,
            state,
            DAMPING_PLUS_WALL,
            {vehicle_id: None for vehicle_id in live},
        )

    assert set(policy.last_audit["wall_override_ids"]) == {"cav.0.far", "cav.1.far"}
    assert set(actions) == {"cav.0.near", "cav.out"}
    assert policy.last_audit["formation"]["longitudinal_span_m"] == 25.0
    assert wall_factory.call_args.kwargs["d_lo"] == 0.0
    assert wall_factory.call_args.kwargs["d_hi"] == 1.0e9


def test_trajectory_coherent_compact_geometry_prefers_speed_matched_members():
    base = _BasePolicy()
    wall = MagicMock()
    policy = CounterfactualEscalationPolicy(
        base_policy=base,
        wall_factory=MagicMock(return_value=wall),
        wall_params={"d_lo": 200.0, "d_hi": 1000.0},
        geometry_mode="compact_moving",
        planned_treatment_duration_s=20.0,
        compact_release_boundary_m=50.0,
        compact_selection_mode="trajectory_coherent",
        max_longitudinal_span_m=50.0,
    )
    policy.reset(SimpleNamespace(simulation_step=0.5), seed=101)
    live = ["cav.fast", "cav.matched", "cav.lane1"]
    roads = {vehicle_id: PRE_MERGE_EDGE for vehicle_id in live}
    lanes = {
        "cav.fast": f"{PRE_MERGE_EDGE}_0",
        "cav.matched": f"{PRE_MERGE_EDGE}_0",
        "cav.lane1": f"{PRE_MERGE_EDGE}_1",
    }
    distances = {
        "cav.fast": 800.0,
        "cav.matched": 760.0,
        "cav.lane1": 790.0,
    }
    speeds = {
        "cav.fast": 30.0,
        "cav.matched": 20.0,
        "cav.lane1": 20.0,
    }
    base.actions = {vehicle_id: 1 for vehicle_id in live}
    state = (live, roads, lanes, distances, False, False)

    with patch(
        "traci.vehicle.getSpeed",
        side_effect=lambda vehicle_id: speeds[vehicle_id],
    ):
        actions = _act(
            policy,
            state,
            DAMPING_PLUS_WALL,
            {vehicle_id: None for vehicle_id in live},
        )

    assert set(policy.last_audit["wall_override_ids"]) == {
        "cav.matched",
        "cav.lane1",
    }
    assert set(actions) == {"cav.fast"}
    assert policy.last_audit["formation"]["predicted_max_span_m"] <= 50.0


def test_adaptive_trajectory_projection_rejects_saturated_servo_divergence():
    base = _BasePolicy()
    policy = CounterfactualEscalationPolicy(
        base_policy=base,
        wall_factory=MagicMock(return_value=MagicMock()),
        wall_params={
            "d_lo": 200.0,
            "d_hi": 1000.0,
            "wall_speed": 10.0,
        },
        geometry_mode="compact_moving",
        planned_treatment_duration_s=20.0,
        compact_release_boundary_m=50.0,
        compact_selection_mode="trajectory_coherent",
        compact_control_mode="adaptive_servo",
        compact_max_speed_reduction_fraction=0.30,
        compact_servo_a_max_dec_mps2=1.0,
        compact_assembly_speed_tolerance_mps=2.0,
        max_longitudinal_span_m=50.0,
    )
    policy.reset(SimpleNamespace(simulation_step=1.0), seed=612)
    live = ["mainline.103", "mainline.104"]
    roads = {vehicle_id: PRE_MERGE_EDGE for vehicle_id in live}
    lanes = {
        "mainline.103": f"{PRE_MERGE_EDGE}_0",
        "mainline.104": f"{PRE_MERGE_EDGE}_1",
    }
    distances = {
        "mainline.103": 822.444045117892,
        "mainline.104": 862.7740327789147,
    }
    speeds = {
        "mainline.103": 29.103095144228575,
        "mainline.104": 26.344533497755204,
    }
    base.actions = {vehicle_id: 1 for vehicle_id in live}
    state = (live, roads, lanes, distances, False, False)

    with (
        patch(
            "traci.vehicle.getSpeed",
            side_effect=lambda vehicle_id: speeds[vehicle_id],
        ),
        patch.object(
            policy,
            "_mean_speed_in_distance_band",
            return_value=10.0,
        ),
    ):
        actions = _act(
            policy,
            state,
            DAMPING_PLUS_WALL,
            {vehicle_id: None for vehicle_id in live},
        )

    formation = policy.last_audit["formation"]
    assert actions is base.actions
    assert formation["feasible"] is False
    assert formation["reason"] == "predicted_trajectory_span_exceeded"
    assert formation["predicted_max_span_m"] > 50.0


def test_compact_assembly_is_a_real_phase_before_holding():
    base = _BasePolicy()
    wall = MagicMock()
    policy = CounterfactualEscalationPolicy(
        base_policy=base,
        wall_factory=MagicMock(return_value=wall),
        wall_params={"d_lo": 200.0, "d_hi": 1000.0},
        geometry_mode="compact_moving",
        planned_treatment_duration_s=20.0,
        compact_release_boundary_m=50.0,
        compact_selection_mode="trajectory_coherent",
        compact_assembly_duration_s=2.0,
        compact_assembly_speed_tolerance_mps=1.0,
        max_longitudinal_span_m=50.0,
    )
    policy.reset(SimpleNamespace(simulation_step=0.5), seed=101)
    live = ["cav.0", "cav.1"]
    roads = {vehicle_id: PRE_MERGE_EDGE for vehicle_id in live}
    lanes = {
        "cav.0": f"{PRE_MERGE_EDGE}_0",
        "cav.1": f"{PRE_MERGE_EDGE}_1",
    }
    distances = {"cav.0": 780.0, "cav.1": 790.0}
    speeds = {"cav.0": 22.0, "cav.1": 20.0}
    base.actions = {vehicle_id: 1 for vehicle_id in live}
    state = (live, roads, lanes, distances, False, False)

    with patch("traci.vehicle.getSpeed", side_effect=lambda vid: speeds[vid]):
        _act(policy, state, DAMPING_PLUS_WALL, {vid: None for vid in live})

    assert policy.treatment_state == TreatmentState.ASSEMBLING
    assert policy.last_audit["transition_reason"] == "wall_assembly_started"
    assert policy.last_audit["assembly_elapsed_steps"] == 1
    assert policy.last_audit["assembly_target_speed_mps"] == 20.0

    speeds.update({"cav.0": 20.5, "cav.1": 20.0})
    with patch("traci.vehicle.getSpeed", side_effect=lambda vid: speeds[vid]):
        _act(policy, state, DAMPING_PLUS_WALL, {vid: None for vid in live})

    assert policy.treatment_state == TreatmentState.HOLDING
    assert policy.last_audit["transition_reason"] == "wall_assembly_complete"
    assert policy.wall_active is True


def test_compact_assembly_timeout_is_an_explicit_treatment_failure():
    base = _BasePolicy()
    wall = MagicMock()
    policy = CounterfactualEscalationPolicy(
        base_policy=base,
        wall_factory=MagicMock(return_value=wall),
        wall_params={"d_lo": 200.0, "d_hi": 1000.0},
        geometry_mode="compact_moving",
        planned_treatment_duration_s=20.0,
        compact_release_boundary_m=50.0,
        compact_selection_mode="trajectory_coherent",
        compact_assembly_duration_s=0.5,
        compact_assembly_speed_tolerance_mps=1.0,
        max_longitudinal_span_m=50.0,
    )
    policy.reset(SimpleNamespace(simulation_step=0.5), seed=101)
    live = ["cav.0", "cav.1"]
    roads = {vehicle_id: PRE_MERGE_EDGE for vehicle_id in live}
    lanes = {
        "cav.0": f"{PRE_MERGE_EDGE}_0",
        "cav.1": f"{PRE_MERGE_EDGE}_1",
    }
    distances = {"cav.0": 780.0, "cav.1": 790.0}
    speeds = {"cav.0": 22.0, "cav.1": 20.0}
    base.actions = {vehicle_id: 1 for vehicle_id in live}
    state = (live, roads, lanes, distances, False, False)

    with patch("traci.vehicle.getSpeed", side_effect=lambda vid: speeds[vid]):
        _act(policy, state, DAMPING_PLUS_WALL, {vid: None for vid in live})
        actions = _act(
            policy,
            state,
            DAMPING_PLUS_WALL,
            {vid: None for vid in live},
        )

    assert actions is base.actions
    assert policy.wall_active is False
    assert policy.last_audit["abort_reason"] == "assembly_speed_mismatch_timeout"
    assert policy.last_audit["transition_reason"] == ("assembly_abort:speed_mismatch_timeout")


def test_compact_geometry_vetoes_when_full_pulse_headroom_is_unavailable():
    base = _BasePolicy()
    wall = MagicMock()
    policy = CounterfactualEscalationPolicy(
        base_policy=base,
        wall_factory=MagicMock(return_value=wall),
        wall_params={"d_lo": 200.0, "d_hi": 800.0},
        geometry_mode="compact_moving",
        planned_treatment_duration_s=30.0,
        max_longitudinal_span_m=50.0,
    )
    policy.reset(SimpleNamespace(simulation_step=0.5), seed=101)
    with patch("traci.vehicle.getSpeed", return_value=25.0):
        actions = _act(policy, _traci_state(), DAMPING_PLUS_WALL)

    assert actions is base.actions
    assert policy.last_audit["transition_reason"] == (
        "formation_veto:insufficient_kinematic_headroom"
    )
    wall.act.assert_not_called()


def test_compact_release_boundary_is_separate_from_recruitment_lower_bound():
    policy = CounterfactualEscalationPolicy(
        base_policy=_BasePolicy(),
        wall_factory=MagicMock(return_value=MagicMock()),
        wall_params={"d_lo": 200.0, "d_hi": 800.0},
        geometry_mode="compact_moving",
        planned_treatment_duration_s=30.0,
        compact_release_boundary_m=100.0,
        max_longitudinal_span_m=50.0,
    )
    policy.reset(SimpleNamespace(simulation_step=0.5), seed=101)

    with patch("traci.vehicle.getSpeed", return_value=15.0):
        assert policy._has_full_pulse_headroom("cav.in", 500.0) is True

    assert policy.last_audit == {}


def test_wall_tau_ramp_increases_headway_incrementally():
    wall = WallHoldPolicy(
        d_lo=200.0,
        d_hi=400.0,
        wall_speed=None,
        wall_tau=1.0,
        tau_ramp_step=0.05,
    )
    wall.reset(SimpleNamespace(simulation_step=0.5), seed=101)

    with (
        patch("policies.inflow_control.configure_cav"),
        patch("traci.vehicle.getIDList", return_value=["cav.in"]),
        patch("traci.vehicle.getTau", return_value=1.0),
        patch(
            "policies.inflow_control.distance_to_merge",
            return_value=300.0,
        ),
        patch("traci.vehicle.setTau") as set_tau,
    ):
        wall.act({"cav.in": None})
        wall.act({"cav.in": None})

    assert [call.args for call in set_tau.call_args_list] == [
        ("cav.in", 1.05),
        ("cav.in", 1.1),
    ]


def test_adaptive_common_target_is_latched_from_predecision_speed_gap():
    policy = CounterfactualEscalationPolicy(
        base_policy=_BasePolicy(),
        wall_factory=MagicMock(return_value=MagicMock()),
        wall_params={"d_lo": 200.0, "d_hi": 1000.0, "wall_speed": 10.0},
        geometry_mode="compact_moving",
        planned_treatment_duration_s=20.0,
        compact_control_mode="adaptive_servo",
    )
    policy.reset(SimpleNamespace(simulation_step=1.0), seed=101)
    speeds = {"cav.0": 20.0, "cav.1": 22.0, "hdv.down": 10.0}
    distances = {"cav.0": 700.0, "cav.1": 710.0, "hdv.down": 100.0}
    with (
        patch("traci.vehicle.getIDList", return_value=list(speeds)),
        patch("traci.vehicle.getRoadID", return_value=PRE_MERGE_EDGE),
        patch("traci.vehicle.getSpeed", side_effect=lambda vid: speeds[vid]),
        patch(
            "policies.counterfactual_escalation.distance_to_merge",
            side_effect=lambda vid: distances[vid],
        ),
    ):
        target = policy._freeze_adaptive_common_target(("cav.0", "cav.1"))

    assert target == 18.0
    assert policy._adaptive_reference_speed_mps == 20.0
    assert policy._adaptive_downstream_speed_mps == 10.0
    assert policy._adaptive_speed_reduction_fraction == 0.1


def test_adaptive_servo_pulls_rear_member_forward_and_front_member_back():
    policy = CounterfactualEscalationPolicy(
        base_policy=_BasePolicy(),
        wall_factory=MagicMock(return_value=MagicMock()),
        wall_params={"d_lo": 200.0, "d_hi": 1000.0},
        geometry_mode="compact_moving",
        planned_treatment_duration_s=20.0,
        compact_control_mode="adaptive_servo",
        compact_position_kp_s_inv=0.05,
        compact_position_speed_offset_cap_mps=1.0,
    )
    policy.reset(SimpleNamespace(simulation_step=1.0), seed=101)
    distances = {"cav.front": 300.0, "cav.rear": 340.0}
    with (
            patch(
                "policies.counterfactual_escalation.distance_to_merge",
                side_effect=lambda vid: distances[vid],
            ),
            patch(
                "traci.vehicle.getRoadID",
                return_value="1191554636",
            ),
            patch("traci.vehicle.getSpeed", return_value=20.0),
        patch("traci.vehicle.getLeader", return_value=None),
        patch("traci.vehicle.getFollower", return_value=None),
    ):
        targets = policy._adaptive_member_targets(("cav.front", "cav.rear"), 18.0)

    assert targets == {"cav.front": 17.0, "cav.rear": 19.0}
    assert policy._last_safety_projection["cav.front"]["position_error_m"] == -20.0
    assert policy._last_safety_projection["cav.rear"]["position_error_m"] == 20.0


def test_adaptive_servo_derivative_term_damps_relative_speed():
    policy = CounterfactualEscalationPolicy(
        base_policy=_BasePolicy(),
        wall_factory=MagicMock(return_value=MagicMock()),
        wall_params={"d_lo": 200.0, "d_hi": 1000.0},
        geometry_mode="compact_moving",
        planned_treatment_duration_s=20.0,
        compact_control_mode="adaptive_servo",
        compact_position_kp_s_inv=0.05,
        compact_position_kd=0.5,
        compact_position_speed_offset_cap_mps=5.0,
    )
    policy.reset(SimpleNamespace(simulation_step=1.0), seed=101)
    distances = {"cav.fast": 320.0, "cav.slow": 320.0}
    speeds = {"cav.fast": 22.0, "cav.slow": 18.0}
    with (
        patch(
            "policies.counterfactual_escalation.distance_to_merge",
            side_effect=lambda vid: distances[vid],
        ),
        patch("traci.vehicle.getSpeed", side_effect=lambda vid: speeds[vid]),
        patch("traci.vehicle.getLeader", return_value=None),
        patch("traci.vehicle.getFollower", return_value=None),
    ):
        targets = policy._adaptive_member_targets(("cav.fast", "cav.slow"), 20.0)

    assert targets == {"cav.fast": 19.0, "cav.slow": 21.0}
    assert policy._last_safety_projection["cav.fast"]["relative_speed_mps"] == 2.0
    assert policy._last_safety_projection["cav.slow"]["relative_speed_mps"] == -2.0


def test_adaptive_target_projection_protects_fast_closing_follower():
    policy = CounterfactualEscalationPolicy(
        base_policy=_BasePolicy(),
        wall_factory=MagicMock(return_value=MagicMock()),
        wall_params={"d_lo": 200.0, "d_hi": 1000.0},
        geometry_mode="compact_moving",
        planned_treatment_duration_s=20.0,
        compact_control_mode="adaptive_servo",
        compact_safety_min_rear_ttc_s=4.0,
    )
    policy.reset(SimpleNamespace(simulation_step=1.0), seed=101)
    speeds = {"cav.0": 20.0, "hdv.follow": 25.0}
    with (
        patch("traci.vehicle.getSpeed", side_effect=lambda vid: speeds[vid]),
        patch("traci.vehicle.getLeader", return_value=None),
        patch("traci.vehicle.getFollower", return_value=("hdv.follow", 10.0)),
    ):
        target, detail = policy._safety_project_target("cav.0", 18.0)

    assert target == 22.5
    assert detail["clip_reasons"] == ("rear_ttc_clip",)


def test_adaptive_servo_uses_frozen_deceleration_bound_during_holding():
    wall = MagicMock()
    wall.a_max_dec = 0.3
    wall.last_speed_commands_mps = {"cav.0": 19.0}
    observed_bounds = []
    wall.act_with_targets.side_effect = lambda _obs, _targets: observed_bounds.append(
        wall.a_max_dec
    )
    policy = CounterfactualEscalationPolicy(
        base_policy=_BasePolicy(),
        wall_factory=MagicMock(return_value=wall),
        wall_params={"d_lo": 200.0, "d_hi": 1000.0, "a_max_dec": 0.3},
        geometry_mode="compact_moving",
        planned_treatment_duration_s=20.0,
        compact_control_mode="adaptive_servo",
        compact_servo_a_max_dec_mps2=1.0,
    )
    policy.reset(SimpleNamespace(simulation_step=1.0), seed=101)
    policy._treatment_state = TreatmentState.HOLDING
    policy._adaptive_common_target_mps = 18.0

    with patch.object(
        policy,
        "_adaptive_member_targets",
        return_value={"cav.0": 18.0},
    ):
        policy._apply_wall_actuator({"cav.0": None})

    assert observed_bounds == [1.0]
    assert wall.a_max_dec == 0.3
    assert policy._last_wall_a_max_dec_mps2 == 1.0
    with patch.object(policy, "_vehicle_float", return_value=20.0):
        audit = policy._control_delta_audit(("cav.0",), {"cav.0": 2})
    assert audit["cav.0"]["member_applied_speed_command_mps"] == 19.0
    assert audit["cav.0"]["wall_a_max_dec_mps2"] == 1.0


def test_fixed_target_holding_keeps_base_wall_deceleration_bound():
    wall = MagicMock()
    wall.a_max_dec = 0.3
    observed_bounds = []
    wall.act.side_effect = lambda _obs: observed_bounds.append(wall.a_max_dec)
    policy = CounterfactualEscalationPolicy(
        base_policy=_BasePolicy(),
        wall_factory=MagicMock(return_value=wall),
        wall_params={"d_lo": 200.0, "d_hi": 1000.0, "a_max_dec": 0.3},
        geometry_mode="compact_moving",
        planned_treatment_duration_s=20.0,
        compact_control_mode="fixed_target",
        compact_servo_a_max_dec_mps2=1.0,
    )
    policy.reset(SimpleNamespace(simulation_step=1.0), seed=101)
    policy._treatment_state = TreatmentState.HOLDING

    policy._apply_wall_actuator({"cav.0": None})

    assert observed_bounds == [0.3]
    assert policy._last_wall_a_max_dec_mps2 == 0.3


def test_wall_actuator_accepts_distinct_rate_limited_member_targets():
    wall = WallHoldPolicy(
        d_lo=0.0,
        d_hi=1.0e9,
        wall_speed=10.0,
        wall_tau=0.0,
        a_max_dec=0.3,
    )
    wall.reset(SimpleNamespace(simulation_step=1.0), seed=101)
    speeds = {"cav.0": 20.0, "cav.1": 20.0}
    with (
        patch("policies.inflow_control.configure_cav"),
        patch("traci.vehicle.getIDList", return_value=list(speeds)),
        patch("traci.vehicle.getTau", return_value=1.0),
        patch("traci.vehicle.getSpeed", side_effect=lambda vid: speeds[vid]),
        patch("traci.vehicle.setTau"),
        patch("traci.vehicle.setSpeed") as set_speed,
    ):
        wall.act_with_targets(
            {"cav.0": None, "cav.1": None},
            {"cav.0": 18.0, "cav.1": 22.0},
        )

    assert [call.args for call in set_speed.call_args_list] == [
        ("cav.0", 19.7),
        ("cav.1", 21.0),
    ]
    assert wall.last_speed_commands_mps == {
        "cav.0": 19.7,
        "cav.1": 21.0,
    }


def test_v5_3_linear_taper_reaches_identical_base_target_and_zero_tau() -> None:
    wall = MagicMock()
    wall.a_max_dec = 0.3
    wall.set_tau_blend.return_value = []
    policy = CounterfactualEscalationPolicy(
        base_policy=_BasePolicy(),
        wall_factory=MagicMock(return_value=wall),
        wall_params={"d_lo": 200.0, "d_hi": 1000.0, "a_max_dec": 0.3},
        geometry_mode="compact_moving",
        planned_treatment_duration_s=20.0,
        compact_control_mode="adaptive_servo",
        treatment_integrity_version="v5.3",
        taper_duration_s=5.0,
        taper_function="linear",
    )
    policy.reset(SimpleNamespace(simulation_step=1.0), seed=101)
    policy._treatment_state = TreatmentState.TAPERING
    policy._taper_reason = "planned_in_horizon_release"
    policy._taper_origin_targets_mps = {"cav.0": 10.0}
    with (
        patch.object(policy, "_base_target_speed_mps", return_value=20.0),
        patch.object(
            policy,
            "_safety_project_target",
            side_effect=lambda _member, target: (target, {}),
        ),
    ):
        for _ in range(5):
            policy._apply_taper_actuator({"cav.0": None})

    final_targets = wall.act_with_targets.call_args.args[1]
    assert final_targets == {"cav.0": 20.0}
    assert policy._last_taper_blend_fraction == 0.0
    assert wall.set_tau_blend.call_args.args == (("cav.0",), 0.0)


def test_v5_4_safety_abort_during_taper_records_zero_damping_handoff() -> None:
    base = _BasePolicy()
    wall = MagicMock()
    policy = CounterfactualEscalationPolicy(
        base_policy=base,
        wall_factory=MagicMock(return_value=wall),
        min_wall_cavs=1,
        min_wall_lanes=1,
        planned_treatment_duration_s=20.0,
        treatment_integrity_version="v5.4",
        taper_duration_s=5.0,
        taper_function="linear",
    )
    policy.reset(SimpleNamespace(simulation_step=1.0), seed=101)
    _act(policy, _traci_state(), DAMPING_PLUS_WALL)
    policy._treatment_state = TreatmentState.TAPERING
    policy._taper_reason = "planned_in_horizon_release"
    policy._last_taper_blend_fraction = 0.8
    policy._last_controlled_ids = ("cav.in",)
    wall.reset_mock()

    with patch.object(policy, "_base_target_speed_mps", return_value=20.0):
        actions = _act(
            policy,
            _traci_state(hard_brake=True),
            DAMPING_PLUS_WALL,
        )

    assert actions is base.actions
    assert policy.wall_active is False
    assert policy.treatment_state == TreatmentState.RELEASED
    assert policy.last_audit["abort_reason"] == "controlled_cav_hard_brake"
    assert policy.last_audit["emergency_taper_zero_handoff"] is True
    assert policy.last_audit["taper_blend_fraction"] == 0.0
    assert policy.last_audit["wall_override_ids"] == ()
    wall.act.assert_not_called()


def test_v5_3_compact_handoff_is_limited_to_one_per_lane() -> None:
    policy = CounterfactualEscalationPolicy(
        base_policy=_BasePolicy(),
        wall_factory=MagicMock(),
        geometry_mode="compact_moving",
        planned_treatment_duration_s=20.0,
        treatment_integrity_version="v5.3",
        compact_handoff_enabled=True,
        max_handoffs_per_lane=1,
        taper_duration_s=5.0,
        taper_function="linear",
    )
    lane = f"{PRE_MERGE_EDGE}_0"
    policy._frozen_lane_ids = (lane,)
    policy._member_slots = {lane: [None]}
    policy._reserve_ids_by_lane = {lane: ["reserve.1", "reserve.2"]}
    policy._handoff_counts_by_lane = {lane: 1}
    policy._treatment_state = TreatmentState.HOLDING
    members, events, _rejected = policy._refresh_members(
        {"reserve.1", "reserve.2"}, {"reserve.1", "reserve.2"}
    )
    assert members == ()
    assert events == []


class _DirectDampingPolicy:
    def __init__(self):
        self.calls = []
        self.last_diagnostics = {}

    def reset(self, env, seed=None):
        self.env = env

    def act(self, obs):
        self.calls.append(tuple(obs))
        self.last_diagnostics = {
            "commands": [
                {"vehicle_id": "cav.in", "target_speed_mps": 18.0},
                {"vehicle_id": "cav.out", "target_speed_mps": 22.0},
            ]
        }
        return {}

    def close(self):
        pass


def test_direct_control_damping_is_audited_outside_wall_members():
    base = _DirectDampingPolicy()
    wall = MagicMock()
    policy = CounterfactualEscalationPolicy(
        base_policy=base,
        wall_factory=MagicMock(return_value=wall),
        min_wall_cavs=1,
        min_wall_lanes=1,
        cooldown_steps=0,
    )
    policy.reset(SimpleNamespace(simulation_step=1.0), seed=101)

    actions = _act(policy, _traci_state(), DAMPING_PLUS_WALL)

    assert actions == {}
    assert policy.last_audit["base_direct_control_ids"] == ("cav.in", "cav.out")
    assert policy.last_audit["wall_override_ids"] == ("cav.in",)
    assert policy.last_audit["out_of_band_damping_ids"] == ("cav.out",)
    wall.act.assert_called_once()

    actions = _act(policy, _traci_state(), DAMPING_ONLY)

    assert actions == {}
    assert len(base.calls) == 2
    assert policy.last_audit["wall_active"] is False
    assert policy.last_audit["out_of_band_damping_ids"] == ("cav.in", "cav.out")
