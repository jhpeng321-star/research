from config.constants import MERGE_EDGE, PRE_MERGE_EDGE, PRE_MERGE_LANE_LENGTH
from counterfactual.outcomes import (
    OutcomeDefinition,
    summarize_branch,
    summarize_predecision_congestion,
)
from counterfactual.pilot import (
    features_from_snapshot,
    features_from_trace,
    find_decision_step,
    placement_features,
)
from counterfactual.twin_replay import ReplayTrace, StepSnapshot


def _row(
    vehicle_id,
    road,
    lane,
    position,
    speed,
    acceleration,
    time_loss,
):
    return (
        vehicle_id,
        road,
        lane,
        position,
        speed,
        acceleration,
        time_loss,
        0,
        "car",
    )


def _snapshot(
    step,
    rows,
    *,
    health=0.4,
    trend=-0.1,
    inband=0.2,
    action=1,
    active_cavs=("a",),
):
    observation = (
        health,
        trend,
        0.4,
        0.5,
        0.1,
        0.2,
        inband,
        0.5,
        0.0,
        0.0,
        0.1,
        0.3,
        0.5,
        0.6,
        0.7,
        0.8,
    )
    return StepSnapshot(
        relative_step=step,
        sim_time=step * 0.5,
        action=action,
        supervisor_observation=observation,
        vehicles=tuple(rows),
        active_cavs=tuple(active_cavs),
        collision_vehicle_ids=(),
        starting_teleport_vehicle_ids=(),
        digest=f"step-{step}",
        vehicle_count=len(rows),
        mean_speed=sum(row[4] for row in rows) / len(rows),
        active_time_loss_sum=sum(row[6] for row in rows),
    )


def _trace(snapshots, *, controller_audits=()):
    return ReplayTrace(
        seed=101,
        decision_step=0,
        horizon_steps=len(snapshots) - 1,
        prefix_action=0,
        post_action=1,
        treatment_steps=2,
        reset_info={},
        snapshots=tuple(snapshots),
        terminated_early=False,
        controller_audits=tuple(controller_audits),
    )


def test_outcomes_use_future_traffic_not_health_score():
    snapshots = []
    for step in range(4):
        rows = [
            _row(
                "a", PRE_MERGE_EDGE, PRE_MERGE_EDGE + "_0", 10, 14, -5 if step == 1 else 0, 1 + step
            ),
            _row("b", PRE_MERGE_EDGE, PRE_MERGE_EDGE + "_0", 20, 10, 0, step),
            _row(
                "c",
                MERGE_EDGE if step == 0 else "downstream",
                (MERGE_EDGE if step == 0 else "downstream") + "_0",
                5,
                15,
                0,
                0,
            ),
        ]
        snapshots.append(_snapshot(step, rows))

    result = summarize_branch(
        _trace(snapshots),
        definition=OutcomeDefinition(
            congestion_queue_count=2,
            sustain_steps=3,
            queue_speed_threshold_mps=15,
            congestion_speed_threshold_mps=15,
        ),
    )

    assert result["sustained_congestion"] is True
    assert result["congestion_onset_step"] == 0
    assert result["queue_area_vehicle_s"] == 3.0
    assert result["incremental_delay_s"] == 6.0
    assert result["merge_discharge_count"] == 1
    assert result["hard_brake_vehicle_steps"] == 1
    assert result["low_ttc_exposure_steps"] == 3


def test_predecision_congestion_uses_no_postdecision_snapshots():
    snapshots = []
    for step in range(6):
        speed = 10.0 if step < 3 else 30.0
        rows = [
            _row("a", PRE_MERGE_EDGE, PRE_MERGE_EDGE + "_0", 10, speed, 0, 0),
            _row("b", PRE_MERGE_EDGE, PRE_MERGE_EDGE + "_0", 20, speed, 0, 0),
        ]
        snapshots.append(_snapshot(step, rows))
    definition = OutcomeDefinition(
        congestion_queue_count=2,
        sustain_steps=3,
        queue_speed_threshold_mps=15.0,
        congestion_speed_threshold_mps=15.0,
    )

    before_persistence = summarize_predecision_congestion(
        _trace(snapshots),
        decision_step=1,
        definition=definition,
    )
    after_persistence = summarize_predecision_congestion(
        _trace(snapshots),
        decision_step=2,
        definition=definition,
    )

    assert before_persistence["sustained_congestion"] is False
    assert after_persistence["sustained_congestion"] is True
    assert after_persistence["congestion_onset_step"] == 0


def test_decision_features_encode_health_path_and_action_availability():
    snapshots = [
        _snapshot(step, [_row("a", PRE_MERGE_EDGE, PRE_MERGE_EDGE + "_0", 10, 12, 0, 0)])
        for step in range(11)
    ]
    trace = _trace(snapshots)

    assert find_decision_step(trace, min_step=10) == 10
    features = features_from_snapshot(snapshots[-1], OutcomeDefinition())
    assert features["health_path_chronological"] == [0.8, 0.7, 0.6, 0.5, 0.4]
    assert features["health_falling_segments"] == 4
    assert features["decision_active_cav_pre_merge_lane_count"] == 1

    temporal = features_from_trace(
        trace,
        decision_step=10,
        definition=OutcomeDefinition(),
    )
    assert temporal["traffic_lag_5_actual_step"] == 5
    assert temporal["traffic_lag_60_actual_step"] == 0
    assert temporal["pattern_window_start_step"] == 0
    assert temporal["pattern60_queue_count_mean"] == 1.0


def test_hard_brake_attribution_identifies_hdv_following_wall_cav():
    snapshots = []
    for step in range(4):
        rows = [
            _row(
                "follower",
                PRE_MERGE_EDGE,
                PRE_MERGE_EDGE + "_0",
                PRE_MERGE_LANE_LENGTH - 310.0,
                20.0,
                -5.0 if step in (1, 2) else 0.0,
                0.0,
            ),
            _row(
                "leader",
                PRE_MERGE_EDGE,
                PRE_MERGE_EDGE + "_0",
                PRE_MERGE_LANE_LENGTH - 300.0,
                10.0,
                0.0,
                0.0,
            ),
        ]
        snapshots.append(
            _snapshot(step, rows, action=1 if step <= 2 else 0, active_cavs=("leader",))
        )

    result = summarize_branch(_trace(snapshots), definition=OutcomeDefinition())
    attribution = result["hard_brake_attribution"]

    assert attribution["exposure_vehicle_steps"] == 2
    assert attribution["event_start_count"] == 1
    assert attribution["exposure_counts"]["vehicle_class"] == {"hdv": 2}
    assert attribution["exposure_counts"]["phase"] == {"treatment_window": 2}
    assert attribution["exposure_context_counts"]["leader_controlled_wall_cav"] == 2
    assert attribution["event_start_context_counts"]["leader_controlled_wall_cav"] == 1
    assert attribution["event_starts"][0]["leader_gap_m"] == 5.0
    assert attribution["event_starts"][0]["leader_ttc_s"] == 0.5


def test_low_ttc_attribution_uses_exact_wall_ids_and_treatment_phase():
    snapshots = []
    for step in range(4):
        follower_speed = 10.0 if step == 0 else 20.0
        rows = [
            _row(
                "follower",
                PRE_MERGE_EDGE,
                PRE_MERGE_EDGE + "_0",
                PRE_MERGE_LANE_LENGTH - 910.0,
                follower_speed,
                0.0,
                0.0,
            ),
            _row(
                "leader",
                PRE_MERGE_EDGE,
                PRE_MERGE_EDGE + "_0",
                PRE_MERGE_LANE_LENGTH - 900.0,
                10.0,
                0.0,
                0.0,
            ),
        ]
        snapshots.append(
            _snapshot(
                step,
                rows,
                action=1 if step <= 2 else 0,
                active_cavs=("leader",),
            )
        )
    audits = (
        {"relative_step": 0, "treatment_state": "DAMPING"},
        {
            "relative_step": 1,
            "treatment_state": "HOLDING",
            "wall_override_ids": ("leader",),
        },
        {
            "relative_step": 2,
            "treatment_state": "HOLDING",
            "wall_override_ids": ("leader",),
        },
        {
            "relative_step": 3,
            "treatment_state": "COOLDOWN",
            "wall_override_ids": (),
        },
    )

    result = summarize_branch(
        _trace(snapshots, controller_audits=audits),
        definition=OutcomeDefinition(),
    )
    attribution = result["low_ttc_attribution"]

    assert attribution["exposure_pair_steps"] == 3
    assert attribution["event_start_count"] == 1
    assert attribution["exposure_counts"]["phase"] == {
        "cooldown": 1,
        "holding": 2,
    }
    assert attribution["exposure_context_counts"]["leader_controlled_wall_cav"] == 2
    assert attribution["event_start_context_counts"]["leader_controlled_wall_cav"] == 1
    assert attribution["events"][0]["gap_m"] == 5.0
    assert attribution["events"][0]["ttc_s"] == 0.5


def test_placement_features_encode_realized_penetration_and_wall_geometry():
    rows = [
        _row("c0", PRE_MERGE_EDGE, PRE_MERGE_EDGE + "_0", PRE_MERGE_LANE_LENGTH - 300, 10, 0, 0),
        _row("c1", PRE_MERGE_EDGE, PRE_MERGE_EDGE + "_1", PRE_MERGE_LANE_LENGTH - 500, 10, 0, 0),
        _row("c2", PRE_MERGE_EDGE, PRE_MERGE_EDGE + "_2", PRE_MERGE_LANE_LENGTH - 900, 10, 0, 0),
        _row("h0", PRE_MERGE_EDGE, PRE_MERGE_EDGE + "_0", PRE_MERGE_LANE_LENGTH - 100, 10, 0, 0),
    ]
    features = placement_features(_snapshot(0, rows, active_cavs=("c0", "c1", "c2")))

    assert features["placement_network_cav_fraction"] == 0.75
    assert features["placement_pre_merge_cav_fraction"] == 0.75
    assert features["placement_longitudinal_span_m"] == 600.0
    assert features["placement_soft_band_cav_count"] == 1
    assert features["placement_soft_band_lane_coverage_fraction"] == 1 / 3
    assert features["placement_strong_band_cav_count"] == 2
    assert features["placement_strong_band_lane_coverage_fraction"] == 2 / 3
