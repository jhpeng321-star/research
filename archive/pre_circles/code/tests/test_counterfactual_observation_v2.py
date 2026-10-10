from config.constants import DOWNSTREAM_EDGE, PRE_MERGE_EDGE
from counterfactual.outcomes import OutcomeDefinition
from counterfactual.pilot import observation_v2_features
from counterfactual.twin_replay import ReplayTrace, StepSnapshot


def _snapshot(step: int, vehicles: tuple[tuple, ...]) -> StepSnapshot:
    return StepSnapshot(
        relative_step=step,
        sim_time=float(step),
        action=0,
        supervisor_observation=tuple([0.0] * 16),
        vehicles=vehicles,
        active_cavs=("c0",),
        collision_vehicle_ids=(),
        starting_teleport_vehicle_ids=(),
        digest=f"d{step}",
        vehicle_count=len(vehicles),
        mean_speed=10.0,
        active_time_loss_sum=0.0,
    )


def _row(vehicle_id: str, road: str, lane: int, position: float, speed: float, accel=0.0):
    return (vehicle_id, road, f"{road}_{lane}", position, speed, accel, 0.0, 0, "car", 1.0)


def _trace() -> ReplayTrace:
    snapshots = (
        _snapshot(0, (_row("c0", PRE_MERGE_EDGE, 0, 900.0, 20.0),)),
        _snapshot(
            1,
            (
                _row("c0", PRE_MERGE_EDGE, 0, 920.0, 18.0),
                _row("d0", DOWNSTREAM_EDGE, 0, 10.0, 10.0),
            ),
        ),
        _snapshot(
            2,
            (
                _row("c0", PRE_MERGE_EDGE, 0, 940.0, 16.0),
                _row("d0", DOWNSTREAM_EDGE, 0, 20.0, 8.0, -5.0),
                _row("d1", DOWNSTREAM_EDGE, 0, 5.0, 12.0),
            ),
        ),
        _snapshot(
            3,
            (
                _row("c0", PRE_MERGE_EDGE, 0, 960.0, 14.0),
                _row("d0", DOWNSTREAM_EDGE, 0, 28.0, 8.0),
                _row("d1", DOWNSTREAM_EDGE, 0, 17.0, 12.0),
            ),
        ),
    )
    return ReplayTrace(
        seed=1,
        decision_step=3,
        horizon_steps=3,
        prefix_action=0,
        post_action=0,
        treatment_steps=None,
        reset_info={},
        snapshots=snapshots,
        terminated_early=False,
    )


def test_observation_v2_is_predecision_and_has_postmerge_early_warning_and_dose() -> None:
    features = observation_v2_features(
        _trace(),
        decision_step=2,
        definition=OutcomeDefinition(),
        dose_context={
            "proposed_wall_speed_mps": 10.0,
            "release_boundary_m": 50.0,
            "selected_member_ids": ["c0"],
        },
    )
    assert features["obs_v2_postmerge_vehicle_count"] == 2.0
    assert features["obs_v2_postmerge_hard_brake_count"] == 1.0
    assert features["obs_v2_postmerge_discharge_rate_20s_vph"] == 360.0
    assert features["obs_v2_dose_proposed_wall_speed_mps"] == 10.0
    assert features["obs_v2_dose_release_boundary_margin_m"] > 0.0
    assert "seed" not in features
    # The future step 3 removes the hard brake; decision-step extraction must
    # still retain the step-2 value and therefore cannot have read the future.
    assert features["obs_v2_postmerge_hard_brake_count"] == 1.0


def test_observation_v2_gap_availability_fails_closed_to_numeric_zero() -> None:
    features = observation_v2_features(
        _trace(), decision_step=1, definition=OutcomeDefinition()
    )
    assert features["obs_v2_premerge_gap_available_20s"] == 0.0
    assert features["obs_v2_premerge_gap_lag1_autocorr_20s"] == 0.0
