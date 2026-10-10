from dataclasses import replace

import numpy as np
import pytest

from env.native_damping import (
    FRAME_FIELDS,
    DampingContract,
    LocalFrame,
    LocalHistory,
    NeighborMeasurement,
    OwnIDMParameters,
    nominal_idm_acceleration,
    project_residual,
)
from idm.damping_credit import (
    TeamCostScales,
    TeamIntervalCost,
    gather_actor_advantages,
    team_advantages,
    team_reward,
)


def frame(**changes):
    return replace(
        LocalFrame("cav", 10, 30, 0, 30, 1, front_coverage_m=100, rear_coverage_m=100), **changes
    )


OWN = OwnIDMParameters(2.6, 4.5, 1.5, 2.5, 9)


def test_manifest_and_observation_order_are_explicit():
    cfg = DampingContract()
    assert cfg.observation_dim == 85
    assert cfg.manifest() == DampingContract().manifest()
    assert cfg.manifest()["sha256"] != replace(cfg, sensor_range_m=50).manifest()["sha256"]
    assert cfg.manifest()["zero_action"] == "native_passthrough_not_formula_command"


@pytest.mark.parametrize(
    "changes",
    [
        {"sensor_range_m": 0},
        {"history_frames": 1.2},
        {"max_age_s": -1},
        {"residual_max": -0.1},
        {"jerk_limit": float("nan")},
    ],
)
def test_invalid_contract_rejected(changes):
    with pytest.raises(ValueError):
        DampingContract(**changes)


def test_history_missing_frames_and_reads_do_not_advance_state():
    history = LocalHistory()
    history.add(frame(timestamp_s=8), now_s=8)
    history.add(frame(timestamp_s=10), now_s=10)
    history.add(frame(timestamp_s=10), now_s=10)  # exact duplicate is idempotent
    before = history.vector("cav", now_s=10)
    np.testing.assert_array_equal(before, history.vector("cav", now_s=10))
    valid = before.reshape(5, -1)[:, FRAME_FIELDS.index("frame_valid")]
    np.testing.assert_array_equal(valid, [0, 0, 1, 0, 1])
    assert not history.vector("unknown", now_s=10).any()


def test_history_identity_and_absolute_time_are_not_features():
    first, second = LocalHistory(), LocalHistory()
    first.add(frame(), now_s=10)
    second.add(frame(vehicle_id="another_id", timestamp_s=1010), now_s=1010)
    np.testing.assert_array_equal(
        first.vector("cav", now_s=10), second.vector("another_id", now_s=1010)
    )


def test_future_conflicting_and_out_of_range_measurements_rejected():
    history = LocalHistory()
    with pytest.raises(ValueError, match="future"):
        history.add(frame(), now_s=9)
    history.add(frame(), now_s=10)
    with pytest.raises(ValueError, match="conflicting"):
        history.add(frame(speed_mps=20), now_s=10)
    with pytest.raises(ValueError):
        history.vector("cav", now_s=9)
    with pytest.raises(ValueError, match="coverage"):
        frame(front=NeighborMeasurement(110, 0))


def test_no_detection_and_no_coverage_are_distinct():
    h = LocalHistory()
    h.add(frame(), now_s=10)
    observed_clear = h.vector("cav", now_s=10)
    other = LocalHistory()
    other.add(frame(front_coverage_m=0), now_s=10)
    assert not np.array_equal(observed_clear, other.vector("cav", now_s=10))
    assert project_residual(frame(front_coverage_m=0), OWN, -0.2, now_s=10).target_speed is None


def test_nominal_idm_is_not_claimed_as_native_zero_command():
    state = frame(speed_mps=10)
    assert nominal_idm_acceleration(state, OWN) > 2
    result = project_residual(state, OWN, 0, now_s=10)
    assert result.target_speed is None
    assert result.reasons == ("zero_native_passthrough",)


def test_free_cruise_braking_authority_survives_without_selector_or_filter():
    result = project_residual(frame(), OWN, -0.2, now_s=10)
    assert result.target_acceleration == pytest.approx(-0.2)
    assert result.target_speed == pytest.approx(29.8)
    assert result.projected_delta == pytest.approx(-0.2)


def test_positive_authority_survives_when_physical_interval_permits():
    state = frame(speed_mps=29.9, acceleration_mps2=0.04)
    result = project_residual(state, OWN, 0.05, now_s=10)
    assert result.target_acceleration is not None
    assert result.projected_delta == pytest.approx(0.05)


def test_jerk_and_residual_bounds_intersect_without_action_reversal():
    result = project_residual(frame(acceleration_mps2=0.3), OWN, -0.5, now_s=10)
    assert result.target_acceleration == pytest.approx(-0.2)
    assert "safety_projection" in result.reasons
    result = project_residual(frame(), OWN, -99, now_s=10)
    assert result.bounded_delta == -0.5
    assert "residual_clip" in result.reasons


def test_conflicting_front_rear_constraints_yield_native_not_unsafe_target():
    state = frame(front=NeighborMeasurement(5, -15), rear=NeighborMeasurement(1, 10))
    result = project_residual(state, OWN, -0.2, now_s=10)
    assert result.target_speed is None
    assert "infeasible_command_interval" in result.reasons


def test_rear_closing_vehicle_prevents_extra_braking():
    result = project_residual(frame(rear=NeighborMeasurement(4, 1)), OWN, -0.2, now_s=10)
    assert result.target_speed is None


@pytest.mark.parametrize("action", [float("nan"), float("inf")])
def test_invalid_actions_fail_instead_of_becoming_training_samples(action):
    with pytest.raises(ValueError):
        project_residual(frame(), OWN, action, now_s=10)


def test_stale_measurement_yields_native():
    result = project_residual(frame(), OWN, -0.2, now_s=12)
    assert result.reasons == ("stale_observation",)


def test_late_team_harm_reaches_actor_that_has_already_left():
    advantages, returns = team_advantages([0, 0, -3], [0, 0, 0, 0], [False, False, True])
    mask = np.array([[1, 0], [0, 1], [0, 0]], dtype=bool)
    credited = gather_actor_advantages(advantages, mask, normalize=False)
    assert credited[0, 0] == -3
    assert credited[1, 1] == -3
    assert credited[2].sum() == 0
    np.testing.assert_array_equal(returns, [-3, -3, -3])


def test_team_ends_stop_credit_but_sampling_chunks_bootstrap():
    a, _ = team_advantages([1, 2], [0, 0, 9], [True, False])
    np.testing.assert_array_equal(a, [1, 11])


def test_constant_delayed_cost_is_not_erased_by_normalization():
    result = gather_actor_advantages([-3, -3], [[True], [True]])
    np.testing.assert_array_equal(result, [[-3], [-3]])


def test_actor_padding_and_population_do_not_change_team_normalization():
    mask = np.array([[1], [1], [0]], dtype=bool)
    first = gather_actor_advantages([1, 3, 100], mask)
    expanded = np.pad(mask, ((0, 0), (0, 4)))
    expanded[0, 1] = True
    second = gather_actor_advantages([1, 3, 100], expanded)
    np.testing.assert_array_equal(first[:, 0], second[:, 0])
    assert np.all(second[:, 2:] == 0)
    assert second[0, 1] == second[0, 0]


def test_physical_reward_components_and_source_queue_cost_cannot_hide_harm():
    scales = TeamCostScales(10, 100, 1000)
    baseline = team_reward(TeamIntervalCost(2, 20, 10), scales)
    more_braking = team_reward(TeamIntervalCost(4, 20, 10), scales)
    source_backlog = team_reward(TeamIntervalCost(2, 20, 110), scales)
    collision = team_reward(TeamIntervalCost(2, 20, 10, 1), scales)
    assert more_braking["reward"] < baseline["reward"]
    assert source_backlog["reward"] < baseline["reward"]
    assert collision["reward"] < baseline["reward"] - 1.9
    assert baseline["braking"] == baseline["energy"] == -0.1


def test_invalid_credit_shapes_and_nonfinite_inputs_rejected():
    with pytest.raises(ValueError):
        team_advantages([0], [0], [False])
    with pytest.raises(ValueError):
        team_advantages([float("nan")], [0, 0], [True])
    with pytest.raises(ValueError):
        gather_actor_advantages([1], [[0.5]])
