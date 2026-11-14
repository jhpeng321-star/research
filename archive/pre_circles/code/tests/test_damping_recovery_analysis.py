from math import exp
import numpy as np
import pytest

from idm.damping_recovery_analysis import (
    constant_leader_visibility_margin,
    large_gap_acceleration,
    margin_from_local_observation,
)


@pytest.mark.parametrize("correction", [0.0, 0.3, -0.1])
def test_derived_asymptote_matches_independent_ema_and_speed_recurrence(correction):
    speed = filtered = 10.0
    q = exp(-1 / 5)
    for _ in range(600):
        acceleration = (filtered + 2 - speed) / 5 + correction
        speed += acceleration
        filtered = q * filtered + (1 - q) * speed
    assert acceleration == pytest.approx(
        large_gap_acceleration(correction_mps2=correction), abs=1e-11
    )
    assert acceleration < 0.5


def test_visibility_boundary_matches_the_extremum_of_relative_motion():
    result = constant_leader_visibility_margin(gap_m=64, ego_speed_mps=10, leader_speed_mps=16)
    time_to_match = 6 / 0.5
    max_gap = 64 + 6 * time_to_match - 0.5 * 0.5 * time_to_match**2
    assert max_gap == 100
    assert result.margin_m == 0 and result.required_acceleration_mps2 == 0.5
    late = constant_leader_visibility_margin(gap_m=65, ego_speed_mps=10, leader_speed_mps=16)
    assert not late.constant_leader_retention_possible_under_acceleration_bound
    assert late.margin_m == -1


def test_missing_leader_is_unknown_and_existing_state_contains_the_margin():
    x = np.zeros(89)
    x[68] = 10 / 30
    x[75] = 0.64
    x[76] = -6 / 30
    x[77] = 1
    x[78] = 1
    x[84] = 1
    assert margin_from_local_observation(x)["margin_m"] == pytest.approx(0)
    x[77] = 0
    assert margin_from_local_observation(x) is None


def test_invalid_or_out_of_range_inputs_fail_instead_of_becoming_safe():
    with pytest.raises(ValueError):
        constant_leader_visibility_margin(gap_m=101, ego_speed_mps=10, leader_speed_mps=15)
    with pytest.raises(ValueError):
        constant_leader_visibility_margin(gap_m=50, ego_speed_mps=float("nan"), leader_speed_mps=15)
    edge = constant_leader_visibility_margin(gap_m=100, ego_speed_mps=10, leader_speed_mps=15)
    assert edge.required_acceleration_mps2 is None
    assert edge.margin_m < 0
