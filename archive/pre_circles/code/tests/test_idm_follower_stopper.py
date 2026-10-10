"""Pure tests for the modified FollowerStopper-shaped ceiling."""

from __future__ import annotations

from dataclasses import replace

import pytest

from idm.follower_stopper import (
    MODIFIED_FOLLOWER_STOPPER_P30_V1,
    modified_follower_stopper_speed,
)


def _result(gap: float):
    return modified_follower_stopper_speed(
        ego_speed_mps=5.0,
        reference_speed_mps=8.0,
        leader_gap_m=gap,
        leader_speed_mps=4.0,
    )


def test_modified_boundaries_include_closing_energy_and_time_headway():
    result = _result(20.0)
    assert result.relative_speed_mps == pytest.approx(-1.0)
    assert result.closing_speed_mps == pytest.approx(-1.0)
    assert result.switching_boundaries_m == pytest.approx(
        (
            4.5 + 1.0 / 3.0 + 0.4 * 5.0,
            5.25 + 0.5 + 1.2 * 5.0,
            6.0 + 1.0 + 1.8 * 5.0,
        )
    )


def test_piecewise_speed_is_continuous_and_bounded_by_reference():
    boundaries = _result(20.0).switching_boundaries_m
    assert boundaries is not None
    x1, x2, x3 = boundaries
    assert _result(x1).desired_speed_mps == pytest.approx(0.0)
    assert _result(0.5 * (x1 + x2)).desired_speed_mps == pytest.approx(2.0)
    assert _result(x2).desired_speed_mps == pytest.approx(4.0)
    assert _result(0.5 * (x2 + x3)).desired_speed_mps == pytest.approx(6.0)
    assert _result(x3).desired_speed_mps == pytest.approx(8.0)
    assert _result(x3 + 1.0).desired_speed_mps == pytest.approx(8.0)


def test_missing_leader_is_an_unconstrained_reference_ceiling():
    result = modified_follower_stopper_speed(
        ego_speed_mps=25.0,
        reference_speed_mps=21.0,
        leader_gap_m=None,
        leader_speed_mps=None,
    )
    assert result.desired_speed_mps == pytest.approx(21.0)
    assert result.zone == "free"
    assert result.switching_boundaries_m is None


def test_positive_relative_speed_does_not_expand_switching_boundaries():
    result = modified_follower_stopper_speed(
        ego_speed_mps=20.0,
        reference_speed_mps=25.0,
        leader_gap_m=100.0,
        leader_speed_mps=22.0,
    )
    assert result.closing_speed_mps == 0.0
    assert result.switching_boundaries_m == pytest.approx((12.5, 29.25, 42.0))


def test_invalid_contract_fails_closed():
    invalid = replace(
        MODIFIED_FOLLOWER_STOPPER_P30_V1,
        closing_decelerations_mps2=(1.5, 0.0, 0.5),
    )
    with pytest.raises(ValueError, match="must be positive"):
        modified_follower_stopper_speed(
            ego_speed_mps=20.0,
            reference_speed_mps=15.0,
            leader_gap_m=20.0,
            leader_speed_mps=10.0,
            contract=invalid,
        )
