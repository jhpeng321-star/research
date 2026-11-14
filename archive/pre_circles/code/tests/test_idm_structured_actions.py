import pytest

from idm.structured_actions import (
    adjusted_reference_speed,
    clip_action,
    gain_desired_acceleration,
    low_pass_action,
    nominal_action,
)


def test_nominal_actions_preserve_each_family():
    assert nominal_action("residual") == 0.0
    assert nominal_action("gain") == 1.0
    assert nominal_action("reference_offset") == 0.0


def test_structured_actions_use_frozen_bounds():
    assert clip_action("gain", 2.0, residual_min_mps2=-0.5, residual_max_mps2=0.3) == 1.5
    assert (
        clip_action(
            "reference_offset", -3.0, residual_min_mps2=-0.5, residual_max_mps2=0.3
        )
        == -2.0
    )
    assert (
        clip_action("residual", 0.4, residual_min_mps2=-0.5, residual_max_mps2=0.3)
        == 0.3
    )


def test_low_pass_transition_is_causal_and_returns_toward_nominal():
    filtered = low_pass_action(1.0, 1.5, dt_s=1.0, tau_s=5.0)
    assert filtered == pytest.approx(1.0 + 0.5 / 6.0)
    assert low_pass_action(filtered, 1.0, dt_s=1.0, tau_s=5.0) < filtered


def test_gain_and_reference_transform_only_analytical_request():
    assert gain_desired_acceleration(-0.4, 1.25) == -0.5
    assert adjusted_reference_speed(9.0, -2.0, floor_mps=8.0, allowed_speed_mps=30.0) == 8.0
    assert adjusted_reference_speed(29.0, 2.0, floor_mps=8.0, allowed_speed_mps=30.0) == 30.0
