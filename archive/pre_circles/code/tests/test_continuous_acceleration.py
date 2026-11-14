"""Tests for continuous acceleration actions used by Selfish and Flat-MARL."""

from unittest.mock import MagicMock, patch

from env.actions import (
    _as_scalar_acceleration,
    RL_SPEED_MODE,
    apply_continuous_acceleration,
)


def test_continuous_acceleration_is_clamped_to_action_range():
    assert _as_scalar_acceleration([-10.0]) == -3.0
    assert _as_scalar_acceleration([10.0]) == 3.0
    assert _as_scalar_acceleration([1.25]) == 1.25


def test_continuous_acceleration_commands_integrated_speed():
    traci = MagicMock()
    traci.vehicle.getIDList.return_value = ["cav_1"]
    traci.vehicle.getSpeed.return_value = 10.0

    with patch("env.actions.traci", traci):
        apply_continuous_acceleration("cav_1", [1.5], dt=1.0, max_speed=35.0)

    traci.vehicle.setSpeedMode.assert_called_once_with("cav_1", RL_SPEED_MODE)
    traci.vehicle.setSpeed.assert_called_once_with("cav_1", 11.5)


def test_continuous_acceleration_respects_speed_bounds():
    traci = MagicMock()
    traci.vehicle.getIDList.return_value = ["cav_1"]
    traci.vehicle.getSpeed.return_value = 34.0

    with patch("env.actions.traci", traci):
        apply_continuous_acceleration("cav_1", [3.0], dt=1.0, max_speed=35.0)

    traci.vehicle.setSpeed.assert_called_once_with("cav_1", 35.0)
