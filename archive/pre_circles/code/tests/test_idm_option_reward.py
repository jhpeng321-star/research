"""Pure tests for the fixed-horizon paired binary option reward."""

from __future__ import annotations

import inspect
import math

import pytest

from idm.option_reward import (
    IDM_BINARY_OPTION_REWARD_V1,
    BinaryOptionRewardContract,
    binary_option_reward,
    paired_constraints,
)


def _branch(**updates):
    row = {
        "completed_horizon_steps": 60,
        "target_command_step_count": 20,
        "unexpected_command_count": 0,
        "collision_observations": 0,
        "teleport_observations": 0,
        "low_ttc_exposure_steps": 0,
        "mean_corridor_delay_rate": 0.1,
        "arrived_vehicles": 40,
        "merge_discharge_count": 38,
        "severe_braking_count_sum": 12.0,
        "acceleration_energy_sum": 600.0,
    }
    row.update(updates)
    return row


def _reward(control, action):
    return binary_option_reward(
        control,
        action,
        severe_reference_per_step=0.2,
        energy_reference_per_step=10.0,
    )


def test_reward_is_exact_centered_safety_effect_followed_by_tanh():
    control = _branch()
    action = _branch(severe_braking_count_sum=6.0, acceleration_energy_sum=480.0)
    result = _reward(control, action)
    assert result["severe_braking_effect_reference_units"] == pytest.approx(0.5)
    assert result["energy_effect_reference_units"] == pytest.approx(0.2)
    assert result["physical_utility"] == pytest.approx(0.35)
    assert result["activate_reward"] == pytest.approx(math.tanh(0.35))
    assert result["abstain_reward"] == 0.0
    assert result["constraints_pass"] is True


def test_identical_abstain_replay_has_exact_zero_reward():
    control = _branch()
    result = _reward(control, dict(control))
    assert result["physical_utility"] == 0.0
    assert result["activate_reward"] == 0.0


def test_helpful_and_harmful_effects_have_the_correct_sign():
    control = _branch()
    helpful = _reward(control, _branch(severe_braking_count_sum=10.0))
    harmful = _reward(control, _branch(severe_braking_count_sum=14.0))
    assert helpful["activate_reward"] > 0.0
    assert harmful["activate_reward"] < 0.0


def test_service_values_do_not_shape_reward_inside_the_permitted_bound():
    control = _branch()
    nominal = _branch(severe_braking_count_sum=10.0)
    changed = dict(nominal)
    changed.update(
        mean_corridor_delay_rate=1.09,
        arrived_vehicles=39,
        merge_discharge_count=37,
    )
    assert _reward(control, nominal)["activate_reward"] == _reward(control, changed)[
        "activate_reward"
    ]


def test_variable_lifetime_agent_count_queue_and_old_returns_are_ignored():
    control = _branch()
    action = _branch(severe_braking_count_sum=10.0)
    contaminated = dict(action)
    contaminated.update(
        selected_cav_lifetime=1,
        selected_agent_steps=1,
        agent_count=10_000,
        queue_area=1e9,
        role_reward_sum=-1e12,
        all_agent_reward_sum=1e12,
        seed=999999,
        decision_step=999999,
    )
    assert _reward(control, action) == _reward(control, contaminated)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("completed_horizon_steps", 59),
        ("target_command_step_count", 15),
        ("unexpected_command_count", 1),
        ("collision_observations", 1),
        ("teleport_observations", 1),
        ("low_ttc_exposure_steps", 3),
        ("mean_corridor_delay_rate", 1.1000001),
        ("arrived_vehicles", 38),
        ("merge_discharge_count", 36),
    ),
)
def test_each_constraint_failure_overrides_positive_physics(field, value):
    control = _branch()
    action = _branch(
        severe_braking_count_sum=0.0,
        acceleration_energy_sum=0.0,
        **{field: value},
    )
    result = _reward(control, action)
    assert result["physical_utility"] > 0.0
    assert result["constraints_pass"] is False
    assert result["activate_reward"] == -2.0


def test_treatment_safety_abort_is_a_hard_failure_when_diagnostic_exists():
    control = _branch()
    action = _branch(
        severe_braking_count_sum=0.0,
        acceleration_energy_sum=0.0,
        treatment_safety_abort_rows=[{"reason": "projection_interval_conflict"}],
    )
    assert paired_constraints(control, action)["zero_treatment_safety_aborts"] is False
    assert _reward(control, action)["activate_reward"] == -2.0


def test_reward_api_has_no_lifetime_agent_count_or_queue_input():
    parameters = set(inspect.signature(binary_option_reward).parameters)
    forbidden = {
        "selected_cav_lifetime",
        "agent_count",
        "queue_area",
        "seed",
        "decision_step",
        "recurrence",
    }
    assert parameters.isdisjoint(forbidden)


def test_contract_rejects_noncausal_activation_cost_and_bad_weights():
    with pytest.raises(ValueError, match="zero activation cost"):
        BinaryOptionRewardContract(activation_cost=0.01)
    with pytest.raises(ValueError, match="sum to one"):
        BinaryOptionRewardContract(severe_weight=0.7, energy_weight=0.7)


def test_references_must_be_positive_and_finite():
    with pytest.raises(ValueError, match="severe reference"):
        binary_option_reward(
            _branch(),
            _branch(),
            severe_reference_per_step=0.0,
            energy_reference_per_step=10.0,
        )
    with pytest.raises(ValueError, match="energy reference"):
        binary_option_reward(
            _branch(),
            _branch(),
            severe_reference_per_step=0.2,
            energy_reference_per_step=float("nan"),
        )


def test_frozen_contract_is_counterfactual_and_fixed_horizon():
    assert IDM_BINARY_OPTION_REWARD_V1.counterfactual_control_required is True
    assert IDM_BINARY_OPTION_REWARD_V1.horizon_steps == 60
    assert IDM_BINARY_OPTION_REWARD_V1.option_steps == 20
    assert IDM_BINARY_OPTION_REWARD_V1.minimum_dose_steps == 16
