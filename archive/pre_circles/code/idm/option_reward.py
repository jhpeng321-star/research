"""Fixed-horizon paired reward for decentralized IDM activation options.

The ordinary multi-agent environment cannot supply this reward because several
constraints and the physical utility are defined against an identical abstain
branch.  A paired option wrapper must call these pure functions after both
60-step branches complete.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from idm.action_local import safety_composite


SCHEMA = "idm_binary_option_reward/1"


@dataclass(frozen=True)
class BinaryOptionRewardContract:
    horizon_steps: int = 60
    option_steps: int = 20
    minimum_dose_steps: int = 16
    severe_weight: float = 0.5
    energy_weight: float = 0.5
    hard_failure_reward: float = -2.0
    deadband: float = 0.02
    substantial_effect: float = 0.06
    low_ttc_excess_budget_steps: int = 2
    maximum_delay_rate_regression: float = 1.0
    maximum_arrival_loss: int = 1
    maximum_discharge_loss: int = 1
    counterfactual_control_required: bool = True
    activation_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.horizon_steps <= 0 or self.option_steps <= 0:
            raise ValueError("reward horizons must be positive")
        if not 0 < self.minimum_dose_steps <= self.option_steps:
            raise ValueError("minimum dose must lie within the option")
        if self.severe_weight < 0.0 or self.energy_weight < 0.0:
            raise ValueError("physical weights must be nonnegative")
        if not math.isclose(
            self.severe_weight + self.energy_weight, 1.0, abs_tol=1e-12
        ):
            raise ValueError("physical weights must sum to one")
        if self.hard_failure_reward >= -1.0:
            raise ValueError("hard-failure reward must rank below tanh utility")
        if self.activation_cost != 0.0:
            raise ValueError("version 1 freezes zero activation cost")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, **asdict(self)}


IDM_BINARY_OPTION_REWARD_V1 = BinaryOptionRewardContract()


def _number(row: Mapping[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def paired_constraints(
    control: Mapping[str, Any],
    action: Mapping[str, Any],
    contract: BinaryOptionRewardContract = IDM_BINARY_OPTION_REWARD_V1,
) -> dict[str, bool]:
    """Return the frozen paired constraint map used by the reward override."""

    constraints = {
        "complete_horizon": int(action["completed_horizon_steps"])
        == contract.horizon_steps,
        "dose_at_least_minimum": int(action["target_command_step_count"])
        >= contract.minimum_dose_steps,
        "zero_unexpected_commands": int(action["unexpected_command_count"]) == 0,
        "zero_added_collisions": int(action["collision_observations"])
        <= int(control["collision_observations"]),
        "zero_added_teleports": int(action["teleport_observations"])
        <= int(control["teleport_observations"]),
        "low_ttc_within_budget": int(action["low_ttc_exposure_steps"])
        - int(control["low_ttc_exposure_steps"])
        <= contract.low_ttc_excess_budget_steps,
        "delay_within_budget": _number(action, "mean_corridor_delay_rate")
        - _number(control, "mean_corridor_delay_rate")
        <= contract.maximum_delay_rate_regression,
        "arrival_within_budget": int(action["arrived_vehicles"])
        - int(control["arrived_vehicles"])
        >= -contract.maximum_arrival_loss,
        "discharge_within_budget": int(action["merge_discharge_count"])
        - int(control["merge_discharge_count"])
        >= -contract.maximum_discharge_loss,
    }
    if "treatment_safety_abort_rows" in action:
        constraints["zero_treatment_safety_aborts"] = not bool(
            action.get("treatment_safety_abort_rows")
        )
    return constraints


def paired_physical_utility(
    control: Mapping[str, Any],
    action: Mapping[str, Any],
    *,
    severe_reference_per_step: float,
    energy_reference_per_step: float,
    contract: BinaryOptionRewardContract = IDM_BINARY_OPTION_REWARD_V1,
) -> dict[str, float]:
    """Compute the centered 60-step severe/energy treatment effect."""

    severe_reference = float(severe_reference_per_step)
    energy_reference = float(energy_reference_per_step)
    if not math.isfinite(severe_reference) or severe_reference <= 0.0:
        raise ValueError("severe reference must be positive and finite")
    if not math.isfinite(energy_reference) or energy_reference <= 0.0:
        raise ValueError("energy reference must be positive and finite")
    physical = safety_composite(
        zero_severe=_number(control, "severe_braking_count_sum"),
        action_severe=_number(action, "severe_braking_count_sum"),
        severe_reference=severe_reference,
        zero_wave=_number(control, "acceleration_energy_sum"),
        action_wave=_number(action, "acceleration_energy_sum"),
        wave_reference=energy_reference,
        horizon_steps=contract.horizon_steps,
    )
    severe = float(physical["severe_braking_effect_reference_units"])
    energy = float(physical["wave_effect_reference_units"])
    utility = contract.severe_weight * severe + contract.energy_weight * energy
    return {
        "severe_braking_effect_reference_units": severe,
        "energy_effect_reference_units": energy,
        "severe_contribution": contract.severe_weight * severe,
        "energy_contribution": contract.energy_weight * energy,
        "physical_utility": float(utility),
    }


def binary_option_reward(
    control: Mapping[str, Any],
    action: Mapping[str, Any],
    *,
    severe_reference_per_step: float,
    energy_reference_per_step: float,
    contract: BinaryOptionRewardContract = IDM_BINARY_OPTION_REWARD_V1,
) -> dict[str, Any]:
    """Return the paired activation reward and complete audit components."""

    physical = paired_physical_utility(
        control,
        action,
        severe_reference_per_step=severe_reference_per_step,
        energy_reference_per_step=energy_reference_per_step,
        contract=contract,
    )
    constraints = paired_constraints(control, action, contract)
    constraints_pass = all(constraints.values())
    reward = (
        math.tanh(float(physical["physical_utility"]))
        if constraints_pass
        else contract.hard_failure_reward
    )
    return {
        "schema": SCHEMA,
        "abstain_reward": 0.0,
        "activate_reward": float(reward),
        "constraints": constraints,
        "constraints_pass": constraints_pass,
        **physical,
    }
