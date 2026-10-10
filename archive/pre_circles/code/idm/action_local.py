"""Pure contracts for the Phase-5 action-local residual diagnostic.

The SUMO runner lives in :mod:`tools.audit_idm_phase5_decision_state_sufficiency`.
This module keeps landmark detection, replay-noise handling, and paired-effect
classification deterministic and independently testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class LandmarkContract:
    level_threshold: float = 1.0
    trend_threshold: float = 0.10
    persistence_steps: int = 3
    minimum_runway_m: float = 150.0
    collapse_speed_mps: float = 15.0
    collapse_min_slow_vehicles: int = 5
    collapse_sustain_steps: int = 60
    minimum_landmark_separation_steps: int = 60
    maximum_landmarks_per_cell: int = 2
    priority: tuple[str, ...] = ("wave_onset", "severe_onset")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OutcomeContract:
    pulse_steps: int = 10
    safety_horizon_steps: int = 20
    service_horizon_steps: int = 60
    base_deadband: float = 0.02
    low_ttc_excess_budget_steps: int = 2
    maximum_time_loss_regression_s_per_vehicle: float = 1.0
    maximum_arrival_loss: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


def landmark_signal(
    landmark: str,
    context: Mapping[str, float],
    contract: LandmarkContract,
) -> bool:
    """Return whether one causal landmark condition is currently satisfied."""
    if landmark == "wave_onset":
        level = float(context["idm_wave_energy_level"])
        trend = float(context["idm_wave_energy_trend"])
    elif landmark == "severe_onset":
        level = float(context["idm_severe_level"])
        trend = float(context["idm_severe_trend"])
    else:
        raise ValueError(f"unknown landmark {landmark!r}")
    return level >= contract.level_threshold and trend >= contract.trend_threshold


def collapse_condition(premerge_speeds_mps: Sequence[float], contract: LandmarkContract) -> bool:
    """One causal controller-step of the frozen sustained-collapse definition."""
    speeds = [float(value) for value in premerge_speeds_mps]
    if not speeds:
        return False
    slow = sum(value < contract.collapse_speed_mps for value in speeds)
    return (
        slow >= contract.collapse_min_slow_vehicles
        and sum(speeds) / len(speeds) <= contract.collapse_speed_mps
    )


def replay_deadband(base_deadband: float, absolute_replay_effects: Sequence[float]) -> float:
    """Freeze a symmetric deadband before treatment effects are inspected.

    The caller supplies zero-versus-zero absolute safety-composite effects.  A
    deterministic nearest-rank p99 avoids a NumPy dependency in this contract.
    """
    values = sorted(abs(float(value)) for value in absolute_replay_effects)
    if not values:
        return float(base_deadband)
    index = max(0, min(len(values) - 1, (99 * len(values) + 99) // 100 - 1))
    return max(float(base_deadband), 2.0 * values[index])


def classify_effect(
    safety_composite: float,
    *,
    deadband: float,
    constraints_pass: bool,
) -> str:
    """Classify one action-state pair without erasing its action identity."""
    if not constraints_pass or float(safety_composite) < -float(deadband):
        return "harmful"
    if float(safety_composite) > float(deadband):
        return "helpful"
    return "neutral"


def safety_composite(
    *,
    zero_severe: float,
    action_severe: float,
    severe_reference: float,
    zero_wave: float,
    action_wave: float,
    wave_reference: float,
    horizon_steps: int,
) -> dict[str, float]:
    """Return separate reference-normalized effects and their frozen mean."""
    severe = (float(zero_severe) - float(action_severe)) / max(
        float(severe_reference) * int(horizon_steps), 1e-12
    )
    wave = (float(zero_wave) - float(action_wave)) / max(
        float(wave_reference) * int(horizon_steps), 1e-12
    )
    return {
        "severe_braking_effect_reference_units": severe,
        "wave_effect_reference_units": wave,
        "safety_composite_reference_units": 0.5 * (severe + wave),
    }
