"""Pure contract for the rear-aware decentralized feedback pulse.

The treatment is deliberately distinct from fixed P30.  A five-second probe
ramps from native IDM toward a bilateral FollowerStopper-shaped command.  A
hold is permitted only after causal local response and projection checks pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any


REAR_AWARE_FEEDBACK_SCHEMA = "idm_rear_aware_feedback_pulse/1"
REAR_AWARE_FEEDBACK_V2_SCHEMA = "idm_rear_aware_feedback_pulse/2"


@dataclass(frozen=True)
class RearAwareFeedbackContract:
    schema: str = REAR_AWARE_FEEDBACK_SCHEMA
    reference_percentile: str = "p50"
    probe_steps: int = 5
    maximum_active_steps: int = 20
    probe_strengths: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
    minimum_projection_survival: float = 0.8
    meaningful_command_delta_mps2: float = 0.05
    minimum_meaningful_probe_commands: int = 2
    rear_margin_m: float = 5.0
    maximum_probe_rear_gap_loss_m: float = 2.5
    maximum_hold_rear_gap_loss_m: float = 3.5
    maximum_probe_closing_increase_mps: float = 0.75
    maximum_hold_closing_increase_mps: float = 1.0
    severe_follower_braking_mps2: float = -3.0
    minimum_closing_improvement_mps: float = 0.25
    minimum_local_energy_improvement_m2ps4: float = 0.05
    minimum_disturbance_std_improvement_mps: float = 0.10
    maximum_probe_disturbance_std_increase_mps: float = 0.75
    maximum_hold_disturbance_std_increase_mps: float = 1.0
    release_contract_schema: str = "idm_p30_projected_release/2"
    hard_projection_authoritative: bool = True
    efficacy_conditioned_hold: bool = True
    soft_response_checks_after_probe: bool = False
    risk_conditioned_rear_response: bool = False
    rear_response_guard_gap_m: float = 20.0
    rear_response_guard_ttc_s: float = 6.0

    def __post_init__(self) -> None:
        if len(self.probe_strengths) != self.probe_steps:
            raise ValueError("probe strength schedule must match probe duration")
        if not 0 < self.probe_steps < self.maximum_active_steps:
            raise ValueError("probe must be shorter than the active treatment")
        if any(not 0.0 < value <= 1.0 for value in self.probe_strengths):
            raise ValueError("probe strengths must lie in (0, 1]")
        if tuple(sorted(self.probe_strengths)) != self.probe_strengths:
            raise ValueError("probe strengths must be nondecreasing")
        if not 0.0 < self.minimum_projection_survival <= 1.0:
            raise ValueError("projection survival must lie in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REAR_AWARE_FEEDBACK_V1 = RearAwareFeedbackContract()
REAR_AWARE_FEEDBACK_V2 = RearAwareFeedbackContract(
    schema=REAR_AWARE_FEEDBACK_V2_SCHEMA,
    soft_response_checks_after_probe=True,
    risk_conditioned_rear_response=True,
)


@dataclass(frozen=True)
class FeedbackResponse:
    disturbance_present: bool
    disturbance_std_mps: float | None
    rear_gap_m: float | None
    rear_closing_speed_mps: float | None
    follower_acceleration_mps2: float | None
    local_acceleration_energy_m2ps4: float
    rear_floor_feasible: bool
    projection_survived: bool
    command_delta_from_native_mps2: float
    follower_id: str | None = None

    def __post_init__(self) -> None:
        optional = (
            self.disturbance_std_mps,
            self.rear_gap_m,
            self.rear_closing_speed_mps,
            self.follower_acceleration_mps2,
        )
        if any(value is not None and not isfinite(float(value)) for value in optional):
            raise ValueError("feedback response contains a non-finite field")
        if not isfinite(float(self.local_acceleration_energy_m2ps4)):
            raise ValueError("local acceleration energy must be finite")
        if self.follower_id is not None and not isinstance(self.follower_id, str):
            raise ValueError("follower identity must be a string or None")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def treatment_strength(
    active_age_steps: int,
    contract: RearAwareFeedbackContract = REAR_AWARE_FEEDBACK_V2,
) -> float:
    """Return the frozen one-based probe/hold authority schedule."""

    age = int(active_age_steps)
    if age < 1 or age > contract.maximum_active_steps:
        raise ValueError("active age lies outside the treatment contract")
    if age <= contract.probe_steps:
        return float(contract.probe_strengths[age - 1])
    return 1.0


def feedback_release_reason(
    *,
    active_age_steps: int,
    baseline: FeedbackResponse,
    current: FeedbackResponse,
    projection_survival_count: int,
    meaningful_command_count: int,
    contract: RearAwareFeedbackContract = REAR_AWARE_FEEDBACK_V2,
) -> str | None:
    """Return a causal early-release reason, or ``None`` to continue/hold.

    Follower identity is part of the causal contract: rear-gap, closing-speed
    and follower-acceleration measurements belong to one physical follower.
    Comparing baseline and current values across different followers would
    fabricate trends, so an identity change is honoured as an early release
    even when it coincides with a hard rear/safety condition.
    """

    age = int(active_age_steps)
    if baseline.follower_id != current.follower_id:
        return "follower_identity_changed"
    if not current.rear_floor_feasible:
        return "rear_floor_infeasible"
    if not current.projection_survived:
        return "projection_conflict"
    if (
        current.follower_acceleration_mps2 is not None
        and current.follower_acceleration_mps2 < contract.severe_follower_braking_mps2
    ):
        return "severe_follower_braking"
    rear_gap_loss = (
        None
        if baseline.rear_gap_m is None or current.rear_gap_m is None
        else baseline.rear_gap_m - current.rear_gap_m
    )
    closing_increase = (
        None
        if baseline.rear_closing_speed_mps is None or current.rear_closing_speed_mps is None
        else current.rear_closing_speed_mps - baseline.rear_closing_speed_mps
    )
    std_increase = (
        None
        if baseline.disturbance_std_mps is None or current.disturbance_std_mps is None
        else current.disturbance_std_mps - baseline.disturbance_std_mps
    )
    gap_limit = (
        contract.maximum_probe_rear_gap_loss_m
        if age <= contract.probe_steps
        else contract.maximum_hold_rear_gap_loss_m
    )
    closing_limit = (
        contract.maximum_probe_closing_increase_mps
        if age <= contract.probe_steps
        else contract.maximum_hold_closing_increase_mps
    )
    std_limit = (
        contract.maximum_probe_disturbance_std_increase_mps
        if age <= contract.probe_steps
        else contract.maximum_hold_disturbance_std_increase_mps
    )
    rear_ttc_s = (
        None
        if current.rear_gap_m is None
        or current.rear_closing_speed_mps is None
        or current.rear_closing_speed_mps <= 0.0
        else current.rear_gap_m / current.rear_closing_speed_mps
    )
    rear_response_risk = (
        not contract.risk_conditioned_rear_response
        or (
            current.rear_gap_m is not None
            and current.rear_gap_m <= contract.rear_response_guard_gap_m
        )
        or (
            rear_ttc_s is not None
            and rear_ttc_s <= contract.rear_response_guard_ttc_s
        )
    )
    if contract.soft_response_checks_after_probe and age < contract.probe_steps:
        return None
    if rear_response_risk and rear_gap_loss is not None and rear_gap_loss > gap_limit:
        return "rear_gap_compression"
    if (
        rear_response_risk
        and closing_increase is not None
        and closing_increase > closing_limit
    ):
        return "rear_closing_deterioration"
    if std_increase is not None and std_increase > std_limit:
        return "disturbance_worsening"
    if not current.disturbance_present:
        return "disturbance_resolved"
    if age < contract.probe_steps:
        return None
    if age == contract.probe_steps:
        survival = projection_survival_count / contract.probe_steps
        if survival < contract.minimum_projection_survival:
            return "insufficient_projection_survival"
        if meaningful_command_count < contract.minimum_meaningful_probe_commands:
            return "insufficient_command_authority"
        disturbance_improved = (
            baseline.disturbance_std_mps is not None
            and current.disturbance_std_mps is not None
            and baseline.disturbance_std_mps - current.disturbance_std_mps
            >= contract.minimum_disturbance_std_improvement_mps
        )
        closing_improved = (
            baseline.rear_closing_speed_mps is not None
            and current.rear_closing_speed_mps is not None
            and baseline.rear_closing_speed_mps - current.rear_closing_speed_mps
            >= contract.minimum_closing_improvement_mps
        )
        energy_improved = (
            baseline.local_acceleration_energy_m2ps4 - current.local_acceleration_energy_m2ps4
            >= contract.minimum_local_energy_improvement_m2ps4
        )
        if not (disturbance_improved or closing_improved or energy_improved):
            return "probe_no_improving_response"
    return None
