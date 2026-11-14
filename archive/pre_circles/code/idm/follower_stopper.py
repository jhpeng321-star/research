"""Pure contract for the modified FollowerStopper-shaped P30 ceiling.

This module contains no TraCI calls.  It implements the switching geometry of
FollowerStopper, augmented by the speed-dependent headway terms proposed for
higher-speed operation.  The environment uses the returned desired speed only
as an upper bound on the existing analytical P30 command; the repository's
hard projection remains authoritative.

Primary sources:

* Stern et al. (2018), https://arxiv.org/abs/1705.01693
* Chou et al. (2021), https://arxiv.org/abs/2112.14345
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Optional


MODIFIED_FOLLOWER_STOPPER_P30_SCHEMA = (
    "idm_modified_follower_stopper_p30_ceiling/1"
)


@dataclass(frozen=True)
class ModifiedFollowerStopperContract:
    """Literature-seeded switching geometry for a highway-speed probe.

    ``base_gaps_m`` and ``closing_decelerations_mps2`` are the original
    FollowerStopper parameters.  ``time_headways_s`` are the higher-speed
    modification.  These values define a reproducible development treatment;
    they are not claimed as a safety proof for this merge network.
    """

    schema: str = MODIFIED_FOLLOWER_STOPPER_P30_SCHEMA
    base_gaps_m: tuple[float, float, float] = (4.5, 5.25, 6.0)
    closing_decelerations_mps2: tuple[float, float, float] = (1.5, 1.0, 0.5)
    time_headways_s: tuple[float, float, float] = (0.4, 1.2, 1.8)
    reference_percentile: str = "p30"
    application: str = "front_feedback_speed_ceiling_on_analytical_p30"
    option_steps: int = 20
    efficacy_conditioned_hold: bool = False
    hard_projection_authoritative: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


MODIFIED_FOLLOWER_STOPPER_P30_V1 = ModifiedFollowerStopperContract()


@dataclass(frozen=True)
class ModifiedFollowerStopperResult:
    desired_speed_mps: float
    reference_speed_mps: float
    clipped_leader_speed_mps: Optional[float]
    relative_speed_mps: Optional[float]
    closing_speed_mps: Optional[float]
    switching_boundaries_m: Optional[tuple[float, float, float]]
    leader_gap_m: Optional[float]
    zone: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def modified_follower_stopper_speed(
    *,
    ego_speed_mps: float,
    reference_speed_mps: float,
    leader_gap_m: float | None,
    leader_speed_mps: float | None,
    contract: ModifiedFollowerStopperContract = MODIFIED_FOLLOWER_STOPPER_P30_V1,
) -> ModifiedFollowerStopperResult:
    """Return the causal modified-FollowerStopper desired speed.

    The caller is responsible for clamping ``reference_speed_mps`` to the road
    and treatment bounds.  Invalid finite inputs are rejected rather than
    silently creating a command.  A missing leader yields the reference speed,
    exactly matching an unconstrained speed ceiling.
    """

    ego = float(ego_speed_mps)
    reference = float(reference_speed_mps)
    if not isfinite(ego) or ego < 0.0:
        raise ValueError("ego speed must be finite and nonnegative")
    if not isfinite(reference) or reference < 0.0:
        raise ValueError("reference speed must be finite and nonnegative")

    if leader_gap_m is None or leader_speed_mps is None:
        return ModifiedFollowerStopperResult(
            desired_speed_mps=reference,
            reference_speed_mps=reference,
            clipped_leader_speed_mps=None,
            relative_speed_mps=None,
            closing_speed_mps=None,
            switching_boundaries_m=None,
            leader_gap_m=None,
            zone="free",
        )

    gap = float(leader_gap_m)
    leader = float(leader_speed_mps)
    if not isfinite(gap) or not isfinite(leader):
        raise ValueError("leader state must be finite")

    if not (
        len(contract.base_gaps_m)
        == len(contract.closing_decelerations_mps2)
        == len(contract.time_headways_s)
        == 3
    ):
        raise ValueError("FollowerStopper contract must define three switching zones")
    if any(decel <= 0.0 for decel in contract.closing_decelerations_mps2):
        raise ValueError("closing-deceleration parameters must be positive")

    relative_speed = leader - ego
    closing_speed = min(relative_speed, 0.0)
    boundaries = tuple(
        float(base + closing_speed * closing_speed / (2.0 * decel) + headway * ego)
        for base, decel, headway in zip(
            contract.base_gaps_m,
            contract.closing_decelerations_mps2,
            contract.time_headways_s,
        )
    )
    if not boundaries[0] < boundaries[1] < boundaries[2]:
        raise ValueError("FollowerStopper switching boundaries must be ordered")

    clipped_leader = min(max(leader, 0.0), reference)
    x1, x2, x3 = boundaries
    if gap <= x1:
        desired, zone = 0.0, "stop"
    elif gap <= x2:
        fraction = (gap - x1) / (x2 - x1)
        desired, zone = clipped_leader * fraction, "closing_transition"
    elif gap <= x3:
        fraction = (gap - x2) / (x3 - x2)
        desired = clipped_leader + (reference - clipped_leader) * fraction
        zone = "recovery_transition"
    else:
        desired, zone = reference, "free"

    desired = min(reference, max(0.0, float(desired)))
    return ModifiedFollowerStopperResult(
        desired_speed_mps=desired,
        reference_speed_mps=reference,
        clipped_leader_speed_mps=clipped_leader,
        relative_speed_mps=relative_speed,
        closing_speed_mps=closing_speed,
        switching_boundaries_m=boundaries,
        leader_gap_m=gap,
        zone=zone,
    )
