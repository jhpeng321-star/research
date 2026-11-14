"""Analytical diagnostics for the frozen v2 recovery failure, not a policy.

These calculations neither simulate traffic nor supply an online native
counterfactual. The visibility margin is conditional on a constant-speed
leader; it is not a prediction of future leader motion or a safety guarantee.
"""

from dataclasses import asdict, dataclass
from math import exp, isfinite


def large_gap_acceleration(
    *,
    catchup_offset_mps=2.0,
    response_seconds=5.0,
    filter_seconds=5.0,
    interval_seconds=1.0,
    correction_mps2=0.0,
):
    """Constant-acceleration limit on the fully blended, saturated-gap branch.

    Assumptions: mode stays active, the leader remains detected, actuator
    projection/native safety do not bind, correction is constant, and the
    five-second ego EMA is updated at each one-second observation.
    This is a branch-specific asymptote, not a global acceleration ceiling.
    """
    inputs = (
        catchup_offset_mps,
        response_seconds,
        filter_seconds,
        interval_seconds,
        correction_mps2,
    )
    if not all(isfinite(value) for value in inputs):
        raise ValueError("finite inputs required")
    if min(response_seconds, filter_seconds, interval_seconds) <= 0:
        raise ValueError("positive time constants required")
    q = exp(-interval_seconds / filter_seconds)
    lag_seconds = interval_seconds * q / (1 - q)
    return (catchup_offset_mps + response_seconds * correction_mps2) / (
        response_seconds + lag_seconds
    )


@dataclass(frozen=True)
class VisibilityMargin:
    opening_speed_mps: float
    remaining_sensor_distance_m: float
    minimum_additional_gap_m: float
    margin_m: float
    required_acceleration_mps2: float | None
    constant_leader_retention_possible_under_acceleration_bound: bool


def constant_leader_visibility_margin(
    *,
    gap_m: float,
    ego_speed_mps: float,
    leader_speed_mps: float,
    sensor_range_m: float = 100.0,
    acceleration_bound_mps2: float = 0.5,
) -> VisibilityMargin:
    """Optimistic retention bound using only an already detected leader.

    With opening speed w>0 and a constant-speed leader, instant maximum ego
    acceleration still adds w²/(2a) to the gap before matching speed. Ignoring
    jerk/response/safety limitations is optimistic: a negative margin proves
    the constrained active controller cannot retain that constant-speed
    leader within range from this state. Positive margin is not a guarantee.
    """
    inputs = (gap_m, ego_speed_mps, leader_speed_mps, sensor_range_m, acceleration_bound_mps2)
    if not all(isfinite(value) for value in inputs):
        raise ValueError("finite local measurements required")
    if (
        min(gap_m, ego_speed_mps, leader_speed_mps) < 0
        or min(sensor_range_m, acceleration_bound_mps2) <= 0
    ):
        raise ValueError("invalid local geometry or acceleration bound")
    if gap_m > sensor_range_m:
        raise ValueError("leader outside declared observation range")
    opening = max(0.0, leader_speed_mps - ego_speed_mps)
    room = sensor_range_m - gap_m
    extra = opening * opening / (2 * acceleration_bound_mps2)
    required = 0.0 if opening == 0 else None if room == 0 else opening * opening / (2 * room)
    margin = room - extra
    return VisibilityMargin(opening, room, extra, margin, required, margin >= 0)


def margin_from_local_observation(observation):
    """Read only the current frame of the existing 89-field actor observation."""
    if len(observation) != 89:
        raise ValueError("v2 local observation required")
    current = observation[68:85]
    if current[9] <= 0.5 or current[16] <= 0.5:
        return None  # Absence is not zero gap or a measured zero-speed leader.
    return asdict(
        constant_leader_visibility_margin(
            gap_m=float(current[7]) * 100,
            ego_speed_mps=float(current[0]) * 30,
            leader_speed_mps=float(current[0] - current[8]) * 30,
            sensor_range_m=float(current[10]) * 100,
        )
    )
