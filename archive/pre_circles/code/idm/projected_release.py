"""Pure rear-aware contracts for the P30 projected-release treatment.

The module deliberately contains no TraCI calls.  The environment supplies an
optional SUMO secure-gap evaluator, while the pure solver supplies the
one-step kinematics, ballistic fallback and deterministic bisection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Callable, Optional


PROJECTED_RELEASE_SCHEMA = "idm_p30_projected_release/1"
PROJECTED_RELEASE_SMOOTHSTEP_SCHEMA = "idm_p30_projected_release/2"
PROJECTED_RELEASE_SETTLE_SCHEMA = "idm_p30_projected_release/3"


@dataclass(frozen=True)
class ProjectedReleaseContract:
    schema: str = PROJECTED_RELEASE_SCHEMA
    blend_schedule: str = "linear"
    settle_steps: int = 0
    blend_origin: str = "last_p30_acceleration"
    rear_margin_m: float = 5.0
    convergence_steps: int = 2
    nominal_budget_steps: int = 10
    cooldown_steps: int = 0
    speed_tolerance_mps: float = 0.5
    acceleration_tolerance_mps2: float = 0.25
    bisection_iterations: int = 32

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


PROJECTED_RELEASE_V1 = ProjectedReleaseContract()
PROJECTED_RELEASE_V2 = ProjectedReleaseContract(
    schema=PROJECTED_RELEASE_SMOOTHSTEP_SCHEMA,
    blend_schedule="smoothstep_3x2_minus_2x3",
)
PROJECTED_RELEASE_V3 = ProjectedReleaseContract(
    schema=PROJECTED_RELEASE_SETTLE_SCHEMA,
    blend_schedule="settle_then_smoothstep_3x2_minus_2x3",
    settle_steps=5,
    blend_origin="zero_acceleration",
)


def release_blend_fraction(
    release_step: int,
    contract: ProjectedReleaseContract = PROJECTED_RELEASE_V1,
) -> float:
    x = min(
        1.0,
        max(0.0, float(release_step) / max(1, contract.nominal_budget_steps)),
    )
    if contract.blend_schedule == "linear":
        return x
    if contract.blend_schedule == "smoothstep_3x2_minus_2x3":
        return x * x * (3.0 - 2.0 * x)
    if contract.blend_schedule == "settle_then_smoothstep_3x2_minus_2x3":
        if release_step <= contract.settle_steps:
            return 0.0
        release_span = max(1, contract.nominal_budget_steps - contract.settle_steps)
        shifted = min(
            1.0,
            max(0.0, (release_step - contract.settle_steps) / release_span),
        )
        return shifted * shifted * (3.0 - 2.0 * shifted)
    raise ValueError(f"unknown projected-release blend {contract.blend_schedule!r}")


@dataclass(frozen=True)
class RearFloorInput:
    current_ego_speed_mps: float
    follower_speed_mps: float
    follower_gap_m: float
    minimum_next_speed_mps: float
    maximum_next_speed_mps: float
    dt_s: float
    rear_ttc_floor_s: float
    follower_decel_mps2: float


@dataclass(frozen=True)
class RearFloorResult:
    minimum_next_speed_mps: float
    ttc_floor_mps: float
    searched_floor_mps: float
    minimum_search_speed_mps: float
    maximum_next_speed_mps: float
    follower_response_available: bool
    feasible_within_upper_bound: bool
    predicted_gap_m: float
    secure_gap_m: Optional[float]
    secure_headroom_m: Optional[float]
    predicted_follower_speed_mps: Optional[float]
    follower_deceleration_margin_mps: Optional[float]
    ttc_margin_mps: float
    reasons: tuple[str, ...]


def predicted_rear_gap_m(inp: RearFloorInput, next_ego_speed_mps: float) -> float:
    """Conservative one-step gap under constant follower speed."""
    dt = max(float(inp.dt_s), 1e-6)
    ego_mean = 0.5 * (
        max(0.0, float(inp.current_ego_speed_mps))
        + max(0.0, float(next_ego_speed_mps))
    )
    return float(inp.follower_gap_m) + (
        ego_mean - max(0.0, float(inp.follower_speed_mps))
    ) * dt


def solve_rear_speed_floor(
    inp: RearFloorInput,
    *,
    follower_response_at_speed: Callable[[float], tuple[float, float]] | None = None,
    contract: ProjectedReleaseContract = PROJECTED_RELEASE_V1,
) -> RearFloorResult:
    """Find the least next speed satisfying SUMO follower response constraints."""
    minimum = max(0.0, float(inp.minimum_next_speed_mps))
    upper = max(0.0, float(inp.maximum_next_speed_mps))
    ttc_floor = max(
        0.0,
        float(inp.follower_speed_mps)
        - max(0.0, float(inp.follower_gap_m))
        / max(float(inp.rear_ttc_floor_s), 1e-6),
    )
    raw_search_low = max(minimum, ttc_floor)
    search_low = min(upper, raw_search_low)
    response_available = follower_response_at_speed is not None
    response_error = False

    def requirements(
        speed: float,
    ) -> tuple[bool, float | None, float | None, float | None, float | None, float]:
        nonlocal response_error
        predicted_follower: float | None = None
        secure_gap: float | None = None
        if follower_response_at_speed is not None:
            try:
                follower_value, secure_value = follower_response_at_speed(float(speed))
                follower_value = float(follower_value)
                secure_value = float(secure_value)
                if (
                    isfinite(follower_value)
                    and follower_value >= 0.0
                    and isfinite(secure_value)
                    and secure_value >= 0.0
                ):
                    predicted_follower = follower_value
                    secure_gap = secure_value
                else:
                    response_error = True
            except Exception:
                response_error = True
        if predicted_follower is None or secure_gap is None:
            return False, None, None, None, None, float("-inf")
        dt = max(float(inp.dt_s), 1e-6)
        ego_mean = 0.5 * (max(0.0, inp.current_ego_speed_mps) + max(0.0, speed))
        follower_mean = 0.5 * (
            max(0.0, inp.follower_speed_mps) + predicted_follower
        )
        gap = float(inp.follower_gap_m) + (ego_mean - follower_mean) * dt
        headroom = gap - secure_gap - contract.rear_margin_m
        minimum_follower_speed = max(
            0.0,
            float(inp.follower_speed_mps)
            - max(0.1, float(inp.follower_decel_mps2)) * dt,
        )
        decel_margin = predicted_follower - minimum_follower_speed
        ttc_margin = speed - ttc_floor
        feasible = min(headroom, decel_margin, ttc_margin) >= -1e-9
        return feasible, predicted_follower, secure_gap, headroom, decel_margin, ttc_margin

    upper_values = requirements(upper)
    upper_ok = raw_search_low <= upper + 1e-9 and upper_values[0]
    if upper_ok:
        low, high = search_low, upper
        for _ in range(max(1, int(contract.bisection_iterations))):
            middle = 0.5 * (low + high)
            if requirements(middle)[0]:
                high = middle
            else:
                low = middle
        searched_floor = high
    else:
        searched_floor = upper

    floor = min(upper, max(search_low, searched_floor))
    _, follower_predicted, secure_gap, headroom, decel_margin, ttc_margin = requirements(
        floor
    )
    if follower_predicted is None:
        predicted_gap = predicted_rear_gap_m(inp, floor)
    else:
        dt = max(float(inp.dt_s), 1e-6)
        predicted_gap = float(inp.follower_gap_m) + 0.5 * (
            (max(0.0, inp.current_ego_speed_mps) + floor)
            - (max(0.0, inp.follower_speed_mps) + follower_predicted)
        ) * dt
    reasons: list[str] = []
    if floor > min(minimum, upper) + 1e-9:
        reasons.append("rear_release_floor")
    if not response_available or response_error or follower_predicted is None:
        reasons.append("sumo_follower_response_unavailable")
    if raw_search_low > upper + 1e-9:
        reasons.append("rear_raw_floor_above_upper_bound")
    if not upper_ok:
        reasons.append("rear_floor_above_upper_bound")
    return RearFloorResult(
        minimum_next_speed_mps=float(floor),
        ttc_floor_mps=float(ttc_floor),
        searched_floor_mps=float(searched_floor),
        minimum_search_speed_mps=float(raw_search_low),
        maximum_next_speed_mps=float(upper),
        follower_response_available=bool(
            response_available and not response_error and follower_predicted is not None
        ),
        feasible_within_upper_bound=bool(upper_ok),
        predicted_gap_m=float(predicted_gap),
        secure_gap_m=(None if secure_gap is None else float(secure_gap)),
        secure_headroom_m=(None if headroom is None else float(headroom)),
        predicted_follower_speed_mps=(
            None if follower_predicted is None else float(follower_predicted)
        ),
        follower_deceleration_margin_mps=(
            None if decel_margin is None else float(decel_margin)
        ),
        ttc_margin_mps=float(ttc_margin),
        reasons=tuple(reasons),
    )


def handoff_converged(
    *,
    projected_target_speed_mps: float,
    native_target_speed_mps: float,
    projected_acceleration_mps2: float,
    native_acceleration_mps2: float,
    constraint_conflict: bool,
    rear_feasible: bool,
    contract: ProjectedReleaseContract = PROJECTED_RELEASE_V1,
) -> bool:
    return (
        abs(float(projected_target_speed_mps) - float(native_target_speed_mps))
        <= contract.speed_tolerance_mps
        and abs(
            float(projected_acceleration_mps2)
            - float(native_acceleration_mps2)
        )
        <= contract.acceleration_tolerance_mps2
        and not bool(constraint_conflict)
        and bool(rear_feasible)
    )
