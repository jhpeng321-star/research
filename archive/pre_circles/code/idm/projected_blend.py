"""Pure contract for the development-only native-IDM-to-P30 blend."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Optional


PROJECTED_BLEND_SCHEMA = "idm_p30_projected_blend/1"


@dataclass(frozen=True)
class ProjectedBlendContract:
    schema: str = PROJECTED_BLEND_SCHEMA
    active_strengths: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    strength_zero_semantics: str = "true_abstain_no_traci_command"
    native_endpoint: str = "allowed_acceleration_follow_speed_proxy"
    p30_endpoint: str = "frozen_filtered_p30_analytical_speed_ceiling"
    composition: str = "project_endpoints_then_convex_blend_then_project"
    terminal_reference_percentile: str = "p30"
    option_steps: int = 20
    release: str = "legacy_controlled_release_with_lane_blocking"
    efficacy_conditioned_hold: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


IDM_P30_PROJECTED_BLEND_V1 = ProjectedBlendContract()


def native_proxy_acceleration(
    *,
    current_speed_mps: float,
    allowed_speed_mps: float,
    maximum_acceleration_mps2: float,
    follow_speed_mps: Optional[float],
    dt_s: float,
) -> tuple[float, float]:
    """Return the treatment-independent one-step native speed/acceleration proxy."""
    values = (
        current_speed_mps,
        allowed_speed_mps,
        maximum_acceleration_mps2,
        dt_s,
    )
    if not all(isfinite(float(value)) for value in values):
        raise ValueError("native proxy inputs must be finite")
    if float(dt_s) <= 0.0:
        raise ValueError("dt_s must be positive")
    speed = max(0.0, float(current_speed_mps))
    allowed = max(0.0, float(allowed_speed_mps))
    acceleration_limit = max(0.0, float(maximum_acceleration_mps2))
    target = min(allowed, speed + acceleration_limit * float(dt_s))
    if follow_speed_mps is not None:
        if not isfinite(float(follow_speed_mps)):
            raise ValueError("follow speed must be finite when present")
        target = min(target, max(0.0, float(follow_speed_mps)))
    acceleration = (target - speed) / float(dt_s)
    return float(target), float(acceleration)


def projected_blend_acceleration(
    *,
    projected_native_acceleration_mps2: float,
    projected_p30_acceleration_mps2: float,
    strength: float,
) -> float:
    """Convexly blend two already projected deliverable accelerations."""
    native = float(projected_native_acceleration_mps2)
    p30 = float(projected_p30_acceleration_mps2)
    blend = float(strength)
    if not all(isfinite(value) for value in (native, p30, blend)):
        raise ValueError("projected blend inputs must be finite")
    if blend < 0.0 or blend > 1.0:
        raise ValueError("strength must remain in [0, 1]")
    return float((1.0 - blend) * native + blend * p30)
