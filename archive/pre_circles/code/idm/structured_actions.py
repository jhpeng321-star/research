"""Pure contracts for safe high-level actions around analytical IDM damping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ActionFamily = Literal["residual", "gain", "reference_offset"]


@dataclass(frozen=True)
class StructuredActionBounds:
    gain_min: float = 0.50
    gain_max: float = 1.50
    reference_offset_min_mps: float = -2.0
    reference_offset_max_mps: float = 2.0


def nominal_action(family: ActionFamily) -> float:
    """Return the action that exactly preserves analytical damping."""
    if family == "gain":
        return 1.0
    if family in {"residual", "reference_offset"}:
        return 0.0
    raise ValueError(f"unknown structured action family {family!r}")


def clip_action(
    family: ActionFamily,
    value: float,
    *,
    residual_min_mps2: float,
    residual_max_mps2: float,
    bounds: StructuredActionBounds = StructuredActionBounds(),
) -> float:
    """Clip a requested action without changing its family semantics."""
    limits = {
        "residual": (residual_min_mps2, residual_max_mps2),
        "gain": (bounds.gain_min, bounds.gain_max),
        "reference_offset": (
            bounds.reference_offset_min_mps,
            bounds.reference_offset_max_mps,
        ),
    }
    try:
        low, high = limits[family]
    except KeyError as exc:
        raise ValueError(f"unknown structured action family {family!r}") from exc
    return float(np.clip(float(value), float(low), float(high)))


def low_pass_action(
    previous: float,
    requested: float,
    *,
    dt_s: float,
    tau_s: float,
) -> float:
    """Causal first-order action transition used by fixed arms and future RL."""
    if tau_s <= 0.0:
        return float(requested)
    alpha = float(dt_s) / (float(tau_s) + float(dt_s))
    return float(previous + alpha * (requested - previous))


def gain_desired_acceleration(nominal_acceleration_mps2: float, gain: float) -> float:
    return float(gain) * float(nominal_acceleration_mps2)


def adjusted_reference_speed(
    analytical_reference_mps: float,
    offset_mps: float,
    *,
    floor_mps: float,
    allowed_speed_mps: float,
) -> float:
    return float(
        np.clip(
            float(analytical_reference_mps) + float(offset_mps),
            float(floor_mps),
            float(allowed_speed_mps),
        )
    )


MODE_TO_GAIN = {"mild": 0.75, "nominal": 1.00, "strong": 1.25}
