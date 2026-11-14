"""Differentiable Intelligent Driver Model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from torch import Tensor


@dataclass(frozen=True)
class IDMParams:
    """IDM parameters; every field is a float or a tensor broadcastable to ``[B, K]``."""

    v0: float | Tensor
    time_headway: float | Tensor
    s0: float | Tensor
    a_max: float | Tensor
    b_comf: float | Tensor
    delta: int = 4

    @classmethod
    def from_driver_params(cls, params: Mapping[str, Tensor], delta: int = 4) -> "IDMParams":
        """Per-driver parameters inferred by the driver encoder."""
        return cls(v0=params["v0"].clamp_min(1.0e-3),
                   time_headway=params["time_headway"], s0=params["min_gap"],
                   a_max=params["accel"], b_comf=params["decel"], delta=delta)


def idm_accel(v: Tensor, leader_v: Tensor, gap: Tensor, p: IDMParams, *,
              accel_min: float, accel_max: float) -> Tensor:
    """IDM acceleration clamped to ``[accel_min, accel_max]`` (m/s^2).

    ``gap`` is the bumper gap and is floored at 0.3 m.
    """
    closing = v - leader_v
    desired = p.s0 + (v * p.time_headway
                      + v * closing / (2.0 * (p.a_max * p.b_comf) ** 0.5)).clamp(min=0.0)
    accel = p.a_max * (1.0 - (v / p.v0).pow(p.delta)
                       - (desired / gap.clamp_min(0.3)).pow(2))
    return accel.clamp(accel_min, accel_max)
