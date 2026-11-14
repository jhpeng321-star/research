"""SLOWRES: learned slowly varying temporal residual state.

The round-5 temporal path emits its acceleration delta directly every step.  That
path carries useful low-frequency correction but also the one-step brake pulses
(``WAVE.md``).  ``model.temporal_residual_mode='slow'`` replaces the direct
emission by the first-order recurrence

    r_t = r_{t-1} + alpha * (target_t - r_{t-1}),
    alpha = dt / (tau + dt),

with a learned positive time constant ``tau`` (softplus parameterisation) and an
optional learned slew-rate cap ``|r_t - r_{t-1}| <= slew_max * dt``.  The
functions here are the pure arithmetic; the model carries ``r_t`` in its rollout
state (including across truncated-BPTT chunks) and resets it for new cars.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def inverse_softplus(value: float) -> float:
    """``x`` with ``softplus(x) == value`` (value must be positive and finite)."""
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("inverse_softplus needs a finite positive value")
    return math.log(math.expm1(value)) if value < 20.0 else value


def softplus_positive(raw: Tensor, floor: float = 1.0e-4) -> Tensor:
    """Positive scale from an unconstrained parameter."""
    return F.softplus(raw) + float(floor)


def slow_residual_step(target: Tensor, previous: Tensor, tau_s: Tensor | float,
                       dt_s: float, *, slew_max_mps2: Tensor | float | None = None) -> Tensor:
    """One step of the learned low-pass recurrence (values are accelerations)."""
    dt_s = float(dt_s)
    tau = torch.as_tensor(tau_s, dtype=target.dtype, device=target.device).clamp_min(1.0e-4)
    alpha = dt_s / (tau + dt_s)
    proposed = previous + alpha * (target - previous)
    if slew_max_mps2 is not None:
        limit = torch.as_tensor(slew_max_mps2, dtype=target.dtype,
                                device=target.device).clamp_min(0.0) * dt_s
        proposed = previous + (proposed - previous).clamp(-limit, limit)
    return proposed


__all__ = ["inverse_softplus", "softplus_positive", "slow_residual_step"]
