"""Bounds on the learned corrections (tanh caps and the soft residual bound)."""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor


def validate_bounds(bound: Optional[float], neg_bound: Optional[float],
                    cap: Optional[float] = None, cap_neg: Optional[float] = None) -> None:
    """Fail closed on invalid or mutually exclusive residual bounds."""
    for value in (bound, neg_bound, cap, cap_neg):
        if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
            raise ValueError("residual soft bounds must be finite and > 0")
    if neg_bound is not None and bound is None:
        raise ValueError("negative soft bound requires residual_soft_bound_mps2")
    if bound is not None and (cap is not None or cap_neg is not None):
        raise ValueError("residual soft bounds cannot be combined with legacy residual caps")


class _StableTanhBound(torch.autograd.Function):
    """``scale * tanh(x / scale)`` with a gradient that survives float32 saturation."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(x / scale)
        return scale * torch.tanh(x / scale)

    @staticmethod
    def backward(ctx, grad):
        (z,) = ctx.saved_tensors
        # 1 - tanh(z)^2 rounds to zero for float32 |z| ~ 9; sech^2 computed
        # directly keeps the small slope.  The floor only handles underflow.
        e = torch.exp(-2 * z.abs()).clamp_min(torch.finfo(z.dtype).tiny)
        return grad * (4 * e / (1 + e).square()), None


def soft_bound(x: Tensor, bound: Optional[float],
               neg_bound: Optional[float] = None) -> Tensor:
    """Smooth ``(-neg_bound, bound)`` limit of ``x``; ``None`` is the identity."""
    validate_bounds(bound, neg_bound)
    if bound is None:
        return x
    scale = torch.full_like(x, float(bound))
    if neg_bound is not None:
        scale = torch.where(x < 0, torch.full_like(x, float(neg_bound)), scale)
    return _StableTanhBound.apply(x, scale)


def bounded_value(x: Tensor, cap: Optional[float]) -> Tensor:
    """Smoothly cap ``x`` to ``(-cap, cap)`` with ``tanh``; ``None`` is a no-op."""
    if cap is None:
        return x
    if cap <= 0.0:
        raise ValueError("cap must be > 0")
    return float(cap) * torch.tanh(x / float(cap))


def bounded_value_asym(x: Tensor, cap_pos: Optional[float],
                       cap_neg: Optional[float]) -> Tensor:
    """tanh cap with separate positive/negative limits (``None`` leaves a side free).

    The two branches have unit slope at 0, so the function is C^1.
    """
    for cap in (cap_pos, cap_neg):
        if cap is not None and (not math.isfinite(float(cap)) or float(cap) <= 0.0):
            raise ValueError("caps must be finite and > 0")
    pos = x if cap_pos is None else float(cap_pos) * torch.tanh(x / float(cap_pos))
    neg = x if cap_neg is None else float(cap_neg) * torch.tanh(x / float(cap_neg))
    return torch.where(x < 0, neg, pos)


def bounded_accel_correction(raw: Tensor, *, soft_bound_pos: Optional[float],
                     soft_bound_neg: Optional[float], cap_pos: Optional[float],
                     cap_neg: Optional[float]) -> Tensor:
    """Bounded learned acceleration correction.

    The soft bound takes precedence; otherwise the (possibly asymmetric) tanh cap
    applies, and with no bounds at all the input is returned unchanged.
    """
    if soft_bound_pos is not None:
        return soft_bound(raw, soft_bound_pos, soft_bound_neg)
    if cap_neg is None:
        return bounded_value(raw, cap_pos)
    return bounded_value_asym(raw, cap_pos, cap_neg)
