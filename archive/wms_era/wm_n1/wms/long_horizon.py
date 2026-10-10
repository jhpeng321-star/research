"""Opt-in long-horizon loss and checkpoint-selection contracts (CPU-testable)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

CONFIG_KEYS = ("residual_soft_bound_mps2", "residual_soft_bound_neg_mps2",
               "residual_mag_weight", "lane_rate_weight", "lane_rate_target",
               "select_by_long_horizon", "val_every")
TERMS = ("residual_mag", "lane_rate")
MONITORS = ("predicted_lane_change_rate", "raw_residual_abs_mean_mps2",
            "raw_residual_abs_p95_mps2")


def validate_bounds(bound, neg_bound, cap=None, cap_neg=None) -> None:
    for value in (bound, neg_bound, cap, cap_neg):
        if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
            raise ValueError("residual soft bounds must be finite and > 0")
    if neg_bound is not None and bound is None:
        raise ValueError("negative soft bound requires residual_soft_bound_mps2")
    if bound is not None and (cap is not None or cap_neg is not None):
        raise ValueError("residual soft bounds cannot be combined with legacy residual caps")


class _StableTanhBound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(x / scale)
        return scale * torch.tanh(x / scale)

    @staticmethod
    def backward(ctx, grad):
        (z,) = ctx.saved_tensors
        # 1-tanh(z)^2 rounds to zero for float32 |z| ~9. Computing sech^2
        # directly retains the small slope. The tiny floor handles underflow;
        # it affects only derivatives below the dtype's normal numeric range.
        e = torch.exp(-2 * z.abs()).clamp_min(torch.finfo(z.dtype).tiny)
        return grad * (4 * e / (1 + e).square()), None


def soft_bound(x: Tensor, bound: float | None,
               neg_bound: float | None = None) -> Tensor:
    validate_bounds(bound, neg_bound)
    if bound is None:
        return x
    scale = torch.full_like(x, float(bound))
    if neg_bound is not None:
        scale = torch.where(x < 0, torch.full_like(x, float(neg_bound)), scale)
    return _StableTanhBound.apply(x, scale)


def residual_mag_loss(raw: Tensor, valid: Tensor, weight: float) -> Tensor:
    selected = raw.masked_select(valid)
    return float(weight) * selected.square().sum() / max(selected.numel(), 1)


def lane_rate_loss(probabilities: Tensor, valid: Tensor, weight: float,
                   target: float = 0.0123) -> Tensor:
    selected = probabilities.masked_select(valid)
    if selected.numel() == 0:
        return selected.sum() * 0.0
    return float(weight) * (selected.mean() - float(target)).square()


def validation_horizons(stage_s: float) -> tuple[float, ...]:
    return tuple(dict.fromkeys((60.0, float(stage_s)))) if stage_s >= 60 else (float(stage_s),)


@dataclass
class LongHorizonSelection:
    """Scores are comparable only at the longest eligible horizon reached."""

    horizon_s: float = 0.0
    score: float = float("inf")
    best: dict | None = None

    def consider(self, record: dict) -> bool:
        horizon = float(record["horizon_s"])
        score = float(record["mean_position"])
        stage = float(record["stage_s"])
        if (not math.isfinite(stage) or not math.isfinite(horizon)
                or stage < 60 or horizon < 60
                or not math.isfinite(score) or record.get("valid_targets", 0) <= 0):
            return False
        if horizon > self.horizon_s or (horizon == self.horizon_s and score < self.score):
            self.horizon_s, self.score, self.best = horizon, score, dict(record)
            return True
        return False
