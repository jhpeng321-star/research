"""SLOWRES / astra #3: free-rollout onset-statistics auxiliary.

The C6 diagnosis measures braking onsets as fresh steps with ``a < -1.5 m/s^2`` and
characterises an event by its duration (consecutive braking steps).  The WM's
free rollout over-produces one-step onsets.  This chunk term matches the rollout's
onset rate, one-step share and mean duration to the recorded chunk's statistics
using a differentiable soft-onset surrogate.

The threshold is the measurement definition already used by C6
(``onset_decel_mps2``); ``temperature_mps2`` only controls the sigmoid softness and
is not a behaviour rule.  Default weight 0: the term is never built, so a
checkpoint loaded with the existing configuration is bit-identical.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import LossContext, LossTerm, TermValue


@dataclass(frozen=True)
class OnsetStatsConfig:
    """Hyper-parameters of the onset-statistics auxiliary (all default OFF)."""

    weight: float = 0.0
    onset_decel_mps2: float = 1.5
    temperature_mps2: float = 0.25
    max_duration_s: float = 10.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(float(x)) for x in (
                self.weight, self.onset_decel_mps2, self.temperature_mps2,
                self.max_duration_s)):
            raise ValueError("onset_stats hyper-parameters must be finite")
        if float(self.weight) < 0.0:
            raise ValueError("onset_stats.weight must be >= 0 (0 = off)")
        if float(self.onset_decel_mps2) <= 0.0 or float(self.temperature_mps2) <= 0.0:
            raise ValueError("onset_stats thresholds / temperature must be positive")
        if not 2.0 <= float(self.max_duration_s) <= 20.0:
            raise ValueError("onset_stats.max_duration_s must be in [2, 20] seconds")

    @property
    def enabled(self) -> bool:
        return float(self.weight) > 0.0

    def duration_steps_for(self, dt_s: float) -> int:
        return max(1, int(round(float(self.max_duration_s) / float(dt_s))))


def _shift_left(x: Tensor, steps: int) -> Tensor:
    """``out[:, t] = x[:, t + steps]`` (zero beyond the chunk)."""
    steps = int(steps)
    out = torch.zeros_like(x)
    if steps <= 0:
        return x
    if steps < x.shape[1]:
        out[:, :x.shape[1] - steps] = x[:, steps:]
    return out


def _pred_accel(ctx: LossContext) -> Tensor:
    sequence = torch.cat((ctx.current["v_mps"].unsqueeze(1), ctx.pred["speeds"]), dim=1)
    return (sequence[:, 1:] - sequence[:, :-1]) / float(ctx.targets.dt_s)


def _soft_stats(braking: Tensor, valid: Tensor, previous: Tensor, *,
                max_steps: int) -> tuple[Tensor, Tensor, Tensor]:
    """Soft onset rate (per 1000 HDV steps), one-step share and mean duration."""
    prev_seq = torch.cat((previous.unsqueeze(1), braking[:, :-1]), dim=1)
    onset = braking * (1.0 - prev_seq)
    weights = onset * valid.to(onset.dtype)
    mass = weights.sum().clamp_min(1.0e-6)
    rate = 1000.0 * mass / valid.to(onset.dtype).sum().clamp_min(1.0)
    one_step = (weights * (1.0 - _shift_left(braking, 1))).sum() / mass
    duration = torch.ones_like(braking)
    running = torch.ones_like(braking)
    for step in range(1, int(max_steps)):
        running = running * _shift_left(braking, step)
        duration = duration + running
    mean_duration = (weights * duration).sum() / mass
    return rate, one_step, mean_duration


class OnsetStatsLoss(LossTerm):
    """Match free-rollout onset incidence / one-step share / duration to the recording."""

    name = "onset_stats"

    def __init__(self, config: OnsetStatsConfig) -> None:
        self.config = config

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        if not self.config.enabled:
            return None
        dt_s = float(ctx.targets.dt_s)
        max_steps = self.config.duration_steps_for(dt_s)
        a_pred = _pred_accel(ctx)
        a_rec = ctx.targets.states[:, ctx.c0 + 1:ctx.c1 + 1, :, 5]
        valid = (ctx.targets.hdv_mask[:, ctx.c0:ctx.c1]
                 & ~ctx.targets.is_cav[:, ctx.c0 + 1:ctx.c1 + 1]
                 & ctx.targets.loss_valid[:, ctx.c0 + 1:ctx.c1 + 1])
        theta = float(self.config.onset_decel_mps2)
        temperature = float(self.config.temperature_mps2)
        braking = torch.sigmoid((-a_pred - theta) / temperature)
        previous = ctx.current.get("a_mps2")
        if previous is None:
            previous = torch.zeros_like(a_pred[:, 0])
        else:
            previous = previous.to(a_pred.dtype)
        previous = torch.sigmoid((-previous - theta) / temperature)
        pred = _soft_stats(braking, valid, previous, max_steps=max_steps)

        # Recorded chunk statistics with the same measurement definition (detached).
        with torch.no_grad():
            a_prev = (ctx.targets.states[:, ctx.c0, :, 5] if int(ctx.c0) > 0
                      else torch.zeros_like(a_rec[:, 0]))
            rec_braking = (a_rec < -theta).to(a_pred.dtype)
            rec_previous = (a_prev < -theta).to(a_pred.dtype)
            rec = _soft_stats(rec_braking, valid, rec_previous, max_steps=max_steps)
        loss = (F.smooth_l1_loss(pred[0] / 100.0, rec[0].detach() / 100.0, beta=0.1)
                + F.smooth_l1_loss(pred[1], rec[1].detach(), beta=0.1)
                + F.smooth_l1_loss(pred[2] / 10.0, rec[2].detach() / 10.0, beta=0.1)) / 3.0
        return TermValue(float(self.config.weight) * loss, log=loss)


def onset_stats_terms(config: OnsetStatsConfig) -> list[LossTerm]:
    """The enabled onset-statistics term(s); empty when the weight is 0."""
    return [OnsetStatsLoss(config)] if config.enabled else []


__all__ = ["OnsetStatsConfig", "OnsetStatsLoss", "onset_stats_terms"]
