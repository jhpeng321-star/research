"""Flip-rate lane guard: penalise the per-step probability of leaving the current lane.

Unlike the lane-rate loss (a squared deviation of the mean change probability, which
R4-DIAG-LANE found ineffective: it saturated far above target with <=0.7% gradient
share), this term is a hinge on every step's leave-lane probability above a small
allowance, so its gradient acts wherever the model over-flips.  The allowance is the
TRUE recorded training rate (LANECAL: 0.00124327163 lane changes per HDV-step on the
R3/R4 911-window TRAIN list, junction lane -1 excluded — the shared supervision mask
already drops junction steps).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .base import LossContext, LossTerm, TermValue


def flip_rate_loss(probabilities: Tensor, valid: Tensor, weight: float,
                   allowance: float) -> Tensor:
    """Weighted mean hinge of per-step leave-lane probability above ``allowance``."""
    selected = probabilities.masked_select(valid)
    if selected.numel() == 0:
        return selected.sum() * 0.0
    return float(weight) * torch.relu(selected - float(allowance)).mean()


class FlipRateLoss(LossTerm):
    """Per-chunk hinge on the per-step probability of leaving the current lane.

    Chunk-local (no whole-batch reference replay): every chunk's leave-lane
    probabilities above the allowance contribute ``weight * mean_hinge`` directly.
    """

    name = "flip_rate"

    def __init__(self, weight: float, allowance: float) -> None:
        self.weight = float(weight)
        self.allowance = float(allowance)

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        comp = flip_rate_loss(ctx.change_probabilities, ctx.valid_aux,
                              self.weight, self.allowance)
        log = comp / self.weight if self.weight else comp
        return TermValue(comp, log=log)
