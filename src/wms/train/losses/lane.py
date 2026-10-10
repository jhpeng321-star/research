"""Lane term: per-slot cross-entropy, or the class-balanced action loss of the history head."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import LossContext, LossTerm, TermValue


def balanced_lane_loss(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Per-step class-balanced cross entropy, capped for stable rare events.

    A recorded transition whose directional action is infeasible under the
    current hard feasibility mask (``HistoryLaneHead`` fills that logit with
    ``finfo.min``) is not a learnable target: including it makes the weighted
    cross-entropy overflow to ``inf``.  Such steps are dropped from the loss.
    """
    if logits.shape[:-1] != target.shape or mask.shape != target.shape:
        raise ValueError("lane logits/target/mask shape mismatch")
    valid = mask.bool()
    chosen = logits.gather(-1, target.long().clamp(0, 2).unsqueeze(-1)).squeeze(-1)
    valid = valid & (chosen > torch.finfo(logits.dtype).min / 2)
    if not bool(valid.any()):
        # A finite zero connected to the graph: ``logits.sum() * 0`` would be
        # ``-inf * 0 = nan`` once a hard-masked (``finfo.min``) logit is present.
        return (logits * 0.0).sum()
    y = target[valid].long()
    counts = torch.bincount(y, minlength=3).to(logits.dtype)
    weights = counts.sum() / (3.0 * counts.clamp_min(1.0))
    weights = weights.clamp(max=20.0)
    return F.cross_entropy(logits[valid], y, weight=weights)


class LaneLoss(LossTerm):
    """Lane classification loss; logged without its weight."""

    name = "lane"

    def __init__(self, weight: float, *, history_head: bool) -> None:
        self.weight = float(weight)
        self.history_head = history_head

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        t = ctx.targets
        logits = ctx.pred["lane_logits"]
        if self.history_head:
            prev_lane = t.states[:, ctx.c0:ctx.c1, :, 1:4].argmax(-1)
            next_lane = t.lane_target[:, ctx.c0:ctx.c1]
            action = torch.where(next_lane == prev_lane, 0,
                                 torch.where(next_lane < prev_lane, 1, 2))
            comp = balanced_lane_loss(logits, action, t.hdv_mask[:, ctx.c0:ctx.c1])
        else:
            per_slot = F.cross_entropy(logits.reshape(-1, 3),
                                       t.lane_target[:, ctx.c0:ctx.c1].reshape(-1),
                                       reduction="none").reshape_as(ctx.m)
            comp = (per_slot * ctx.w * ctx.m).sum() / t.denom
        return TermValue(self.weight * comp, log=comp)
