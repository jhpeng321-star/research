"""V10NEGOT matched non-yield negatives for the merge-negotiation block.

For every recorded ramp car near the lane end with a lane-1 follower, the follower's
recorded acceleration one step later gives the yield label:

* follower decelerating (< ``yield_decel_mps2``) -> positive (the follower yielded);
* otherwise -> a matched non-yield negative (same geometry, no courtesy response).

The negotiation block's ``yield_logit`` is trained with BCE on that label, and its
merge-logit correction gets a hinge: positive on a recorded merge with a yielding
follower, and ``softplus(delta)`` on a matched non-yield negative (do not open the gap
without the follower's response).  All inputs are the current recorded state, so this
is a teacher-forced auxiliary next to the free-rollout and merge-survival losses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import LossContext, LossTerm, TermValue, UnrollTargets

TERM_NAME = "merge_negotiation"


@dataclass(frozen=True)
class MergeNegotiationLossConfig:
    """Weight and the matching geometry / yield threshold."""

    weight: float = 0.0
    yield_decel_mps2: float = -0.5
    behind_m: float = 40.0
    dend_m: float = 60.0
    merge_lane: int = 2
    target_lane: int = 1

    def __post_init__(self) -> None:
        if not (math.isfinite(float(self.weight)) and self.weight >= 0.0):
            raise ValueError("merge negotiation weight must be finite and non-negative")
        if self.behind_m <= 0.0 or self.dend_m <= 0.0:
            raise ValueError("merge negotiation matching distances must be positive")


def _lane(kin: Tensor) -> Tensor:
    onehot = kin[..., 1:4]
    lane = onehot.argmax(-1)
    return torch.where(onehot.sum(-1) > 0, lane, torch.full_like(lane, -1))


class MergeNegotiationTerm(LossTerm):
    """BCE on the follower-yield label + a merge-delta hinge on matched negatives."""

    name = TERM_NAME

    def __init__(self, cfg: MergeNegotiationLossConfig) -> None:
        self.cfg = cfg

    def prepare(self, targets: UnrollTargets) -> None:
        cfg, n = self.cfg, targets.n_steps
        with torch.no_grad():
            rec = targets.states[:, :n + 1].float()
            k = rec.shape[2]
            lanes = _lane(rec)
            valid = targets.loss_valid[:, :n + 1, :k].bool() & ~targets.is_cav[:, :n + 1, :k]
            s, a = rec[..., 0], rec[..., 5]
            ramp = valid[:, :n] & (lanes[:, :n] == cfg.merge_lane)
            end = 75.44
            near = ramp & ((end - s[:, :n]) < float(cfg.dend_m))
            target = valid[:, :n] & (lanes[:, :n] == cfg.target_lane)
            ds = s[:, :n].unsqueeze(-1) - s[:, :n].unsqueeze(-2)       # [B, t, ramp, target]
            behind = (target.unsqueeze(-2) & (ds >= 0.0) & (ds <= float(cfg.behind_m)))
            gap = torch.where(behind, ds, torch.full_like(ds, float("inf")))
            gap_min, idx = gap.min(-1)
            has = torch.isfinite(gap_min) & near
            safe_idx = idx.clamp(0, max(k - 1, 0))
            f_acc = a[:, 1:n + 1].gather(-1, safe_idx)                 # next-step accel
            yield_label = (f_acc < float(cfg.yield_decel_mps2)) & has
            merged = (lanes[:, 1:n + 1] == cfg.target_lane) & ramp
            targets.prepared["merge_negotiation"] = {
                "valid": has, "yield": yield_label, "merged": merged,
                "n_valid": float(has.sum()), "n_pos": float(yield_label.sum()),
                "n_merge_pos": float((yield_label & merged).sum()),
                "n_neg": float((has & ~yield_label).sum())}

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        cfg, prep = self.cfg, ctx.targets.prepared.get("merge_negotiation")
        pred = ctx.pred
        logit = pred.get("merge_negotiation_yield_logit")
        delta = pred.get("merge_negotiation_merge_delta")
        if prep is None or logit is None or delta is None:
            return None
        c0, c1 = ctx.c0, ctx.c1
        k = prep["valid"].shape[-1]
        logit = logit[:, :, :k]
        delta = delta[:, :, :k]
        valid = prep["valid"][:, c0:c1]
        y = prep["yield"][:, c0:c1].to(logit.dtype)
        merged = prep["merged"][:, c0:c1]
        if not bool(valid.any()):
            return None
        bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
        bce = torch.where(valid, bce, torch.zeros_like(bce)).sum() / valid.sum().clamp_min(1)
        neg = valid & ~prep["yield"][:, c0:c1] & ~merged
        pos = valid & prep["yield"][:, c0:c1] & merged
        hinge = (torch.where(neg, F.softplus(delta), torch.zeros_like(delta)).sum()
                 + torch.where(pos, F.softplus(-delta), torch.zeros_like(delta)).sum())
        hinge = hinge / valid.sum().clamp_min(1)
        value = cfg.weight * (bce + 0.5 * hinge)
        return TermValue(value, log=bce.detach())


def merge_negotiation_terms(cfg: MergeNegotiationLossConfig) -> list[MergeNegotiationTerm]:
    """The auxiliary term (empty when its weight is 0)."""
    return [MergeNegotiationTerm(cfg)] if cfg.weight > 0.0 else []


__all__ = ["TERM_NAME", "MergeNegotiationLossConfig", "MergeNegotiationTerm",
           "merge_negotiation_terms"]

