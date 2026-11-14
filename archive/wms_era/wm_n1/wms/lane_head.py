"""History-conditioned lane-change hazard head and rare-event loss."""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class HistoryLaneHead(nn.Module):
    """Predict stay/left/right from pre-t0 driver embedding and local context."""

    def __init__(self, history_dim: int, hidden: int = 64):
        super().__init__()
        # kinematics (8), 3 lanes x (leader/follower gap + relative speed),
        # merge-end distance and dwell (2), MOBIL incentives (3).
        self.net = nn.Sequential(nn.Linear(history_dim + 23, hidden), nn.GELU(),
                                 nn.Linear(hidden, 3))
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        with torch.no_grad():
            final.bias.copy_(torch.tensor((0.0, -3.5, -3.5)))

    def forward(self, history_embedding: Tensor, context: Tensor,
                feasible: Tensor) -> Tensor:
        if history_embedding.shape[:-1] != context.shape[:-1]:
            raise ValueError("history embedding/context batch dimensions differ")
        if context.shape[-1] != 23 or feasible.shape != (*context.shape[:-1], 3):
            raise ValueError("lane context must end in 23 and feasible in 3")
        logits = self.net(torch.cat((history_embedding, context), dim=-1))
        # Directional actions are hard masked; context may hard-force the
        # mandatory merge when it is the last safe opportunity.
        feasible = feasible.bool()
        return logits.masked_fill(~feasible, torch.finfo(logits.dtype).min)


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
    # Drop steps whose target action is hard-masked as infeasible.
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
