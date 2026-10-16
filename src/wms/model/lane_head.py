"""History-conditioned lane-change hazard head, its context and rare-event loss."""
from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from wms.physics.config import PhysicsConfig
from wms.physics.neighbours import lane_neighbour, lane_of

#: Width of the local context vector built by :func:`lane_context`: kinematics (8),
#: 3 lanes x (leader/follower gap + relative speed) = 12, merge-end distance and
#: dwell (2), baseline MOBIL logits (3).
LANE_CONTEXT_F = 23
#: Lane-change action classes: stay, left (lane - 1), right (lane + 1).
N_LANE_ACTIONS = 3
#: A change must have dwelt this many steps in its lane (except mandatory merges).
MIN_LANE_DWELL_STEPS = 6
#: Distance before the merge end within which a merge is mandatory (m).
MANDATORY_MERGE_WINDOW_M = 25.0


class HistoryLaneHead(nn.Module):
    """Predict stay/left/right from the pre-t0 driver embedding and local context."""

    def __init__(self, history_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(history_dim + LANE_CONTEXT_F, hidden), nn.GELU(),
                                 nn.Linear(hidden, N_LANE_ACTIONS))
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        with torch.no_grad():
            final.bias.copy_(torch.tensor((0.0, -3.5, -3.5)))

    def forward(self, history_embedding: Tensor, context: Tensor,
                feasible: Tensor) -> Tensor:
        """Logits ``[B, K, 3]``; infeasible directional actions are hard masked."""
        if history_embedding.shape[:-1] != context.shape[:-1]:
            raise ValueError("history embedding/context batch dimensions differ")
        if context.shape[-1] != LANE_CONTEXT_F \
                or feasible.shape != (*context.shape[:-1], N_LANE_ACTIONS):
            raise ValueError("lane context must end in 23 and feasible in 3")
        logits = self.net(torch.cat((history_embedding, context), dim=-1))
        return logits.masked_fill(~feasible.bool(), torch.finfo(logits.dtype).min)


def lane_context(kin: Tensor, present: Tensor, lane_base: Tensor,
                 physics: PhysicsConfig, driver_params: Optional[Mapping[str, Tensor]],
                 lane_age: Optional[Tensor]) -> tuple[Tensor, Tensor]:
    """Local context ``[B, K, 23]`` and feasible-action mask ``[B, K, 3]`` for the head.

    The context holds the vehicle's kinematics, the leader/follower gap and relative
    speed in each of the three lanes, the distance to the merge end, the dwell in the
    current lane and the baseline lane logits.  The mask allows *stay*, *left* and
    *right* only when the target lane is safe and the vehicle has dwelt long enough;
    the mandatory merge near the lane end forces the merge when it is possible.
    """
    s, v = kin[..., 0], kin[..., 4]
    lane = lane_of(kin)
    chunks = [s.unsqueeze(-1) / 250.0, v.unsqueeze(-1) / 30.0,
              kin[..., 5:6] / 3.0, kin[..., 1:4]]
    target_safe = torch.zeros_like(lane_base, dtype=torch.bool)
    min_gap = physics.lane_safe_gap_m if driver_params is None else driver_params["merge_gap"]
    for target in range(3):
        target_lane = torch.full_like(lane, target)
        lead_gap, lead_v = lane_neighbour(kin, present, target_lane, physics, ahead=True)
        foll_gap, foll_v = lane_neighbour(kin, present, target_lane, physics, ahead=False)
        # Missing neighbours have the free-road gap and zero relative speed; clamp
        # to a finite range before feeding the head.
        chunks.extend(((lead_gap.clamp(0, 250) / 100).unsqueeze(-1),
                       ((lead_v - v) / 30).unsqueeze(-1),
                       (foll_gap.clamp(0, 250) / 100).unsqueeze(-1),
                       ((foll_v - v) / 30).unsqueeze(-1)))
        safe = (lead_gap >= min_gap) & (foll_gap >= min_gap)
        adjacent = (lane - target).abs() == 1
        mandatory = (lane == physics.merge_lane) & (s <= physics.merge_zone_s1_m)
        allowed = adjacent & safe & (lane_base[..., target] > -7.9)
        target_safe[..., target] = allowed
        if target != physics.merge_target_lane:
            target_safe[..., target] &= ~mandatory
    end_distance = (physics.merge_zone_s1_m - s).clamp(-250, 250) / 100
    age = torch.full_like(s, 999.0) if lane_age is None else lane_age
    age_feature = (age * physics.dt_s).clamp(0, 60) / 10
    context = torch.cat((*chunks, end_distance.unsqueeze(-1),
                         age_feature.unsqueeze(-1), lane_base), dim=-1)
    # Three direction classes map stay/left/right whereas the baseline logits are
    # indexed by absolute lane: translate safety by the adjacent target lane.
    action_safe = torch.zeros_like(lane_base, dtype=torch.bool)
    action_safe[..., 0] = True
    for action, target in ((1, lane - 1), (2, lane + 1)):
        valid_target = (target >= 0) & (target < 3)
        target_clamped = target.clamp(0, 2)
        action_safe[..., action] = valid_target & target_safe.gather(
            -1, target_clamped.unsqueeze(-1)).squeeze(-1)
    mandatory = ((lane == physics.merge_lane)
                 & (s >= physics.merge_zone_s1_m - MANDATORY_MERGE_WINDOW_M)
                 & (s <= physics.merge_zone_s1_m))
    action_safe[..., 1:] &= ((age >= MIN_LANE_DWELL_STEPS) | mandatory).unsqueeze(-1)
    merge_action = 1 if physics.merge_target_lane < physics.merge_lane else 2
    can_merge = action_safe[..., merge_action]
    action_safe[..., 0] &= ~(mandatory & can_merge)
    other_action = 3 - merge_action
    action_safe[..., other_action] &= ~mandatory
    return context, action_safe


def balanced_lane_loss(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Per-step class-balanced cross entropy, capped for stable rare events.

    A recorded transition whose directional action is infeasible under the hard
    feasibility mask (that logit is ``finfo.min``) is not a learnable target and is
    dropped from the loss; keeping it would overflow the weighted cross entropy.
    """
    if logits.shape[:-1] != target.shape or mask.shape != target.shape:
        raise ValueError("lane logits/target/mask shape mismatch")
    valid = mask.bool()
    chosen = logits.gather(-1, target.long().clamp(0, 2).unsqueeze(-1)).squeeze(-1)
    valid = valid & (chosen > torch.finfo(logits.dtype).min / 2)
    if not bool(valid.any()):
        # A finite zero connected to the graph: ``logits.sum() * 0`` would be
        # ``-inf * 0 = nan`` once a hard-masked logit is present.
        return (logits * 0.0).sum()
    y = target[valid].long()
    counts = torch.bincount(y, minlength=3).to(logits.dtype)
    weights = counts.sum() / (3.0 * counts.clamp_min(1.0))
    weights = weights.clamp(max=20.0)
    return F.cross_entropy(logits[valid], y, weight=weights)
