"""Per-car trajectory terms: Huber position and speed against the recorded rollout."""
from __future__ import annotations

from typing import Optional

import torch

from .base import LossContext, LossTerm, TermValue, UnrollTargets

#: V10BRAKEWAVE: same-lane leader gap and recorded leader deceleration of the follower mask.
BRAKE_FOLLOW_MAX_GAP_M = 40.0
BRAKE_FOLLOW_LEADER_DECEL_MPS2 = -1.0


def brake_follow_mask(targets: UnrollTargets, *, max_gap_m: float = BRAKE_FOLLOW_MAX_GAP_M,
                      leader_decel_mps2: float = BRAKE_FOLLOW_LEADER_DECEL_MPS2,
                      leader_is_cav: bool = False) -> torch.Tensor:
    """[B, n_steps, K] followers whose nearest same-lane recorded leader brakes hard.

    The leader is the nearest vehicle ahead in the same recorded lane within ``max_gap_m``;
    the follower is flagged when that leader's recorded acceleration at the target step is
    below ``leader_decel_mps2``.  ``leader_is_cav`` selects CAV-led events (the leader must be
    a recorded CAV) instead of the default HDV-led events.  Computed once per unroll and
    cached on the targets.
    """
    key = "brake_follow_cav" if leader_is_cav else "brake_follow"
    cached = targets.prepared.get(key)
    if cached is not None and float(cached["max_gap_m"]) == float(max_gap_m) \
            and float(cached["leader_decel_mps2"]) == float(leader_decel_mps2):
        return cached["mask"]
    states = targets.states
    n = targets.n_steps
    accel = states[:, 1:n + 1, :, 5].to(torch.float64)
    s = states[:, 1:n + 1, :, 0].to(torch.float64)
    lane_onehot = states[:, 1:n + 1, :, 1:4]
    lane = lane_onehot.argmax(dim=-1)
    on_road = lane_onehot.sum(dim=-1) > 0
    valid = targets.loss_valid[:, 1:n + 1] & on_road
    leader_cav = targets.is_cav[:, 1:n + 1]
    leader = valid & (leader_cav if leader_is_cav else ~leader_cav) \
        & (accel < float(leader_decel_mps2))
    mask = torch.zeros(valid.shape, dtype=torch.bool, device=valid.device)
    for t in range(n):
        s_t, lane_t, valid_t, leader_t = s[:, t, :], lane[:, t, :], valid[:, t, :], leader[:, t, :]
        gap = s_t[:, None, :] - s_t[:, :, None]
        same_lane = lane_t[:, :, None] == lane_t[:, None, :]
        ahead = (valid_t[:, :, None] & valid_t[:, None, :] & same_lane
                 & (gap > 0.0) & (gap <= float(max_gap_m)))
        nearest_gap = torch.where(ahead, gap, torch.full_like(gap, float("inf")))
        nearest = nearest_gap.argmin(dim=-1)
        has_leader = torch.isfinite(nearest_gap.min(dim=-1).values)
        mask[:, t] = has_leader & leader_t.gather(1, nearest)
    mask = mask & targets.hdv_mask
    targets.prepared[key] = {
        "mask": mask, "count": int(mask.sum()),
        "max_gap_m": float(max_gap_m),
        "leader_decel_mps2": float(leader_decel_mps2)}
    return mask


def brake_follow_step_weights(targets: UnrollTargets, weight: float) -> Optional[torch.Tensor]:
    """[B, n_steps, K] multiplier (``weight`` on flagged steps, 1 elsewhere) or None if off."""
    weight = float(weight)
    if weight <= 1.0:
        return None
    mask = brake_follow_mask(targets).to(targets.states.dtype)
    return 1.0 + (weight - 1.0) * mask


def brake_follow_cav_step_weights(targets: UnrollTargets,
                                  weight: float) -> Optional[torch.Tensor]:
    """[B, n_steps, K] extra multiplier (``1 + weight`` on CAV-led steps) or None if off.

    V10CAVFOLLOW: the default ``0`` leaves the loss untouched; positive values add
    emphasis on HDV followers of a recorded CAV leader braking below -1 m/s^2.
    """
    weight = float(weight)
    if weight <= 0.0:
        return None
    mask = brake_follow_mask(targets, leader_is_cav=True).to(targets.states.dtype)
    return 1.0 + weight * mask


class _HuberTerm(LossTerm):
    """Step-weighted, valid-count-normalised Huber term logged without its weight."""

    def __init__(self, weight: float, brake_follow_weight: float = 1.0,
                 brake_follow_cav_weight: float = 0.0) -> None:
        self.weight = float(weight)
        self.brake_follow_weight = float(brake_follow_weight)
        self.brake_follow_cav_weight = float(brake_follow_cav_weight)

    def _huber(self, ctx: LossContext):
        raise NotImplementedError

    def _step_multiplier(self, ctx: LossContext):
        mult = None
        if self.brake_follow_weight > 1.0:
            mult = brake_follow_step_weights(ctx.targets, self.brake_follow_weight)
            if mult is not None:
                mult = mult[:, ctx.c0:ctx.c1]
        if self.brake_follow_cav_weight > 0.0:
            cav = brake_follow_cav_step_weights(ctx.targets, self.brake_follow_cav_weight)
            if cav is not None:
                cav = cav[:, ctx.c0:ctx.c1]
                mult = cav if mult is None else mult * cav
        return mult

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        huber = self._huber(ctx)
        mult = self._step_multiplier(ctx)          # V10BRAKEWAVE follower-chain weight
        if mult is not None:
            huber = huber * mult
        boost = ctx.courtesy_boost                 # V10COURTESY yield-event weight
        m = ctx.m if boost is None else ctx.m * boost.to(ctx.m.dtype)
        comp = (huber * ctx.w * m).sum() / ctx.targets.denom
        return TermValue(self.weight * comp, log=comp)


class PositionLoss(_HuberTerm):
    name = "position"

    def _huber(self, ctx: LossContext):
        return ctx.pos_huber


class SpeedLoss(_HuberTerm):
    name = "speed"

    def _huber(self, ctx: LossContext):
        return ctx.spd_huber
