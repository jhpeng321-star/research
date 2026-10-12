"""MOBIL lane-change logits with the mandatory accel-lane merge."""
from __future__ import annotations

from typing import Mapping, Optional

import torch
from torch import Tensor

from .config import PhysicsConfig
from .idm import IDMParams, idm_accel
from .merge_cal import merge_cal_target_logit, merge_urgency
from .neighbours import lane_neighbour, lane_of, lead_neighbour

#: Logit of a forbidden lane change.
MASKED_LOGIT = -8.0
_DEFAULT_POLITENESS = 0.2


def _idm(physics: PhysicsConfig, v: Tensor, leader_v: Tensor, gap: Tensor,
         params: IDMParams) -> Tensor:
    return idm_accel(v, leader_v, gap, params, accel_min=physics.accel_min_mps2,
                     accel_max=physics.accel_max_mps2)


def _follower_gain(kin: Tensor, present: Tensor, lane: Tensor, target_lane: Tensor,
                   a_current: Tensor, gap_f: Tensor, follower_v: Tensor,
                   driver_params: Optional[Mapping[str, Tensor]],
                   default: IDMParams, physics: PhysicsConfig) -> tuple[Tensor, Tensor]:
    """Acceleration change of the target-lane follower caused by the lane change.

    Returns ``(follower_accel_after, follower_gain)``.
    """
    s, v = kin[..., 0], kin[..., 4]
    ds = s.unsqueeze(-2) - s.unsqueeze(-1)
    follower = (present.unsqueeze(-2) & present.unsqueeze(-1)
                & (lane.unsqueeze(-2) == target_lane.unsqueeze(-1)) & (ds < 0))
    follower_dist = torch.where(follower, -ds, torch.full_like(ds, float("inf")))
    follower_gap, follower_index = follower_dist.min(-1)
    follower_has = torch.isfinite(follower_gap)
    follower_old = torch.gather(a_current, -1, follower_index)
    if driver_params is None:
        follower_idm = default
    else:
        follower_idm = IDMParams.from_driver_params(
            {name: torch.gather(value, -1, follower_index)
             for name, value in driver_params.items()})
    a_follow = _idm(physics, follower_v, v, gap_f, follower_idm)
    gain = torch.where(follower_has, a_follow - follower_old, torch.zeros_like(a_follow))
    return a_follow, gain


def mobil_merge_logits(kin: Tensor, present: Tensor, physics: PhysicsConfig,
                       driver_params: Optional[Mapping[str, Tensor]] = None) -> Tensor:
    """MOBIL plus mandatory-merge baseline lane logits ``[B, K, 3]``.

    ``stay`` has logit 0; a change to an adjacent lane has
    ``3 tanh((incentive - threshold (1 - u)) / 4) - 2.5 (+ 6 u^2 on the accel lane)``
    and is masked (-8) unless the leader/follower gaps and the follower braking in
    the target lane are safe (the braking limit grows with merge urgency ``u``).

    With ``physics.lane_rule == 'forced_merge_only'`` every discretionary change
    is masked as well: only the mandatory ramp merge (lane ``merge_lane`` ->
    ``merge_target_lane`` inside the merge zone, with the same gap safety and
    urgency) can keep a non-masked logit (C4; owner decision 2026-10-12).
    """
    p = physics
    s, v = kin[..., 0], kin[..., 4]
    lane = lane_of(kin)
    default = p.idm_params()
    own = default if driver_params is None else IDMParams.from_driver_params(driver_params)
    gap0, lv0 = lead_neighbour(kin, present, p)
    a_cur = _idm(p, v, lv0, gap0, own)
    urgency = merge_urgency(s, p)
    threshold = p.mobil_threshold_mps2
    politeness = torch.full_like(v, _DEFAULT_POLITENESS)
    min_gap = torch.full_like(v, p.lane_safe_gap_m)
    if driver_params is not None:
        threshold = driver_params["mobil_threshold"]
        politeness = driver_params["politeness"]
        min_gap = driver_params["merge_gap"]
        urgency = (urgency * driver_params["merge_urgency"]).clamp(0.0, 1.0)
    in_zone = (s >= p.merge_zone_s0_m) & (s <= p.merge_zone_s1_m)
    b_safe = p.b_safe_base_mps2 + (p.b_safe_end_mps2 - p.b_safe_base_mps2) * urgency
    logits = torch.full((*s.shape, 3), MASKED_LOGIT, dtype=kin.dtype, device=kin.device)
    for target in range(3):
        target_lane = torch.full_like(lane, target)
        gap_l, lv_l = lane_neighbour(kin, present, target_lane, p, ahead=True)
        gap_f, fv_f = lane_neighbour(kin, present, target_lane, p, ahead=False)
        a_new = _idm(p, v, lv_l, gap_l, own)
        a_follow, follower_gain = _follower_gain(
            kin, present, lane, target_lane, a_cur, gap_f, fv_f, driver_params,
            default, p)
        adjacent = (lane - target).abs() == 1
        ramp_ok = (lane != p.merge_lane) | (in_zone if target == p.merge_target_lane
                                            else torch.zeros_like(in_zone))
        no_to_ramp = torch.full_like(lane, target) != p.merge_lane
        forced = (lane == p.merge_lane) & (target == p.merge_target_lane)
        safe = (gap_l >= min_gap) & (gap_f >= min_gap) & (a_follow >= -b_safe)
        allowed = adjacent & ramp_ok & no_to_ramp & safe
        if p.lane_rule == "forced_merge_only":
            # No discretionary changes: the mandatory merge alone stays possible
            # (ramp_ok already confines it to the in-zone lane 2 -> 1 change).
            allowed = allowed & forced
        u = urgency * forced.to(kin.dtype)
        if driver_params is None:
            score = (a_new - a_cur) - p.mobil_threshold_mps2 * (1.0 - u)
        else:
            score = (a_new - a_cur) + politeness * follower_gain - threshold * (1.0 - u)
        move = 3.0 * torch.tanh(score / 4.0) - 2.5 + 6.0 * u * u
        logits[..., target] = torch.where(allowed, move, logits[..., target])
    if p.merge_cal:
        logits = merge_cal_target_logit(logits, kin, present, p, urgency, in_zone)
    logits = logits.scatter(-1, lane.unsqueeze(-1), torch.zeros_like(s).unsqueeze(-1))
    return logits * present.unsqueeze(-1).to(kin.dtype)
