"""Calibrated SUMO-like mandatory merge and cooperative mainline yielding.

Both pieces are active only when ``PhysicsConfig.merge_cal`` is set; the
parameters are fitted on TRAIN merge statistics and carried by the config.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .config import PhysicsConfig
from .idm import idm_accel
from .neighbours import lane_neighbour, lane_of


def merge_urgency(s: Tensor, physics: PhysicsConfig) -> Tensor:
    """Fraction of the merge zone already travelled, clamped to ``[0, 1]``."""
    span = max(physics.merge_zone_s1_m - physics.merge_zone_s0_m, 1.0)
    return ((s - physics.merge_zone_s0_m) / span).clamp(0.0, 1.0)


def merge_cal_yield(kin: Tensor, present: Tensor, accel: Tensor,
                    physics: PhysicsConfig) -> Tensor:
    """Bounded cooperative yielding of target-lane vehicles to urgent mergers."""
    p = physics
    lane = lane_of(kin)
    s, v, length = kin[..., 0], kin[..., 4], kin[..., 6]
    urgency = merge_urgency(s, p)
    urgent = present & (lane == p.merge_lane) & (urgency >= p.u_yield)
    lane1 = present & (lane == p.merge_target_lane)
    ds = s.unsqueeze(-2) - s.unsqueeze(-1)          # [B,i,j] = s_j - s_i
    valid = urgent.unsqueeze(-2) & lane1.unsqueeze(-1) & (ds > 0.0) \
        & (ds <= p.yield_lookahead_m) \
        & (v.unsqueeze(-1) >= p.yield_min_speed_mps) \
        & (v.unsqueeze(-2) >= p.yield_min_merger_speed_mps)
    gap_v = ds - length.unsqueeze(-2)
    a_v = idm_accel(v.unsqueeze(-1), v.unsqueeze(-2), gap_v, p.idm_params(),
                    accel_min=p.accel_min_mps2, accel_max=p.accel_max_mps2)
    a_v = torch.where(valid, a_v, torch.full_like(a_v, float("inf")))
    a_min = a_v.min(dim=-1).values.clamp_min(-p.b_yield_max_mps2)
    has = valid.any(dim=-1)
    return torch.where(lane1 & has, torch.minimum(accel, a_min), accel)


def merge_cal_target_logit(logits: Tensor, kin: Tensor, present: Tensor,
                           physics: PhysicsConfig, urgency: Tensor,
                           in_zone: Tensor) -> Tensor:
    """Override the merge-target logit of mandatory mergers (strategic merge).

    A merger in the zone accepts (logit 3) whenever the calibrated, urgency-relaxed
    lead/follower ceilings are satisfied and rejects (logit -8) otherwise.
    """
    p = physics
    s, v = kin[..., 0], kin[..., 4]
    lane = lane_of(kin)
    target = p.merge_target_lane
    target_lane = torch.full_like(lane, target)
    gap_l, lv_l = lane_neighbour(kin, present, target_lane, p, ahead=True)
    gap_f, fv_f = lane_neighbour(kin, present, target_lane, p, ahead=False)
    floor = p.gap_floor_m
    s0_idm = p.idm()[3]
    t_lead = p.t_lead_base_s + (p.t_lead_end_s - p.t_lead_base_s) * urgency
    b_acc = p.b_accept_base_mps2 + (p.b_accept_end_mps2 - p.b_accept_base_mps2) * urgency
    b_req_foll = (fv_f ** 2 - v ** 2) / (2.0 * gap_f.clamp_min(floor))
    b_req_lead = (v ** 2 - lv_l ** 2) / (2.0 * gap_l.clamp_min(floor))
    cal_safe = (gap_l >= torch.maximum(torch.full_like(gap_l, floor),
                                       s0_idm + v * t_lead)) \
        & (gap_f >= floor) & (gap_l >= floor) \
        & (b_req_foll <= b_acc) & (b_req_lead <= b_acc)
    mandatory = (lane == p.merge_lane) & in_zone
    logits = logits.clone()
    logits[..., target] = torch.where(
        mandatory & cal_safe, torch.full_like(s, 3.0),
        torch.where(mandatory, torch.full_like(s, -8.0), logits[..., target]))
    return logits
