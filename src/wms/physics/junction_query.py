"""Exact ego-restricted evaluation of junction_merge.merge_geometry.

Sequential reservation consumes ONE ramp ego per batch row. Recomputing the
whole-fleet [B,N,N] geometry at each rank is unnecessary: [B,1,N] preserves
the identical other-vehicle order, min tie-breaking, footprint and arithmetic.
This is a tensor-query optimization, not a different merge/acceptance model.
"""
from __future__ import annotations

import torch


def merge_geometry_at(kin, present, indices, physics, driver_params=None):
    from .junction_merge import parameters, secure_gap

    s, v, length = kin[..., 0], kin[..., 4], kin[..., 6]
    lane = kin[..., 1:4].argmax(-1)
    ego_s = s.gather(-1, indices)
    ego_v = v.gather(-1, indices)
    ego_length = length.gather(-1, indices)
    delta = s.unsqueeze(-2) - ego_s.unsqueeze(-1)
    target = present & (lane == physics.merge_target_lane)
    other = torch.arange(s.shape[-1], device=s.device)
    valid = target.unsqueeze(-2) & (other != indices.unsqueeze(-1))
    inf = torch.full_like(delta, float('inf'))
    da, li = torch.where(valid & (delta >= 0), delta, inf).min(-1)
    db, fi = torch.where(valid & (delta < 0), -delta, inf).min(-1)
    gl, gf = da - length.gather(-1, li), db - ego_length
    overlap = (valid & (delta < length.unsqueeze(-2))
               & (delta > -ego_length.unsqueeze(-1))).any(-1)
    vl, vf = v.gather(-1, li), v.gather(-1, fi)
    all_params = parameters(kin, physics, driver_params)
    p = {name: value.gather(-1, indices) for name, value in all_params.items()}
    fp = {name: value.gather(-1, fi) for name, value in all_params.items()}
    lead_need = p['min_gap'] + secure_gap(ego_v, vl, p['accel'], p['decel'], p['time_headway'])
    follow_need = fp['min_gap'] + secure_gap(vf, ego_v, fp['accel'], fp['decel'], fp['time_headway'])
    ramp = present.gather(-1, indices) & (lane.gather(-1, indices) == physics.merge_lane)
    legal = ramp & (ego_s >= physics.merge_zone_s0_m) & (ego_s <= physics.merge_zone_s1_m)
    blocked_l = da.isfinite() & (gl < lead_need)
    blocked_f = db.isfinite() & (gf < follow_need)
    return dict(gap_l=gl, gap_f=gf, leader=li, follower=fi,
        has_l=da.isfinite(), has_f=db.isfinite(), overlap=overlap, v_l=vl, v_f=vf,
        ramp=ramp, legal=legal, lead_need=lead_need, follow_need=follow_need,
        blocked_l=blocked_l, blocked_f=blocked_f,
        safe=legal & ~overlap & ~blocked_l & ~blocked_f)
