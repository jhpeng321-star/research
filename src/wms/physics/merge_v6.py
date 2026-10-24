"""V6MERGE ``mainline_merge_yield``: cooperative lane-1 slowing for a blocked lane-end merger.

A physical interaction rule applied equally to every vehicle (HDV or CAV): no controller,
no vehicle-class policy, no learned parameters.  Fully vectorised ``[B, K, K]`` (no loops
over batch or slots, no host syncs).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .config import PhysicsConfig
from .idm import IDMParams, idm_accel
from .mobil import (anticipated_kin, impatient_margins, merge_end_blocked, merge_end_waiting,
                    secure_gaps)
from .neighbours import _CLOSED_ROAD_FREE_GAP_M, MIN_GAP_M, lane_of


def _pair_neighbours(s: Tensor, v: Tensor, length: Tensor, cand: Tensor, free: float
                     ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Target-lane leader/follower bumper gaps and speeds of every ego ``j`` among its own
    candidate set ``cand [B, j, k]`` (same convention as ``lane_neighbour``)."""
    ds = s.unsqueeze(-2) - s.unsqueeze(-1)                                    # s_k - s_j
    inf = float("inf")
    gap_l, i_l = torch.where(cand & (ds > 0), ds - length.unsqueeze(-2), inf).min(-1)
    gap_f, i_f = torch.where(cand & (ds < 0), -ds - length.unsqueeze(-1), inf).min(-1)
    has_l, has_f = torch.isfinite(gap_l), torch.isfinite(gap_f)
    return (torch.where(has_l, gap_l.clamp_min(MIN_GAP_M), torch.full_like(gap_l, free)),
            torch.where(has_l, v.gather(-1, i_l), v),
            torch.where(has_f, gap_f.clamp_min(MIN_GAP_M), torch.full_like(gap_f, free)),
            torch.where(has_f, v.gather(-1, i_f), v))


def _pair_overlap(s: Tensor, length: Tensor, cand: Tensor, margin: float) -> Tensor:
    """``[B, j]``: some candidate ``k`` overlaps ego ``j``'s footprint (``target_overlap``)."""
    rear = s - length
    eye = torch.eye(s.shape[-1], dtype=torch.bool, device=s.device)
    return (cand & ~eye & (rear.unsqueeze(-2) < s.unsqueeze(-1) + margin)
            & (s.unsqueeze(-2) > rear.unsqueeze(-1) - margin)).any(-1)


def merge_feasible_among(kin: Tensor, present: Tensor, physics: PhysicsConfig,
                         cand: Tensor, impatience: Optional[Tensor] = None) -> Tensor:
    """``[B, j]``: the speed-rule merge feasibility (lane ``merge_lane`` -> ``merge_target_lane``,
    in the merge zone) of every ego ``j`` judged against the lane-1 vehicles ``cand [B, j, k]``
    only.  With ``cand`` = every present lane-1 vehicle it equals the physics mask of
    ``mobil_merge_logits`` (speed gap rule on the anticipated configuration, the stopped-merger
    relaxation, the ``merge_end_wait`` veto; tested).  Discrete: no autograd graph.
    V8WM3 ``impatience`` (level ``[B, j]``, ``None`` = legacy): the relaxed margins of
    :func:`wms.physics.mobil.speed_gap_safe` / :func:`wms.physics.mobil.secure_gaps`."""
    p = physics
    with torch.no_grad():
        kin = kin.detach()
        s, v, length = kin[..., 0], kin[..., 4], kin[..., 6]
        free = p.free_road_gap_m if p.open_road else _CLOSED_ROAD_FREE_GAP_M
        kin_a = anticipated_kin(kin, p.gap_anticipation_s)
        sa, va = kin_a[..., 0], kin_a[..., 4]
        gl, lvl, gf, fvf = _pair_neighbours(sa, va, length, cand, free)
        if impatience is None:
            req_l = p.gap_s0_lead_m + p.gap_t_lead_s * va \
                + (va * va - lvl * lvl).clamp_min(0.0) / (2.0 * p.gap_b_lead_mps2)
            req_f = p.gap_s0_follow_m + p.gap_t_follow_s * fvf \
                + (fvf * fvf - va * va).clamp_min(0.0) / (2.0 * p.gap_b_follow_mps2)
        else:
            s0l, tl, s0f, tf = impatient_margins(p, impatience)
            req_l = s0l + tl * va + (va * va - lvl * lvl).clamp_min(0.0) / (
                2.0 * p.gap_b_lead_mps2)
            req_f = s0f + tf * fvf + (fvf * fvf - va * va).clamp_min(0.0) / (
                2.0 * p.gap_b_follow_mps2)
        ov_a = _pair_overlap(sa, length, cand, p.gap_overlap_margin_m)
        safe = (gl >= req_l) & (gf >= req_f) & ~ov_a
        if p.gap_stopped_relax or p.merge_end_wait:
            glc, lvc, gfc, fvc = _pair_neighbours(s, v, length, cand, free)
        if p.gap_stopped_relax:
            req_fs = p.gap_stopped_min_m + (fvc * fvc - v * v).clamp_min(0.0) / (
                2.0 * p.gap_b_follow_mps2)
            safe = safe | ((v < p.gap_stopped_speed_mps) & (glc >= p.gap_stopped_min_m)
                           & (gfc >= req_fs)
                           & ~_pair_overlap(s, length, cand, p.gap_stopped_min_m))
        if p.merge_end_wait:
            secure = (secure_gaps(v, glc, lvc, gfc, fvc, p, impatience)
                      & ~_pair_overlap(s, length, cand, p.gap_overlap_margin_m)
                      & secure_gaps(va, gl, lvl, gf, fvf, p, impatience) & ~ov_a)
            safe = safe & ~(merge_end_waiting(kin, present, p) & ~secure)
        ramp = (present & (lane_of(kin) == p.merge_lane) & (s >= p.merge_zone_s0_m)
                & (s <= p.merge_zone_s1_m))
        return ramp & safe


def zone_mergers(kin: Tensor, present: Tensor, physics: PhysicsConfig,
                 impatience: Optional[Tensor] = None) -> Tensor:
    """V8WM2 ``mainline_merge_yield_scope='zone'``: in-zone ramp vehicles ``j`` ``[B, K]`` whose
    merge the physics mask blocks ONLY from behind / beside: blocked against every lane-1
    vehicle, but feasible once the lane-1 vehicles whose front is at or behind ``j``'s front
    and within ``mainline_merge_yield_distance_m`` are ignored (i.e. the leader side and the
    cars ahead of ``j`` already admit the merge; what blocks it is an overlapping car or the
    follower gap - the cars a zipper can open)."""
    p = physics
    with torch.no_grad():
        kin = kin.detach()
        s = kin[..., 0]
        lane1 = present & (lane_of(kin) == p.merge_target_lane)
        cand = lane1.unsqueeze(-2).expand(*lane1.shape[:-1], lane1.shape[-1], lane1.shape[-1])
        ds = s.unsqueeze(-2) - s.unsqueeze(-1)                                # s_k - s_j
        behind = (ds <= 0.0) & (-ds <= p.mainline_merge_yield_distance_m)
        blocked = ~merge_feasible_among(kin, present, p, cand, impatience)
        free_ahead = merge_feasible_among(kin, present, p, cand & ~behind, impatience)
        return blocked & free_ahead


def mainline_merge_yield(kin: Tensor, present: Tensor, accel: Tensor, physics: PhysicsConfig,
                         params: IDMParams, driver_decel: Optional[Tensor] = None,
                         blocked: Optional[Tensor] = None,
                         impatience: Optional[Tensor] = None, *,
                         learned_decel: Optional[Tensor] = None,
                         learned_distance: Optional[Tensor] = None,
                         return_selected: bool = False,
                         return_influence: bool = False):
    """``accel`` with each blocked merger acting as a virtual leader of its lane-1 follower.

    Merger ``j`` (:func:`wms.physics.mobil.merge_end_blocked`) is seen by the NEAREST present
    lane-1 vehicle ``i`` whose front is behind the merger's rear (``gap = s_j - L_j - s_i > 0``,
    so the merger is ahead of it; no phantom braking for a merger behind or already overlapped)
    and within ``mainline_merge_yield_distance_m`` (front to front).  The follower applies its
    own IDM (``params``, per-driver when learned) to a virtual leader at the merger's speed
    standing the road's ``s0`` behind the merger's rear (room for the merger's secure gap),
    never accelerating because of it and braking by at most ``mainline_merge_yield_decel_mps2``
    (and its own ``driver_decel``).  Vehicles further back react through ordinary IDM.
    ``blocked``: a precomputed :func:`merge_end_blocked` (computed here when ``None``).

    V8WM2 ``physics.mainline_merge_yield_scope = 'zone'`` (zipper; ``'lane_end'`` = the rule
    above, unchanged).  The mergers are the lane-end set above PLUS every in-zone ramp vehicle
    whose merge is blocked only from behind / beside (:func:`zone_mergers`), and:

    1. follower yield: exactly the rule above for that merger set (the nearest lane-1 vehicle
       whose front is behind the merger's rear, within the distance, virtual leader ``s0``
       behind the merger's rear, never accelerating, at most the decel bound);
    2. overlap yield (zone mergers only): a lane-1 vehicle overlapping the merger's footprint
       (``gap_overlap_margin_m``) with its front at or behind the merger's front brakes by the
       decel bound (and its own ``driver_decel``) only while the merger is FASTER than it, so
       the merger pulls ahead; a slower or stopped merger is passed (braking beside it would
       only prolong the overlap: no deadlock).  A follower facing several pulls takes the
       strongest.  Neither rule ever raises an acceleration, so no follower is pushed through a
       merger.

    V8WM3 ``impatience`` (level ``[B, K]``, ``None`` = legacy): the merger masks use the
    impatience-relaxed merge feasibility (the same as the lane-change mask).

    R13Y optional ``learned_decel`` / ``learned_distance`` (both required together):
    positive live shared tensors. Lane-end scope only, with a continuous range gate
    sigmoid((distance - front separation) / follower learned s0), and the follower's
    learned s0 as virtual clearance reserve. Only the discrete nearest-pair/mask is
    detached; no continuous state or parameter is detached. ``return_selected`` also
    returns the pair mask so the caller can compose competing acceleration caps.
    Omitting these arguments preserves the legacy operator exactly.
    """
    p = physics
    learned = learned_distance is not None
    if learned != (learned_decel is not None):
        raise ValueError("learned yield needs both distance and deceleration")
    s, v, length = kin[..., 0], kin[..., 4], kin[..., 6]
    if blocked is None:
        blocked = merge_end_blocked(kin, present, p, impatience)      # mergers j  [B, K]
    zone = not learned and getattr(p, "mainline_merge_yield_scope", "lane_end") == "zone"
    if zone:
        in_zone = zone_mergers(kin, present, p, impatience)
        blocked = blocked | in_zone
    with torch.no_grad():
        # Pair selection is discrete: no autograd graph over the [B, K, K] pair tensors.
        sd, ld = s.detach(), length.detach()
        follower = present & (lane_of(kin) == p.merge_target_lane)    # followers i [B, K]
        ds = sd.unsqueeze(-2) - sd.unsqueeze(-1)                      # [B, i, j] = s_j - s_i
        gap = ds - ld.unsqueeze(-2)                                   # i front -> j rear
        eligible = follower.unsqueeze(-1) & blocked.unsqueeze(-2) & (gap > 0.0)
        if not learned:
            eligible = eligible & (ds <= p.mainline_merge_yield_distance_m)
        inf = torch.full_like(ds, float("inf"))
        dist = torch.where(eligible, gap, inf)
        # Only the immediate follower of each merger yields ...
        nearest = dist == dist.min(dim=-2, keepdim=True).values
        dist = torch.where(eligible & nearest, dist, inf)
        # ... and a follower facing several mergers reacts to its closest one.
        best, index = dist.min(dim=-1)
        has = torch.isfinite(best)
    # Differentiable [B, K] gap to the selected merger (follower front -> merger rear).
    best = s.gather(-1, index) - length.gather(-1, index) - s
    # The virtual leader stands the road's minimum gap s0 behind the merger's rear: a follower
    # that stops for it leaves the merger's standstill secure follower gap (s0 + T*0 + 0), so
    # yielding can never itself block the merge (deadlock when the driver's own s0 < road s0).
    s0_road = params.s0 if learned else p.idm()[3]
    virtual_gap = torch.where(has, best - s0_road, torch.full_like(best, p.free_road_gap_m))
    virtual_v = torch.where(has, v.gather(-1, index), v)
    virtual = idm_accel(v, virtual_v, virtual_gap, params,
                        accel_min=p.accel_min_mps2, accel_max=p.accel_max_mps2)
    if learned:
        virtual = torch.maximum(virtual.clamp_max(0.0), -learned_decel)
    else:
        virtual = virtual.clamp(-p.mainline_merge_yield_decel_mps2, 0.0)
    if driver_decel is not None:
        virtual = torch.maximum(virtual, -driver_decel)
    out = torch.where(has, torch.minimum(accel, virtual), accel)
    if learned:
        # Continuous counterpart of the old <= distance cutoff. The transition
        # scale is a learned driver setting, not a new hand-set threshold.
        ds_live = s.gather(-1, index) - s
        weight = torch.sigmoid((learned_distance - ds_live) / params.s0)
        out = torch.where(has, accel + weight * (out - accel), accel)
    if zone:
        with torch.no_grad():
            sd, ld, vd = s.detach(), length.detach(), v.detach()
            m = p.gap_overlap_margin_m
            rear = sd - ld
            ds = sd.unsqueeze(-2) - sd.unsqueeze(-1)                  # [B, i, j] = s_j - s_i
            overlap = ((rear.unsqueeze(-1) < sd.unsqueeze(-2) + m)    # i's rear < j's front
                       & (sd.unsqueeze(-1) > rear.unsqueeze(-2) - m))  # i's front > j's rear
            pull = (follower.unsqueeze(-1) & in_zone.unsqueeze(-2) & overlap & (ds >= 0.0)
                    & (vd.unsqueeze(-2) > vd.unsqueeze(-1))).any(-1)
        brake = torch.full_like(accel, -p.mainline_merge_yield_decel_mps2)
        if driver_decel is not None:
            brake = torch.maximum(brake, -driver_decel)
        out = torch.where(pull, torch.minimum(out, brake), out)
    if return_influence:
        if not learned:
            raise ValueError("continuous influence is available only for learned yield")
        return out, has, weight
    return (out, has) if return_selected else out
