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
from .mobil import merge_end_blocked
from .neighbours import lane_of


def mainline_merge_yield(kin: Tensor, present: Tensor, accel: Tensor, physics: PhysicsConfig,
                         params: IDMParams, driver_decel: Optional[Tensor] = None,
                         blocked: Optional[Tensor] = None) -> Tensor:
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
    """
    p = physics
    s, v, length = kin[..., 0], kin[..., 4], kin[..., 6]
    if blocked is None:
        blocked = merge_end_blocked(kin, present, p)                  # mergers j  [B, K]
    with torch.no_grad():
        # Pair selection is discrete: no autograd graph over the [B, K, K] pair tensors.
        sd, ld = s.detach(), length.detach()
        follower = present & (lane_of(kin) == p.merge_target_lane)    # followers i [B, K]
        ds = sd.unsqueeze(-2) - sd.unsqueeze(-1)                      # [B, i, j] = s_j - s_i
        gap = ds - ld.unsqueeze(-2)                                   # i front -> j rear
        eligible = (follower.unsqueeze(-1) & blocked.unsqueeze(-2) & (gap > 0.0)
                    & (ds <= p.mainline_merge_yield_distance_m))
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
    s0_road = p.idm()[3]
    virtual_gap = torch.where(has, best - s0_road, torch.full_like(best, p.free_road_gap_m))
    virtual_v = torch.where(has, v.gather(-1, index), v)
    virtual = idm_accel(v, virtual_v, virtual_gap, params,
                        accel_min=p.accel_min_mps2, accel_max=p.accel_max_mps2)
    virtual = virtual.clamp(-p.mainline_merge_yield_decel_mps2, 0.0)
    if driver_decel is not None:
        virtual = torch.maximum(virtual, -driver_decel)
    return torch.where(has, torch.minimum(accel, virtual), accel)
