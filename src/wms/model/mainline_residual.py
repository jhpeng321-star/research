"""S-RAMPFIX mainline interaction residual (default OFF).

RAMPORACLE/WMDIAG ranking #1: after the ramp/lanes oracles, the remaining
120/300 s paired-dJ bias is upstream+merge dominated and belongs to the MAINLINE
vehicles' response to ramp/merge interaction.  This module adds ONE learned
acceleration offset to mainline HDVs (both mainline lanes, upstream and inside
the merge zone) conditioned on local interaction state only:

* own kinematics (speed, acceleration);
* same-lane leader and follower bumper gaps / relative speeds / presence;
* adjacent-mainline-lane leader and follower gaps / relative speeds / presence;
* nearest ramp-lane car ahead (gap / relative speed / presence);
* local density and mean speed of mainline traffic ahead within the residual
  network's attention radius (geometry scale, not a behaviour window);
* distance to the merge-zone start/end (road geometry);
* ramp queue state: number of ramp-lane cars, their mean slowness and how far
  back the queue reaches (per-row features, identical for every slot).

Design contract (mirrors :mod:`wms.model.follower_courtesy`):

* every input is a smoothly normalised current-state quantity: no threshold,
  bin, window or indicator of a behavioural regime (``asinh`` scales follow the
  courtesy module); the only clamps are saturation-free normalisation bounds;
* the output is ``scale * tanh(MLP(features))`` with the last layer
  zero-initialised, so a freshly enabled head is an EXACT no-op and old
  checkpoints load unchanged;
* the offset is added to the combined learned ``accel_delta`` BEFORE the
  existing gate, soft bound / cap and plant clamp, i.e. it is gated and bounded
  exactly like the existing residual;
* CAV slots are excluded by the rollout's ``learned`` mask (the offset is added
  to ``accel_delta``, which only applies where ``learned``); the head itself
  masks to mainline HDV geometry (lanes 0/1, ``s <= merge_zone_s1_m``: upstream
  and merge zone, not downstream);
* inputs keep their autograd graph: rollout losses (loop fine-tune, long-run
  paired loss) train the head through both its weights and the state.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from wms.physics.neighbours import MIN_GAP_M, lane_neighbour, lane_of

#: Per-slot input fields, in network-input order.
MAINLINE_FIELDS = (
    "v",                 # own speed / 20 m/s
    "a",                 # own acceleration / 3 m/s^2
    "gap_l",             # asinh(same-lane leader bumper gap / 10 m) / 3
    "dv_l",              # (v_leader - v) / 5 m/s
    "has_l",             # a same-lane leader exists
    "gap_f",             # asinh(same-lane follower bumper gap / 10 m) / 3
    "dv_f",              # (v_follower - v) / 5 m/s
    "has_f",             # a same-lane follower exists
    "gap_al",            # asinh(adjacent-mainline-lane leader gap / 10 m) / 3
    "dv_al",             # (v_adj_leader - v) / 5 m/s
    "has_al",
    "gap_af",            # asinh(adjacent-mainline-lane follower gap / 10 m) / 3
    "dv_af",             # (v_adj_follower - v) / 5 m/s
    "has_af",
    "gap_r",             # asinh(nearest ramp-lane car ahead gap / 10 m) / 3
    "dv_r",              # (v_ramp_ahead - v) / 5 m/s
    "has_r",
    "dens_ahead",        # mainline vehicles ahead within 150 m / 10
    "vspd_ahead",        # their mean speed / 20 m/s (0 when none)
    "d_merge_start",     # asinh((merge_zone_s0_m - s) / 50 m) / 3
    "d_merge_end",       # asinh((merge_zone_s1_m - s) / 50 m) / 3
    "queue_n",           # ramp-lane cars at or upstream of the zone end / 10
    "queue_slow",        # their mean clamp(1 - v / 8, 0, 1) (0 when none)
    "queue_extent",      # asinh((zone end - rearmost ramp car) / 50 m) / 3 (0 when none)
)
MAINLINE_F = len(MAINLINE_FIELDS)

#: Look-ahead radius of the local density / speed features (m; geometry scale of
#: the residual network's neighbourhood, not a behavioural threshold).
LOOKAHEAD_M = 150.0


def mainline_lane_mask(kin: Tensor, present: Tensor, physics) -> Tensor:
    """Mainline HDV geometry mask ``[B, K]``: present, lane 0/1, upstream or merge zone.

    ``s <= merge_zone_s1_m`` covers upstream (``s < 0``) and the merge zone; the
    downstream road is excluded because the oracle attribution locates the
    remaining bias upstream and in the merge zone.
    """
    lane = lane_of(kin)
    s = kin[..., 0]
    main = torch.zeros_like(s, dtype=torch.bool)
    main = main | (lane == 0) | (lane == 1)
    return (present.bool() & main & (s <= float(getattr(physics, "merge_zone_s1_m", 75.44))))


def mainline_interaction_features(kin: Tensor, present: Tensor, physics) -> Tensor:
    """Per-slot interaction features ``[B, K, MAINLINE_F]`` from the step-start state.

    Everything is computed from the current slot kinematics exactly as the
    courtesy pair features are: the same function serves the teacher-forced
    replay (recorded state) and the rollout (rolled-out state), and the returned
    tensor keeps the input graph so rollout losses reach the state through it.
    """
    lane = lane_of(kin)
    s, v, a = kin[..., 0], kin[..., 4], kin[..., 5]
    present = present.bool()

    free = (float(getattr(physics, "free_road_gap_m", 250.0))
            if bool(getattr(physics, "open_road", True)) else 1000.0)

    def neighbour(query_lane: Tensor, *, ahead: bool):
        gap, other_v = lane_neighbour(kin, present, query_lane, physics, ahead=ahead)
        # Missing neighbours are filled with the free-road gap; a real gap beyond
        # it is free-road-equivalent (the physics' own convention).
        has = gap < free
        return gap, other_v, has

    # Same-lane and adjacent-mainline-lane leader / follower.
    other_main = 1 - lane
    gap_l, v_l, has_l = neighbour(lane, ahead=True)
    gap_f, v_f, has_f = neighbour(lane, ahead=False)
    gap_al, v_al, has_al = neighbour(other_main, ahead=True)
    gap_af, v_af, has_af = neighbour(other_main, ahead=False)
    ramp_lane = torch.full_like(lane, int(getattr(physics, "merge_lane", 2)))
    gap_r, v_r, has_r = neighbour(ramp_lane, ahead=True)

    # Local mainline traffic ahead within the lookahead radius (any mainline lane).
    ds = s.unsqueeze(-2) - s.unsqueeze(-1)                      # [B,i,j] = s_j - s_i
    main_j = present.unsqueeze(-2) & ((lane.unsqueeze(-2) == 0) | (lane.unsqueeze(-2) == 1))
    ahead_near = main_j & (ds > 0) & (ds <= LOOKAHEAD_M)
    count = ahead_near.sum(-1).to(s.dtype)
    vsum = (ahead_near.to(s.dtype) * v.unsqueeze(-2)).sum(-1)
    vmean = vsum / count.clamp_min(1.0)
    dens_ahead = count / 10.0
    vspd_ahead = torch.where(count > 0, vmean / 20.0, torch.zeros_like(s))

    # Merge geometry.
    s0 = float(getattr(physics, "merge_zone_s0_m", 0.0))
    s1 = float(getattr(physics, "merge_zone_s1_m", 75.44))
    d_start = torch.asinh((s0 - s) / 50.0) / 3.0
    d_end = torch.asinh((s1 - s) / 50.0) / 3.0

    # Ramp queue state (per-row): cars on the ramp lane at or upstream of the zone end.
    ramp = present & (lane == int(getattr(physics, "merge_lane", 2))) & (s <= s1)
    n_queue = ramp.sum(-1, keepdim=True).to(s.dtype)
    slow = (1.0 - v / 8.0).clamp(0.0, 1.0)
    slow_sum = (ramp.to(s.dtype) * slow).sum(-1, keepdim=True)
    queue_slow = slow_sum / n_queue.clamp_min(1.0)
    queue_slow = torch.where(n_queue > 0, queue_slow, torch.zeros_like(queue_slow))
    rear = torch.where(ramp, s, torch.full_like(s, s1)).amin(dim=-1, keepdim=True)
    queue_extent = torch.where(
        n_queue > 0, torch.asinh((s1 - rear).clamp_min(MIN_GAP_M) / 50.0) / 3.0,
        torch.zeros_like(rear))
    # Per-row queue scalars broadcast onto every slot of the row.
    n_queue = n_queue.expand_as(s)
    queue_slow = queue_slow.expand_as(s)
    queue_extent = queue_extent.expand_as(s)

    def gap_feat(gap: Tensor) -> Tensor:
        return torch.asinh(gap.clamp_min(MIN_GAP_M) / 10.0) / 3.0

    feats = torch.stack((
        (v / 20.0).clamp(-1.0, 2.0),
        (a / 3.0).clamp(-3.0, 3.0),
        gap_feat(gap_l), ((v_l - v) / 5.0).clamp(-4.0, 4.0), has_l.to(s.dtype),
        gap_feat(gap_f), ((v_f - v) / 5.0).clamp(-4.0, 4.0), has_f.to(s.dtype),
        gap_feat(gap_al), ((v_al - v) / 5.0).clamp(-4.0, 4.0), has_al.to(s.dtype),
        gap_feat(gap_af), ((v_af - v) / 5.0).clamp(-4.0, 4.0), has_af.to(s.dtype),
        gap_feat(gap_r), ((v_r - v) / 5.0).clamp(-4.0, 4.0), has_r.to(s.dtype),
        dens_ahead, vspd_ahead,
        d_start, d_end,
        (n_queue / 10.0).clamp(0.0, 3.0), queue_slow, queue_extent,
    ), dim=-1)
    return torch.where(present.unsqueeze(-1), feats, torch.zeros_like(feats))


class MainlineInteractionResidual(nn.Module):
    """Zero-initialised per-slot acceleration offset for mainline HDVs.

    ``forward`` returns ``{"accel_delta_mps2": [B, K]}`` — exactly zero on every
    non-mainline / absent slot and (before training) on the mainline slots too,
    because the read-out layer starts at zero.
    """

    def __init__(self, hidden: int = 64, scale_mps2: float = 1.5) -> None:
        super().__init__()
        self.scale_mps2 = float(scale_mps2)
        self.net = nn.Sequential(
            nn.Linear(MAINLINE_F, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: Tensor, mask: Optional[Tensor] = None) -> dict[str, Tensor]:
        delta = self.scale_mps2 * torch.tanh(self.net(features).squeeze(-1))
        if mask is not None:
            delta = delta * mask.to(delta.dtype)
        return {"accel_delta_mps2": delta}
