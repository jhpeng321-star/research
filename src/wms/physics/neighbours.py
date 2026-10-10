"""Nearest same-lane / target-lane neighbours on the batched slot representation."""
from __future__ import annotations

import torch
from torch import Tensor

from .config import PhysicsConfig

#: Gaps are floored at this bumper distance (m).
MIN_GAP_M = 0.3
_CLOSED_ROAD_FREE_GAP_M = 1.0e3


def lane_of(kin: Tensor) -> Tensor:
    """Lane index ``[B, K]`` from the one-hot columns of ``kin``."""
    return kin[..., 1:4].argmax(-1)


def lane_neighbour(kin: Tensor, present: Tensor, query_lane: Tensor,
                   physics: PhysicsConfig, *, ahead: bool) -> tuple[Tensor, Tensor]:
    """Bumper gap and speed of the nearest vehicle ahead/behind in ``query_lane``.

    Missing neighbours yield the free-road gap and the vehicle's own speed.
    """
    s, v, length = kin[..., 0], kin[..., 4], kin[..., 6]
    lane = lane_of(kin)
    ds = s.unsqueeze(-2) - s.unsqueeze(-1)          # [B,i,j] = s_j - s_i
    other = present.unsqueeze(-2) & (lane.unsqueeze(-2) == query_lane.unsqueeze(-1))
    valid = other & ((ds > 0) if ahead else (ds < 0))
    if ahead:
        bumper = ds - length.unsqueeze(-2)
    else:
        bumper = -ds - length.unsqueeze(-1)
    gaps = torch.where(valid, bumper, torch.full_like(ds, float("inf")))
    gap, index = gaps.min(dim=-1)
    has = torch.isfinite(gap)
    other_v = torch.gather(v, -1, index)
    free = physics.free_road_gap_m if physics.open_road else _CLOSED_ROAD_FREE_GAP_M
    return (torch.where(has, gap.clamp_min(MIN_GAP_M), torch.full_like(gap, free)),
            torch.where(has, other_v, v))


def lead_neighbour(kin: Tensor, present: Tensor,
                   physics: PhysicsConfig) -> tuple[Tensor, Tensor]:
    """Same-lane leader gap/speed; the accel-lane end is a standing obstacle."""
    lane = lane_of(kin)
    gap, lead_v = lane_neighbour(kin, present, lane, physics, ahead=True)
    end_gap = (physics.merge_zone_s1_m - kin[..., 0]).clamp_min(MIN_GAP_M)
    obstacle = (lane == physics.merge_lane) & (end_gap < gap)
    if physics.merge_cal:
        obstacle = obstacle & (end_gap <= physics.end_obstacle_range_m)
    return (torch.where(obstacle, end_gap, gap),
            torch.where(obstacle, torch.zeros_like(lead_v), lead_v))
