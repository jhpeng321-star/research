"""Nearest same-lane / target-lane neighbours on the batched slot representation.

S-EXECPARITY3: with ``C4_SUMO_LEADER_RULE=1`` (default OFF) the neighbour
search runs in the lane-chain coordinate of
:func:`wms.physics.lane_chain.chain_s` (corridor ``s`` plus the registered
network's hidden junction-connector lengths), matching SUMO's lane/link-based
leader rule: no phantom corridor-projected leaders between the ramp and the
accel lane or across the mainline junctions, and junction-transit slots (empty
lane one-hot) are neither candidates nor egos of a lane search - they are on a
connector. With the flag off the legacy body below is byte-for-byte unchanged
(bitwise identity with the WM-S checkpoints' training-time geometry).
"""
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


def _sumo_link_geometry(kin: Tensor):
    """Chain coordinates and junction-transit mask for the SUMO leader rule."""
    from .lane_chain import chain_s
    lane = lane_of(kin)
    # Empty lane one-hot = junction transit (traffic_shepherd.c4.state.Frame.kin);
    # SUMO-wise such a slot is on a connector, not on lane 0 (argmax of zeros).
    junction_row = kin[..., 1:4].sum(-1) <= 0
    chain = chain_s(lane, kin[..., 0])
    return lane, chain, junction_row


def lane_neighbour(kin: Tensor, present: Tensor, query_lane: Tensor,
                   physics: PhysicsConfig, *, ahead: bool) -> tuple[Tensor, Tensor]:
    """Bumper gap and speed of the nearest vehicle ahead/behind in ``query_lane``.

    Missing neighbours yield the free-road gap and the vehicle's own speed.
    """
    from .lane_chain import leader_rule_enabled
    if not leader_rule_enabled():
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
    # SUMO lane/link rule: chain coordinate, junction slots on neither side.
    lane, chain, junction_row = _sumo_link_geometry(kin)
    s, v, length = chain, kin[..., 4], kin[..., 6]
    modelled = present & ~junction_row
    ds = s.unsqueeze(-2) - s.unsqueeze(-1)              # [B,i,j] = c_j - c_i
    other = modelled.unsqueeze(-2) & (lane.unsqueeze(-2) == query_lane.unsqueeze(-1))
    valid = other & ((ds > 0) if ahead else (ds < 0))
    if ahead:
        bumper = ds - length.unsqueeze(-2)
    else:
        bumper = -ds - length.unsqueeze(-1)
    gaps = torch.where(valid, bumper, torch.full_like(ds, float("inf")))
    gap, index = gaps.min(dim=-1)
    has = torch.isfinite(gap) & ~junction_row           # a junction ego has no lane
    other_v = torch.gather(v, -1, index)
    free = physics.free_road_gap_m if physics.open_road else _CLOSED_ROAD_FREE_GAP_M
    return (torch.where(has, gap.clamp_min(MIN_GAP_M), torch.full_like(gap, free)),
            torch.where(has, other_v, v))


def lead_neighbour(kin: Tensor, present: Tensor,
                   physics: PhysicsConfig) -> tuple[Tensor, Tensor]:
    """Same-lane leader gap/speed; the accel-lane end is a standing obstacle.

    Under the SUMO leader rule the dead end sits at the chain coordinate
    ``MERGE_S1 + RAMP_CONNECTOR`` so the obstacle competes with chain-coordinate
    leader gaps on the ego's own path (the ramp reaches it THROUGH the
    connector)."""
    from .lane_chain import leader_rule_enabled
    lane = lane_of(kin)
    gap, lead_v = lane_neighbour(kin, present, lane, physics, ahead=True)
    if getattr(physics, "merge_model", "legacy") == "sumo":
        # The SUMO merge adapter applies IDM::stopSpeed separately. Treating
        # a lane end as a car also adds minGap and stops cars short of the end.
        return gap, lead_v
    if leader_rule_enabled():
        from .lane_chain import MERGE_CONNECTOR_RAMP_LEN_M
        chain_lane, chain, _ = _sumo_link_geometry(kin)
        end_gap = (float(physics.merge_zone_s1_m) + float(MERGE_CONNECTOR_RAMP_LEN_M)
                   - chain).clamp_min(MIN_GAP_M)
        obstacle = (chain_lane == physics.merge_lane) & (end_gap < gap)
    else:
        end_gap = (physics.merge_zone_s1_m - kin[..., 0]).clamp_min(MIN_GAP_M)
        obstacle = (lane == physics.merge_lane) & (end_gap < gap)
    if physics.merge_cal:
        obstacle = obstacle & (end_gap <= physics.end_obstacle_range_m)
    return (torch.where(obstacle, end_gap, gap),
            torch.where(obstacle, torch.zeros_like(lead_v), lead_v))
