"""LC2013 best-lane lengths in the frozen observable corridor frame.

Geometry only: traffic_shepherd/c4/geometry.py LANE_TABLE/HERE_OFFSETS and
rb-shared/obsstate's completed netframe contract, independently checked in
tools/rblane/receipts/GEOMETRY_AUDIT.json. No driver/type/route assignments.
SUMO LaneQ.length sums EDGE lengths, omitting internal connectors; MSEdge
uses its first lane length, so the upstream edge is 2644.22 for both lanes.
This is not remaining distance: _wantsChange subtracts posOnLane only in
its strategic tests, not its anticipation/keep-right/speed-gain tests.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .batched import MODEL_FRAME_ROAD_END_M
from .config import PhysicsConfig

UPSTREAM_POSITION_OFFSETS_M = (2645.67, 2644.22, 378.66)
UPSTREAM_EDGE_LENGTH_M = 2644.22


def lane_position_shift(s: Tensor, source: Tensor, destination: Tensor, physics) -> Tensor:
    """SUMO enterLaneAtLaneChange preserves EDGE-LOCAL position (not model s)."""
    shifts = s.new_tensor([[a - b for b in UPSTREAM_POSITION_OFFSETS_M]
                          for a in UPSTREAM_POSITION_OFFSETS_M])
    through = (source >= 0) & (source < 2) & (destination >= 0) & (destination < 2)
    return torch.where((s < physics.merge_zone_s0_m) & through,
                        shifts[source.clamp(0, 2), destination.clamp(0, 2)], 0.)


def corridor_context(s: Tensor, lane: Tensor, physics: PhysicsConfig) -> dict[str, Tensor]:
    upstream = s < physics.merge_zone_s0_m
    downstream = s > physics.merge_zone_s1_m
    offsets = s.new_tensor(UPSTREAM_POSITION_OFFSETS_M)[lane.clamp(0, 2)]
    position = torch.where(upstream, s + offsets,
                           torch.where(downstream, s - physics.merge_zone_s1_m,
                                       s - physics.merge_zone_s0_m))
    through_length = torch.where(upstream,
        UPSTREAM_EDGE_LENGTH_M + MODEL_FRAME_ROAD_END_M - physics.merge_zone_s0_m,
        torch.where(downstream,
            s.new_tensor(MODEL_FRAME_ROAD_END_M - physics.merge_zone_s1_m),
            s.new_tensor(MODEL_FRAME_ROAD_END_M - physics.merge_zone_s0_m)))
    ramp_length = torch.where(upstream,
        s.new_tensor(UPSTREAM_POSITION_OFFSETS_M[2] + physics.merge_zone_s1_m - physics.merge_zone_s0_m),
        s.new_tensor(physics.merge_zone_s1_m - physics.merge_zone_s0_m))
    length = torch.where(lane == physics.merge_lane, ramp_length, through_length)
    segment = torch.where(upstream, 0, torch.where(downstream, 2, 1))
    return dict(position=position, current_length=length, target_length=through_length,
                segment=segment)


def neighbour_displacements(s: Tensor, lane: Tensor, context, physics):
    """SUMO edge-local pair distances, including collapsed junction lengths.

    Same-edge upstream lane positions have different origins (1.45m between
    through lanes). Across edges, normal model-frame distances omit the 3m
    mainline /2.93m ramp entry link and8m downstream link; SUMO includes them
    in getLeaderOnConsecutive/getFollowersOnConsecutive bumper gaps.
    """
    segment = context['segment']
    ego, other = segment.unsqueeze(-1), segment.unsqueeze(-2)
    ds = s.unsqueeze(-2) - s.unsqueeze(-1)
    direction = torch.sign(other - ego)
    earlier_lane = torch.where(ego < other, lane.unsqueeze(-1), lane.unsqueeze(-2))
    first_link = torch.where(earlier_lane == physics.merge_lane, ds.new_tensor(2.93), ds.new_tensor(3.0))
    crosses_first = (ego == 0) != (other == 0)
    crosses_second = (ego == 2) != (other == 2)
    ds = ds + direction * (crosses_first * first_link + crosses_second * 8.0)
    raw = context['position'].unsqueeze(-2) - context['position'].unsqueeze(-1)
    ds = torch.where((ego == 0) & (other == 0), raw, ds)
    # A leader on a following NORMAL edge is inspected once the preceding
    # normal lane's end is in braking range. Internal lanes are always traversed
    # by the search, even if their added length exceeds that range (MSLane.cpp
    # 3036/3136-3161). A leader's position within that next edge is not clipped.
    incoming_link = torch.where(other == 1, first_link, 8.0)
    preceding_end = ds - context['position'].unsqueeze(-2) - incoming_link
    return ds, ego == other, preceding_end
