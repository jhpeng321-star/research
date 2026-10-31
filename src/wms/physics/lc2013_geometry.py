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
    return dict(position=position, current_length=length, target_length=through_length)
