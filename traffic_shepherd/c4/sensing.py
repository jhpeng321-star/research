"""Table-backed CAV sensor: ``SumoLocalSensor.capture`` semantics without per-vehicle calls.

``traffic_shepherd.sensing.SumoLocalSensor.capture`` issues ~2 TraCI calls per vehicle on
the CAV's lane (≈20 ms per CAV on Windows loopback).  This builds the identical
:class:`traffic_shepherd.sensing.SensorFrame` (same nearest same-lane front/rear selection,
bumper gaps, lane-end coverage, ``distance_to_merge_m``) from the per-step subscription
table of :class:`traffic_shepherd.c4.sumo_io.VehicleReader`.  Only lane lengths are
queried (once per lane id, cached).  The smoke run checks frame equality against
``SumoLocalSensor.capture`` on live CAVs.
"""
from __future__ import annotations

from collections import defaultdict

from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame


class TableSensor:
    def __init__(self, api, contract, lane_end_distance_to_merge):
        self.api = api
        self.contract = contract
        self.lane_end_distance_to_merge = dict(lane_end_distance_to_merge or {})
        self._lane_length: dict[str, float] = {}
        self._by_lane: dict = {}

    def lane_length(self, lane_id: str) -> float:
        value = self._lane_length.get(lane_id)
        if value is None:
            value = self._lane_length[lane_id] = float(self.api.lane.getLength(lane_id))
        return value

    def bind(self, table: dict) -> None:
        """Use ``table`` (the current step's vehicle table) for the following captures."""
        self.table = table
        by_lane = defaultdict(list)
        for vid, veh in table.items():
            by_lane[veh["lane_id"]].append(vid)
        self._by_lane = by_lane

    def capture(self, cav_id: str, *, now_s: float, previous_command_delta: float = 0
                ) -> SensorFrame:
        table = self.table
        me = table[cav_id]
        lane_id = me["lane_id"]
        position = me["lane_position_m"]
        length = me["length_m"]
        lane_length = self.lane_length(lane_id)
        speed = me["speed_mps"]
        limit = self.contract.sensor_range_m
        front_coverage = min(limit, max(0.0, lane_length - position))
        rear_coverage = min(limit, max(0.0, position - length))
        front, rear = [], []
        for other in self._by_lane.get(lane_id, ()):
            if other == cav_id:
                continue
            o = table[other]
            other_position = o["lane_position_m"]
            if other_position >= position:
                gap = other_position - o["length_m"] - position
                if gap <= front_coverage:
                    front.append((gap, str(other)))
            else:
                gap = position - length - other_position
                if gap <= rear_coverage:
                    rear.append((gap, str(other)))

        def measurement(candidates):
            if not candidates:
                return None
            gap, other = min(candidates)
            return NeighborMeasurement(gap, table[other]["speed_mps"] - speed)

        distance = self.lane_end_distance_to_merge.get(lane_id)
        if distance is not None:
            distance += lane_length - position
        return SensorFrame(cav_id, now_s, speed, me["acceleration_mps2"],
                           me["allowed_speed_mps"], int(me["lane_index"]), distance,
                           measurement(front), measurement(rear), front_coverage,
                           rear_coverage, previous_command_delta, lane_id, position)
