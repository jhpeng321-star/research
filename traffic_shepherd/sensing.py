"""Ideal same-lane CAV perception, extracted from native_damping{,_sumo}.

Preserves tested nearest-front/rear selection, bumper gaps and lane-end coverage.
SUMO lane enumeration is a sensor emulator only: hidden neighbor speeds and
private driver parameters never cross the frame boundary. No sensor noise or
cross-lane visibility is implied. Actor receives only its selected fields.
"""
from dataclasses import dataclass
from math import isfinite
from typing import Mapping
from traffic_shepherd.config import load_config


def _finite(**values):
    if not all(isfinite(float(v)) for v in values.values()):
        raise ValueError("nonfinite measured state")


@dataclass(frozen=True)
class NeighborMeasurement:
    """Measured bumper gap and other-minus-ego speed; no hidden type fields."""

    gap_m: float
    relative_speed_mps: float

    def __post_init__(self) -> None:
        _finite(gap=self.gap_m, relative_speed=self.relative_speed_mps)
        if self.gap_m < 0:
            raise ValueError("overlap is invalid for this measurement contract")


@dataclass(frozen=True)
class SensorFrame:
    vehicle_id: str
    timestamp_s: float
    speed_mps: float
    acceleration_mps2: float
    allowed_speed_mps: float
    lane_index: int
    distance_to_merge_m: float | None = None
    front: NeighborMeasurement | None = None
    rear: NeighborMeasurement | None = None
    front_coverage_m: float = 0.0
    rear_coverage_m: float = 0.0
    previous_command_delta: float = 0.0
    lane_id: str | None = None
    position_m: float | None = None

    def __post_init__(self) -> None:
        _finite(
            timestamp=self.timestamp_s,
            speed=self.speed_mps,
            acceleration=self.acceleration_mps2,
            allowed=self.allowed_speed_mps,
            front_coverage=self.front_coverage_m,
            rear_coverage=self.rear_coverage_m,
            command=self.previous_command_delta,
        )
        if (
            min(self.speed_mps, self.allowed_speed_mps, self.front_coverage_m, self.rear_coverage_m)
            < 0
        ):
            raise ValueError("speed and coverage must be nonnegative")
        if self.distance_to_merge_m is not None:
            _finite(distance=self.distance_to_merge_m)
        if self.position_m is not None:
            _finite(position=self.position_m)
        if self.lane_index < 0 or int(self.lane_index) != self.lane_index:
            raise ValueError("invalid measured lane index")
        for neighbor, coverage in (
            (self.front, self.front_coverage_m),
            (self.rear, self.rear_coverage_m),
        ):
            if neighbor is not None and neighbor.gap_m > coverage:
                raise ValueError("detection beyond observed coverage")


class SumoLocalSensor:
    def __init__(
        self,
        api,
        contract=None,
        lane_end_distance_to_merge: Mapping[str, float] | None = None,
    ):
        self.api = api
        self.contract = contract if contract is not None else load_config()
        self.lane_end_distance_to_merge = dict(lane_end_distance_to_merge or {})

    def capture(
        self, cav_id: str, *, now_s: float, previous_command_delta: float = 0
    ) -> SensorFrame:
        vehicle, lane = self.api.vehicle, self.api.lane
        lane_id = vehicle.getLaneID(cav_id)
        position = float(vehicle.getLanePosition(cav_id))
        length = float(vehicle.getLength(cav_id))
        lane_length = float(lane.getLength(lane_id))
        speed = float(vehicle.getSpeed(cav_id))
        limit = self.contract.sensor_range_m
        front_coverage = min(limit, max(0.0, lane_length - position))
        rear_coverage = min(limit, max(0.0, position - length))
        front_candidates, rear_candidates = [], []
        for other in lane.getLastStepVehicleIDs(lane_id):
            if other == cav_id:
                continue
            other_position = float(vehicle.getLanePosition(other))
            if other_position >= position:
                gap = other_position - float(vehicle.getLength(other)) - position
                if gap <= front_coverage:
                    front_candidates.append((gap, str(other)))
            else:
                gap = position - length - other_position
                if gap <= rear_coverage:
                    rear_candidates.append((gap, str(other)))

        def measurement(candidates):
            if not candidates:
                return None
            gap, other = min(candidates)
            # Overlaps are errors, not silently recoded as an empty road.
            return NeighborMeasurement(gap, float(vehicle.getSpeed(other)) - speed)

        distance = self.lane_end_distance_to_merge.get(lane_id)
        if distance is not None:
            distance += lane_length - position
        return SensorFrame(
            cav_id,
            now_s,
            speed,
            float(vehicle.getAcceleration(cav_id)),
            float(vehicle.getAllowedSpeed(cav_id)),
            int(vehicle.getLaneIndex(cav_id)),
            distance,
            measurement(front_candidates),
            measurement(rear_candidates),
            front_coverage,
            rear_coverage,
            previous_command_delta,
            lane_id,
            position,
        )

