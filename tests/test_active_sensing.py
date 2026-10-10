"""Ported sensor regressions from test_native_damping_sumo; no old imports."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame, SumoLocalSensor


def api():
    positions = {"cav": 200, "front": 250, "occluded": 280, "far": 450, "rear": 150}
    speeds = {key: 30 for key in positions}
    reads = []
    vehicle = SimpleNamespace(
        getLaneID=lambda _: "lane0", getLanePosition=lambda v: positions[v],
        getLength=lambda _: 5.0, getAcceleration=lambda _: 0.0,
        getAllowedSpeed=lambda _: 30.0, getLaneIndex=lambda _: 0,
        getSpeed=lambda v: (reads.append(v), speeds[v])[1],
    )
    return SimpleNamespace(vehicle=vehicle, lane=SimpleNamespace(
        getLength=lambda _: 1000, getLastStepVehicleIDs=lambda _: list(positions)),
        positions=positions, speeds=speeds, reads=reads)


def test_nearest_measurements_exclude_occluded_far_and_private_state():
    connection = api()
    sensor = SumoLocalSensor(connection)
    first = sensor.capture("cav", now_s=0)
    assert first.front.gap_m == first.rear.gap_m == 45
    assert set(connection.reads) == {"cav", "front", "rear"}
    connection.speeds.update(occluded=0, far=100)
    assert sensor.capture("cav", now_s=0) == first
    assert first.lane_id == "lane0" and first.position_m == 200


def test_lane_boundary_is_partial_coverage_and_not_empty_100m():
    connection = api()
    connection.positions.clear()
    connection.positions["cav"] = 999
    frame = SumoLocalSensor(connection).capture("cav", now_s=0)
    assert frame.front is None and frame.front_coverage_m == 1
    connection.positions["cav"] = 6
    assert SumoLocalSensor(connection).capture("cav", now_s=0).rear_coverage_m == 1


def test_front_speed_is_other_minus_ego_and_distance_uses_map_only():
    connection = api()
    connection.speeds["front"] = 25
    frame = SumoLocalSensor(connection, lane_end_distance_to_merge={"lane0": 10}).capture("cav", now_s=2)
    assert frame.front.relative_speed_mps == -5
    assert frame.distance_to_merge_m == 810 and frame.timestamp_s == 2


@pytest.mark.parametrize("change", [dict(speed_mps=-1), dict(timestamp_s=float("nan")),
                                  dict(position_m=float("inf")), dict(lane_index=-1)])
def test_invalid_frame_measurements_rejected(change):
    with pytest.raises(ValueError):
        replace(SensorFrame("cav", 0, 20, 0, 30, 0), **change)


def test_overlaps_and_out_of_coverage_detections_are_not_empty_road():
    with pytest.raises(ValueError, match="overlap"):
        NeighborMeasurement(-1, 0)
    with pytest.raises(ValueError, match="coverage"):
        SensorFrame("cav", 0, 20, 0, 30, 0, front=NeighborMeasurement(20, 0), front_coverage_m=10)
