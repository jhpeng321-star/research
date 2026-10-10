"""Obstacle free-gap placement + traci insertion tests."""
from __future__ import annotations

import pytest

from inc1 import constants as C
from inc1 import obstacles as O
from inc1.tests.fake_traci import FakeTraci


def test_free_slot_exact_when_lane_empty():
    slot = O.find_free_slot([], 40.0, length=5.0)
    assert slot["position_m"] == 40.0
    assert slot["shift_m"] == 0.0
    assert slot["shifted"] is False


def test_slot_shifts_around_occupant_within_window():
    # An occupant front at 40 m occupies [35, 40]; desired 40 overlaps.
    slot = O.find_free_slot([(40.0, 5.0)], 40.0, length=5.0,
                            min_gap=15.0, window=20.0)
    assert slot["shifted"] is True
    assert abs(slot["shift_m"]) <= C.OBSTACLE_SEARCH_WINDOW_M
    rear, front = slot["position_m"] - 5.0, slot["position_m"]
    assert front + 15.0 <= 35.0 or rear - 15.0 >= 40.0


def test_no_slot_raises_when_lane_is_packed():
    packed = [(float(p), 5.0) for p in range(2, 60, 6)]
    with pytest.raises(O.ObstacleError):
        O.find_free_slot(packed, 30.0, length=5.0, min_gap=15.0, window=20.0)


def test_place_obstacle_inserts_and_logs_shift():
    api = FakeTraci()
    lane = "1487606179_1"
    api.set_lane(lane, length=80.0)
    api.set_vehicle("mainline.0", lane_id=lane, position=40.0, speed=10.0)
    record = O.place_obstacle(api, lane_id=lane, desired_s_down_m=40.0,
                              s_down_base_m=0.0, obstacle_id="incident.obstacle")
    assert record["obstacle_id"] == "incident.obstacle"
    assert "incident.obstacle" in api.vehicles
    assert api.stops and api.stops[0][3] == C.OBSTACLE_HOLD_S
    assert api.moves
    assert record["hold_s"] == C.OBSTACLE_HOLD_S


def test_incident_lane_mapping():
    assert O.incident_lane_id(40.0) == "1487606179_1"
    assert O.incident_lane_base(40.0) == 0.0
    assert O.incident_lane_id(-150.0) == "1191554636_0"
    assert O.incident_lane_base(-150.0) == C.MAIN_RIGHT_LANE_OFFSET_M
