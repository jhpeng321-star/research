import pytest

from idm.spatial_response import _population, _vehicle_state


class _Snapshot:
    def __init__(self):
        self.ids = {"a", "b"}

    def has(self, vehicle_id):
        return vehicle_id in self.ids

    def position(self, vehicle_id):
        return (10.0, 2.0)

    def speed(self, vehicle_id):
        return {"a": 10.0, "b": 8.0}.get(vehicle_id)

    def accel(self, vehicle_id):
        return {"a": -4.6, "b": 1.0}.get(vehicle_id)

    def lane_pos(self, vehicle_id):
        return {"a": 20.0, "b": 30.0}.get(vehicle_id)

    def road(self, _vehicle_id):
        return "road"

    def lane_id(self, _vehicle_id):
        return "road_0"

    def leader(self, vehicle_id):
        return ("b", 4.0) if vehicle_id == "a" else None


def test_vehicle_state_has_causal_jerk_and_position():
    result = _vehicle_state(_Snapshot(), "a", {"a": -4.0})
    assert result["position_xy_m"] == [10.0, 2.0]
    assert result["jerk_mps3"] == pytest.approx(-0.6)


def test_population_records_hard_braking_and_ttc():
    result = _population(
        _Snapshot(),
        ["a", "b"],
        {"a": -4.0, "b": 0.0},
        severe_threshold_mps2=-3.0,
        hard_threshold_mps2=-4.5,
        low_ttc_threshold_s=3.0,
    )
    assert result["severe_braking_count"] == 1
    assert result["hard_braking_count"] == 1
    assert result["minimum_ttc_s"] == 2.0
    assert result["low_ttc_count"] == 1
