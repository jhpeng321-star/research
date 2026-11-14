"""Shared-state interface tests: VehState + ObstacleRecord (zero SUMO)."""
from __future__ import annotations

from inc1_gen.states import VehState, distance_to_obstacle

from inc1_gen.tests._helpers import OBSTACLE


def test_obstacle_record_defaults_are_a_pinned_stopped_hdv():
    assert OBSTACLE.v == 0.0 and OBSTACLE.a == 0.0
    assert OBSTACLE.lane_phys == 1
    assert OBSTACLE.length_m == 5.0


def test_distance_is_s_obs_minus_s_down():
    state = VehState(vehicle_id="c", is_cav=True, lane_phys=0,
                     s_down=-60.0, v=20.0, a=0.0)
    assert OBSTACLE.distance_to(state) == 100.0
    assert distance_to_obstacle(state, OBSTACLE.s_down) == 100.0
    downstream = VehState(vehicle_id="d", is_cav=True, lane_phys=0,
                          s_down=50.0, v=20.0, a=0.0)
    assert OBSTACLE.distance_to(downstream) == -10.0


def test_state_optional_fields_default_to_unknown_leader_and_native():
    state = VehState(vehicle_id="c", is_cav=True, lane_phys=1,
                     s_down=0.0, v=10.0, a=0.5)
    assert state.leader_gap is None
    assert state.leader_v is None
    assert state.native_accel is None
    assert state.present is True
