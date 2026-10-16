"""sim_bridge: lazy access to the frozen controllers, frames and acceleration clamping."""
from __future__ import annotations

import pytest

from wms import sim_bridge


@pytest.fixture(scope="module")
def deps():
    try:
        return sim_bridge.load_controller_dependencies()
    except sim_bridge.SimBridgeError as exc:
        pytest.skip(f"frozen controller sources unavailable: {exc}")


def frame(deps, *, gap=30.0, rel=-2.0, speed=20.0):
    return sim_bridge.make_sensor_frame(
        deps, vehicle_id="0:1", timestamp_s=0.0, speed_mps=speed, acceleration_mps2=0.0,
        allowed_speed_mps=31.29, lane_index=0, position_m=100.0, front_gap_m=gap,
        front_relative_speed_mps=rel)


def test_missing_source_tree_is_a_bridge_error(tmp_path):
    with pytest.raises(sim_bridge.SimBridgeError, match="missing"):
        sim_bridge.load_controller_dependencies(tmp_path)


def test_unknown_kind_is_refused():
    with pytest.raises(sim_bridge.SimBridgeError, match="unknown CAV controller kind"):
        sim_bridge.make_cav_controller("wall")


def test_frame_has_leader_and_default_coverage(deps):
    built = frame(deps)
    assert built.front.gap_m == 30.0 and built.front.relative_speed_mps == -2.0
    assert built.front_coverage_m == deps.config.sensor_range_m
    no_leader = sim_bridge.make_sensor_frame(
        deps, vehicle_id="0:2", timestamp_s=1.0, speed_mps=10.0, acceleration_mps2=0.0,
        allowed_speed_mps=31.29, lane_index=1, position_m=0.0)
    assert no_leader.front is None


def test_damping_controller_keeps_one_instance_per_vehicle(deps):
    controller = sim_bridge.make_cav_controller("damping", deps)
    target = controller.target_speed("0:1", frame(deps), now_s=0.0)
    assert isinstance(target, float) and 0.0 <= target <= 31.29
    first = controller._by_vehicle["0:1"]
    controller.target_speed("0:1", frame(deps), now_s=1.0)
    assert controller._by_vehicle["0:1"] is first
    controller.invalidate("0:1")
    assert "0:1" not in controller._by_vehicle


def test_pi_controller_hands_back_without_a_leader_and_tracks_activation(deps):
    controller = sim_bridge.make_cav_controller("pi", deps)
    lone = sim_bridge.make_sensor_frame(
        deps, vehicle_id="0:1", timestamp_s=0.0, speed_mps=20.0, acceleration_mps2=0.0,
        allowed_speed_mps=31.29, lane_index=0, position_m=100.0)
    assert controller.target_speed("0:1", lone, now_s=0.0) is None
    assert controller._active == set()
    target = controller.target_speed("0:1", frame(deps), now_s=0.0)
    assert isinstance(target, float) and "0:1" in controller._active
    controller.invalidate("0:1")
    assert controller._active == set()


def test_acceleration_request_is_clamped_to_controller_limits(deps):
    cfg = deps.config
    assert sim_bridge.clamp_acceleration(deps, 25.0, 20.0) == min(cfg.acceleration_max_mps2, 5.0)
    assert sim_bridge.clamp_acceleration(deps, 0.0, 30.0) == cfg.acceleration_min_mps2
