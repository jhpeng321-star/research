"""CommandOwnership exclusivity tests (real ownership, stub controller)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from traffic_shepherd.damping import DampingDecision
from traffic_shepherd.ownership import CommandOwnership
from traffic_shepherd.sensing import SensorFrame

from inc1.tests.fake_traci import FakeTraci


class _StubController:
    def __init__(self):
        self.config = SimpleNamespace(
            control_dt_s=1.0, acceleration_min_mps2=-3.0,
            acceleration_max_mps2=1.5, sensor_range_m=100.0)

    def decide(self, frame, *, now_s):
        return DampingDecision(fallback_reason="stub_damping")


def _frame(vid, speed, now, allowed=30.0):
    return SensorFrame(vehicle_id=vid, timestamp_s=now, speed_mps=speed,
                       acceleration_mps2=0.0, allowed_speed_mps=allowed,
                       lane_index=1, lane_id="1487606179_1", position_m=100.0)


def _api():
    api = FakeTraci()
    api.set_lane("1487606179_1")
    api.set_vehicle("cav.0", lane_id="1487606179_1", position=100.0, speed=10.0,
                    type_id="cav.0")
    api.set_vehicle("cav.1", lane_id="1487606179_1", position=80.0, speed=10.0,
                    type_id="cav.0")
    return api


def test_wall_and_native_must_be_disjoint():
    api = _api()
    ownership = CommandOwnership(api, ("cav.0", "cav.1"), _StubController(),
                                 cav_speed_mode=31)
    frames = {"cav.0": _frame("cav.0", 10.0, 300.0)}
    with pytest.raises(ValueError):
        ownership.step(frames, now_s=300.0, live_ids=["cav.0"],
                       wall_commands={"cav.0": -2.0}, native_ids=("cav.0",))


def test_wall_mask_rejects_foreign_vehicle():
    api = _api()
    ownership = CommandOwnership(api, ("cav.0", "cav.1"), _StubController(),
                                 cav_speed_mode=31)
    with pytest.raises(ValueError):
        ownership.step({}, now_s=300.0, live_ids=["cav.0"],
                       wall_commands={"hdv.0": -2.0})


def test_foreign_frame_rejected():
    api = _api()
    ownership = CommandOwnership(api, ("cav.0", "cav.1"), _StubController(),
                                 cav_speed_mode=31)
    bad = {"hdv.0": _frame("hdv.0", 10.0, 300.0)}
    with pytest.raises(ValueError):
        ownership.step(bad, now_s=300.0, live_ids=["cav.0", "cav.1"])


def test_wall_issues_command_and_native_releases():
    api = _api()
    ownership = CommandOwnership(api, ("cav.0", "cav.1"), _StubController(),
                                 cav_speed_mode=31)
    frames = {"cav.0": _frame("cav.0", 10.0, 300.0),
              "cav.1": _frame("cav.1", 10.0, 300.0)}
    records = ownership.step(frames, now_s=300.0, live_ids=["cav.0", "cav.1"],
                             wall_commands={"cav.0": -2.0},
                             native_ids=("cav.1",))
    owners = {r.vehicle_id: r.owner for r in records}
    assert owners["cav.0"] == "wall"
    assert owners["cav.1"] == "native"
    assert ("cav.0", 8.0, 1.0) in api.issued       # target 10 + 1*(-2)
    records = ownership.step(frames, now_s=301.0, live_ids=["cav.0", "cav.1"],
                             wall_commands={}, native_ids=("cav.0", "cav.1"))
    assert api.released


def test_command_clock_must_advance_on_grid():
    api = _api()
    ownership = CommandOwnership(api, ("cav.0",), _StubController(),
                                 cav_speed_mode=31)
    with pytest.raises(ValueError):
        ownership.step({}, now_s=300.5, live_ids=["cav.0"])


def _mode27_api():
    api = FakeTraci()
    api.set_lane("1487606179_1")
    api.set_vehicle("cav.0", lane_id="1487606179_1", position=100.0,
                    speed=10.0, accel=0.5, type_id="cav.0")
    api.set_vehicle("cav.1", lane_id="1487606179_1", position=80.0,
                    speed=10.0, accel=0.5, type_id="cav.0")
    return api


def test_r1_command_runs_through_ownership_mode27():
    """R1's bounded -2.0 command is issued via CommandOwnership's wall path
    with chunk-4 speedMode 27 (safety checks on)."""
    from inc1 import constants as C
    from inc1 import schedules as S

    api = _mode27_api()
    ownership = CommandOwnership(api, ("cav.0", "cav.1"), _StubController(),
                                 cav_speed_mode=27)
    states = [S.CavState("cav.0", speed_mps=20.0, s_down_m=20.0, lane_phys=1)]
    plan = S.make_schedule("R1", states, 40.0).step(C.T0_S, states)
    assert plan.wall == {"cav.0": C.R1_BRAKE_MPS2}
    records = ownership.step(
        {"cav.0": _frame("cav.0", 10.0, C.T0_S)}, now_s=C.T0_S,
        live_ids=["cav.0", "cav.1"], wall_commands=plan.wall,
        native_ids=plan.native)
    assert api.speed_modes["cav.0"] == 27 and api.speed_modes["cav.1"] == 27
    assert api.issued[0][:2] == ("cav.0", 8.0)      # 10 + 1 * (-2.0)
    assert {r.vehicle_id: r.owner for r in records}["cav.0"] == "wall"


def test_r2_command_is_bounded_then_ownership_clips_it():
    """R2's native-accel bound and the frozen ownership clip [-3, 1.5] both
    apply; the issued target never exceeds the ownership ceiling."""
    from inc1 import constants as C
    from inc1 import schedules as S

    api = _mode27_api()
    ownership = CommandOwnership(api, ("cav.0",), _StubController(),
                                 cav_speed_mode=27)
    states = [S.CavState("cav.0", speed_mps=2.0, s_down_m=-500.0, lane_phys=1,
                         native_accel_mps2=2.6)]
    plan = S.make_schedule("R2", states, 40.0).step(C.T0_S, states)
    assert plan.wall == {"cav.0": 2.6}              # schedule bound
    records = ownership.step(
        {"cav.0": _frame("cav.0", 2.0, C.T0_S)}, now_s=C.T0_S,
        live_ids=["cav.0"], wall_commands=plan.wall, native_ids=plan.native)
    assert records[0].owner == "wall"
    assert records[0].issued_acceleration_mps2 == pytest.approx(1.5)  # clip
    assert api.issued[0][:2] == ("cav.0", pytest.approx(3.5))        # 2 + 1*1.5
    assert api.speed_modes["cav.0"] == 27

