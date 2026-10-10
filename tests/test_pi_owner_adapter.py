"""CV07 tests use real CommandOwnership with fake TraCI only."""
from types import SimpleNamespace

import pytest

from traffic_shepherd.ownership import CommandOwnership
from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame
from output.controller_verification_v7.pi_owner_adapter import (
    PIOwnerAdapter, PIOwnerSessionError)


def frame(vehicle, t, speed=10., gap=20.):
    return SensorFrame(vehicle, t, speed, 0., 30., 0, front=NeighborMeasurement(gap, -2.),
                       rear=NeighborMeasurement(10., 0.), front_coverage_m=100., rear_coverage_m=100.)


class FakeVehicle:
    def __init__(self, fail=None):
        self.commands, self.fail = [], fail

    def getSpeedMode(self, _vehicle):
        return 31

    def slowDown(self, vehicle, target, duration):
        if vehicle == self.fail:
            raise RuntimeError("slowDown failure")
        self.commands.append(("slowDown", vehicle, target, duration))

    def setSpeed(self, vehicle, target):
        self.commands.append(("setSpeed", vehicle, target))


def owner(adapter, vehicle):
    return CommandOwnership(SimpleNamespace(vehicle=vehicle), set(adapter.cav_ids), adapter)


def test_real_owner_writes_and_callback_commits_raw_pi_proposal():
    adapter = PIOwnerAdapter(cav_ids={"a", "b"})
    vehicle = FakeVehicle()
    own = owner(adapter, vehicle)
    records = adapter.step(own, {"a": frame("a", 1.), "b": frame("b", 1., gap=5.75)},
                           now_s=1., live_ids={"a", "b"})
    assert [r.owner for r in records] == ["damping", "damping"]
    assert [r.status for r in adapter.receipts].count("committed_raw_proposal") == 2
    assert len(vehicle.commands) == 2


def test_adapter_prepares_then_calls_real_owner_once():
    adapter = PIOwnerAdapter(cav_ids={"a", "b"})
    vehicle = FakeVehicle()
    records = adapter.step(owner(adapter, vehicle), {"a": frame("a", 1.), "b": frame("b", 1.)},
                           now_s=1., live_ids={"a", "b"})
    assert len(records) == 2 and len(vehicle.commands) == 2
    assert [r.status for r in adapter.receipts].count("committed_raw_proposal") == 2


def test_a_success_b_failure_c_never_written_and_adapter_stops():
    adapter = PIOwnerAdapter(cav_ids={"a", "b", "c"})
    vehicle = FakeVehicle(fail="b")
    own = owner(adapter, vehicle)
    with pytest.raises(RuntimeError, match="slowDown failure"):
        adapter.step(own, {v: frame(v, 1.) for v in ("a", "b", "c")}, now_s=1.,
                     live_ids={"a", "b", "c"})
    assert [x[1] for x in vehicle.commands] == ["a"]
    assert [r.vehicle_id for r in adapter.receipts] == ["a", "a"]
    assert adapter.receipts[-1].status == "committed_raw_proposal"


def test_native_and_wall_rows_are_valid_receipts_without_pi_commit():
    adapter = PIOwnerAdapter(cav_ids={"a", "b"})
    vehicle = FakeVehicle()
    own = owner(adapter, vehicle)
    adapter.invalidate(live_ids={"a", "b"}, native_ids={"a"}, wall_ids={"b"})
    records = adapter.step(own, {"a": frame("a", 1.), "b": frame("b", 1.)}, now_s=1.,
                           live_ids={"a", "b"}, native_ids={"a"}, wall_commands={"b": 0.})
    assert [r.owner for r in records] == ["native", "wall"]
    assert len(adapter.receipts) == 2 and all(r.status == "owner_row_without_pi" for r in adapter.receipts)


def test_missing_stale_departure_and_id_reuse_invalidate_then_reentry_resets():
    adapter = PIOwnerAdapter(cav_ids={"a"})
    adapter.invalidate(live_ids={"a"}, missing_ids={"a"}, stale_ids={"a"}, departed_ids={"a"}, id_reuse_ids={"a"})
    assert not adapter._initialized
    vehicle = FakeVehicle()
    adapter.step(owner(adapter, vehicle), {"a": frame("a", 2., speed=12.)},
                 now_s=2., live_ids={"a"})
    assert "a" in adapter._initialized


def test_callback_issued_mismatch_latches_stop_and_no_retry():
    adapter = PIOwnerAdapter(cav_ids={"a"})
    class BadOwner:
        controller = adapter
        cav_ids = frozenset({"a"})
        cav_speed_mode = 31
        def step(self, frames, *, now_s, live_ids, record_callback, **_kwargs):
            decision = adapter.decide(frames["a"], now_s=now_s)
            record_callback(SimpleNamespace(vehicle_id="a", timestamp_s=now_s, owner="damping",
                                            fallback_reason=None,
                                            issued_acceleration_mps2=decision.command_acceleration_mps2 + 1.,
                                            issued_target_speed_mps=decision.target_speed_mps))
    bad = BadOwner()
    with pytest.raises(PIOwnerSessionError):
        adapter.step(bad, {"a": frame("a", 1.)}, now_s=1., live_ids={"a"})
    assert adapter.stopped
