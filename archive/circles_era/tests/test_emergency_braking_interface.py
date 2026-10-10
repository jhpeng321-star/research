"""Mocked OE-31 command-interface regressions; never launches SUMO."""

from types import SimpleNamespace

import pytest

import traffic_shepherd.ownership as ownership
from traffic_shepherd.damping import DampingDecision
from traffic_shepherd.escalation_evidence import paired_effect
from traffic_shepherd.ownership import CommandOwnership


class VehicleApi:
    def __init__(self, modes):
        self.modes = dict(modes)
        self.calls = []
        self.vehicle = self

    def getSpeedMode(self, vehicle_id):
        return self.modes[vehicle_id]

    def setSpeedMode(self, vehicle_id, mode):
        self.calls.append(("mode", vehicle_id, mode))
        self.modes[vehicle_id] = mode

    def slowDown(self, vehicle_id, target, duration):
        self.calls.append(("slowDown", vehicle_id, target, duration))

    def setSpeed(self, vehicle_id, target):
        self.calls.append(("setSpeed", vehicle_id, target))


def _controller():
    config = SimpleNamespace(control_dt_s=1.0, speed_mode=31)

    class Controller:
        def __init__(self):
            self.config = config

        def decide(self, frame, *, now_s):
            return DampingDecision(command_acceleration_mps2=0.1, target_speed_mps=20.0)

    return Controller()


def _frames(*ids):
    return {vehicle_id: SimpleNamespace(vehicle_id=vehicle_id) for vehicle_id in ids}


def _patch_wall(monkeypatch):
    monkeypatch.setattr(ownership, "frame_problem", lambda *args: None)
    monkeypatch.setattr(ownership, "direct_target", lambda *args: (0.2, 21.0))


def test_default31_keeps_existing_mode_and_writes_only_cav_command():
    api = VehicleApi({"cav": 31, "hdv": 31})
    owner = CommandOwnership(api, {"cav"}, _controller())
    rows = owner.step(_frames("cav"), now_s=0.0, live_ids={"cav", "hdv"})
    assert rows[0].owner == "damping"
    assert api.calls == [("slowDown", "cav", 20.0, 1.0)]


def test_explicit27_configures_all_current_and_late_cavs_but_never_hdv():
    api = VehicleApi({"a": 31, "b": 31, "late": 31, "hdv": 31})
    owner = CommandOwnership(api, {"a", "b", "late"}, _controller(), cav_speed_mode=27)
    owner.step(_frames("a", "b"), now_s=0.0, live_ids={"a", "b", "hdv"})
    owner.step(_frames("a", "b", "late"), now_s=1.0, live_ids={"a", "b", "late", "hdv"})
    mode_calls = [call for call in api.calls if call[0] == "mode"]
    assert mode_calls == [("mode", "a", 27), ("mode", "b", 27), ("mode", "late", 27)]
    assert "hdv" not in {call[1] for call in mode_calls}


def test_external_reset_of_configured_cav_is_rejected_before_new_command():
    api = VehicleApi({"cav": 31})
    owner = CommandOwnership(api, {"cav"}, _controller(), cav_speed_mode=27)
    owner.step(_frames("cav"), now_s=0.0, live_ids={"cav"})
    api.calls.clear()
    api.modes["cav"] = 31
    with pytest.raises(ValueError, match="changed outside"):
        owner.step(_frames("cav"), now_s=1.0, live_ids={"cav"})
    assert api.calls == []


def test_native_release_then_reengagement_keeps_same_candidate_mode():
    api = VehicleApi({"cav": 31})
    owner = CommandOwnership(api, {"cav"}, _controller(), cav_speed_mode=27)
    owner.step(_frames("cav"), now_s=0.0, live_ids={"cav"})
    owner.step(_frames("cav"), now_s=1.0, live_ids={"cav"}, native_ids={"cav"})
    owner.step(_frames("cav"), now_s=2.0, live_ids={"cav"})
    assert ("setSpeed", "cav", -1) in api.calls
    assert api.modes["cav"] == 27
    assert [call[0] for call in api.calls].count("slowDown") == 2


def test_wall_and_damping_share_the_same_mode_and_writer(monkeypatch):
    _patch_wall(monkeypatch)
    api = VehicleApi({"wall": 31, "damping": 31})
    owner = CommandOwnership(api, {"wall", "damping"}, _controller(), cav_speed_mode=27)
    rows = owner.step(_frames("wall", "damping"), now_s=0.0, live_ids={"wall", "damping"}, wall_commands={"wall": 0.4})
    assert {row.owner for row in rows} == {"wall", "damping"}
    assert {api.modes[vehicle_id] for vehicle_id in ("wall", "damping")} == {27}
    assert {call[0] for call in api.calls} == {"mode", "slowDown"}


def test_invalid_hdv_masks_and_modes_fail_without_any_write():
    api = VehicleApi({"cav": 31, "hdv": 31})
    owner = CommandOwnership(api, {"cav"}, _controller(), cav_speed_mode=27)
    with pytest.raises(ValueError, match="HDV"):
        owner.step(_frames("cav"), now_s=0.0, live_ids={"cav", "hdv"}, wall_commands={"hdv": 0.1})
    assert api.calls == []
    with pytest.raises(ValueError, match="native mask"):
        owner.step(_frames("cav"), now_s=0.0, live_ids={"cav"}, native_ids={"hdv"})
    assert api.calls == []
    with pytest.raises(ValueError, match="only legacy"):
        CommandOwnership(VehicleApi({"cav": 31}), {"cav"}, _controller(), cav_speed_mode=29)


def test_paired_effect_rejects_mode_mismatched_baselines():
    control = {"cav_speed_mode": 31}
    treatment = {"cav_speed_mode": 27}
    with pytest.raises(ValueError, match="braking interfaces differ"):
        paired_effect(control, {}, treatment, {})
