"""Sole ownership, explicit native handback, timing and measurement semantics."""
from types import SimpleNamespace
import json
import pytest

from traffic_shepherd.config import load_config
from traffic_shepherd.damping import DampingDecision
from traffic_shepherd.ownership import CommandOwnership


class VehicleAPI:
    def __init__(self):
        self.calls = []
        self.mode = 31
    def getSpeedMode(self, vehicle):
        return self.mode
    def slowDown(self, *args):
        self.calls.append(("slowDown", *args))
    def setSpeed(self, *args):
        self.calls.append(("setSpeed", *args))


class FixedController:
    config = load_config()
    def decide(self, frame, *, now_s):
        if frame is None:
            return DampingDecision(fallback_reason="missing_observation")
        return DampingDecision(-2., -2., frame.speed_mps-2., actor_input_si=(15.,15.,30.))


def frame(now=0.):
    return SimpleNamespace(vehicle_id="cav", timestamp_s=now, speed_mps=15., allowed_speed_mps=30.,
        front_coverage_m=100., rear_coverage_m=100., front=None, rear=None)


def setup():
    vehicle = VehicleAPI()
    controller = FixedController()
    owner = CommandOwnership(SimpleNamespace(vehicle=vehicle), {"cav"}, controller)
    return owner, vehicle, controller


def test_damping_wall_damping_is_same_controller_and_sole_command():
    owner, api, controller = setup()
    a = owner.step({"cav":frame()}, now_s=0., live_ids={"cav","hdv"})[0]
    b = owner.step({"cav":frame(1.)}, now_s=1., live_ids={"cav","hdv"}, wall_commands={"cav":-.5})[0]
    c = owner.step({"cav":frame(2.)}, now_s=2., live_ids={"cav","hdv"})[0]
    assert [a.owner,b.owner,c.owner] == ["damping","wall","damping"]
    assert a.network_output_mps2 == c.network_output_mps2 == -2.
    assert b.network_output_mps2 is None and b.wall_request_mps2 == -.5
    assert owner.controller is controller
    assert api.calls == [("slowDown","cav",13.,1.),("slowDown","cav",14.5,1.),("slowDown","cav",13.,1.)]


def test_missing_frame_releases_owned_command_and_reengages():
    owner, api, _ = setup()
    owner.step({"cav":frame()}, now_s=0., live_ids={"cav"})
    row = owner.step({}, now_s=1., live_ids={"cav"})[0]
    assert row.owner == "native" and row.released_to_native
    assert row.issued_acceleration_mps2 is None
    owner.step({"cav":frame(2.)}, now_s=2., live_ids={"cav"})
    assert api.calls[1] == ("setSpeed","cav",-1)
    assert api.calls[2] == ("slowDown","cav",13.,1.)


@pytest.mark.parametrize("wall", [{"hdv":-.5},{"absent":-.5},{"cav":float('nan')}])
def test_wall_masks_validated_before_any_commands(wall):
    owner, api, _ = setup()
    with pytest.raises(ValueError):
        owner.step({"cav":frame()}, now_s=0., live_ids={"cav","hdv"}, wall_commands=wall)
    assert api.calls == []


def test_wrong_sensor_identity_and_safety_mode_rejected():
    owner, api, _ = setup()
    wrong = frame()
    wrong.vehicle_id = "hdv"
    with pytest.raises(ValueError, match="sensor mask"):
        owner.step({"cav":wrong}, now_s=0., live_ids={"cav","hdv"})
    api.mode = 0
    with pytest.raises(ValueError, match="speedMode31"):
        owner.step({"cav":frame()}, now_s=0., live_ids={"cav"})
    assert api.calls == []


def test_response_is_separate_measured_value_and_lifecycle_drops_absent():
    owner, api, _ = setup()
    command = owner.step({"cav":frame()}, now_s=0., live_ids={"cav"})[0]
    response = owner.record_response("cav", timestamp_s=.5, speed_mps=14.8, acceleration_mps2=-.4)
    assert command.issued_acceleration_mps2 == -2.
    assert response.measured_acceleration_mps2 == -.4
    with pytest.raises(ValueError):
        owner.record_response("cav",timestamp_s=0.,speed_mps=15.,acceleration_mps2=0.)
    owner.step({}, now_s=1., live_ids={"hdv"})
    owner.release_all(live_ids={"hdv"})
    assert len(api.calls) == 1


def test_duplicate_control_tick_forbidden():
    owner, api, _ = setup()
    owner.step({"cav":frame()}, now_s=0., live_ids={"cav"})
    with pytest.raises(ValueError, match="clock"):
        owner.step({"cav":frame()}, now_s=0., live_ids={"cav"})
    assert len(api.calls) == 1


@pytest.mark.parametrize("observation", [None, frame(0.)])
def test_wall_missing_or_stale_observation_releases(observation):
    owner, api, _ = setup()
    owner.step({"cav":frame()}, now_s=0., live_ids={"cav"}, wall_commands={"cav":-.5})
    row = owner.step({"cav":observation}, now_s=1., live_ids={"cav"}, wall_commands={"cav":-.5})[0]
    assert row.owner == "native" and row.released_to_native
    assert api.calls[-1] == ("setSpeed","cav",-1)


def test_invalid_model_output_releases_and_serializes_without_nan():
    owner, api, controller = setup()
    owner.step({"cav":frame()}, now_s=0., live_ids={"cav"})
    decision = DampingDecision(network_output_mps2=float("nan"), fallback_reason="invalid_model_output")
    controller.decide = lambda *a, **kw: decision
    row = owner.step({"cav":frame(1.)}, now_s=1., live_ids={"cav"})[0]
    assert row.owner == "native" and row.released_to_native
    for obj in (decision, row):
        output = obj.as_dict()
        assert output["network_output_mps2"] is None
        assert output["network_output_mps2_nonfinite"] == "nan"
        assert output["fallback_reason"] == "invalid_model_output"
        json.dumps(output, allow_nan=False)
    response = owner.record_response("cav",timestamp_s=1.5,speed_mps=14.,acceleration_mps2=-.2)
    assert response.as_dict()["measured_acceleration_mps2"] == -.2
    json.dumps(response.as_dict(), allow_nan=False)


def test_malformed_wall_value_is_rejected_before_commands():
    owner, api, _ = setup()
    with pytest.raises(ValueError, match="numeric"):
        owner.step({"cav":frame()}, now_s=0., live_ids={"cav"}, wall_commands={"cav":object()})
    assert api.calls == []


def test_half_step_command_is_rejected():
    owner, api, _ = setup()
    with pytest.raises(ValueError, match="grid"):
        owner.step({"cav":frame(.5)}, now_s=.5, live_ids={"cav"})
    assert api.calls == []


def test_explicit_native_bypasses_onnx_releases_and_reengages():
    owner, api, controller = setup()
    first = owner.step({"cav":frame()}, now_s=0., live_ids={"cav"})[0]
    response = owner.record_response("cav",timestamp_s=.5,speed_mps=14.8,acceleration_mps2=-.4)
    assert first.issued_acceleration_mps2 == -2. and response.measured_acceleration_mps2 == -.4
    original = controller.decide
    def forbidden(*a, **kw):
        raise AssertionError("native arm must not call ONNX")
    controller.decide = forbidden
    row = owner.step({},now_s=1.,live_ids={"cav"},native_ids={"cav"})[0]
    assert row.owner == "native" and row.fallback_reason == "selected_native"
    assert row.released_to_native and row.network_output_mps2 is None
    controller.decide = original
    resumed = owner.step({"cav":frame(2.)},now_s=2.,live_ids={"cav"})[0]
    assert resumed.issued_acceleration_mps2 == first.issued_acceleration_mps2
    assert api.calls == [("slowDown","cav",13.,1.),("setSpeed","cav",-1),("slowDown","cav",13.,1.)]


@pytest.mark.parametrize("native,wall", [({"hdv"},{}),({"absent"},{}),({"cav"},{"cav":-.5})])
def test_explicit_native_mask_validated_atomically(native,wall):
    owner, api, _ = setup()
    with pytest.raises(ValueError, match="native mask"):
        owner.step({"cav":frame()},now_s=0.,live_ids={"cav","hdv"},native_ids=native,wall_commands=wall)
    assert api.calls == []
