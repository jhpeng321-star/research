from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from env.native_damping import DampingContract
from policies.sensor_local_damping import SensorLocalDampingController


def connection():
    writes = []
    api = SimpleNamespace(
        vehicle=SimpleNamespace(
            getLaneID=lambda _: "lane0",
            getLanePosition=lambda _: 200,
            getLength=lambda _: 5,
            getSpeed=lambda _: 30,
            getAcceleration=lambda _: 0,
            getAllowedSpeed=lambda _: 30,
            getLaneIndex=lambda _: 0,
            getAccel=lambda _: 2.6,
            getDecel=lambda _: 4.5,
            getTau=lambda _: 1.5,
            getMinGap=lambda _: 2.5,
            getEmergencyDecel=lambda _: 9,
            getSpeedMode=lambda _: 31,
            slowDown=lambda *args: writes.append(("slowDown", *args)),
            setSpeed=lambda *args: writes.append(("setSpeed", *args)),
        ),
        lane=SimpleNamespace(getLength=lambda _: 1000, getLastStepVehicleIDs=lambda _: ["cav"]),
    )
    return api, writes


def test_complete_zero_actor_path_does_not_write_sumo_commands():
    api, writes = connection()
    controller = SensorLocalDampingController(api)
    first = controller.step({"cav"}, now_s=0)
    second = controller.step({"cav"}, now_s=1)
    assert abs(first["actions"]["cav"]) <= controller.contract.zero_tolerance
    assert second["previous_execution"][0]["observed_average_acceleration"] == 0
    assert writes == []
    with pytest.raises(ValueError, match="once"):
        controller.step({"cav"}, now_s=1)


def test_actor_weights_actually_drive_commands_and_restore_under_identical_contract(tmp_path):
    api, writes = connection()
    controller = SensorLocalDampingController(api)
    with torch.no_grad():
        # Native head mean is shifted positive; raw zero requests -0.1 here.
        controller.actor.mean_head.bias.zero_()
    path = tmp_path / "weights.pt"
    controller.save_weights(path)
    result = controller.step({"cav"}, now_s=0)
    assert result["actions"]["cav"] == pytest.approx(-0.1)
    assert writes[0][0] == "slowDown"
    other_api, _ = connection()
    restored = SensorLocalDampingController.from_weights(other_api, path)
    other = restored.step({"cav"}, now_s=0)
    np.testing.assert_array_equal(other["observations"]["cav"], result["observations"]["cav"])
    assert other["actions"] == result["actions"]
    with pytest.raises(FileExistsError):
        controller.save_weights(path)
    with pytest.raises(ValueError, match="contract"):
        SensorLocalDampingController.from_weights(
            other_api, path, contract=replace(DampingContract(), history_frames=3)
        )


def test_cav_departure_removes_history_and_does_not_create_ghost_actor():
    api, _ = connection()
    controller = SensorLocalDampingController(api)
    controller.step({"cav"}, now_s=0)
    result = controller.step(set(), now_s=1)
    assert result["observations"] == result["actions"] == {}
    assert "cav" not in controller.history.frames
    assert result["previous_execution"][0]["reason"] == "vehicle_departed"


def test_checkpoint_preserves_map_inputs_and_rejects_silent_map_change(tmp_path):
    api, _ = connection()
    controller = SensorLocalDampingController(api, lane_end_distance_to_merge={"lane0": 0.0})
    path = tmp_path / "mapped.pt"
    controller.save_weights(path)
    restored = SensorLocalDampingController.from_weights(api, path)
    assert restored.sensor.lane_end_distance_to_merge == {"lane0": 0.0}
    with pytest.raises(ValueError, match="map mismatch"):
        SensorLocalDampingController.from_weights(
            api, path, lane_end_distance_to_merge={"lane0": 50}
        )
