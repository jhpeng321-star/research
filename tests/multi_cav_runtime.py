"""Shared synthetic multi-CAV runtime fixture (relocated from the retired
evaluation tests; generic fake-TraCI wiring, no controller-specific logic)."""
import sys
from types import SimpleNamespace

import pytest

from traffic_shepherd import workflow
from traffic_shepherd.scenarios import TripRequest
from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame


@pytest.fixture
def multi_cav_runtime(tmp_path, monkeypatch):
    state = {"time": 0., "writes": [], "closed": False, "callbacks": []}
    vehicle = SimpleNamespace(
        getIDList=lambda: ["a", "b"] if state["time"] >= 1. else [],
        getSpeedMode=lambda _: 31, getSpeed=lambda _: 10., getAcceleration=lambda _: 0.,
        getLeader=lambda *_: None, slowDown=lambda *args: state["writes"].append(("slowDown", *args)),
        setSpeed=lambda *args: state["writes"].append(("setSpeed", *args)),
    )
    api = SimpleNamespace(vehicle=vehicle, simulationStep=lambda: state.update(time=state["time"] + .5),
        isLoaded=lambda: False, simulation=SimpleNamespace(getTime=lambda: state["time"],
            getDepartedIDList=lambda: ["a", "b"] if state["time"] == 1. else [],
            getArrivedIDList=lambda: [], getCollidingVehiclesIDList=lambda: [],
            getStartingTeleportNumber=lambda: 0), getVersion=lambda: [1, "synthetic"])
    monkeypatch.setitem(sys.modules, "traci", api)
    binary = tmp_path / "sumo-placeholder"
    binary.write_bytes(b"synthetic")
    monkeypatch.setattr(workflow.shutil, "which", lambda _: str(binary))
    monkeypatch.setattr(workflow, "start_sumo", lambda *args, **kwargs: kwargs["command_record"].append(["sumo"]))
    monkeypatch.setattr(workflow, "close_sumo", lambda *_: state.update(closed=True))
    monkeypatch.setattr(workflow, "observation_map", lambda *_: {})
    monkeypatch.setattr(workflow, "SumoLocalSensor", lambda *args: SimpleNamespace(capture=lambda cav, *, now_s:
        SensorFrame(cav, now_s, 10., 0., 30., 0, front=NeighborMeasurement(5., 0.),
                    front_coverage_m=100., rear_coverage_m=100.)))
    requests = [TripRequest("a", 0., "r", "normal_car", True), TripRequest("b", 0., "r", "normal_car", True)]
    def build(root, *args, **kwargs):
        root.mkdir(parents=True)
        scene = root / "scene.sumocfg"
        scene.write_text("synthetic scene")
        (root / "requests.rou.xml").write_text("synthetic requests")
        return scene, requests, {"schema": "test"}
    monkeypatch.setattr(workflow, "build_scenario", build)
    return state
