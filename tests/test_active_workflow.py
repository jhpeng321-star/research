"""CLI/workflow checks use a synthetic TraCI API, never launch SUMO."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from traffic_shepherd import workflow
from traffic_shepherd import controller_factory
from traffic_shepherd.__main__ import main
from traffic_shepherd.damping import DampingDecision
from traffic_shepherd.scenarios import TripRequest
from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    state = dict(time=0., writes=[], closed=False, start_calls=0, actor_calls=0)
    def live():
        return (["a"] if state["time"] >= .5 else []) + (["b"] if state["time"] >= 1.5 else [])
    def step():
        state["time"] += .5
    vehicle = SimpleNamespace(
        getIDList=live, getSpeedMode=lambda _: 31, getSpeed=lambda _: 20.,
        getAcceleration=lambda v: -.25 if v == "a" else 0., getLeader=lambda *_: None,
        slowDown=lambda *args: state["writes"].append(("slowDown", *args)),
        setSpeed=lambda *args: state["writes"].append(("setSpeed", *args)),
    )
    api = SimpleNamespace(vehicle=vehicle, simulationStep=step, isLoaded=lambda: False,
        simulation=SimpleNamespace(getTime=lambda: state["time"],
            getDepartedIDList=lambda: ["a"] if state["time"] == .5 else ["b"] if state["time"] == 1.5 else [],
            getArrivedIDList=lambda: [], getCollidingVehiclesIDList=lambda: [],
            getStartingTeleportNumber=lambda: 0), getVersion=lambda: [1,"synthetic"])
    monkeypatch.setitem(sys.modules, "traci", api)
    binary = tmp_path / "sumo-placeholder"
    binary.write_bytes(b"no executable")
    monkeypatch.setattr(workflow.shutil, "which", lambda _: str(binary))
    def start(*args, **kwargs):
        state["start_calls"] += 1
    monkeypatch.setattr(workflow, "start_sumo", start)
    monkeypatch.setattr(workflow, "close_sumo", lambda *_: state.update(closed=True))
    monkeypatch.setattr(workflow, "observation_map", lambda *_: {})
    class Controller:
        def __init__(self, config):
            self.config = config
        def decide(self, frame, *, now_s):
            state["actor_calls"] += 1
            return DampingDecision(-.7, -.5, 19.5, None, (20.,20.,20.))
    monkeypatch.setattr(controller_factory, "CirclesController", Controller)
    class Sensor:
        def __init__(self, *args):
            pass
        def capture(self, cav, *, now_s):
            return SensorFrame(cav, now_s, 20, 0, 30, 0,
                front=NeighborMeasurement(20,0), front_coverage_m=100,rear_coverage_m=100)
    monkeypatch.setattr(workflow, "SumoLocalSensor", Sensor)
    requests = [TripRequest("a",0,"r","normal_car",True),
                TripRequest("b",1.25,"r","normal_car",False),
                TripRequest("future",10,"r","normal_car",False)]
    def build(root, *args, **kwargs):
        root.mkdir()
        scene = root / "scenario.sumocfg"
        scene.write_text("synthetic scenario")
        (root / "requests.rou.xml").write_text("synthetic routes")
        return scene, requests, dict(schema="synthetic fixture")
    monkeypatch.setattr(workflow, "build_scenario", build)
    return state


@pytest.mark.parametrize("controller", ["native", "damping"])
def test_fixed_grid_accounting_and_separate_telemetry(tmp_path, fake_runtime, controller):
    output = tmp_path / "evaluation"
    result = workflow.evaluate(seed=4107,loading=2000,penetration=.2,
                               controller_name=controller,output=output,duration=2)
    assert result["mechanics_only"] and not result["accepted_checkpoint"]
    assert result["population"] == 3 and result["requested_due_at_horizon"] == 2
    assert result["system_seconds_per_requested_vehicle"] == pytest.approx(2.75/3)
    assert result["source_wait_vehicle_seconds"] == .75
    assert result["totals"]["acceleration_energy"] == .125
    assert fake_runtime["closed"] and fake_runtime["start_calls"] == 1
    rows = [json.loads(s) for s in (output / "telemetry.jsonl").read_text().splitlines()]
    commands = [r for r in rows if r["kind"] == "command"]
    responses = [r for r in rows if r["kind"] == "response"]
    assert [r["timestamp_s"] for r in commands] == [1.]
    assert [r["timestamp_s"] for r in responses] == [1.5,2.]
    assert all(r["vehicle_id"] == "a" for r in commands + responses)
    assert all(r["measured_acceleration_mps2"] == -.25 for r in responses)
    if controller == "damping":
        assert commands[0]["network_output_mps2"] == -.7
        assert commands[0]["issued_acceleration_mps2"] == -.5
        assert fake_runtime["writes"] == [("slowDown","a",19.5,1.),("setSpeed","a",-1)]
    else:
        assert fake_runtime["actor_calls"] == 0 and fake_runtime["writes"] == []
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["allocated_simulated_seconds"] == 2
    assert manifest["telemetry_sha256"] == workflow.sha(output / "telemetry.jsonl")


def test_existing_scientific_output_refused_without_launch(tmp_path, fake_runtime):
    output = tmp_path / "existing"
    output.mkdir()
    evidence = output / "summary.json"
    evidence.write_text("preserve evidence")
    with pytest.raises(FileExistsError):
        workflow.evaluate(seed=4107,loading=2000,penetration=.2,controller_name="damping",output=output,duration=2)
    assert evidence.read_text() == "preserve evidence" and fake_runtime["start_calls"] == 0


@pytest.mark.parametrize("seed,role", [(912001,"development"),(5001,"exposed"),(910001,"training")])
def test_sealed_and_training_cases_refused_before_output_or_launch(tmp_path, fake_runtime, seed, role):
    output = tmp_path / "forbidden"
    with pytest.raises(ValueError):
        workflow.evaluate(seed=seed,loading=2000,penetration=.2,controller_name="damping",output=output,role=role,duration=2)
    assert not output.exists() and fake_runtime["start_calls"] == 0


@pytest.mark.parametrize("duration", [0,2401,1.5])
def test_invalid_duration_cannot_allocate_a_run(tmp_path,fake_runtime,duration):
    with pytest.raises(ValueError):
        workflow.evaluate(seed=4107,loading=2000,penetration=.2,controller_name="damping",output=tmp_path/"invalid",duration=duration)
    assert fake_runtime["start_calls"] == 0


def test_failure_is_preserved_and_never_marked_complete(tmp_path,fake_runtime,monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate synthetic failure")
    monkeypatch.setattr(workflow,"ttc_exposure",fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError,match="deliberate"):
        workflow.evaluate(seed=4107,loading=2000,penetration=.2,controller_name="damping",output=output,duration=2)
    assert json.loads((output/"manifest.json").read_text())["status"] == "failed"
    assert not (output/"summary.json").exists() and fake_runtime["closed"]


def test_cli_inspect_is_read_only_and_reports_expected_identity(capsys):
    main(["inspect"])
    result = json.loads(capsys.readouterr().out)
    assert result["model_sha256"] == result["expected_model_sha256"]
    assert result["scientifically_accepted"] is False
    assert Path(result["model_path"]).name == "model.onnx"


def test_cli_has_no_training_or_acceptance_entry_point():
    with pytest.raises(SystemExit):
        main(["train"])


def test_active_entry_point_dependency_graph_has_no_archived_package_imports():
    package = Path(workflow.__file__).parent
    forbidden = {"archive", "env", "idm", "config", "experiments", "policies", "tools", "ray"}
    for source in package.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            assert not {name.split(".")[0] for name in names} & forbidden, source.name


def test_wrong_physics_grid_is_a_failed_run(tmp_path, fake_runtime, monkeypatch):
    api = sys.modules["traci"]
    def wrong_step():
        fake_runtime["time"] += .25
    monkeypatch.setattr(api, "simulationStep", wrong_step)
    output = tmp_path / "bad-clock"
    with pytest.raises(RuntimeError, match="physics time grid"):
        workflow.evaluate(seed=4107, loading=2000, penetration=.2,
                          controller_name="damping", output=output, duration=2)
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"


def test_preexisting_connection_is_never_closed_or_modified(tmp_path, fake_runtime, monkeypatch):
    monkeypatch.setattr(sys.modules["traci"], "isLoaded", lambda: True)
    output = tmp_path / "must-not-start"
    with pytest.raises(RuntimeError, match="pre-existing"):
        workflow.evaluate(seed=4107, loading=2000, penetration=.2,
                          controller_name="damping", output=output, duration=2)
    assert not output.exists()
    assert not fake_runtime["closed"] and fake_runtime["writes"] == []
    assert fake_runtime["start_calls"] == 0
