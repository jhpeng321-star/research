"""Actual supported PI evaluation into independent CV15 oracle; fake physics only."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import pytest
from types import SimpleNamespace

from traffic_shepherd import workflow
from traffic_shepherd.config import CONFIG_PATH, load_config
from traffic_shepherd.scenarios import TripRequest, scenario_identity
from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame
from output.controller_verification_v15 import independent_review as oracle


@pytest.mark.parametrize("suffix", ["damping", "native", "pi-external-u"])
def test_actual_multicav_workflow_passes_independent_fullhorizon_oracle(
    tmp_path, monkeypatch, suffix
):
    state = {"time": 0.0, "closed": False}
    requests = [TripRequest(v, 0.25, "r", "normal_car", True) for v in ("a", "b")]

    def live():
        return ["a", "b"] if 599 <= state["time"] < 604 else []

    api = SimpleNamespace(
        isLoaded=lambda: False,
        getVersion=lambda: [22, "SUMO 1.27.1 MOCK"],
        simulationStep=lambda: state.update(time=state["time"] + 0.5),
        vehicle=SimpleNamespace(
            getIDList=live,
            getSpeedMode=lambda _: 31,
            getSpeed=lambda _: 10.0,
            getAcceleration=lambda _: 0.0,
            getLeader=lambda *_: None,
            slowDown=lambda *a: None,
            setSpeed=lambda *a: None,
        ),
        simulation=SimpleNamespace(
            getTime=lambda: state["time"],
            getDepartedIDList=lambda: ["a", "b"] if state["time"] == 599 else [],
            getArrivedIDList=lambda: ["a", "b"] if state["time"] == 604 else [],
            getCollidingVehiclesIDList=lambda: [],
            getStartingTeleportNumber=lambda: 0,
        ),
    )
    monkeypatch.setitem(sys.modules, "traci", api)
    binary = tmp_path / "mock-sumo"
    binary.write_bytes(b"fake physics, not SUMO")
    monkeypatch.setattr(workflow.shutil, "which", lambda _: str(binary))
    monkeypatch.setattr(
        workflow, "start_sumo", lambda *a, **k: k["command_record"].append(["mock"])
    )
    monkeypatch.setattr(workflow, "close_sumo", lambda _: state.update(closed=True))
    monkeypatch.setattr(workflow, "observation_map", lambda *a: {})

    def capture(vehicle, *, now_s):
        if vehicle == "b" and now_s == 601:
            raise ValueError("mock missed report")
        return SensorFrame(
            vehicle,
            now_s,
            10.0,
            0.0,
            30.0,
            0,
            front=NeighborMeasurement(50.0, 0.0) if now_s != 602 else None,
            front_coverage_m=100.0,
            rear_coverage_m=100.0,
        )

    monkeypatch.setattr(workflow, "SumoLocalSensor", lambda *a: SimpleNamespace(capture=capture))
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    route = (
        "<routes>"
        + "".join(
            f'<vehicle id="{r.vehicle_id}" depart=".25" route="r" type="normal_car"/>'
            for r in requests
        )
        + "</routes>"
    )
    metadata = dict(
        requests_sha256=hashlib.sha256(
            json.dumps([asdict(r) for r in requests], sort_keys=True).encode()
        ).hexdigest(),
        routes_sha256=hashlib.sha256(route.encode()).hexdigest(),
    )

    def build(root, *args, **kwargs):
        root.mkdir(parents=True)
        scene = root / "scenario.sumocfg"
        scene.write_text("<configuration/>")
        (root / "requests.rou.xml").write_text(route)
        return scene, requests, metadata

    scene, _, _ = build(frozen / "scenario")
    reqpath = frozen / "requests.json"
    reqpath.write_text(json.dumps([asdict(r) for r in requests]))
    monkeypatch.setattr(workflow, "build_scenario", build)
    cfg = load_config()
    root = Path(__file__).resolve().parents[1]
    expected = dict(
        seed=4108,
        loading=2700,
        penetration=0.2,
        duration_s=2400,
        desired_speed_mps=31.29,
        config_sha256=oracle.sha(CONFIG_PATH),
        model_sha256=oracle.sha(cfg.model_path),
        sumo_binary_sha256=oracle.sha(binary),
        scenario_identity=scenario_identity(scene, requests, network_path=cfg.network_path),
        active_source_sha256={
            p.name: oracle.sha(p) for p in sorted((root / "traffic_shepherd").glob("*.py"))
        },
        pi_transitive_source_sha256={
            str(root / p): oracle.sha(root / p)
            for p in [
                "output/controller_verification_v2/pi_saturation_v1.py",
                "output/controller_verification_v4/pi_external_u.py",
                "output/controller_verification_v7/pi_owner_adapter.py",
                "traffic_shepherd/pi_session.py",
            ]
        },
        requests_path=str(reqpath),
        requests_sha256=oracle.sha(reqpath),
        scene_sha256={p.name: oracle.sha(p) for p in (scene, scene.parent / "requests.rou.xml")},
    )
    output = tmp_path / "actual"
    expected.update(
        model_path=str(cfg.model_path),
        suffix=suffix,
        fleet_switch=dict(prefix="damping", suffix=suffix, switch_time_s=600.0),
    )

    def progress(t):
        row = dict(
            time_s=t,
            live_ids=live(),
            departed_ids=api.simulation.getDepartedIDList(),
            arrived_ids=api.simulation.getArrivedIDList(),
            speed_modes={v: 31 for v in live()},
        )
        if t <= 600:
            row["vehicles"] = {
                v: dict(
                    road_id="road",
                    lane_id="lane",
                    lane_position_m=10.0,
                    position_xy=[10.0, 0.0],
                    speed_mps=10.0,
                    acceleration_mps2=0.0,
                    type_id="normal_car",
                    speed_mode=31,
                )
                for v in live()
            }
        with (output / "physics_steps.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    from traffic_shepherd.fleet_switch import SingleSwitchSpec

    def writer(method):
        def call(*args):
            with (output / "writer_trace.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(time_s=state["time"], method=method, args=list(args), succeeded=True)
                    )
                    + "\n"
                )

        return call

    api.vehicle.slowDown = writer("slowDown")
    api.vehicle.setSpeed = writer("setSpeed")
    workflow.evaluate(
        seed=4108,
        loading=2700,
        penetration=0.2,
        controller_name="damping",
        fleet_switch=SingleSwitchSpec("damping", suffix, 600.0),
        desired_speed_mps=31.29 if suffix == "pi-external-u" else None,
        output=output,
        duration=2400,
        progress_callback=progress,
    )
    result = oracle.verify_run(output, expected)
    assert result["integrity_pass"], result
    assert result["safety_pass"] and state["closed"]
    assert result["physical_steps"] == 4800
