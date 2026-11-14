"""Root integration evidence: real workflow/owner, mocked physical system only."""

import json
import sys
from types import SimpleNamespace

import pytest
from traffic_shepherd import workflow
from traffic_shepherd.scenarios import TripRequest
from traffic_shepherd.sensing import SensorFrame, NeighborMeasurement


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    state = dict(time=0.0, writes=[], closed=False)

    def live():
        t = state["time"]
        return ["hdv"] + (["a", "b"] if t >= 1 else []) + (["c"] if t >= 3 and t != 4.5 else [])

    def write(kind, *args):
        state["writes"].append((state["time"], kind, *args))

    api = SimpleNamespace(
        isLoaded=lambda: False,
        getVersion=lambda: [22, "mock only"],
        simulationStep=lambda: state.update(time=state["time"] + 0.5),
        vehicle=SimpleNamespace(
            getIDList=live,
            getSpeedMode=lambda _: 31,
            getSpeed=lambda v: dict(a=10.0, b=12.0, c=14.0, hdv=15.0)[v],
            getAcceleration=lambda _: 0.0,
            getLeader=lambda *_: None,
            slowDown=lambda *a: write("slowDown", *a),
            setSpeed=lambda *a: write("setSpeed", *a),
        ),
        simulation=SimpleNamespace(
            getTime=lambda: state["time"],
            getDepartedIDList=lambda: (
                ["a", "b"] if state["time"] == 1 else ["c"] if state["time"] == 3 else []
            ),
            getArrivedIDList=lambda: [],
            getCollidingVehiclesIDList=lambda: [],
            getStartingTeleportNumber=lambda: 0,
        ),
    )
    monkeypatch.setitem(sys.modules, "traci", api)
    binary = tmp_path / "fake-sumo"
    binary.write_bytes(b"not physical evidence")
    monkeypatch.setattr(workflow.shutil, "which", lambda _: str(binary))
    monkeypatch.setattr(
        workflow, "start_sumo", lambda *a, **k: k["command_record"].append(["mock"])
    )
    monkeypatch.setattr(workflow, "close_sumo", lambda _: state.update(closed=True))
    monkeypatch.setattr(workflow, "observation_map", lambda *a: {})

    def capture(v, *, now_s):
        if v == "b" and now_s == 4:
            raise ValueError("missing local frame")
        return SensorFrame(
            v,
            now_s,
            api.vehicle.getSpeed(v),
            0.0,
            30.0,
            0,
            front=NeighborMeasurement(50.0, 0.0),
            front_coverage_m=100.0,
            rear_coverage_m=100.0,
        )

    monkeypatch.setattr(workflow, "SumoLocalSensor", lambda *a: SimpleNamespace(capture=capture))

    def build(folder, *a, **k):
        folder.mkdir(parents=True)
        scene = folder / "scene.sumocfg"
        scene.write_text("mock")
        (folder / "requests.rou.xml").write_text("mock")
        req = [
            TripRequest(v, 0.5 if v != "c" else 2.5, "r", "normal_car", v != "hdv")
            for v in ["a", "b", "c", "hdv"]
        ]
        return scene, req, dict(schema="mock")

    monkeypatch.setattr(workflow, "build_scenario", build)
    return state, api


def run(output, suffix, prefix="damping"):
    from traffic_shepherd.fleet_switch import SingleSwitchSpec

    return workflow.evaluate(
        seed=4108,
        loading=2000,
        penetration=0.2,
        controller_name=prefix,
        fleet_switch=SingleSwitchSpec(prefix, suffix, 3.0),
        output=output,
        duration=6,
        desired_speed_mps=31.29 if "pi-external-u" in (prefix, suffix) else None,
    )


def test_shared_prefix_real_release_and_pi_entry(tmp_path, runtime):
    state, api = runtime
    traces = {}
    outputs = {}
    for arm in ["native", "damping", "pi-external-u"]:
        state.update(time=0.0, writes=[], closed=False)
        out = tmp_path / arm
        run(out, arm)
        assert state["closed"]
        outputs[arm] = rows(out / "telemetry.jsonl")
        traces[arm] = list(state["writes"])
        assert all(row[2] != "hdv" for row in traces[arm])
    for arm in ["native", "pi-external-u"]:
        assert [x for x in traces[arm] if x[0] < 3] == [x for x in traces["damping"] if x[0] < 3]
        kinds = {"command", "response", "cav_report", "observation_unavailable"}
        assert [
            x for x in outputs[arm] if x["kind"] in kinds and x.get("timestamp_s", 999) < 3
        ] == [x for x in outputs["damping"] if x["kind"] in kinds and x.get("timestamp_s", 999) < 3]
    assert [x for x in traces["native"] if x[0] == 3 and x[1] == "setSpeed"] == [
        (3.0, "setSpeed", "a", -1),
        (3.0, "setSpeed", "b", -1),
    ]
    ev = {
        (r["vehicle_id"], r["timestamp_s"]): r
        for r in outputs["pi-external-u"]
        if r["kind"] == "pi_source_evidence"
    }
    for v, ego in [("a", 10.0), ("b", 12.0), ("c", 14.0)]:
        assert ev[v, 3.0]["previous_raw_command_mps"] == ego
        assert ev[v, 3.0]["raw_proposal_mps"] == pytest.approx(0.5 * 32.29 + 0.5 * ego)
    assert ev["a", 4.0]["previous_raw_command_mps"] == ev["a", 3.0]["raw_proposal_mps"]
    assert ("b", 4.0) not in ev
    assert ev["b", 5.0]["previous_raw_command_mps"] == 12.0
    assert ev["c", 5.0]["previous_raw_command_mps"] == 14.0
    assert ev["c", 5.0]["generation"] > ev["c", 4.0]["generation"]


def test_boundary_partial_failure_preserves_real_acknowledgement(tmp_path, runtime):
    state, api = runtime
    old = api.vehicle.slowDown

    def fail(v, *a):
        if state["time"] == 3 and v == "b":
            raise RuntimeError("boundary partial failure")
        old(v, *a)

    api.vehicle.slowDown = fail
    out = tmp_path / "fail"
    with pytest.raises(RuntimeError, match="boundary partial failure"):
        run(out, "pi-external-u")
    assert state["closed"] and state["time"] == 3
    assert not (out / "summary.json").exists()
    assert json.loads((out / "manifest.json").read_text())["status"] == "failed"
    evidence = [r for r in rows(out / "telemetry.jsonl") if r["kind"] == "pi_source_evidence"]
    byid = {r["vehicle_id"]: r for r in evidence}
    assert byid["a"]["owner_record"] is not None
    assert byid["b"]["owner_record"] is None
    assert (out / "pi_evidence_rescue.json").exists()


def test_actual_telemetry_flush_failure_preserves_pending_evidence(tmp_path, runtime, monkeypatch):
    from pathlib import Path

    original = Path.open

    class BrokenTelemetry:
        def __init__(self, stream):
            self.stream = stream
            self.has_pi = False

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, text):
            self.has_pi |= "pi_source_evidence" in text
            return self.stream.write(text)

        def flush(self):
            if self.has_pi:
                raise OSError("telemetry flush failed")
            return self.stream.flush()

    def open_file(path, *args, **kwargs):
        stream = original(path, *args, **kwargs)
        if path.name == "telemetry.jsonl" and args and args[0] == "x":
            return BrokenTelemetry(stream)
        return stream

    monkeypatch.setattr(Path, "open", open_file)
    output = tmp_path / "sink_failure"
    with pytest.raises(OSError, match="telemetry flush failed"):
        run(output, "pi-external-u")
    rescue = json.loads((output / "pi_evidence_rescue.json").read_text())
    receipts = [
        x
        for x in rescue["rows"]
        if x["kind"] == "pi_owner_receipt" and x["status"] == "committed_raw_proposal"
    ]
    assert {x["vehicle_id"] for x in receipts} == {"a", "b", "c"}
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
    assert runtime[0]["closed"] and runtime[0]["time"] == 3


def test_session_construction_failure_precedes_launch(tmp_path, runtime, monkeypatch):
    starts = []
    monkeypatch.setattr(workflow, "start_sumo", lambda *a, **k: starts.append(1))

    def fail(*a, **k):
        raise ValueError("invalid session construction")

    monkeypatch.setattr(workflow, "FleetSwitchSession", fail)
    with pytest.raises(ValueError, match="invalid session construction"):
        run(tmp_path / "prelaunch", "pi-external-u")
    assert starts == []
    assert runtime[0]["writes"] == []


def test_suffix_pi_identity_bound_before_start(tmp_path, runtime, monkeypatch):
    output = tmp_path / "identity"
    captured = []

    def start(*a, **k):
        manifest = json.loads((output / "manifest.json").read_text())
        captured.append(manifest["fleet_controller_identities"])
        k["command_record"].append(["mock"])

    monkeypatch.setattr(workflow, "start_sumo", start)
    summary = run(output, "pi-external-u")
    assert len(captured) == 1
    assert captured[0]["pi"]["u_mps"] == 31.29
    assert captured[0]["pi"]["cav_ids"] == ["a", "b", "c"]
    assert captured[0]["pi"]["session_version"] == "pi_session/1"
    assert captured[0]["circles"]["model_sha256"] == workflow.load_config().model_sha256
    assert summary["fleet_controller_identities"] == captured[0]
