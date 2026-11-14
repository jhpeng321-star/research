from dataclasses import dataclass
import pytest

import output.controller_verification_v8.pi_mechanics as mechanics
from output.controller_verification_v7.pi_owner_adapter import PIOwnerAdapter
from output.controller_verification_v8.pi_mechanics import (
    CONTROL_TICKS,
    PI_TICKS,
    analyze,
    prepare_fixture,
    run_fixture,
)
from traffic_shepherd.config import load_config
from traffic_shepherd.ownership import CommandOwnership
from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame


@dataclass
class Frame:
    vehicle_id: str
    timestamp_s: float


class Record:
    def __init__(self, tick, owner):
        self.timestamp_s = tick
        self.vehicle_id = "cav"
        self.owner = owner
        self.released_to_native = owner == "native"
        self.fallback_reason = "selected_native" if owner == "native" and tick == 3.0 else (
            "missing_observation" if owner == "native" else None)

    def as_dict(self):
        return {"vehicle_id": self.vehicle_id, "timestamp_s": self.timestamp_s,
                "owner": self.owner, "fallback_reason": self.fallback_reason,
                "released_to_native": self.released_to_native}


class Adapter:
    def __init__(self):
        self.evidence = []

    def step(self, owner, frame_map, *, now_s, live_ids, native_ids):
        if now_s in PI_TICKS and frame_map["cav"] is not None:
            self.evidence.append({"vehicle_id": "cav", "timestamp_s": now_s,
                                  "raw_proposal_mps": 20.0,
                                  "issued_target_speed_mps": 16.5,
                                  "frame": {"timestamp_s": now_s},
                                  "owner_record": {"timestamp_s": now_s}})
        return [Record(now_s, "native" if native_ids or frame_map["cav"] is None else "damping")]


def _fixture_inputs():
    frames = {tick: Frame("cav", tick) for tick in CONTROL_TICKS}
    physical = [{"time_s": i / 2, "collisions": 0, "teleports": 0,
                 "speed_mps": 15.0, "acceleration_mps2": 0.0}
                for i in range(1, 17)]
    return frames, physical


def test_prepare_is_explicit_and_exclusive(tmp_path):
    path = tmp_path / "run"
    manifest = prepare_fixture(path, source_identities={"adapter": "sha"})
    assert manifest["budget_seconds"] == 8.0
    assert manifest["pi_ticks"] == list(PI_TICKS)
    with pytest.raises(FileExistsError):
        prepare_fixture(path)


def test_run_persists_frames_evidence_and_progress_before_rows(tmp_path):
    path = tmp_path / "run"
    prepare_fixture(path)
    frames, physical = _fixture_inputs()
    progress = []
    result = run_fixture(path, owner=object(), adapter=Adapter(), frames=frames,
                         physical=physical, progress_callback=progress.append)
    assert progress == [row["time_s"] for row in physical]
    assert len(result["frames"]) == 7
    assert result["frames"][4]["provided_to_controller"] is False
    assert len(result["adapter_evidence"]) == 5
    assert (path / "adapter_evidence.json").exists()
    assert (path / "owner_receipts.jsonl").read_text()


def test_analyze_requires_lifecycle_and_physics_evidence(tmp_path):
    path = tmp_path / "run"
    prepare_fixture(path)
    frames, physical = _fixture_inputs()
    result = run_fixture(path, owner=object(), adapter=Adapter(), frames=frames,
                         physical=physical)
    report = analyze(result)
    assert report["passed"] is True
    result["physical"] = result["physical"][:-1]
    assert analyze(result)["passed"] is False


def test_missing_halfstep_is_refused(tmp_path):
    path = tmp_path / "run"
    prepare_fixture(path)
    frames, physical = _fixture_inputs()
    with pytest.raises(ValueError, match="physical observations"):
        run_fixture(path, owner=object(), adapter=Adapter(), frames=frames,
                    physical=physical[:-1])


class _Vehicle:
    def __init__(self):
        self.commands = []

    def getSpeedMode(self, vehicle):
        assert vehicle == "cav"
        return 31

    def slowDown(self, vehicle, target, duration):
        self.commands.append(("slowDown", vehicle, target, duration))

    def setSpeed(self, vehicle, target):
        self.commands.append(("setSpeed", vehicle, target))


def test_real_adapter_and_command_ownership_are_used(tmp_path):
    path = tmp_path / "run"
    prepare_fixture(path)
    config = load_config()
    adapter = PIOwnerAdapter(config=config, cav_ids={"cav"})
    vehicle = _Vehicle()
    owner = CommandOwnership(type("API", (), {"vehicle": vehicle})(), {"cav"}, adapter)
    frames = {
        tick: SensorFrame("cav", tick, 15.0, 0.0, 30.0, 0,
                          front=NeighborMeasurement(45.0, 0.0),
                          front_coverage_m=100.0, rear_coverage_m=100.0)
        for tick in CONTROL_TICKS
    }
    frames[5.0] = None
    _, physical = _fixture_inputs()
    result = run_fixture(path, owner=owner, adapter=adapter, frames=frames,
                         physical=physical)
    assert len(result["adapter_evidence"]) == 5
    assert len(result["owner_receipts"]) == 7
    assert any(command[0] == "setSpeed" for command in vehicle.commands)
    assert all("owner_record" in evidence for evidence in result["adapter_evidence"])


def test_registered_runner_reserves_eight_seconds_before_live_boundary(tmp_path, monkeypatch):
    ledger = tmp_path / "budget.json"
    registration = dict(mechanics.EXPECTED, ledger_path=str(ledger), automatic_retries=0,
                        budget={"ceilings": {"categories": {
                            "mechanics": {"max_allocations": 1, "max_seconds": 8.0}},
                            "total_seconds": 8.0}})
    called = []

    def fake_live(reg, output, binary, guard):
        called.append((output, binary))
        return {"status": "complete", "actual_progress_s": 8.0}

    monkeypatch.setattr(mechanics, "_live_run", fake_live)
    import output.controller_verification_v8.launch_contract as launch_contract
    monkeypatch.setattr(launch_contract, "validate", lambda reg, output, binary: None)
    with pytest.raises((RuntimeError, KeyError)):
        mechanics.run_registered(registration, tmp_path / "run", "sumo")
    assert called == [((tmp_path / "run").resolve(), "sumo")]
