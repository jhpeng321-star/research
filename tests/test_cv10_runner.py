from types import SimpleNamespace

import pytest

import output.controller_verification_v10.runner as runner
from output.controller_verification_v10.independent_review import verify


class Sim:
    def __init__(self):
        self.time = 0.0
    def simulationStep(self): self.time += 0.5
    def getTime(self): return self.time
    def getCollidingVehiclesNumber(self): return 0
    def getStartingTeleportNumber(self): return 0


class Vehicle:
    ids = ("cav", "leader_hdv", "rear_hdv")
    def __init__(self, sim):
        self.sim = sim
        self.writes = []
    def getIDList(self): return [] if self.sim.time == .5 else list(self.ids)
    def getTypeID(self, v): return {"cav":"idm", "leader_hdv":"leader_type", "rear_hdv":"rear_type"}[v]
    def getSpeedMode(self, v): return 31
    def getSpeed(self, v): return 1.0 if v == "cav" else .5 if v == "leader_hdv" else 1.0
    def getAcceleration(self, v): return 0.0
    def getAllowedSpeed(self, v): return 30.0 if v == "cav" else 15.0
    def getLaneIndex(self, v): return 0
    def getLaneID(self, v): return "road_0"
    def getLanePosition(self, v): return {"cav":100., "leader_hdv":109.06, "rear_hdv":50.}[v]
    def getLength(self, v): return 5.0
    def getMaxSpeed(self, v): return 30.0 if v == "cav" else .5 if v == "leader_hdv" else 1.0
    def slowDown(self, v, target, duration): self.writes.append(("slowDown", v, target, duration))
    def setSpeed(self, v, target): self.writes.append(("setSpeed", v, target))


class Lane:
    def __init__(self, vehicle): self.vehicle = vehicle
    def getLength(self, lane): return 2000.0
    def getLastStepVehicleIDs(self, lane): return self.vehicle.getIDList()


def test_causal_case_uses_real_adapter_owner_sensor_and_persists_evidence(tmp_path, monkeypatch):
    sim = Sim()
    vehicle = Vehicle(sim)
    traci = SimpleNamespace(simulation=sim, simulationStep=sim.simulationStep,
                            vehicle=vehicle, lane=Lane(vehicle))
    monkeypatch.setitem(__import__("sys").modules, "traci", traci)
    monkeypatch.setattr(runner, "start_sumo", lambda *a, **k: 1)
    monkeypatch.setattr(runner, "close_sumo", lambda: None)
    progress = []
    reg = {"scene_paths": {"PI-UNSAT-MIXED": "scene"}}
    result = runner.run_case(reg, tmp_path / "run", "sumo", "PI-UNSAT-MIXED", progress.append)
    assert progress == [.5, 1.0, 1.5, 2.0]
    assert len(result["physical"]) == 4
    assert len(result["decisions"]) == len(result["owner_receipts"]) == 1
    assert len(result["adapter_evidence"]) == 1
    assert result["frames"][0]["provided_to_controller"] is True
    assert [row["method"] for row in result["owner_commands"]] == ["slowDown", "setSpeed"]
    assert result["status"] == "complete"
    report = verify(result, "PI-UNSAT-MIXED")
    assert report["passed"] is True, report


def test_telemetry_failure_preserves_progress_and_failure_artifact(tmp_path, monkeypatch):
    sim = Sim()
    vehicle = Vehicle(sim)
    original = vehicle.getSpeed
    def fail(v):
        if sim.time >= 1.5:
            raise RuntimeError("telemetry failure")
        return original(v)
    vehicle.getSpeed = fail
    traci = SimpleNamespace(simulation=sim, simulationStep=sim.simulationStep,
                            vehicle=vehicle, lane=Lane(vehicle))
    monkeypatch.setitem(__import__("sys").modules, "traci", traci)
    monkeypatch.setattr(runner, "start_sumo", lambda *a, **k: 1)
    monkeypatch.setattr(runner, "close_sumo", lambda: None)
    with pytest.raises(RuntimeError, match="telemetry failure"):
        runner.run_case({"scene_paths": {"PI-UNSAT-MIXED": "scene"}},
                        tmp_path / "run", "sumo", "PI-UNSAT-MIXED")
    assert (tmp_path / "run" / "result.json").exists()
    assert (tmp_path / "run" / "result.partial.json").exists()
