"""Mocked live boundary for CV08; no SUMO process is started."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import output.controller_verification_v8.launch_contract as launch_contract
import output.controller_verification_v8.pi_mechanics as mechanics
import traffic_shepherd.simulation as simulation


class FakeSimulation:
    def __init__(self):
        self.time = 0.0

    def simulationStep(self):
        self.time += 0.5

    def getTime(self):
        return self.time

    def getCollidingVehiclesNumber(self):
        return 0

    def getStartingTeleportNumber(self):
        return 0


class FakeVehicle:
    ids = ("cav", "leader_hdv", "rear_hdv")

    def __init__(self, simulation):
        self.simulation = simulation
        self.writes = []

    def getIDList(self):
        return [] if self.simulation.time == 0.5 else list(self.ids)

    def getSpeedMode(self, vehicle):
        return 31

    def getTypeID(self, vehicle):
        return {"cav": "idm", "leader_hdv": "leader_type", "rear_hdv": "rear_type"}[vehicle]

    def getSpeed(self, vehicle):
        return 15.0

    def getAcceleration(self, vehicle):
        return 0.0

    def getAllowedSpeed(self, vehicle):
        return 30.0

    def getLaneIndex(self, vehicle):
        return 0

    def getLaneID(self, vehicle):
        return "road_0"

    def getLanePosition(self, vehicle):
        return {"cav": 100.0, "leader_hdv": 150.0, "rear_hdv": 50.0}[vehicle]

    def getLength(self, vehicle):
        return 5.0

    def getMaxSpeed(self, vehicle):
        return 30.0 if vehicle == "cav" else 15.0

    def slowDown(self, vehicle, target, duration):
        self.writes.append(("slowDown", vehicle, target, duration))

    def setSpeed(self, vehicle, target):
        self.writes.append(("setSpeed", vehicle, target))


class FakeLane:
    def __init__(self, vehicle):
        self.vehicle = vehicle

    def getLength(self, lane_id):
        return 2000.0

    def getLastStepVehicleIDs(self, lane_id):
        return self.vehicle.getIDList()


def _registration(tmp_path, monkeypatch):
    base = tmp_path / "base"
    base.mkdir()
    frozen = base / "frozen_inputs"
    frozen.mkdir()
    (frozen / "nodes.xml").write_text('<nodes><node id="a" x="0" y="0"/><node id="b" x="2000" y="0"/></nodes>')
    (frozen / "edges.xml").write_text('<edges><edge id="road" from="a" to="b" numLanes="1" speed="30"/></edges>')
    common = 'carFollowModel="IDM" accel="2.6" decel="4.5" tau="1.1" minGap="2.5" sigma="0" speedFactor="1" speedDev="0" length="5"'
    routes = '<routes>\n<vType id="idm" maxSpeed="30" ' + common + '/>\n'
    routes += '<vType id="leader_type" maxSpeed="15" ' + common + '/>\n<vType id="rear_type" maxSpeed="15" ' + common + '/>\n<route id="route" edges="road"/>\n'
    for vehicle, kind, position in (("leader_hdv", "leader_type", 150), ("cav", "idm", 100), ("rear_hdv", "rear_type", 50)):
        routes += f'<vehicle id="{vehicle}" type="{kind}" route="route" depart="0.5" departPos="{position}" departSpeed="15"/>\n'
    (frozen / "routes.rou.xml").write_text(routes + '</routes>')
    (frozen / "network.net.xml").write_text("network")
    (frozen / "scenario.sumocfg").write_text("scene")
    binary = tmp_path / "sumo.exe"
    converter = tmp_path / "netconvert.exe"
    shutil.copy2(Path("C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe"), binary)
    shutil.copy2(binary, converter)
    output = tmp_path / "runs" / "cv08"
    ledger = tmp_path / "budget.json"
    monkeypatch.setattr(launch_contract, "BASE", base)
    monkeypatch.setattr(launch_contract, "OUTPUT", output)
    monkeypatch.setattr(launch_contract, "LEDGER", ledger)
    monkeypatch.setattr(launch_contract, "BINARY", binary)
    monkeypatch.setattr(launch_contract, "sources", lambda: {"test_source": "fixture"})
    hashes = {str(frozen / name): launch_contract.sha(frozen / name)
              for name in ("nodes.xml", "edges.xml", "routes.rou.xml", "network.net.xml", "scenario.sumocfg")}
    reg = dict(mechanics.EXPECTED, experiment_id="CV-08", seed=1, automatic_retries=0,
               output_dir=str(output), ledger_path=str(ledger),
               budget={"ceilings": launch_contract.CEILINGS},
               source_sha256={"test_source": "fixture"},
               executables={str(binary): {"sha256": launch_contract.sha(binary)},
                            str(converter): {"sha256": launch_contract.sha(converter)}},
               scene_path=str(frozen / "scenario.sumocfg"), fixture_sha256=hashes)
    return reg, binary, output


def test_registered_pipeline_reaches_real_owner_and_persists_boundary(tmp_path, monkeypatch):
    reg, binary, output = _registration(tmp_path, monkeypatch)
    sim = FakeSimulation()
    vehicle = FakeVehicle(sim)
    traci = SimpleNamespace(simulation=sim, simulationStep=sim.simulationStep,
                            vehicle=vehicle, lane=FakeLane(vehicle),
                            getVersion=lambda: ("SUMO", "1.27.1"))
    monkeypatch.setitem(__import__("sys").modules, "traci", traci)
    monkeypatch.setattr(simulation, "start_sumo", lambda *args, **kwargs: 8811)
    monkeypatch.setattr(simulation, "close_sumo", lambda: None)
    from traffic_shepherd.mechanics import main
    registration_path = tmp_path / 'registration.json'
    registration_path.write_text(json.dumps(reg))
    exit_code = main(['--candidate', 'pi-external-u', '--registration', str(registration_path),
                      '--output', str(output), '--sumo-binary', str(binary)])
    saved = json.loads((output / "result.json").read_text())
    assert exit_code == 0 and saved['status'] == 'complete'
    assert json.loads((output / 'independent_review.json').read_text())['passed'] is True
    assert saved["actual_progress_s"] == 8.0
    assert [row["time_s"] for row in saved["physical"]] == [i / 2 for i in range(1, 17)]
    assert len(saved["decisions"]) == 7
    assert len(saved["adapter_evidence"]) == 5
    assert any(row["succeeded"] for row in saved["owner_commands"])
    assert saved["frames"][4]["provided_to_controller"] is False


def test_live_telemetry_failure_leaves_progress_and_failed_receipt(tmp_path, monkeypatch):
    reg, binary, output = _registration(tmp_path, monkeypatch)
    sim = FakeSimulation()
    vehicle = FakeVehicle(sim)
    original_speed = vehicle.getSpeed
    def fail_after_first(vehicle_id):
        if sim.time >= 1.0:
            raise RuntimeError("telemetry failure")
        return original_speed(vehicle_id)
    vehicle.getSpeed = fail_after_first
    traci = SimpleNamespace(simulation=sim, simulationStep=sim.simulationStep,
                            vehicle=vehicle, lane=FakeLane(vehicle),
                            getVersion=lambda: ("SUMO", "1.27.1"))
    monkeypatch.setitem(__import__("sys").modules, "traci", traci)
    monkeypatch.setattr(simulation, "start_sumo", lambda *args, **kwargs: 8812)
    monkeypatch.setattr(simulation, "close_sumo", lambda: None)
    with pytest.raises(RuntimeError, match="telemetry failure"):
        mechanics.run_registered(reg, output, str(binary))
    progress = json.loads((output / "progress.json").read_text())
    saved = json.loads((output / "result.json").read_text())
    assert progress["actual_progress_s"] == 0.5
    assert saved["status"] == "failed"
    assert saved["failure"]
    ledger = json.loads((tmp_path / "budget.json").read_text())
    assert ledger["reservations"][0]["status"] == "failed"
    assert ledger['reservations'][0]['completed_simulated_s'] == 1.0
