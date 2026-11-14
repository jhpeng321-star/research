"""CV10 launch-contract integration with a fake TraCI boundary only."""
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import output.controller_verification_v10.independent_review as review
import output.controller_verification_v10.launch_contract as contract
import output.controller_verification_v10.runner as runner


class World:
    def __init__(self, leader_front):
        self.time = 0.0
        self.leader_front = leader_front

    def step(self):
        self.time += 0.5


class Vehicle:
    def __init__(self, world):
        self.world = world
        self.writes = []

    def getIDList(self):
        return [] if self.world.time == 0.5 else ["cav", "leader_hdv", "rear_hdv"]

    def getTypeID(self, vehicle):
        return {"cav": "idm", "leader_hdv": "leader_type", "rear_hdv": "rear_type"}[vehicle]

    def getSpeedMode(self, vehicle): return 31
    def getSpeed(self, vehicle): return 1.0 if vehicle != "leader_hdv" else 0.5
    def getAcceleration(self, vehicle): return 0.0
    def getAllowedSpeed(self, vehicle): return 30.0 if vehicle == "cav" else 1.0
    def getLaneIndex(self, vehicle): return 0
    def getLaneID(self, vehicle): return "road_0"
    def getLanePosition(self, vehicle):
        return {"cav": 100.0, "leader_hdv": self.world.leader_front, "rear_hdv": 50.0}[vehicle]
    def getLength(self, vehicle): return 5.0
    def getMaxSpeed(self, vehicle):
        return {"cav": 30.0, "leader_hdv": .5, "rear_hdv": 1.0}[vehicle]
    def slowDown(self, vehicle, target, duration): self.writes.append(("slowDown", vehicle, target, duration))
    def setSpeed(self, vehicle, target): self.writes.append(("setSpeed", vehicle, target))


class Lane:
    def __init__(self, vehicle): self.vehicle = vehicle
    def getLength(self, lane): return 2000.0
    def getLastStepVehicleIDs(self, lane): return self.vehicle.getIDList()


def _setup(tmp_path, monkeypatch):
    base = tmp_path / "base"
    base.mkdir()
    frozen_root = base / "frozen_inputs"
    frozen_root.mkdir()
    folders = {}
    specs = {
        "PI-UNSAT-MIXED": {"leader_front_m": 109.06, "gap_m": 4.06, "raw_mps": 1.4173445},
        "PI-BRAKE-ZERO": {"leader_front_m": 108.8, "gap_m": 3.8, "raw_mps": .5},
    }
    for case, spec in specs.items():
        folder = frozen_root / case
        folder.mkdir()
        (folder / "nodes.xml").write_text('<nodes><node id="a" x="0" y="0"/><node id="b" x="2000" y="0"/></nodes>')
        (folder / "edges.xml").write_text('<edges><edge id="road" from="a" to="b" numLanes="1" speed="30"/></edges>')
        common = 'carFollowModel="IDM" accel="2.6" decel="4.5" tau="1.1" minGap="2.5" sigma="0" speedFactor="1" speedDev="0" length="5"'
        routes = '<routes>\n<vType id="leader_type" maxSpeed="0.5" ' + common + '/>\n<vType id="idm" maxSpeed="30" ' + common + '/>\n<vType id="rear_type" maxSpeed="1" ' + common + '/>\n<route id="route" edges="road"/>\n'
        for vehicle, kind, position, speed in (("leader_hdv", "leader_type", spec["leader_front_m"], .5), ("cav", "idm", 100, 1), ("rear_hdv", "rear_type", 50, 1)):
            routes += f'<vehicle id="{vehicle}" type="{kind}" route="route" depart="0.5" departPos="{position}" departSpeed="{speed}"/>\n'
        (folder / "routes.rou.xml").write_text(routes + '</routes>')
        (folder / "network.net.xml").write_text("network")
        (folder / "scenario.sumocfg").write_text("scene")
        folders[case] = folder
    binary = tmp_path / "sumo.exe"
    converter = tmp_path / "netconvert.exe"
    shutil.copy2(Path("C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe"), binary)
    shutil.copy2(binary, converter)
    output = tmp_path / "output"
    ledger = tmp_path / "budget.json"
    monkeypatch.setattr(contract, "BASE", base)
    monkeypatch.setattr(contract, "OUTPUT", output)
    monkeypatch.setattr(contract, "LEDGER", ledger)
    monkeypatch.setattr(contract, "BINARY", binary)
    monkeypatch.setattr(contract, "CASES", specs)
    monkeypatch.setattr(contract, "sources", lambda: {"test_source": "fixture"})
    scene_paths = {case: str(folder / "scenario.sumocfg") for case, folder in folders.items()}
    scene_hash = {case: contract.scene_hashes(folder) for case, folder in folders.items()}
    reg = dict(schema=contract.VERSION, experiment_id="CV-10", campaign=contract.CAMPAIGN,
               seed=1, seconds_per_case=2.0, budget_seconds=4.0, automatic_retries=0,
               cases=specs, output_dir=str(output), ledger_path=str(ledger), ceilings=contract.CEILINGS,
               source_sha256={"test_source": "fixture"},
               executables={str(binary): {"sha256": contract.sha(binary)},
                            str(converter): {"sha256": contract.sha(converter)}},
               scene_paths=scene_paths, scene_sha256=scene_hash)
    return reg, binary, output, specs


@pytest.mark.parametrize('fail_first', [False, True])
def test_real_launch_contract_runs_both_cases_and_reviews_them(tmp_path, monkeypatch, fail_first):
    reg, binary, output, specs = _setup(tmp_path, monkeypatch)
    current = {}
    traci = SimpleNamespace()

    def start(*args, **kwargs):
        case = "PI-UNSAT-MIXED" if current.get("index", 0) == 0 else "PI-BRAKE-ZERO"
        world = World(specs[case]["leader_front_m"])
        vehicle = Vehicle(world)
        current["world"] = world
        traci.simulation = SimpleNamespace(simulationStep=world.step, getTime=lambda: world.time,
                                           getCollidingVehiclesNumber=lambda: 2 if fail_first and world.time==1.5 else 0,
                                           getStartingTeleportNumber=lambda: 0)
        traci.simulationStep = world.step
        traci.vehicle = vehicle
        traci.lane = Lane(vehicle)
        return 1

    monkeypatch.setitem(sys.modules, "traci", traci)
    monkeypatch.setattr(runner, "start_sumo", start)
    monkeypatch.setattr(runner, "close_sumo", lambda: None)
    original_run_case = runner.run_case
    def counted_run_case(*args, **kwargs):
        result = original_run_case(*args, **kwargs)
        current["index"] = current.get("index", 0) + 1
        return result
    monkeypatch.setattr(runner, "run_case", counted_run_case)
    contract.validate(reg, output, binary)
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(json.dumps(reg))
    from traffic_shepherd.mechanics import main
    if fail_first:
        with pytest.raises(RuntimeError, match='collision'):
            main(['--candidate','pi-regions','--registration',str(registration_path),
                  '--output',str(output),'--sumo-binary',str(binary)])
        allocations=json.loads(Path(reg['ledger_path']).read_text())['reservations']
        assert len(allocations)==1 and allocations[0]['status']=='failed'
        assert allocations[0]['amount_s']==2 and allocations[0]['completed_simulated_s']==1.5
        assert not (output/'PI-BRAKE-ZERO').exists()
        failed=json.loads((output/'PI-UNSAT-MIXED/result.json').read_text())
        assert failed['physical'][-1]['collisions']==2
        assert failed['status']=='failed' and (output/'STOP.json').exists()
        return
    assert main(['--candidate','pi-regions','--registration',str(registration_path),
                 '--output',str(output),'--sumo-binary',str(binary)]) == 0
    summary = json.loads((output/'summary.json').read_text())
    assert summary["status"] == "complete"
    assert set(summary["results"]) == set(specs)
    for case in specs:
        saved = json.loads((output / case / "result.json").read_text())
        assert review.verify(saved, case)["passed"] is True
        assert json.loads((output / case / "independent_review.json").read_text())["passed"] is True
    reservations = json.loads(Path(reg['ledger_path']).read_text())['reservations']
    assert len(reservations) == 2
    assert sum(r['amount_s'] for r in reservations) == 4
    assert sum(r['completed_simulated_s'] for r in reservations) == 4
