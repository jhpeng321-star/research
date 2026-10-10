"""Full-runner integration test for the SELECTOR mechanics fixture (offline).

Drives the REAL ppo_selector_mech.run_case path — real SelectorWallPolicy
(wall_ppo/3), real offer binding, real CommandOwnership, real CirclesController
(frozen ONNX damping), real RequestLedger/collectors, budget reservation,
clock contract, completion checks and summary construction — against a MOCKED
simulator boundary only (same point-mass world shape as the mechanics
integration test, populated with the FOUR selector-fixture CAVs and two HDV
followers). No SUMO process is launched.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from traffic_shepherd import ppo_selector_mech
from traffic_shepherd.config import load_config

ALL_IDS = (*ppo_selector_mech.MEMBERS, *ppo_selector_mech.FOLLOWERS)
CONFIG = load_config()
POSITIONS = {"member_a": 1000.0, "member_b": 990.0,
             "member_c": 400.0, "member_d": 390.0,
             "follower_a": 610.0, "follower_b": 600.0}
LANES_BY_ID = {"member_a": "road_0", "member_b": "road_1",
               "member_c": "road_0", "member_d": "road_1",
               "follower_a": "road_0", "follower_b": "road_1"}


class FakeVehicleDomain:
    """Point-mass kinematics: slowDown/setSpeed set a desired speed
    approached at a bounded 2.6 m/s^2 rate; setSpeed(-1) releases."""

    def __init__(self, world):
        self.world = world

    def _v(self, vid):
        return self.world.vehicles[vid]

    def getIDList(self):
        return list(self.world.vehicles)

    def getLaneID(self, vid):
        return self._v(vid)["lane"]

    def getLanePosition(self, vid):
        return self._v(vid)["position"]

    def getLength(self, vid):
        return 5.0

    def getSpeed(self, vid):
        return self._v(vid)["speed"]

    def getAcceleration(self, vid):
        return self._v(vid)["accel"]

    def getAllowedSpeed(self, vid):
        return 33.0

    def getLaneIndex(self, vid):
        return 0 if self._v(vid)["lane"].endswith("_0") else 1

    def getSpeedMode(self, vid):
        return CONFIG.speed_mode

    def setLaneChangeMode(self, vid, mode):
        self.world.lane_modes[vid] = mode

    def setSpeed(self, vid, speed):
        self._v(vid)["desired"] = 27.0 if speed == -1 else float(speed)

    def slowDown(self, vid, target_speed, duration):
        self._v(vid)["desired"] = float(target_speed)

    def getMinGap(self, vid):
        return 2.5

    def getLeader(self, vid, dist):
        me = self._v(vid)
        candidates = []
        for other in self.world.order:
            if other == vid:
                continue
            state = self.world.vehicles[other]
            if state["lane"] != me["lane"]:
                continue
            gap = state["position"] - 5.0 - me["position"]
            if 0.0 <= gap <= dist:
                candidates.append((gap, other))
        if not candidates:
            return None
        gap, other = min(candidates)
        return (other, gap)


class FakeLaneDomain:
    def __init__(self, world):
        self.world = world

    def getLastStepVehicleIDs(self, lane_id):
        return [vid for vid in self.world.order
                if self.world.vehicles[vid]["lane"] == lane_id]

    def getLength(self, lane_id):
        return 3500.0


class FakeSimulationDomain:
    def __init__(self, world):
        self.world = world

    def simulationStep(self, *_args):
        world = self.world
        world.calls += 1
        for vid in world.vehicles:
            state = world.vehicles[vid]
            previous = state["speed"]
            desired = state.get("desired", 27.0)
            change = max(-2.6 * .5, min(2.6 * .5, desired - previous))
            state["speed"] = max(0.0, previous + change)
            state["accel"] = (state["speed"] - previous) / .5
            state["position"] += state["speed"] * .5
        world.time = round(world.time + .5, 6)
        if world.calls == 1 and not world.departed_emitted:
            world.departed_emitted = True
            world.last_departed = list(world.order)
        return world.time

    def getTime(self):
        return self.world.time

    def getDepartedIDList(self):
        out, self.world.last_departed = self.world.last_departed, []
        return out

    def getArrivedIDList(self):
        return []

    def getCollidingVehiclesNumber(self):
        return 0

    def getStartingTeleportNumber(self):
        return 0

    def getCollidingVehiclesIDList(self):
        return []


class FakeTraci:
    def __init__(self):
        self.world = SimpleNamespace(
            time=0.0, calls=0, departed_emitted=False, last_departed=[],
            lane_modes={}, order=list(ALL_IDS), anomalies={},
            vehicles={vid: dict(lane=LANES_BY_ID[vid],
                                position=POSITIONS[vid], speed=27.0,
                                accel=0.0, desired=27.0)
                      for vid in ALL_IDS})
        self.vehicle = FakeVehicleDomain(self.world)
        self.lane = FakeLaneDomain(self.world)
        self.simulation = FakeSimulationDomain(self.world)

    def simulationStep(self, *_args):
        return self.simulation.simulationStep(*_args)

    def getVersion(self):
        return [22, "SUMO 1.27.1 (mock)"]

    def isLoaded(self):
        return False

    @property
    def steps(self):
        return self.world.calls


def install_fake_traci(monkeypatch):
    fake = FakeTraci()
    monkeypatch.setitem(sys.modules, "traci", fake)
    monkeypatch.setattr(ppo_selector_mech, "start_sumo",
                        lambda *a, **k: 54321)
    return fake


def stub_build_fixture(monkeypatch):
    requests = [SimpleNamespace(vehicle_id=vid, depart_s=0.0,
                                driver_type="idm", is_cav=vid in
                                ppo_selector_mech.MEMBERS)
                for vid in ALL_IDS]

    def _stub(output, sumo_binary):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "MOCKED_NO_SIMULATOR.txt").write_text(
            "scenario builder mocked; simulator boundary mocked",
            encoding="utf-8")
        return output / "scenario.sumocfg", requests
    monkeypatch.setattr(ppo_selector_mech, "build_selector_fixture", _stub)


def generous_budget(tmp_path):
    return dict(campaign="ppo_sel_mech_int", category="mechanics",
                amount_s=120.0, ledger_path=str(tmp_path / "ledger.json"),
                ceilings=dict(categories={
                    "mechanics": {"max_allocations": 2, "max_seconds": 240.0}},
                    total_seconds=134640.0),
                campaign_stop_on_failure=True)


def run_case(tmp_path, monkeypatch, budget):
    fake = install_fake_traci(monkeypatch)
    stub_build_fixture(monkeypatch)
    data = dict(registration_sha256="x", sumo_binary="mocked-sumo")
    summary = ppo_selector_mech.run_case(tmp_path, data, "selector_ppo_chain",
                                         budget)
    return fake, summary


# ---- valid complete selector chain ------------------------------------------

def test_populated_selector_chain_passes_every_check(tmp_path, monkeypatch):
    fake, summary = run_case(tmp_path, monkeypatch, generous_budget(tmp_path))
    assert summary["status"] == "complete" and summary["passed"] is True
    for name, value in summary["checks"].items():
        assert value is True, f"check {name} failed"
    assert summary["completed_simulated_seconds"] == pytest.approx(120.0)
    assert fake.steps == 240
    assert len(summary["clock_history"]) == 240
    # the required action transitions of the verified chain
    log = summary["decision_log"]
    assert any(d["executed_action"] == "activate" for d in log)
    assert any(d["executed_action"] == "extend" for d in log)
    assert any(d["executed_action"] == "release" for d in log)
    assert any(d["requested_action"] == 1 and d["executed_action"] is None
               and d["rejection_reason"] == "activate_masked_or_out_of_window"
               for d in log)
    # offer binding evidence: nontrivial deterministic search, one committed
    # offer, fingerprint equality with the committed admission
    inactive_offers = [d.get("offer") for d in log
                       if d.get("phase") == "inactive"
                       and d.get("offer")]
    assert max(o["assessed_pairs"] for o in inactive_offers) == 4
    assert len({tuple(sorted(o["member_ids"])) for o in inactive_offers}) == 1
    activation = [d for d in log if d["executed_action"] == "activate"][0]
    assert activation["offer"]["committed"] is True
    assert activation["committed_proposal_sha256"] == \
        activation["offer"]["proposal_fingerprint"]
    assert summary["activation_used"] == 1
    assert summary["delivered_hold_s"] == pytest.approx(30.0)
    assert summary["ownership"]["collisions"] == 0
    assert summary["ownership"]["teleports"] == 0
    assert summary["ownership"]["protective_abort"] is False
    # search-timing instrumentation present (fixture scale only)
    timing = summary["selector_timing"]
    assert timing["calls"] >= 1 and timing["total_s"] >= 0.0
    assert any("search_time_s" in d for d in log)
    # rollout rows carry the offered proposal metadata
    rollout = json.loads((tmp_path / "jobs" / "selector_ppo_chain" /
                          "rollout.json").read_text(encoding="utf-8"))
    offers = [r.get("offer") for r in rollout if r.get("offer")]
    assert offers and any(o.get("committed") for o in offers)
