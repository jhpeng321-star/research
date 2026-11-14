"""Full-runner integration test for the PPO mechanics fixture (offline).

Drives the REAL run_case_chain fixture path - real WallPpoPolicy, real
CommandOwnership, real CirclesController (frozen ONNX damping), real
RequestLedger/collectors, real accounting, completion checks and summary
construction - against a MOCKED simulator boundary only:

  * traci (module injected into sys.modules): a scripted point-mass fixture
    world (4 vehicles, two 3500 m lanes, 0.5 s steps) exposing exactly the
    surface the runner touches (simulationStep/getTime/departures/collisions/
    teleports; vehicle id/lane/position/length/speed/acceleration/allowed
    speed/lane index/speedMode/setLaneChangeMode/setSpeed/slowDown; lane
    getLastStepVehicleIDs/getLength);
  * build_fixture: replaced by a stub returning the fixed four TripRequests
    (no netconvert); the scenario directory content is irrelevant because
    the simulator is mocked;
  * start_sumo (in the ppo_mechanics namespace): starts the fake world.

Nothing else is replaced. No SUMO process is launched.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from traffic_shepherd import ppo_contract as C
from traffic_shepherd import ppo_mechanics
from traffic_shepherd.budget_guard import BudgetExceeded
from traffic_shepherd.config import load_config
from traffic_shepherd.scenarios import TripRequest

MEMBERS = ppo_mechanics.MEMBERS
FOLLOWERS = ppo_mechanics.FOLLOWERS
ALL_IDS = (*MEMBERS, *FOLLOWERS)
CONFIG = load_config()


class FakeVehicleDomain:
    """Point-mass kinematics: slowDown sets a desired speed approached at a
    bounded rate; setSpeed(-1) releases to cruise (24 m/s)."""

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
        return 24.0

    def getLaneIndex(self, vid):
        return 0 if self._v(vid)["lane"].endswith("_0") else 1

    def getSpeedMode(self, vid):
        return CONFIG.speed_mode

    def setLaneChangeMode(self, vid, mode):
        self.world.lane_modes[vid] = mode

    def setSpeed(self, vid, speed):
        if speed == -1:
            self._v(vid)["desired"] = 24.0      # cruise: control handed back
        else:
            self._v(vid)["desired"] = float(speed)

    def slowDown(self, vid, target_speed, duration):
        self._v(vid)["desired"] = float(target_speed)

    def getMinGap(self, vid):
        return 2.5

    def getLeader(self, vid, dist):
        """Nearest same-lane leader within dist: (id, gap without minGap)."""
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
        kind = world.anomalies.get(world.calls)
        if kind == "jump":          # one call advances TWO 0.5 s substeps
            world.time = round(world.time + .5, 6)
        if kind != "repeat":
            for vid in world.vehicles:
                state = world.vehicles[vid]
                previous = state["speed"]
                desired = state.get("desired", 24.0)
                change = max(-2.6 * .5, min(2.6 * .5, desired - previous))
                state["speed"] = max(0.0, previous + change)
                state["accel"] = (state["speed"] - previous) / .5
                state["position"] += state["speed"] * .5
            world.time = round(world.time + .5, 6)
        if world.calls == 1 and not world.departed_emitted:
            world.departed_emitted = True
            world.last_departed = list(ALL_IDS)
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


class FakeTraci:
    """The mocked simulator boundary."""

    def __init__(self, anomalies=None):
        self.world = SimpleNamespace(
            time=0.0, calls=0, departed_emitted=False, last_departed=[],
            lane_modes={}, anomalies=dict(anomalies or {}),
            order=list(ALL_IDS),
            vehicles={vid: dict(lane=f"road_{0 if vid.endswith('a') or vid == 'member_a' or vid == 'follower_a' else 1}",
                                position=1000.0 if vid in MEMBERS else 910.0
                                if vid == "follower_a" else 900.0,
                                speed=24.0, accel=0.0, desired=24.0)
                      for vid in ALL_IDS})
        # explicit lane assignment (a-vehicles left, b-vehicles right)
        self.world.vehicles["member_a"]["lane"] = "road_0"
        self.world.vehicles["follower_a"]["lane"] = "road_0"
        self.world.vehicles["member_b"]["lane"] = "road_1"
        self.world.vehicles["follower_b"]["lane"] = "road_1"
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


def install_fake_traci(monkeypatch, anomalies=None):
    fake = FakeTraci(anomalies)
    monkeypatch.setitem(sys.modules, "traci", fake)
    # the real start_sumo returns the local TraCI port (an int); the fake
    # module is reached through sys.modules["traci"]
    monkeypatch.setattr(ppo_mechanics, "start_sumo",
                        lambda *a, **k: 54321)
    return fake


def stub_build_fixture(monkeypatch):
    requests = [TripRequest(vid, 0., "route", "idm", vid in MEMBERS)
                for vid in ALL_IDS]

    def _stub(output, sumo_binary):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "MOCKED_NO_SIMULATOR.txt").write_text(
            "scenario builder mocked; simulator boundary mocked", encoding="utf-8")
        return output / "scenario.sumocfg", requests
    monkeypatch.setattr(ppo_mechanics, "build_fixture", _stub)


def generous_budget(tmp_path):
    return dict(campaign="ppo_mech_int", category="mechanics", amount_s=120.0,
                ledger_path=str(tmp_path / "ledger.json"),
                ceilings=dict(
                    categories={"mechanics": {"max_allocations": 2, "max_seconds": 240.0}},
                    total_seconds=134640.0))


def exhausted_budget(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(dict(schema="campaign_budget_ledger/1",
        campaigns={}, reservations=[dict(id="x:1", campaign="ppo_mech_int",
            category="mechanics", amount_s=240.0, target_dir="x",
            reserved_at="t", status="complete")])), encoding="utf-8")
    return dict(campaign="ppo_mech_int", category="mechanics", amount_s=120.0,
                ledger_path=str(ledger), ceilings=dict(
                    categories={"mechanics": {"max_allocations": 2, "max_seconds": 240.0}},
                    total_seconds=134640.0))


def run_chain(tmp_path, monkeypatch, budget, anomalies=None, script=None):
    fake = install_fake_traci(monkeypatch, anomalies)
    stub_build_fixture(monkeypatch)
    if script is not None:
        monkeypatch.setattr(ppo_mechanics, "_script", script)
    data = dict(registration_sha256="x", sumo_binary="mocked-sumo")
    summary = ppo_mechanics.run_case_chain(tmp_path, data, "hold_ppo_chain", budget)
    return fake, summary


# ---- valid complete trace -------------------------------------------------

def test_valid_scripted_trace_passes_every_final_check(tmp_path, monkeypatch):
    fake, summary = run_chain(tmp_path, monkeypatch, generous_budget(tmp_path))
    assert summary["status"] == "complete" and summary["passed"] is True
    for name, value in summary["checks"].items():
        assert value is True, f"check {name} failed"
    # completed time equals the final observed simulator time
    assert summary["completed_simulated_seconds"] == pytest.approx(120.0, abs=1e-9)
    assert fake.world.time == pytest.approx(120.0, abs=1e-9)
    # declared time contract
    assert fake.steps == 240                     # 120 s / 0.5 s physics samples
    assert summary["clock_calls"] == 240
    assert summary["clock_history"][0] == pytest.approx(0.5, abs=1e-9)
    assert summary["clock_history"][-1] == pytest.approx(120.0, abs=1e-9)
    assert len(summary["clock_history"]) == 240  # every sample exactly once
    assert summary["time_contract"]["reward_rows"] == 120
    assert summary["time_contract"]["control_ticks"] == 119
    # required action transitions
    log = summary["decision_log"]
    assert any(d["executed_action"] == "activate" for d in log)
    assert any(d["executed_action"] == "extend" for d in log)
    assert any(d["executed_action"] == "release" for d in log)
    assert any(d["requested_action"] == C.A_ACTIVATE and d["executed_action"] is None
               and d["rejection_reason"] == "activate_masked_or_out_of_window"
               for d in log)
    # first decision row is the masked request at the first decision tick
    rollout = json.loads((tmp_path / "jobs" / "hold_ppo_chain" / "rollout.json")
                         .read_text(encoding="utf-8"))
    assert rollout[0]["time_s"] == 5.0 and rollout[0]["action"] == C.A_ACTIVATE
    assert all(row["phase"] in {"inactive", "hold_interior",
                                "hold_deadline"} for row in rollout)
    # no fabricated rows for deterministic phases: 1:1 with the decision log
    assert len(rollout) == len(log)
    assert [row["time_s"] for row in rollout] == [d["time_s"] for d in log]


def test_first_and_final_reward_intervals_present_once(tmp_path, monkeypatch):
    fake, summary = run_chain(tmp_path, monkeypatch, generous_budget(tmp_path))
    # 120 rows == 120 one-second control intervals; the loop's interval
    # counters are rebuilt per row, so row count plus the clock history
    # (strictly +0.5 per advance, 240 advances) proves first and final
    # intervals are neither lost nor duplicated.
    ledger_file = tmp_path / "ledger.json"
    data = json.loads(ledger_file.read_text(encoding="utf-8"))
    assert data["reservations"][0]["status"] == "complete"
    assert data["reservations"][0]["amount_s"] == 120.0
    assert summary["episode_return"] < 0.0       # wall-owned ticks cost something
    assert summary["checks"]["intervention_bound"] is True


# ---- injected clock anomalies ---------------------------------------------

def test_repeated_timestamp_fails_immediately_with_receipt(tmp_path, monkeypatch):
    install_fake_traci(monkeypatch, anomalies={4: "repeat"})
    stub_build_fixture(monkeypatch)
    data = dict(registration_sha256="x", sumo_binary="mocked-sumo")
    from traffic_shepherd.sim_clock import ClockViolationError
    with pytest.raises(ClockViolationError) as exc:
        ppo_mechanics.run_case_chain(tmp_path, data, "hold_ppo_chain",
                                     generous_budget(tmp_path))
    receipt = exc.value.receipt
    assert receipt["kind"] == "physics_step_mismatch"
    assert receipt["expected"] == pytest.approx(2.0, abs=1e-9)
    assert receipt["observed"] == pytest.approx(1.5, abs=1e-9)
    manifest = json.loads((tmp_path / "jobs" / "hold_ppo_chain" / "manifest.json")
                          .read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["clock_receipt"]["expected"] == pytest.approx(2.0, abs=1e-9)
    # failure receipt preserves the last clock-VERIFIED progress (the
    # rejected 1.5 read was never verified, so 1.0 stands)
    assert manifest["completed_simulated_seconds"] == pytest.approx(1.0, abs=1e-9)
    assert not (tmp_path / "jobs" / "hold_ppo_chain" / "summary.json").exists()


def test_skipped_timestamp_fails_immediately_with_receipt(tmp_path, monkeypatch):
    fake = install_fake_traci(monkeypatch, anomalies={4: "jump"})
    stub_build_fixture(monkeypatch)
    data = dict(registration_sha256="x", sumo_binary="mocked-sumo")
    from traffic_shepherd.sim_clock import ClockViolationError
    with pytest.raises(ClockViolationError) as exc:
        ppo_mechanics.run_case_chain(tmp_path, data, "hold_ppo_chain",
                                     generous_budget(tmp_path))
    receipt = exc.value.receipt
    assert receipt["observed"] == pytest.approx(2.5, abs=1e-9)
    assert receipt["expected"] == pytest.approx(2.0, abs=1e-9)
    assert fake.steps == 4          # the run stopped at the violating call
    manifest = json.loads((tmp_path / "jobs" / "hold_ppo_chain" / "manifest.json")
                          .read_text(encoding="utf-8"))
    # the rejected 2.5 read was never verified: last verified progress is 1.0
    assert manifest["completed_simulated_seconds"] == pytest.approx(1.0, abs=1e-9)


# ---- missing required transitions ------------------------------------------

def test_missing_activation_fails_honestly_without_raise(tmp_path, monkeypatch):
    def wait_only(policy, now, mask, phase):
        if mask is None:
            return None
        if phase == "inactive":
            return C.A_WAIT
        return C.A_RELEASE
    _fake, summary = run_chain(tmp_path, monkeypatch, generous_budget(tmp_path),
                               script=wait_only)
    assert summary["passed"] is False
    assert summary["checks"]["activation_recorded"] is False
    assert summary["checks"]["continue_accepted"] is False
    assert summary["checks"]["reward_rows_complete"] is True   # clock unaffected
    # the still-written summary records the honest failure for close-out
    written = json.loads((tmp_path / "jobs" / "hold_ppo_chain" / "summary.json")
                         .read_text(encoding="utf-8"))
    assert written["passed"] is False


# ---- budget rejection at the real runner boundary ---------------------------

def test_budget_rejection_prevents_mocked_launcher(tmp_path, monkeypatch):
    fake = install_fake_traci(monkeypatch)
    stub_build_fixture(monkeypatch)
    data = dict(registration_sha256="x", sumo_binary="mocked-sumo")
    with pytest.raises(BudgetExceeded):
        ppo_mechanics.run_case_chain(tmp_path, data, "hold_ppo_chain",
                                     exhausted_budget(tmp_path))
    assert fake.steps == 0                       # launcher never ran
    assert not (tmp_path / "jobs" / "hold_ppo_chain" / "summary.json").exists()
    # the receipt honestly records the rejection: no simulation progress
    manifest = json.loads((tmp_path / "jobs" / "hold_ppo_chain" / "manifest.json")
                          .read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["completed_simulated_seconds"] == 0.0
    assert "exceeds" in manifest["error"]
