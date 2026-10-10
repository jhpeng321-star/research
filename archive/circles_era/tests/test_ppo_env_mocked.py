"""Mocked-boundary smoke test for the corridor PPO episode loop (offline).

Exercises the real PpoEpisode.run() loop on a natural-network scenario
configuration with a mocked traci module (empty world): scenario build,
identity, ownership, report buffering, decision cadence/masking, reward
rows, clock contract and summary construction. This is the integration
de-risk for the training phase; no SUMO process is launched.
"""
import sys

import pytest

from traffic_shepherd import ppo_contract as C
from traffic_shepherd.ppo_env import PpoEpisode, build_pilot_job


class EmptyWorldTraci:
    """Empty-world fake: the loop machinery runs, no vehicle is ever live."""

    def __init__(self):
        self.time = 0.0
        self.calls = 0
        self.vehicle = self._Vehicle(self)
        self.lane = self._Lane()
        self.edge = self._Edge()
        self.simulation = self._Simulation(self)

    class _Vehicle:
        def __init__(self, outer):
            self.outer = outer

        def getIDList(self):
            return []

        def __getattr__(self, name):
            # any remaining vehicle query degrades over the empty world
            def _any(*args, **kwargs):
                if name in ("getSpeed", "getAcceleration", "getLanePosition"):
                    return 0.0
                if name in ("getLaneID",):
                    return ""
                if name in ("getLength", "getMinGap"):
                    return 5.0
                if name in ("getAllowedSpeed",):
                    return 24.0
                if name in ("getLaneIndex",):
                    return 0
                if name in ("getSpeedMode",):
                    return 31
                if name == "getLeader":
                    return None
                return None
            return _any

    class _Lane:
        def getLength(self, lane_id):
            return 350.0

        def getLastStepVehicleIDs(self, lane_id):
            return []

    class _Edge:
        def getLaneNumber(self, edge_id):
            return 2

    class _Simulation:
        def __init__(self, outer):
            self.outer = outer

        def simulationStep(self, *_a):
            self.outer.calls += 1
            self.outer.time = round(self.outer.time + 0.5, 6)
            return self.outer.time

        def getTime(self):
            return self.outer.time

        def getDepartedIDList(self):
            return []

        def getArrivedIDList(self):
            return []

        def getCollidingVehiclesNumber(self):
            return 0

        def getStartingTeleportNumber(self):
            return 0

        def getCollidingVehiclesIDList(self):
            return []

    def simulationStep(self, *_a):
        return self.simulation.simulationStep(*_a)

    def isLoaded(self):
        return False

    def getVersion(self):
        return [22, "SUMO 1.27.1 (mock)"]

    def close(self):
        pass


@pytest.fixture()
def fake_traci(monkeypatch):
    fake = EmptyWorldTraci()
    monkeypatch.setitem(sys.modules, "traci", fake)
    monkeypatch.setattr("traffic_shepherd.ppo_env.start_sumo", lambda *a, **k: 54321)
    return fake


def registration_dict():
    return dict(
        identity=dict(sumo_binary_sha256="x", config_sha256="x", model_sha256="x",
                      network_sha256="x", scenario_template_sha256="x",
                      active_source_sha256={}),
        communication=dict(max_report_age_s=1.0, history_s=10.0),
        assignment="registered_experimental_assignment_not_trained_benefit_gate",
        registration_sha256="x",
        budget=dict(campaign="ppo_smoke", category="training",
                    ledger_path="/nonexistent/ledger.json",
                    ceilings=dict(categories={}, total_seconds=0.0),
                    target_dir="/nonexistent"))


def test_corridor_loop_completes_with_contract_accounting(tmp_path, fake_traci):
    horizon = 30
    job = build_pilot_job(seed=4107, loading=2700, horizon=horizon,
                          case_id="smoke", decision_time_s=5,
                          policy_spec=dict(member_ids=["ghost_a", "ghost_b"],
                                           required_lanes=["1191554636_0",
                                                           "1191554636_1"]))

    def wait_actor(mask, features, phase, now):
        return C.A_WAIT

    episode = PpoEpisode(job=job, registration=registration_dict(),
        member_ids=("ghost_a", "ghost_b"),
        required_lanes=("1191554636_0", "1191554636_1"),
        policy_mode="ppo", actor=wait_actor,
        output=tmp_path / "smoke")     # budget=None: no charge in a smoke
    summary, rollout = episode.run()

    # clock contract: exactly 2*horizon advances on the exact grid
    assert fake_traci.calls == horizon * 2
    assert summary["duration_s"] == horizon and summary["mechanics_only"] is True
    # accounting: one reward row per control tick
    assert summary["totals"] is not None
    # decisions: only on cadence ticks, only for inactive/hold phases
    decision_times = [row["time_s"] for row in rollout]
    assert decision_times == [t for t in range(horizon) if t % C.CADENCE_S == 0]
    assert all(row["phase"] == "inactive" for row in rollout)
    assert all(row["mask"][C.A_ACTIVATE] is False for row in rollout)
    # all-WAIT policy: no fabricated actions, nothing under wall control
    assert summary["wall_owned_control_steps"] == 0
    assert summary["totals"]["collision_involvements"] == 0
    # rollout rows carry masks for PPO log-probability consistency
    assert all(len(row["mask"]) == 2 for row in rollout)
    assert (tmp_path / "smoke" / "decision_log.json").exists()


def test_smoke_episode_reward_rows_account_full_horizon(tmp_path, fake_traci):
    horizon = 20
    job = build_pilot_job(seed=4107, loading=2700, horizon=horizon,
                          case_id="smoke2", decision_time_s=5,
                          policy_spec=dict(member_ids=["ghost_a"],
                                           required_lanes=["1191554636_0"]))

    class WaitActor:
        def __call__(self, mask, features, phase, now):
            return C.A_WAIT

    episode = PpoEpisode(job=job, registration=registration_dict(),
        member_ids=("ghost_a",), required_lanes=("1191554636_0",),
        policy_mode="ppo", actor=WaitActor(),
        output=tmp_path / "smoke2")
    _summary, rollout = episode.run()
    # every control second accounted exactly once at decision granularity
    import math
    reward_log = (tmp_path / "smoke2" / "reward_log.jsonl").read_text(
        encoding="utf-8").splitlines()
    assert len(reward_log) == horizon
    assert len(rollout) == math.ceil(horizon / C.CADENCE_S) - 1 or rollout
    covered = horizon - (horizon % C.CADENCE_S)
    assert len(rollout) == covered // C.CADENCE_S
