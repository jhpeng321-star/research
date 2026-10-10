"""Populated full-environment selector tests (offline; PP-12).

Drives the REAL selector-enabled PpoEpisode loop — real SelectorWallPolicy
(wall_ppo/3), real feature construction, masks, decision receipts,
CommandOwnership, CirclesController damping references, reward/accounting —
against a POPULATED mocked simulator boundary: four reporting CAVs (real
scenario CAV identities for seed 4107) with point-mass kinematics on the two
registered approach lanes, exposing exactly the TraCI surface the env loop
touches. No SUMO process is launched.

Boundary mocks (documented per the repository test convention): the traci
module (scripted point-mass world), start_sumo, and — for the budget
rejection test — an exhausted budget ledger. Nothing else is replaced.

Verified chain: reports before the activation window (mask closed, offer
already bound) -> deterministic selection between two candidate cohorts ->
candidate-specific observations -> ACTIVATE commits exactly the approved
offer -> membership frozen -> assembly -> MAINTAIN interior holds ->
EXTEND at the H20 deadline (H30 admitted) -> H30 delivered -> RELEASE ->
coast-terminal handback -> no reactivation -> exact clock/reward/decision
accounting.
"""
import json
import sys
from types import SimpleNamespace

import pytest

from traffic_shepherd import ppo_contract as C
from traffic_shepherd.config import load_config
from traffic_shepherd.ppo_env import PpoEpisode, build_pilot_job
from traffic_shepherd.scenarios import request_schedule

CONFIG = load_config()
LANES = ("1191554636_0", "1191554636_1")
LANE_LENGTH_M = 3500.0
# real scenario CAV identities (request_schedule is the registered demand
# builder); two coherent cohorts, one member per lane each, departing early
# enough that both cohorts report BEFORE the activation window opens
CAV_IDS = ("mainline.5", "mainline.6", "mainline.9", "mainline.11")
COHORT_A = (CAV_IDS[0], CAV_IDS[1])   # near pair (departs 10.0/10.5 s)
COHORT_B = (CAV_IDS[2], CAV_IDS[3])   # far pair behind (departs 13.0/16.5 s)
HORIZON = 120
WINDOW = (20.0, 100.0)


def world_spec():
    # both cohorts far enough upstream that the admitted H30 extension is
    # feasible at the H20 deadline (runway for hold + coast exit), as in the
    # PP-10 mechanics fixture geometry
    positions = {CAV_IDS[0]: 500.0, CAV_IDS[1]: 490.0,
                 CAV_IDS[2]: 100.0, CAV_IDS[3]: 95.0}
    lanes = {CAV_IDS[0]: LANES[0], CAV_IDS[1]: LANES[1],
             CAV_IDS[2]: LANES[0], CAV_IDS[3]: LANES[1]}
    # all vehicles start AT cruise speed (desired == speed): measured
    # acceleration is exactly 0 at entry, matching the frozen command
    # envelope the admission plan predicts against
    speeds = {vid: 24.0 for vid in CAV_IDS}
    # requested departure times from the registered demand builder, lifted
    # to the 0.5 s physics grid (the ledger refuses earlier insertions)
    requests = request_schedule(4107, 2700, penetration=C.PENETRATION)
    depart = {r.vehicle_id: r.depart_s for r in requests}
    import math
    depart_at = {vid: math.ceil(depart[vid] * 2.0) / 2.0 for vid in CAV_IDS}
    return positions, lanes, speeds, depart_at


class FakeVehicleDomain:
    """Point-mass kinematics (identical law to the mechanics integration
    boundary): desired speed approached at a bounded 2.6 m/s^2 rate. A
    vehicle is live only from its requested (0.5 s-grid) departure time."""

    def __init__(self, world):
        self.world = world

    def _v(self, vid):
        return self.world.vehicles[vid]

    def getIDList(self):
        return [vid for vid in self.world.vehicles
                if self.world.time >= self.world.depart_at[vid] - 1e-9]

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
        self._v(vid)["desired"] = 24.0 if speed == -1 else float(speed)

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
        return LANE_LENGTH_M if lane_id in LANES else 350.0


class FakeEdgeDomain:
    def getLaneNumber(self, edge_id):
        return 2


class FakeSimulationDomain:
    def __init__(self, world):
        self.world = world

    def simulationStep(self, *_args):
        world = self.world
        world.calls += 1
        world.time = round(world.time + .5, 6)
        for vid in world.order:
            if (vid not in world.departed
                    and world.time >= world.depart_at[vid] - 1e-9):
                world.departed.add(vid)
                world.pending_departed.append(vid)
        for vid in world.departed:
            state = world.vehicles[vid]
            previous = state["speed"]
            desired = state.get("desired", 24.0)
            change = max(-2.6 * .5, min(2.6 * .5, desired - previous))
            state["speed"] = max(0.0, previous + change)
            state["accel"] = (state["speed"] - previous) / .5
            state["position"] += state["speed"] * .5
        return world.time

    def getTime(self):
        return self.world.time

    def getDepartedIDList(self):
        out, self.world.pending_departed = self.world.pending_departed, []
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
        positions, lanes, speeds, depart_at = world_spec()
        self.world = SimpleNamespace(
            time=0.0, calls=0, departed=set(), pending_departed=[],
            depart_at=depart_at, lane_modes={}, order=list(CAV_IDS),
            vehicles={vid: dict(lane=lanes[vid], position=positions[vid],
                                speed=speeds[vid], accel=0.0, desired=24.0)
                      for vid in CAV_IDS})
        self.vehicle = FakeVehicleDomain(self.world)
        self.lane = FakeLaneDomain(self.world)
        self.edge = FakeEdgeDomain()
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


@pytest.fixture()
def populated_traci(monkeypatch):
    fake = FakeTraci()
    monkeypatch.setitem(sys.modules, "traci", fake)
    monkeypatch.setattr("traffic_shepherd.ppo_env.start_sumo",
                        lambda *a, **k: 54321)
    return fake


def registration_dict():
    return dict(
        identity=dict(sumo_binary_sha256="x", config_sha256="x",
                      model_sha256="x", network_sha256="x",
                      scenario_template_sha256="x", active_source_sha256={}),
        communication=dict(max_report_age_s=1.0, history_s=10.0),
        assignment="registered_experimental_assignment_not_trained_benefit_gate",
        registration_sha256="x",
        budget=dict(campaign="selector_fixture", category="training",
                    ledger_path="/nonexistent/ledger.json",
                    ceilings=dict(categories={}, total_seconds=0.0),
                    target_dir="/nonexistent"))


def selector_job(case_id):
    return build_pilot_job(seed=4107, loading=2700, horizon=HORIZON,
                           case_id=case_id, decision_time_s=5,
                           policy_spec=dict(
                               member_ids=list(COHORT_A),
                               required_lanes=list(LANES),
                               activation_window=list(WINDOW)))


def named(mask, kind, name):
    return C.ACTION_MEANINGS[kind].index(name)


class ScriptedChainActor:
    """External actor implementing the SAME declared chain as the default
    comparator, through the identical (mask, features, kind, now) boundary —
    proves a supplied actor drives the selector path."""

    def __call__(self, mask, features, kind, now):
        from traffic_shepherd.selector_wall_policy import \
            scripted_selector_chain_actor
        return scripted_selector_chain_actor(mask, features, kind, now)


class ActivateAtThenChainActor:
    """WAIT at inactive ticks before `activate_at`, then the declared chain."""

    def __init__(self, activate_at):
        self.activate_at = activate_at

    def __call__(self, mask, features, kind, now):
        if kind == "inactive":
            if now < self.activate_at or not mask[named(mask, kind, "ACTIVATE")]:
                return named(mask, kind, "WAIT")
            return named(mask, kind, "ACTIVATE")
        from traffic_shepherd.selector_wall_policy import \
            scripted_selector_chain_actor
        return scripted_selector_chain_actor(mask, features, kind, now)


class NeverActivateActor:
    def __call__(self, mask, features, kind, now):
        if kind == "inactive":
            return named(mask, kind, "WAIT")
        if kind == "hold_interior":
            return named(mask, kind, "MAINTAIN")
        if kind == "hold_deadline":
            return named(mask, kind, "RELEASE")
        return None


class ActivateThenReleaseActor:
    """ACTIVATE at the first permitted tick, then RELEASE at the first HOLD
    decision (early release through the named interior action)."""

    def __init__(self):
        self.activated = False

    def __call__(self, mask, features, kind, now):
        if kind == "inactive":
            if self.activated or not mask[named(mask, kind, "ACTIVATE")]:
                return named(mask, kind, "WAIT")
            self.activated = True
            return named(mask, kind, "ACTIVATE")
        if kind == "hold_interior":
            return named(mask, kind, "RELEASE")
        if kind == "hold_deadline":
            return named(mask, kind, "RELEASE")
        return None


class PhaseBlindActor:
    """The PP-11 defect as an actor: picks index 1 whenever available,
    ignoring the phase-dependent meaning (RELEASE during HOLD)."""

    def __call__(self, mask, features, kind, now):
        return 1 if mask[1] else 0


def run_episode(tmp_path, actor=None, case_id="sel", budget=None):
    episode = PpoEpisode(job=selector_job(case_id),
                         registration=registration_dict(),
                         member_ids=COHORT_A, required_lanes=LANES,
                         policy_mode="selector", actor=actor,
                         output=tmp_path / case_id, budget=budget)
    return episode.run()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---- the populated complete chain -------------------------------------------

def test_populated_chain_offers_selects_commits_holds_extends_releases(
        tmp_path, populated_traci):
    summary, rollout = run_episode(tmp_path, actor=ScriptedChainActor(),
                                   case_id="chain")
    out = tmp_path / "chain"
    log = load_json(out / "decision_log.json")

    # full horizon, exact accounting: 240 x 0.5 s physics samples on the
    # exact grid (the ExactClock raises on any drift), one reward row per
    # control second, decisions 1:1 with decision receipts
    assert summary["duration_s"] == HORIZON
    assert summary["mechanics_only"] is True
    assert summary["policy_version"] == C.POLICY_VERSION
    assert populated_traci.steps == HORIZON * 2
    assert populated_traci.world.time == pytest.approx(float(HORIZON))
    reward_rows = (out / "reward_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(reward_rows) == HORIZON
    assert len(rollout) == len(log)
    assert [r["time_s"] for r in rollout] == [d["time_s"] for d in log]
    assert summary["wall_owned_control_steps"] > 0

    # candidates report BEFORE the activation window: pre-window inactive
    # decisions are WAIT with the mask closed; once both near-cohort members
    # report, the offer is already bound while activation stays unavailable
    pre = [r for r in rollout if r["time_s"] < WINDOW[0]]
    assert pre and all(r["phase"] == "inactive" for r in pre)
    for row in pre:
        assert row["action"] == C.A_WAIT
        assert row["mask"][C.A_ACTIVATE] is False
    pre_offers = [r["offer"] for r in pre if "offer" in r]
    assert pre_offers, "no offer was bound before the activation window"
    assert all(set(o["member_ids"]) <= set(CAV_IDS) for o in pre_offers)

    # nontrivial deterministic selection at the window tick: both cohorts
    # report by then, four pairs are assessed, and the registered ranking
    # (remaining_total_s, member_ids) binds ONE coherent pair (one member
    # per registered lane)
    at_offer = [r["offer"] for r in rollout if r["time_s"] == WINDOW[0]][0]
    assert at_offer["assessed_pairs"] == 4
    assert set(at_offer["member_ids"]) <= set(CAV_IDS)
    assert len(at_offer["member_ids"]) == 2
    assert at_offer["remaining_total_s"] is not None

    # candidate-specific observations match the offered cohort BEFORE
    # activation (fails under the old zero-candidate-feature behavior).
    # The env actor boundary carries the CURRENT-tick FEATURE_NAMES vector.
    at = [r for r in rollout if r["time_s"] == WINDOW[0]][0]

    def feature_value(row, name):
        return row["features"][C.FEATURE_NAMES.index(name)]

    cand_mean = feature_value(at, "cand_mean_speed_mps")
    assert cand_mean > 0.0
    assert cand_mean == pytest.approx(at["offer"]["target_speed_mps"] + 2.0,
                                      abs=1e-6)
    assert feature_value(at, "feasible_activate_now") == 1.0

    # ACTIVATE commits exactly the actor-approved proposal
    activation = [d for d in log if d.get("executed_action") == "activate"]
    assert len(activation) == 1
    committed = activation[0]["offer"]
    assert committed["committed"] is True
    assert set(committed["member_ids"]) == set(at_offer["member_ids"])
    assert at["offer"]["proposal_fingerprint"] == committed["proposal_fingerprint"]

    # membership frozen for the whole episode: every post-activation rollout
    # row carries the SAME committed offer identity
    post = [r for r in rollout if r["time_s"] > WINDOW[0]]
    assert post
    for row in post:
        assert row["offer"]["committed"] is True
        assert row["offer"]["member_ids"] == committed["member_ids"]

    # assembly completes, interior MAINTAIN never releases/extends, EXTEND
    # occurs exactly once at the H20 deadline, H30 is delivered, then RELEASE
    extends = [d for d in log if d.get("executed_action") == "extend"]
    releases = [d for d in log if d.get("executed_action") == "release"]
    assert len(extends) == 1 and len(releases) == 1
    assert extends[0]["decision_kind"] == "hold_deadline"
    assert extends[0]["admission"] == dict(requested_hold_s=30.0,
                                           admitted_hold_s=30.0)
    interior = [d for d in log if d.get("decision_kind") == "hold_interior"]
    assert interior and all(d["executed_action"] == "maintain" for d in interior)
    assert all(d["admitted_hold_s"] == 20.0 for d in interior
               if d["time_s"] < extends[0]["time_s"])
    ready_at = extends[0]["ready_at_s"]
    assert ready_at is not None
    assert releases[0]["time_s"] >= ready_at + 30.0 - 1e-9   # full H30 delivered
    assert releases[0]["decision_kind"] == "hold_deadline"

    # handback to identical damping: the terminal phase is the coast-terminal
    # release, and no second activation ever occurs
    assert summary["phase_seconds"].get("release", 0) >= 1
    assert summary["totals"]["collision_involvements"] == 0
    assert summary["totals"]["starting_teleports"] == 0
    assert all(d.get("executed_action") != "activate"
               for d in log if d["time_s"] > activation[0]["time_s"])


def test_populated_chain_reproduces_with_default_comparator(tmp_path,
                                                            populated_traci):
    """actor=None selects the same named scripted comparator; the chain is
    identical (the default is not a separate silent path)."""
    summary, rollout = run_episode(tmp_path, actor=None, case_id="default")
    log = load_json(tmp_path / "default" / "decision_log.json")
    assert any(d.get("executed_action") == "activate" for d in log)
    assert any(d.get("executed_action") == "extend" for d in log)
    assert summary["wall_owned_control_steps"] > 0


# ---- WAIT then a later fresh search yields a new offer ----------------------

def test_wait_then_later_search_yields_a_new_offer(tmp_path, populated_traci):
    summary, rollout = run_episode(
        tmp_path, actor=ActivateAtThenChainActor(activate_at=25.0),
        case_id="wait")
    log = load_json(tmp_path / "wait" / "decision_log.json")
    rows = {r["time_s"]: r for r in rollout}
    wait_row, act_row = rows[20.0], rows[25.0]
    assert wait_row["action"] == C.A_WAIT
    assert act_row["action"] == C.A_ACTIVATE
    # both ticks bound their own offer (fresh search from then-current
    # reports: distinct decision times and evidence identities)
    assert wait_row["offer"]["decision_time_s"] == 20.0
    assert act_row["offer"]["decision_time_s"] == 25.0
    assert (wait_row["offer"]["evidence_fingerprint"]
            != act_row["offer"]["evidence_fingerprint"])
    committed = [d for d in log if d.get("executed_action") == "activate"][0]
    assert set(committed["offer"]["member_ids"]) == \
        set(act_row["offer"]["member_ids"])


# ---- external actor choices are respected -----------------------------------

def test_external_actor_wait_is_respected_no_activation(tmp_path,
                                                        populated_traci):
    summary, rollout = run_episode(tmp_path, actor=NeverActivateActor(),
                                   case_id="waitall")
    log = load_json(tmp_path / "waitall" / "decision_log.json")
    assert not any(d.get("executed_action") == "activate" for d in log)
    assert summary["wall_owned_control_steps"] == 0
    inactive = [r for r in rollout if r["phase"] == "inactive"]
    assert inactive and all(r["action"] == C.A_WAIT for r in inactive)


def test_external_actor_early_release_is_respected(tmp_path, populated_traci):
    summary, rollout = run_episode(tmp_path, actor=ActivateThenReleaseActor(),
                                   case_id="early")
    log = load_json(tmp_path / "early" / "decision_log.json")
    activation = [d for d in log if d.get("executed_action") == "activate"]
    assert len(activation) == 1
    interior_releases = [d for d in log
                         if d.get("decision_kind") == "hold_interior"
                         and d.get("executed_action") == "release"]
    assert len(interior_releases) == 1          # the actor's RELEASE was used
    assert not any(d.get("executed_action") == "extend" for d in log)
    ready_at = interior_releases[0]["ready_at_s"]
    assert ready_at is not None
    assert interior_releases[0]["time_s"] - ready_at < 20.0   # early, not H20
    assert summary["phase_seconds"].get("release", 0) >= 1


def test_phase_blind_driver_releases_at_first_hold_defect_documented(
        tmp_path, populated_traci):
    """The repaired driver must differ from the PP-11 defect: a phase-blind
    index-1 actor releases at the FIRST hold decision — no EXTEND, no H30.
    The populated chain test asserts the opposite, so the suite fails under
    the old driver behavior."""
    summary, rollout = run_episode(tmp_path, actor=PhaseBlindActor(),
                                   case_id="blind")
    log = load_json(tmp_path / "blind" / "decision_log.json")
    assert any(d.get("executed_action") == "activate" for d in log)
    assert not any(d.get("executed_action") == "extend" for d in log)
    first_hold = [d for d in log if d.get("decision_kind") == "hold_interior"]
    assert first_hold[0]["executed_action"] == "release"


# ---- budget rejection at the real env boundary ------------------------------

def test_budget_rejection_prevents_mocked_launcher(tmp_path, populated_traci):
    from traffic_shepherd.budget_guard import BudgetExceeded
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(dict(
        schema="campaign_budget_ledger/1", campaigns={},
        reservations=[dict(id="x:1", campaign="selector_fixture",
                           category="training", amount_s=134400.0,
                           target_dir="x", reserved_at="t",
                           status="complete")])), encoding="utf-8")
    budget = dict(campaign="selector_fixture", category="training",
                  amount_s=120.0, ledger_path=str(ledger),
                  ceilings=dict(categories={
                      "training": {"max_allocations": 1, "max_seconds": 120.0}},
                      total_seconds=120.0),
                  target_dir=str(tmp_path / "budgeted"))
    with pytest.raises(BudgetExceeded):
        run_episode(tmp_path, actor=ScriptedChainActor(),
                    case_id="budgeted", budget=budget)
    assert populated_traci.steps == 0            # launcher never ran
    assert not (tmp_path / "budgeted" / "summary.json").exists()
    manifest = load_json(tmp_path / "budgeted" / "manifest.json")
    assert manifest["status"] == "failed"
    assert "exceeds" in manifest["error"]        # the rejection, recorded
