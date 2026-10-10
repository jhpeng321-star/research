"""Admission-repair and budget-guard verification (offline, zero simulation).

The admission tests exercise the repaired hold_extend/2 evaluation on
synthetic proposals/plans (remaining_plan is stubbed, so these verify the
admission logic itself: which plan is assessed, commit/decline behavior,
state mutation, binding identity). The budget tests exercise the persistent
guard on synthetic temporary records with mocked launchers.
"""
import json
import threading
from dataclasses import asdict
from pathlib import Path

import pytest

from traffic_shepherd.formation import FormationProposal, WallMember
from traffic_shepherd.target_adjust_wall import ContinuationTargetProfile

from traffic_shepherd.budget_guard import (BudgetExceeded, BlockedByStop,
                                           BudgetGuard, PrerequisiteMissing)
from traffic_shepherd.hold_extend_wall import (HoldExtendPolicy)
from traffic_shepherd.lifecycle_wall import LifecycleProfile


def _policy_with_stub_plan(monkeypatch, original_feasible, ext_feasible,
                           ext_remaining_hold=10, original_exit_feasible=True,
                           ext_exit_feasible=True):
    """HoldExtendPolicy whose remaining_plan is stubbed per profile."""
    job = dict(arm="wall", decision_time_s=600, policy_spec=dict(
        member_ids=["a", "b"], required_lanes=["x", "y"],
        profile=asdict(LifecycleProfile()),
        target_adjust_branch=dict(branch_mode="absolute", adjust_mps=2.0,
                                  expected_hold_start_s=612.0, branch_time_s=617.0),
        hold_extension=dict(branch_mode="absolute", extension_s=10.0,
                            expected_hold_start_s=612.0, branch_time_s=632.0)))
    policy = HoldExtendPolicy(json.loads(json.dumps(job)), registration=None)
    policy.proposal = FormationProposal(
        600.0, ("x", "y"),
        (WallMember("a", "x"), WallMember("b", "y")),
        26.0, ContinuationTargetProfile())
    policy.ready_at = 612.0
    policy.taper_at = None
    policy.last_output_phase = "hold"
    policy.previous_commands = {"a": 0.0, "b": 0.0}

    def fake_remaining_plan(proposal, reports, *, now_s, previous_commands,
                            readiness_count=0, ready_at_s=None, exit_now=False):
        hold = proposal.profile.hold_s
        remaining_hold = max(0, hold - (now_s - ready_at_s))
        if remaining_hold == 0:
            feasible = bool(original_feasible and original_exit_feasible)
        else:
            feasible = bool(ext_feasible and ext_exit_feasible)
        return dict(
            feasible=feasible,
            reason="stub", remaining_hold_s=remaining_hold,
            remaining_exit_s=1 if (original_exit_feasible if remaining_hold == 0
                                   else ext_exit_feasible) else 0,
            remaining_total_s=remaining_hold + 1,
            minimum_map_margin_m=300.0, minimum_traffic_margin_m=33.4)
    monkeypatch.setattr("traffic_shepherd.hold_extend_wall.remaining_plan",
                        fake_remaining_plan)
    return policy


def _load_v7_like():
    from traffic_shepherd.hold_extend_wall import HoldExtendPolicy
    return HoldExtendPolicy


# ---- corrected physical signs (authoritative summaries, exact) ----

def test_hold_extend_4107_correct_signs_and_guardrails():
    t = json.loads(Path("D:/research-dense/runs/active/hold_extend_v1/jobs/"
                        "s4107_m2700_p20_t600_hold26_ext10_wall/summary.json").read_text())
    r = json.loads(Path("D:/research-dense/runs/active/observer_replay_v1/jobs/"
                        "s4107_m2700_p20_damping/summary.json").read_text())
    pop = r["population"]
    diff = dict(
        service=(t["totals"]["requested_not_arrived_vehicle_seconds"]
                 - r["totals"]["requested_not_arrived_vehicle_seconds"]) / pop,
        braking=(t["totals"]["braking_vehicle_seconds"]
                 - r["totals"]["braking_vehicle_seconds"]) / pop,
        effort=(t["totals"]["acceleration_energy"]
                - r["totals"]["acceleration_energy"]) / pop,
        ttc=(t["totals"]["ttc_below_3_vehicle_seconds"]
             - r["totals"]["ttc_below_3_vehicle_seconds"]) / pop,
        wait=(t["source_wait_vehicle_seconds"]
              - r["source_wait_vehicle_seconds"]) / pop)
    assert diff["service"] == pytest.approx(-8.488693, abs=2e-6)
    assert diff["braking"] == pytest.approx(-0.063958, abs=2e-6)
    assert diff["effort"] == pytest.approx(-4.214348, abs=2e-6)
    assert diff["ttc"] == pytest.approx(0.020141, abs=2e-6)
    assert diff["wait"] == pytest.approx(-0.154417, abs=2e-6)
    # Exact guardrail failures: TTC barely above allowance, plus collisions.
    allowances = {"service": 0.5, "braking": 0.005, "effort": 0.5,
                  "ttc": 0.02, "wait": 0.1}
    failed = [m for m, d in diff.items() if d > allowances[m]]
    assert failed == ["ttc"]
    assert t["totals"]["collision_involvements"] == 2
    assert r["totals"]["collision_involvements"] == 4
    # delivered H26 in 4107
    assert t["phase_seconds"]["hold"] == 26
    assert "taper" not in t["phase_seconds"] and "abort" in t["phase_seconds"]


def test_hold_extend_4108_correct_signs_and_wait():
    t = json.loads(Path("D:/research-dense/runs/active/hold_extend_v2/jobs/"
                        "s4108_m2700_p20_t600_hold26_ext10_wall/summary.json").read_text())
    r = json.loads(Path("D:/research-dense/runs/active/observer_replay_v1/jobs/"
                        "s4108_m2700_p20_damping/summary.json").read_text())
    pop = r["population"]
    assert (t["totals"]["requested_not_arrived_vehicle_seconds"]
            - r["totals"]["requested_not_arrived_vehicle_seconds"]) / pop \
        == pytest.approx(13.258990, abs=2e-6)
    assert (t["source_wait_vehicle_seconds"] - r["source_wait_vehicle_seconds"]) / pop \
        == pytest.approx(0.005738, abs=2e-6)
    assert t["phase_seconds"]["hold"] == 30  # full delivery, release at 649
    # vs raise-H20: inert (all near zero)
    k = json.loads(Path("D:/research-dense/runs/active/cohort_eligibility_v1/jobs/"
                        "s4108_m2700_p20_t600_cohort_v1_wall/summary.json").read_text())
    assert (t["totals"]["requested_not_arrived_vehicle_seconds"]
            - k["totals"]["requested_not_arrived_vehicle_seconds"]) / 1307 \
        == pytest.approx(-0.0099, abs=1e-3)


# ---- admission repair ----

def test_admission_checks_extended_plan_not_original(monkeypatch):
    """Original H20 exit feasible but proposed extension infeasible -> decline."""
    policy = _policy_with_stub_plan(monkeypatch, original_feasible=True,
                                    ext_feasible=False)
    policy._request_extension(632.0, reports=())
    assert policy.ext_receipt["requested"] is True
    assert policy.ext_receipt["applied"] is False
    assert "proposed_extension_infeasible" in policy.ext_receipt["declined_reason"]
    # assessments saved separately, bound to the same report snapshot
    a = policy.ext_receipt["assessments"]
    assert a["original_plan"]["remaining_hold_s"] == 0
    assert a["extension_plan"]["remaining_hold_s"] == 10
    assert a["selected"] == "decline"
    assert a["bound_to"]["now_s"] == 632.0 and a["bound_to"]["elapsed_hold_s"] == 20.0
    # no state mutation on decline
    assert policy.profile.hold_s == 20.0
    assert policy.proposal.profile.hold_s == 20.0


def test_admission_commits_only_after_extended_plan_passes(monkeypatch):
    policy = _policy_with_stub_plan(monkeypatch, original_feasible=True,
                                    ext_feasible=True)
    policy._request_extension(632.0, reports=())
    assert policy.ext_receipt["applied"] is True
    assert policy.profile.hold_s == 30.0
    assert policy.proposal.profile is policy.profile
    assert policy.ext_receipt["assessments"]["selected"] == "extend"


def test_admission_binds_exact_proposal_and_report_snapshot(monkeypatch):
    from traffic_shepherd.paired_experiments import digest
    policy = _policy_with_stub_plan(monkeypatch, original_feasible=True,
                                    ext_feasible=True)
    fingerprint_before = policy.proposal.fingerprint
    policy._request_extension(632.0, reports=())
    bound = policy.ext_receipt["assessments"]["bound_to"]
    assert bound["proposal_fingerprint"] == fingerprint_before
    assert bound["evidence_digest"] == digest([])


def test_admission_rejects_infeasible_extended_exit(monkeypatch):
    """The extended assessment covers the WHOLE remaining commitment: a
    feasible extended hold with an infeasible remaining exit still declines."""
    policy = _policy_with_stub_plan(monkeypatch, original_feasible=True,
                                    ext_feasible=True, ext_exit_feasible=False)
    policy._request_extension(632.0, reports=())
    assert policy.ext_receipt["applied"] is False
    assert "proposed_extension_infeasible" in policy.ext_receipt["declined_reason"]
    assert policy.profile.hold_s == 20.0  # original schedule proceeds unchanged


def test_assessment_receipts_identify_both_commitments(monkeypatch):
    policy = _policy_with_stub_plan(monkeypatch, original_feasible=True,
                                    ext_feasible=True)
    adjust_receipt_before = dict(policy.receipt)
    policy._request_extension(632.0, reports=())
    a = policy.ext_receipt["assessments"]
    assert a["original_plan"]["assessed_hold_s"] == 20.0
    assert a["extension_plan"]["assessed_hold_s"] == 30.0
    assert a["original_plan"]["remaining_hold_s"] == 0.0
    assert a["extension_plan"]["remaining_hold_s"] == 10.0
    # acceptance committed exactly the assessed extended proposal
    assert policy.proposal.profile.hold_s == 30.0
    # the target-adjust receipt was never touched by the extension path
    assert policy.receipt == adjust_receipt_before


# ---- budget guard ----

CEILINGS = dict(
    categories={"mechanics": {"max_allocations": 1, "max_seconds": 120.0},
                "corridor": {"max_allocations": 2, "max_seconds": 4800.0}},
    total_seconds=4920.0)


def test_budget_reserve_complete_flow(tmp_path):
    guard = BudgetGuard(tmp_path / "ledger.json")
    calls = []
    res, outcome = guard.reserve(campaign="hold_extend", category="mechanics",
        amount_s=120.0, target_dir=tmp_path / "m1", ceilings=CEILINGS,
        launcher=lambda: calls.append(1) or "ok")
    assert outcome == "ok" and calls == [1]
    data = json.loads((tmp_path / "ledger.json").read_text())
    assert data["reservations"][0]["status"] == "complete"
    assert data["reservations"][0]["amount_s"] == 120.0


def test_budget_category_and_total_ceilings_block(tmp_path):
    ledger = tmp_path / "ledger.json"
    guard = BudgetGuard(ledger)
    for i in range(2):
        guard.reserve(campaign="c", category="corridor", amount_s=2400.0,
            target_dir=tmp_path / f"c{i}", ceilings=CEILINGS, launcher=lambda: None)
    with pytest.raises(BudgetExceeded):
        guard.reserve(campaign="c", category="corridor", amount_s=1.0,
            target_dir=tmp_path / "c9", ceilings=CEILINGS, launcher=lambda: None)
    # total ceiling: mechanics 120 + corridor 4800 = exactly 4920 -> allowed
    res, _ = guard.reserve(campaign="c", category="mechanics", amount_s=120.0,
        target_dir=tmp_path / "m9", ceilings=CEILINGS, launcher=lambda: None)
    assert res["amount_s"] == 120.0
    # one more second beyond 4920 is blocked
    with pytest.raises(BudgetExceeded):
        guard.reserve(campaign="c", category="mechanics", amount_s=1.0,
            target_dir=tmp_path / "m10", ceilings=CEILINGS, launcher=lambda: None)


def test_budget_survives_new_version_directory(tmp_path):
    """A versioned re-registration must not reset consumed budget."""
    guard = BudgetGuard(tmp_path / "ledger.json")
    guard.reserve(campaign="hold_extend", category="mechanics", amount_s=120.0,
        target_dir=tmp_path / "v1", ceilings=CEILINGS, launcher=lambda: None)
    fresh_version = tmp_path / "v13_fresh_registration"
    with pytest.raises(BudgetExceeded):
        guard.reserve(campaign="hold_extend", category="mechanics", amount_s=120.0,
            target_dir=fresh_version, ceilings=CEILINGS, launcher=lambda: None)


def test_budget_failed_reservation_stays_charged(tmp_path):
    guard = BudgetGuard(tmp_path / "ledger.json")

    def boom():
        raise RuntimeError("simulated crash after reservation")
    with pytest.raises(RuntimeError):
        guard.reserve(campaign="c", category="mechanics", amount_s=120.0,
            target_dir=tmp_path / "m1", ceilings=CEILINGS, launcher=boom)
    data = json.loads((tmp_path / "ledger.json").read_text())
    assert data["reservations"][0]["status"] == "failed"
    # crashed reservation still consumes the category ceiling
    with pytest.raises(BudgetExceeded):
        guard.reserve(campaign="c", category="mechanics", amount_s=120.0,
            target_dir=tmp_path / "m2", ceilings=CEILINGS, launcher=lambda: None)


def test_budget_stop_receipt_blocks_corridor(tmp_path):
    guard = BudgetGuard(tmp_path / "ledger.json")
    target = tmp_path / "v1"
    target.mkdir()
    (target / "STOP.json").write_text('{"reason": "verifier defect"}')
    with pytest.raises(BlockedByStop):
        guard.reserve(campaign="c", category="corridor", amount_s=2400.0,
            target_dir=target, ceilings=CEILINGS, launcher=lambda: None)
    # an agent-generated resolution cannot clear the block (hardened contract)
    (target / "verification_resolution.json").write_text('{"ok": true, "author": "agent"}')
    with pytest.raises(BlockedByStop, match="agent-generated"):
        guard.reserve(campaign="c", category="corridor", amount_s=2400.0,
            target_dir=target, ceilings=CEILINGS, launcher=lambda: None)
    # only an owner-authored resolution clears the per-directory block
    (target / "verification_resolution.json").write_text('{"ok": true, "author": "owner"}')
    res, _ = guard.reserve(campaign="c", category="corridor", amount_s=2400.0,
        target_dir=target, ceilings=CEILINGS, launcher=lambda: None)
    data = json.loads((tmp_path / "ledger.json").read_text())
    assert data["reservations"][-1]["status"] == "complete"


def test_budget_requires_prerequisite_before_corridor(tmp_path):
    guard = BudgetGuard(tmp_path / "ledger.json")
    guard.reserve(campaign="c", category="mechanics", amount_s=120.0,
        target_dir=tmp_path / "m1", ceilings=CEILINGS, launcher=lambda: None)
    with pytest.raises((PrerequisiteMissing, BudgetExceeded)):
        guard.reserve(campaign="c", category="corridor", amount_s=2400.0,
            target_dir=tmp_path / "c1", ceilings=CEILINGS, launcher=lambda: None,
            prerequisite=("mechanics_registration", str(tmp_path / "mech.json")))
    # recorded-and-passed prerequisite admits the corridor reservation
    data = json.loads((tmp_path / "ledger.json").read_text())
    data.setdefault("prerequisites", {}).setdefault("c", {})[
        "mechanics_registration"] = dict(path=str(tmp_path / "mech.json"),
                                         passed=True)
    (tmp_path / "ledger.json").write_text(json.dumps(data))
    res, _ = guard.reserve(campaign="c", category="corridor", amount_s=2400.0,
        target_dir=tmp_path / "c1", ceilings=CEILINGS, launcher=lambda: None,
        prerequisite=("mechanics_registration", str(tmp_path / "mech.json")))
    data = json.loads((tmp_path / "ledger.json").read_text())
    assert data["reservations"][-1]["status"] == "complete"


def test_budget_parallel_launch_attempts_do_not_overspend(tmp_path):
    """Two threads race for the single mechanics allocation: exactly one wins."""
    guard = BudgetGuard(tmp_path / "ledger.json")
    results, errors = [], []

    def attempt(i):
        try:
            res, _ = guard.reserve(campaign="c", category="mechanics",
                amount_s=120.0, target_dir=tmp_path / f"m{i}", ceilings=CEILINGS,
                launcher=lambda: None)
            results.append(res)
        except BudgetExceeded:
            errors.append("exceeded")
    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 1 and len(errors) == 1
