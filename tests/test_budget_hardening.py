"""Budget-accounting hardening tests (mocked launchers only).

Covers: sticky campaign-level stop after a failed mechanics allocation,
reclassified/moved charges never restoring capacity, author-tagged STOP
resolutions, actual-progress recording distinct from charged amounts, and
crash/concurrency safety on the hardened guard.
"""
import json
import threading

import pytest

from traffic_shepherd.budget_guard import (BlockedByStop, BudgetExceeded,
                                           BudgetGuard)

CEILINGS = dict(
    categories={"mechanics": {"max_allocations": 2, "max_seconds": 240.0},
                "training": {"max_allocations": 32, "max_seconds": 76800.0}},
    total_seconds=134640.0)


def boom():
    raise RuntimeError("simulator failure")


def test_first_failed_mechanics_allocation_stops_the_whole_campaign(tmp_path):
    """The registered stopping rule, enforced by the guard: one launcher
    failure with campaign_stop_on_failure stops EVERY category, including
    fresh version directories."""
    guard = BudgetGuard(tmp_path / "ledger.json")
    with pytest.raises(RuntimeError):
        guard.reserve(campaign="ppo_pilot", category="mechanics", amount_s=120.0,
            target_dir=tmp_path / "m1", ceilings=CEILINGS, launcher=boom,
            campaign_stop_on_failure=True)
    state = guard.campaign_state("ppo_pilot")
    assert state["stopped"] is True and "launcher failure" in state["stop_reason"]
    with pytest.raises(BlockedByStop, match="stopped"):
        guard.reserve(campaign="ppo_pilot", category="mechanics", amount_s=120.0,
            target_dir=tmp_path / "m2", ceilings=CEILINGS, launcher=lambda: None,
            campaign_stop_on_failure=True)
    with pytest.raises(BlockedByStop, match="stopped"):
        guard.reserve(campaign="ppo_pilot", category="training", amount_s=2400.0,
            target_dir=tmp_path / "fresh_version_dir", ceilings=CEILINGS,
            launcher=lambda: None)


def test_campaign_stop_is_sticky_and_survives_reclassification(tmp_path):
    """Moving a charge to another ledger (tombstone) never restores
    capacity, and the sticky stop is never cleared by reclassification."""
    ledger = tmp_path / "ledger.json"
    guard = BudgetGuard(ledger)
    guard.reserve(campaign="c", category="mechanics", amount_s=120.0,
        target_dir=tmp_path / "a", ceilings=CEILINGS, launcher=lambda: None)
    guard.exclude_charge("c", "mechanics", 120.0, "c:mechanics:1",
                         "moved to diagnostics ledger (disclosed deviation)")
    data = json.loads(ledger.read_text(encoding="utf-8"))
    assert data["reservations"] == []                       # live list is empty
    assert data["excluded_charges"][0]["amount_s"] == 120.0  # but still counted
    assert guard.charged_seconds("c", "mechanics") == 120.0
    guard.stop_campaign("c", "mechanics allocation failed")
    with pytest.raises(BlockedByStop):
        guard.reserve(campaign="c", category="mechanics", amount_s=120.0,
            target_dir=tmp_path / "b", ceilings=CEILINGS, launcher=lambda: None)


def test_reclassification_cannot_free_exhausted_category_capacity(tmp_path):
    ledger = tmp_path / "ledger.json"
    guard = BudgetGuard(ledger)
    guard.reserve(campaign="c", category="training", amount_s=76800.0,
        target_dir=tmp_path / "t1", ceilings=CEILINGS, launcher=lambda: None)
    with pytest.raises(BudgetExceeded):
        guard.reserve(campaign="c", category="training", amount_s=2400.0,
            target_dir=tmp_path / "t2", ceilings=CEILINGS, launcher=lambda: None)
    guard.exclude_charge("c", "training", 76800.0, "c:training:1", "moved away")
    with pytest.raises(BudgetExceeded):     # tombstoned charge still counts
        guard.reserve(campaign="c", category="training", amount_s=2400.0,
            target_dir=tmp_path / "t3", ceilings=CEILINGS, launcher=lambda: None)


def test_agent_generated_resolution_cannot_clear_stop_owner_can(tmp_path):
    target = tmp_path / "v1"
    target.mkdir()
    (target / "STOP.json").write_text('{"reason": "mechanics failure"}',
                                      encoding="utf-8")
    guard = BudgetGuard(tmp_path / "ledger.json")
    # agent-generated resolution (no author == owner): still blocked
    (target / "verification_resolution.json").write_text(
        json.dumps(dict(author="agent", note="self-cleared")), encoding="utf-8")
    with pytest.raises(BlockedByStop, match="agent-generated"):
        guard.reserve(campaign="c", category="training", amount_s=2400.0,
            target_dir=target, ceilings=CEILINGS, launcher=lambda: None)
    # an owner-authored resolution clears the per-directory block
    (target / "verification_resolution.json").write_text(
        json.dumps(dict(author="owner", note="owner authorization record")),
        encoding="utf-8")
    _res, outcome = guard.reserve(campaign="c", category="training",
        amount_s=2400.0, target_dir=target, ceilings=CEILINGS, launcher=lambda: "ok")
    assert outcome == "ok"


def test_mark_progress_records_actual_time_separately_from_charge(tmp_path):
    guard = BudgetGuard(tmp_path / "ledger.json")
    res, _ = guard.reserve(campaign="c", category="mechanics", amount_s=120.0,
        target_dir=tmp_path / "m", ceilings=CEILINGS, launcher=lambda: "ran")
    guard.mark_progress(res["id"], completed_simulated_s=3.0)
    data = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    row = data["reservations"][0]
    assert row["amount_s"] == 120.0                  # charged unchanged
    assert row["completed_simulated_s"] == 3.0       # actual progress separate
    assert guard.charged_seconds("c") == 120.0       # charge accounting unchanged


def test_failed_launcher_remains_charged_and_counts(tmp_path):
    guard = BudgetGuard(tmp_path / "ledger.json")
    with pytest.raises(RuntimeError):
        guard.reserve(campaign="c", category="training", amount_s=2400.0,
            target_dir=tmp_path / "t", ceilings=CEILINGS, launcher=boom)
    assert guard.charged_seconds("c", "training") == 2400.0


def test_concurrent_launch_attempts_cannot_overspend_hardened(tmp_path):
    guard = BudgetGuard(tmp_path / "ledger.json")
    results, errors = [], []

    def attempt(i):
        try:
            res, _ = guard.reserve(campaign="c", category="mechanics",
                amount_s=120.0, target_dir=tmp_path / f"m{i}", ceilings=CEILINGS,
                launcher=lambda: None, campaign_stop_on_failure=True)
            results.append(res)
        except (BudgetExceeded, BlockedByStop):
            errors.append("blocked")
    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) + len(errors) == 2
    assert guard.charged_seconds("c", "mechanics") <= 240.0
