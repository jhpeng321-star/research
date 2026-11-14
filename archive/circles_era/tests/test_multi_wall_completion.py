"""Accounting tests for the completion wrapper; no simulator is started."""
import importlib.util
import json
from pathlib import Path

import pytest
from traffic_shepherd.budget_guard import BlockedByStop, BudgetExceeded


PATH = Path(__file__).parents[1] / "output/multi_wall_probe_v1/completion.py"
SPEC = importlib.util.spec_from_file_location("multi_wall_completion_under_test", PATH)
COMPLETION = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(COMPLETION)


def _ceilings(mechanics_seconds=120, total_seconds=240):
    return {
        "categories": {
            "mechanics": {"max_allocations": 1, "max_seconds": mechanics_seconds},
            "corridor": {"max_allocations": 6, "max_seconds": 120},
        },
        "total_seconds": total_seconds,
    }


def test_real_guard_returns_reservation_and_outcome_and_records_actual(tmp_path):
    ledger = tmp_path / "ledger.json"
    target = tmp_path / "first"
    guard = COMPLETION.BudgetGuard(ledger)
    reservation, outcome = guard.reserve(
        campaign="completion-test",
        category="mechanics",
        amount_s=120,
        target_dir=target,
        ceilings=_ceilings(),
        launcher=lambda: {"passed": True, "completed_simulated_seconds": 120},
    )
    guard.mark_progress(reservation["id"], outcome["completed_simulated_seconds"])
    record = json.loads(ledger.read_text(encoding="utf-8"))["reservations"][0]
    assert reservation["amount_s"] == 120
    assert outcome["completed_simulated_seconds"] == 120
    assert record["status"] == "complete"
    assert record["completed_simulated_s"] == 120
    with pytest.raises(BudgetExceeded):
        guard.reserve(
            campaign="completion-test",
            category="mechanics",
            amount_s=120,
            target_dir=tmp_path / "retry",
            ceilings=_ceilings(),
            launcher=lambda: pytest.fail("retry launcher ran"),
        )


def test_failed_allocation_stops_campaign_and_blocks_retry(tmp_path):
    ledger = tmp_path / "ledger.json"
    guard = COMPLETION.BudgetGuard(ledger)
    with pytest.raises(RuntimeError):
        guard.reserve(
            campaign="sticky-test",
            category="mechanics",
            amount_s=120,
            target_dir=tmp_path / "failed",
            ceilings=_ceilings(),
            launcher=lambda: (_ for _ in ()).throw(RuntimeError("fixture failure")),
            campaign_stop_on_failure=True,
        )
    with pytest.raises(BlockedByStop):
        guard.reserve(
            campaign="sticky-test",
            category="mechanics",
            amount_s=120,
            target_dir=tmp_path / "retry",
            ceilings=_ceilings(),
            launcher=lambda: pytest.fail("stopped launcher ran"),
        )


def test_run_mechanics_unpacks_real_guard_tuple_and_charges_120(tmp_path, monkeypatch):
    from output.multi_wall_probe_v1 import study

    runs = tmp_path / "runs"
    (runs / "mechanics").mkdir(parents=True)
    out = tmp_path / "out"
    ledger = tmp_path / "ledger.json"
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "RUNS", runs)
    monkeypatch.setattr(study, "LEDGER", ledger)
    monkeypatch.setattr(study, "OUT", out)
    monkeypatch.setattr(COMPLETION, "REG", runs / "completion_registration.json")
    COMPLETION.REG.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(COMPLETION, "CAMPAIGN", "completion-wrapper-test")
    monkeypatch.setattr(COMPLETION, "CEILINGS", _ceilings(120, 120))
    original = {"registration_sha256": "registration"}
    monkeypatch.setattr(COMPLETION, "load", lambda: (original, {}))
    monkeypatch.setattr(COMPLETION, "verify_first", lambda: {"passed": True})
    first = runs / "mechanics" / "same_cohort_extend20"
    first.mkdir()
    (first / "manifest.json").write_text("{}", encoding="utf-8")
    (first / "summary.json").write_text("{}", encoding="utf-8")

    def fake_execute(output, _registration, mode):
        output.mkdir(parents=True)
        (output / "manifest.json").write_text("{}", encoding="utf-8")
        (output / "summary.json").write_text("{}", encoding="utf-8")
        return {"mode": mode, "passed": True, "completed_simulated_seconds": 120}

    monkeypatch.setattr(COMPLETION.mechanics, "_execute", fake_execute)
    summary = COMPLETION.run_mechanics()
    reservations = json.loads(ledger.read_text(encoding="utf-8"))["reservations"]
    assert summary["completed_simulated_seconds"] == 120
    assert reservations[0]["amount_s"] == 120
    assert reservations[0]["completed_simulated_s"] == 120
    assert (out / "MECHANICS_GATE.json").exists()
