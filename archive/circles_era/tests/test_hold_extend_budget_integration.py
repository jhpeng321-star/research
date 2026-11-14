"""Runner-level integration of the campaign budget guard at the launch boundary.

The real hold-extend runners (mechanics fixture and corridor study) are
driven against temporary campaign ledgers with the REAL launcher spied:
a rejected reservation, a blocking STOP receipt or a missing prerequisite
must raise out of the runner before the simulator start is ever invoked.
Unit coverage of the guard alone (tests/test_hold_extend_admission.py)
does not establish this.
"""
import json
from pathlib import Path

import pytest

from traffic_shepherd import hold_extend_mechanics, hold_extend_study
from traffic_shepherd import paired_experiments as paired
from traffic_shepherd.budget_guard import (BlockedByStop, BudgetExceeded,
                                           BudgetGuard, PrerequisiteMissing)
from traffic_shepherd.dissolution_study import REFERENCES
from traffic_shepherd.hold_extend_mechanics import (CAMPAIGN, CEILINGS,
                                                    PREREQUISITE_KIND)

EXPECTED_HOLD_START_S = hold_extend_study.EXPECTED_HOLD_START_S
EXTENSION_BRANCH_S = hold_extend_study.EXTENSION_BRANCH_S
SOURCE_JOBS = hold_extend_study.SOURCE_JOBS


def _seed_ledger(tmp_path, reservations):
    """Write a campaign ledger pre-charged with the given reservations."""
    ledger = tmp_path / "ledger.json"
    now = "2026-09-07T00:00:00+00:00"
    ledger.write_text(json.dumps(dict(
        schema="campaign_budget_ledger/1", campaigns={},
        reservations=[dict(id=f"seed:{i}", campaign=CAMPAIGN, category=category,
                           amount_s=amount_s, target_dir=str(tmp_path / f"seed{i}"),
                           reserved_at=now, status=timeline)
                      for i, (category, amount_s, timeline) in enumerate(reservations)])),
        encoding="utf-8")
    return ledger


def _spy(monkeypatch, owner, name="start_sumo"):
    calls = []
    def launcher(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("simulator must not start under a rejected reservation")
    monkeypatch.setattr(owner, name, launcher)
    return calls


# ---- mechanics runner (direct start_sumo call site) ----

def test_mechanics_runner_rejects_exhausted_campaign(tmp_path, monkeypatch):
    ledger = _seed_ledger(tmp_path, [("mechanics", 120.0, "complete")])
    monkeypatch.setattr(hold_extend_mechanics, "LEDGER_PATH", ledger)
    registered = hold_extend_mechanics.register(tmp_path / "mech")
    calls = _spy(monkeypatch, hold_extend_mechanics)
    with pytest.raises(BudgetExceeded):
        hold_extend_mechanics.run_one(registered["registration"])
    assert calls == []  # the real runner path never invoked its launcher
    stop = json.loads((tmp_path / "mech" / "STOP.json").read_text(encoding="utf-8"))
    assert stop["reason"] == "BudgetExceeded"
    # rejection never wrote to the ledger
    data = json.loads(ledger.read_text(encoding="utf-8"))
    assert len(data["reservations"]) == 1  # only the seed charge


def test_mechanics_runner_blocked_by_stop_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(hold_extend_mechanics, "LEDGER_PATH", tmp_path / "ledger.json")
    registered = hold_extend_mechanics.register(tmp_path / "mech")
    (tmp_path / "mech" / "STOP.json").write_text('{"reason": "prior defect"}',
                                                 encoding="utf-8")
    calls = _spy(monkeypatch, hold_extend_mechanics)
    with pytest.raises(BlockedByStop):
        hold_extend_mechanics.run_one(registered["registration"])
    assert calls == []


def test_mechanics_runner_registration_without_budget_is_refused(tmp_path, monkeypatch):
    """A registration that lacks the budget block cannot even be read back."""
    monkeypatch.setattr(hold_extend_mechanics, "LEDGER_PATH", tmp_path / "ledger.json")
    registered = hold_extend_mechanics.register(tmp_path / "mech")
    calls = _spy(monkeypatch, hold_extend_mechanics)
    # strip the budget block and fix the registration hash
    path = Path(registered["registration"])
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("budget")
    data.pop("registration_sha256")
    data["registration_sha256"] = hold_extend_mechanics.digest(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="exact bounded plan"):
        hold_extend_mechanics.run_one(str(path))
    assert calls == []


# ---- corridor runner (launch boundary inside paired.evaluate_arm) ----

def _corridor_registration(tmp_path, ledger, prerequisite=True):
    data = dict(
        schema=hold_extend_study.SCHEMA,
        registered_at_utc="2026-09-07T00:00:00+00:00",
        identity=paired.current_identity(),
        jobs=hold_extend_study.jobs(),
        extension_s=10.0,
        maximum_attempts=1, maximum_simulated_seconds=4800, automatic_retries=0,
        automatic_successors=False, duration_variants=False,
        new_mechanics_simulated_seconds=120.0,
        interaction_diagnostics=True, policy_version=hold_extend_study.VERSION,
        assignment="registered_experimental_assignment_not_trained_benefit_gate",
        budget=dict(campaign=CAMPAIGN, category="corridor", amount_s=2400.0,
                    ledger_path=str(ledger), ceilings=CEILINGS,
                    prerequisite=[PREREQUISITE_KIND, str(tmp_path / "mech_registration.json")]
                    if prerequisite else None),
        protocol_path=hold_extend_study.PROTOCOL,
        protocol_sha256=paired.sha(hold_extend_study.ROOT / hold_extend_study.PROTOCOL))
    original = json.loads(hold_extend_study.SOURCE_REGISTRATION.read_text(encoding="utf-8"))
    data["criteria"] = original["criteria"]
    data["communication"] = original["communication"]
    references, sources = {}, {}
    for job in data["jobs"]:
        seed = job["seed"]
        # The frozen references' manifest proofs, computed directly. The
        # registration-time reference_proof() identity bridge is intentionally
        # NOT used here: it refuses reuse across changed baseline execution
        # code (paired_experiments.py now carries the budget gate), which is
        # registration governance, not the launch boundary under test.
        reference_dir = REFERENCES / "jobs" / job["reference_job_id"]
        references[job["id"]] = dict(
            path=str(reference_dir),
            manifest_sha256=paired.sha(reference_dir / "manifest.json"))
        block, job_id = SOURCE_JOBS[seed]
        source_root = Path("runs/active") / block
        sources[job["id"]] = dict(
            path=str(source_root / "jobs" / job_id),
            manifest_sha256=paired.sha(source_root / "jobs" / job_id / "manifest.json"),
            registration_path=str(source_root / "registration.json"),
            registration_file_sha256=paired.sha(source_root / "registration.json"),
            measured_hold_start_s=EXPECTED_HOLD_START_S[seed],
            extension_branch_s=EXTENSION_BRANCH_S[seed])
    data["references"] = references
    data["sources"] = sources
    data["registration_sha256"] = paired.digest(data)
    out = tmp_path / "corridor"
    snapshot = out / "source_snapshot"
    snapshot.mkdir(parents=True)
    import shutil
    from traffic_shepherd.config import CONFIG_PATH
    for name in data["identity"]["active_source_sha256"]:
        shutil.copy2(paired.ROOT / "traffic_shepherd" / name, snapshot / name)
    shutil.copy2(CONFIG_PATH, snapshot / "adapter.json")
    (out / "registration.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    return out / "registration.json"


def test_corridor_runner_rejects_exhausted_campaign(tmp_path, monkeypatch):
    ledger = _seed_ledger(tmp_path, [("corridor", 2400.0, "complete"),
                                     ("corridor", 2400.0, "complete")])
    registration = _corridor_registration(tmp_path, ledger)
    job_id = json.loads(registration.read_text(encoding="utf-8"))["jobs"][0]["id"]
    calls = _spy(monkeypatch, paired)
    with pytest.raises(BudgetExceeded):
        hold_extend_study.run_one(registration, job_id)
    assert calls == []  # paired.evaluate_arm never reached start_sumo
    stop = json.loads((tmp_path / "corridor" / "STOP.json").read_text(encoding="utf-8"))
    assert "exceeds" in stop["reason"]
    data = json.loads(ledger.read_text(encoding="utf-8"))
    assert len(data["reservations"]) == 2  # rejection wrote nothing


def test_corridor_runner_requires_recorded_prerequisite(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(dict(schema="campaign_budget_ledger/1",
                                      campaigns={}, reservations=[])), encoding="utf-8")
    registration = _corridor_registration(tmp_path, ledger, prerequisite=True)
    job_id = json.loads(registration.read_text(encoding="utf-8"))["jobs"][1]["id"]
    calls = _spy(monkeypatch, paired)
    with pytest.raises(PrerequisiteMissing):
        hold_extend_study.run_one(registration, job_id)
    assert calls == []
    assert json.loads(ledger.read_text(encoding="utf-8"))["reservations"] == []


def test_corridor_runner_still_blocked_when_budget_has_room(tmp_path, monkeypatch):
    """Prerequisite satisfied but category allocation count exhausted."""
    ledger = _seed_ledger(tmp_path, [("corridor", 2400.0, "complete"),
                                     ("corridor", 2400.0, "failed")])
    registration = _corridor_registration(tmp_path, ledger)
    # the seeded charges stand in for the recorded mechanics prerequisite
    guard = BudgetGuard(ledger)
    guard.record_prerequisite(CAMPAIGN, PREREQUISITE_KIND,
                              tmp_path / "mech_registration.json")
    job_id = json.loads(registration.read_text(encoding="utf-8"))["jobs"][0]["id"]
    calls = _spy(monkeypatch, paired)
    with pytest.raises(BudgetExceeded):
        hold_extend_study.run_one(registration, job_id)
    assert calls == []
