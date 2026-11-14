"""Bounded atomic-ledger replace retry tests; launchers are never retried."""
import errno
import json

import pytest

from traffic_shepherd import budget_guard


def _guard(tmp_path):
    return budget_guard.BudgetGuard(tmp_path / "ledger.json")


def _permission(winerror):
    error = PermissionError(errno.EACCES, "denied")
    error.winerror = winerror
    return error


def test_transient_windows_replace_failure_retries_same_payload(monkeypatch, tmp_path):
    guard = _guard(tmp_path)
    original = budget_guard.os.replace
    calls = []
    sleeps = []

    def replace(source, target):
        calls.append((source.read_text(encoding="utf-8"), str(target)))
        if len(calls) < 3:
            raise _permission(32)
        return original(source, target)

    monkeypatch.setattr(budget_guard.os, "replace", replace)
    monkeypatch.setattr(budget_guard.time, "sleep", sleeps.append)
    payload = {"schema": budget_guard.LEDGER_SCHEMA, "campaigns": {}, "reservations": []}
    guard._save(payload)
    assert len(calls) == 3
    assert len({item[0] for item in calls}) == 1
    assert sleeps == [0.05, 0.05]
    assert json.loads(guard.ledger_path.read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_each_eligible_windows_error_is_bounded(monkeypatch, tmp_path, winerror):
    guard = _guard(tmp_path)
    calls, sleeps = [], []
    monkeypatch.setattr(budget_guard.os, "replace",
                        lambda *_args: (calls.append(1), (_ for _ in ()).throw(_permission(winerror)))[1])
    monkeypatch.setattr(budget_guard.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError):
        guard._save({"schema": budget_guard.LEDGER_SCHEMA, "campaigns": {}, "reservations": []})
    assert len(calls) == 5
    assert sleeps == [0.05, 0.05, 0.05, 0.05]


def test_unrelated_permission_error_is_not_retried(monkeypatch, tmp_path):
    guard = _guard(tmp_path)
    calls = []
    monkeypatch.setattr(budget_guard.os, "replace",
                        lambda *_args: (calls.append(1), (_ for _ in ()).throw(_permission(13)))[1])
    with pytest.raises(PermissionError):
        guard._save({"schema": budget_guard.LEDGER_SCHEMA, "campaigns": {}, "reservations": []})
    assert calls == [1]


def test_successful_first_replace_does_not_sleep(monkeypatch, tmp_path):
    guard = _guard(tmp_path)
    sleeps = []
    monkeypatch.setattr(budget_guard.time, "sleep", sleeps.append)
    guard._save({"schema": budget_guard.LEDGER_SCHEMA, "campaigns": {}, "reservations": []})
    assert sleeps == []


def test_replace_failure_cannot_invoke_launcher(monkeypatch, tmp_path):
    guard = _guard(tmp_path)
    monkeypatch.setattr(budget_guard.os, "replace",
                        lambda *_args: (_ for _ in ()).throw(_permission(5)))
    launched = []
    with pytest.raises(PermissionError):
        guard.reserve(
            campaign="cv16", category="test", amount_s=1.0,
            target_dir=tmp_path / "run", ceilings=dict(
                categories=dict(test=dict(max_allocations=1, max_seconds=1.0)),
                total_seconds=1.0), launcher=lambda: launched.append(True))
    assert launched == []
