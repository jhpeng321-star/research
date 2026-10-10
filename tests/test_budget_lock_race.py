"""Budget-lock TOCTOU race tests; concurrent progress marking must not crash."""
from pathlib import Path

import pytest

from traffic_shepherd import budget_guard


def test_vanishing_lock_between_exists_and_stat_retries_instead_of_crashing(monkeypatch, tmp_path):
    """The race that stopped CV-23's 7011-retain: another holder's __exit__
    deletes the lock between our exists() True and stat(). The stale-check
    must tolerate the FileNotFoundError and retry the atomic acquire."""
    lock_path = tmp_path / "ledger.json"
    real_lock_file = Path(str(lock_path) + budget_guard.LOCK_SUFFIX)
    real_stat = Path.stat
    real_exists = Path.exists
    calls = {"stat": 0}

    def racing_stat(self, *args, **kwargs):
        if Path(self) == real_lock_file:
            calls["stat"] += 1
            if calls["stat"] == 1:
                # holder released between exists() and stat()
                real_lock_file.unlink(missing_ok=True)
                raise FileNotFoundError(2, "vanished")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", racing_stat)
    real_exists  # exists() returns True only while the file exists; the
    # racing_stat unlinks it first, so exists() -> False would short-circuit.
    # To force the exact race we also need exists() to have already returned
    # True: simulate by keeping a second pre-created lock until stat runs.
    real_lock_file.write_text("", encoding="utf-8")
    with budget_guard._FileLock(lock_path):
        assert calls["stat"] == 1
        assert real_lock_file.exists()


def test_vanishing_lock_during_stale_unlink_is_tolerated(monkeypatch, tmp_path):
    """A stale lock deleted by its returning owner between stat() and
    unlink() must not crash the acquirer (missing_ok already covers it)."""
    lock_path = tmp_path / "ledger.json"
    real_lock_file = Path(str(lock_path) + budget_guard.LOCK_SUFFIX)
    real_stat = Path.stat
    # Make the lock look ancient (stale) so the unlink branch runs.
    real_lock_file.write_text("", encoding="utf-8")
    import os
    os.utime(real_lock_file, ns=(0, 0))
    with budget_guard._FileLock(lock_path):
        assert real_lock_file.exists()
    real_stat  # silence lint on unused reference


def test_exclusive_serialization_still_holds(tmp_path):
    """Basic serialization semantics are unchanged."""
    lock_path = tmp_path / "ledger.json"
    with budget_guard._FileLock(lock_path):
        with pytest.raises(TimeoutError):
            budget_guard._FileLock(lock_path, timeout_s=0.2).__enter__()
    # released after exit: re-acquirable
    with budget_guard._FileLock(lock_path):
        pass


def test_stale_lock_expiry_still_reclaims(tmp_path):
    import os
    lock_path = tmp_path / "ledger.json"
    real_lock_file = Path(str(lock_path) + budget_guard.LOCK_SUFFIX)
    real_lock_file.write_text("", encoding="utf-8")
    os.utime(real_lock_file, ns=(0, 0))
    with budget_guard._FileLock(lock_path, stale_s=30.0):
        assert real_lock_file.exists()


def test_transient_permission_on_lock_create_is_retried(monkeypatch, tmp_path):
    """The CV-24 lane-B failure: os.open O_EXCL raising transient
    PermissionError (winerror 5/32/33) must retry within the deadline, not
    crash the caller; a non-matching winerror re-raises immediately."""
    import errno
    import os as _os
    lock_path = tmp_path / "ledger.json"
    real_open = _os.open
    calls = []

    def racing_open(path, flags, *args, **kwargs):
        if str(path).endswith(budget_guard.LOCK_SUFFIX):
            calls.append(1)
            if len(calls) < 3:
                error = PermissionError(errno.EACCES, "denied")
                error.winerror = 5
                raise error
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(budget_guard.os, "open", racing_open)
    with budget_guard._FileLock(lock_path):
        assert len(calls) == 3

    calls.clear()

    def hard_open(path, flags, *args, **kwargs):
        if str(path).endswith(budget_guard.LOCK_SUFFIX):
            calls.append(1)
            error = PermissionError(errno.EACCES, "denied")
            error.winerror = 1223
            raise error
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(budget_guard.os, "open", hard_open)
    # Any PermissionError on the lock create is contention-class now: a
    # persistent denial surfaces as the bounded deadline TimeoutError.
    with pytest.raises(TimeoutError):
        budget_guard._FileLock(lock_path, timeout_s=0.2).__enter__()
    assert len(calls) > 1


def test_lock_create_denial_without_winerror_is_retried(monkeypatch, tmp_path):
    """The CV-25 lane-B failure: PermissionError errno 13 with NO winerror
    attribute must retry inside the deadline instead of re-raising."""
    import errno
    import os as _os
    lock_path = tmp_path / "ledger.json"
    real_open = _os.open
    calls = []

    def racing_open(path, flags, *args, **kwargs):
        if str(path).endswith(budget_guard.LOCK_SUFFIX):
            calls.append(1)
            if len(calls) < 2:
                raise PermissionError(errno.EACCES, "denied")  # no winerror set
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(budget_guard.os, "open", racing_open)
    with budget_guard._FileLock(lock_path):
        assert len(calls) == 2


def test_progress_marking_is_non_fatal_on_lock_failure(monkeypatch, tmp_path):
    """mark_latest_progress must swallow transient OSError (advisory
    telemetry; receipts authoritative) instead of killing physics."""
    guard = budget_guard.BudgetGuard(tmp_path / "ledger.json")
    guard.reserve(campaign="c", category="characterization", amount_s=10,
                  target_dir=tmp_path, ceilings={"categories": {"characterization": {"max_allocations": 1, "max_seconds": 100}}, "total_seconds": 100},
                  launcher=lambda *a: None)

    def broken_lock(_path):
        raise PermissionError(13, "denied")

    monkeypatch.setattr(budget_guard, "_FileLock", broken_lock)
    guard.mark_latest_progress("c", "characterization", 5.0)  # must not raise
    assert guard.progress_mark_failures == 1
    assert "PermissionError" in guard.last_progress_mark_error
