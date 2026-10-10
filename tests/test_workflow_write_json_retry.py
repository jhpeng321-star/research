"""Bounded write_json atomic-replace retry tests; launchers are never retried."""
import errno
import json

import pytest

from traffic_shepherd import workflow


def _permission(winerror):
    error = PermissionError(errno.EACCES, "denied")
    error.winerror = winerror
    return error


def test_transient_replace_failure_retries_same_payload(monkeypatch, tmp_path):
    original = workflow.os.replace
    calls = []
    sleeps = []
    monkeypatch.setattr(workflow.os, "replace",
                        lambda s, t: calls.append(str(t)) or (_ for _ in ()).throw(_permission(5)))
    monkeypatch.setattr(workflow.time, "sleep", lambda s: sleeps.append(s))
    # first implementation attempt: route through real os.replace after N calls
    real = workflow.os.replace

    def replace_after_two(source, target):
        calls.append(str(target))
        if len(calls) < 3:
            raise _permission(5)
        return original(source, target)

    monkeypatch.setattr(workflow.os, "replace", replace_after_two)
    target = tmp_path / "manifest.json"
    workflow.write_json(target, {"status": "complete", "seed": 7001})
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "complete", "seed": 7001}
    assert len(calls) == 3
    assert len(sleeps) == 2 and all(s == workflow._REPLACE_RETRY_DELAY_S for s in sleeps)
    assert not (tmp_path / "manifest.json.tmp").exists()


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_persistent_denial_raises_after_bounded_attempts(monkeypatch, tmp_path, winerror):
    calls = []
    monkeypatch.setattr(workflow.os, "replace",
                        lambda s, t: calls.append(1) or (_ for _ in ()).throw(_permission(winerror)))
    monkeypatch.setattr(workflow.time, "sleep", lambda _s: None)
    with pytest.raises(PermissionError):
        workflow.write_json(tmp_path / "manifest.json", {"a": 1})
    assert len(calls) == workflow._REPLACE_ATTEMPTS


def test_non_permission_replace_error_raises_immediately(monkeypatch, tmp_path):
    calls = []

    def replace(source, target):
        calls.append(1)
        raise OSError(errno.EIO, "io error")

    monkeypatch.setattr(workflow.os, "replace", replace)
    monkeypatch.setattr(workflow.time, "sleep", lambda _s: None)
    with pytest.raises(OSError):
        workflow.write_json(tmp_path / "manifest.json", {"a": 1})
    assert len(calls) == 1


def test_semantics_unchanged_payload_and_no_temp_leftover(tmp_path):
    target = tmp_path / "summary.json"
    workflow.write_json(target, {"seed": 7001, "value": 1.5})
    assert json.loads(target.read_text(encoding="utf-8")) == {"seed": 7001, "value": 1.5}
    assert not (tmp_path / "summary.json.tmp").exists()
