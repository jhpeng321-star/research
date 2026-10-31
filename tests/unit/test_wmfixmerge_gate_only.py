import pytest

from tools.wmfixmerge import gate_only_watch as w


def test_gate_only_requires_the_actual_bound_thirty_step_pass():
    assert not w.check_gate(dict(passed=False, status="BLOCKED"))
    assert w.check_gate(dict(passed=True, completed_steps=30,
                             protocol=dict(recovery_sha256=w.RECOVERY_SHA)))
    for doc in (dict(passed=True, completed_steps=29, protocol=dict(recovery_sha256=w.RECOVERY_SHA)),
                dict(passed=True, completed_steps=30, protocol=dict(recovery_sha256="other"))):
        with pytest.raises(ValueError):
            w.check_gate(doc)


def test_process_identity_refuses_pid_reuse_other_outputs_and_other_commands(tmp_path):
    proc = tmp_path / "123"
    proc.mkdir()
    # stat field22 = starttime; a process name with spaces must not shift it.
    (proc / "stat").write_text("123 (own process) " + " ".join(["S"] + ["0"] * 18 + ["456"]))
    argv = ["python", "-m", "tools.wmfixmerge.pilot_v2", "--registered-data-pilot",
            "--out", str(w.OUT), "--data-recovery-sha256", w.RECOVERY_SHA]
    (proc / "cmdline").write_bytes("\0".join(argv).encode())
    assert w.owned(123, "456", proc=tmp_path)
    assert not w.owned(123, "457", proc=tmp_path)
    assert not w.owned(999, "456", proc=tmp_path)
    argv[5] = "different-output"
    (proc / "cmdline").write_bytes("\0".join(argv).encode())
    assert not w.owned(123, "456", proc=tmp_path)
    argv[5] = str(w.OUT)
    argv[2] = "peer.job"
    (proc / "cmdline").write_bytes("\0".join(argv).encode())
    assert not w.owned(123, "456", proc=tmp_path)


def test_pidfd_signal_requires_final_ownership_check_and_always_closes(monkeypatch):
    calls = []
    monkeypatch.setattr(w, "open_pidfd", lambda pid: 7)
    monkeypatch.setattr(w.os, "close", lambda fd: calls.append(("close", fd)))
    monkeypatch.setattr(w, "send_pidfd", lambda fd: calls.append(("signal", fd, w.signal.SIGINT)))
    monkeypatch.setattr(w, "owned", lambda *a: False)
    assert not w.interrupt_owned(123, "456")
    assert calls == [("close", 7)]
    calls.clear()
    monkeypatch.setattr(w, "owned", lambda *a: True)
    assert w.interrupt_owned(123, "456")
    assert calls == [("signal", 7, w.signal.SIGINT), ("close", 7)]


def test_conda_wrapper_fallback_uses_pidfd_syscalls_without_pid_based_kill(monkeypatch):
    monkeypatch.delattr(w.os, "pidfd_open", raising=False)
    monkeypatch.delattr(w.signal, "pidfd_send_signal", raising=False)
    calls = []
    monkeypatch.setattr(w, "_syscall", lambda *args: calls.append(args) or 8)
    assert w.open_pidfd(123) == 8
    w.send_pidfd(8)
    assert calls == [(434, 123, 0), (424, 8, w.signal.SIGINT, 0, 0)]
