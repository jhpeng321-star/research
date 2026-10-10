from types import SimpleNamespace

import pytest

from traffic_shepherd import recovery_policy as mod
from traffic_shepherd.duration_study import DurationAssignmentError
from traffic_shepherd.paired_experiments import PolicyOutput, PolicyInput


class Base:
    def __init__(self, phase="hold"):
        self.calls = []
        self.phase = phase
        self.proposal = SimpleNamespace(member_ids=("m",), fingerprint="p")
        self.previous_commands = {"m": 0.2}
        self.ready_at = 10
        self.taper_at = None
        self.terminal = None
        self.last_time = 14
        self.profile = SimpleNamespace()
        self.readiness = 3

    def __call__(self, context):
        self.calls.append(context.now_s)
        return PolicyOutput({}, self.phase, "base", ("m",), "p", "e", {})


def policy(base, branch=15):
    p = object.__new__(mod.RecoveryReleasePolicy)
    p.spec = {"branch_time_s": branch}
    p.base = base
    p.recovery = None
    p.recovery_receipt = None
    p.coasting = False
    p.finished = None
    p.last_output_phase = "hold"
    p.receipt = {"requested": False}
    return p


def ctx(t):
    return PolicyInput(t, (), (), {}, frozenset())


def test_exact_prebranch_delegates_original(monkeypatch):
    b = Base()
    p = policy(b)
    monkeypatch.setattr(mod, "_inner", lambda x: x)
    monkeypatch.setattr("traffic_shepherd.duration_study._inner", lambda x: x)
    out = p(ctx(14))
    assert out.phase == "hold" and b.calls == [14]


def test_branch_verifies_before_recovery_and_active_skips_base(monkeypatch):
    b = Base()
    p = policy(b)
    order = []
    monkeypatch.setattr(mod, "_inner", lambda x: x)
    p._verify_branch = lambda c, i: order.append("verify") or {"ok": True}

    class Core:
        def __init__(self, *args):
            order.append("core")

        def step(self, c):
            order.append("step")
            return {"status": "active", "commands": {"m": .1},
                    "reason": "r", "details": {}}
    monkeypatch.setattr(mod, "RecoveryController", Core)
    out = p(ctx(15))
    assert order == ["verify", "core", "step"]
    assert b.calls == [] and out.phase == "taper"


def test_fallback_calls_base_once_and_terminal_is_sticky(monkeypatch):
    b = Base(phase="release")
    p = policy(b)
    monkeypatch.setattr(mod, "_inner", lambda x: x)
    p._verify_branch = lambda c, i: {}
    class Core:
        def __init__(self, *args):
            pass

        def step(self, c):
            return {"status": "fallback", "commands": {},
                    "reason": "unsafe", "details": {}}
    monkeypatch.setattr(mod, "RecoveryController", Core)
    first = p(ctx(15))
    second = p(ctx(16))
    assert first.phase == second.phase == "release"
    assert b.calls == [15]


def test_missed_branch_rejects_without_base_or_core(monkeypatch):
    b = Base()
    p = policy(b)
    monkeypatch.setattr(mod, "_inner", lambda x: x)
    with pytest.raises(DurationAssignmentError):
        p(ctx(16))
    assert b.calls == []
