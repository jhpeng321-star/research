"""Focused tests for the minimum-hold wall policy (pure state machine)."""

from __future__ import annotations

from policies.sustained_wall_minhold import MinimumHoldWallPolicy


class _FakeEnv:
    def __init__(self, health_sequence):
        self._seq = list(health_sequence)
        self.calls = 0

    def metastable_health_detail(self):
        self.calls += 1
        if not self._seq:
            return (1.0, 0.0, 25.0)
        value = self._seq.pop(0) if len(self._seq) > 1 else self._seq[0]
        return (float(value), 0.0, 25.0)


def test_minhold_does_not_release_before_h_min():
    # Health above 0.45 immediately: the wall must still stay on for H_min.
    env = _FakeEnv([1.0] * 400)
    policy = MinimumHoldWallPolicy(decision_step=250, min_hold_s=120.0, recovery_persist_s=30.0)
    policy.reset(env, seed=7)
    wall_steps = []
    first_release_step = None
    for step in range(1, 420):
        policy.act({})
        if policy.last_audit["wall_active"]:
            wall_steps.append(step)
        elif first_release_step is None and step > 251:
            first_release_step = step
    assert wall_steps[0] == 251
    # 120 s of wall from step 251 = steps 251..370; the persistence window
    # starts at 371 and the 30th consecutive healthy reading lands at 400 ->
    # release there. Wall steps = 251..399 = 149.
    assert len(wall_steps) == 149
    assert first_release_step == 400


def test_minhold_health_dips_reset_persistence():
    # High health from the min-hold window start for 20 s, a dip, then high
    # forever: release happens after the persistence run completes again.
    seq = [1.0] * 371 + [0.1] * 3 + [1.0] * 300
    env = _FakeEnv(seq)
    policy = MinimumHoldWallPolicy(decision_step=250, min_hold_s=120.0, recovery_persist_s=30.0)
    policy.reset(env, seed=7)
    wall_steps = []
    first_release_step = None
    for step in range(1, 460):
        policy.act({})
        if policy.last_audit["wall_active"]:
            wall_steps.append(step)
        elif first_release_step is None and step > 251:
            first_release_step = step
    assert wall_steps[0] == 251
    # 120 s minimum hold (251..370); persistence from 371: 20 healthy readings
    # (371..390), dip resets (391..393), then healthy -> 30th reading at 404
    # -> release at 404. Wall steps 251..403 = 153.
    assert len(wall_steps) == 153
    assert first_release_step == 404
