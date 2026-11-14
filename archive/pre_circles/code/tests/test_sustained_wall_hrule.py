"""Focused tests for the HRULE / SE rule arms (pure state-machine logic)."""

from __future__ import annotations

from policies.sustained_wall_hrule import HysteresisWallPolicy, SingleEngagementWallPolicy


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


def test_hrule_engagement_gate_and_hysteresis():
    # h < 0.25 from the first possible read; never releases (no health > 0.45).
    env = _FakeEnv([0.1] * 300)
    policy = HysteresisWallPolicy(arm_start_step=100, trigger_h=0.25, release_h=0.45)
    policy.reset(env, seed=7)
    wall_steps = []
    for step in range(1, 140):
        policy.act({})
        if policy.last_audit["wall_active"]:
            wall_steps.append(step)
    assert wall_steps[0] == 101  # first command after observing h at t=100
    assert wall_steps == list(range(101, 140))
    assert policy.engagements == 1
    assert policy.first_engage_step == 101


def test_hrule_release_and_reengagement():
    # h: 0.1 (arm), then 0.5 (release), then 0.1 again (re-engage).
    env = _FakeEnv([0.1] * 102 + [0.5] * 60 + [0.1] * 60)
    policy = HysteresisWallPolicy(arm_start_step=100, trigger_h=0.25, release_h=0.45)
    policy.reset(env, seed=7)
    wall_steps = []
    for step in range(1, 240):
        policy.act({})
        if policy.last_audit["wall_active"]:
            wall_steps.append(step)
    # Step 101 engages; health 0.5 from call 103 -> release at step 102+1=103?
    # calls: 1..101 -> 0.1 (arm at 101); 102 -> 0.1; 103..162 -> 0.5 : at
    # act(103) the pre-step read is 0.5 -> release; re-arm at 163 -> 0.1.
    assert 101 in wall_steps and 102 in wall_steps
    assert 103 not in wall_steps
    assert 163 in wall_steps
    assert policy.engagements == 2
    assert policy.first_engage_step == 101


def test_hrule_no_health_means_no_arm():
    policy = HysteresisWallPolicy(arm_start_step=100)
    policy.reset(_FakeEnv([]), seed=7)  # health always 1.0
    wall_steps = []
    for step in range(1, 130):
        policy.act({})
        if policy.last_audit["wall_active"]:
            wall_steps.append(step)
    assert wall_steps == []


def test_se_single_engagement_timing_and_release():
    # Health always 1.0: engagement applies one second, then releases.
    env = _FakeEnv([1.0] * 400)
    policy = SingleEngagementWallPolicy(decision_step=150, release_h=0.45, demand_end_s=900.0)
    policy.reset(env, seed=7)
    wall_steps = []
    for step in range(1, 200):
        policy.act({})
        if policy.last_audit["wall_active"]:
            wall_steps.append(step)
    assert wall_steps == [151]
    assert policy.last_audit["treatment_state"] == "RELEASED"


def test_se_no_reengagement_after_health_release():
    # Health dips again after release; no re-engagement allowed.
    env = _FakeEnv([1.0] * 160 + [0.1] * 100)
    policy = SingleEngagementWallPolicy(decision_step=150, release_h=0.45, demand_end_s=900.0)
    policy.reset(env, seed=7)
    wall_steps = []
    for step in range(1, 280):
        policy.act({})
        if policy.last_audit["wall_active"]:
            wall_steps.append(step)
    assert wall_steps == [151]
    assert len(wall_steps) == 1
