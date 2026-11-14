"""Focused tests for the damping-plus-wall composite (pure state logic)."""

from __future__ import annotations

import policies.damping_plus_wall as dpw


class _FakeDamping:
    def __init__(self):
        self.calls = 0
        self.last_diagnostics = {"commands": []}

    def reset(self, env, seed=None):
        self.calls = 0

    def act(self, obs):
        self.calls += 1
        return {}

    def close(self):
        pass


def test_composite_damping_called_every_step_and_wall_gated():
    original_factory = dpw.build_frozen_idm_analytical_policy
    fake = _FakeDamping()
    dpw.build_frozen_idm_analytical_policy = lambda: fake
    try:
        policy = dpw.DampingPlusWallPolicy(decision_step=250, wall_params={})
        policy.reset(None, seed=7)
        wall_steps = []
        for step in range(1, 320):
            policy.act({})
            if policy.last_audit["wall_active"]:
                wall_steps.append(step)
        assert policy._damping is fake
        assert fake.calls == 319  # damping ran every step
        assert wall_steps[0] == 251  # first wall command one step after the decision
        assert wall_steps == list(range(251, 320))  # holds until the cap
    finally:
        dpw.build_frozen_idm_analytical_policy = original_factory


def test_composite_releases_at_cap_with_taper():
    original_factory = dpw.build_frozen_idm_analytical_policy
    fake = _FakeDamping()
    dpw.build_frozen_idm_analytical_policy = lambda: fake
    try:
        policy = dpw.DampingPlusWallPolicy(decision_step=250, wall_params={})
        policy.reset(None, seed=7)
        wall_steps = []
        taper_steps = []
        released_step = None
        for step in range(1, 920):
            policy.act({})
            audit = policy.last_audit
            state = audit["treatment_state"]
            if state == "TAPERING":
                taper_steps.append(step)
            if audit["wall_active"]:
                wall_steps.append(step)
            if released_step is None and step > 251 and state == "RELEASED":
                released_step = step
        # Wall holds 251..900 (650 s); taper audits 901..905; released at 905.
        assert wall_steps[0] == 251 and wall_steps[649] == 900
        assert len(wall_steps) == 655  # hold + taper
        assert taper_steps == [901, 902, 903, 904, 905]
        assert released_step == 906  # first audit labeled RELEASED
    finally:
        dpw.build_frozen_idm_analytical_policy = original_factory
