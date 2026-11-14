"""Tests for the RSU wall supervisor (env adapter + eval wrapper)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from env.wall_supervisor import OBS_DIM, RUNGS, corridor_state
from policies.supervisor_policy import SupervisorPolicy


def _mock_ma_env(h=(0.5, 3.0, 20.0)):
    return SimpleNamespace(
        metastable_health_detail=lambda: h,
        active_cavs=set(),
        simulation_step=1.0,
        max_episode_steps=100,
    )


class _FakePolicy:
    def __init__(self, action):
        self._action = action

    def compute_single_action(self, obs, explore=False):
        return (self._action, None, {})


def test_rungs_match_measured_operating_points():
    # soft = the p=0.2 champion config (w10_t30, short band) sans trigger
    assert RUNGS[1] == {"d_lo": 200.0, "d_hi": 400.0, "wall_speed": 10.0, "wall_tau": 3.0}
    # strong = the low-pen champion config (w12_t30, long band) sans trigger
    assert RUNGS[2] == {"d_lo": 200.0, "d_hi": 800.0, "wall_speed": 12.0, "wall_tau": 3.0}


def test_corridor_state_shape_and_health_fallback():
    with patch("env.snapshot.get_active_snapshot", return_value=None):
        obs = corridor_state(_mock_ma_env(), None, 0, 0, 0, {})
    assert obs.shape == (OBS_DIM,)
    assert np.all(np.isfinite(obs))
    # h=0.5 flows to slot 0; dh=0 at the first step
    assert abs(obs[0] - 0.5) < 1e-6 and obs[1] == 0.0
    # None health detail -> healthy default h=1.0
    env = _mock_ma_env()
    env.metastable_health_detail = lambda: None
    with patch("env.snapshot.get_active_snapshot", return_value=None):
        obs = corridor_state(env, None, 0, 0, 0, {})
    assert abs(obs[0] - 1.0) < 1e-6


def test_supervisor_policy_off_releases_all_cavs():
    pol = SupervisorPolicy("unused")
    pol._policy = _FakePolicy(0)  # pre-set so reset() skips checkpoint loading
    pol.reset(_mock_ma_env())
    with (
        patch("policies.supervisor_policy.release_to_sumo") as rel,
        patch("traci.vehicle.getIDList", return_value=["cav0", "cav1"]),
    ):
        out = pol.act({"cav0": None, "cav1": None})
    assert out == {}
    assert rel.call_count == 2
    assert pol._rung == 0


def test_supervisor_policy_rung_broadcasts_wall():
    pol = SupervisorPolicy("unused")
    pol._policy = _FakePolicy(1)  # wall on (binary: 0=off, 1=wall)
    pol.reset(_mock_ma_env())
    pol._walls[1] = MagicMock()
    out = pol.act({"cav0": None, "cav1": None})
    assert out == {}
    pol._walls[1].act.assert_called_once()  # every CAV id forwarded
    assert set(pol._walls[1].act.call_args.args[0].keys()) == {"cav0", "cav1"}
    assert pol._rung == 1 and pol._engaged == 1
