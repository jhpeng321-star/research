"""Tests for the hybrid (mode, magnitude) action space — composite form.

v3 (2026-08-05): mode is a model-side Categorical (mask applied to logits
exactly like the discrete champion); the dial is a Gaussian scalar in
[-1, 1] mapped to [0, 1] by the env parser. v1/v2's env-side argmax over
Gaussian scores broke the PPO gradient (executed action ~ independent of
logged action) — the composite eliminates the mismatch.
"""

import numpy as np

from env.core import TrafficShepherdMultiAgentEnv
from tests.test_shepherd_actions import _make_engine, _mock_traci
from unittest.mock import patch


# -- parser (tuple in -> (mode, mag [0,1]) out) ----------------------------
def _parser_env():
    env = TrafficShepherdMultiAgentEnv.__new__(TrafficShepherdMultiAgentEnv)
    env.config = {"shepherd_modes": ["coast", "fair_balance", "recover", "vacate_drop", "guard"]}
    return env


def test_parser_tuple_passthrough():
    env = _parser_env()
    mode, mag = env._parse_hybrid_action("cav0", (1, 1.0))
    assert mode == 1 and mag == 1.0
    mode, mag = env._parse_hybrid_action("cav0", (2, -1.0))
    assert mode == 2 and mag == 0.0
    mode, mag = env._parse_hybrid_action("cav0", (0, 0.0))
    assert mode == 0 and mag == 0.5


def test_parser_numpy_and_clip():
    env = _parser_env()
    mode, mag = env._parse_hybrid_action("cav0", (np.int64(3), np.array([5.0])))
    assert mode == 3 and mag == 1.0  # clipped
    mode, mag = env._parse_hybrid_action("cav0", [4, -0.4])
    assert mode == 4 and abs(mag - 0.3) < 1e-6


# -- engine magnitude mappings (unchanged from v1) --------------------------
def _engine_with_spy():
    engine = _make_engine()
    calls = {}

    def _spy(cav_id, current_speed, v_target, tau):
        calls["v_target"] = v_target
        calls["current_speed"] = current_speed

    engine._cmd_speed = _spy
    engine._apply_coast = lambda cav_id: calls.setdefault("coast", True)
    return engine, calls


def test_fair_balance_mag_full_and_zero():
    engine, calls = _engine_with_spy()
    with patch.object(engine, "_fair_balance_eligible", return_value=True):
        engine._fair_balance("cav0", 20.0, mag=1.0)
    assert abs(calls["v_target"] - 20.0 * 0.80) < 1e-6

    engine, calls = _engine_with_spy()
    with patch.object(engine, "_fair_balance_eligible", return_value=True):
        engine._fair_balance("cav0", 20.0, mag=0.0)
    assert abs(calls["v_target"] - 20.0 * 1.0) < 1e-6


def test_protect_mag_scales_tau_not_speed():
    # 2026-08-11 redesign: protect's job is to CREATE HEADWAY — the dial
    # scales the tau bump (m=1 -> full protect_tau_delta), while the speed
    # factor stays mild (0.97..1.0). m=0.5 -> half tau bump, factor 0.985.
    engine, calls = _engine_with_spy()
    with patch.object(engine, "_protect_eligible", return_value=True):
        engine._protect("cav0", 20.0, mag=1.0)
    assert abs(calls["v_target"] - 20.0 * 0.97) < 1e-6

    engine, calls = _engine_with_spy()
    with patch.object(engine, "_protect_eligible", return_value=True):
        engine._protect("cav0", 20.0, mag=0.5)
    assert abs(calls["v_target"] - 20.0 * 0.985) < 1e-6


def test_fair_balance_mag_none_is_legacy():
    engine, calls = _engine_with_spy()
    with patch.object(engine, "_fair_balance_eligible", return_value=True):
        engine._fair_balance("cav0", 20.0, mag=None)
    assert abs(calls["v_target"] - 20.0 * 0.90) < 1e-6


def test_recover_mag_maps_push():
    engine, calls = _engine_with_spy()
    with patch.object(engine, "_recover_eligible", return_value=True):
        engine._recover_speed("cav0", 10.0, mag=1.0)
    assert abs(calls["v_target"] - (10.0 + 2.0 * engine.dt)) < 1e-6


def test_near_zero_mag_is_coast_not_lock():
    engine, calls = _engine_with_spy()
    mock_traci = _mock_traci(speed=20.0)
    with patch("env.actions.traci", mock_traci):
        engine.apply("cav0", 1, mag=0.01)
    assert calls.get("coast") is True
    assert "v_target" not in calls


def test_apply_all_tuple_parsing():
    engine, calls = _engine_with_spy()
    mock_traci = _mock_traci(speed=20.0)
    with patch("env.actions.traci", mock_traci):
        with patch.object(engine, "_fair_balance_eligible", return_value=True):
            engine.apply_all({"cav0": (1, 1.0), "cav1": 0})
    assert abs(calls["v_target"] - 20.0 * 0.80) < 1e-6


# -- model composite head ---------------------------------------------------
def _hybrid_model(local_dim, max_cavs=4, hidden=64, mask_dim=0):
    import gymnasium as gym
    from env.mappo import mappo_obs_dim
    from models import MAPPOModel

    obs_dim = mappo_obs_dim(local_dim, max_cavs)
    return MAPPOModel(
        obs_space=gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32),
        action_space=gym.spaces.Tuple(
            (gym.spaces.Discrete(5), gym.spaces.Box(-1.0, 1.0, (1,), np.float32))
        ),
        num_outputs=7,
        model_config={
            "custom_model_config": {
                "max_neighbors": 20,
                "max_cavs_critic": max_cavs,
                "local_obs_dim": local_dim,
                "hidden": hidden,
                "action_mask_dim": mask_dim,
                "raw_local_dim": local_dim - mask_dim,
            }
        },
        name="test",
    )


def test_hybrid_forward_layout_and_init():
    import torch
    from env.features import raw_obs_dim

    local_dim = raw_obs_dim(20)
    model = _hybrid_model(local_dim)
    logits, _ = model({"obs": torch.zeros(3, model.obs_space.shape[0])}, [], None)
    assert logits.shape == (3, 7)
    dial_mean = logits[:, 5]
    dial_log_std = logits[:, 6]
    assert float(dial_mean.detach().abs().max()) <= 1.0
    dial_log_std = dial_log_std.detach()
    assert -5.0 <= float(dial_log_std.min()) and float(dial_log_std.max()) <= 2.0
    # init: dial mean -0.7 (gentle), dial log_std -2.0 (tight exploration)
    final = model.policy_head[-1]
    assert abs(float(final.bias[5].detach()) - (-0.7)) < 1e-6
    assert abs(float(final.bias[6].detach()) - (-2.0)) < 1e-6


def test_hybrid_mask_applied_model_side():
    import torch
    from env.features import raw_obs_dim

    mask_dim = 5
    local_dim = raw_obs_dim(20) + mask_dim
    model = _hybrid_model(local_dim, mask_dim=mask_dim)
    obs_dim = model.obs_space.shape[0]
    obs = torch.zeros(2, obs_dim)
    # mask in the local segment tail: fair_balance (idx 1) soft-masked 0.1,
    # rest legal
    mask = torch.tensor([1.0, 0.1, 1.0, 1.0, 1.0])
    obs[:, local_dim - mask_dim : local_dim] = mask
    # force fair_balance to lead raw logits by 1.5 (< 2.3 mask penalty);
    # zero the full final layer so random init terms can't confound
    with torch.no_grad():
        model.policy_head[-1].weight[:] = 0.0
        model.policy_head[-1].bias[:] = 0.0
        model.policy_head[-1].bias[1] = 1.5
        model.policy_head[-1].bias[5] = -0.7  # restore dial init
        model.policy_head[-1].bias[6] = -2.0
    logits, _ = model({"obs": obs}, [], None)
    mode_logits = logits[:, :5]
    # coast (0) should beat fair_balance (1) after the -2.3 mask penalty
    assert int(mode_logits[0].argmax()) == 0
