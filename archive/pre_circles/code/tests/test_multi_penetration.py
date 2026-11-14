"""Tests for multi-penetration training (per-episode penetration sampling)."""

import numpy as np

from env.core import TrafficShepherdMultiAgentEnv
from train import _build_env_config


def _env_stub(choices, seed=0):
    env = TrafficShepherdMultiAgentEnv.__new__(TrafficShepherdMultiAgentEnv)
    env._penetration_choices = list(choices) if choices else None
    env._episode_seed_rng = np.random.default_rng(seed)
    env.config = {"cav_penetration": 0.1}
    return env


def test_penetration_sampling_picks_from_choices():
    env = _env_stub([0.05, 0.2])
    env._apply_episode_penetration()
    assert env.config["cav_penetration"] in (0.05, 0.2)


def test_penetration_sampling_covers_all_choices():
    env = _env_stub([0.05, 0.1, 0.2])
    seen = set()
    for _ in range(60):
        env._apply_episode_penetration()
        seen.add(env.config["cav_penetration"])
    assert seen == {0.05, 0.1, 0.2}


def test_penetration_sampling_disabled_without_choices():
    env = _env_stub(None)
    env._apply_episode_penetration()
    assert env.config["cav_penetration"] == 0.1


class _Args:
    density = "medium"
    curriculum_densities = None
    multi_density = False
    multi_penetration = True
    sumo_binary = "sumo"
    cav_penetration = 0.1
    max_episode_steps = 1200
    max_neighbors = 20
    max_cavs_critic = 32
    hdv_mixture = None
    baseline = "shepherd"


def test_build_env_config_multi_penetration():
    cfg = _build_env_config(_Args())
    assert cfg["penetration_choices"] == [0.05, 0.1, 0.2, 0.3]


def test_build_env_config_default_no_penetration_choices():
    args = _Args()
    args.multi_penetration = False
    cfg = _build_env_config(args)
    assert "penetration_choices" not in cfg
