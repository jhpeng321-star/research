"""Unit tests for MAPPO packing and Flat-MARL control-cost mapping."""

import numpy as np

from env.mappo import (
    acceleration_to_shepherd_mode,
    mappo_obs_dim,
    pack_mappo_observation,
    unpack_mappo_observation,
)


def test_near_zero_acceleration_is_no_intervention_for_control_cost():
    assert acceleration_to_shepherd_mode(0.0) == 0
    assert acceleration_to_shepherd_mode([5e-4]) == 0
    assert acceleration_to_shepherd_mode(-0.1) == 1
    assert acceleration_to_shepherd_mode(0.1) == 1


def test_pack_unpack_mappo_obs():
    local_dim = 4
    max_cavs = 3
    a = np.arange(local_dim, dtype=np.float32)
    b = np.arange(local_dim, dtype=np.float32) + 10
    all_obs = {"cav_b": b, "cav_a": a}
    packed = pack_mappo_observation(a, all_obs, max_cavs)
    assert packed.shape[0] == mappo_obs_dim(local_dim, max_cavs)
    local, peers, trust_weights = unpack_mappo_observation(packed, local_dim, max_cavs)
    np.testing.assert_array_equal(local, a)
    assert trust_weights[0] == 1 and trust_weights[1] == 1 and trust_weights[2] == 0
    # Without ego_id / positions: sorted keys cav_a, cav_b
    np.testing.assert_array_equal(peers[0], a)
    np.testing.assert_array_equal(peers[1], b)


def test_pack_overflow_warns_on_stderr(capsys):
    """More active CAVs than max_cavs: still truncate deterministically, but
    emit a loud stderr warning instead of silently dropping peers."""
    local_dim = 4
    max_cavs = 2
    a = np.zeros(local_dim, dtype=np.float32)
    all_obs = {
        "cav_c": np.full(local_dim, 3.0, dtype=np.float32),
        "cav_a": np.full(local_dim, 1.0, dtype=np.float32),
        "cav_b": np.full(local_dim, 2.0, dtype=np.float32),
    }
    packed = pack_mappo_observation(a, all_obs, max_cavs)
    assert packed.shape[0] == mappo_obs_dim(local_dim, max_cavs)
    err = capsys.readouterr().err
    assert "max_cavs" in err and "3" in err, f"expected overflow warning, got: {err!r}"
    pack_mappo_observation(a, all_obs, max_cavs)
    assert capsys.readouterr().err == ""
    # Deterministic truncation: first max_cavs sorted IDs are packed.
    _, peers, trust_weights = unpack_mappo_observation(packed, local_dim, max_cavs)
    np.testing.assert_array_equal(peers[0], all_obs["cav_a"])
    np.testing.assert_array_equal(peers[1], all_obs["cav_b"])
    assert trust_weights[0] == 1 and trust_weights[1] == 1


def test_pack_at_capacity_does_not_warn(capsys):
    local_dim = 4
    max_cavs = 2
    a = np.zeros(local_dim, dtype=np.float32)
    all_obs = {
        "cav_a": np.ones(local_dim, dtype=np.float32),
        "cav_b": np.ones(local_dim, dtype=np.float32),
    }
    pack_mappo_observation(a, all_obs, max_cavs)
    assert capsys.readouterr().err == ""


def test_mappo_model_forward_shapes():
    import gymnasium as gym
    import torch

    from env.features import raw_obs_dim
    from models import MAPPOModel

    local_dim = raw_obs_dim(20)
    max_cavs = 4
    obs_dim = mappo_obs_dim(local_dim, max_cavs)
    model = MAPPOModel(
        obs_space=gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32),
        action_space=gym.spaces.Discrete(5),
        num_outputs=5,
        model_config={
            "custom_model_config": {
                "max_neighbors": 20,
                "max_cavs_critic": max_cavs,
                "local_obs_dim": local_dim,
                "hidden": 64,
            }
        },
        name="test",
    )
    batch = 3
    obs = torch.zeros(batch, obs_dim)
    # mark first two peer slots valid
    obs[:, -max_cavs : -max_cavs + 2] = 1.0
    logits, _ = model({"obs": obs}, [], None)
    assert logits.shape == (batch, 5)
    v = model.value_function()
    assert v.shape == (batch,)


def test_default_max_cavs_critic_covers_measured_max():
    """Default must exceed the measured max concurrent active CAVs (26 at p20
    dense, seeds 42/43/44 — see tools/measure_max_active_cavs.py)."""
    from config.defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["max_cavs_critic"] > 26
    assert DEFAULT_CONFIG["max_cavs_critic"] == 32


def test_env_obs_space_follows_max_cavs_config():
    """env obs space and MAPPOModel must both derive from max_cavs_critic."""
    import gymnasium as gym
    import torch

    from config.constants import NUM_MODES  # noqa: F401
    from config.defaults import DEFAULT_CONFIG
    from env.core import TrafficShepherdMultiAgentEnv
    from env.features import raw_obs_dim
    from models import MAPPOModel

    env = TrafficShepherdMultiAgentEnv({"centralized_critic": True, "max_cavs_critic": 32})
    # Defaults enable action_masking → local = assembler + default mode count.
    local_dim = raw_obs_dim(20) + len(DEFAULT_CONFIG["shepherd_modes"])
    expected_dim = mappo_obs_dim(local_dim, 32)
    assert env.observation_space.shape == (expected_dim,)
    assert env._action_mask_dim == len(DEFAULT_CONFIG["shepherd_modes"])

    model = MAPPOModel(
        obs_space=env.observation_space,
        action_space=gym.spaces.Discrete(5),
        num_outputs=5,
        model_config={
            "custom_model_config": {
                "max_neighbors": 20,
                "max_cavs_critic": 32,
                "local_obs_dim": local_dim,
                "action_mask_dim": len(DEFAULT_CONFIG["shepherd_modes"]),
                "raw_local_dim": raw_obs_dim(20),
                "hidden": 64,
            }
        },
        name="test_config_flow",
    )
    obs = torch.zeros(2, expected_dim)
    # coast always legal in mask tail of each local slot — set ego local mask
    obs[:, raw_obs_dim(20) : raw_obs_dim(20) + NUM_MODES] = 0.0
    obs[:, raw_obs_dim(20)] = 1.0  # coast
    logits, _ = model({"obs": obs}, [], None)
    assert logits.shape == (2, 5)
    obs[:, -32:-30] = 1.0  # two active peers
    logits, _ = model({"obs": obs}, [], None)
    assert logits.shape == (2, 5)
    assert model.value_function().shape == (2,)


def test_decentralized_idm_latched_mask_does_not_reopen_abstain():
    import gymnasium as gym
    import torch

    from env.features import raw_obs_dim
    from models import MAPPOModel

    raw_dim = raw_obs_dim(20)
    local_dim = raw_dim + 5
    max_cavs = 2
    obs_dim = mappo_obs_dim(local_dim, max_cavs)
    model = MAPPOModel(
        obs_space=gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32),
        action_space=gym.spaces.Discrete(5),
        num_outputs=5,
        model_config={
            "custom_model_config": {
                "max_neighbors": 20,
                "max_cavs_critic": max_cavs,
                "local_obs_dim": local_dim,
                "action_mask_dim": 5,
                "raw_local_dim": raw_dim,
                "force_action_zero_legal": False,
            }
        },
        name="test_decentralized_latch",
    )
    obs = torch.zeros(1, obs_dim)
    obs[0, raw_dim + 3] = 1.0
    logits, _ = model({"obs": obs}, [], None)
    assert logits[0, 3] > -20.0
    assert torch.all(logits[0, [0, 1, 2, 4]] < -20.0)


def test_binary_decentralized_gate_starts_at_exact_half_probability():
    import gymnasium as gym
    import torch

    from env.features import raw_obs_dim
    from models import MAPPOModel

    raw_dim = raw_obs_dim(20, include_wave_signal=True, include_idm_absorber_gate=True,
                          include_idm_controller_context=True)
    local_dim = raw_dim + 2
    max_cavs = 2
    obs_dim = mappo_obs_dim(local_dim, max_cavs)
    model = MAPPOModel(
        obs_space=gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32),
        action_space=gym.spaces.Discrete(2),
        num_outputs=2,
        model_config={
            "custom_model_config": {
                "max_neighbors": 20,
                "max_cavs_critic": max_cavs,
                "local_obs_dim": local_dim,
                "action_mask_dim": 2,
                "raw_local_dim": raw_dim,
                "force_action_zero_legal": False,
                "context_dim": 13,
                "zero_mean_init": True,
            }
        },
        name="test_binary_zero_init",
    )
    obs = torch.randn(4, obs_dim)
    obs[:, raw_dim : raw_dim + 2] = 1.0
    logits, _ = model({"obs": obs}, [], None)
    probabilities = torch.softmax(logits, dim=-1)
    assert torch.all(probabilities == 0.5)


def test_ippo_actor_supports_the_same_decentralized_mask():
    import gymnasium as gym
    import torch

    from env.features import raw_obs_dim
    from models import SelfishPPOModel

    raw_dim = raw_obs_dim(20)
    local_dim = raw_dim + 5
    model = SelfishPPOModel(
        obs_space=gym.spaces.Box(-np.inf, np.inf, (local_dim,), np.float32),
        action_space=gym.spaces.Discrete(5),
        num_outputs=5,
        model_config={
            "custom_model_config": {
                "max_neighbors": 20,
                "local_obs_dim": local_dim,
                "action_mask_dim": 5,
                "raw_local_dim": raw_dim,
                "force_action_zero_legal": False,
            }
        },
        name="test_decentralized_ippo",
    )
    obs = torch.zeros(1, local_dim)
    obs[0, raw_dim + 4] = 1.0
    logits, _ = model({"obs": obs}, [], None)
    assert logits[0, 4] > -20.0
    assert torch.all(logits[0, :4] < -20.0)


if __name__ == "__main__":
    test_near_zero_acceleration_is_no_intervention_for_control_cost()
    test_pack_unpack_mappo_obs()
    test_mappo_model_forward_shapes()
    print("ok")
