"""Phase-4 IDM residual-learning contract tests."""

import argparse
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from env.rewards import compute_idm_damping_rewards
from models import _init_log_std_bias
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from train import (
    IDMResidualTrainingCallbacks,
    _acquire_train_lock,
    _build_env_config,
    _context_dim_from_args,
    _extract_train_metrics,
    _idm_decision_batch_metrics,
    _idm_absorber_gate_index,
    _obs_action_spaces,
    _release_train_lock,
)


def _args():
    return argparse.Namespace(
        sumo_binary="sumo",
        sumo_config="osm.dense.idm_m2200r400.sumocfg",
        sumo_root=None,
        scenario_refs=None,
        sumo_config_choices=[
            "osm.dense.idm_m1900r400.sumocfg",
            "osm.dense.idm_m2000r400.sumocfg",
            "osm.dense.idm_m2100r400.sumocfg",
            "osm.dense.idm_m2200r400.sumocfg",
        ],
        density="dense",
        cav_penetration=0.2,
        hdv_mixture=None,
        max_episode_steps=1200,
        max_neighbors=20,
        max_cavs_critic=48,
        baseline="shepherd",
        control_mode="idm_residual_smoothing",
        multi_density=False,
        curriculum_densities=None,
        multi_penetration=False,
        penetration_choices=[0.1, 0.2],
        use_meter_load=False,
        use_cav_count=False,
        use_wave_signal=False,
        use_health_obs=False,
        use_dv_obs=False,
        use_margin_obs=False,
        pruned_modes=False,
        enable_jad=False,
        enable_merge_match=False,
        enable_wave_damp=False,
    )


def test_phase4_config_freezes_analytical_reference_and_policy_context():
    args = _args()
    cfg = _build_env_config(args)
    assert cfg["control_mode"] == "idm_residual_smoothing"
    assert cfg["reward_mode"] == "idm_damping"
    assert cfg["idm_residual_reference_mode"] == "disturbance_p30"
    assert cfg["use_wave_signal"] is True
    assert cfg["use_idm_absorber_gate"] is True
    assert cfg["use_idm_controller_context"] is True
    assert cfg["idm_damping_reward_weights"]["residual_effort"] == 0.0
    assert cfg["idm_damping_reward_weights"]["delay_budget"] == 0.10
    assert cfg["idm_damping_reward_weights"]["corridor_severe_braking"] == 1.0
    assert cfg["idm_damping_reward_weights"]["corridor_wave_energy"] == 1.0
    assert cfg["idm_damping_reward_weights"]["wave"] == 0.0
    assert len(cfg["sumo_config_choices"]) == 4
    assert cfg["penetration_choices"] == [0.1, 0.2]
    obs_space, action_space, local_dim, mask_dim = _obs_action_spaces(args, True)
    assert mask_dim == 0
    assert tuple(action_space.low) == pytest.approx((-0.5,))
    assert tuple(action_space.high) == pytest.approx((0.3,))
    assert _context_dim_from_args(args) == 13  # wave(2), gate, controller/temporal(10)
    assert _idm_absorber_gate_index(args) == 123
    assert obs_space.shape[0] > local_dim


def test_decentralized_idm_uses_five_local_modes_and_action_mask():
    args = _args()
    args.idm_decentralized_modes = True
    cfg = _build_env_config(args)
    obs_space, action_space, local_dim, mask_dim = _obs_action_spaces(args, True)
    assert cfg["idm_residual_decentralized_damping_enabled"] is True
    assert action_space.n == 5
    assert mask_dim == 5
    assert obs_space.shape[0] > local_dim

    from env.core import TrafficShepherdMultiAgentEnv

    cfg["action_masking"] = True
    cfg["local_obs_dim"] = local_dim
    env = TrafficShepherdMultiAgentEnv(cfg)
    assert env.action_space.n == 5
    assert env.action_masking is True
    assert env._action_mask_dim == 5
    assert env._raw_local_dim == local_dim - 5
    placeholder = env._zero_agent_observation()
    assert placeholder[env._raw_local_dim : env._raw_local_dim + 5].tolist() == [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    args.idm_decentralized_action_set = "acceleration_caps_v1"
    cap_cfg = _build_env_config(args)
    _, cap_action_space, cap_local_dim, cap_mask_dim = _obs_action_spaces(args, True)
    assert cap_cfg["idm_residual_decentralized_action_set"] == "acceleration_caps_v1"
    assert cap_action_space.n == 4
    assert cap_mask_dim == 4
    assert cap_local_dim == local_dim - 1


def test_idm_reward_zeros_nonselected_and_shares_global_terms_once():
    weights = {
        "efficiency": 0.35,
        "safety": 2.0,
        "stability": 0.5,
        "delay_budget": 0.5,
        "corridor_ttc_penalty": 1.0,
        "corridor_severe_braking": 1.5,
        "corridor_wave_energy": 1.2,
        "wave": 1.0,
        "residual_effort": 0.15,
    }
    diagnostics = {}
    with (
        patch(
            "env.rewards.compute_shepherd_rewards",
            return_value={"a": -1.0, "b": -2.0},
        ),
        patch("env.rewards._efficiency", return_value=0.6),
        patch("env.rewards._delay_budget", return_value=-0.4),
        patch("env.rewards._corridor_ttc_penalty", return_value=-3.0),
        patch(
            "env.rewards.wave_risk_signal",
            side_effect=lambda agent_id: ({"a": 0.2, "b": 0.5}[agent_id], 1.0),
        ),
    ):
        rewards = compute_idm_damping_rewards(
            {"a": 0.1, "b": -0.5},
            ["a", "b", "nonselected"],
            35.0,
            weights,
            200.0,
            corridor_ttc_count=3,
            corridor_severe_braking_count=2,
            corridor_severe_braking_count_ref_per_s=4.0,
            corridor_acceleration_energy_sum_m2ps4=2.0,
            corridor_acceleration_energy_sum_ref_m2ps4=4.0,
            residual_scale_mps2=0.5,
            diagnostics=diagnostics,
        )
    # One global term: .35*.6 + .5*(-.4) + 1*(-3) = -2.99, shared / 2.
    assert rewards["nonselected"] == 0.0
    assert rewards["a"] == pytest.approx(-1.0 - 1.495 - 0.375 - 0.3 - 0.2 - 0.15 * 0.2**2)
    assert rewards["b"] == pytest.approx(-2.0 - 1.495 - 0.375 - 0.3 - 0.5 - 0.15)
    assert set(diagnostics) == {"a", "b"}
    assert diagnostics["a"]["total"] == pytest.approx(rewards["a"])
    assert diagnostics["b"]["total"] == pytest.approx(rewards["b"])
    for row in diagnostics.values():
        assert row["total"] == pytest.approx(
            sum(
                row[key]
                for key in (
                    "local",
                    "efficiency",
                    "delay_budget",
                    "corridor_ttc",
                    "corridor_severe_braking",
                    "corridor_wave_energy",
                    "wave",
                    "residual_effort",
                )
            )
        )


def test_decentralized_reward_gives_feasible_abstainer_corridor_return():
    diagnostics = {}
    with (
        patch("env.rewards.compute_shepherd_rewards", return_value={"a": 0.0, "b": 0.0}),
        patch("env.rewards._efficiency", return_value=0.0),
        patch("env.rewards._delay_budget", return_value=0.0),
        patch("env.rewards.wave_risk_signal", return_value=(0.0, 0.0)),
    ):
        rewards = compute_idm_damping_rewards(
            {"a": -0.2},
            ["a", "b", "ineligible"],
            35.0,
            {
                "corridor_severe_braking": 1.0,
                "corridor_wave_energy": 1.0,
                "residual_effort": 0.0,
            },
            200.0,
            corridor_severe_braking_count=1,
            corridor_severe_braking_count_ref_per_s=2.0,
            corridor_acceleration_energy_sum_m2ps4=2.0,
            corridor_acceleration_energy_sum_ref_m2ps4=4.0,
            shared_reward_ids={"a", "b"},
            diagnostics=diagnostics,
        )
    assert rewards["a"] == pytest.approx(-1.0)
    assert rewards["b"] == pytest.approx(-1.0)
    assert rewards["ineligible"] == 0.0
    assert set(diagnostics) == {"a", "b"}


def test_centered_corridor_reward_is_positive_below_frozen_reference():
    with (
        patch("env.rewards.compute_shepherd_rewards", return_value={"a": 0.0}),
        patch("env.rewards._efficiency", return_value=0.0),
        patch("env.rewards._delay_budget", return_value=0.0),
        patch("env.rewards.wave_risk_signal", return_value=(0.0, 0.0)),
    ):
        rewards = compute_idm_damping_rewards(
            {},
            ["a"],
            35.0,
            {"corridor_severe_braking": 1.0, "corridor_wave_energy": 1.0},
            200.0,
            corridor_severe_braking_count=1,
            corridor_severe_braking_count_ref_per_s=2.0,
            corridor_acceleration_energy_sum_m2ps4=2.0,
            corridor_acceleration_energy_sum_ref_m2ps4=4.0,
            shared_reward_ids={"a"},
            center_corridor_terms=True,
        )
    assert rewards["a"] == pytest.approx(1.0)


def test_idm_corridor_severe_braking_fraction_is_density_normalized():
    from env.core import TrafficShepherdMultiAgentEnv

    env = TrafficShepherdMultiAgentEnv.__new__(TrafficShepherdMultiAgentEnv)
    env.config = {"idm_severe_brake_threshold_mps2": -3.0}
    env._pre_edge = "pre"
    env._merge_edge = "merge"
    env._ramp_edge = "ramp"
    env._shepherd_engine = type("Engine", (), {"down_edge": "down"})()
    roads = {"a": "pre", "b": "merge", "c": "down", "outside": "other"}
    accelerations = {"a": -3.1, "b": -3.0, "c": 0.0, "outside": -9.0}
    snapshot = type(
        "Snapshot",
        (),
        {
            "ordered_ids": lambda self: list(roads),
            "road": lambda self, vehicle_id: roads[vehicle_id],
            "accel": lambda self, vehicle_id: accelerations[vehicle_id],
        },
    )()
    with patch("env.core.get_active_snapshot", return_value=snapshot):
        fraction = env._corridor_severe_braking_fraction()
        count = env._corridor_severe_braking_count()
        energy = env._corridor_acceleration_energy_m2ps4()
        energy_sum = env._corridor_acceleration_energy_sum_m2ps4()
    assert fraction == pytest.approx(1.0 / 3.0)
    assert count == 1
    assert energy == pytest.approx((3.1**2 + 3.0**2) / 3.0)
    assert energy_sum == pytest.approx(3.1**2 + 3.0**2)


def test_idm_pattern_context_is_causal_normalized_and_tracks_rising_risk():
    from env.core import TrafficShepherdMultiAgentEnv

    env = TrafficShepherdMultiAgentEnv.__new__(TrafficShepherdMultiAgentEnv)
    env.config = {"idm_pattern_short_tau_s": 10.0, "idm_pattern_long_tau_s": 30.0}
    env.simulation_step = 1.0
    env.corridor_severe_braking_count_ref_per_s = 2.0
    env.corridor_acceleration_energy_sum_ref_m2ps4 = 10.0
    severe = iter((2, 4))
    wave = iter((10.0, 30.0))
    env._corridor_severe_braking_count = lambda: next(severe)
    env._corridor_acceleration_energy_sum_m2ps4 = lambda: next(wave)

    env._reset_idm_pattern_context()
    env._update_idm_pattern_context()
    assert env._idm_pattern_context == pytest.approx(
        {
            "idm_severe_level": 1.0,
            "idm_wave_energy_level": 1.0,
            "idm_severe_short_ema": 1.0,
            "idm_wave_short_ema": 1.0,
            "idm_severe_trend": 0.0,
            "idm_wave_energy_trend": 0.0,
        }
    )
    env._update_idm_pattern_context()
    context = env._idm_pattern_context
    assert context["idm_severe_level"] == pytest.approx(2.0)
    assert context["idm_wave_energy_level"] == pytest.approx(3.0)
    assert 1.0 < context["idm_severe_short_ema"] < 2.0
    assert 1.0 < context["idm_wave_short_ema"] < 3.0
    assert context["idm_severe_trend"] > 0.0
    assert context["idm_wave_energy_trend"] > 0.0


def test_continuous_head_starts_at_exact_zero_mean_and_narrow_std():
    head = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))
    _init_log_std_bias(head, -2.0, zero_mean=True)
    output = head(torch.randn(32, 4))
    assert torch.all(output[:, 0] == 0.0)
    assert torch.all(output[:, 1] == -2.0)


def test_inactive_absorber_advantages_are_exactly_zero_after_centering():
    callback = IDMResidualTrainingCallbacks()
    local_dim = 4
    obs = np.zeros((5, 12), dtype=np.float32)
    gate_index = 1
    obs[[1, 3], gate_index] = 1.0
    batch = {
        SampleBatch.OBS: obs,
        Postprocessing.ADVANTAGES: np.array([9.0, 2.0, -4.0, 6.0, 3.0]),
    }
    policy = type(
        "P",
        (),
        {
            "config": {
                "model": {
                    "custom_model_config": {
                        "local_obs_dim": local_dim,
                        "idm_absorber_gate_index": gate_index,
                    }
                }
            }
        },
    )()
    callback.on_postprocess_trajectory(
        policies={"cav_policy": policy},
        policy_id="cav_policy",
        postprocessed_batch=batch,
    )
    assert batch[Postprocessing.ADVANTAGES].tolist() == [0.0, -2.0, 0.0, 2.0, 0.0]
    assert float(np.mean(batch[Postprocessing.ADVANTAGES])) == 0.0


def test_decentralized_callback_keeps_only_true_binary_decision_advantages():
    callback = IDMResidualTrainingCallbacks()
    obs = np.zeros((4, 8), dtype=np.float32)
    obs[:, 4:6] = [[1, 1], [1, 0], [1, 1], [0, 1]]
    batch = {
        SampleBatch.OBS: obs,
        SampleBatch.ACTIONS: np.array([0, 0, 1, 1]),
        SampleBatch.ACTION_DIST_INPUTS: np.zeros((4, 2), dtype=np.float32),
        Postprocessing.ADVANTAGES: np.array([1.0, 10.0, 3.0, 20.0]),
    }
    custom = {
        "local_obs_dim": 6,
        "raw_local_dim": 4,
        "action_mask_dim": 2,
        "force_action_zero_legal": False,
        "idm_absorber_gate_index": 1,
    }
    metrics = _idm_decision_batch_metrics(batch, custom)
    assert metrics == pytest.approx(
        {
            "policy_eligible_sample_count": 2.0,
            "policy_masked_sample_count": 2.0,
            "policy_activate_probability_sum": 1.0,
            "policy_activate_executed_count": 1.0,
            "policy_abstain_executed_count": 1.0,
        }
    )
    policy = type(
        "P",
        (),
        {"config": {"model": {"custom_model_config": custom}}},
    )()
    callback.on_postprocess_trajectory(
        policies={"cav_policy": policy},
        policy_id="cav_policy",
        postprocessed_batch=batch,
    )
    assert batch[Postprocessing.ADVANTAGES].tolist() == [1.0, 0.0, 3.0, 0.0]


def test_training_metric_extraction_includes_learner_and_idm_diagnostics():
    result = {
        "num_env_steps_sampled_lifetime": 2000,
        "info": {
            "learner": {
                "cav_policy": {
                    "learner_stats": {
                        "entropy": 0.41,
                        "kl": 0.003,
                        "policy_loss": -0.02,
                        "vf_loss": 3.0,
                        "vf_explained_var": 0.25,
                    }
                }
            }
        },
        "env_runners": {
            "episode_reward_mean": -10.0,
            "policy_reward_mean": {"cav_policy": -2.0},
            "episode_len_mean": 900.0,
            "num_episodes": 2,
            "hist_stats": {
                "idm_policy_eligible_sample_count": [2.0, 3.0],
                "idm_policy_activate_probability_sum": [1.0, 2.25],
                "idm_policy_activate_executed_count": [1.0, 2.0],
                "idm_policy_abstain_executed_count": [1.0, 1.0],
                "idm_winning_activation_count": [1.0, 1.0],
                "idm_minimum_ttc_s": [2.0, 3.0],
            },
        },
    }
    metrics = _extract_train_metrics(result)
    assert metrics["policy_activate_probability"] == pytest.approx(0.65)
    assert metrics["policy_activate_executed_count"] == 3.0
    assert metrics["policy_abstain_executed_count"] == 2.0
    assert metrics["winning_activation_count"] == 2.0
    assert metrics["minimum_ttc_s"] == pytest.approx(2.5)
    assert metrics["entropy"] == pytest.approx(0.41)
    assert metrics["kl"] == pytest.approx(0.003)
    assert metrics["value_explained_variance"] == pytest.approx(0.25)

    result["env_runners"]["num_episodes"] = 0
    stale = _extract_train_metrics(result)
    assert stale["policy_eligible_sample_count"] == 0.0
    assert stale["winning_activation_count"] == 0.0
    assert np.isnan(stale["policy_activate_probability"])
    assert np.isnan(stale["minimum_ttc_s"])


def test_training_lock_rejects_a_second_live_writer(tmp_path):
    lock = _acquire_train_lock(tmp_path)
    with pytest.raises(RuntimeError, match="Refusing concurrent trainer"):
        _acquire_train_lock(tmp_path)
    _release_train_lock(lock)
    assert not lock.exists()


def test_episode_telemetry_prefers_once_per_step_core_metrics_over_stale_agent_info():
    callback = IDMResidualTrainingCallbacks()
    episode = type(
        "Episode",
        (),
        {
            "user_data": {},
            "get_agents": lambda self: ["stale"],
            "last_info_for": lambda self, _agent: {
                "idm_training_metrics": {"step": 1, "eligible_decision_count": 99}
            },
        },
    )()
    core = type(
        "Core",
        (),
        {
            "_last_idm_training_metrics": {
                "step": 2,
                "eligible_decision_count": 3,
                "controlled_cav_seconds": 1.0,
                "reward_components": {},
            }
        },
    )()
    wrapped = type("Wrapped", (), {"_env": core})()
    base_env = type(
        "BaseEnv", (), {"get_sub_environments": lambda self: [wrapped]}
    )()
    callback.on_episode_step(episode=episode, base_env=base_env, env_index=0)
    store = episode.user_data["idm_training"]
    assert store["last_info_step"] == 2
    assert store["sums"]["eligible_decision_count"] == 3.0
    assert store["sums"]["controlled_cav_seconds"] == 1.0


def test_scenario_reward_refs_replace_mismatched_dense_defaults():
    from env.core import TrafficShepherdMultiAgentEnv

    env = TrafficShepherdMultiAgentEnv.__new__(TrafficShepherdMultiAgentEnv)
    env.config = {"sumo_config": "osm.dense.idm_m2200r400.sumocfg"}
    env._scenario_reward_refs = {
        "osm.dense.idm_m2200r400.sumocfg": {
            "delay_rate_ref": 0.133,
            "flow_q_ref": 0.637,
            "flow_v_ref": 27.39,
            "corridor_severe_braking_count_ref_per_s": 0.21,
            "corridor_acceleration_energy_sum_ref_m2ps4": 12.4,
        }
    }
    env.delay_rate_ref = None
    env.flow_q_ref = 0.95
    env.flow_v_ref = 17.5
    env.corridor_severe_braking_count_ref_per_s = 1.0
    env.corridor_acceleration_energy_sum_ref_m2ps4 = 1.0
    env._apply_scenario_reward_refs()
    assert env.delay_rate_ref == pytest.approx(0.133)
    assert env.flow_q_ref == pytest.approx(0.637)
    assert env.flow_v_ref == pytest.approx(27.39)
    assert env.corridor_severe_braking_count_ref_per_s == pytest.approx(0.21)
    assert env.corridor_acceleration_energy_sum_ref_m2ps4 == pytest.approx(12.4)
