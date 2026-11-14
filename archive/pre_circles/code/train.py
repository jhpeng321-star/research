"""Train baselines with Ray RLlib (parallel SUMO env runners)."""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.tune.registry import register_env

from config.constants import (
    ACCELERATION_MAX,
    ACCELERATION_MIN,
    FLAT_MARL_ACCEL_MAX,
    FLAT_MARL_ACCEL_MIN,
)
from config.defaults import (
    FAIRNESS_R_REF_BY_DENSITY,
    TAIL_R_REF_BY_DENSITY,
    FLOW_Q_REF_BY_DENSITY,
    FLOW_V_REF_BY_DENSITY,
)
from env.features import raw_obs_dim
from env.mappo import mappo_obs_dim
from env.rllib_wrapper import POLICY_ID, make_traffic_shepherd_env
from models import (
    MAPPO_MODEL_ID,
    SELFISH_MODEL_ID,
    register_mappo_model,
    register_selfish_ppo_model,
)
from policies.rllib_policy import is_rllib_checkpoint, resolve_rllib_checkpoint_dir

TRAIN_STATE_NAME = "train_state.json"
TRAIN_LOCK_NAME = "active_trainer.lock"

_IDM_EPISODE_SUM_FIELDS = (
    "policy_action_count",
    "eligible_decision_count",
    "safety_masked_decision_count",
    "nondecision_masked_count",
    "executed_activate_count",
    "executed_abstain_count",
    "winning_activation_count",
    "treatment_completion_count",
    "safety_abort_count",
    "controlled_cav_seconds",
    "collision_observations",
    "teleport_starts",
    "arrivals",
    "severe_braking_count",
    "acceleration_energy_sum_m2ps4",
)
_IDM_EPISODE_MEAN_FIELDS = (
    "active_treatment_count",
    "corridor_delay_rate",
    "merge_discharge_rate",
    "severe_braking_fraction",
)


def _idm_decision_batch_metrics(batch, custom: dict) -> dict:
    """Extract eligible binary-policy probabilities/actions from one batch."""
    mask_dim = int(custom.get("action_mask_dim", 0))
    raw_local_dim = int(custom.get("raw_local_dim", 0))
    if mask_dim != 2 or raw_local_dim <= 0:
        return {}
    obs = np.asarray(batch.get(SampleBatch.OBS, ()), dtype=np.float32)
    logits = np.asarray(batch.get(SampleBatch.ACTION_DIST_INPUTS, ()), dtype=np.float64)
    actions = np.asarray(batch.get(SampleBatch.ACTIONS, ())).reshape(-1)
    if obs.ndim != 2 or logits.ndim != 2 or len(actions) != len(obs):
        return {}
    masks = obs[:, raw_local_dim : raw_local_dim + mask_dim]
    if masks.shape != (len(obs), 2) or logits.shape[1] < 2:
        return {}
    eligible = np.all(masks > 0.5, axis=1)
    eligible_count = int(np.sum(eligible))
    if eligible_count == 0:
        return {
            "policy_eligible_sample_count": 0.0,
            "policy_masked_sample_count": float(len(obs)),
            "policy_activate_probability_sum": 0.0,
            "policy_activate_executed_count": 0.0,
            "policy_abstain_executed_count": 0.0,
        }
    eligible_logits = logits[eligible, :2]
    eligible_logits -= np.max(eligible_logits, axis=1, keepdims=True)
    probabilities = np.exp(eligible_logits)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    eligible_actions = actions[eligible]
    return {
        "policy_eligible_sample_count": float(eligible_count),
        "policy_masked_sample_count": float(len(obs) - eligible_count),
        "policy_activate_probability_sum": float(np.sum(probabilities[:, 1])),
        "policy_activate_executed_count": float(np.sum(eligible_actions == 1)),
        "policy_abstain_executed_count": float(np.sum(eligible_actions == 0)),
    }


class IDMResidualTrainingCallbacks(DefaultCallbacks):
    """Remove inactive absorber samples from the PPO actor objective.

    The critic still sees every traffic state. Centering the active advantages
    before zeroing inactive rows keeps the later RLlib batch standardization
    from turning the zeros back into nonzero policy-gradient weights.
    """

    @staticmethod
    def _episode_store(episode) -> dict:
        return episode.user_data.setdefault(
            "idm_training",
            {
                "last_info_step": None,
                "steps": 0,
                "sums": {},
                "means": {},
                "reward_components": {},
                "minimum_ttc_s": float("inf"),
                "policy": {},
            },
        )

    def on_episode_start(self, *, episode, **kwargs) -> None:
        del kwargs
        self._episode_store(episode)

    def on_episode_step(self, *, episode, **kwargs) -> None:
        info = None
        base_env = kwargs.get("base_env")
        env_index = int(kwargs.get("env_index", 0) or 0)
        if base_env is not None:
            try:
                sub_env = base_env.get_sub_environments()[env_index]
                core_env = getattr(sub_env, "_env", sub_env)
                candidate = getattr(core_env, "_last_idm_training_metrics", None)
                if isinstance(candidate, dict):
                    info = candidate
            except (AttributeError, IndexError, TypeError):
                info = None
        # Fixture/compatibility fallback.  Do not use this as the primary path:
        # a first agent can retain stale shared info after its SUMO vehicle exits.
        for agent_id in episode.get_agents():
            if info is not None:
                break
            candidate = episode.last_info_for(agent_id)
            if isinstance(candidate, dict) and "idm_training_metrics" in candidate:
                info = candidate["idm_training_metrics"]
                break
        if not isinstance(info, dict):
            return
        store = self._episode_store(episode)
        step = info.get("step")
        if step == store["last_info_step"]:
            return
        store["last_info_step"] = step
        store["steps"] += 1
        for key in _IDM_EPISODE_SUM_FIELDS:
            value = info.get(key)
            if value is not None:
                store["sums"][key] = store["sums"].get(key, 0.0) + float(value)
        for key in _IDM_EPISODE_MEAN_FIELDS:
            value = info.get(key)
            if value is not None:
                total, count = store["means"].get(key, (0.0, 0))
                store["means"][key] = (total + float(value), count + 1)
        minimum_ttc = info.get("minimum_ttc_s")
        if minimum_ttc is not None and np.isfinite(float(minimum_ttc)):
            store["minimum_ttc_s"] = min(store["minimum_ttc_s"], float(minimum_ttc))
        for key, value in (info.get("reward_components") or {}).items():
            store["reward_components"][key] = (
                store["reward_components"].get(key, 0.0) + float(value)
            )

    def on_postprocess_trajectory(
        self, *, policies, policy_id, postprocessed_batch, **kwargs
    ) -> None:
        policy = policies[policy_id]
        custom = policy.config["model"]["custom_model_config"]
        gate_index = int(custom["idm_absorber_gate_index"])
        obs = np.asarray(postprocessed_batch[SampleBatch.OBS])
        advantages = np.asarray(
            postprocessed_batch[Postprocessing.ADVANTAGES], dtype=np.float32
        ).copy()
        decentralized = bool(
            int(custom.get("action_mask_dim", 0)) == 2
            and not bool(custom.get("force_action_zero_legal", True))
        )
        if decentralized:
            raw_local_dim = int(custom["raw_local_dim"])
            masks = obs[:, raw_local_dim : raw_local_dim + 2]
            active = np.all(masks > 0.5, axis=1)
            # A latched option has one legal action and therefore no decision.
            # Keep only true binary decision advantages; PPO's categorical mask
            # makes every other sample policy-gradient neutral.
            advantages[~active] = 0.0
            episode = kwargs.get("episode")
            if episode is not None:
                store = self._episode_store(episode)
                for key, value in _idm_decision_batch_metrics(
                    postprocessed_batch, custom
                ).items():
                    store["policy"][key] = store["policy"].get(key, 0.0) + value
        else:
            active = obs[:, gate_index] > 0.5
            if np.any(active):
                advantages[active] -= float(np.mean(advantages[active]))
            advantages[~active] = 0.0
        postprocessed_batch[Postprocessing.ADVANTAGES] = advantages

    def on_episode_end(self, *, episode, **kwargs) -> None:
        del kwargs
        store = self._episode_store(episode)
        values = dict(store["sums"])
        for key, (total, count) in store["means"].items():
            values[key] = total / max(count, 1)
        if store["minimum_ttc_s"] != float("inf"):
            values["minimum_ttc_s"] = store["minimum_ttc_s"]
        for key, value in store["reward_components"].items():
            values[f"reward_{key}"] = value
        values.update(store["policy"])
        eligible = values.get("policy_eligible_sample_count", 0.0)
        values["policy_activate_probability"] = (
            values.get("policy_activate_probability_sum", 0.0) / eligible
            if eligible > 0
            else float("nan")
        )
        for key, value in values.items():
            if np.isfinite(float(value)):
                episode.custom_metrics[f"idm_{key}"] = float(value)
                episode.hist_data.setdefault(f"idm_{key}", []).append(float(value))


def _build_env_config(args) -> dict:
    density = args.density
    density_choices = None
    if getattr(args, "curriculum_densities", None) is not None:
        density_choices = list(args.curriculum_densities)
        density = density_choices[0]
    elif getattr(args, "multi_density", False):
        density_choices = ["light", "medium", "dense", "hyper"]
    cfg = {
        "sumo_binary": args.sumo_binary,
        # --sumo-config overrides the per-density config (e.g. train on the
        # 2700 critical metastable scene _light/osm.dense.light_m2700r400.sumocfg).
        "sumo_config": (getattr(args, "sumo_config", None) or f"osm.{density}.sumocfg"),
        # Keep train penetration identical to eval (no silent dense p/2).
        "cav_penetration": args.cav_penetration,
        "max_episode_steps": args.max_episode_steps,
        "eval_mode": False,
        "max_neighbors": args.max_neighbors,
        "cav_warmup_steps": 400,
        "max_cavs_critic": args.max_cavs_critic,
        "fairness_r_ref": FAIRNESS_R_REF_BY_DENSITY.get(density, 0.5),
        "tail_r_ref": TAIL_R_REF_BY_DENSITY.get(density, 0.34),
        "flow_q_ref": FLOW_Q_REF_BY_DENSITY.get(density, 0.9),
        "flow_v_ref": FLOW_V_REF_BY_DENSITY.get(density, 27.0),
    }
    if density_choices is not None:
        # Env samples one density per episode reset (not once at algo build).
        cfg["density_choices"] = density_choices
    if getattr(args, "sumo_config_choices", None):
        cfg["sumo_config_choices"] = list(args.sumo_config_choices)
    if getattr(args, "scenario_refs", None):
        payload = json.loads(Path(args.scenario_refs).read_text(encoding="utf-8"))
        cfg["scenario_reward_refs"] = dict(payload.get("refs") or payload)
    if getattr(args, "penetration_choices", None):
        cfg["penetration_choices"] = [float(p) for p in args.penetration_choices]
    elif getattr(args, "multi_penetration", False):
        # Legacy broad randomization.
        cfg["penetration_choices"] = [0.05, 0.1, 0.2, 0.3]
    paired_refs_path = getattr(args, "paired_refs", None)
    if paired_refs_path:
        # Per-seed All-HDV reward baselines (fairness / flow q / flow v).
        # Env samples S from paired_seed_pool each reset and loads refs[S].
        payload = json.loads(Path(paired_refs_path).read_text(encoding="utf-8"))
        refs = payload.get("refs") or payload
        cfg["paired_hdv_refs"] = {str(k): dict(v) for k, v in refs.items()}
        # Prefer explicit pool; otherwise train on every seed that has refs.
        pool = payload.get("train_seeds")
        if pool:
            cfg["paired_seed_pool"] = [int(s) for s in pool]
        else:
            cfg["paired_seed_pool"] = sorted(int(k) for k in refs.keys())
        print(
            f"Paired refs: {paired_refs_path} "
            f"(n_refs={len(cfg['paired_hdv_refs'])}, "
            f"pool={len(cfg['paired_seed_pool'])})",
            flush=True,
        )
    if args.hdv_mixture is not None:
        # Explicit mixture: re-enable TraCI-side type assignment, overriding
        # the route file's vTypeDistribution (assign_hdv_types default False).
        cfg["hdv_type_proportions"] = list(args.hdv_mixture)
        cfg["assign_hdv_types"] = True
    if args.baseline == "selfish":
        cfg.update(
            {
                "control_mode": "continuous_acceleration",
                "reward_mode": "selfish",
                "use_v2_obs": True,
                # Independent PPO/IPPO baseline: actor and critic both receive
                # this CAV's local observation only.
                "centralized_critic": False,
            }
        )
    elif args.baseline == "flat_marl":
        cfg.update(
            {
                "control_mode": "continuous_acceleration",
                "reward_mode": "shepherd",
                "use_v2_obs": True,
                "centralized_critic": True,
                "acceleration_min": FLAT_MARL_ACCEL_MIN,
                "acceleration_max": FLAT_MARL_ACCEL_MAX,
            }
        )
    elif args.baseline == "shepherd":
        cfg.update(
            {
                "control_mode": "shepherd",
                "reward_mode": "shepherd",
                "use_v2_obs": True,
                "centralized_critic": True,
            }
        )
    else:
        raise ValueError(f"Unknown baseline: {args.baseline}")
    if getattr(args, "control_mode", None):
        cfg["control_mode"] = args.control_mode
    if getattr(args, "control_mode", None) == "idm_residual_smoothing":
        # Phase-4 contract: train a bounded residual around the analytical
        # disturbance_p30 controller. The actor sees both the disturbance
        # signal and the deterministic absorber gate.
        cfg["reward_mode"] = "idm_damping"
        cfg["idm_residual_reference_mode"] = "disturbance_p30"
        cfg["use_wave_signal"] = True
        cfg["use_idm_absorber_gate"] = True
        cfg["use_idm_controller_context"] = True
        cfg["idm_pattern_short_tau_s"] = 10.0
        cfg["idm_pattern_long_tau_s"] = 30.0
        cfg["idm_residual_filter_tau_s"] = float(getattr(args, "idm_residual_filter_tau", 0.0))
        cfg["idm_residual_decentralized_damping_enabled"] = bool(
            getattr(args, "idm_decentralized_modes", False)
        )
        cfg["idm_residual_decentralized_action_set"] = str(
            getattr(args, "idm_decentralized_action_set", "gain_reference_v1")
        )
        # Freeze the audited IDM-only weights into the run/checkpoint config;
        # do not rely on an implicit environment default.
        from config.defaults import DEFAULT_CONFIG as _IDM_DEFAULTS

        cfg["idm_damping_reward_weights"] = dict(_IDM_DEFAULTS["idm_damping_reward_weights"])
    # Single-variable experiment knobs (2026-08-05): fair_balance tip zone and
    # corridor-TTC penalty toggle, so warm runs can differ from DEFAULT_CONFIG
    # in exactly one dimension (shallow merge: reward_weights replaced whole).
    if getattr(args, "disable_modes", None):
        cfg["shepherd_disabled_modes"] = [
            m.strip() for m in str(args.disable_modes).split(",") if m.strip()
        ]
    if getattr(args, "enable_wave_damp", False):
        # Opt-in 6th mode (research_v2 flow_stabilization port): active wave
        # damping. Sizes the action space + mask to 6 for this run only.
        modes = list(cfg.get("shepherd_modes") or [])
        if "wave_damp" not in modes:
            modes.append("wave_damp")
        cfg["shepherd_modes"] = modes
    if getattr(args, "enable_merge_match", False):
        # Opt-in 7th mode (merge-stream smoothing, global index 6). Index 6 is
        # only reachable when wave_damp (5) is also in the set, so a 7-mode
        # action space: [.. protect, wave_damp, merge_match].
        modes = list(cfg.get("shepherd_modes") or [])
        if "wave_damp" not in modes:
            modes.append("wave_damp")
        if "merge_match" not in modes:
            modes.append("merge_match")
        cfg["shepherd_modes"] = modes
    if getattr(args, "enable_jad", False):
        # Opt-in JAD modes (global indices 7,8) reachable only with the index
        # fillers (wave_damp=5, merge_match=6) present -> 9-mode action space:
        # [.. protect, wave_damp, merge_match, jad_absorb, jad_support].
        modes = list(cfg.get("shepherd_modes") or [])
        for filler in ("wave_damp", "merge_match", "jad_absorb", "jad_support"):
            if filler not in modes:
                modes.append(filler)
        cfg["shepherd_modes"] = modes
    if getattr(args, "pruned_modes", False):
        # Pruned champion (2026-08-20): remove the modes that don't work at
        # dense demand — fair_balance (metering leaks: isolated CAV braking
        # cannot throttle inflow; the HDV follower keeps its own tau and
        # re-closes the gap — the central finding of the wall program) and
        # vacate_drop (disabled in every probe since the autopsies).
        # Discrete(3): coast · recover_speed · protect — the merge-assist
        # modes behind the champion's medium-density wins, retrained clean
        # as the paper's baseline. The env passes this list to the engine as
        # its index->mode mapping (no prefix-of-registry assumption).
        cfg["shepherd_modes"] = ["coast", "recover", "guard"]
    if getattr(args, "use_wave_signal", False):
        # RSU compound shockwave detector feature (2 slots) in the local obs.
        cfg["use_wave_signal"] = True
    if getattr(args, "fair_balance_tip", None) is not None:
        cfg["fair_balance_tip"] = float(args.fair_balance_tip)
    if getattr(args, "no_corridor_ttc_penalty", False):
        from config.defaults import DEFAULT_CONFIG as _DC

        rw = dict(_DC["reward_weights"])
        rw["corridor_ttc_penalty"] = 0.0
        cfg["reward_weights"] = rw
    if getattr(args, "no_easy_intervention_mask", False):
        # Experiment B (hybrid, 2026-08-11): in the metastable regime metering
        # (fair_balance) is the PRIMARY density-control lever. The default
        # easy-intervention soft mask discounts fair_balance exactly when the
        # corridor is already fair — i.e. the calm window where we WANT the
        # policy free to meter to hold rho below the metastable ceiling.
        cfg["shepherd_easy_intervention_mask"] = False
    if getattr(args, "metastable_health_mask", False):
        # Experiment C (2026-08-11): RSU metastable-health graduated soft mask
        # on the discretionary modes. Mutually exclusive intent with the
        # easy-intervention mask (which gates on fairness instead of health);
        # when both are enabled the health mask is the Experiment-C signal and
        # the easy mask is disabled to keep the ablation clean.
        cfg["metastable_health_mask"] = True
        cfg["shepherd_easy_intervention_mask"] = False
    if getattr(args, "health_mask_includes_protect", False):
        # Retry (2026-08-12): also gate protect — at healthy h the policy is
        # pushed to coast (protect discounted like metering/recover); when the
        # merge sickens, protect opens alongside fair_balance/recover so the
        # TTC-avoiding lever is fully available in the regime that needs it.
        modes = list(cfg.get("metastable_health_mask_modes") or ["fair_balance", "recover"])
        if "guard" not in modes:
            modes.append("guard")
        cfg["metastable_health_mask_modes"] = modes
    if getattr(args, "reward_weights", None):
        # Generic reward override, e.g. "safety=3.0,delay_budget=0.5,flow=0.6"
        # (campaign-4 TTC-first pivot). Applied on top of DEFAULT_CONFIG.
        from config.defaults import DEFAULT_CONFIG as _DC

        rw = dict(cfg.get("reward_weights") or _DC["reward_weights"])
        for pair in str(args.reward_weights).split(","):
            key, _, value = pair.strip().partition("=")
            if key and value:
                rw[key.strip()] = float(value)
        cfg["reward_weights"] = rw
    if getattr(args, "sumo_root", None):
        from pathlib import Path as _P

        cfg["project_root"] = _P(args.sumo_root)
    if getattr(args, "zone_comm_range", None) is not None:
        cfg["zone_comm_range"] = float(args.zone_comm_range)
    return cfg


def _obs_action_spaces(args, centralized: bool):
    from config.constants import NUM_MODES  # noqa: F401 (registry size)
    from config.defaults import DEFAULT_CONFIG

    n_modes = (
        3 if getattr(args, "pruned_modes", False) else len(DEFAULT_CONFIG["shepherd_modes"])
    ) + (
        4
        if getattr(args, "enable_jad", False)
        else 2
        if getattr(args, "enable_merge_match", False)
        else 1
        if getattr(args, "enable_wave_damp", False)
        else 0
    )
    local_dim = raw_obs_dim(
        args.max_neighbors,
        include_meter_load=bool(getattr(args, "use_meter_load", False)),
        include_cav_count=bool(getattr(args, "use_cav_count", False)),
        include_wave_signal=bool(
            getattr(args, "use_wave_signal", False)
            or getattr(args, "control_mode", None) == "idm_residual_smoothing"
        ),
        include_health=bool(getattr(args, "use_health_obs", False)),
        include_dv=bool(getattr(args, "use_dv_obs", False)),
        include_margin=bool(getattr(args, "use_margin_obs", False)),
        include_idm_absorber_gate=bool(
            getattr(args, "control_mode", None) == "idm_residual_smoothing"
        ),
        include_idm_controller_context=bool(
            getattr(args, "control_mode", None) == "idm_residual_smoothing"
        ),
    )
    decentralized_idm = bool(
        getattr(args, "control_mode", None) == "idm_residual_smoothing"
        and getattr(args, "idm_decentralized_modes", False)
    )
    decentralized_idm_mode_count = (
        4
        if getattr(args, "idm_decentralized_action_set", "gain_reference_v1")
        == "acceleration_caps_v1"
        else 2
        if getattr(args, "idm_decentralized_action_set", "gain_reference_v1")
        in {
            "activation_only_v1",
            "activation_speed_ceiling_v1",
            "activation_speed_ceiling_p50_v1",
            "activation_speed_ceiling_p70_v1",
        }
        else 5
    )
    action_mask_dim = 0
    # Shepherd Discrete + action_masking: append eligibility bits to local obs.
    if (
        args.baseline == "shepherd"
        and getattr(args, "control_mode", None)
        not in ("continuous_acceleration", "idm_residual_smoothing")
        and bool(DEFAULT_CONFIG.get("action_masking", False))
    ):
        action_mask_dim = n_modes
        local_dim += action_mask_dim
    elif decentralized_idm:
        action_mask_dim = decentralized_idm_mode_count
        local_dim += action_mask_dim
    obs_dim = mappo_obs_dim(local_dim, args.max_cavs_critic) if centralized else local_dim
    obs_space = gym.spaces.Box(
        low=-float("inf"), high=float("inf"), shape=(obs_dim,), dtype="float32"
    )
    if args.baseline == "flat_marl":
        act_space = gym.spaces.Box(
            low=FLAT_MARL_ACCEL_MIN,
            high=FLAT_MARL_ACCEL_MAX,
            shape=(1,),
            dtype="float32",
        )
    elif args.baseline == "selfish":
        act_space = gym.spaces.Box(
            low=ACCELERATION_MIN,
            high=ACCELERATION_MAX,
            shape=(1,),
            dtype="float32",
        )
    elif decentralized_idm:
        act_space = gym.spaces.Discrete(decentralized_idm_mode_count)
    elif getattr(args, "control_mode", None) == "idm_residual_smoothing":
        act_space = gym.spaces.Box(low=-0.5, high=0.3, shape=(1,), dtype="float32")
    elif getattr(args, "control_mode", None) == "shepherd_hybrid":
        # Hybrid composite: Categorical(n_modes) x Gaussian dial. RLlib
        # auto-builds the multi distribution for the Tuple space.
        act_space = gym.spaces.Tuple(
            (
                gym.spaces.Discrete(n_modes),
                gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype="float32"),
            )
        )
    else:
        act_space = gym.spaces.Discrete(n_modes)
    return obs_space, act_space, local_dim, action_mask_dim


def _context_dim_from_args(args) -> int:
    from env.features import context_feature_names

    return len(
        context_feature_names(
            include_meter_load=bool(getattr(args, "use_meter_load", False)),
            include_cav_count=bool(getattr(args, "use_cav_count", False)),
            include_wave_signal=bool(
                getattr(args, "use_wave_signal", False)
                or getattr(args, "control_mode", None) == "idm_residual_smoothing"
            ),
            include_health=bool(getattr(args, "use_health_obs", False)),
            include_dv=bool(getattr(args, "use_dv_obs", False)),
            include_margin=bool(getattr(args, "use_margin_obs", False)),
            include_idm_absorber_gate=bool(
                getattr(args, "control_mode", None) == "idm_residual_smoothing"
            ),
            include_idm_controller_context=bool(
                getattr(args, "control_mode", None) == "idm_residual_smoothing"
            ),
        )
    )


def _idm_absorber_gate_index(args) -> int:
    """Return the actor-local index of the deterministic absorber gate."""
    from env.features import context_feature_names

    names = context_feature_names(
        include_meter_load=bool(getattr(args, "use_meter_load", False)),
        include_cav_count=bool(getattr(args, "use_cav_count", False)),
        include_wave_signal=bool(
            getattr(args, "use_wave_signal", False)
            or getattr(args, "control_mode", None) == "idm_residual_smoothing"
        ),
        include_health=bool(getattr(args, "use_health_obs", False)),
        include_dv=bool(getattr(args, "use_dv_obs", False)),
        include_margin=bool(getattr(args, "use_margin_obs", False)),
        include_idm_absorber_gate=True,
        include_idm_controller_context=bool(
            getattr(args, "control_mode", None) == "idm_residual_smoothing"
        ),
    )
    return raw_obs_dim(args.max_neighbors) + names.index("idm_absorber_gate")


def _rollout_fragment_length(args) -> int | str:
    """Match train_batch_size to num_env_runners (RLlib validates this)."""
    raw = getattr(args, "rollout_fragment_length", "auto")
    if isinstance(raw, str) and raw.lower() == "auto":
        return "auto"
    return int(raw)


def _as_float(value, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # Windows os.kill(pid, 0) is not the POSIX no-op probe: CPython may
        # route it through TerminateProcess.  Query a process handle instead.
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information, False, int(pid)
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(
                handle, ctypes.byref(exit_code)
            ):
                return False
            return int(exit_code.value) == 259  # STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _acquire_train_lock(logdir: Path) -> Path:
    """Prevent two train.py parents from writing the same run directory."""
    path = logdir / TRAIN_LOCK_NAME
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            owner_pid = int(payload.get("pid", -1))
        except (OSError, ValueError, json.JSONDecodeError):
            owner_pid = -1
        if _pid_is_alive(owner_pid):
            raise RuntimeError(
                f"Refusing concurrent trainer for {logdir}: PID {owner_pid} owns {path}"
            )
        path.unlink(missing_ok=True)
    payload = {"pid": os.getpid(), "created_unix_s": time.time(), "logdir": str(logdir)}
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def _release_train_lock(path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("pid", -1)) == os.getpid():
            path.unlink(missing_ok=True)
    except (OSError, ValueError, json.JSONDecodeError):
        pass


def _extract_train_metrics(result: dict) -> dict:
    """Pull the key scalars we care about from an RLlib train() result."""
    ep_stats = result.get("env_runners", {}) or {}
    policy_means = ep_stats.get("policy_reward_mean") or {}
    if isinstance(policy_means, dict):
        policy_reward = policy_means.get(POLICY_ID, policy_means.get("cav_policy"))
    else:
        policy_reward = None

    episode_return = ep_stats.get(
        "episode_return_mean",
        ep_stats.get(
            "episode_reward_mean",
            result.get("episode_reward_mean"),
        ),
    )
    timesteps = int(
        result.get("num_env_steps_sampled_lifetime")
        or result.get("timesteps_total")
        or result.get("num_env_steps_sampled")
        or result.get("info", {}).get("num_env_steps_sampled", 0)
        or result.get("counters", {}).get("num_env_steps_sampled", 0)
        or 0
    )
    restarts = int(
        result.get("num_remote_worker_restarts")
        or result.get("counters", {}).get("total_num_restored_workers", 0)
        or 0
    )
    learner_stats = (
        result.get("info", {})
        .get("learner", {})
        .get(POLICY_ID, {})
        .get("learner_stats", {})
        or {}
    )
    hist_stats = ep_stats.get("hist_stats", {}) or {}
    custom_metrics = ep_stats.get("custom_metrics", {}) or {}
    completed_episodes = int(ep_stats.get("num_episodes") or 0)

    def hist_values(key: str) -> list[float]:
        if completed_episodes <= 0:
            return []
        values = hist_stats.get(f"idm_{key}", ()) or ()
        return [float(value) for value in values if value is not None]

    def episode_sum(key: str) -> float:
        if completed_episodes <= 0:
            return 0.0
        values = hist_values(key)
        if values:
            return float(sum(values))
        return _as_float(custom_metrics.get(f"idm_{key}_mean"), 0.0)

    def episode_mean(key: str) -> float:
        if completed_episodes <= 0:
            return float("nan")
        values = hist_values(key)
        if values:
            return float(np.mean(values))
        return _as_float(custom_metrics.get(f"idm_{key}_mean"))

    eligible_samples = episode_sum("policy_eligible_sample_count")
    activation_probability_sum = episode_sum("policy_activate_probability_sum")
    diagnostics = {
        "policy_activate_probability": (
            activation_probability_sum / eligible_samples
            if eligible_samples > 0
            else float("nan")
        ),
        "policy_abstain_probability": (
            1.0 - activation_probability_sum / eligible_samples
            if eligible_samples > 0
            else float("nan")
        ),
        "policy_eligible_sample_count": eligible_samples,
        "policy_masked_sample_count": episode_sum("policy_masked_sample_count"),
        "policy_activate_executed_count": episode_sum(
            "policy_activate_executed_count"
        ),
        "policy_abstain_executed_count": episode_sum("policy_abstain_executed_count"),
    }
    for key in _IDM_EPISODE_SUM_FIELDS:
        diagnostics[key] = episode_sum(key)
    for key in _IDM_EPISODE_MEAN_FIELDS:
        diagnostics[key] = episode_mean(key)
    diagnostics["minimum_ttc_s"] = episode_mean("minimum_ttc_s")
    for key in (
        "local",
        "efficiency",
        "delay_budget",
        "corridor_ttc",
        "corridor_severe_braking",
        "corridor_wave_energy",
        "wave",
        "residual_effort",
    ):
        diagnostics[f"reward_{key}"] = episode_sum(f"reward_{key}")

    return {
        "timesteps": timesteps,
        "episode_return_mean": _as_float(episode_return),
        "policy_reward_mean": _as_float(policy_reward),
        "episode_len_mean": _as_float(ep_stats.get("episode_len_mean")),
        "num_episodes": completed_episodes,
        "worker_restarts": restarts,
        "entropy": _as_float(learner_stats.get("entropy")),
        "kl": _as_float(learner_stats.get("kl")),
        "policy_loss": _as_float(learner_stats.get("policy_loss")),
        "value_loss": _as_float(learner_stats.get("vf_loss")),
        "value_explained_variance": _as_float(
            learner_stats.get("vf_explained_var")
        ),
        **diagnostics,
    }


def _append_metrics(logdir: Path, row: dict) -> None:
    """Append one training iteration to metrics.jsonl and metrics.csv."""
    jsonl_path = logdir / "metrics.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")

    csv_path = logdir / "metrics.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _make_tb_writer(logdir: Path):
    """Create a TensorBoard writer under ``{logdir}/tb``, or None if unavailable."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            print(
                "TensorBoard not available (install tensorboard). "
                "Continuing with CSV/JSONL metrics only."
            )
            return None
    tb_dir = logdir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(tb_dir))


def _log_tb(writer, row: dict) -> None:
    """Log one training-iteration row to TensorBoard (x-axis = env timesteps)."""
    if writer is None:
        return
    step = int(row.get("timesteps") or row.get("iteration") or 0)
    for key, val in row.items():
        if key in {"iteration", "timesteps", "total_timesteps", "steps_remaining"}:
            continue
        if val is None:
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if fval != fval:  # NaN
            continue
        writer.add_scalar(f"train/{key}", fval, step)
    writer.flush()


def _load_train_state(logdir: Path) -> dict:
    path = logdir / TRAIN_STATE_NAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_train_state(logdir: Path, state: dict) -> None:
    path = logdir / TRAIN_STATE_NAME
    path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def _format_duration(seconds: float) -> str:
    if seconds != seconds or seconds == float("inf"):
        return "?"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _write_progress(
    logdir: Path,
    iteration: int,
    timesteps: int,
    total_timesteps: int,
    best_reward: float,
    steps_per_sec: float | None = None,
    eta_s: float | None = None,
) -> None:
    """Overwrite a small progress.txt so status is visible without scrolling logs."""
    pct = 100.0 * timesteps / max(total_timesteps, 1)
    bar_w = 28
    filled = int(bar_w * min(timesteps, total_timesteps) / max(total_timesteps, 1))
    bar = "#" * filled + "-" * (bar_w - filled)
    lines = [
        f"iteration:        {iteration}",
        f"timesteps:        {timesteps:,} / {total_timesteps:,}  ({pct:.1f}%)",
        f"progress:         [{bar}]",
        f"steps_remaining:  {max(total_timesteps - timesteps, 0):,}",
    ]
    if steps_per_sec is not None:
        lines.append(f"throughput:       {steps_per_sec:.1f} env steps/s")
    if eta_s is not None:
        lines.append(f"eta:              {_format_duration(eta_s)}")
    if best_reward == best_reward and best_reward != float("-inf"):
        lines.append(f"best_pol_reward:  {best_reward:.4f}")
    (logdir / "progress.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _find_latest_checkpoint(ckpt_root: Path) -> Path | None:
    """Pick the newest training checkpoint under checkpoints/."""
    if not ckpt_root.is_dir():
        return None

    iter_dirs = []
    for child in ckpt_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith("iter_"):
            try:
                n = int(name.split("_", 1)[1])
            except ValueError:
                continue
            if is_rllib_checkpoint(child):
                iter_dirs.append((n, child))

    if iter_dirs:
        iter_dirs.sort(key=lambda x: x[0])
        return resolve_rllib_checkpoint_dir(iter_dirs[-1][1])

    for name in ("final", "best"):
        candidate = ckpt_root / name
        if is_rllib_checkpoint(candidate):
            return resolve_rllib_checkpoint_dir(candidate)
    return None


def _resolve_resume_checkpoint(args, logdir: Path) -> Path | None:
    """Return checkpoint dir to restore, or None for a fresh run."""
    explicit = getattr(args, "from_checkpoint", None)
    if explicit:
        path = resolve_rllib_checkpoint_dir(explicit)
        if not is_rllib_checkpoint(path):
            raise FileNotFoundError(f"--from-checkpoint is not a valid RLlib checkpoint: {path}")
        return path

    if not getattr(args, "resume", False):
        return None

    found = _find_latest_checkpoint(logdir / "checkpoints")
    if found is None:
        raise FileNotFoundError(
            f"--resume set but no checkpoint found under {logdir / 'checkpoints'}. "
            "Train once without --resume, or pass --from-checkpoint PATH."
        )
    return found


def _algo_timesteps(algo) -> int:
    counters = getattr(algo, "_counters", None) or {}
    for key in (
        "num_env_steps_sampled",
        "num_env_steps_sampled_lifetime",
    ):
        if key in counters and counters[key]:
            return int(counters[key])
    for attr in ("_timesteps_total",):
        val = getattr(algo, attr, None)
        if val:
            return int(val)
    return 0


def _algo_iteration(algo) -> int:
    for attr in ("_iteration", "iteration", "training_iteration"):
        val = getattr(algo, attr, None)
        if val:
            return int(val)
    return 0


def build_selfish_ppo_config(args) -> PPOConfig:
    """Independent PPO/IPPO: shared policy, local actor, and local critic."""
    register_selfish_ppo_model()
    register_env("traffic_shepherd_ma", make_traffic_shepherd_env)

    obs_space, act_space, local_dim, action_mask_dim = _obs_action_spaces(
        args, centralized=False
    )
    env_config = _build_env_config(args)
    env_config["local_obs_dim"] = local_dim
    env_config["action_masking"] = action_mask_dim > 0
    decentralized_idm = bool(
        getattr(args, "control_mode", None) == "idm_residual_smoothing"
        and getattr(args, "idm_decentralized_modes", False)
    )

    def policy_mapping_fn(agent_id, *a, **k):
        return POLICY_ID

    return (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env="traffic_shepherd_ma",
            env_config=env_config,
            normalize_actions=not (getattr(args, "control_mode", None) == "idm_residual_smoothing"),
            clip_actions=bool(
                getattr(args, "control_mode", None) == "idm_residual_smoothing"
                and not decentralized_idm
            ),
        )
        .framework("torch")
        .callbacks(
            IDMResidualTrainingCallbacks
            if getattr(args, "control_mode", None) == "idm_residual_smoothing"
            else DefaultCallbacks
        )
        .env_runners(
            num_env_runners=args.num_env_runners,
            num_envs_per_env_runner=1,
            rollout_fragment_length=_rollout_fragment_length(args),
            sample_timeout_s=600.0,
            create_local_env_runner=False,
            batch_mode="truncate_episodes",
        )
        .resources(num_gpus=args.num_gpus)
        .multi_agent(
            policies={POLICY_ID: (None, obs_space, act_space, {})},
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=[POLICY_ID],
        )
        .training(
            lr=args.lr,
            gamma=args.gamma,
            lambda_=args.gae_lambda,
            clip_param=args.clip_eps,
            train_batch_size=args.train_batch_size,
            minibatch_size=args.sgd_minibatch_size,
            num_epochs=args.num_sgd_iter,
            entropy_coeff=args.entropy_coeff,
            vf_clip_param=args.vf_clip_param,
            model={
                "custom_model": SELFISH_MODEL_ID,
                "custom_model_config": {
                    "max_neighbors": args.max_neighbors,
                    "hidden": args.hidden,
                    "local_obs_dim": local_dim,
                    "action_mask_dim": action_mask_dim,
                    "raw_local_dim": local_dim - action_mask_dim,
                    "force_action_zero_legal": not decentralized_idm,
                    "context_dim": _context_dim_from_args(args),
                    "zero_mean_init": bool(
                        getattr(args, "control_mode", None) == "idm_residual_smoothing"
                    ),
                    "idm_absorber_gate_index": (
                        _idm_absorber_gate_index(args)
                        if getattr(args, "control_mode", None) == "idm_residual_smoothing"
                        else None
                    ),
                },
            },
        )
    )


def build_mappo_config(args) -> PPOConfig:
    """MAPPO: decentralized actor plus centralized per-component pooled critic."""
    register_mappo_model()
    register_env("traffic_shepherd_ma", make_traffic_shepherd_env)

    obs_space, act_space, local_dim, action_mask_dim = _obs_action_spaces(args, centralized=True)
    env_config = _build_env_config(args)
    # Keep env local vector size aligned with the policy obs space.
    env_config["local_obs_dim"] = local_dim
    env_config["action_masking"] = action_mask_dim > 0
    decentralized_idm = bool(
        getattr(args, "control_mode", None) == "idm_residual_smoothing"
        and getattr(args, "idm_decentralized_modes", False)
    )
    env_config["use_meter_load"] = bool(getattr(args, "use_meter_load", False))
    env_config["use_cav_count"] = bool(getattr(args, "use_cav_count", False))
    env_config["use_wave_signal"] = bool(
        getattr(args, "use_wave_signal", False)
        or getattr(args, "control_mode", None) == "idm_residual_smoothing"
    )
    env_config["use_idm_absorber_gate"] = bool(
        getattr(args, "control_mode", None) == "idm_residual_smoothing"
    )
    env_config["use_idm_controller_context"] = bool(
        getattr(args, "control_mode", None) == "idm_residual_smoothing"
    )
    env_config["use_health_obs"] = bool(getattr(args, "use_health_obs", False))
    env_config["use_dv_obs"] = bool(getattr(args, "use_dv_obs", False))
    env_config["use_margin_obs"] = bool(getattr(args, "use_margin_obs", False))
    env_config["feasibility_mask"] = bool(getattr(args, "feasibility_mask", False))
    env_config["feasibility_lmin_a"] = float(getattr(args, "feasibility_lmin_a", 0.0))
    env_config["feasibility_lmin_b"] = float(getattr(args, "feasibility_lmin_b", 0.0))

    def policy_mapping_fn(agent_id, *a, **k):
        return POLICY_ID

    return (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env="traffic_shepherd_ma",
            env_config=env_config,
            normalize_actions=not (getattr(args, "control_mode", None) == "idm_residual_smoothing"),
            clip_actions=bool(
                getattr(args, "control_mode", None) == "idm_residual_smoothing"
                and not decentralized_idm
            ),
        )
        .framework("torch")
        .callbacks(
            IDMResidualTrainingCallbacks
            if getattr(args, "control_mode", None) == "idm_residual_smoothing"
            else DefaultCallbacks
        )
        .env_runners(
            num_env_runners=args.num_env_runners,
            num_envs_per_env_runner=1,
            rollout_fragment_length=_rollout_fragment_length(args),
            sample_timeout_s=600.0,
            create_local_env_runner=False,
            batch_mode="truncate_episodes",
        )
        .resources(num_gpus=args.num_gpus)
        .multi_agent(
            policies={POLICY_ID: (None, obs_space, act_space, {})},
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=[POLICY_ID],
        )
        .training(
            lr=args.lr,
            gamma=args.gamma,
            lambda_=args.gae_lambda,
            clip_param=args.clip_eps,
            train_batch_size=args.train_batch_size,
            minibatch_size=args.sgd_minibatch_size,
            num_epochs=args.num_sgd_iter,
            entropy_coeff=args.entropy_coeff,
            vf_clip_param=args.vf_clip_param,
            model={
                "custom_model": MAPPO_MODEL_ID,
                "custom_model_config": {
                    "max_neighbors": args.max_neighbors,
                    "hidden": args.hidden,
                    "max_cavs_critic": args.max_cavs_critic,
                    "local_obs_dim": local_dim,
                    "action_mask_dim": action_mask_dim,
                    "raw_local_dim": local_dim - action_mask_dim,
                    "force_action_zero_legal": not decentralized_idm,
                    "context_dim": _context_dim_from_args(args),
                    "zero_mean_init": bool(
                        getattr(args, "control_mode", None) == "idm_residual_smoothing"
                    ),
                    "idm_absorber_gate_index": (
                        _idm_absorber_gate_index(args)
                        if getattr(args, "control_mode", None) == "idm_residual_smoothing"
                        else None
                    ),
                    "hybrid_dial_init": float(getattr(args, "hybrid_dial_init", -0.7)),
                },
            },
        )
    )


def train(args):
    logdir = Path(args.logdir).resolve()
    logdir.mkdir(parents=True, exist_ok=True)
    train_lock = _acquire_train_lock(logdir)
    atexit.register(_release_train_lock, train_lock)
    manifest_path = logdir / "run_manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(
            json.dumps(vars(args), indent=2, default=str, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    project_root = str(Path(__file__).resolve().parent)
    env_vars = {
        "PYTHONPATH": project_root + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    if os.environ.get("SUMO_HOME"):
        env_vars["SUMO_HOME"] = os.environ["SUMO_HOME"]
        sumo_bin = str(Path(os.environ["SUMO_HOME"]) / "bin")
        env_vars["PATH"] = sumo_bin + os.pathsep + os.environ.get("PATH", "")

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        runtime_env={"env_vars": env_vars},
    )

    if args.baseline == "selfish":
        config = build_selfish_ppo_config(args)
    elif args.baseline in ("flat_marl", "shepherd"):
        config = build_mappo_config(args)
    else:
        raise ValueError(f"Unknown baseline: {args.baseline}")

    os.environ["RAY_AIR_LOCAL_CACHE_DIR"] = str(logdir)
    os.environ["TUNE_RESULT_DIR"] = str(logdir)
    config = config.debugging(seed=args.seed)

    resume_ckpt = _resolve_resume_checkpoint(args, logdir)
    algo = config.build_algo()
    print(f"Algorithm logdir: {getattr(algo, 'logdir', logdir)}")

    ckpt_root = logdir / "checkpoints"
    ckpt_root.mkdir(exist_ok=True)
    train_state = _load_train_state(logdir)
    best_reward = float(train_state.get("best_reward", float("-inf")))
    timesteps = 0
    iteration = 0

    if resume_ckpt is not None:
        print(f"Resuming from checkpoint: {resume_ckpt}")
        algo.restore(str(resume_ckpt))
        timesteps = max(
            _algo_timesteps(algo),
            int(train_state.get("timesteps", 0) or 0),
        )
        iteration = max(
            _algo_iteration(algo),
            int(train_state.get("iteration", 0) or 0),
        )
        print(
            f"  restored state | iteration={iteration} | timesteps={timesteps} | "
            f"best_reward={best_reward}"
        )
        if timesteps >= args.total_timesteps:
            print(
                f"Already at {timesteps} >= --total-timesteps {args.total_timesteps}. "
                "Raise --total-timesteps to continue, or omit --resume for a fresh run."
            )
            algo.stop()
            ray.shutdown()
            _release_train_lock(train_lock)
            return
    else:
        initial_path = algo.save(str(ckpt_root / "step_0"))
        print(f"Saved frozen initialization checkpoint -> {initial_path}")

    print(
        f"Training {args.baseline} with {args.num_env_runners} env runners | "
        f"progress {timesteps:,} / {args.total_timesteps:,} "
        f"({100.0 * timesteps / max(args.total_timesteps, 1):.1f}%)"
    )
    print(f"Results / metrics under: {logdir}")
    print(f"TensorBoard: tensorboard --logdir {logdir / 'tb'}")
    _write_progress(logdir, iteration, timesteps, args.total_timesteps, best_reward)

    tb_writer = _make_tb_writer(logdir)
    stalled_iters = 0
    last_t = time.time()
    last_steps = timesteps
    try:
        while timesteps < args.total_timesteps:
            result = algo.train()
            iteration += 1
            prev_timesteps = timesteps
            metrics = _extract_train_metrics(result)
            # Lifetime counters after restore should keep growing; never go backwards.
            timesteps = max(timesteps, metrics["timesteps"], _algo_timesteps(algo))
            ep_reward = metrics["episode_return_mean"]
            policy_reward = metrics["policy_reward_mean"]

            now = time.time()
            dt = max(now - last_t, 1e-6)
            dsteps = max(timesteps - last_steps, 0)
            steps_per_sec = dsteps / dt
            remaining = max(args.total_timesteps - timesteps, 0)
            eta_s = remaining / steps_per_sec if steps_per_sec > 0 else float("inf")
            pct = 100.0 * timesteps / max(args.total_timesteps, 1)
            last_t = now
            last_steps = timesteps

            # In-train "best" uses policy_reward_mean as a cheap proxy. For
            # fair+safe final picks, prefer the per-seed scorecard
            # (tools/select_checkpoint.py / talk-seed eval vs paired no_op).
            score = policy_reward if policy_reward == policy_reward else ep_reward

            row = {
                "iteration": iteration,
                **metrics,
                "timesteps": timesteps,
                "total_timesteps": args.total_timesteps,
                "progress_pct": round(pct, 2),
                "steps_remaining": remaining,
                "steps_per_sec": round(steps_per_sec, 2),
                "best_score": score if score == score else None,
            }
            _append_metrics(logdir, row)
            _log_tb(tb_writer, row)
            _write_progress(
                logdir,
                iteration,
                timesteps,
                args.total_timesteps,
                best_reward if best_reward != float("-inf") else score,
                steps_per_sec=steps_per_sec,
                eta_s=eta_s,
            )

            eta_str = _format_duration(eta_s)
            print(
                f"Iter {iteration} | "
                f"steps {timesteps:,}/{args.total_timesteps:,} ({pct:.1f}%) | "
                f"left {remaining:,} | "
                f"{steps_per_sec:.1f} steps/s | ETA {eta_str} | "
                f"ep_ret={ep_reward:.3f} | "
                f"pol_ret={policy_reward:.3f} | "
                f"restarts={metrics['worker_restarts']}"
            )

            if timesteps <= prev_timesteps:
                stalled_iters += 1
                if stalled_iters >= 5:
                    raise RuntimeError(
                        "Training stalled: no env steps sampled for 5 iterations. "
                        "Check SUMO/TraCI worker logs (ports, empty CAV episodes)."
                    )
            else:
                stalled_iters = 0

            if score == score and score > best_reward:
                best_reward = float(score)
                path = algo.save(str(ckpt_root / "best"))
                print(f"  saved best checkpoint (policy_reward_mean={best_reward:.3f}) -> {path}")

            if iteration % args.save_every == 0:
                path = algo.save(str(ckpt_root / f"iter_{iteration}"))
                print(f"  saved checkpoint -> {path}")

            _save_train_state(
                logdir,
                {
                    "baseline": args.baseline,
                    "iteration": iteration,
                    "timesteps": timesteps,
                    "best_reward": best_reward,
                    "total_timesteps_target": args.total_timesteps,
                },
            )

        final = algo.save(str(ckpt_root / "final"))
        _save_train_state(
            logdir,
            {
                "baseline": args.baseline,
                "iteration": iteration,
                "timesteps": timesteps,
                "best_reward": best_reward,
                "total_timesteps_target": args.total_timesteps,
                "final_checkpoint": str(ckpt_root / "final"),
            },
        )
        print(f"Training complete. Final checkpoint: {final}")
        print(f"Metrics: {logdir / 'metrics.csv'}")
        print(f"TensorBoard: tensorboard --logdir {logdir / 'tb'}")
    finally:
        if tb_writer is not None:
            tb_writer.close()
        algo.stop()
        ray.shutdown()
        _release_train_lock(train_lock)


def main():
    parser = argparse.ArgumentParser(
        description="RLlib parallel training for Traffic Shepherd baselines"
    )
    parser.add_argument(
        "--baseline",
        default="selfish",
        choices=["selfish", "flat_marl", "shepherd"],
        help="selfish=IPPO; flat_marl/shepherd=MAPPO (shared features, Shepherd reward).",
    )
    parser.add_argument(
        "--density", default="medium", choices=["light", "medium", "dense", "hyper"]
    )
    parser.add_argument(
        "--multi-density",
        action="store_true",
        help=(
            "Sample density (light/medium/dense) each episode inside the env "
            "(density_choices), not once at config build."
        ),
    )
    parser.add_argument(
        "--scenario-refs",
        default=None,
        help="JSON mapping SUMO config names to delay/flow reward references.",
    )
    parser.add_argument(
        "--sumo-config-choices",
        nargs="+",
        default=None,
        help="Explicit SUMO config filenames sampled per training episode.",
    )
    parser.add_argument("--cav-penetration", type=float, default=0.1)
    parser.add_argument(
        "--penetration-choices",
        type=float,
        nargs="+",
        default=None,
        help="Explicit CAV penetrations sampled per episode (e.g. 0.1 0.2).",
    )
    parser.add_argument(
        "--multi-penetration",
        action="store_true",
        help=(
            "Sample CAV penetration (0.05/0.1/0.2) each episode inside the env "
            "(penetration_choices), not once at config build."
        ),
    )
    parser.add_argument(
        "--use-meter-load",
        action="store_true",
        help=(
            "Append the V2V metering-load scalar to each CAV's observation "
            "(obs dim +1, 127 vs 126). Default OFF: clean 126-dim obs."
        ),
    )
    parser.add_argument(
        "--use-cav-count",
        action="store_true",
        help=(
            "Append the normalized concurrent-CAV count feature (penetration-"
            "context, obs dim +1) so the policy can calibrate metering to how "
            "many CAVs are in the scenario. Default OFF."
        ),
    )
    parser.add_argument(
        "--use-health-obs",
        action="store_true",
        help=(
            "Append the RSU merge-zone metastable-health scalar h in [0,1] to "
            "each CAV's observation (obs dim +1) so the policy can learn the "
            "h -> mode mapping explicitly (retry 2026-08-12). Default OFF."
        ),
    )
    parser.add_argument(
        "--use-margin-obs",
        action="store_true",
        help=(
            "Append the runway margin L_pre - L_min(rho) (m) to each CAV's "
            "observation (+1 dim): positive = this highway has room to "
            "control at this density, negative = futile (feasibility, "
            "2026-08-13). Default OFF."
        ),
    )
    parser.add_argument(
        "--feasibility-mask",
        action="store_true",
        help=(
            "Gate the discretionary modes on the runway margin (feasibility "
            "mask): infeasible margin -> coast pressure, feasible -> open. "
            "Fitted L_min coefficients via --feasibility-lmin-a/b. Default OFF."
        ),
    )
    parser.add_argument(
        "--feasibility-lmin-a",
        type=float,
        default=0.0,
        help="L_min(rho) = a + b*rho intercept (fitted from the grid).",
    )
    parser.add_argument(
        "--feasibility-lmin-b",
        type=float,
        default=0.0,
        help="L_min(rho) = a + b*rho slope (fitted from the grid).",
    )
    parser.add_argument(
        "--use-dv-obs",
        action="store_true",
        help=(
            "Append the injection velocity delta |v_merge - v_ramp_approach| "
            "(m/s) to each CAV's observation (obs dim +1): the KWG breakdown "
            "trigger is the injection speed difference (retry 2026-08-13). "
            "Default OFF."
        ),
    )
    parser.add_argument(
        "--health-mask-includes-protect",
        action="store_true",
        help=(
            "Extend the metastable-health mask to gate protect as well as "
            "fair_balance/recover: healthy h pushes protect to coast, sick h "
            "opens the TTC-avoiding lever (retry 2026-08-12). Default OFF."
        ),
    )
    parser.add_argument(
        "--paired-refs",
        default=None,
        help=(
            "JSON from tools/build_paired_hdv_refs.py. Enables per-seed "
            "fairness/flow reward refs via paired_seed_pool + paired_hdv_refs "
            "(no training started by this flag alone)."
        ),
    )
    parser.add_argument(
        "--hdv-mixture",
        type=float,
        nargs=3,
        default=None,
        metavar=("AGGRESSIVE", "NORMAL", "CAUTIOUS"),
        help=(
            "HDV proportions (e.g. 0.1 0.7 0.2); re-enables TraCI-side type "
            "assignment. Default: types come from the route vTypeDistribution."
        ),
    )
    parser.add_argument(
        "--control-mode",
        default=None,
        choices=[
            "shepherd",
            "continuous_acceleration",
            "shepherd_hybrid",
            "idm_residual_smoothing",
        ],
        help="Override the baseline's control_mode (e.g. shepherd_hybrid = (mode, magnitude) hybrid action space).",
    )
    parser.add_argument(
        "--idm-residual-filter-tau",
        type=float,
        default=0.0,
        help="Optional low-pass time constant in seconds for IDM residual commands.",
    )
    parser.add_argument(
        "--idm-decentralized-modes",
        action="store_true",
        help=(
            "Use the decentralized five-mode IDM actor: abstain, mild, nominal, "
            "strong, or conservative-reference damping. Safe local V2V "
            "contention selects the active absorber without a learned coordinator."
        ),
    )
    parser.add_argument(
        "--idm-decentralized-action-set",
        choices=(
            "gain_reference_v1",
            "acceleration_caps_v1",
            "activation_only_v1",
            "activation_speed_ceiling_v1",
            "activation_speed_ceiling_p50_v1",
            "activation_speed_ceiling_p70_v1",
        ),
        default="gain_reference_v1",
        help="Versioned local IDM mode contract used with --idm-decentralized-modes.",
    )
    parser.add_argument(
        "--hybrid-dial-init",
        type=float,
        default=-0.7,
        help="Hybrid dial mean init (raw): -0.7 = gentle default (mag ~0.15); +0.5 = open start (mag ~0.75, top-down).",
    )
    parser.add_argument(
        "--reward-weights",
        default=None,
        help="Comma-separated reward overrides, e.g. 'safety=3.0,delay_budget=0.5,flow=0.6' (applied on top of DEFAULT_CONFIG).",
    )
    parser.add_argument(
        "--enable-wave-damp",
        action="store_true",
        help="Opt-in 6th mode (research_v2 flow_stabilization port: active wave damping).",
    )
    parser.add_argument(
        "--enable-merge-match",
        action="store_true",
        help="Opt-in 7th mode (merge-stream smoothing: match merge-zone speed from "
        "a buffered far-ahead RSU range). Implies wave_damp to fill index 5.",
    )
    parser.add_argument(
        "--enable-jad",
        action="store_true",
        help="Opt-in JAD modes 8-9 (jad_absorb + jad_support): single absorber at "
        "the wave front + large-gap support, gated by the RSU wave signal. "
        "Implies the index-filler modes (wave_damp, merge_match) -> 9 modes.",
    )
    parser.add_argument(
        "--pruned-modes",
        action="store_true",
        help="Pruned champion action set Discrete(3): coast/recover_speed/protect. "
        "fair_balance + vacate_drop are removed (metering leaks at dense demand; "
        "vacate_drop was disabled in every probe). For the retrained baseline "
        "champion; do not combine with the opt-in mode flags.",
    )
    parser.add_argument(
        "--use-wave-signal",
        action="store_true",
        help="Append the RSU compound shockwave feature [wave_risk, wave_forming] "
        "to each CAV's local observation (+2 dims).",
    )
    parser.add_argument("--sumo-binary", default="sumo")
    parser.add_argument(
        "--sumo-root",
        default=None,
        help="Override the SUMO project/scene root (default: D:/research/highway_merge_shepherd). "
        "Point at a NEW longer map dir to train/eval on it.",
    )
    parser.add_argument(
        "--sumo-config",
        default=None,
        help="Override the SUMO config filename relative to project root "
        "(e.g. '_light/osm.dense.light_m2700r400.sumocfg' for the 2700 critical scene).",
    )
    parser.add_argument(
        "--zone-comm-range",
        type=float,
        default=None,
        help="Override RSU/zone communication range in metres (default 500.0). "
        "e.g. 1500.0 to let CAVs see the merge-zone speed far upstream.",
    )
    parser.add_argument(
        "--disable-modes",
        default=None,
        help="Comma-separated shepherd modes to disable for this run (e.g. 'fair_balance').",
    )
    parser.add_argument(
        "--fair-balance-tip",
        type=float,
        default=None,
        help="Override fair_balance tip zone (m) in the training env (default: keep DEFAULT_CONFIG).",
    )
    parser.add_argument(
        "--no-corridor-ttc-penalty",
        action="store_true",
        help="Zero the corridor_ttc_penalty reward weight for this run (single-variable control).",
    )
    parser.add_argument(
        "--no-easy-intervention-mask",
        action="store_true",
        help="Disable the easy-intervention soft mask so fair_balance (metering) "
        "is free in calm corridors — Experiment B: metering is the primary lever "
        "to hold merge density below the metastable ceiling.",
    )
    parser.add_argument(
        "--metastable-health-mask",
        action="store_true",
        help="Experiment C: RSU metastable-health graduated soft mask — h in [0,1] "
        "(Krauss 1997 §V hysteresis + §VI pre-breakdown variance) modulates the "
        "action space: healthy merge → strong coast pressure; sick merge → metering/"
        "recover open. Disables the easy-intervention mask for a clean ablation.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=1200,
        help="Training episode cap; 1200 covers the 900s demand window + clearing.",
    )
    # Sized for ~16-core / 32GB laptop (SUMO-bound). 8 runners reduces port
    # contention and sample-timeout risk vs 12; raise if CPU stays underutilized.
    parser.add_argument("--num-env-runners", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--train-batch-size", type=int, default=6000)
    parser.add_argument("--sgd-minibatch-size", type=int, default=128)
    parser.add_argument("--num-sgd-iter", type=int, default=4)
    parser.add_argument(
        "--rollout-fragment-length",
        default="auto",
        help="Steps per worker per sample ('auto' matches train_batch_size / workers).",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument(
        "--entropy-coeff",
        type=float,
        default=0.015,
        help="Keep exploration alive while reward is being reshaped.",
    )
    parser.add_argument(
        "--vf-clip-param",
        type=float,
        default=40.0,
        help="Larger than default 10 to match per-agent return scale.",
    )
    parser.add_argument("--max-neighbors", type=int, default=20)
    parser.add_argument("--max-cavs-critic", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-gpus", type=float, default=0)
    parser.add_argument("--logdir", default=None)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue from the latest checkpoint under {logdir}/checkpoints "
            "(highest iter_N, else final, else best)."
        ),
    )
    parser.add_argument(
        "--from-checkpoint",
        default=None,
        help="Explicit RLlib checkpoint dir to restore (overrides --resume).",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Train with curriculum: light(200k) → light+medium(500k) → all(1M).",
    )
    args = parser.parse_args()

    if args.logdir is None:
        args.logdir = f"runs/rllib/{args.baseline}"

    project_root = str(Path(__file__).resolve().parent)
    if project_root not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = project_root + os.pathsep + os.environ.get("PYTHONPATH", "")

    if args.curriculum:
        stages = [
            (["light"], 200_000),
            (["light", "medium"], 700_000),
            (["light", "medium", "dense"], 1_700_000),
        ]
        # Stage 1: from scratch
        args.curriculum_densities = stages[0][0]
        args.total_timesteps = stages[0][1]
        args.resume = False
        print(f"\n=== Curriculum stage 1/3: {stages[0][0]} → {stages[0][1]:,} steps ===\n")
        train(args)

        # Stages 2-3: resume from previous
        for i, (densities, cumulative_steps) in enumerate(stages[1:], start=1):
            args.curriculum_densities = densities
            args.total_timesteps = cumulative_steps
            args.resume = True
            print(
                f"\n=== Curriculum stage {i + 1}/3: {densities} → {cumulative_steps:,} steps (resume) ===\n"
            )
            train(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
