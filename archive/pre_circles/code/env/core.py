"""Multi-agent Gymnasium environment for TrafficShepherd SUMO simulation."""

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import traci
from statistics import mean, pvariance

from config.constants import (
    ACCELERATION_MAX,
    ACCELERATION_MIN,
    CAV_VTYPE,
    MERGE_EDGE,
    PRE_MERGE_EDGE,
    RAMP_EDGE,
    RAMP_LANE_LENGTH,
)
from config.defaults import (
    FAIRNESS_R_REF_BY_DENSITY,
    FLOW_Q_REF_BY_DENSITY,
    FLOW_V_REF_BY_DENSITY,
    TAIL_R_REF_BY_DENSITY,
    merge_config,
)
from env.actions import (
    ShepherdActionEngine,
    apply_continuous_accelerations,
    make_action_space,
)
from env.features import ObservationAssembler, TrustTracker, raw_obs_dim
from env.idm_residual import IDMResidualActionEngine, IDMResidualConfig
from idm.decentralized_damping import mode_count as idm_decentralized_mode_count
from env.mappo import acceleration_to_shepherd_mode, mappo_obs_dim, pack_mappo_observation
from env.rewards import (
    JerkTracker,
    MergeOutflowTracker,
    TimeLossTracker,
    compute_idm_damping_rewards,
    compute_selfish_rewards,
    compute_shepherd_rewards,
)
from env.snapshot import (
    TrafficSnapshot,
    clear_active_snapshot,
    get_active_snapshot,
    set_active_snapshot,
)
from env.sumo_utils import (
    close_sumo,
    get_active_vehicle_ids,
    get_arrived_ids,
    get_departed_ids,
    start_sumo,
)
from env.vehicles import CAVSelector, HDVTypeAssigner

# Merge-zone lane count used by the RSU metastable-health density estimator
# (veh/km/lane). The merge edge has 3 lanes on this scene (supervisor uses
# the same constant).
MERGE_NLANES = 3


def paired_pool_sample_probs(
    pool: list,
    refs: dict,
    easy_quantile: Optional[float] = None,
    easy_boost: float = 1.0,
) -> Optional[list]:
    """Per-seed sampling probabilities aligned to ``pool`` order.

    Seeds whose paired no_op ``fair_gap`` is at or below the
    ``easy_quantile`` quantile of the pool get ``easy_boost`` x weight, so
    easy episodes (nothing to win, abstention is the lesson) are seen more
    often in training. Returns None (= uniform) when disabled or degenerate.
    """
    if not pool or easy_quantile is None or float(easy_boost) == 1.0:
        return None
    gaps = [float((refs.get(str(s)) or {}).get("fair_gap", 0.0)) for s in pool]
    cutoff = float(np.quantile(gaps, float(easy_quantile)))
    w = [float(easy_boost) if g <= cutoff else 1.0 for g in gaps]
    total = sum(w)
    if total <= 0:
        return None
    return [x / total for x in w]


class TrafficShepherdMultiAgentEnv(gym.Env):
    """Gymnasium multi-agent env where each CAV is an agent keyed by vehicle ID.

    Customization hooks (for experimenting without editing this file):
      reward_fn:     callable ``(env, action_dict, obs) -> Dict[agent_id, float]``
                     that replaces the built-in reward computation
                     (see ``env/rewards.py`` for the defaults).
      action_engine: object with ``apply_all(action_dict, cav_ids=...)`` and
                     ``reset()`` that replaces the default
                     :class:`env.actions.ShepherdActionEngine` for discrete modes.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_config: Optional[Dict[str, Any]] = None,
        *,
        reward_fn: Optional[Any] = None,
        action_engine: Optional[Any] = None,
    ):
        super().__init__()
        self._reward_fn = reward_fn
        self.config = merge_config(env_config)
        self.project_root = Path(self.config["project_root"])
        self._sumo_port: Optional[int] = None
        self._cav_selector: Optional[CAVSelector] = None
        self._hdv_type_assigner: Optional[HDVTypeAssigner] = None
        # Master RNG for per-episode seed randomization: each reset() without
        # an explicit seed draws fresh CAV-selection / HDV-type seeds, so
        # training episodes see different CAV sets and driver mixes while the
        # whole run stays reproducible from config["cav_selection_seed"].
        # reset(seed=...) bypasses this (paired eval determinism).
        # RLlib env runners diversify the master seed via worker/vector index
        # (forwarded by env/rllib_wrapper.py); without these keys (single env,
        # tests, direct construction) the base seed is used unchanged.
        base_seed = self.config["cav_selection_seed"]
        worker_index = self.config.get("worker_index")
        vector_index = self.config.get("vector_index")
        if base_seed is not None and (worker_index is not None or vector_index is not None):
            base_seed = base_seed + int(worker_index or 0) * 1000 + int(vector_index or 0)
        self._episode_seed_rng = np.random.default_rng(base_seed)
        self._active_cavs: set = set()
        self._elapsed_steps = 0
        self._prev_obs_ids: set = set()
        # Agents that already emitted a terminal transition this episode.
        # RLlib forbids the same agent_id from appearing again after done=True.
        self._finished_ids: set = set()
        self.eval_tripinfo_output: Optional[Path] = None
        self.eval_fcd_output: Optional[Path] = None
        self.eval_stats_output: Optional[Path] = None

        self.neighbor_count = self.config["neighbor_count"]
        self.observation_radius = self.config["observation_radius"]
        self.max_speed = self.config["max_speed"]
        self.simulation_step = self.config["simulation_step"]
        self.max_episode_steps = self.config["max_episode_steps"]
        self.reward_weights = self.config["reward_weights"]
        self.eval_mode = self.config["eval_mode"]
        self.base_tau = self.config["base_tau"]
        self.tau_deltas = self.config["tau_deltas"]
        self.speed_factors = self.config["speed_factors"]
        self.control_mode = self.config.get("control_mode", "shepherd")
        decentralized_idm = bool(
            self.control_mode == "idm_residual_smoothing"
            and self.config.get("idm_residual_decentralized_damping_enabled", False)
        )
        decentralized_idm_mode_count = idm_decentralized_mode_count(
            str(
                self.config.get(
                    "idm_residual_decentralized_action_set", "gain_reference_v1"
                )
            )
        )
        # Eligibility bits mask discrete Shepherd modes and the opt-in local
        # decentralized IDM modes. Continuous IDM families remain unmasked.
        self.action_masking = bool(self.config.get("action_masking", False)) and (
            decentralized_idm
            or self.control_mode
            not in ("continuous_acceleration", "idm_residual_smoothing")
        )
        self._action_mask_dim = (
            decentralized_idm_mode_count
            if self.action_masking and decentralized_idm
            else len(self.config["shepherd_modes"])
            if self.action_masking
            else 0
        )
        self.acceleration_min = float(self.config.get("acceleration_min", ACCELERATION_MIN))
        self.acceleration_max = float(self.config.get("acceleration_max", ACCELERATION_MAX))
        self.selfish_reward_weights = self.config.get("selfish_reward_weights", {})
        self.rl_speed_mode = int(self.config.get("rl_speed_mode", 6))
        # All-HDV baseline per-step time-loss rate for the fairness term
        # (per-density values: config.defaults.FAIRNESS_R_REF_BY_DENSITY).
        self.fairness_r_ref = float(self.config.get("fairness_r_ref", 0.5))
        # All-HDV local-p90 delay reference for the R_tail penalty
        # (per-density values: config.defaults.TAIL_R_REF_BY_DENSITY).
        self.tail_r_ref = float(self.config.get("tail_r_ref", 0.34))
        # All-HDV merge-edge discharge reference for the R_eff flow term
        # (per-density values: config.defaults.FLOW_Q_REF_BY_DENSITY).
        self.flow_q_ref = float(self.config.get("flow_q_ref", 0.9))
        # All-HDV upstream mean-speed reference (FLOW_V_REF_BY_DENSITY).
        self.flow_v_ref = float(self.config.get("flow_v_ref", 27.0))
        # Paired no_op baselines for relative fairness / delay budget (rates).
        self.stream_delay_gap_ref = float(self.config.get("stream_delay_gap_ref", 0.0))
        self.delay_rate_ref = self.config.get("delay_rate_ref")
        if self.delay_rate_ref is not None:
            self.delay_rate_ref = float(self.delay_rate_ref)
        self.corridor_severe_braking_count_ref_per_s = float(
            self.config.get("corridor_severe_braking_count_ref_per_s", 1.0)
        )
        self.corridor_acceleration_energy_sum_ref_m2ps4 = float(
            self.config.get("corridor_acceleration_energy_sum_ref_m2ps4", 1.0)
        )

        # Reward: Flat-MARL uses accel actions + Shepherd reward.
        inferred_reward = (
            "selfish" if self.control_mode == "continuous_acceleration" else "shepherd"
        )
        self.reward_mode = self.config.get("reward_mode") or inferred_reward

        # v2 + trust features for Selfish / Flat-MARL / Shepherd learning.
        use_v2 = self.config.get("use_v2_obs")
        if use_v2 is None:
            use_v2 = True
        self.use_v2_obs = bool(use_v2)

        self.centralized_critic = bool(self.config.get("centralized_critic", False))
        self.max_cavs_critic = int(self.config.get("max_cavs_critic", 32))
        # Optional per-episode density sampling (training multi-density / curriculum).
        choices = self.config.get("density_choices")
        self._density_choices = list(choices) if choices else None
        config_choices = self.config.get("sumo_config_choices")
        self._sumo_config_choices = (
            [str(value) for value in config_choices] if config_choices else None
        )
        self._scenario_reward_refs = dict(self.config.get("scenario_reward_refs") or {})
        pchoices = self.config.get("penetration_choices")
        self._penetration_choices = [float(p) for p in pchoices] if pchoices else None
        # Optional paired multi-seed training: sample a seed each reset and
        # load that seed's All-HDV reward refs (see paired_hdv_refs_medium.json).
        pool = self.config.get("paired_seed_pool")
        self._paired_seed_pool = [int(s) for s in pool] if pool else None
        raw_refs = self.config.get("paired_hdv_refs") or {}
        self._paired_hdv_refs = {str(k): dict(v) for k, v in raw_refs.items()}
        self._paired_pool_probs = paired_pool_sample_probs(
            self._paired_seed_pool,
            self._paired_hdv_refs,
            easy_quantile=self.config.get("paired_easy_quantile"),
            easy_boost=float(self.config.get("paired_easy_boost", 1.0)),
        )
        self._paired_episode_seed: Optional[int] = None
        self._cav_empty_retries = int(self.config.get("cav_empty_retries", 2))
        # Gridlock early termination: end the episode (truncated) when the
        # whole network is effectively stopped for this many consecutive
        # steps. Jammed states are unlearnable advantage noise and cost 5-7x
        # wall time per step; 0 disables the check.
        self._gridlock_terminate_steps = int(self.config.get("gridlock_terminate_steps", 60))
        self._gridlock_steps = 0

        self._trust = TrustTracker(lambda_ema=self.config.get("trust_lambda", 0.2))
        self._jerk = JerkTracker(dt=self.simulation_step)
        self._time_loss = TimeLossTracker()
        # Per-step bulk vehicle state (TraCI subscriptions); activated for the
        # lifetime of each SUMO connection. Observation/reward/traci_query
        # readers transparently serve from it (same values, no per-vehicle
        # TCP round trips).
        self._snapshot = TrafficSnapshot()
        self._obs_assembler: Optional[ObservationAssembler] = None
        self._local_obs_dim: int

        # Shepherd control law, packaged in one engine (built from config).
        self._shepherd_engine = ShepherdActionEngine(
            base_tau=self.base_tau,
            tau_deltas=self.tau_deltas,
            speed_factors=self.speed_factors,
            max_speed=self.max_speed,
            dt=self.simulation_step,
            mode_names=list(self.config["shepherd_modes"]),
            flow_stab_config={
                "near": self.config.get("flow_stab_near", 150.0),
                "far": self.config.get("flow_stab_far", 200.0),
                "ema_alpha": self.config.get("flow_stab_ema_alpha", 0.3),
                "deadband": self.config.get("flow_stab_deadband", 0.5),
                "a_max_dec": self.config.get("flow_stab_a_max_dec", 1.0),
                "a_max_acc": self.config.get("flow_stab_a_max_acc", 1.0),
                "preview_gain": self.config.get("flow_stab_preview_gain", 0.5),
            },
            mode_config={
                "mode_count": len(self.config["shepherd_modes"]),
                "max_decel": float(self.config.get("mode_max_decel", 1.0)),
                "max_accel": float(self.config.get("mode_max_accel", 1.0)),
                "deadband": float(self.config.get("mode_deadband", 0.5)),
                "guard_speed": float(self.config.get("congestion_guard_speed", 8.0)),
                "disabled_modes": list(self.config.get("shepherd_disabled_modes") or []),
                "health_mask": bool(self.config.get("shepherd_health_mask", False)),
                "health_mask_threshold": float(
                    self.config.get("shepherd_health_mask_threshold", 0.8)
                ),
                "health_mask_modes": list(self.config.get("shepherd_health_mask_modes") or []),
                "fairness_mask": bool(self.config.get("shepherd_fairness_mask", False)),
                "fairness_mask_threshold": float(
                    self.config.get("shepherd_fairness_mask_threshold", 0.1)
                ),
                "fairness_mask_modes": list(self.config.get("shepherd_fairness_mask_modes") or []),
                "easy_intervention_mask": bool(
                    self.config.get("shepherd_easy_intervention_mask", False)
                ),
                "easy_intervention_mask_modes": list(
                    self.config.get("shepherd_easy_intervention_mask_modes") or []
                ),
                "easy_intervention_mask_discount": float(
                    self.config.get("shepherd_easy_intervention_mask_discount", 0.0)
                ),
                "easy_intervention_mask_gap_abs": float(
                    self.config.get("shepherd_easy_intervention_mask_gap_abs", 0.0)
                ),
                "easy_intervention_mask_graduated": bool(
                    self.config.get("shepherd_easy_intervention_mask_graduated", False)
                ),
                "easy_intervention_mask_gap_outer": float(
                    self.config.get("shepherd_easy_intervention_mask_gap_outer", 0.10)
                ),
                "easy_intervention_mask_hard_below": float(
                    self.config.get("shepherd_easy_intervention_mask_hard_below", 0.0)
                ),
                # Experiment C: RSU metastable-health graduated action-space mask.
                "metastable_health_mask": bool(self.config.get("metastable_health_mask", False)),
                "metastable_health_mask_modes": list(
                    self.config.get("metastable_health_mask_modes") or []
                ),
                "metastable_health_mask_discount": float(
                    self.config.get("metastable_health_mask_discount", 0.05)
                ),
                "metastable_health_mask_h_lo": float(
                    self.config.get("metastable_health_mask_h_lo", 0.2)
                ),
                "metastable_health_mask_h_hi": float(
                    self.config.get("metastable_health_mask_h_hi", 0.7)
                ),
                # Feasibility mask (2026-08-13): runway-margin-gated soft mask.
                "feasibility_mask": bool(self.config.get("feasibility_mask", False)),
                "feasibility_mask_discount": float(
                    self.config.get("feasibility_mask_discount", 0.05)
                ),
                "feasibility_margin_lo": float(self.config.get("feasibility_margin_lo", 0.0)),
                "feasibility_margin_hi": float(self.config.get("feasibility_margin_hi", 300.0)),
                "fair_balance_gap_tau": float(self.config.get("fair_balance_gap_tau", 0.05)),
                "stream_delay_gap_ref": float(self.config.get("stream_delay_gap_ref", 0.0)),
                "coast_force": bool(self.config.get("shepherd_coast_force", False)),
                "coast_force_delay_slack": float(
                    self.config.get("shepherd_coast_force_delay_slack", 0.0)
                ),
                "coast_force_modes": (
                    None
                    if bool(self.config.get("shepherd_coast_force_all_modes", False))
                    else list(
                        self.config.get("shepherd_coast_force_modes") or ["fair_balance", "guard"]
                    )
                ),
                "coast_force_hold_steps": int(
                    self.config.get("shepherd_coast_force_hold_steps", 1)
                ),
                "coast_force_min_step": int(self.config.get("shepherd_coast_force_min_step", 0)),
                "delay_rate_ref": self.config.get("delay_rate_ref"),
                "fair_balance_tip": float(self.config.get("fair_balance_tip", 120.0)),
                "fair_balance_factor": float(self.config.get("fair_balance_factor", 0.90)),
                "fair_balance_floor": float(self.config.get("fair_balance_floor", 8.0)),
                "fair_balance_ema_alpha": float(self.config.get("fair_balance_ema_alpha", 0.2)),
                "adaptive_gentle": bool(self.config.get("shepherd_adaptive_gentle", False)),
                "adaptive_gentle_factor": float(
                    self.config.get("shepherd_adaptive_gentle_factor", 0.85)
                ),
                "adaptive_gentle_ttc_lo": float(
                    self.config.get("shepherd_adaptive_gentle_ttc_lo", 0.5)
                ),
                "adaptive_gentle_ttc_hi": float(
                    self.config.get("shepherd_adaptive_gentle_ttc_hi", 2.0)
                ),
                "adaptive_gentle_ema_alpha": float(
                    self.config.get("shepherd_adaptive_gentle_ema_alpha", 0.1)
                ),
                "recover_ttc_gate": bool(self.config.get("shepherd_recover_ttc_gate", False)),
                "recover_ttc_lo": float(self.config.get("shepherd_recover_ttc_lo", 0.5)),
                "ttc_gate": bool(self.config.get("shepherd_ttc_gate", False)),
                "ttc_gate_lo": float(self.config.get("shepherd_ttc_gate_lo", 1.0)),
                "ttc_gate_modes": list(
                    self.config.get("shepherd_ttc_gate_modes")
                    or ("fair_balance", "recover", "vacate_drop")
                ),
                "meter_cap": self.config.get("shepherd_meter_cap"),
                "dissolve_exec": bool(self.config.get("shepherd_dissolve", False)),
                "dissolve_slow_v": float(self.config.get("shepherd_dissolve_slow_v", 12.0)),
                "dissolve_recover_v": float(self.config.get("shepherd_dissolve_recover_v", 20.0)),
                "dissolve_look": float(self.config.get("shepherd_dissolve_look", 500.0)),
                "dissolve_lead_min": float(self.config.get("shepherd_dissolve_lead_min", 150.0)),
                "dissolve_support_tau": float(
                    self.config.get("shepherd_dissolve_support_tau", 0.0)
                ),
                "dissolve_hold_gap": float(self.config.get("shepherd_dissolve_hold_gap", 8.0)),
                "merge_match_exec": bool(self.config.get("shepherd_merge_match", False)),
                "merge_match_min_speed": float(
                    self.config.get("shepherd_merge_match_min_speed", 15.0)
                ),
                "merge_match_comm": float(self.config.get("shepherd_merge_match_comm", 1500.0)),
                "jad_exec": bool(self.config.get("shepherd_jad", False)),
                "jad_absorb_speed": float(self.config.get("shepherd_jad_absorb_speed", 13.0)),
                "jad_congest_thresh": float(self.config.get("shepherd_jad_congest_thresh", 15.0)),
                "jad_support_tau": float(self.config.get("shepherd_jad_support_tau", 4.0)),
                "gap_select_exec": bool(self.config.get("shepherd_gap_select", False)),
                "gap_select_range": float(self.config.get("shepherd_gap_select_range", 150.0)),
                "gap_select_factor": float(self.config.get("shepherd_gap_select_factor", 0.85)),
                "gap_select_floor": float(self.config.get("shepherd_gap_select_floor", 12.0)),
                "follower_aware_exec": bool(self.config.get("shepherd_follower_aware_exec", False)),
                "follower_gentle_cap": float(self.config.get("shepherd_follower_gentle_cap", 0.97)),
                "follower_gap_near": float(self.config.get("shepherd_follower_gap_near", 15.0)),
                "follower_gap_far": float(self.config.get("shepherd_follower_gap_far", 60.0)),
                "smooth_exec": bool(self.config.get("shepherd_smooth_exec", False)),
                "smooth_a_max_dec": float(self.config.get("shepherd_smooth_a_max_dec", 0.4)),
                "smooth_a_max_acc": float(self.config.get("shepherd_smooth_a_max_acc", 0.4)),
                "smooth_deadband": float(self.config.get("shepherd_smooth_deadband", 0.3)),
                "smooth_ema_alpha": float(self.config.get("shepherd_smooth_ema_alpha", 0.3)),
                "look_behind_exec": bool(self.config.get("shepherd_look_behind_exec", False)),
                "behind_near": float(self.config.get("shepherd_behind_near", 150.0)),
                "behind_far": float(self.config.get("shepherd_behind_far", 200.0)),
                "behind_ema_alpha": float(self.config.get("shepherd_behind_ema_alpha", 0.3)),
                "recover_merge_clear": float(self.config.get("recover_merge_clear", 10.0)),
                "recover_down_clear": float(self.config.get("recover_down_clear", 18.0)),
                "recover_ego_max": float(self.config.get("recover_ego_max", 12.0)),
                "recover_push": float(self.config.get("recover_push", 1.5)),
                "recover_v_cap": float(self.config.get("recover_v_cap", 18.0)),
                "recover_gap_sign_gate": bool(
                    self.config.get("shepherd_recover_gap_sign_gate", True)
                ),
                "recover_gap_tau": float(self.config.get("shepherd_recover_gap_tau", 0.05)),
                "vacate_lane_duration": float(self.config.get("vacate_lane_duration", 3.0)),
                "vacate_soft_factor": float(self.config.get("vacate_soft_factor", 0.90)),
                "protect_ttc": float(self.config.get("protect_ttc", 1.5)),
                "protect_factor": float(self.config.get("protect_factor", 0.85)),
                "protect_tau_delta": float(self.config.get("protect_tau_delta", 0.5)),
            },
        )
        if action_engine is not None:
            self._shepherd_engine = action_engine

        idm_kwargs = {
            f.name: self.config[f"idm_residual_{f.name}"]
            for f in fields(IDMResidualConfig)
            if f.name != "dt_s" and f"idm_residual_{f.name}" in self.config
        }
        self._idm_residual_engine = IDMResidualActionEngine(
            IDMResidualConfig(dt_s=self.simulation_step, **idm_kwargs)
        )
        self._last_idm_residual_diag: Dict[str, Any] = {}
        self._last_idm_reward_components: Dict[str, Dict[str, float]] = {}
        self._reset_idm_pattern_context()

        # The shared v2 observation (ego + zone + trust-weighted neighbours) is
        # the only observation path; ``use_v2_obs`` is kept in the config for
        # backward compatibility but is always treated as True.
        self._obs_assembler = ObservationAssembler(
            trust_tracker=self._trust,
            neighbor_radius=self.config.get("neighbor_radius", 100.0),
            max_neighbors=self.config.get("max_neighbors", 20),
            zone_comm_range=self.config.get("zone_comm_range", 500.0),
            rsu_observation_contract=self.config.get(
                "rsu_observation_contract", "classic_rsu_configured_range_v1"
            ),
            max_speed=self.max_speed,
            slow_speed_threshold=self.config.get("slow_speed_threshold", 5.0),
            include_meter_load=bool(self.config.get("use_meter_load", False)),
            include_cav_count=bool(self.config.get("use_cav_count", False)),
            include_wave_signal=bool(self.config.get("use_wave_signal", False)),
            include_health=bool(self.config.get("use_health_obs", False)),
            include_dv=bool(self.config.get("use_dv_obs", False)),
            include_margin=bool(self.config.get("use_margin_obs", False)),
            include_idm_absorber_gate=bool(self.config.get("use_idm_absorber_gate", False)),
            include_idm_controller_context=bool(
                self.config.get("use_idm_controller_context", False)
            ),
        )
        # Detect scenario from sumo config path
        sumo_config = self.config["sumo_config"]
        density_name = sumo_config.split(".")[1]
        self._density_id = {
            "light": 0.0,
            "medium": 0.5,
            "dense": 1.0,
            "hyper": 1.0,
        }[density_name]
        if "bottleneck" in sumo_config or "lane_drop" in sumo_config:
            self._pre_edge = "upstream"
            self._merge_edge = "downstream"
            self._ramp_edge = "upstream"  # same edge as pre; lane-filter below
            self._ramp_lane = "upstream_2"  # dropped-lane filter
            self._zone_config = None  # edge-based, same code path as merge
            down_edge = "downstream"
        else:
            self._pre_edge = PRE_MERGE_EDGE
            self._merge_edge = MERGE_EDGE
            self._ramp_edge = RAMP_EDGE
            self._ramp_lane = None  # ramp is a dedicated edge
            self._zone_config = None
            down_edge = "1487606178"
        # Wire scenario edges into the action engine (built above before edges
        # were known).
        eng = self._shepherd_engine
        eng.pre_edge = self._pre_edge
        eng.merge_edge = self._merge_edge
        eng.ramp_edge = self._ramp_edge
        eng.down_edge = down_edge
        assembler_dim = raw_obs_dim(
            self.config.get("max_neighbors", 20),
            include_meter_load=bool(self.config.get("use_meter_load", False)),
            include_cav_count=bool(self.config.get("use_cav_count", False)),
            include_wave_signal=bool(self.config.get("use_wave_signal", False)),
            include_health=bool(self.config.get("use_health_obs", False)),
            include_dv=bool(self.config.get("use_dv_obs", False)),
            include_margin=bool(self.config.get("use_margin_obs", False)),
            include_idm_absorber_gate=bool(self.config.get("use_idm_absorber_gate", False)),
            include_idm_controller_context=bool(
                self.config.get("use_idm_controller_context", False)
            ),
        )
        cfg_local = self.config.get("local_obs_dim")
        if cfg_local is not None:
            # Full local vector size (includes mask bits when action_masking).
            self._local_obs_dim = int(cfg_local)
            self._raw_local_dim = (
                self._local_obs_dim - self._action_mask_dim
                if self.action_masking
                else self._local_obs_dim
            )
        else:
            self._raw_local_dim = assembler_dim
            self._local_obs_dim = assembler_dim + self._action_mask_dim

        # Merge-edge outflow tracker for the R_eff flow term (reward + zone
        # feature). Created after the scenario's merge edge is known; the
        # upstream speed EMA watches the pre-merge edge.
        self._flow_tracker = MergeOutflowTracker(
            self._merge_edge,
            alpha=float(self.config.get("flow_ema_alpha", 0.1)),
            speed_edge_id=self._pre_edge,
        )
        # Attach upstream health for the optional hard action mask.
        eng = self._shepherd_engine
        if getattr(eng, "health_mask_enabled", False):
            eng.health_fn = lambda: float(
                min(max(self._flow_tracker.s_norm(self.flow_v_ref), 0.0), 1.0)
            )
        self._fairness_ema = 0.0
        if getattr(eng, "fairness_mask_enabled", False):
            eng.fairness_fn = lambda: float(self._fairness_ema)
        # Adaptive gentleness / recover TTC gate / benefit gate feed on the
        # snapshot-based corridor TTC count (same source as corridor_ttc_penalty).
        if (
            getattr(eng, "adaptive_gentle", False)
            or getattr(eng, "recover_ttc_gate", False)
            or getattr(eng, "ttc_gate_enabled", False)
        ):
            eng.corridor_ttc_fn = self._corridor_ttc_count
        # Experiment C / retry: RSU metastable-health provider for the
        # graduated action-space mask (h = merge-zone rho/variance/speed
        # health). The retry (2026-08-12) also feeds h to the observation and
        # the health reward; h is computed ONCE per step in the obs phase and
        # shared by all consumers so the EMA advances exactly once per step.
        self._met_health_enabled = bool(
            getattr(eng, "met_health_mask_enabled", False)
            or self.config.get("use_health_obs", False)
            or self.config.get("use_dv_obs", False)
            or float(self.config.get("reward_weights", {}).get("health", 0.0)) > 0
        )
        # Feasibility (2026-08-13): runway margin feature + mask. margin =
        # L_pre - L_min(rho), where L_min(rho) = a + b*rho is the minimum
        # upstream runway for sparse-CAV metastable control at density rho
        # (fitted from the density x length controllability grid).
        self._feasibility_enabled = bool(
            self.config.get("use_margin_obs", False) or self.config.get("feasibility_mask", False)
        )
        self._last_met_health = None
        self._last_met_detail = None
        self._last_merge_dv = None
        self._last_margin = None
        self._pre_edge_len: Optional[float] = None
        if getattr(eng, "met_health_mask_enabled", False):
            eng.met_health_fn = self._last_met_health_stored
        if self.config.get("feasibility_mask", False):
            eng.feas_fn = self._last_margin_stored

        if self.control_mode == "idm_residual_smoothing":
            cfg = self._idm_residual_engine.cfg
            if cfg.decentralized_damping_enabled:
                self.action_space = gym.spaces.Discrete(
                    idm_decentralized_mode_count(cfg.decentralized_action_set)
                )
            elif cfg.action_family == "gain":
                low, high = cfg.gain_min, cfg.gain_max
            elif cfg.action_family == "reference_offset":
                low, high = cfg.reference_offset_min_mps, cfg.reference_offset_max_mps
            else:
                low, high = cfg.residual_min_mps2, cfg.residual_max_mps2
            if not cfg.decentralized_damping_enabled:
                self.action_space = gym.spaces.Box(
                    low=np.array([low], dtype=np.float32),
                    high=np.array([high], dtype=np.float32),
                    dtype=np.float32,
                )
        elif self.control_mode == "continuous_acceleration":
            self.action_space = gym.spaces.Box(
                low=np.array([self.acceleration_min], dtype=np.float32),
                high=np.array([self.acceleration_max], dtype=np.float32),
                dtype=np.float32,
            )
        elif self.control_mode == "shepherd_hybrid":
            # Hybrid composite: Discrete mode + Gaussian dial, matching the
            # model's Tuple action space (RLlib multi-distribution).
            self.action_space = gym.spaces.Tuple(
                (
                    gym.spaces.Discrete(len(self.config["shepherd_modes"])),
                    gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                )
            )
        else:
            self.action_space = make_action_space(len(self.config["shepherd_modes"]))

        if self.centralized_critic:
            obs_len = mappo_obs_dim(self._local_obs_dim, self.max_cavs_critic)
        else:
            obs_len = self._local_obs_dim

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32
        )

    @property
    def _sumo_is_running(self) -> bool:
        try:
            return traci.isLoaded()
        except Exception:
            return False

    def _apply_episode_sumo_config(self) -> None:
        """Sample an explicit scenario filename for this training episode."""
        if not self._sumo_config_choices:
            return
        self.config["sumo_config"] = str(self._episode_seed_rng.choice(self._sumo_config_choices))

    def _apply_scenario_reward_refs(self) -> None:
        """Apply delay/discharge baselines matched to the active IDM config."""
        ref = self._scenario_reward_refs.get(str(self.config["sumo_config"]))
        if not ref:
            return
        for name in (
            "delay_rate_ref",
            "flow_q_ref",
            "flow_v_ref",
            "corridor_severe_braking_count_ref_per_s",
            "corridor_acceleration_energy_sum_ref_m2ps4",
        ):
            if name in ref:
                setattr(self, name, float(ref[name]))

    def _apply_episode_density(self) -> None:
        """Sample density for this episode when density_choices is configured."""
        if not self._density_choices:
            return
        density = str(self._episode_seed_rng.choice(self._density_choices))
        self.config["sumo_config"] = f"osm.{density}.sumocfg"
        self.fairness_r_ref = float(FAIRNESS_R_REF_BY_DENSITY.get(density, self.fairness_r_ref))
        self.tail_r_ref = float(TAIL_R_REF_BY_DENSITY.get(density, self.tail_r_ref))
        self.flow_q_ref = float(FLOW_Q_REF_BY_DENSITY.get(density, self.flow_q_ref))
        self.flow_v_ref = float(FLOW_V_REF_BY_DENSITY.get(density, self.flow_v_ref))
        self._density_id = {
            "light": 0.0,
            "medium": 0.5,
            "dense": 1.0,
            "hyper": 1.0,
        }.get(density, 0.5)

    def _apply_episode_penetration(self) -> None:
        """Sample CAV penetration for this episode when penetration_choices is set.

        Multi-penetration training (generalization across CAV rates): the
        CAVSelector is rebuilt per reset from config["cav_penetration"], so
        mutating it here takes effect for the upcoming episode. Explicit-seed
        resets (paired eval) skip this entirely (see reset()).
        """
        if not self._penetration_choices:
            return
        p = float(self._episode_seed_rng.choice(self._penetration_choices))
        self.config["cav_penetration"] = p

    def _warmup_until_cav(self, warmup_steps: int, extend_steps: int) -> int:
        """Step SUMO until ≥1 CAV is active or budgets are exhausted.

        Returns the number of simulation steps consumed.
        """
        waited = 0
        limit = max(int(warmup_steps), 0) + max(int(extend_steps), 0)
        while waited < limit:
            target_time = traci.simulation.getTime() + self.simulation_step
            if traci.simulation.getMinExpectedNumber() <= 0 and self._active_cavs:
                break
            if traci.simulation.getMinExpectedNumber() <= 0 and waited >= warmup_steps:
                break
            traci.simulationStep(target_time)
            self._snapshot.update()
            self._update_idm_pattern_context()
            self._update_cav_set()
            waited += 1
            if self._active_cavs:
                break
        return waited

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed, options=options)

        if float(self.config.get("cav_penetration", 0.0)) <= 0.0:
            raise ValueError(
                "cav_penetration <= 0 yields an agent-less env: no CAV can "
                "ever depart, so reset() would wait out the warmup budget and "
                "fail. Use evaluate_sumo_baseline.py for all-HDV runs."
            )

        last_error: Optional[Exception] = None
        for attempt in range(self._cav_empty_retries + 1):
            self.close()

            self._elapsed_steps = 0
            self._gridlock_steps = 0
            self._active_cavs = set()
            self._prev_obs_ids = set()
            self._finished_ids = set()
            self._trust.reset()
            self._jerk.reset()
            self._time_loss.reset()
            self._flow_tracker.reset()
            self._shepherd_engine.reset()
            self._idm_residual_engine.reset()
            self._last_idm_residual_diag = {}
            self._reset_idm_pattern_context()
            self._fairness_ema = 0.0
            self._met_health_rho_ema = None  # Experiment C RSU health state
            self._met_health_var_ema = None
            self._met_health_var_base = None
            self._met_health_var_steps = 0
            self._met_health_merge_len = None  # cached static merge length
            self._last_met_health = None  # once-per-step h (retry: obs/mask/reward share it)
            self._last_met_detail = None  # once-per-step (h, rho, v_mean)
            self._last_merge_dv = None  # once-per-step injection velocity delta
            self._last_margin = None  # once-per-step runway margin (feasibility)
            self._pre_edge_len = None  # cached static pre-edge length
            if self._obs_assembler is not None:
                self._obs_assembler.reset()

            # Per-episode density before SUMO starts (training multi-density).
            if seed is None:
                self._apply_episode_sumo_config()
                self._apply_episode_density()
                self._apply_episode_penetration()
            self._apply_scenario_reward_refs()

            if seed is not None:
                # Explicit seed (paired evaluation): deterministic, unchanged.
                cav_seed = seed
                hdv_seed = seed
                self._paired_episode_seed = int(seed)
                self._apply_paired_refs(int(seed))
            elif self._paired_seed_pool:
                # Paired multi-seed training: one seed controls CAV pick, SUMO
                # RNG, and All-HDV reward refs for this episode.
                S = int(
                    self._episode_seed_rng.choice(self._paired_seed_pool, p=self._paired_pool_probs)
                )
                cav_seed = S
                hdv_seed = S
                self.config["sumo_seed"] = S
                self._paired_episode_seed = S
                self._apply_paired_refs(S)
            else:
                # Training reset: draw two independent per-episode seeds from the
                # master RNG so CAV selection and HDV types vary across episodes.
                episode_seeds = self._episode_seed_rng.integers(0, 2**31 - 1, size=2)
                cav_seed = int(episode_seeds[0])
                hdv_seed = int(episode_seeds[1])
                self._paired_episode_seed = None
            self._cav_selector = CAVSelector(
                penetration=self.config["cav_penetration"],
                seed=cav_seed,
                placement=self.config.get("cav_placement", "scatter"),
                block_size=int(self.config.get("cav_block_size", 3)),
            )
            if self.config.get("assign_hdv_types", True):
                self._hdv_type_assigner = HDVTypeAssigner(
                    proportions=self.config["hdv_type_proportions"], seed=hdv_seed
                )
            else:
                self._hdv_type_assigner = None

            # SUMO integration dt (may be finer than env/policy ``simulation_step``).
            # TraCI still advances by ``simulation_step`` per env.step, so e.g.
            # sumo_step_length=0.5 with simulation_step=1.0 runs two SUMO ticks
            # per decision — finer CF without changing the RL action rate.
            sumo_dt = float(self.config.get("sumo_step_length", self.simulation_step))
            extra_args = ["--step-length", str(sumo_dt)]
            if self.eval_mode:
                self.eval_tripinfo_output = Path(
                    self.config["eval_tripinfo_output"] or self.project_root / "tripinfos.eval.xml"
                )
                self.eval_fcd_output = Path(
                    self.config["eval_fcd_output"] or self.project_root / "fcd.eval.xml"
                )
                self.eval_stats_output = Path(
                    self.config["eval_stats_output"] or self.project_root / "stats.eval.xml"
                )
                extra_args.extend(
                    [
                        "--tripinfo-output",
                        str(self.eval_tripinfo_output),
                        "--fcd-output",
                        str(self.eval_fcd_output),
                        "--device.fcd.period",
                        str(self.config["eval_fcd_period"]),
                        "--statistic-output",
                        str(self.eval_stats_output),
                    ]
                )
            else:
                self.eval_tripinfo_output = None
                self.eval_fcd_output = None
                self.eval_stats_output = None

            self._sumo_port = start_sumo(
                sumo_binary=self.config["sumo_binary"],
                sumo_config=self.config["sumo_config"],
                project_root=self.project_root,
                port=None,
                extra_args=extra_args,
                seed=self.config.get("sumo_seed"),
            )
            self._snapshot.reset()
            set_active_snapshot(self._snapshot)

            warmup_steps = int(self.config.get("cav_warmup_steps", 300))
            # First attempt: normal warmup budget only. Retries add the
            # empty-extend budget (fresh SUMO start + longer wait).
            extend = self._cav_empty_extend_steps if attempt > 0 else 0
            waited = self._warmup_until_cav(warmup_steps, extend)

            if self._active_cavs:
                obs = self._get_observations()
                self._prev_obs_ids = set(obs.keys())
                info = {
                    "episode_time": traci.simulation.getTime(),
                    "active_cavs": list(self._active_cavs),
                    "warmup_steps": waited,
                    "sumo_config": self.config["sumo_config"],
                    "fairness_r_ref": self.fairness_r_ref,
                    "reset_attempt": attempt,
                }
                if self._paired_episode_seed is not None:
                    info["paired_seed"] = self._paired_episode_seed
                    info["flow_q_ref"] = self.flow_q_ref
                    info["flow_v_ref"] = self.flow_v_ref
                return obs, info

            last_error = RuntimeError(
                f"No CAV after warmup/extend "
                f"(attempt {attempt + 1}/{self._cav_empty_retries + 1}, "
                f"steps={waited}, config={self.config['sumo_config']}, "
                f"penetration={self.config['cav_penetration']})."
            )

        raise last_error or RuntimeError("No CAV after reset retries.")

    def step(self, action_dict: Dict[str, object]):
        if not self._sumo_is_running:
            raise RuntimeError("SUMO is not running. Call reset() before step().")

        if self.control_mode == "idm_residual_smoothing":
            # Zero is the frozen analytical controller; the residual engine
            # must therefore execute even when the policy mapping is empty.
            self._last_idm_residual_diag = self._idm_residual_engine.apply_all(
                action_dict, cav_ids=self._active_cavs
            )
            self._last_mask_fired = {}
            self._last_ineligible = {}
        elif action_dict:
            if self.control_mode == "continuous_acceleration":
                apply_continuous_accelerations(
                    action_dict,
                    dt=self.simulation_step,
                    max_speed=self.max_speed,
                    speed_mode=self.rl_speed_mode,
                    acceleration_min=self.acceleration_min,
                    acceleration_max=self.acceleration_max,
                )
                self._last_mask_fired = {}
                self._last_ineligible = {}
            elif self.control_mode == "shepherd_hybrid":
                parsed = {
                    cav_id: self._parse_hybrid_action(cav_id, action)
                    for cav_id, action in action_dict.items()
                }
                self._last_mask_fired = self._shepherd_engine.apply_all(
                    parsed, cav_ids=self._active_cavs
                )
                self._last_ineligible = dict(
                    getattr(self._shepherd_engine, "last_ineligible", {}) or {}
                )
                self._last_hybrid_parsed = parsed
            else:
                self._last_mask_fired = self._shepherd_engine.apply_all(
                    action_dict, cav_ids=self._active_cavs
                )
                self._last_ineligible = dict(
                    getattr(self._shepherd_engine, "last_ineligible", {}) or {}
                )
        else:
            self._last_mask_fired = {}
            self._last_ineligible = {}

        current_time = traci.simulation.getTime()
        next_time = current_time + self.simulation_step
        traci.simulationStep(next_time)
        self._elapsed_steps += 1
        self._snapshot.update()
        self._update_idm_pattern_context()
        self._update_gridlock_counter()
        self._update_cav_set()

        if self._met_health_enabled or self._feasibility_enabled:
            # Once per step: the RSU merge-zone health h (and the injection
            # velocity delta / runway margin when requested) for the mask
            # engine (next apply), the observation and the reward all read
            # the same values — the state after this step's actions, exactly
            # what the policy observes. One EMA advance per step, no
            # double-counting.
            detail = self.metastable_health_detail()
            self._last_met_detail = detail
            self._last_met_health = None if detail is None else detail[0]
            self._last_merge_dv = (
                self.merge_injection_dv() if bool(self.config.get("use_dv_obs", False)) else None
            )
            if self._feasibility_enabled:
                self._last_margin = self.feasibility_margin()

        obs = self._get_observations()
        # Never re-emit agents that already sent a terminal transition.
        for aid in list(obs.keys()):
            if aid in self._finished_ids:
                obs.pop(aid, None)

        rewards = self._compute_rewards(action_dict, obs)
        # Keep PPO advantages finite if a TraCI term blows up.
        rewards = {
            aid: float(0.0 if (r != r or r in (float("inf"), float("-inf"))) else r)
            for aid, r in rewards.items()
        }
        # Stream delay EMAs for fair_balance (after TimeLossTracker.update in rewards).
        self._refresh_stream_delay_emas()
        if getattr(self._shepherd_engine, "fairness_mask_enabled", False):
            self._refresh_fairness_ema()

        prev_ids = (
            set(action_dict.keys()) | set(getattr(self, "_prev_obs_ids", set()))
        ) - self._finished_ids
        current_ids = set(obs.keys())
        left_ids = prev_ids - current_ids

        # Only terminate if the vehicle truly left the network (not a one-step
        # TraCI/obs glitch). Temporarily missing CAVs keep a zero obs, not done.
        live_vehicles = set(get_active_vehicle_ids())
        truly_left = {aid for aid in left_ids if aid not in live_vehicles}
        missing_alive = left_ids - truly_left
        for aid in missing_alive:
            obs[aid] = self._zero_agent_observation()
            rewards.setdefault(aid, 0.0)
        current_ids = set(obs.keys())

        terminated = {cav_id: False for cav_id in current_ids}
        truncated = {cav_id: False for cav_id in current_ids}
        for cav_id in truly_left:
            terminated[cav_id] = True
            truncated[cav_id] = False
            rewards.setdefault(cav_id, 0.0)
            if cav_id not in obs:
                obs[cav_id] = self._zero_agent_observation()
            self._finished_ids.add(cav_id)

        end_reason = self._episode_end_reason()
        done = end_reason is not None
        # Gymnasium: time-limit / forced stop → truncated; natural empty → terminated.
        terminated["__all__"] = end_reason == "terminated"
        truncated["__all__"] = end_reason == "truncated"
        if done:
            for cav_id in list(obs.keys()):
                if cav_id not in self._finished_ids:
                    if end_reason == "truncated":
                        truncated[cav_id] = True
                    else:
                        terminated[cav_id] = True
                    self._finished_ids.add(cav_id)

        # Next step: only continue with still-active (non-terminal) agents.
        self._prev_obs_ids = {aid for aid in current_ids if aid not in self._finished_ids}

        info = {
            "episode_time": traci.simulation.getTime(),
            "active_cavs": list(self._prev_obs_ids),
            "total_vehicles": len(live_vehicles),
        }
        if self.control_mode == "idm_residual_smoothing":
            info["idm_residual"] = self._last_idm_residual_diag
            info["idm_reward_components"] = self._last_idm_reward_components
            self._last_idm_training_metrics = self._idm_training_metrics(action_dict)
            info["idm_training_metrics"] = self._last_idm_training_metrics
        if done:
            info["end_reason"] = (
                "gridlock"
                if self._gridlock_steps >= self._gridlock_terminate_steps > 0
                else end_reason
            )

        return obs, rewards, terminated, truncated, info

    def _zero_agent_observation(self) -> np.ndarray:
        """Return a finite placeholder with a legal no-op action mask."""
        value = np.zeros(self.observation_space.shape, dtype=np.float32)
        if self.action_masking and self._action_mask_dim > 0:
            # The actor-local vector is first in both IPPO and packed MAPPO.
            value[self._raw_local_dim] = 1.0
        return value

    def _parse_hybrid_action(self, cav_id: str, action: object) -> tuple:
        """Hybrid (mode, magnitude) parsing: the action arrives as
        (mode_int, dial in [-1,1]) from the composite distribution — the
        mode was sampled model-side WITH the eligibility mask applied
        (identical semantics to the discrete champion), so no env-side
        argmax. Only the dial needs mapping to [0, 1]. (2026-08-05, v3.)
        """
        if isinstance(action, (tuple, list)) and len(action) >= 2:
            mode_raw, dial_raw = action[0], action[1]
        else:
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
            mode_raw, dial_raw = arr[0], arr[1] if arr.shape[0] > 1 else 1.0
        mode = int(np.asarray(mode_raw).reshape(-1)[0])
        dial = float(np.asarray(dial_raw, dtype=np.float32).reshape(-1)[0])
        mag = (min(max(dial, -1.0), 1.0) + 1.0) / 2.0
        return mode, mag

    def close(self) -> None:
        clear_active_snapshot()
        close_sumo()
        self._active_cavs = set()

    def __del__(self):
        self.close()

    def _compute_rewards(
        self, action_dict: Dict[str, object], obs: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        if self._reward_fn is not None:
            return self._reward_fn(self, action_dict, obs)

        if self.reward_mode == "selfish":
            return compute_selfish_rewards(
                set(obs.keys()) | set(action_dict.keys()),
                self.max_speed,
                self.selfish_reward_weights,
            )

        if self.control_mode == "idm_residual_smoothing":
            selected = {
                str(row["vehicle_id"]): float(row["residual_mps2"])
                for row in self._last_idm_residual_diag.get("commands", [])
            }
            components: Dict[str, Dict[str, float]] = {}
            shared_reward_ids = None
            if self._idm_residual_engine.cfg.decentralized_damping_enabled:
                shared_reward_ids = set(action_dict)
            rewards = compute_idm_damping_rewards(
                selected,
                action_dict.keys(),
                self.max_speed,
                self.config.get("idm_damping_reward_weights", {}),
                self.observation_radius,
                cav_ids=self._active_cavs,
                jerk_tracker=self._jerk,
                time_loss_tracker=self._time_loss,
                delay_rate_ref=self.delay_rate_ref,
                delay_budget_scale=float(self.config.get("delay_budget_scale", 0.20)),
                corridor_edges=(self._pre_edge, self._merge_edge, self._ramp_edge),
                flow_tracker=self._flow_tracker,
                flow_q_ref=self.flow_q_ref,
                flow_v_ref=self.flow_v_ref,
                corridor_ttc_count=self._corridor_ttc_count(),
                corridor_severe_braking_count=(self._corridor_severe_braking_count()),
                corridor_severe_braking_count_ref_per_s=(
                    self.corridor_severe_braking_count_ref_per_s
                ),
                corridor_acceleration_energy_sum_m2ps4=(
                    self._corridor_acceleration_energy_sum_m2ps4()
                ),
                corridor_acceleration_energy_sum_ref_m2ps4=(
                    self.corridor_acceleration_energy_sum_ref_m2ps4
                ),
                residual_scale_mps2=max(
                    abs(self._idm_residual_engine.cfg.residual_min_mps2),
                    abs(self._idm_residual_engine.cfg.residual_max_mps2),
                ),
                shared_reward_ids=shared_reward_ids,
                center_corridor_terms=(shared_reward_ids is not None),
                diagnostics=components,
            )
            self._last_idm_reward_components = components
            return rewards

        # Shepherd reward (also Flat-MARL): near-zero acceleration is coast.
        reward_actions: Dict[str, int] = {}
        action_mags: Dict[str, float] = {}
        mag_full = float(self.acceleration_max)
        for cav_id, action in action_dict.items():
            if self.control_mode in ("continuous_acceleration", "idm_residual_smoothing"):
                reward_actions[cav_id] = acceleration_to_shepherd_mode(action)
                action_mags[cav_id] = abs(
                    float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
                )
            elif self.control_mode == "shepherd_hybrid":
                # Parsed (mode, mag) from this step's argmax; costs scale by
                # the dial (gentler = cheaper) via action_mags, full = 1.0.
                mode, mag = getattr(self, "_last_hybrid_parsed", {}).get(cav_id, (0, 0.0))
                reward_actions[cav_id] = int(mode)
                action_mags[cav_id] = float(mag)
            else:
                reward_actions[cav_id] = int(action)
        if self.control_mode == "shepherd_hybrid":
            mag_full = 1.0
        return compute_shepherd_rewards(
            reward_actions,
            self.max_speed,
            self.reward_weights,
            self.observation_radius,
            cav_ids=self._active_cavs,
            jerk_tracker=self._jerk,
            time_loss_tracker=self._time_loss,
            fairness_r_ref=self.fairness_r_ref,
            stream_delay_gap=self._shepherd_engine.stream_delay_gap(),
            stream_fairness_gap_scale=float(self.config.get("stream_fairness_gap_scale", 0.15)),
            stream_delay_gap_ref=float(self.stream_delay_gap_ref),
            # Computed inside compute_shepherd_rewards after TimeLossTracker.update
            # (calling it here would double-update and zero the rates).
            corridor_edges=(self._pre_edge, self._merge_edge, self._ramp_edge),
            delay_rate_ref=self.delay_rate_ref,
            delay_budget_scale=float(self.config.get("delay_budget_scale", 0.20)),
            delay_win_scale=float(self.config.get("delay_win_scale", 0.20)),
            flow_tracker=self._flow_tracker,
            flow_q_ref=self.flow_q_ref,
            flow_v_ref=self.flow_v_ref,
            tail_r_ref=self.tail_r_ref,
            control_cost_need_gated=bool(self.config.get("control_cost_need_gated", False)),
            control_cost_c_min=float(self.config.get("control_cost_c_min", 0.25)),
            control_cost_health=str(self.config.get("control_cost_health", "upstream")),
            mask_fired=getattr(self, "_last_mask_fired", None),
            ineligible=getattr(self, "_last_ineligible", None),
            mode_use_bonus_modes=self.config.get("mode_use_bonus_modes"),
            mode_use_bonus_overrides=self.config.get("mode_use_bonus_overrides"),
            safety_collision_penalty=float(self.config.get("safety_collision_penalty", 6.5)),
            corridor_ttc_count=self._corridor_ttc_count(),
            action_mags=action_mags or None,
            action_mag_full=mag_full,
            action_mag_cost_floor=float(self.config.get("action_mag_cost_floor", 0.3)),
            metastable_health=self._last_met_health,
        )

    def _apply_paired_refs(self, seed: int) -> None:
        """Load paired no_op refs for ``seed`` into live reward baselines."""
        ref = self._paired_hdv_refs.get(str(seed))
        if not ref:
            return
        if "fairness_r_ref" in ref:
            self.fairness_r_ref = float(ref["fairness_r_ref"])
        if "flow_q_ref" in ref:
            self.flow_q_ref = float(ref["flow_q_ref"])
        if "flow_v_ref" in ref:
            self.flow_v_ref = float(ref["flow_v_ref"])
        if "tail_r_ref" in ref:
            self.tail_r_ref = float(ref["tail_r_ref"])
        if "stream_delay_gap_ref" in ref:
            self.stream_delay_gap_ref = float(ref["stream_delay_gap_ref"])
        elif "fair_gap" in ref and ref.get("mean_tt"):
            mt = float(ref["mean_tt"])
            if mt > 1e-6:
                self.stream_delay_gap_ref = float(ref["fair_gap"]) / mt
        self._shepherd_engine.set_stream_delay_gap_ref(self.stream_delay_gap_ref)
        if "delay_rate_ref" in ref:
            self.delay_rate_ref = float(ref["delay_rate_ref"])
        elif ref.get("mean_tt") and ref.get("mean_time_loss") is not None:
            mt = float(ref["mean_tt"])
            if mt > 1e-6:
                self.delay_rate_ref = float(ref["mean_time_loss"]) / mt
        self._shepherd_engine.set_delay_rate_ref(self.delay_rate_ref)

    def _refresh_stream_delay_emas(self) -> None:
        """Mean per-step delay rate on pre vs ramp → action-engine EMAs."""
        snap = get_active_snapshot()
        main_rates: list[float] = []
        ramp_rates: list[float] = []
        if snap is not None:
            for vid in snap.ordered_ids():
                road = snap.road(vid)
                rate = float(self._time_loss.get(vid))
                if road == self._pre_edge:
                    main_rates.append(rate)
                elif road == self._ramp_edge:
                    ramp_rates.append(rate)
        main_m = sum(main_rates) / len(main_rates) if main_rates else None
        ramp_m = sum(ramp_rates) / len(ramp_rates) if ramp_rates else None
        self._shepherd_engine.update_stream_delay_rates(main_m, ramp_m)

    def _corridor_fairness_raw(self) -> float:
        """Signed corridor HDV delay vs paired ``fairness_r_ref`` (no_op seed).

        F = (r_ref - mean_rate) / r_ref over HDVs on pre ∪ merge ∪ ramp.
        Matching the ref scores ~0; worse delay is negative. Empty → 0.
        """
        snap = get_active_snapshot()
        corridor = {self._pre_edge, self._merge_edge, self._ramp_edge}
        rates: list[float] = []
        if snap is not None:
            for vid in snap.ordered_ids():
                if vid in self._active_cavs:
                    continue
                road = snap.road(vid)
                if road is None or road not in corridor:
                    continue
                rates.append(float(self._time_loss.get(vid)))
        else:
            try:
                ids = list(traci.vehicle.getIDList())
            except Exception:
                ids = []
            for vid in ids:
                if vid in self._active_cavs:
                    continue
                try:
                    road = traci.vehicle.getRoadID(vid)
                except Exception:
                    continue
                if road not in corridor:
                    continue
                rates.append(float(self._time_loss.get(vid)))
        if not rates:
            return 0.0
        mean_r = sum(rates) / len(rates)
        r_ref = max(float(self.fairness_r_ref), 1e-6)
        return float(min(max((r_ref - mean_r) / r_ref, -1.0), 1.0))

    def _refresh_fairness_ema(self) -> None:
        """EMA of corridor fairness score for meter/yield gating."""
        raw = self._corridor_fairness_raw()
        alpha = float(self.config.get("shepherd_fairness_mask_ema_alpha", 0.2))
        alpha = min(max(alpha, 0.0), 1.0)
        self._fairness_ema = (1.0 - alpha) * float(self._fairness_ema) + alpha * raw

    def _vtype_lookup(self) -> Dict[str, str]:
        snap = get_active_snapshot()
        if snap is not None:
            return {
                vid: snap.vtype(vid) for vid in snap.ordered_ids() if snap.vtype(vid) is not None
            }
        lookup = {}
        for vid in traci.vehicle.getIDList():
            try:
                lookup[vid] = traci.vehicle.getTypeID(vid)
            except Exception:
                continue
        return lookup

    def _placement_pool(self, departed: list) -> list:
        """Placement-aware CAV eligibility (probe, 2026-08-06).

        ``cav_placement="upstream"`` restricts CAVs to vehicles that traverse
        the upstream mainline approach (PRE_MERGE_EDGE), i.e. ramp vehicles and
        downstream-entering vehicles are never CAVs. ``"uniform"`` (default)
        keeps the current behavior: any vehicle can be a CAV.
        """
        mode = self.config.get("cav_placement", "uniform")
        if mode != "upstream":
            return departed
        eligible = []
        for vid in departed:
            try:
                route = traci.vehicle.getRoute(vid)
            except Exception:
                route = []
            if PRE_MERGE_EDGE in route:  # traverses the mainline approach to the merge
                eligible.append(vid)
        return eligible

    def _update_cav_set(self) -> None:
        """Track newly departed and arrived CAVs and assign HDV types."""
        departed = get_departed_ids()
        arrived = get_arrived_ids()

        if self._hdv_type_assigner is not None:
            self._hdv_type_assigner.assign_all_departed(departed, skip=self._active_cavs)

        new_cavs: list = []
        if self._cav_selector is not None:
            pool = self._placement_pool(departed)
            if self.config.get("cav_aggressive_first", False):
                types = (
                    self._hdv_type_assigner.get_assigned_types()
                    if self._hdv_type_assigner is not None
                    else {}
                )
                new_cavs = self._cav_selector.add_departed_aggressive_first(pool, types)
            else:
                new_cavs = self._cav_selector.add_departed(pool)
            self._cav_selector.remove_arrived(arrived)

        all_vehicles = set(get_active_vehicle_ids())
        if self._cav_selector is not None:
            self._active_cavs = self._cav_selector.get_active_cavs(list(all_vehicles))
        else:
            self._active_cavs = set()

        # Coast / SUMO-in-control CAVs must drive as normal, not as the
        # aggressive/cautious draw from the route vTypeDistribution.
        if new_cavs and bool(self.config.get("cav_force_normal_type", True)):
            for vid in new_cavs:
                try:
                    traci.vehicle.setType(vid, CAV_VTYPE)
                except Exception:
                    continue
                # Drop any cached tau from the pre-retype vType.
                self._shepherd_engine._tau_cache.pop(vid, None)

        self._trust.set_cav_ids(self._active_cavs)
        self._trust.prune(all_vehicles)

    def _corridor_delay_rate_live(self) -> Optional[float]:
        """Mean per-step timeLoss rate on pre∪merge∪ramp (for zone obs)."""
        snap = get_active_snapshot()
        if snap is None:
            return None
        edges = {self._pre_edge, self._merge_edge, self._ramp_edge}
        rates: list[float] = []
        for vid in snap.ordered_ids():
            road = snap.road(vid)
            if road is None or road not in edges:
                continue
            rates.append(float(self._time_loss.get(vid)))
        if not rates:
            return None
        return float(sum(rates) / len(rates))

    def _last_met_health_stored(self) -> Optional[float]:
        """Stored h for the Experiment-C mask engine (computed once per step).

        The retry computes h once in the obs phase and shares it with the
        mask, the observation and the reward — keeping the health EMA at one
        advance per step regardless of how many consumers exist.
        """
        return getattr(self, "_last_met_health", None)

    def _last_margin_stored(self) -> Optional[float]:
        """Stored runway margin for the feasibility mask (once per step)."""
        return getattr(self, "_last_margin", None)

    def _pre_edge_length(self) -> float:
        """Static length of the upstream mainline pre-edge (m), cached."""
        if self._pre_edge_len is None:
            try:
                self._pre_edge_len = sum(
                    traci.lane.getLength(lane_id)
                    for lane_id in traci.lane.getIDList()
                    if lane_id.startswith(PRE_MERGE_EDGE)
                )
            except Exception:
                self._pre_edge_len = 1144.0
        return self._pre_edge_len

    def feasibility_margin(self) -> Optional[float]:
        """Runway margin L_pre - L_min(rho) (m), or None when rho is unknown.

        Feasibility feature (2026-08-13): L_min(rho) = a + b*rho is the
        minimum upstream runway for sparse-CAV metastable control at density
        rho, fitted from the density x length controllability grid. Positive
        margin = this highway has room to act at this density; negative =
        control is futile (coast). Uses the ONCE-PER-STEP stored health
        detail (never advances the EMA itself).
        """
        detail = getattr(self, "_last_met_detail", None)
        if detail is None:
            return None
        rho = detail[1]
        if rho is None:
            return None
        a = float(self.config.get("feasibility_lmin_a", 0.0))
        b = float(self.config.get("feasibility_lmin_b", 0.0))
        lmin = a + b * rho
        return float(self._pre_edge_length() - lmin)

    def merge_injection_dv(self) -> Optional[float]:
        """Injection velocity delta |v_mean_merge - v_ramp_approach| (m/s).

        Retry feature (2026-08-13): the KWG breakdown trigger is the speed
        DIFFERENCE of injections into the stream, so the magnitude of the
        ramp-approach speed gap vs the merge-edge mean is the quantity the
        policy needs to see. Uses the ONCE-PER-STEP stored health detail
        (never calls ``metastable_health_detail`` — that would advance the
        EMA a second time). None when no ramp vehicle is near the junction.
        """
        detail = getattr(self, "_last_met_detail", None)
        if detail is None:
            return None
        v_mean = detail[2]
        snap = get_active_snapshot()
        if snap is None:
            return None
        speeds = [
            snap.speed(v)
            for v in snap.ordered_ids()
            if snap.road(v) == self._ramp_edge
            and snap.lane_pos(v) is not None
            and snap.lane_pos(v) >= RAMP_LANE_LENGTH - 120.0
        ]
        speeds = [s for s in speeds if s is not None]
        if not speeds:
            return None
        return float(abs(v_mean - sum(speeds) / len(speeds)))

    def metastable_health_detail(self) -> Optional[tuple]:
        """RSU merge-zone metastable health detail (feasibility probe).

        Public accessor: returns ``(h, rho_e, v_mean)`` — h in [0,1] (see
        ``_merge_metastable_health`` for the semantics), the merge-zone
        density EMA in veh/km/lane, and the merge-zone mean speed in m/s.
        Snapshot-only (zero extra TraCI calls); None when the snapshot is
        unavailable or the merge edge has no vehicles. The EMA state advances
        once per call — the probe calls this once per step and keeps the
        Experiment-C mask disabled, so the EMA advances exactly once per step.
        """
        snap = get_active_snapshot()
        if snap is None:
            return None
        try:
            merge_ids = [v for v in snap.ordered_ids() if snap.road(v) == MERGE_EDGE]
        except Exception:
            return None
        speeds = [snap.speed(v) for v in merge_ids if snap.speed(v) is not None]
        if not speeds:
            return None
        # Merge-edge total length is STATIC (network geometry never changes) —
        # cache it after the first call instead of re-querying every lane ID via
        # TraCI on every CAV step (this was the training slowdown: 20+ CAVs x
        # per-step getIDList+getLength). Snapshot-only stays true afterwards.
        if self._met_health_merge_len is None:
            try:
                self._met_health_merge_len = sum(
                    traci.lane.getLength(lane_id)
                    for lane_id in traci.lane.getIDList()
                    if lane_id.startswith(MERGE_EDGE)
                )
            except Exception:
                self._met_health_merge_len = 300.0
        mlen = self._met_health_merge_len
        rho = len(merge_ids) / max(mlen, 1e-6) * 1000.0 / MERGE_NLANES
        v_mean = float(mean(speeds))
        var = float(pvariance(speeds)) if len(speeds) > 1 else 0.0

        a = float(self.config.get("metastable_health_ema_alpha", 0.3))
        if self._met_health_rho_ema is None:
            self._met_health_rho_ema = rho
        else:
            self._met_health_rho_ema = (1 - a) * self._met_health_rho_ema + a * rho
        if self._met_health_var_ema is None:
            self._met_health_var_ema = var
        else:
            self._met_health_var_ema = (1 - a) * self._met_health_var_ema + a * var
        rho_e = self._met_health_rho_ema
        var_e = self._met_health_var_ema

        # Free-flow variance baseline: sample the var_ema over the first
        # ``var_baseline_window`` steps (steady window) and hold it — NOT a
        # running minimum. On the critical 2700 scene the merge-zone variance
        # is intrinsically high (mean ~66 even when healthy), so a soft-min
        # baseline captures a transient dip and makes every later step look
        # "about to collapse" (h stuck at ~0). A fixed early-window baseline
        # reflects the scene's normal operating variance instead.
        if self._met_health_var_base is None:
            self._met_health_var_base = var_e
        window = int(self.config.get("metastable_health_var_baseline_window", 120))
        if self._met_health_var_steps < window:
            self._met_health_var_base = 0.95 * self._met_health_var_base + 0.05 * var_e
        self._met_health_var_steps += 1

        rho_free = float(self.config.get("metastable_health_rho_free", 2.0))
        rho_crit = float(self.config.get("metastable_health_rho_crit", 5.5))
        v_free = float(self.config.get("metastable_health_v_free", 25.0))
        v_break = float(self.config.get("metastable_health_v_break", 12.0))
        var_lo = float(self.config.get("metastable_health_var_ratio_lo", 3.0))
        var_hi = float(self.config.get("metastable_health_var_ratio_hi", 6.0))

        rho_h = 1.0 - min(max((rho_e - rho_free) / max(rho_crit - rho_free, 1e-6), 0.0), 1.0)
        v_h = min(max((v_mean - v_break) / max(v_free - v_break, 1e-6), 0.0), 1.0)
        # Krauss 1997 §VI: speed variance rises in FREE FLOW shortly before
        # breakdown. The warning is only informative while the density is
        # still healthy (rho_h > 0.5) — once rho is already climbing toward
        # crit, the density term carries the signal and variance (intrinsically
        # high at critical load) adds nothing but noise.
        if rho_h > 0.5:
            var_ratio = var_e / max(self._met_health_var_base, 1e-3)
            var_h = 1.0 - min(max((var_ratio - var_lo) / max(var_hi - var_lo, 1e-6), 0.0), 1.0)
            return float(min(rho_h, v_h, var_h)), rho_e, v_mean
        return float(min(rho_h, v_h)), rho_e, v_mean

    def _merge_metastable_health(self) -> Optional[float]:
        """RSU merge-zone metastable health h in [0,1] (Experiment C).

        Snapshot-only (zero extra TraCI calls): aggregate merge-edge vehicles
        into density (veh/km/lane), mean speed and speed VARIANCE, then map to
        health via three linear ramps AND-combined, following Krauß-Wagner-
        Gawron 1997 (Phys. Rev. E 55, 5597 — the metastable-state paper behind
        the SUMO Krauss model):

          rho ramp (§V hysteresis):  rho_e <= rho_free -> 1.0 (upper branch)
                                     rho_e >= rho_crit -> 0.0 (collapse)
          v   ramp (§V):             v_mean >= v_free  -> 1.0 (free flow)
                                     v_mean <= v_break -> 0.0 (crawling jam)
          var ramp (§VI pre-breakdown signal): speed variance rises ~160x
                                     (1.1 -> 177.8) in the ~30 s BEFORE the
                                     merge collapses, even while rho/v still
                                     look healthy — so a rising var_ratio
                                     (var_ema / free-flow baseline) drops h
                                     EARLY, opening metering before the crash.

        h = min(rho_h, v_h, var_h). 1 = deep metastable (free coast); falling
        h = merge sickening (density rising, speed dropping, or variance
        amplifying); 0 = collapse imminent (meter to restore). Calibration
        follows the measured window: rho_free 2.0, rho_crit 5.5 veh/km/lane,
        v_free 25, v_break 12 m/s (docs/metastable_merge_control.md §1.3).
        """
        detail = self.metastable_health_detail()
        return None if detail is None else detail[0]

    def _corridor_ttc_count(self) -> Optional[int]:
        """Count sub-threshold TTC pairs on corridor lanes this step.

        Snapshot-only (zero extra TraCI calls): group corridor vehicles by
        lane, sort by position, count consecutive pairs with
        ttc = (gap - veh_len) / (v_fol - v_lead) below
        ``corridor_ttc_threshold``. This is the reward-side view of the
        eval's TTC safety leg — the compression-wave redistribution the wake
        probe documented is corridor-systemic, so the penalty is too.
        """
        snap = get_active_snapshot()
        if snap is None:
            return None
        edges = {
            self._pre_edge,
            self._merge_edge,
            self._ramp_edge,
            self._shepherd_engine.down_edge,
        }
        thr = float(self.config.get("corridor_ttc_threshold", 1.5))
        lanes: dict = {}
        for vid in snap.ordered_ids():
            road = snap.road(vid)
            if road is None or road not in edges:
                continue
            lane = snap.lane_id(vid)
            pos = snap.lane_pos(vid)
            spd = snap.speed(vid)
            if lane is None or pos is None or spd is None:
                continue
            lanes.setdefault(lane, []).append((float(pos), float(spd)))
        count = 0
        for vehs in lanes.values():
            vehs.sort()
            for i in range(1, len(vehs)):
                (p_f, v_f), (p_l, v_l) = vehs[i - 1], vehs[i]
                closing = v_f - v_l
                if closing <= 0.01:
                    continue
                ttc = (p_l - p_f - 5.0) / closing
                if ttc < thr:
                    count += 1
        return count

    def _corridor_min_ttc(self) -> Optional[float]:
        """Minimum finite bumper-to-bumper TTC on the controlled corridor."""
        snap = get_active_snapshot()
        if snap is None:
            return None
        edges = {
            self._pre_edge,
            self._merge_edge,
            self._ramp_edge,
            self._shepherd_engine.down_edge,
        }
        lanes: dict = {}
        for vehicle_id in snap.ordered_ids():
            if snap.road(vehicle_id) not in edges:
                continue
            lane = snap.lane_id(vehicle_id)
            position = snap.lane_pos(vehicle_id)
            speed = snap.speed(vehicle_id)
            if lane is None or position is None or speed is None:
                continue
            lanes.setdefault(lane, []).append((float(position), float(speed)))
        minimum = float("inf")
        for vehicles in lanes.values():
            vehicles.sort()
            for index in range(1, len(vehicles)):
                (follower_pos, follower_speed), (leader_pos, leader_speed) = (
                    vehicles[index - 1],
                    vehicles[index],
                )
                closing_speed = follower_speed - leader_speed
                if closing_speed <= 0.01:
                    continue
                ttc = max(leader_pos - follower_pos - 5.0, 0.0) / closing_speed
                minimum = min(minimum, float(ttc))
        return None if minimum == float("inf") else minimum

    def _idm_training_metrics(self, action_dict: Dict[str, object]) -> Dict[str, object]:
        """One-step observability contract for decentralized IDM training.

        These diagnostics deliberately separate genuine two-action decision
        points from latched/non-candidate samples.  They are copied into the
        RLlib info stream and aggregated by :class:`IDMResidualTrainingCallbacks`.
        """
        diag = dict(self._last_idm_residual_diag or {})
        bids = list(diag.get("local_bids") or [])
        changes = list(diag.get("member_changes") or [])
        eligible = [row for row in bids if bool(row.get("admissible", False))]
        safety_masked = [row for row in bids if not bool(row.get("admissible", False))]
        activation_bids = [row for row in eligible if int(row.get("mode_index", 0)) != 0]
        abstentions = [row for row in eligible if int(row.get("mode_index", 0)) == 0]
        winners = [row for row in activation_bids if row.get("outcome") == "selected"]
        completions = [row for row in changes if row.get("reason") == "option_completed"]
        safety_aborts = [row for row in changes if row.get("reason") == "unsafe_or_exited"]

        reward_totals: Dict[str, float] = {}
        for component_row in (self._last_idm_reward_components or {}).values():
            for key, value in component_row.items():
                if key in {"total", "wave_risk", "executed_residual_mps2"}:
                    continue
                try:
                    reward_totals[key] = reward_totals.get(key, 0.0) + float(value)
                except (TypeError, ValueError):
                    continue

        def simulation_count(name: str) -> int:
            try:
                return int(getattr(traci.simulation, name)())
            except Exception:
                return 0

        return {
            "step": int(diag.get("step", self._elapsed_steps)),
            "policy_action_count": len(action_dict),
            "eligible_decision_count": len(eligible),
            "safety_masked_decision_count": len(safety_masked),
            "nondecision_masked_count": max(len(action_dict) - len(eligible), 0),
            "executed_activate_count": len(activation_bids),
            "executed_abstain_count": len(abstentions),
            "winning_activation_count": len(winners),
            "treatment_completion_count": len(completions),
            "safety_abort_count": len(safety_aborts),
            "controlled_cav_seconds": float(diag.get("controlled_cav_seconds", 0.0)),
            "active_treatment_count": int(diag.get("simultaneous_absorber_count", 0)),
            "collision_observations": simulation_count("getCollidingVehiclesNumber"),
            "teleport_starts": simulation_count("getStartingTeleportNumber"),
            "arrivals": simulation_count("getArrivedNumber"),
            "minimum_ttc_s": self._corridor_min_ttc(),
            "corridor_delay_rate": self._corridor_delay_rate_live(),
            "merge_discharge_rate": float(self._flow_tracker.rate()),
            "severe_braking_count": int(self._corridor_severe_braking_count() or 0),
            "severe_braking_fraction": float(
                self._corridor_severe_braking_fraction() or 0.0
            ),
            "acceleration_energy_sum_m2ps4": float(
                self._corridor_acceleration_energy_sum_m2ps4() or 0.0
            ),
            "reward_components": reward_totals,
            "safety_abort_reasons": [
                list(row.get("reason_details") or ()) for row in safety_aborts
            ],
        }

    def _corridor_severe_braking_fraction(self) -> Optional[float]:
        """Density-normalized share of corridor vehicles braking below -3 m/s²."""
        snap = get_active_snapshot()
        if snap is None:
            return None
        edges = {
            self._pre_edge,
            self._merge_edge,
            self._ramp_edge,
            self._shepherd_engine.down_edge,
        }
        threshold = float(self.config.get("idm_severe_brake_threshold_mps2", -3.0))
        observed = 0
        severe = 0
        for vehicle_id in snap.ordered_ids():
            if snap.road(vehicle_id) not in edges:
                continue
            acceleration = snap.accel(vehicle_id)
            if acceleration is None:
                continue
            observed += 1
            severe += int(float(acceleration) < threshold)
        return float(severe / observed) if observed else 0.0

    def _corridor_severe_braking_count(self) -> Optional[int]:
        """Raw per-step corridor events matching the integrated paper endpoint."""
        snap = get_active_snapshot()
        if snap is None:
            return None
        edges = {
            self._pre_edge,
            self._merge_edge,
            self._ramp_edge,
            self._shepherd_engine.down_edge,
        }
        threshold = float(self.config.get("idm_severe_brake_threshold_mps2", -3.0))
        return sum(
            1
            for vehicle_id in snap.ordered_ids()
            if snap.road(vehicle_id) in edges
            and snap.accel(vehicle_id) is not None
            and float(snap.accel(vehicle_id)) < threshold
        )

    def _corridor_acceleration_energy_m2ps4(self) -> Optional[float]:
        """Mean squared corridor acceleration (diagnostic only)."""
        snap = get_active_snapshot()
        if snap is None:
            return None
        edges = {
            self._pre_edge,
            self._merge_edge,
            self._ramp_edge,
            self._shepherd_engine.down_edge,
        }
        squared = []
        for vehicle_id in snap.ordered_ids():
            if snap.road(vehicle_id) not in edges:
                continue
            acceleration = snap.accel(vehicle_id)
            if acceleration is not None:
                squared.append(float(acceleration) ** 2)
        return float(sum(squared) / len(squared)) if squared else 0.0

    def _corridor_acceleration_energy_sum_m2ps4(self) -> Optional[float]:
        """Raw per-step acceleration-squared sum matching oscillation energy."""
        snap = get_active_snapshot()
        if snap is None:
            return None
        edges = {
            self._pre_edge,
            self._merge_edge,
            self._ramp_edge,
            self._shepherd_engine.down_edge,
        }
        total = 0.0
        for vehicle_id in snap.ordered_ids():
            if snap.road(vehicle_id) not in edges:
                continue
            acceleration = snap.accel(vehicle_id)
            if acceleration is not None:
                total += float(acceleration) ** 2
        return total

    def _reset_idm_pattern_context(self) -> None:
        """Reset causal corridor-level temporal context at episode boundaries."""
        self._idm_pattern_initialized = False
        self._idm_severe_short_ema = 0.0
        self._idm_severe_long_ema = 0.0
        self._idm_wave_short_ema = 0.0
        self._idm_wave_long_ema = 0.0
        self._idm_pattern_context = {
            "idm_severe_level": 0.0,
            "idm_wave_energy_level": 0.0,
            "idm_severe_short_ema": 0.0,
            "idm_wave_short_ema": 0.0,
            "idm_severe_trend": 0.0,
            "idm_wave_energy_trend": 0.0,
        }

    def _update_idm_pattern_context(self) -> None:
        """Advance causal safety/oscillation levels and short-vs-long trends."""
        severe_count = self._corridor_severe_braking_count()
        wave_sum = self._corridor_acceleration_energy_sum_m2ps4()
        severe_level = float(severe_count or 0.0) / max(
            self.corridor_severe_braking_count_ref_per_s, 1e-6
        )
        wave_level = float(wave_sum or 0.0) / max(
            self.corridor_acceleration_energy_sum_ref_m2ps4, 1e-6
        )
        if not self._idm_pattern_initialized:
            self._idm_severe_short_ema = severe_level
            self._idm_severe_long_ema = severe_level
            self._idm_wave_short_ema = wave_level
            self._idm_wave_long_ema = wave_level
            self._idm_pattern_initialized = True
        else:
            short_tau = max(float(self.config.get("idm_pattern_short_tau_s", 10.0)), 1e-6)
            long_tau = max(float(self.config.get("idm_pattern_long_tau_s", 30.0)), 1e-6)
            short_alpha = 1.0 - float(np.exp(-self.simulation_step / short_tau))
            long_alpha = 1.0 - float(np.exp(-self.simulation_step / long_tau))
            self._idm_severe_short_ema += short_alpha * (severe_level - self._idm_severe_short_ema)
            self._idm_severe_long_ema += long_alpha * (severe_level - self._idm_severe_long_ema)
            self._idm_wave_short_ema += short_alpha * (wave_level - self._idm_wave_short_ema)
            self._idm_wave_long_ema += long_alpha * (wave_level - self._idm_wave_long_ema)
        self._idm_pattern_context = {
            "idm_severe_level": severe_level,
            "idm_wave_energy_level": wave_level,
            "idm_severe_short_ema": self._idm_severe_short_ema,
            "idm_wave_short_ema": self._idm_wave_short_ema,
            "idm_severe_trend": self._idm_severe_short_ema - self._idm_severe_long_ema,
            "idm_wave_energy_trend": self._idm_wave_short_ema - self._idm_wave_long_ema,
        }

    def _get_local_observations(self) -> Dict[str, np.ndarray]:
        """Build per-CAV local observations via the shared v2 assembler."""
        obs: Dict[str, np.ndarray] = {}
        vtypes = self._vtype_lookup()
        snap = get_active_snapshot()
        live_ids = snap.id_set if snap is not None else None
        merge_outflow = self._flow_tracker.sys_norm(self.flow_q_ref, self.flow_v_ref)
        eng = self._shepherd_engine
        delay_kw = dict(
            corridor_delay_rate=self._corridor_delay_rate_live(),
            delay_rate_ref=self.delay_rate_ref,
            delay_budget_scale=float(self.config.get("delay_budget_scale", 0.20)),
            stream_delay_gap=eng.stream_delay_gap(),
            stream_delay_gap_ref=float(self.stream_delay_gap_ref),
            stream_fairness_gap_scale=float(self.config.get("stream_fairness_gap_scale", 0.40)),
            ema_main_delay=eng.ema_main_delay,
            ema_ramp_delay=eng.ema_ramp_delay,
            zone_delay_abs_scale=float(self.config.get("zone_delay_abs_scale", 0.50)),
        )
        idm_preview_members: set[str] = set()
        idm_preview_masks: dict[str, tuple[int, ...]] = {}
        if (
            self.control_mode == "idm_residual_smoothing"
            and bool(self.config.get("use_idm_absorber_gate", False))
        ):
            if self._idm_residual_engine.cfg.decentralized_damping_enabled:
                idm_preview_members, idm_preview_masks, _ = (
                    self._idm_residual_engine.preview_decentralized_state(
                        self._active_cavs
                    )
                )
            else:
                idm_preview_members = self._idm_residual_engine.preview_active_members(
                    self._active_cavs
                )
        idm_context_by_cav: Dict[str, dict] = {}
        if bool(self.config.get("use_idm_controller_context", False)):
            for command in self._last_idm_residual_diag.get("commands", []):
                reference = command.get("reference", {})
                projected = command.get("projected", {})
                desired = float(command.get("desired_acceleration_mps2", 0.0))
                executed = float(projected.get("acceleration_mps2", desired))
                idm_context_by_cav[str(command["vehicle_id"])] = {
                    "idm_nominal_acceleration": float(
                        command.get("nominal_acceleration_mps2", 0.0)
                    ),
                    "idm_filtered_reference": float(reference.get("filtered_reference_mps", 0.0)),
                    "idm_reference_error": float(
                        reference.get("raw_reference_mps", 0.0)
                        - reference.get("filtered_reference_mps", 0.0)
                    ),
                    "idm_projection_delta": executed - desired,
                }
        for cav_id in self._active_cavs:
            if live_ids is not None:
                if cav_id not in live_ids:
                    continue
            elif cav_id not in traci.vehicle.getIDList():
                continue
            loc = self._obs_assembler.build(
                cav_id,
                cav_ids=self._active_cavs,
                vtype_lookup=vtypes,
                density_id=self._density_id,
                pre_edge=self._pre_edge,
                merge_edge=self._merge_edge,
                ramp_edge=self._ramp_edge,
                ramp_lane=self._ramp_lane,
                zone_config=self._zone_config,
                merge_outflow=merge_outflow,
                metering_load=float(getattr(self._shepherd_engine, "_metering_load", 0.0)),
                cav_count_norm=float(
                    len(self._active_cavs) / 48.0
                    if bool(self.config.get("use_cav_count", False))
                    else 0.0
                ),
                metastable_health=self._last_met_health,
                merge_dv=self._last_merge_dv,
                runway_margin=self._last_margin,
                idm_absorber_gate=float(cav_id in idm_preview_members),
                **self._idm_pattern_context,
                **idm_context_by_cav.get(cav_id, {}),
                **delay_kw,
            )
            if self.action_masking:
                if self._idm_residual_engine.cfg.decentralized_damping_enabled:
                    mask = np.asarray(
                        idm_preview_masks.get(cav_id, (1, 0, 0, 0, 0)),
                        dtype=np.float32,
                    )
                else:
                    mask = self._shepherd_engine.eligibility_mask(cav_id)
                # Truncate/pad assembler features to raw size, then append mask.
                raw_n = self._raw_local_dim
                if loc.shape[0] < raw_n:
                    loc = np.concatenate([loc, np.zeros(raw_n - loc.shape[0], dtype=np.float32)])
                elif loc.shape[0] > raw_n:
                    loc = loc[:raw_n]
                loc = np.concatenate([loc, mask.astype(np.float32)])
            else:
                if loc.shape[0] < self._local_obs_dim:
                    loc = np.concatenate(
                        [
                            loc,
                            np.zeros(self._local_obs_dim - loc.shape[0], dtype=np.float32),
                        ]
                    )
                elif loc.shape[0] > self._local_obs_dim:
                    loc = loc[: self._local_obs_dim]
            obs[cav_id] = loc
        return obs

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Build observation dict; pack peers + trust weights for MAPPO critic."""
        local = self._get_local_observations()
        if not self.centralized_critic:
            return local

        # Compute peer trust weights for each ego CAV.
        packed = {}
        for cav_id, loc in local.items():
            # Trust weight for each peer CAV from TrustTracker.
            trust_weights = {}
            for peer_id in local:
                if peer_id == cav_id:
                    continue
                # TrustTracker stores (ego, neighbor) trust; default 0.5, CAV prior 0.8.
                trust_weights[peer_id] = self._trust.get(cav_id, peer_id)
            packed[cav_id] = pack_mappo_observation(
                loc,
                local,
                self.max_cavs_critic,
                trust_weights=trust_weights if trust_weights else None,
                ego_id=cav_id,
            )
        return packed

    def _update_gridlock_counter(self) -> None:
        """Count consecutive steps where the whole network is stopped."""
        if self._gridlock_terminate_steps <= 0:
            return
        snap = get_active_snapshot()
        if snap is None or len(snap.ids) < 10:
            self._gridlock_steps = 0
            return
        speeds = [float(s) for s in (snap.speed(vid) for vid in snap.ids) if s is not None]
        if speeds and sum(speeds) / len(speeds) < 1.0:
            self._gridlock_steps += 1
        else:
            self._gridlock_steps = 0

    def _episode_end_reason(self) -> Optional[str]:
        """Return 'truncated', 'terminated', or None if the episode continues."""
        if self.max_episode_steps is not None and self._elapsed_steps >= self.max_episode_steps:
            return "truncated"
        if (
            self._gridlock_terminate_steps > 0
            and self._gridlock_steps >= self._gridlock_terminate_steps
        ):
            return "truncated"
        if traci.simulation.getMinExpectedNumber() == 0 and len(get_active_vehicle_ids()) == 0:
            return "terminated"
        return None

    def _is_done(self) -> bool:
        """Return True when the episode should end."""
        return self._episode_end_reason() is not None

    def get_agent_ids(self) -> list:
        """Return active CAV IDs (used by RLlib for policy mapping)."""
        return list(self._active_cavs)
