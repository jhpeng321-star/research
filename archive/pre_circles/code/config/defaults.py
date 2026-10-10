"""Default configuration for the TrafficShepherd multi-agent SUMO environment."""

from pathlib import Path

from config.constants import RL_SPEED_MODE


# research-dense lives outside D:\research (parallel to research_v2); reuse SUMO net/demand.
_RL_ENV_ROOT = Path(__file__).parent.parent
_SUMO_PROJECT_ROOT = Path(r"D:\research\highway_merge_shepherd")


DEFAULT_CONFIG = {
    "sumo_binary": "sumo",  # or "sumo-gui" for debugging
    "sumo_config": "osm.hyper.sumocfg",  # denser than stock dense (3400+900)
    "cav_penetration": 0.10,  # 0.05, 0.10, 0.20
    "cav_selection_seed": 42,  # master seed: unseeded reset() draws per-episode CAV/HDV seeds from it; reset(seed=...) overrides for eval
    # When True, every newly selected CAV is setType'd to CAV_VTYPE (normal_car).
    # Coast / no-op then follows normal Krauss params, not aggressive/cautious.
    "cav_force_normal_type": True,
    "sumo_seed": None,  # SUMO --seed; None = SUMO default
    "max_episode_steps": 3600,  # one-hour demand
    "simulation_step": 1.0,  # seconds per env / policy step
    # Finer SUMO CF tick (policy still 1 Hz). Probe: L0 creep coll → 0 at 0.5.
    "sumo_step_length": 0.5,
    "max_speed": 35.0,  # m/s (~126 km/h)
    "observation_radius": 200.0,  # meters around CAV (shared by all baselines)
    "neighbor_count": 5,  # K nearest neighbors in observation
    "reward_weights": {
        # v9c gate-only ablation: keep Phase-1 relative gate + fairness_mask,
        # restore v9 delay_win / control_cost (v9b retune hurt hard seeds).
        "efficiency": 0.35,
        "safety": 1.25,
        "stability": 0.25,
        "fairness": 1.20,
        "delay_budget": 1.10,
        "delay_win": 0.30,
        "control_cost": 0.25,
        "mask_penalty": 0.2,
        "ineligible_penalty": 0.5,
        # Slight shaping so recover/vacate are learnable when eligible; kept
        # small vs fairness (1.20) / delay_budget (1.10). fair_balance/protect
        # stay unlisted (no bonus).
        "mode_use_bonus": 0.08,
        # v10 learned abstention (spec 2026-08-04): v10 recipe active.
        # exp2 (2026-08-04): tax raised 0.15 -> 0.25; structural-seed harm (9/50)
        # was the binding constraint at v10 i85 (33/50 dominated).
        "fair_win": 0.20,
        "easy_intervention_tax": 0.15,
        # Corridor-level TTC penalty (2026-08-04 late): the TTC redistribution
        # is systemic (HDV-HDV compression waves anywhere in the corridor), so
        # only a corridor-level term can see it. -count per sub-threshold
        # TTC pair per step on pre/merge/ramp/down lanes; small weight — exp2
        # showed strong penalties push passivity.
        "corridor_ttc_penalty": 0.2,
        "flow": 1.0,
        "tail": 0.0,
    },
    # Threshold (s) for the corridor TTC count fed to corridor_ttc_penalty.
    # Matches the eval-side safety ruler (user steer: 1.5 s).
    "corridor_ttc_threshold": 1.5,
    # Modes that receive ``mode_use_bonus`` when eligible (not remapped to coast).
    # wave_damp (opt-in, 2026-08-05) NEEDS the bonus: free wasn't enough —
    # at 0.0 usage the policy never samples it, so its benefit (calmed
    # neighbors via the safety share term) can never be learned. The bonus
    # breaks the zero-usage/zero-signal cycle; safety then sustains or
    # rejects it on merit. No effect on 5-mode runs (index 5 never picked).
    "mode_use_bonus_modes": ["recover", "vacate_drop", "wave_damp"],
    # Per-mode bonus values (default 1.0). wave_damp gets 0.3: at 0.08 it
    # never beat the trained fair_balance preference (0.0 usage on every
    # seed class), so its benefit never got sampled. 0.3 forces exploration;
    # always-on wave_damp is gentle-coast-like, bounding the distortion.
    "mode_use_bonus_overrides": {"wave_damp": 0.3},
    # Keep ego collision sting for rare CAV-involved events (weight is lower).
    "safety_collision_penalty": 6.5,
    # Need-gated control cost: intervening on calm upstream (high s_norm) costs
    # ~-1; intervening into a jam costs closer to -control_cost_c_min.
    "control_cost_need_gated": True,
    "control_cost_c_min": 0.25,
    "control_cost_health": "upstream",  # "upstream" (s_norm) | "sys" (sys_norm)
    # Eval/train hard mask (legacy flow_stab); off for gated Discrete(5) redesign.
    "shepherd_health_mask": False,
    "shepherd_health_mask_threshold": 0.8,
    "shepherd_health_mask_modes": [],
    # Fairness mask: off by default for frozen eval parity with stock i40/i150.
    # Phase1-only peek on v9 i40 was 10/20 (worse than stock 11/20). Enable via
    # env override / train recipe when intentionally testing the relative gate.
    "shepherd_fairness_mask": False,
    "shepherd_fairness_mask_threshold": 0.1,
    "shepherd_fairness_mask_modes": ["fair_balance"],
    "shepherd_fairness_mask_ema_alpha": 0.2,
    # Easy-intervention mask (mask-v1, 2026-08-04): when the corridor is
    # already REALLY fair in absolute terms (|gap| <= gap_abs), discount the
    # flow-hurting discretionary modes. User steer: absolute fairness, NOT
    # "fair vs the paired no_op ref" (set gap_abs 0 to fall back to the
    # ref-based condition |gap| <= gap_ref). Dense coast |gap| ~ 0.10, so
    # 0.05 = clearly balanced. Soft by default: the mask bit becomes
    # ``discount`` so the model applies log(discount) to the logits
    # (0.1 -> -2.3 penalty) instead of an outright ban; apply() does NOT
    # remap. 0.0 = hard ban (bit 0 + coast remap), 1.0 = no effect.
    # protect is exempt (safety); coast is never gated.
    "shepherd_easy_intervention_mask": True,
    "shepherd_easy_intervention_mask_modes": [
        "fair_balance",
        "recover",
        "vacate_drop",
    ],
    "shepherd_easy_intervention_mask_discount": 0.1,
    "shepherd_easy_intervention_mask_gap_abs": 0.05,
    # Graduated mask (2026-08-04): fade the discount linearly from
    # ``discount`` at |gap| <= gap_abs up to no-effect at |gap| >= gap_outer.
    # The fairer the corridor, the stronger the logit penalty.
    "shepherd_easy_intervention_mask_graduated": True,
    "shepherd_easy_intervention_mask_gap_outer": 0.10,
    # Hybrid mask (2026-08-05): hard BAN when |gap| <= this (0 = off, pure
    # soft/graduated). Targets structural-harm seeds — ultra-fair corridor,
    # nothing to win — without paying the soft mask's leak elsewhere.
    "shepherd_easy_intervention_mask_hard_below": 0.0,
    # Metastable-health mask (Experiment C, 2026-08-11): the RSU aggregates
    # merge-zone CAV data each step into h in [0,1] (Krauss-Wagner-Gawron 1997
    # §V hysteresis + §VI pre-breakdown variance). Graduated soft mask: the
    # healthier the merge (h high), the stronger the logit penalty on the
    # discretionary metering/recovery modes — the policy is pushed to coast;
    # as h falls the discount rises to 1.0 (open) so the CAV can meter/recover
    # to restore metastable. Coast always legal; protect exempt (safety).
    "metastable_health_mask": False,
    "metastable_health_mask_modes": ["fair_balance", "recover"],
    "metastable_health_mask_discount": 0.05,
    # h window: discount at h <= h_lo (worst health), 1.0 (no effect) at
    # h >= h_hi (healthiest). Linear fade between.
    "metastable_health_mask_h_lo": 0.2,
    "metastable_health_mask_h_hi": 0.7,
    # RSU health signal calibration (measured, docs/metastable_merge_control.md):
    # density (veh/km/lane) and speed (m/s) ramps + variance-ratio ramp.
    # The variance warning (Krauss 1997 §VI) only acts while rho is still
    # healthy (rho_h > 0.5) — at critical load the merge-zone variance is
    # intrinsically high, so it is gated off once density takes over.
    "metastable_health_ema_alpha": 0.3,
    "metastable_health_rho_free": 2.0,
    "metastable_health_rho_crit": 5.5,
    "metastable_health_v_free": 25.0,
    "metastable_health_v_break": 12.0,
    "metastable_health_var_ratio_lo": 3.0,
    "metastable_health_var_ratio_hi": 6.0,
    "metastable_health_var_baseline_window": 120,
    # Coast-force ablation: off by default (scoped CF 26/50, protect-exempt
    # peek 10/20 — neither beat stock i40 27/50). Enable only in eval scripts.
    "shepherd_coast_force": False,
    "shepherd_coast_force_delay_slack": 0.0,
    "shepherd_coast_force_modes": ["fair_balance", "guard"],
    "shepherd_coast_force_hold_steps": 1,
    # 0 = arm immediately when healthy; >0 lets hard seeds intervene early.
    "shepherd_coast_force_min_step": 0,
    # Stream delay-gap scale for R_fair (|main-ramp| mean delay *rate*).
    # Dense coast |gap|≈0.10; 0.40 keeps fairness informative. Excess over
    # paired ``stream_delay_gap_ref`` is what gets penalized (relative G).
    "stream_fairness_gap_scale": 0.40,
    # One-sided delay budget: penalize corridor delay rate above paired
    # ``delay_rate_ref`` (= mean_time_loss/mean_tt under no_op).
    "delay_budget_scale": 0.20,
    # Modest delay-win credit when corridor rate beats paired no_op (same units).
    "delay_win_scale": 0.20,
    # Absolute main/ramp delay-rate normalization in zone obs slots 10–11.
    "zone_delay_abs_scale": 0.50,
    # No dead indices — all five modes are live physics (eligibility-gated).
    "shepherd_disabled_modes": [],
    # Append per-mode eligibility bits to local obs and mask illegal logits
    # in MAPPOModel (shepherd Discrete only). Coast always remains legal.
    "action_masking": True,
    "project_root": _SUMO_PROJECT_ROOT,
    "eval_mode": False,  # when True, write tripinfo + FCD for offline metrics
    "eval_tripinfo_output": None,  # default: project_root / "tripinfos.eval.xml"
    "eval_fcd_output": None,  # default: project_root / "fcd.eval.xml"
    "eval_stats_output": None,  # default: project_root / "stats.eval.xml"
    "eval_fcd_period": 1,  # seconds between FCD snapshots
    # Traffic Shepherd discrete action modes (0..4) — single source: constants.MODE_NAMES
    # Default mode set: the original 5. "wave_damp" is registered in
    # MODE_NAMES but opt-in only (train.py --enable-wave-damp appends it) —
    # the champion's 5-mode checkpoints stay compatible.
    "shepherd_modes": [
        "coast",
        "fair_balance",
        "recover",
        "vacate_drop",
        "guard",
    ],
    "base_tau": 1.0,  # default SUMO desired time headway (s)
    # Legacy tau/speed tables kept for tests that still call _gentle_mode; live
    # modes use the gated knob blocks below.
    "tau_deltas": {
        "coast": 0.0,
        "fair_balance": 0.25,
        "recover": -0.2,
        "vacate_drop": 0.0,
        "guard": 0.5,
    },
    "speed_factors": {
        "coast": None,
        "fair_balance": 0.90,
        "recover": None,
        "vacate_drop": 0.90,
        "guard": 0.85,
    },
    # fair_balance: tip-only slowdown of the advantaged stream when |G| large
    "fair_balance_gap_tau": 0.05,  # |ema_main - ema_ramp| delay-rate gap
    "fair_balance_tip": 120.0,  # m remaining to tip
    "fair_balance_factor": 0.90,
    "fair_balance_floor": 8.0,
    "fair_balance_ema_alpha": 0.2,
    # Adaptive gentleness (2026-08-05): fair_balance slowdown factor is
    # modulated by the trailing corridor-TTC EMA — ``adaptive_gentle_factor``
    # when the corridor is smooth, base fair_balance_factor when chaotic.
    # Direction is empirical: global 0.93 was TTC-WORSE, so the live
    # hypothesis is firmer-on-smooth (factor < 0.90). Off by default.
    "shepherd_adaptive_gentle": False,
    "shepherd_adaptive_gentle_factor": 0.85,
    "shepherd_adaptive_gentle_ttc_lo": 0.5,
    "shepherd_adaptive_gentle_ttc_hi": 2.0,
    "shepherd_adaptive_gentle_ema_alpha": 0.1,
    # Recover TTC gate (2026-08-05): recover is load-bearing on chaotic
    # corridors but injects TTC into smooth ones — ban it when the trailing
    # corridor-TTC EMA is below this threshold. Off by default.
    "shepherd_recover_ttc_gate": False,
    "shepherd_recover_ttc_lo": 0.5,
    # Follower-aware execution (user steer 2026-08-05): mode decisions stay
    # discrete, but braking magnitude adapts to the follower behind — full
    # strength when the road behind is open (gap >= far), gentle cap when a
    # close follower (gap <= near) would inherit the brake as a wave.
    "shepherd_follower_aware_exec": False,
    "shepherd_follower_gentle_cap": 0.97,
    "shepherd_follower_gap_near": 15.0,
    "shepherd_follower_gap_far": 60.0,
    # recover_speed: gated queue discharge on pre-merge
    "recover_merge_clear": 10.0,
    "recover_down_clear": 18.0,
    "recover_ego_max": 12.0,
    "recover_push": 1.5,
    "recover_v_cap": 18.0,
    # Recover sign gate (2026-08-04): recover is corrective for a
    # MAIN-disadvantaged corridor — only eligible when the stream gap is
    # known and gap > tau. Firing on fair/ramp-disadvantaged corridors
    # widens the gap (structural-harm mechanism).
    "shepherd_recover_gap_sign_gate": True,
    "shepherd_recover_gap_tau": 0.05,
    # vacate_drop: leave merge L0
    "vacate_lane_duration": 3.0,
    "vacate_soft_factor": 0.90,
    # protect: emergency headway
    "protect_ttc": 1.5,
    "protect_factor": 0.85,
    "protect_tau_delta": 0.5,
    # Retired knobs (compat)
    "meter_trigger_speed": 14.0,
    "meter_factor": 0.88,
    "meter_speed_floor": 8.0,
    "yield_pet_threshold": 2.0,
    "yield_tip_distance": 80.0,
    "yield_factor": 0.85,
    "yield_tau_delta": 0.4,
    # Retired flow_stab knobs (kept so old configs don't KeyError if merged)
    "flow_stab_near": 150.0,
    "flow_stab_far": 200.0,
    "flow_stab_ema_alpha": 0.3,
    "flow_stab_deadband": 0.5,
    "flow_stab_a_max_dec": 1.0,
    "flow_stab_a_max_acc": 1.0,
    "flow_stab_preview_gain": 0.5,
    # Mode comfort bounds
    "mode_max_decel": 1.0,
    "mode_max_accel": 1.25,
    "mode_deadband": 0.5,
    # Driver types come from the route file's vTypeDistribution (single source
    # of truth, shared by All-HDV baseline and RL runs). Set True only for
    # legacy randomTrips route files without a distribution.
    "assign_hdv_types": False,
    "hdv_type_proportions": [0.1, 0.7, 0.2],  # legacy TraCI assignment
    "hdv_type_seed": 42,  # legacy TraCI assignment; unseeded reset() now draws per-episode seeds from the cav_selection_seed master RNG
    # Selfish / Flat-MARL use continuous acceleration; Shepherd uses modes.
    "control_mode": "shepherd",  # "shepherd" | "continuous_acceleration" | "idm_residual_smoothing"
    # IDM-native selected-absorber analytical controller. Learned actions are
    # bounded residual accelerations around the frozen analytical zero.
    "idm_residual_residual_min_mps2": -0.5,
    "idm_residual_residual_max_mps2": 0.3,
    "idm_residual_minimum_runway_m": 150.0,
    "idm_residual_minimum_hold_steps": 10,
    "idm_residual_front_ttc_floor_s": 3.0,
    "idm_residual_rear_ttc_floor_s": 4.0,
    "idm_residual_reference_mode": "disturbance_p30",
    "idm_residual_filter_tau_s": 0.0,
    "idm_residual_action_family": "residual",
    "idm_residual_structured_action_filter_tau_s": 5.0,
    "idm_residual_gain_min": 0.50,
    "idm_residual_gain_max": 1.50,
    "idm_residual_reference_offset_min_mps": -2.0,
    "idm_residual_reference_offset_max_mps": 2.0,
    # Decentralized structured controller: every eligible CAV emits a local
    # five-mode V2V bid; no learned lane-level selector sees other candidates.
    "idm_residual_decentralized_damping_enabled": False,
    "idm_residual_decentralized_option_steps": 20,
    "idm_residual_decentralized_minimum_predicted_dose_s": 16.0,
    "idm_residual_decentralized_require_secure_gap": True,
    "idm_residual_decentralized_terminal_rear_headroom_m": 5.0,
    "idm_residual_decentralized_action_set": "gain_reference_v1",
    "use_idm_absorber_gate": False,
    "use_idm_controller_context": False,
    "idm_pattern_short_tau_s": 10.0,
    "idm_pattern_long_tau_s": 30.0,
    "idm_severe_brake_threshold_mps2": -3.0,
    "corridor_severe_braking_count_ref_per_s": 1.0,
    "corridor_acceleration_energy_sum_ref_m2ps4": 1.0,
    "idm_damping_reward_weights": {
        "efficiency": 0.10,
        "safety": 2.0,
        "stability": 0.5,
        "delay_budget": 0.10,
        "corridor_ttc_penalty": 1.0,
        "corridor_severe_braking": 1.0,
        "corridor_wave_energy": 1.0,
        # Local onset risk remains an observation/gate signal. The Phase-5
        # audit found it can move opposite corridor oscillation energy.
        "wave": 0.0,
        # Phase-5 audit: the actuator is already bounded, absorber-gated,
        # low-pass filtered, and safety-projected. A quadratic effort tax
        # reversed physically beneficial residual corrections.
        "residual_effort": 0.0,
    },
    "neighbor_radius": 200.0,
    "max_neighbors": 20,
    # Legacy observation semantics remain a configured-range contract whose
    # default is 500 m. Extended-range diagnostics must select their own
    # explicit versioned contract instead of mutating this default silently.
    "rsu_observation_contract": "classic_rsu_configured_range_v1",
    "zone_comm_range": 500.0,
    "slow_speed_threshold": 5.0,
    "trust_lambda": 0.2,
    "selfish_reward_weights": {
        # Speed alone must not saturate the return near free-flow.
        "speed": 0.4,
        "comfort": 1.0,
        "safety": 2.0,
    },
    # RL accel control: SUMO clamps setSpeed to the safe car-following speed
    # (see config.constants.RL_SPEED_MODE).
    "rl_speed_mode": RL_SPEED_MODE,
    # "selfish" | "shepherd" — Flat-MARL uses continuous acceleration + Shepherd reward.
    "reward_mode": None,  # None → infer from control_mode
    "use_v2_obs": None,  # None → True for all learned-policy training
    "centralized_critic": False,  # MAPPO: pack all-CAV obs for mean-pool critic
    # Sized from tools/measure_max_active_cavs.py: max concurrent active CAVs
    # at p20 is 26 across light/medium/dense (seeds 42/43/44). 32 gives
    # headroom without a 64-slot padded critic forward every step.
    "max_cavs_critic": 32,
    "cav_warmup_steps": 300,
    # If warmup finds no CAV, keep stepping / restart SUMO this many times
    # before failing (avoids empty multi-agent resets that hang RLlib).
    "cav_empty_extend_steps": 600,
    "cav_empty_retries": 2,
    # When set (e.g. ["light","medium","dense"]), reset() samples a density
    # each episode and rewrites sumo_config / fairness_r_ref / density_id.
    "density_choices": None,
    # Easy-seed oversampling for paired-pool training (v10 recipe active):
    # 0.25 / 2.0 lifts easy-episode share from ~25% to ~40% of sampled episodes.
    "paired_easy_quantile": 0.25,
    "paired_easy_boost": 2.0,
    # When set (e.g. [0.05, 0.1, 0.2]), reset() samples a CAV penetration each
    # episode (multi-penetration training for cross-rate generalization).
    "penetration_choices": None,
    # All-HDV merge-edge discharge reference for the R_eff flow term (per-step
    # EMA of vehicles completing the merge edge, vs FLOW_Q_REF_BY_DENSITY).
    "flow_q_ref": 0.9,
    # All-HDV upstream (pre-merge) mean-speed reference for the R_eff flow
    # term (vs FLOW_V_REF_BY_DENSITY) — the queue-growth detector.
    "flow_v_ref": 27.0,
    "flow_ema_alpha": 0.1,  # smoothing for merge discharge
    # Congestion guard: below this ego speed (m/s), DECELERATING modes
    # (speed_factor < 1.0) are vetoed to non_intervention physics, so the
    # policy can never brake the network into a standing queue. 0 disables.
    "congestion_guard_speed": 8.0,
    # Gridlock early termination: truncate the episode after this many
    # consecutive steps with network mean speed < 1 m/s (>=10 vehicles).
    "gridlock_terminate_steps": 60,
}


# All-HDV merge-edge discharge (veh/step, steady state t=300-900, seed 42),
# measured from fcd.{light,medium,dense}.seed42.xml as vehicles leaving edge
# 1487606179 per step. Reference for the R_eff flow term q_norm = q_ema/q_ref.
FLOW_Q_REF_BY_DENSITY = {
    "light": 0.56,
    "medium": 0.90,
    "dense": 0.95,
    # Provisional (=dense) until All-HDV FCD re-measure on hyper.
    "hyper": 0.95,
}


# All-HDV pre-merge (upstream mainline edge 1191554636) mean speed (m/s,
# steady state t=300-900, seed 42), from fcd.{light,medium,dense}.seed42.xml.
# Reference for the R_eff upstream-speed term s_norm = v_ema / v_ref.
FLOW_V_REF_BY_DENSITY = {
    "light": 29.7,
    "medium": 27.0,
    # 2026-07-30 re-measured across seeds 42-46 (was 19.7 from seed 42 only,
    # a lucky draw: per-seed 13.5-22.0, mean 17.48). Q ref unchanged (0.948≈0.95).
    "dense": 17.5,
    # Provisional (=dense) until All-HDV FCD re-measure on hyper.
    "hyper": 17.5,
}


# All-HDV 局部逐步 timeLoss 率参考值（R_fair 的基线 r_ref）。2026-07-30 重新标定：
# 与 _fairness 完全同口径（同 edge、200 m 内的局部均值），来自 all-HDV FCD
# (fcd.{medium,dense}.seed42-46.xml, runs/eval/analysis/fairness_local_rates.json)。
# 公式已用 network 均值验证（0.218≈0.22, 0.381≈0.38）。light 待补测（沿用旧值）。
FAIRNESS_R_REF_BY_DENSITY = {
    "light": 0.074,
    "medium": 0.24,
    "dense": 0.45,
    # Provisional (=dense) until local timeLoss re-measure on hyper.
    "hyper": 0.45,
}


# R_tail 参考值：all-HDV 局部 p90 timeLoss 率（与 _tail_penalty 同口径：
# 同 edge、200 m 内局部车辆 per-step delay rate 的 p90），来自 all-HDV FCD
# (fcd.{medium,dense}.seed42-46.xml)。mean 版本验证与 R1.2 一致。
# light 待补测（暂用 medium 值）。
TAIL_R_REF_BY_DENSITY = {
    "light": 0.34,
    "medium": 0.34,
    "dense": 0.53,
    # Provisional (=dense) until p90 timeLoss re-measure on hyper.
    "hyper": 0.53,
}


def merge_config(user_config: dict) -> dict:
    """Return a full config by overriding defaults with user-supplied values."""
    config = DEFAULT_CONFIG.copy()
    if user_config is not None:
        config.update(user_config)
    # Resolve project_root to an absolute path so SUMO configs are found regardless of cwd.
    config["project_root"] = Path(config["project_root"]).resolve()
    return config
