"""Flat, validated view of the training configuration used by the train layer.

``TrainSettings`` carries exactly the values the loss composer, unroll runner and
trainer consume.  It is built from the documented compatibility contract
``TrainConfig.to_legacy_flat()`` (one mapping point; the field names are the
legacy ``WmSTrainConfig`` names), so the nested schema owned by ``wms.config`` can
evolve without touching numerics.  Validation mirrors the legacy trainer and adds
the fail-closed refusal of the hard residual clamp (change-log #3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional

CAV_DRIVE_MODES = ("teacher", "closed_loop")
ROLLOUT_MODES = ("residual", "driver_params", "hybrid")
LANE_RULES = ("mobil", "forced_merge_only", "learned_discretionary")
LANE_GAP_RULES = ("fixed", "speed")
JUNCTION_FIX_MODES = ("off", "ffill")
CONTROLLER_HYGIENE_MODES = ("off", "strict")
#: Checkpoint-selection policies (SCHED multi_horizon; INFRAV5 multi_horizon_v2).
SELECTION_MODES = ("long_horizon", "multi_horizon", "multi_horizon_v2")
HARD_CAP_FIELDS = ("residual_cap_mps2", "residual_cap_neg_mps2")


def validate_bounds(bound: Optional[float], neg_bound: Optional[float],
                    cap: Optional[float] = None, cap_neg: Optional[float] = None) -> None:
    """Soft residual bounds are finite and positive and exclude the legacy hard caps."""
    for value in (bound, neg_bound, cap, cap_neg):
        if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
            raise ValueError("residual soft bounds must be finite and > 0")
    if neg_bound is not None and bound is None:
        raise ValueError("negative soft bound requires residual_soft_bound_mps2")
    if bound is not None and (cap is not None or cap_neg is not None):
        raise ValueError("residual soft bounds cannot be combined with legacy residual caps")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class TrainSettings:
    """Immutable training hyper-parameters (defaults == legacy ``WmSTrainConfig``)."""

    unroll_curriculum_s: tuple[float, ...] = (10.0, 30.0, 60.0, 120.0)
    stage_steps: Optional[tuple[int, ...]] = None
    num_workers: int = 0
    epochs_per_stage: int = 1
    steps_per_epoch: int = 2000
    batch_size: int = 8
    #: V10HORIZON: optional per-curriculum-stage batch size (memory control at long horizons).
    batch_size_per_stage: Optional[tuple[int, ...]] = None
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    position_huber_delta_m: float = 5.0
    speed_huber_delta_mps: float = 1.0
    lane_weight: float = 1.0
    position_weight: float = 1.0
    speed_weight: float = 1.0
    # V10BRAKEWAVE (default 1 = OFF, bit-identical): per-step position/speed multiplier on
    # followers whose same-lane recorded leader (<= 40 m) is decelerating below -1 m/s^2.
    brake_follow_weight: float = 1.0
    # V10CAVFOLLOW (default 0 = OFF, bit-identical): extra per-step position/speed multiplier
    # on HDV followers of a recorded same-lane CAV leader (<= 40 m) braking below -1 m/s^2.
    brake_follow_cav_weight: float = 0.0
    # BRAKE_DIAG V10PERSIST (default 0 = OFF, bit-identical): multi-step onset persistence,
    # onset quantiles and optional inverse-frequency event balancing.
    brake_persist_weight: float = 0.0
    brake_persist_quantile_weight: float = 0.0
    brake_persist_event_balance: float = 0.0
    brake_persist_window_s: float = 4.0
    step_weight_decay: float = 0.01
    truncation_steps: int = 20
    truncation_steps_per_stage: Optional[tuple[int, ...]] = None
    lr_per_stage: Optional[tuple[float, ...]] = None
    driver_grad_across_chunks: bool = False
    history_features: int = 6
    history_s: Optional[float] = None
    history_stride_s: float = 1.0
    device: str = "cpu"
    threads: int = 6
    state_noise_std_m: float = 0.0
    state_noise_std_mps: float = 0.0
    rollout_mode: str = "residual"
    lane_head: str = "off"
    lane_rule: str = "mobil"
    lane_gap_rule: str = "fixed"
    residual_magnitude_weight: Optional[float] = None
    driver_sidecar: Optional[str] = None
    out_dir: str = "runs/wms"
    seed: int = 101
    black_box: bool = False
    n_controllers: int = 7
    controller_embed_dim: int = 16
    checkpoint_residual: bool = False
    traffic_loss: bool = False
    traffic_bin_s: float = 30.0
    traffic_weight: float = 0.1
    traffic_free_speed_mps: float = 13.89
    traffic_ramp_start_s: float = 30.0
    traffic_ramp_end_s: float = 120.0
    entries_v2: bool = False
    entry_pad: int = 8
    window_list: Optional[str] = None
    exclude_window_list: Optional[str] = None
    val_window_list: Optional[str] = None
    val_batches: int = 8
    val_max_windows: int = 0
    projection_enabled: bool = False
    projection_min_gap_m: float = 2.0
    residual_cap_mps2: Optional[float] = None
    lane_logit_cap: Optional[float] = None
    stability_weight: float = 0.0
    stability_margin: float = 1.0
    stability_probes: int = 1
    conservation_enabled: bool = False
    conservation_weight: float = 0.0
    loss_share_logging: bool = False
    switch_window_fraction: float = 0.0
    asym_speed_alpha: float = 1.0
    decel_weight: float = 1.0
    decel_threshold_mps2: float = 3.0
    propagation_weight: float = 0.0
    prop_window_s: float = 3.0
    prop_max_gap_m: Optional[float] = None
    residual_cap_neg_mps2: Optional[float] = None
    cav_drive: str = "teacher"
    residual_soft_bound_mps2: Optional[float] = None
    residual_soft_bound_neg_mps2: Optional[float] = None
    residual_mag_weight: float = 0.0
    lane_rate_weight: float = 0.0
    lane_rate_target: float = 0.0123
    select_by_long_horizon: bool = False
    val_every: int = 200
    gate_kind: str = "cumsum"
    gate_g_max_init: float = 0.30
    gate_tau_steps_init: float = 50.0
    lane_gate_kind: str = "shared"
    lane_gate_g_max_init: float = 0.16
    lane_gate_tau_steps_init: float = 30.0
    checkpoint_step: bool = False
    grad_decay_per_step: float = 1.0
    init_from: Optional[str] = None
    flip_rate_weight: float = 0.0
    flip_rate_allowance: float = 0.00124327163
    # WMLANE (default OFF)
    lane_model: str = "residual"
    lane_hazard_hidden: int = 64
    lane_hazard_cap: float = 10.0
    driver_update: str = "off"
    driver_param_mode: str = "legacy"
    driver_ema_update: bool = True
    driver_update_rate: float = 0.3
    driver_reveal_prob: float = 0.0
    driver_reveal_max_s: float = 10.0
    lane_event_weight: float = 0.0
    lane_event_gamma: float = 2.0
    lane_event_max_weight: float = 20.0
    # C4 WM v2 loss (2026-10-13, WMLOSS_NOTES.md): aggregate-consistency loss (0 = off) and
    # the adaptive entry pad / uncapped backlog of the training entry process.
    aggregate_weight: float = 0.0
    aggregate_field_weight: float = 1.0
    aggregate_reward_weight: float = 1.0
    aggregate_flow_weight: float = 0.0
    aggregate_bin_s: float = 5.0
    aggregate_every_steps: int = 1
    aggregate_horizon_start_s: float = 0.0
    aggregate_horizon_full_s: float = 0.0
    aggregate_cell_temp_m: float = 5.0
    # A5 (2026-10-19, default OFF = bit-identical): long-horizon aggregate delay loss.
    agg_delay_weight: float = 0.0
    agg_delay_paired_weight: float = 0.0
    agg_delay_start_s: float = 120.0
    agg_delay_horizon_s: float = 300.0
    agg_delay_fade_s: float = 0.0
    agg_delay_bin_s: float = 5.0
    agg_delay_merge_s0_m: float = 0.0
    agg_delay_merge_s1_m: float = 75.44
    agg_delay_merge_lane: int = 2
    agg_delay_road_end_m: float = 553.33
    agg_delay_sections_m: tuple[float, ...] = (-400.0, 0.0, 75.44, 400.0)
    agg_delay_queue_speed_mps: float = 5.0
    agg_delay_region_temp_m: float = 5.0
    agg_delay_queue_temp_mps: float = 1.0
    agg_delay_delay_scale_veh_s: float = 1.0
    agg_delay_queue_scale_veh: float = 1.0
    agg_delay_throughput_scale_veh: float = 1.0
    agg_delay_huber_delta: float = 1.0
    entry_pad_auto: bool = False
    entry_pad_max: int = 256
    entry_insert_rule: str = "v2"  # ENTRYFIX/V6ENTRY: "v2" | "sumo_like" | "sumo_like_v2"
    entry_queue_state: bool = False  # V9ENTRY: recorded t0 backlog + demand-rate arrivals
    junction_fix: str = "off"  # JUNCTIONFIX: data-side only (wms.data.junction_fix)
    controller_hygiene: str = "off"  # DATAHYG: data-side only (wms.data.controller_hygiene)
    # LANEFIX (2026-10-14, default OFF): model switches mirrored for validation; the
    # decouple flag is applied to the model by the trainer.
    lane_decision: str = "argmax"
    lane_logit_offsets: tuple[float, ...] = (0.0, 0.0, 0.0)
    lane_hazard_features: str = "v1"
    merge_hazard_zone: str = "forced_late"
    gap_stopped_relax: bool = False
    lane_hazard_decouple: bool = False
    # LANEV5 (2026-10-14, default OFF; lane_model='hazard_v5' only).
    lane_v5_gates: tuple[float, ...] = (0.25, 0.10, 0.5)
    lane_v5_head_ap: tuple[float, ...] = (-1.0, -1.0, -1.0)
    lane_v5_pretrain: Optional[str] = None
    # SCHED (2026-10-14, default OFF): validation horizons, selection, mixed-horizon batches.
    val_all_horizons: bool = False
    selection_mode: str = "long_horizon"
    selection_horizons_s: tuple[float, ...] = (10.0, 30.0, 60.0, 120.0)
    selection_reference: tuple[float, ...] = (7.0356, 49.4889, 128.7179, 0.0)
    selection_min_stage_s: float = 30.0
    selection_agg_reward_weight: float = 0.0
    mixed_horizon_frac: float = 0.0
    mixed_horizon_s: tuple[float, ...] = (10.0, 20.0)
    mixed_horizon_min_stage_s: float = 30.0
    # V10HORIZON (2026-10-19, default OFF): advance each curriculum stage when the
    # stage-horizon validation error plateaus instead of after a fixed step budget.
    plateau_advance: bool = False
    plateau_patience: int = 3
    plateau_min_delta: float = 0.01
    plateau_min_steps: int = 2
    plateau_max_steps: int = 0
    # V10HORIZON prioritised mixed replay (default OFF; all multipliers 1 = unchanged).
    replay_enabled: bool = False
    replay_merge_oversample: float = 1.0
    replay_cav_follower_oversample: float = 1.0
    replay_brake_oversample: float = 1.0
    replay_error_max_boost: float = 8.0
    replay_error_alpha: float = 1.0
    replay_error_decay: float = 0.9
    replay_probe_windows: int = 0
    replay_refresh_every: int = 0
    # PAIRLOSS (2026-10-14, default OFF): paired-difference loss on paired SUMO branches.
    paired_diff_weight: float = 0.0
    paired_diff_fraction: float = 0.25
    paired_diff_groups: int = 2
    paired_diff_max_arms: int = 5
    paired_diff_position_weight: float = 1.0
    paired_diff_speed_weight: float = 1.0
    paired_diff_aggregate_weight: float = 1.0
    paired_diff_couple_randomness: bool = False
    paired_diff_return_weight: float = 0.0
    paired_diff_rank_weight: float = 0.0
    paired_diff_rank_margin_veh_s: float = 5.0
    paired_diff_horizon_mode: str = "fixed"
    # INFRAV5 (2026-10-14, default OFF): driver-estimate prior on the batch-mean encoded
    # settings (loss.driver_prior; 0 = off) and the same-horizon checkpoint selection
    # (selection_mode "multi_horizon_v2").
    driver_prior_weight: float = 0.0
    driver_prior_target: str = "explicit"
    driver_prior_accel: float = 2.6
    driver_prior_time_headway: float = 1.1
    driver_prior_decel: float = 4.5
    driver_prior_min_gap: float = 2.5
    driver_prior_v0_ratio: float = 1.0
    # V6INTEG: retained in flat checkpoint configs; absent in older checkpoints.
    stop_time_integration: bool = False
    emergency_decel_mps2: Optional[float] = None
    lane_conflict_resolve: bool = False
    # V6MERGE: retained in flat checkpoint configs; absent in older checkpoints.
    merge_end_wait: bool = False
    mainline_merge_yield: bool = False
    # V10UNBLOCK (2026-10-19, default OFF = bit-identical): physical envelope + all-risk lane
    # learning (wms.physics.config.PhysicsConfig.lane_learnable).
    lane_learnable: bool = False
    # V10LANEOPT (2026-10-19, default 'free' = current behaviour): staged mainline lane changes
    # (wms.physics.config.PhysicsConfig.mainline_lane_mode).
    mainline_lane_mode: str = "free"
    mainline_courtesy_zone_m: float = 300.0
    # V7FT (2026-10-17, default OFF): merge-zone calibration loss (wms.train.losses.merge).
    merge_loss_speed_weight: float = 0.0
    merge_loss_discharge_weight: float = 0.0
    merge_loss_cell_edges_m: tuple[float, ...] = (-100.0, 0.0, 75.44, 300.0)
    merge_loss_line_m: float = 75.44
    merge_loss_bin_s: float = 5.0
    merge_loss_cell_temp_m: float = 2.0
    merge_loss_cross_temp_m: float = 5.0
    merge_loss_speed_scale_mps: float = 1.0
    merge_loss_discharge_scale_veh: float = 2.0
    # V8WM (2026-10-17, all default OFF / unchanged).
    merge_loss_speed_cells: Optional[tuple[int, ...]] = None
    merge_loss_max_share: Optional[float] = None
    merge_timing_weight: float = 0.0
    merge_timing_metrics: bool = False
    physics_overrides: Optional[Mapping[str, float]] = None
    physics_inference: Optional[Mapping[str, Any]] = None
    selection_ramp_weight: float = 0.0
    selection_stuck_weight: float = 0.0
    # V8LANE (2026-10-17, default OFF): mainline decision rule (model switch, mirrored for
    # validation) and the on-policy lane-flow loss.
    lane_v5_mainline: str = "default"
    lane_flow_weight: float = 0.0
    lane_flow_balance: float = 0.0
    # V10COURTESY (2026-10-19, default OFF): learned courtesy inputs and the
    # position/speed event boost for lane-1 followers near a merging ramp car.
    courtesy_inputs: bool = False
    courtesy_weight: float = 1.0
    # V10NEGOT (default OFF): learned merge-negotiation block, censored merge-survival loss
    # and the ramp-only merge teacher-forcing schedule.
    merge_negotiation: bool = False
    merge_survival_weight: float = 0.0
    merge_survival_negative_weight: float = 0.0
    merge_survival_metrics: bool = False
    merge_teacher_frac_schedule: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _require(self.emergency_decel_mps2 is None or (
            not isinstance(self.emergency_decel_mps2, bool)
            and math.isfinite(float(self.emergency_decel_mps2))
            and 0.0 < float(self.emergency_decel_mps2) <= 9.0),
            "emergency_decel_mps2 must be finite and in (0, 9] when set")
        _require(bool(self.unroll_curriculum_s)
                 and all(float(x) > 0 for x in self.unroll_curriculum_s),
                 "unroll_curriculum_s must be positive")
        _require(self.epochs_per_stage >= 1 and self.steps_per_epoch >= 1,
                 "epochs_per_stage and steps_per_epoch must be positive")
        _require(self.truncation_steps >= 1, "truncation_steps must be positive")
        _require(self.entry_pad >= 0, "entry_pad must be non-negative")
        _require(self.stage_steps is None or (
            len(self.stage_steps) == len(self.unroll_curriculum_s)
            and all(int(x) >= 1 for x in self.stage_steps)),
            "stage_steps must give one positive count per curriculum stage")
        _require(self.batch_size_per_stage is None or (
            len(self.batch_size_per_stage) == len(self.unroll_curriculum_s)
            and all(int(x) >= 1 for x in self.batch_size_per_stage)),
            "batch_size_per_stage must give one positive batch size per curriculum stage")
        _require(float(self.traffic_bin_s) > 0.0 and float(self.traffic_weight) >= 0.0,
                 "traffic_bin_s must be positive and traffic_weight non-negative")
        _require(float(self.traffic_ramp_end_s) > float(self.traffic_ramp_start_s),
                 "traffic_ramp_end_s must exceed traffic_ramp_start_s")
        _require(float(self.state_noise_std_m) >= 0.0 and float(self.state_noise_std_mps) >= 0.0,
                 "state noise stds must be non-negative")
        _require(self.rollout_mode in ROLLOUT_MODES,
                 "rollout_mode must be residual, driver_params or hybrid")
        _require(self.history_s is None or not self.black_box,
                 "long history requires structured WM-S")
        _require(self.lane_head in ("off", "history"), "lane_head must be off or history")
        _require(self.lane_rule in LANE_RULES,
                 f"lane_rule must be one of {LANE_RULES}")
        _require(self.lane_gap_rule in LANE_GAP_RULES,
                 f"lane_gap_rule must be one of {LANE_GAP_RULES}")
        _require(self.lane_head != "history" or self.rollout_mode in ("driver_params", "hybrid"),
                 "history lane head requires rollout_mode=driver_params or hybrid")
        _require(self.residual_magnitude_weight is None
                 or float(self.residual_magnitude_weight) >= 0.0,
                 "residual_magnitude_weight must be non-negative when set")
        _require(self.residual_cap_mps2 is None or float(self.residual_cap_mps2) > 0.0,
                 "residual_cap_mps2 must be > 0 when set")
        _require(self.lane_logit_cap is None or float(self.lane_logit_cap) > 0.0,
                 "lane_logit_cap must be > 0 when set")
        _require(float(self.stability_weight) >= 0.0 and float(self.conservation_weight) >= 0.0,
                 "stability/conservation weights must be non-negative")
        _require(float(self.projection_min_gap_m) >= 0.0,
                 "projection_min_gap_m must be non-negative")
        _require(0.0 <= float(self.switch_window_fraction) <= 1.0,
                 "switch_window_fraction must be in [0,1]")
        _require(self.cav_drive in CAV_DRIVE_MODES, f"cav_drive must be one of {CAV_DRIVE_MODES}")
        _require(self.residual_cap_neg_mps2 is None or (
            math.isfinite(float(self.residual_cap_neg_mps2))
            and float(self.residual_cap_neg_mps2) > 0.0),
            "residual_cap_neg_mps2 must be finite and > 0 when set")
        validate_bounds(self.residual_soft_bound_mps2, self.residual_soft_bound_neg_mps2,
                        self.residual_cap_mps2, self.residual_cap_neg_mps2)
        for value in (self.residual_mag_weight, self.lane_rate_weight):
            _require(math.isfinite(value) and value >= 0,
                     "residual_mag/lane_rate weights must be finite and non-negative")
        _require(math.isfinite(self.lane_rate_target) and 0 <= self.lane_rate_target <= 1,
                 "lane_rate_target must be finite and in [0,1]")
        _require(self.gate_kind in ("cumsum", "saturating"),
                 "gate_kind must be cumsum or saturating")
        _require(math.isfinite(self.gate_g_max_init)
                 and 0 < self.gate_g_max_init < 1,
                 "gate_g_max_init must be finite and in (0,1)")
        _require(math.isfinite(self.gate_tau_steps_init) and self.gate_tau_steps_init > 0,
                 "gate_tau_steps_init must be finite and positive")
        _require(self.lane_gate_kind in ("shared", "saturating"),
                 "lane_gate_kind must be shared or saturating")
        _require(math.isfinite(self.lane_gate_g_max_init)
                 and 0 < self.lane_gate_g_max_init < 1,
                 "lane_gate_g_max_init must be finite and in (0,1)")
        _require(math.isfinite(self.lane_gate_tau_steps_init)
                 and self.lane_gate_tau_steps_init > 0,
                 "lane_gate_tau_steps_init must be finite and positive")
        _require(self.truncation_steps_per_stage is None or (
            len(self.truncation_steps_per_stage) == len(self.unroll_curriculum_s)
            and all(int(x) >= 1 for x in self.truncation_steps_per_stage)),
            "truncation_steps_per_stage must give one positive truncation per stage")
        _require(self.lr_per_stage is None or (
            len(self.lr_per_stage) == len(self.unroll_curriculum_s)
            and all(math.isfinite(float(x)) and float(x) > 0 for x in self.lr_per_stage)),
            "lr_per_stage must give one positive learning rate per stage")
        _require(math.isfinite(self.flip_rate_weight) and self.flip_rate_weight >= 0,
                 "flip_rate_weight must be finite and non-negative")
        _require(math.isfinite(self.flip_rate_allowance)
                 and 0 <= self.flip_rate_allowance <= 1,
                 "flip_rate_allowance must be finite and in [0,1]")
        _require(self.val_every >= 1, "val_every must be positive")
        _require(self.lane_model in ("residual", "hazard", "hazard_v5"),
                 "lane_model must be residual, hazard or hazard_v5")
        _require(self.driver_update in ("off", "ema"), "driver_update must be off or ema")
        _require(self.driver_param_mode in ("legacy", "physical"),
                 "driver_param_mode must be legacy or physical")
        _require(isinstance(self.driver_ema_update, bool),
                 "driver_ema_update must be a boolean")
        _require(0.0 <= self.driver_reveal_prob <= 1.0 and self.driver_reveal_max_s > 0.0,
                 "driver_reveal_prob must be in [0,1] and driver_reveal_max_s > 0")
        _require(math.isfinite(self.lane_event_weight) and self.lane_event_weight >= 0.0
                 and self.lane_event_gamma >= 0.0 and self.lane_event_max_weight >= 1.0,
                 "lane_event weight/gamma must be non-negative and max_weight >= 1")
        for value in (self.aggregate_weight, self.aggregate_field_weight,
                      self.aggregate_reward_weight, self.aggregate_flow_weight,
                      self.aggregate_horizon_start_s, self.aggregate_horizon_full_s):
            _require(math.isfinite(value) and value >= 0,
                     "aggregate weights/horizons must be finite and non-negative")
        _require(float(self.aggregate_bin_s) > 0 and float(self.aggregate_cell_temp_m) > 0
                 and int(self.aggregate_every_steps) >= 1,
                 "aggregate_bin_s/cell_temp_m must be positive and every_steps >= 1")
        _require(all(math.isfinite(float(x)) and float(x) >= 0.0 for x in (
            self.agg_delay_weight, self.agg_delay_paired_weight, self.agg_delay_start_s,
            self.agg_delay_horizon_s, self.agg_delay_fade_s)),
            "agg_delay weights/window must be finite and non-negative")
        _require(all(math.isfinite(float(x)) and float(x) > 0.0 for x in (
            self.agg_delay_bin_s, self.agg_delay_region_temp_m, self.agg_delay_queue_temp_mps,
            self.agg_delay_delay_scale_veh_s, self.agg_delay_queue_scale_veh,
            self.agg_delay_throughput_scale_veh, self.agg_delay_huber_delta)),
            "agg_delay bin/temperatures/scales must be finite and positive")
        _require(math.isfinite(float(self.agg_delay_merge_s1_m))
                 and float(self.agg_delay_merge_s1_m) > float(self.agg_delay_merge_s0_m)
                 and int(self.agg_delay_merge_lane) >= 1
                 and math.isfinite(float(self.agg_delay_road_end_m))
                 and float(self.agg_delay_road_end_m) > 0.0,
                 "agg_delay merge geometry must be finite with merge_s1_m > merge_s0_m, "
                 "merge_lane >= 1 and road_end_m > 0")
        _require(len(self.agg_delay_sections_m) >= 1
                 and all(math.isfinite(float(x)) for x in self.agg_delay_sections_m),
                 "agg_delay_sections_m must be non-empty finite positions")
        _require(int(self.entry_pad_max) >= 1, "entry_pad_max must be positive")
        _require(self.junction_fix in JUNCTION_FIX_MODES,
                 f"junction_fix must be one of {JUNCTION_FIX_MODES}")
        _require(self.controller_hygiene in CONTROLLER_HYGIENE_MODES,
                 f"controller_hygiene must be one of {CONTROLLER_HYGIENE_MODES}")
        _require(self.lane_decision in ("argmax", "sample", "gated"),
                 "lane_decision must be argmax, sample or gated")
        _require(len(self.lane_v5_gates) == 3 and len(self.lane_v5_head_ap) == 3
                 and all(math.isfinite(float(x))
                         for x in (*self.lane_v5_gates, *self.lane_v5_head_ap)),
                 "lane_v5_gates / lane_v5_head_ap must be three finite values")
        v5_on = (self.lane_decision == "gated" or self.lane_v5_pretrain is not None
                 or tuple(self.lane_v5_gates) != (0.25, 0.10, 0.5)
                 or tuple(self.lane_v5_head_ap) != (-1.0, -1.0, -1.0))
        _require(not v5_on or self.lane_model == "hazard_v5",
                 "lane_decision='gated' / lane_v5_* require lane_model='hazard_v5'")
        _require(len(self.lane_logit_offsets) == 3
                 and all(math.isfinite(float(x)) for x in self.lane_logit_offsets),
                 "lane_logit_offsets must be three finite values")
        _require(self.lane_hazard_features in ("v1", "v2"), "lane_hazard_features must be v1 or v2")
        _require(self.merge_hazard_zone in ("forced_late", "full"),
                 "merge_hazard_zone must be forced_late or full")
        _require(not self.gap_stopped_relax or self.lane_gap_rule == "speed",
                 "gap_stopped_relax requires lane_gap_rule='speed'")
        for value in (self.paired_diff_weight, self.paired_diff_position_weight,
                      self.paired_diff_speed_weight, self.paired_diff_aggregate_weight,
                      self.paired_diff_return_weight, self.paired_diff_rank_weight):
            _require(math.isfinite(float(value)) and float(value) >= 0,
                     "paired_diff weights must be finite and non-negative")
        _require(isinstance(self.paired_diff_couple_randomness, bool),
                 "paired_diff_couple_randomness must be a bool")
        _require(math.isfinite(float(self.paired_diff_rank_margin_veh_s))
                 and float(self.paired_diff_rank_margin_veh_s) >= 0.0,
                 "paired_diff_rank_margin_veh_s must be finite and non-negative")
        _require(0.0 <= float(self.paired_diff_fraction) <= 1.0,
                 "paired_diff_fraction must be in [0,1]")
        _require(int(self.paired_diff_groups) >= 1 and int(self.paired_diff_max_arms) >= 2,
                 "paired_diff_groups must be >= 1 and paired_diff_max_arms >= 2")
        _require(self.paired_diff_horizon_mode in ("fixed", "stage"),
                 "paired_diff_horizon_mode must be fixed or stage")
        _require(self.lane_v5_mainline in ("default", "sample", "budget", "budget_prop"),
                 "lane_v5_mainline must be default, sample, budget or budget_prop")
        _require(all(math.isfinite(float(x)) and float(x) >= 0.0
                     for x in (self.lane_flow_weight, self.lane_flow_balance)),
                 "lane_flow_weight / lane_flow_balance must be finite and non-negative")
        _require((self.lane_v5_mainline == "default" and float(self.lane_flow_weight) == 0.0)
                 or self.lane_model == "hazard_v5",
                 "lane_v5_mainline / lane_flow_weight require lane_model='hazard_v5'")
        lanefix_on = (self.lane_decision != "argmax"
                      or any(float(x) for x in self.lane_logit_offsets)
                      or self.lane_hazard_features != "v1"
                      or self.merge_hazard_zone != "forced_late" or self.lane_hazard_decouple)
        _require(not lanefix_on or self.lane_model in ("hazard", "hazard_v5"),
                 "LANEFIX lane switches require lane_model='hazard'")
        _require(self.selection_mode in SELECTION_MODES,
                 f"selection_mode must be one of {SELECTION_MODES}")
        _require(self.selection_mode != "multi_horizon_v2" or self.val_all_horizons,
                 "selection_mode multi_horizon_v2 requires val_all_horizons (every candidate "
                 "is compared on the same horizon set)")
        _require(math.isfinite(float(self.driver_prior_weight))
                 and float(self.driver_prior_weight) >= 0.0
                 and all(math.isfinite(float(x)) and float(x) > 0.0 for x in (
                     self.driver_prior_accel, self.driver_prior_time_headway,
                     self.driver_prior_decel, self.driver_prior_min_gap,
                     self.driver_prior_v0_ratio)),
                 "driver_prior weight must be >= 0 and its targets positive")
        _require(self.driver_prior_target in ("explicit", "legacy", "textbook"),
                 "driver_prior_target must be explicit, legacy or textbook")
        _require(not float(self.driver_prior_weight) or (
            self.driver_grad_across_chunks and self.rollout_mode != "residual"),
            "driver_prior_weight > 0 requires driver_grad_across_chunks and an "
            "encoder-driven rollout_mode")
        _require((not self.val_all_horizons and self.selection_mode == "long_horizon")
                 or self.select_by_long_horizon,
                 "val_all_horizons / multi_horizon selection require select_by_long_horizon "
                 "(the validation schedule)")
        _require(len(self.selection_horizons_s) >= 1
                 and len(self.selection_reference) == len(self.selection_horizons_s)
                 and all(math.isfinite(float(x)) and float(x) > 0
                         for x in self.selection_horizons_s)
                 and all(math.isfinite(float(x)) for x in self.selection_reference),
                 "selection_reference needs one finite value per selection horizon")
        _require(all(math.isfinite(float(x)) and float(x) >= 0.0 for x in (
            self.merge_timing_weight, self.selection_ramp_weight, self.selection_stuck_weight)),
            "merge_timing_weight / selection_ramp_weight / selection_stuck_weight must be >= 0")
        _require(self.merge_loss_max_share is None
                 or 0.0 < float(self.merge_loss_max_share) < 1.0,
                 "merge_loss_max_share must be in (0, 1) when set")
        _require(not (self.selection_ramp_weight or self.selection_stuck_weight)
                 or (self.selection_mode == "multi_horizon_v2" and self.merge_timing_on),
                 "selection_ramp_weight / selection_stuck_weight need selection_mode "
                 "multi_horizon_v2 and the merge-timing metrics (merge_timing_weight > 0 or "
                 "merge_timing_metrics)")
        _require(math.isfinite(self.selection_agg_reward_weight)
                 and self.selection_agg_reward_weight >= 0.0,
                 "selection_agg_reward_weight must be finite and non-negative")
        _require(math.isfinite(self.mixed_horizon_frac) and 0.0 <= self.mixed_horizon_frac <= 1.0
                 and len(self.mixed_horizon_s) >= 1
                 and all(math.isfinite(float(x)) and float(x) > 0 for x in self.mixed_horizon_s),
                 "mixed_horizon_frac must be in [0,1] and mixed_horizon_s positive")
        _require(isinstance(self.plateau_advance, bool), "plateau_advance must be a boolean")
        _require(int(self.plateau_patience) >= 1 and int(self.plateau_min_steps) >= 1
                 and int(self.plateau_max_steps) >= 0,
                 "plateau_patience/min_steps must be >= 1 and plateau_max_steps >= 0")
        _require(math.isfinite(float(self.plateau_min_delta)) and float(self.plateau_min_delta) >= 0.0,
                 "plateau_min_delta must be finite and non-negative")
        _require(isinstance(self.courtesy_inputs, bool),
                 "courtesy_inputs must be a boolean")
        _require(math.isfinite(float(self.courtesy_weight))
                 and float(self.courtesy_weight) >= 1.0,
                 "courtesy_weight must be finite and >= 1 (1 = off)")
        _require(isinstance(self.merge_negotiation, bool)
                 and isinstance(self.merge_survival_metrics, bool),
                 "merge_negotiation / merge_survival_metrics must be booleans")
        _require(math.isfinite(float(self.merge_survival_weight))
                 and float(self.merge_survival_weight) >= 0.0
                 and math.isfinite(float(self.merge_survival_negative_weight))
                 and float(self.merge_survival_negative_weight) >= 0.0,
                 "merge_survival weights must be finite and non-negative")
        _require(all(math.isfinite(float(x)) and 0.0 <= float(x) <= 1.0
                     for x in self.merge_teacher_frac_schedule),
                 "merge_teacher_frac_schedule values must be in [0, 1]")
        _require(isinstance(self.replay_enabled, bool)
                 and all(math.isfinite(float(x)) and float(x) >= 1.0 for x in (
                     self.replay_merge_oversample, self.replay_cav_follower_oversample,
                     self.replay_brake_oversample,
                     self.replay_error_max_boost, self.replay_error_alpha))
                 and math.isfinite(float(self.brake_follow_weight))
                 and float(self.brake_follow_weight) >= 1.0
                 and math.isfinite(float(self.brake_follow_cav_weight))
                 and float(self.brake_follow_cav_weight) >= 0.0
                 and math.isfinite(float(self.replay_error_decay))
                 and 0.0 <= float(self.replay_error_decay) < 1.0
                 and int(self.replay_probe_windows) >= 0 and int(self.replay_refresh_every) >= 0,
                 "replay_* multipliers must be >= 1, brake_follow_cav_weight >= 0, "
                 "error_decay in [0,1), probe_windows/refresh_every >= 0")
        _require(all(math.isfinite(float(x)) and float(x) >= 0.0 for x in (
            self.brake_persist_weight, self.brake_persist_quantile_weight))
            and math.isfinite(float(self.brake_persist_event_balance))
            and 0.0 <= float(self.brake_persist_event_balance) <= 4.0
            and math.isfinite(float(self.brake_persist_window_s))
            and 3.0 <= float(self.brake_persist_window_s) <= 5.0,
            "brake_persist weights must be >= 0, event_balance in [0,4], "
            "window_s in [3,5]")

    @classmethod
    def from_flat(cls, data: Mapping[str, Any]) -> "TrainSettings":
        """Build from a legacy flat config dict (unknown keys are refused)."""
        unknown = sorted(set(data) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"unknown WM-S config keys: {unknown}")
        values = dict(data)
        if "unroll_curriculum_s" in values:
            values["unroll_curriculum_s"] = tuple(float(x) for x in values["unroll_curriculum_s"])
        if values.get("stage_steps") is not None:
            values["stage_steps"] = tuple(int(x) for x in values["stage_steps"])
        if values.get("batch_size_per_stage") is not None:
            values["batch_size_per_stage"] = tuple(int(x)
                                                   for x in values["batch_size_per_stage"])
        if values.get("truncation_steps_per_stage") is not None:
            values["truncation_steps_per_stage"] = tuple(
                int(x) for x in values["truncation_steps_per_stage"])
        if values.get("lr_per_stage") is not None:
            values["lr_per_stage"] = tuple(float(x) for x in values["lr_per_stage"])
        for key in ("selection_horizons_s", "selection_reference", "mixed_horizon_s",
                    "merge_loss_cell_edges_m", "agg_delay_sections_m"):
            if values.get(key) is not None:
                values[key] = tuple(float(x) for x in values[key])
        if values.get("merge_loss_speed_cells") is not None:
            values["merge_loss_speed_cells"] = tuple(int(x) for x in values["merge_loss_speed_cells"])
        for key in ("lane_logit_offsets", "lane_v5_gates", "lane_v5_head_ap",
                    "merge_teacher_frac_schedule"):
            if values.get(key) is not None:
                values[key] = tuple(float(x) for x in values[key])
        return cls(**values)

    @classmethod
    def from_config(cls, config: Any) -> "TrainSettings":
        """Build from anything honouring ``to_legacy_flat()`` (``TrainConfig``)."""
        return cls.from_flat(config.to_legacy_flat())

    @property
    def uses_hard_cap(self) -> bool:
        """True when a legacy hard residual clamp is configured (REJECTED, #3)."""
        return any(getattr(self, name) is not None for name in HARD_CAP_FIELDS)

    @property
    def brake_persist_on(self) -> bool:
        """BRAKE_DIAG persistence terms enabled (either weight > 0)."""
        return (float(self.brake_persist_weight) > 0.0
                or float(self.brake_persist_quantile_weight) > 0.0)

    @property
    def merge_loss_on(self) -> bool:
        """V7FT merge-zone calibration loss enabled (either weight > 0)."""
        return (float(self.merge_loss_speed_weight) > 0.0
                or float(self.merge_loss_discharge_weight) > 0.0)

    @property
    def agg_delay_on(self) -> bool:
        """A5 long-horizon aggregate delay loss enabled (either weight > 0)."""
        return float(self.agg_delay_weight) > 0.0 or float(self.agg_delay_paired_weight) > 0.0

    @property
    def merge_timing_on(self) -> bool:
        """V8WM/V10NEGOT merge-timing / survival term built (weight or metrics)."""
        return (float(self.merge_timing_weight) > 0.0 or float(self.merge_survival_weight) > 0.0
                or bool(self.merge_timing_metrics) or bool(self.merge_survival_metrics))

    @property
    def merge_negotiation_on(self) -> bool:
        """V10NEGOT negotiation auxiliary (matched non-yield negatives) built."""
        return (bool(self.merge_negotiation)
                and float(self.merge_survival_negative_weight) > 0.0)

    @property
    def long_horizon_on(self) -> bool:
        """Any of the opt-in long-horizon guards (soft bound / penalties / selection)."""
        return bool(self.residual_soft_bound_mps2 or self.residual_mag_weight
                    or self.lane_rate_weight or self.flip_rate_weight
                    or self.select_by_long_horizon)
