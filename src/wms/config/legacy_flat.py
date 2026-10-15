"""Table linking the legacy flat ``WmSTrainConfig`` keys to nested ``TrainConfig`` paths.

The flat dict is the format stored in checkpoints (``ckpt["config"]``) and in
``TRAIN_RECEIPT.json``; it must stay readable and writable key for key.
"""
from __future__ import annotations

from typing import Any, Mapping, MutableMapping

#: flat key -> dotted path inside the nested config. ``black_box`` and
#: ``exclude_window_list`` need value conversion and are handled in ``TrainConfig``.
FLAT_TO_PATH: dict[str, str] = {
    "unroll_curriculum_s": "curriculum.unroll_s",
    "stage_steps": "curriculum.stage_steps",
    "num_workers": "runtime.num_workers",
    "epochs_per_stage": "curriculum.epochs_per_stage",
    "steps_per_epoch": "curriculum.steps_per_epoch",
    "batch_size": "optim.batch_size",
    "lr": "optim.lr",
    "weight_decay": "optim.weight_decay",
    "grad_clip": "optim.grad_clip",
    "position_huber_delta_m": "loss.position_huber_delta_m",
    "speed_huber_delta_mps": "loss.speed_huber_delta_mps",
    "lane_weight": "loss.lane_weight",
    "position_weight": "loss.position_weight",
    "speed_weight": "loss.speed_weight",
    "step_weight_decay": "loss.step_weight_decay",
    "truncation_steps": "curriculum.truncation_steps",
    "history_features": "model.history_features",
    "history_s": "data.history_s",
    "history_stride_s": "data.history_stride_s",
    "device": "runtime.device",
    "threads": "runtime.threads",
    "state_noise_std_m": "data.state_noise_std_m",
    "state_noise_std_mps": "data.state_noise_std_mps",
    "rollout_mode": "model.rollout_mode",
    "lane_head": "model.lane_head",
    "residual_magnitude_weight": "loss.residual.hybrid_magnitude_weight",
    "driver_sidecar": "data.sidecars.driver_sidecar",
    "out_dir": "runtime.out_dir",
    "seed": "runtime.seed",
    "black_box": "model.kind",
    "n_controllers": "model.n_controllers",
    "controller_embed_dim": "model.controller_embed_dim",
    "checkpoint_residual": "model.checkpoint_residual",
    "traffic_loss": "loss.traffic.enabled",
    "traffic_bin_s": "loss.traffic.bin_s",
    "traffic_weight": "loss.traffic.weight",
    "traffic_free_speed_mps": "loss.traffic.free_speed_mps",
    "traffic_ramp_start_s": "loss.traffic.ramp_start_s",
    "traffic_ramp_end_s": "loss.traffic.ramp_end_s",
    "entries_v2": "data.entries_v2",
    "entry_pad": "data.entry_pad",
    "window_list": "data.windows.train",
    "exclude_window_list": "data.windows.exclude",
    "val_window_list": "data.windows.val",
    "val_batches": "selection.val_batches",
    "val_max_windows": "selection.val_max_windows",
    "projection_enabled": "model.projection.enabled",
    "projection_min_gap_m": "model.projection.min_gap_m",
    "residual_cap_mps2": "model.residual_cap_mps2",
    "lane_logit_cap": "model.lane_logit_cap",
    "stability_weight": "loss.stability_weight",
    "stability_margin": "loss.stability_margin",
    "stability_probes": "loss.stability_probes",
    "conservation_enabled": "loss.conservation_enabled",
    "conservation_weight": "loss.conservation_weight",
    "loss_share_logging": "loss.loss_share_logging",
    "switch_window_fraction": "data.switching.window_fraction",
    "asym_speed_alpha": "loss.braking.asym_speed_alpha",
    "decel_weight": "loss.braking.decel_weight",
    "decel_threshold_mps2": "loss.braking.decel_threshold_mps2",
    "propagation_weight": "loss.braking.propagation_weight",
    "prop_window_s": "loss.braking.prop_window_s",
    "prop_max_gap_m": "loss.braking.prop_max_gap_m",
    "residual_cap_neg_mps2": "model.residual_cap_neg_mps2",
    "cav_drive": "data.cav_drive",
    "residual_soft_bound_mps2": "model.residual_soft_bound_mps2",
    "residual_soft_bound_neg_mps2": "model.residual_soft_bound_neg_mps2",
    "residual_mag_weight": "loss.residual.residual_mag_weight",
    "lane_rate_weight": "loss.lane_rate.weight",
    "lane_rate_target": "loss.lane_rate.target",
    "select_by_long_horizon": "selection.select_by_long_horizon",
    "val_every": "selection.val_every",
    "gate_kind": "model.gate.kind",
    "gate_g_max_init": "model.gate.g_max_init",
    "gate_tau_steps_init": "model.gate.tau_steps_init",
    "lane_gate_kind": "model.lane_gate.kind",
    "lane_gate_g_max_init": "model.lane_gate.g_max_init",
    "lane_gate_tau_steps_init": "model.lane_gate.tau_steps_init",
    "checkpoint_step": "model.checkpoint_step",
    "grad_decay_per_step": "model.grad_decay_per_step",
    "init_from": "model.init_from",
    "truncation_steps_per_stage": "curriculum.truncation_steps_per_stage",
    "lr_per_stage": "optim.lr_per_stage",
    "driver_grad_across_chunks": "curriculum.driver_grad_across_chunks",
    "flip_rate_weight": "loss.flip_rate.weight",
    "flip_rate_allowance": "loss.flip_rate.allowance",
    "lane_rule": "model.lane_rule",
    "lane_gap_rule": "model.lane_gap_rule",  # GAPRULE
    # WMLANE (A hazard lane model + event loss, C driver-setting update)
    "lane_model": "model.lane_model",
    "lane_hazard_hidden": "model.lane_hazard_hidden",
    "lane_hazard_cap": "model.lane_hazard_cap",
    "driver_update": "model.driver_update",
    "driver_param_mode": "model.driver_param_mode",
    "driver_ema_update": "model.driver_ema_update",
    "driver_update_rate": "model.driver_update_rate",
    "driver_reveal_prob": "model.driver_reveal_prob",
    "driver_reveal_max_s": "model.driver_reveal_max_s",
    "lane_event_weight": "loss.lane_event.weight",
    "lane_event_gamma": "loss.lane_event.gamma",
    "lane_event_max_weight": "loss.lane_event.max_weight",
    "aggregate_weight": "loss.aggregate.weight",
    "aggregate_field_weight": "loss.aggregate.field_weight",
    "aggregate_reward_weight": "loss.aggregate.reward_weight",
    "aggregate_flow_weight": "loss.aggregate.flow_weight",
    "aggregate_bin_s": "loss.aggregate.bin_s",
    "aggregate_every_steps": "loss.aggregate.every_steps",
    "aggregate_horizon_start_s": "loss.aggregate.horizon_start_s",
    "aggregate_horizon_full_s": "loss.aggregate.horizon_full_s",
    "aggregate_cell_temp_m": "loss.aggregate.cell_temp_m",
    "entry_pad_auto": "data.entry_pad_auto",
    "entry_pad_max": "data.entry_pad_max",
    "entry_insert_rule": "data.entry_insert_rule",  # ENTRYFIX
    "junction_fix": "data.junction_fix",  # JUNCTIONFIX
    "controller_hygiene": "data.controller_hygiene",  # DATAHYG
    # LANEFIX (hazard decisions, ranking features, merge ownership, stopped-merger gaps)
    "lane_decision": "model.lane_decision",
    "lane_logit_offsets": "model.lane_logit_offsets",
    "lane_hazard_features": "model.lane_hazard_features",
    "merge_hazard_zone": "model.merge_hazard_zone",
    "gap_stopped_relax": "model.gap_stopped_relax",
    "lane_hazard_decouple": "loss.lane_hazard_decouple",
    # LANEV5 (per-transition lane network)
    "lane_v5_gates": "model.lane_v5_gates",
    "lane_v5_head_ap": "model.lane_v5_head_ap",
    "lane_v5_pretrain": "model.lane_v5_pretrain",
    # SCHED (validation horizons, multi-horizon selection, mixed-horizon batches)
    "val_all_horizons": "curriculum.val_all_horizons",
    "selection_mode": "curriculum.selection",
    "selection_horizons_s": "curriculum.selection_horizons_s",
    "selection_reference": "curriculum.selection_reference",
    "selection_min_stage_s": "curriculum.selection_min_stage_s",
    "selection_agg_reward_weight": "curriculum.selection_agg_reward_weight",
    "mixed_horizon_frac": "curriculum.mixed_horizon_frac",
    "mixed_horizon_s": "curriculum.mixed_horizon_s",
    "mixed_horizon_min_stage_s": "curriculum.mixed_horizon_min_stage_s",
    # PAIRLOSS (paired-difference loss on paired SUMO branches)
    "paired_diff_weight": "loss.paired_diff.weight",
    "paired_diff_fraction": "loss.paired_diff.fraction",
    "paired_diff_groups": "loss.paired_diff.groups",
    "paired_diff_max_arms": "loss.paired_diff.max_arms",
    "paired_diff_position_weight": "loss.paired_diff.position_weight",
    "paired_diff_speed_weight": "loss.paired_diff.speed_weight",
    "paired_diff_aggregate_weight": "loss.paired_diff.aggregate_weight",
    "paired_diff_couple_randomness": "loss.paired_diff.couple_randomness",
    "paired_diff_return_weight": "loss.paired_diff.return_weight",
    "paired_diff_rank_weight": "loss.paired_diff.rank_weight",
    "paired_diff_rank_margin_veh_s": "loss.paired_diff.rank_margin_veh_s",
    "paired_diff_horizon_mode": "loss.paired_diff.horizon_mode",
    # INFRAV5 (optional driver-estimate prior on the batch-mean encoded settings)
    "driver_prior_weight": "loss.driver_prior.weight",
    "driver_prior_target": "loss.driver_prior.target",
    "driver_prior_accel": "loss.driver_prior.accel",
    "driver_prior_time_headway": "loss.driver_prior.time_headway",
    "driver_prior_decel": "loss.driver_prior.decel",
    "driver_prior_min_gap": "loss.driver_prior.min_gap",
    "driver_prior_v0_ratio": "loss.driver_prior.v0_ratio",
    # V6INTEG (opt-in integration and simultaneous lane-conflict resolution)
    "stop_time_integration": "physics.stop_time_integration",
    "emergency_decel_mps2": "physics.emergency_decel_mps2",
    "lane_conflict_resolve": "model.lane_conflict_resolve",
    # V6MERGE (opt-in lane-end interaction rules)
    "merge_end_wait": "physics.merge_end_wait",
    "mainline_merge_yield": "physics.mainline_merge_yield",
}

#: Keys the legacy ``to_dict`` omits while they equal their default (braking-fix
#: and long-horizon fields), so old flags-off receipts stay byte-identical.
OMIT_WHEN_DEFAULT: tuple[str, ...] = (
    "asym_speed_alpha", "decel_weight", "decel_threshold_mps2", "propagation_weight",
    "prop_window_s", "prop_max_gap_m", "residual_cap_neg_mps2",
    "residual_soft_bound_mps2", "residual_soft_bound_neg_mps2", "residual_mag_weight",
    "lane_rate_weight", "lane_rate_target", "select_by_long_horizon", "val_every",
    "gate_kind", "gate_g_max_init", "gate_tau_steps_init",
    "lane_gate_kind", "lane_gate_g_max_init", "lane_gate_tau_steps_init",
    "checkpoint_step", "grad_decay_per_step", "init_from", "truncation_steps_per_stage",
    "lr_per_stage",
    "driver_grad_across_chunks", "flip_rate_weight", "flip_rate_allowance",
    "lane_rule", "lane_gap_rule",
    "lane_model", "lane_hazard_hidden", "lane_hazard_cap", "driver_update",
    "driver_param_mode", "driver_ema_update",
    "driver_update_rate", "driver_reveal_prob", "driver_reveal_max_s",
    "lane_event_weight", "lane_event_gamma", "lane_event_max_weight",
    "aggregate_weight", "aggregate_field_weight", "aggregate_reward_weight",
    "aggregate_flow_weight", "aggregate_bin_s", "aggregate_every_steps",
    "aggregate_horizon_start_s", "aggregate_horizon_full_s", "aggregate_cell_temp_m",
    "entry_pad_auto", "entry_pad_max", "entry_insert_rule", "junction_fix", "controller_hygiene",
    "val_max_windows",
    "lane_decision", "lane_logit_offsets", "lane_hazard_features", "merge_hazard_zone",
    "gap_stopped_relax", "lane_hazard_decouple",
    "lane_v5_gates", "lane_v5_head_ap", "lane_v5_pretrain",
    # SCHED
    "val_all_horizons", "selection_mode", "selection_horizons_s", "selection_reference",
    "selection_min_stage_s", "selection_agg_reward_weight", "mixed_horizon_frac",
    "mixed_horizon_s", "mixed_horizon_min_stage_s",
    "paired_diff_weight", "paired_diff_fraction", "paired_diff_groups", "paired_diff_max_arms",
    "paired_diff_position_weight", "paired_diff_speed_weight", "paired_diff_aggregate_weight",
    "paired_diff_couple_randomness", "paired_diff_return_weight", "paired_diff_rank_weight",
    "paired_diff_rank_margin_veh_s",
    "paired_diff_horizon_mode",
    # INFRAV5
    "driver_prior_weight", "driver_prior_target", "driver_prior_accel", "driver_prior_time_headway",
    "driver_prior_decel", "driver_prior_min_gap", "driver_prior_v0_ratio",
    "stop_time_integration", "emergency_decel_mps2", "lane_conflict_resolve",
    "merge_end_wait", "mainline_merge_yield",
)


def get_path(tree: Mapping[str, Any], dotted: str) -> Any:
    """Read ``dotted`` ("a.b.c") from nested mappings."""
    node: Any = tree
    for part in dotted.split("."):
        node = node[part]
    return node


def set_path(tree: MutableMapping[str, Any], dotted: str, value: Any) -> None:
    """Write ``value`` at ``dotted``, creating intermediate tables."""
    parts = dotted.split(".")
    node = tree
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
