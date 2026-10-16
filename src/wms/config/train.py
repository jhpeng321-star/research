"""Training configuration: model, loss, curriculum, optimiser, selection and runtime."""
from __future__ import annotations

import warnings
from dataclasses import field
from typing import Any, Callable, Literal, Mapping, Optional

from ..core.constants import N_CONTROLLERS
from ..core.errors import ConfigError, TrainDataRefusal
from ..core.guard import find_restricted_token
from .data import DataConfig
from .legacy_flat import FLAT_TO_PATH, OMIT_WHEN_DEFAULT, get_path, set_path
from .schema import (config_dataclass, fingerprint, from_mapping, is_finite, require,
                     to_mapping)

ROLLOUT_MODES = ("residual", "driver_params", "hybrid")


@config_dataclass
class PhysicsOptions:
    """Opt-in integration settings; the default global plant braking limit is 9 m/s²."""

    stop_time_integration: bool = False
    emergency_decel_mps2: Optional[float] = None
    merge_end_wait: bool = False
    """V6MERGE: lane-end mergers need both lane-1 gaps SUMO-secure and never pass the end."""
    mainline_merge_yield: bool = False
    """V6MERGE: the nearest lane-1 follower brakes (own IDM) for a blocked lane-end merger."""

    def __post_init__(self) -> None:
        require(self.emergency_decel_mps2 is None or (
            is_finite(self.emergency_decel_mps2) and 0.0 < self.emergency_decel_mps2 <= 9.0),
            "physics.emergency_decel_mps2 must be finite and in (0, 9] when set")


@config_dataclass
class ProjectionConfig:
    """Differentiable minimum-gap projection during rollout (off by default)."""

    enabled: bool = False
    min_gap_m: float = 2.0

    def __post_init__(self) -> None:
        require(is_finite(self.min_gap_m) and self.min_gap_m >= 0.0,
                "model.projection.min_gap_m must be non-negative")


@config_dataclass
class GateConfig:
    """Residual-gate parameterization and saturating-gate initialization."""

    kind: Literal["cumsum", "saturating"] = "cumsum"
    g_max_init: float = 0.30
    tau_steps_init: float = 50.0

    def __post_init__(self) -> None:
        require(0.0 < self.g_max_init < 1.0,
                "model.gate.g_max_init must be in (0, 1)")
        require(is_finite(self.tau_steps_init) and self.tau_steps_init > 0.0,
                "model.gate.tau_steps_init must be finite and positive")


@config_dataclass
class LaneGateConfig:
    """Lane-correction gate parameterization (shared with acceleration by default)."""

    kind: Literal["shared", "saturating"] = "shared"
    g_max_init: float = 0.16
    tau_steps_init: float = 30.0

    def __post_init__(self) -> None:
        require(0.0 < self.g_max_init < 1.0,
                "model.lane_gate.g_max_init must be in (0, 1)")
        require(is_finite(self.tau_steps_init) and self.tau_steps_init > 0.0,
                "model.lane_gate.tau_steps_init must be finite and positive")


@config_dataclass
class ModelConfig:
    """Structure of the WM-S model and its physics options."""

    kind: Literal["structured", "blackbox_traj", "blackbox_id"] = "structured"
    rollout_mode: Literal["residual", "driver_params", "hybrid"] = "residual"
    lane_head: Literal["off", "history"] = "off"
    lane_rule: Literal["mobil", "forced_merge_only", "learned_discretionary"] = "mobil"
    lane_gap_rule: Literal["fixed", "speed"] = "fixed"
    """GAPRULE: gap acceptance of the lane-change feasibility mask ('fixed' = legacy 6 m /
    merge_gap + IDM follower braking; 'speed' = speed-dependent SUMO-like secure gaps)."""
    physics_params: Optional[str] = None
    merge_cal: bool = False
    sumo_fidelity: bool = False
    geometry_only: bool = False
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    lane_gate: LaneGateConfig = field(default_factory=LaneGateConfig)
    residual_cap_mps2: Optional[float] = None
    residual_cap_neg_mps2: Optional[float] = None
    residual_soft_bound_mps2: Optional[float] = None
    residual_soft_bound_neg_mps2: Optional[float] = None
    lane_logit_cap: Optional[float] = None
    n_controllers: int = N_CONTROLLERS
    controller_embed_dim: int = 16
    history_features: int = 6
    checkpoint_residual: bool = False
    checkpoint_step: bool = False
    grad_decay_per_step: float = 1.0
    """Round 5b: backward-only gradient damping per rollout step through the carried state (1.0 = off)."""
    init_from: Optional[str] = None
    allow_rejected_hard_cap: bool = False
    # --- WMLANE switches (default OFF = bit-identical legacy model) ---------------
    lane_model: Literal["residual", "hazard", "hazard_v5"] = "residual"
    """A: 'hazard' = dedicated per-HDV stay/left/right event model (replaces the residual lane delta)."""
    lane_hazard_hidden: int = 64
    lane_hazard_cap: float = 10.0
    driver_update: Literal["off", "ema"] = "off"
    """C: refine a car's driver settings/embedding from its observed motion inside a rollout."""
    driver_param_mode: Literal["legacy", "physical"] = "legacy"
    """V6: physical driver parameters learned from prediction error with a learned shared centre."""
    driver_ema_update: bool = True
    """Enable the EMA driver-parameter update (legacy behaviour)."""
    driver_update_rate: float = 0.3
    driver_reveal_prob: float = 0.0
    """C training: probability that an HDV gets a teacher-observed prefix in a training unroll."""
    driver_reveal_max_s: float = 10.0
    """C training: the revealed prefix length is drawn uniformly from (0, driver_reveal_max_s]."""
    # --- LANEFIX switches (2026-10-14, LANEFIX_NOTES.md; default OFF = bit-identical) -------
    lane_decision: Literal["argmax", "sample", "gated"] = "argmax"
    """Hazard mode: rollout lane decision = argmax, or a seeded draw from the masked categorical
    ('gated', hazard_v5 only: a head samples only if its TRAIN-val AP reaches its gate)."""
    lane_logit_offsets: tuple[float, ...] = (0.0, 0.0, 0.0)
    """Hazard mode: constant logit offsets (overtake 1->0, return 0->1, merge 2->1), calibration."""
    lane_hazard_features: Literal["v1", "v2"] = "v1"
    """Hazard input: 'v2' adds lookahead speed vs v0, leader trend, density ahead, blocked time."""
    merge_hazard_zone: Literal["forced_late", "full"] = "forced_late"
    """'full': the hazard owns ramp merges from the zone start; physics forces only the last 5 m."""
    gap_stopped_relax: bool = False
    """lane_gap_rule='speed': relaxed gaps (0.5 m, no anticipation) for stopped ramp mergers."""
    # --- LANEV5 (2026-10-14, LANEV5_NOTES.md; lane_model='hazard_v5' only) -------------------
    lane_v5_gates: tuple[float, ...] = (0.25, 0.10, 0.5)
    """lane_decision='gated': minimum TRAIN-val AP per head (overtake, return, merge) to sample."""
    lane_v5_head_ap: tuple[float, ...] = (-1.0, -1.0, -1.0)
    """Per-head TRAIN-val AP for the gate (negative = the AP stored in the lane network)."""
    lane_v5_pretrain: Optional[str] = None
    """Initialise the lane network from a tools/lanev5_pretrain.py file before training."""
    lane_conflict_resolve: bool = False
    """V6INTEG: resolve simultaneous lane-change conflicts during rollout."""

    def __post_init__(self) -> None:
        require(self.driver_param_mode in ("legacy", "physical"),
                "model.driver_param_mode must be legacy or physical")
        require(isinstance(self.driver_ema_update, bool),
                "model.driver_ema_update must be a boolean")
        require(self.lane_hazard_hidden >= 1 and is_finite(self.lane_hazard_cap)
                and self.lane_hazard_cap > 0.0,
                "model.lane_hazard_hidden must be >= 1 and lane_hazard_cap > 0")
        require(0.0 < self.driver_update_rate < 1.0, "model.driver_update_rate must be in (0, 1)")
        require(0.0 <= self.driver_reveal_prob <= 1.0 and is_finite(self.driver_reveal_max_s)
                and self.driver_reveal_max_s > 0.0,
                "model.driver_reveal_prob must be in [0,1] and driver_reveal_max_s > 0")
        if self.lane_model in ("hazard", "hazard_v5") or self.driver_update != "off":
            require(self.rollout_mode in ("driver_params", "hybrid") and self.kind == "structured",
                    "model.lane_model='hazard' / driver_update need a structured "
                    "driver_params or hybrid model")
        if self.lane_model in ("hazard", "hazard_v5"):
            require(self.lane_head == "off", "model.lane_model='hazard' requires lane_head='off'")
        require(len(self.lane_logit_offsets) == 3
                and all(is_finite(x) for x in self.lane_logit_offsets),
                "model.lane_logit_offsets must be three finite values (overtake, return, merge)")
        if (self.lane_decision != "argmax" or any(float(x) for x in self.lane_logit_offsets)
                or self.lane_hazard_features != "v1" or self.merge_hazard_zone != "forced_late"):
            require(self.lane_model in ("hazard", "hazard_v5"),
                    "model.lane_decision / lane_logit_offsets / lane_hazard_features / "
                    "merge_hazard_zone need model.lane_model='hazard'")
        require(len(self.lane_v5_gates) == 3 and len(self.lane_v5_head_ap) == 3
                and all(is_finite(x) for x in (*self.lane_v5_gates, *self.lane_v5_head_ap)),
                "model.lane_v5_gates / lane_v5_head_ap must be three finite values")
        if (self.lane_decision == "gated" or self.lane_v5_pretrain is not None
                or tuple(self.lane_v5_gates) != (0.25, 0.10, 0.5)
                or tuple(self.lane_v5_head_ap) != (-1.0, -1.0, -1.0)):
            require(self.lane_model == "hazard_v5",
                    "model.lane_decision='gated' / lane_v5_* need model.lane_model='hazard_v5'")
        require(not self.gap_stopped_relax or self.lane_gap_rule == "speed",
                "model.gap_stopped_relax requires model.lane_gap_rule='speed'")
        require(is_finite(self.grad_decay_per_step) and 0.0 < self.grad_decay_per_step <= 1.0,
                "model.grad_decay_per_step must be in (0, 1]")
        if self.lane_head == "history":
            require(self.rollout_mode in ("driver_params", "hybrid"),
                    "model.lane_head='history' requires rollout_mode driver_params or hybrid")
        if self.kind == "blackbox_traj":
            require(self.rollout_mode == "residual" and self.lane_head == "off",
                    "model.kind='blackbox_traj' requires residual rollout and lane_head off")
        bounds = (self.residual_cap_mps2, self.residual_cap_neg_mps2, self.lane_logit_cap,
                  self.residual_soft_bound_mps2, self.residual_soft_bound_neg_mps2)
        for name, value in zip(("residual_cap_mps2", "residual_cap_neg_mps2", "lane_logit_cap",
                                "residual_soft_bound_mps2", "residual_soft_bound_neg_mps2"), bounds):
            require(value is None or (is_finite(value) and value > 0.0),
                    f"model.{name} must be finite and > 0 when set")
        require(self.residual_soft_bound_neg_mps2 is None or self.residual_soft_bound_mps2 is not None,
                "model.residual_soft_bound_neg_mps2 requires residual_soft_bound_mps2")
        require(self.residual_soft_bound_mps2 is None
                or (self.residual_cap_mps2 is None and self.residual_cap_neg_mps2 is None),
                "residual soft bounds cannot be combined with legacy hard residual caps")
        require(self.n_controllers >= 1 and self.controller_embed_dim >= 1
                and self.history_features >= 1, "model sizes must be positive")
        if self.has_hard_cap and not self.allow_rejected_hard_cap:
            warnings.warn("model.residual_cap_* (hard residual clamp) was REJECTED for training "
                          "(change-log #3); training refuses it unless allow_rejected_hard_cap",
                          stacklevel=2)

    @property
    def has_hard_cap(self) -> bool:
        """True when a legacy hard residual cap is set."""
        return self.residual_cap_mps2 is not None or self.residual_cap_neg_mps2 is not None


@config_dataclass
class BrakingConfig:
    """Braking-response loss terms; defaults are all OFF."""

    asym_speed_alpha: float = 1.0
    decel_weight: float = 1.0
    decel_threshold_mps2: float = 3.0
    propagation_weight: float = 0.0
    prop_window_s: float = 3.0
    prop_max_gap_m: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("asym_speed_alpha", "decel_weight", "decel_threshold_mps2",
                     "propagation_weight", "prop_window_s"):
            require(is_finite(getattr(self, name)), f"loss.braking.{name} must be finite")
        require(self.asym_speed_alpha >= 1.0, "loss.braking.asym_speed_alpha must be >= 1 (1 = off)")
        require(self.decel_weight >= 1.0, "loss.braking.decel_weight must be >= 1 (1 = off)")
        require(self.decel_threshold_mps2 > 0.0, "loss.braking.decel_threshold_mps2 must be > 0")
        require(self.propagation_weight >= 0.0, "loss.braking.propagation_weight must be >= 0")
        require(self.prop_window_s > 0.0, "loss.braking.prop_window_s must be positive")
        require(self.prop_max_gap_m is None or (is_finite(self.prop_max_gap_m) and self.prop_max_gap_m > 0),
                "loss.braking.prop_max_gap_m must be positive when set")

    @property
    def any_on(self) -> bool:
        """True when at least one braking term is enabled."""
        return (self.asym_speed_alpha > 1.0 or self.decel_weight > 1.0
                or self.propagation_weight > 0.0)


@config_dataclass
class ResidualLossConfig:
    """Penalties on the learned residual magnitude."""

    residual_mag_weight: float = 0.0
    hybrid_magnitude_weight: Optional[float] = None

    def __post_init__(self) -> None:
        require(is_finite(self.residual_mag_weight) and self.residual_mag_weight >= 0.0,
                "loss.residual.residual_mag_weight must be finite and non-negative")
        require(self.hybrid_magnitude_weight is None
                or (is_finite(self.hybrid_magnitude_weight) and self.hybrid_magnitude_weight >= 0.0),
                "loss.residual.hybrid_magnitude_weight must be non-negative when set")


@config_dataclass
class LaneRateConfig:
    """Lane-change-rate penalty."""

    weight: float = 0.0
    target: float = 0.0123

    def __post_init__(self) -> None:
        require(is_finite(self.weight) and self.weight >= 0.0,
                "loss.lane_rate.weight must be finite and non-negative")
        require(is_finite(self.target) and 0.0 <= self.target <= 1.0,
                "loss.lane_rate.target must be in [0,1]")


@config_dataclass
class FlipRateConfig:
    """Flip-rate lane guard: hinge on the per-step leave-lane probability.

    ``allowance`` is the free per-HDV-step probability of leaving the current lane;
    the default is the TRUE training rate 0.00124327163 (LANECAL, junction lane -1
    excluded by the shared supervision mask).
    """

    weight: float = 0.0
    allowance: float = 0.00124327163

    def __post_init__(self) -> None:
        require(is_finite(self.weight) and self.weight >= 0.0,
                "loss.flip_rate.weight must be finite and non-negative")
        require(is_finite(self.allowance) and 0.0 <= self.allowance <= 1.0,
                "loss.flip_rate.allowance must be in [0,1]")


@config_dataclass
class LaneEventConfig:
    """WMLANE A: teacher-forced one-step lane-change event loss (off by default).

    Focal, category-balanced cross entropy of the stay/left/right decision on the
    RECORDED state of every HDV-step (0.5 s); event categories (overtake 1->0,
    return 0->1, ramp merge 2->1) are balanced against each other and against
    'stay', each event weight capped at ``max_weight``.
    """

    weight: float = 0.0
    gamma: float = 2.0
    max_weight: float = 20.0

    def __post_init__(self) -> None:
        require(is_finite(self.weight) and self.weight >= 0.0,
                "loss.lane_event.weight must be finite and non-negative")
        require(is_finite(self.gamma) and self.gamma >= 0.0,
                "loss.lane_event.gamma must be finite and non-negative")
        require(is_finite(self.max_weight) and self.max_weight >= 1.0,
                "loss.lane_event.max_weight must be >= 1")


@config_dataclass
class TrafficLossConfig:
    """Legacy differentiable traffic-level loss (off in R3)."""

    enabled: bool = False
    bin_s: float = 30.0
    weight: float = 0.1
    free_speed_mps: float = 13.89
    ramp_start_s: float = 30.0
    ramp_end_s: float = 120.0

    def __post_init__(self) -> None:
        require(self.bin_s > 0.0 and self.weight >= 0.0,
                "loss.traffic.bin_s must be positive and weight non-negative")
        require(self.ramp_end_s > self.ramp_start_s,
                "loss.traffic.ramp_end_s must exceed ramp_start_s")


@config_dataclass
class AggregateLossConfig:
    """C4 aggregate-consistency loss (``wms.train.losses.aggregate``; weight 0 = off).

    Compares soft RSU cell fields (density / mean speed / queue per 50 m cell per lane),
    reward components (on-road delay, throughput past the merge end) and, with
    ``flow_weight`` > 0, every cell-boundary flux between rollout and record, per
    ``bin_s`` bin, weighted by a horizon ramp from ``horizon_start_s`` to
    ``horizon_full_s`` (equal values: a step at ``horizon_start_s``).
    """

    weight: float = 0.0
    field_weight: float = 1.0
    reward_weight: float = 1.0
    flow_weight: float = 0.0
    bin_s: float = 5.0
    every_steps: int = 1
    horizon_start_s: float = 0.0
    horizon_full_s: float = 0.0
    cell_temp_m: float = 5.0

    def __post_init__(self) -> None:
        require(all(is_finite(x) and x >= 0.0 for x in (
            self.weight, self.field_weight, self.reward_weight, self.flow_weight,
            self.horizon_start_s, self.horizon_full_s)),
            "loss.aggregate weights/horizons must be finite and non-negative")
        require(self.bin_s > 0.0 and self.cell_temp_m > 0.0 and self.every_steps >= 1,
                "loss.aggregate.bin_s/cell_temp_m must be positive and every_steps >= 1")

    @property
    def enabled(self) -> bool:
        return self.weight > 0.0


@config_dataclass
class PairedDiffLossConfig:
    """PAIRLOSS paired-difference loss (``wms.train.losses.paired_diff``; weight 0 = off).

    With ``weight`` > 0 and a ``paired`` C4 source, a ``fraction`` of the TRAIN batches are
    PAIRED batches: ``groups`` pair groups (same branch state, arms differ only in the CAV
    actions after t0) with up to ``max_arms`` arms each (reference = the all-native arm, else
    arm 0).  Per group the predicted arm-minus-reference differences of the t0 HDV positions /
    speeds (``position_weight`` / ``speed_weight``) and of the per-bin cohort aggregates
    (road / RSU-zone delay, throughput past the merge end; ``aggregate_weight``) are matched
    to the recorded differences (Huber, normalised).  Optional return-difference and
    ranking terms use ``return_weight`` / ``rank_weight``; ``rank_margin_veh_s`` is
    the ranking margin in vehicle-seconds.  ``couple_randomness`` shares stochastic
    draws across arms of a pair group to reduce noise in their differences.  ``horizon_mode = "fixed"`` keeps the
    legacy branch-length cap; ``"stage"`` selects eligible groups and windows for each
    curriculum stage's unroll horizon.
    """

    weight: float = 0.0
    fraction: float = 0.25
    groups: int = 2
    max_arms: int = 5
    position_weight: float = 1.0
    speed_weight: float = 1.0
    aggregate_weight: float = 1.0
    couple_randomness: bool = False
    return_weight: float = 0.0
    rank_weight: float = 0.0
    rank_margin_veh_s: float = 5.0
    horizon_mode: Literal["fixed", "stage"] = "fixed"

    def __post_init__(self) -> None:
        require(all(is_finite(x) and x >= 0.0 for x in (
            self.weight, self.position_weight, self.speed_weight, self.aggregate_weight,
            self.return_weight, self.rank_weight)),
            "loss.paired_diff weights must be finite and non-negative")
        require(isinstance(self.couple_randomness, bool),
                "loss.paired_diff.couple_randomness must be a bool")
        require(is_finite(self.rank_margin_veh_s) and self.rank_margin_veh_s >= 0.0,
                "loss.paired_diff.rank_margin_veh_s must be finite and non-negative")
        require(is_finite(self.fraction) and 0.0 <= self.fraction <= 1.0,
                "loss.paired_diff.fraction must be in [0,1]")
        require(self.groups >= 1 and self.max_arms >= 2,
                "loss.paired_diff.groups must be >= 1 and max_arms >= 2")
        require(self.horizon_mode in ("fixed", "stage"),
                "loss.paired_diff.horizon_mode must be fixed or stage")

    @property
    def enabled(self) -> bool:
        return self.weight > 0.0


@config_dataclass
class DriverPriorConfig:
    """INFRAV5 optional driver-estimate prior (weight 0 = off).

    A weak L2 pull of the POPULATION MEAN of the encoded driver settings (mean over the
    valid t0 HDV slots of the batch) toward population targets; per-driver identification
    stays free (individual deviations from the batch mean are not penalised).  Targets
    default to the medians of the SUMO driver population (scenarios/idm/driver_routes.rou.xml:
    normal_car 70 % / aggressive 10 % / cautious 20 % -> every median = the normal_car value):
    accel 2.6, time headway 1.1 s, decel 4.5, minGap 2.5 m, v0 / speed limit = speedFactor 1.0.
    Each term is relative: ``((mean - target) / target) ** 2``.  Needs
    ``curriculum.driver_grad_across_chunks`` (the single t0 encoding).
    ``target``: "explicit" (default) uses the five values below; "legacy" / "textbook" use
    the medians of the legacy SUMO population (= the defaults below) / of the c4/popfix
    textbook population (a 1.2, T 1.2, b 2.0, s0 2.0, speedFactor 1.0) and ignore them.
    """

    weight: float = 0.0
    target: Literal["explicit", "legacy", "textbook"] = "explicit"
    accel: float = 2.6
    time_headway: float = 1.1
    decel: float = 4.5
    min_gap: float = 2.5
    v0_ratio: float = 1.0

    def __post_init__(self) -> None:
        require(is_finite(self.weight) and self.weight >= 0.0,
                "loss.driver_prior.weight must be finite and non-negative")
        require(all(is_finite(x) and x > 0.0 for x in (
            self.accel, self.time_headway, self.decel, self.min_gap, self.v0_ratio)),
            "loss.driver_prior targets must be finite and positive")

    @property
    def enabled(self) -> bool:
        return self.weight > 0.0


@config_dataclass
class LossConfig:
    """All loss terms and diagnostics."""

    position_huber_delta_m: float = 5.0
    speed_huber_delta_mps: float = 1.0
    position_weight: float = 1.0
    speed_weight: float = 1.0
    lane_weight: float = 1.0
    step_weight_decay: float = 0.01
    braking: BrakingConfig = field(default_factory=BrakingConfig)
    residual: ResidualLossConfig = field(default_factory=ResidualLossConfig)
    lane_rate: LaneRateConfig = field(default_factory=LaneRateConfig)
    flip_rate: FlipRateConfig = field(default_factory=FlipRateConfig)
    lane_event: LaneEventConfig = field(default_factory=LaneEventConfig)
    traffic: TrafficLossConfig = field(default_factory=TrafficLossConfig)
    aggregate: AggregateLossConfig = field(default_factory=AggregateLossConfig)
    paired_diff: PairedDiffLossConfig = field(default_factory=PairedDiffLossConfig)
    driver_prior: DriverPriorConfig = field(default_factory=DriverPriorConfig)
    stability_weight: float = 0.0
    stability_margin: float = 1.0
    stability_probes: int = 1
    conservation_enabled: bool = False
    conservation_weight: float = 0.0
    loss_share_logging: bool = False
    lane_hazard_decouple: bool = False
    """LANEFIX: in hazard mode the rollout lane CE and the flip-rate guard do not reach the
    hazard parameters (its delta enters the rollout logits detached); loss.lane_event is the
    hazard's only objective."""

    def __post_init__(self) -> None:
        require(self.stability_weight >= 0.0 and self.conservation_weight >= 0.0,
                "stability/conservation weights must be non-negative")
        require(self.stability_probes >= 1, "loss.stability_probes must be positive")


@config_dataclass
class CurriculumConfig:
    """Unroll-horizon curriculum."""

    unroll_s: tuple[float, ...] = (10.0, 30.0, 60.0, 120.0)
    stage_steps: Optional[tuple[int, ...]] = None
    truncation_steps: int = 20
    truncation_steps_per_stage: Optional[tuple[int, ...]] = None
    driver_grad_across_chunks: bool = False
    epochs_per_stage: int = 1
    steps_per_epoch: int = 2000
    max_hours: Optional[float] = None
    # SCHED (2026-10-14, SCHED_NOTES.md; all default OFF = unchanged loop): validate every
    # horizon <= the stage, multi-horizon checkpoint selection, short mixed-horizon batches.
    val_all_horizons: bool = False
    selection: Literal["long_horizon", "multi_horizon", "multi_horizon_v2"] = "long_horizon"
    selection_horizons_s: tuple[float, ...] = (10.0, 30.0, 60.0, 120.0)
    selection_reference: tuple[float, ...] = (7.0356, 49.4889, 128.7179, 0.0)
    """Per-horizon reference ``mean_position`` (C4 main WM best val at 10/30/60 s); a value
    <= 0 normalises by this run's own first validation at that horizon (120 s: none in C4).
    INFRAV5 ``selection = "multi_horizon_v2"``: a value <= 0 normalises by the best candidate
    of the final same-horizon comparison instead (never by the run's own first validation)."""
    selection_min_stage_s: float = 30.0
    selection_agg_reward_weight: float = 0.0
    mixed_horizon_frac: float = 0.0
    mixed_horizon_s: tuple[float, ...] = (10.0, 20.0)
    mixed_horizon_min_stage_s: float = 30.0

    def __post_init__(self) -> None:
        require(len(self.selection_horizons_s) >= 1
                and all(is_finite(x) and x > 0 for x in self.selection_horizons_s)
                and len(self.selection_reference) == len(self.selection_horizons_s)
                and all(is_finite(x) for x in self.selection_reference),
                "curriculum.selection_horizons_s must be positive and selection_reference "
                "one finite value per horizon")
        require(is_finite(self.selection_agg_reward_weight)
                and self.selection_agg_reward_weight >= 0.0
                and is_finite(self.selection_min_stage_s),
                "curriculum.selection_agg_reward_weight must be finite and non-negative")
        require(is_finite(self.mixed_horizon_frac) and 0.0 <= self.mixed_horizon_frac <= 1.0
                and len(self.mixed_horizon_s) >= 1
                and all(is_finite(x) and x > 0 for x in self.mixed_horizon_s)
                and is_finite(self.mixed_horizon_min_stage_s),
                "curriculum.mixed_horizon_frac must be in [0,1] and mixed_horizon_s positive")
        require(bool(self.unroll_s) and all(x > 0 for x in self.unroll_s),
                "curriculum.unroll_s must be a non-empty list of positive horizons")
        require(self.epochs_per_stage >= 1 and self.steps_per_epoch >= 1,
                "curriculum.epochs_per_stage and steps_per_epoch must be positive")
        require(self.truncation_steps >= 1, "curriculum.truncation_steps must be positive")
        if self.stage_steps is not None:
            require(len(self.stage_steps) == len(self.unroll_s) and all(s >= 1 for s in self.stage_steps),
                    "curriculum.stage_steps needs one positive count per curriculum stage")
        if self.truncation_steps_per_stage is not None:
            require(len(self.truncation_steps_per_stage) == len(self.unroll_s)
                    and all(s >= 1 for s in self.truncation_steps_per_stage),
                    "curriculum.truncation_steps_per_stage needs one positive "
                    "truncation per curriculum stage")
        require(self.max_hours is None or (is_finite(self.max_hours) and self.max_hours > 0),
                "curriculum.max_hours must be finite and positive when set")


@config_dataclass
class OptimConfig:
    """Optimiser settings."""

    batch_size: int = 8
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    lr_per_stage: Optional[tuple[float, ...]] = None
    """Optional learning rate per curriculum stage (round 5b); None = ``lr`` throughout."""

    def __post_init__(self) -> None:
        require(self.batch_size >= 1, "optim.batch_size must be positive")
        require(self.lr > 0.0 and self.weight_decay >= 0.0 and self.grad_clip > 0.0,
                "optim.lr/grad_clip must be positive and weight_decay non-negative")
        if self.lr_per_stage is not None:
            require(len(self.lr_per_stage) >= 1
                    and all(is_finite(x) and x > 0.0 for x in self.lr_per_stage),
                    "optim.lr_per_stage needs positive finite learning rates")


@config_dataclass
class SelectionConfig:
    """Validation cadence and long-horizon checkpoint selection."""

    select_by_long_horizon: bool = False
    val_every: int = 200
    val_batches: int = 8
    # 2026-10-14: cap on validation windows per evaluation (0 = all, unchanged). The evaluator ran
    # the WHOLE validation set; with the GENRAND sources (~8,300 val windows) one 60 s validation took
    # >1 h. A fixed, evenly spaced subset keeps validation/selection comparable across steps.
    val_max_windows: int = 0

    def __post_init__(self) -> None:
        require(self.val_every >= 1 and self.val_batches >= 1,
                "selection.val_every and val_batches must be positive")
        require(self.val_max_windows >= 0, "selection.val_max_windows must be >= 0")


@config_dataclass
class RuntimeConfig:
    """Device, threading and output location."""

    device: str = "cpu"
    threads: int = 6
    num_workers: int = 0
    seed: int = 101
    omp: int = 2
    out_dir: str = "runs/active/wms"

    def __post_init__(self) -> None:
        require(self.threads >= 1 and self.omp >= 1, "runtime.threads and omp must be positive")
        require(self.num_workers >= 0, "runtime.num_workers must be non-negative")


@config_dataclass
class TrainConfig:
    """Complete training configuration (schema version 2)."""

    schema_version: int = 2
    name: str = "wms"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    physics: PhysicsOptions = field(default_factory=PhysicsOptions)

    def __post_init__(self) -> None:
        require(self.schema_version == 2, f"unsupported schema_version {self.schema_version}")
        if self.data.history_s is not None:
            require(self.model.kind == "structured", "long history requires structured WM-S")
        if self.model.kind != "structured":
            require(not self.loss.braking.any_on and self.model.residual_cap_neg_mps2 is None,
                    "enabled braking terms require model.kind='structured'")
        require(not self.loss.lane_hazard_decouple
                or self.model.lane_model in ("hazard", "hazard_v5"),
                "loss.lane_hazard_decouple requires model.lane_model='hazard'")

    # -- training-time (stricter than construction) ---------------------
    def validate_for_training(self) -> None:
        """Checks that gate a training run but must not block reading old receipts."""
        require(self.runtime.num_workers == 0,
                "runtime.num_workers must be 0 (lazy closures are not picklable)")
        require(not self.model.has_hard_cap or self.model.allow_rejected_hard_cap,
                "hard residual caps were REJECTED (change-log #3); "
                "set model.allow_rejected_hard_cap only to re-score old runs")
        if self.loss.driver_prior.enabled:
            require(self.curriculum.driver_grad_across_chunks
                    and self.model.rollout_mode != "residual",
                    "loss.driver_prior.weight > 0 requires curriculum.driver_grad_across_chunks "
                    "and an encoder-driven rollout_mode (driver_params / hybrid)")
        if self.loss.paired_diff.enabled:
            require(any(src.paired for src in self.data.c4_sources),
                    "loss.paired_diff.weight > 0 requires a data.c4_sources entry with paired = true")
        if self.selection.select_by_long_horizon:
            require(self.data.windows.val is not None or self.data.windows.val_holdout_frac > 0.0
                    or self.data.c4_base is not None,   # V5DATA: c4 sources' world split
                    "selection.select_by_long_horizon requires a validation window list")
        for text in self._path_strings():
            token = find_restricted_token(text)
            if token is not None:
                raise TrainDataRefusal(f"config path {text!r} contains restricted token {token!r}")

    def _path_strings(self) -> list[str]:
        """Every string in the data section (logical names and paths)."""
        found: list[str] = []

        def walk(node: Any) -> None:
            if isinstance(node, str):
                found.append(node)
            elif isinstance(node, Mapping):
                for value in node.values():
                    walk(value)
            elif isinstance(node, (list, tuple)):
                for value in node:
                    walk(value)

        walk(to_mapping(self.data))
        walk(self.model.physics_params)
        return found

    def required_log_terms(self) -> tuple[str, ...]:
        """Loss-term names the training log must contain, in the shell chain's order."""
        terms = ["position", "speed", "lane"]
        if self.loss.traffic.enabled:
            terms.append("traffic")
        braking = self.loss.braking
        for name, enabled in (("asym_speed", braking.asym_speed_alpha > 1.0),
                              ("decel_weight", braking.decel_weight > 1.0),
                              ("propagation", braking.propagation_weight > 0.0),
                              ("stability", self.loss.stability_weight > 0.0),
                              ("conservation", self.loss.conservation_weight > 0.0)):
            if enabled:
                terms.append(name)
        agg = self.loss.aggregate
        if agg.enabled:
            terms.extend(name for name, w in (("agg_field", agg.field_weight),
                                              ("agg_reward", agg.reward_weight),
                                              ("agg_flow", agg.flow_weight)) if w > 0.0)
        pair = self.loss.paired_diff
        if pair.enabled:
            terms.extend(name for name, w in (("pair_pos", pair.position_weight),
                                              ("pair_speed", pair.speed_weight),
                                              ("pair_agg", pair.aggregate_weight),
                                              ("pair_return", pair.return_weight),
                                              ("pair_rank", pair.rank_weight)) if w > 0.0)
        return tuple(terms)

    # -- legacy flat checkpoint/receipt format --------------------------
    def to_legacy_flat(self, resolve: Optional[Callable[[str], str]] = None) -> dict[str, Any]:
        """Flat dict identical to the legacy ``WmSTrainConfig.to_dict()``.

        ``resolve`` maps logical data names (window lists, driver sidecar) to local
        paths, since checkpoints store resolved paths.
        """
        full = self._flat_full(resolve or (lambda name: name))
        defaults = _default_flat()
        for key in OMIT_WHEN_DEFAULT:
            if full[key] == defaults[key]:
                del full[key]
        if self.data.cav_drive == "teacher":
            del full["cav_drive"]
        if self.data.history_s is None:
            del full["history_s"]
            del full["history_stride_s"]
        return full

    def _flat_full(self, resolve: Callable[[str], str]) -> dict[str, Any]:
        tree = to_mapping(self)
        flat = {key: get_path(tree, path) for key, path in FLAT_TO_PATH.items()}
        flat["black_box"] = self.model.kind == "blackbox_id"
        exclude = [resolve(name) for name in self.data.windows.exclude]
        flat["exclude_window_list"] = exclude or None
        for key in ("window_list", "val_window_list", "driver_sidecar"):
            if flat[key] is not None:
                flat[key] = resolve(flat[key])
        for key in ("unroll_curriculum_s", "stage_steps"):
            if flat[key] is not None:
                flat[key] = list(flat[key])
        return flat

    @classmethod
    def from_legacy_flat(cls, flat: Mapping[str, Any]) -> "TrainConfig":
        """Rebuild a config from a legacy flat dict (``ckpt["config"]``, receipt ``config``).

        Missing keys take their defaults (old receipts predate newer fields); unknown
        keys are an error, as in the legacy ``from_dict``.
        """
        unknown = sorted(set(flat) - set(FLAT_TO_PATH))
        if unknown:
            raise ConfigError(f"unknown WM-S config keys: {unknown}")
        tree: dict[str, Any] = {}
        for key, value in flat.items():
            if key == "black_box":
                value = "blackbox_id" if value else "structured"
            elif key == "exclude_window_list":
                value = _as_list(value)
            set_path(tree, FLAT_TO_PATH[key], value)
        return from_mapping(cls, tree)

    def fingerprint(self) -> str:
        """SHA-256 of the canonical nested JSON of this config."""
        return fingerprint(self)

    def to_dict(self) -> dict[str, Any]:
        """Nested plain-dict form (what ``config_v2`` stores in checkpoints)."""
        return to_mapping(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainConfig":
        """Inverse of :meth:`to_dict`; strict about unknown keys."""
        return from_mapping(cls, data)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _default_flat() -> dict[str, Any]:
    """Flat form of the all-default config (used to decide which keys to omit)."""
    return TrainConfig()._flat_full(lambda name: name)
