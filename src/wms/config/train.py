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
#: V8WM ``physics.overrides``: numeric PhysicsConfig parameters a config may set (gap rule).
PHYSICS_OVERRIDE_KEYS = (
    "gap_s0_lead_m", "gap_t_lead_s", "gap_b_lead_mps2", "gap_s0_follow_m", "gap_t_follow_s",
    "gap_b_follow_mps2", "gap_overlap_margin_m", "gap_anticipation_s",
    "gap_stopped_speed_mps", "gap_stopped_min_m",
    "merge_end_wait_distance_m", "merge_end_wait_speed_mps",
    "mainline_merge_yield_distance_m", "mainline_merge_yield_decel_mps2",
    # V8WM2 (2026-10-17): lane-end hazard floor, merge_end_wait follower constants.
    "forced_window_min_hazard", "merge_end_wait_s0_follow_m", "merge_end_wait_t_follow_s",
    # V8WM3 (2026-10-17): lane-end waiting-time impatience parameters.
    "merge_impatience_distance_m", "merge_impatience_speed_mps", "merge_impatience_hazard_tau_s",
    "merge_impatience_gap_tau_s", "merge_impatience_s0_min_m", "merge_impatience_t_min_s",
    # V10LANEOPT (2026-10-19): courtesy approach-zone length upstream of the merge end.
    "mainline_courtesy_zone_m")
#: V8WM3: boolean PhysicsConfig switches ``physics.overrides`` may set (training AND inference).
PHYSICS_BOOL_OVERRIDE_KEYS = ("merge_impatience", "entry_learned")
#: V8WM2: string-valued PhysicsConfig modes ``physics.overrides`` / ``physics.inference`` may set
#: (key -> allowed values).
PHYSICS_MODE_KEYS = {"forced_window_mode": ("deterministic", "hazard"),
                     "mainline_merge_yield_scope": ("lane_end", "zone"),
                     "mainline_lane_mode": ("free", "courtesy")}
#: V8WM ``physics.inference``: boolean switches that may differ at inference ...
PHYSICS_INFERENCE_SWITCHES = ("merge_end_wait", "mainline_merge_yield", "gap_stopped_relax",
                              "merge_impatience", "lane_learnable")
#: ... plus the numeric override keys (and the V8WM2 mode keys).
PHYSICS_INFERENCE_KEYS = PHYSICS_INFERENCE_SWITCHES + PHYSICS_OVERRIDE_KEYS \
    + tuple(PHYSICS_MODE_KEYS)


def _mode_ok(key: str, value: Any) -> bool:
    return isinstance(value, str) and value in PHYSICS_MODE_KEYS[key]


@config_dataclass
class PhysicsOptions:
    """Opt-in integration settings; the default global plant braking limit is 9 m/s²."""

    stop_time_integration: bool = False
    emergency_decel_mps2: Optional[float] = None
    exit_at_road_end: bool = True
    """Plain-bug fix (default true): rollouts remove vehicles past 553.33 m.  False reproduces
    pre-fix results bit-identically (``wms.physics.config.PhysicsConfig.exit_at_road_end``)."""
    merge_end_wait: bool = False
    """V6MERGE: lane-end mergers need both lane-1 gaps SUMO-secure and never pass the end."""
    mainline_merge_yield: bool = False
    """V6MERGE: the nearest lane-1 follower brakes (own IDM) for a blocked lane-end merger."""
    lane_learnable: bool = False
    """V10UNBLOCK (2026-10-19, default false = bit-identical): replace behavioural lane-change
    vetoes by the physical envelope and learn merge / lane-change / yield from data
    (``wms.physics.config.PhysicsConfig.lane_learnable``)."""
    mainline_lane_mode: str = "free"
    """V10LANEOPT (2026-10-19, default 'free' = current behaviour, ``lane_learnable`` only):
    staged mainline lane changes.  'courtesy' keeps merges fully learned and allows only the
    learned lane-1 -> lane-0 overtakes of HDVs inside the approach zone upstream of the merge
    (``wms.physics.config.PhysicsConfig.mainline_lane_mode``)."""
    mainline_courtesy_zone_m: float = 300.0
    """V10LANEOPT: length (m) of that approach zone upstream of the acceleration-lane end."""
    overrides: dict[str, Any] = field(default_factory=dict)
    """V8WM (2026-10-17, default empty = unchanged): numeric gap-rule / lane-end parameters of
    ``wms.physics.config.PhysicsConfig`` (keys :data:`PHYSICS_OVERRIDE_KEYS`: the speed-rule
    secure-gap constants ``gap_*``, the stopped-merger relaxation ``gap_stopped_*`` and the
    ``merge_end_wait_*`` / ``mainline_merge_yield_*`` thresholds; V8WM2: also
    ``forced_window_min_hazard``, ``merge_end_wait_s0_follow_m`` / ``_t_follow_s`` and the
    string modes :data:`PHYSICS_MODE_KEYS` ``forced_window_mode`` /
    ``mainline_merge_yield_scope``), applied to the TRAINING physics (and therefore also to the
    saved inference physics)."""
    inference: dict[str, Any] = field(default_factory=dict)
    """V8WM (default empty = physics.json equals the training physics): PhysicsConfig fields
    (keys :data:`PHYSICS_INFERENCE_KEYS`: ``merge_end_wait``, ``mainline_merge_yield``,
    ``gap_stopped_relax`` and any :data:`PHYSICS_OVERRIDE_KEYS`) that differ at INFERENCE.  The
    trainer trains with the physics above and writes ``physics.json`` (read by every checkpoint
    loader: evaluators, the Dreamer policy trainer) with these values applied; the training
    physics is kept as ``physics_train.json``."""

    def __post_init__(self) -> None:
        require(self.emergency_decel_mps2 is None or (
            is_finite(self.emergency_decel_mps2) and 0.0 < self.emergency_decel_mps2 <= 9.0),
            "physics.emergency_decel_mps2 must be finite and in (0, 9] when set")
        require(isinstance(self.exit_at_road_end, bool),
                "physics.exit_at_road_end must be a boolean")
        require(_mode_ok("mainline_lane_mode", self.mainline_lane_mode),
                f"physics.mainline_lane_mode must be one of "
                f"{PHYSICS_MODE_KEYS['mainline_lane_mode']}")
        require(is_finite(self.mainline_courtesy_zone_m) and self.mainline_courtesy_zone_m >= 0.0,
                "physics.mainline_courtesy_zone_m must be finite and >= 0")
        valid = (list(PHYSICS_OVERRIDE_KEYS) + list(PHYSICS_MODE_KEYS)
                 + list(PHYSICS_BOOL_OVERRIDE_KEYS))
        bad = sorted(set(self.overrides) - set(valid))
        require(not bad, f"physics.overrides: unknown keys {bad}; valid: {valid}")
        for key, value in self.overrides.items():
            if key in PHYSICS_MODE_KEYS:
                require(_mode_ok(key, value), f"physics.overrides.{key} must be one of "
                                              f"{PHYSICS_MODE_KEYS[key]}")
            elif key in PHYSICS_BOOL_OVERRIDE_KEYS:
                require(isinstance(value, bool), f"physics.overrides.{key} must be a boolean")
            else:
                require(is_finite(value), "physics.overrides values must be finite numbers")
        bad = sorted(set(self.inference) - set(PHYSICS_INFERENCE_KEYS))
        require(not bad, f"physics.inference: unknown keys {bad}; valid: "
                         f"{list(PHYSICS_INFERENCE_KEYS)}")
        for key, value in self.inference.items():
            if key in PHYSICS_INFERENCE_SWITCHES:
                require(isinstance(value, bool), f"physics.inference.{key} must be a boolean")
            elif key in PHYSICS_MODE_KEYS:
                require(_mode_ok(key, value), f"physics.inference.{key} must be one of "
                                              f"{PHYSICS_MODE_KEYS[key]}")
            else:
                require(is_finite(value), f"physics.inference.{key} must be a finite number")


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
    lane_v5_mainline: Literal["default", "sample", "budget", "budget_prop"] = "default"
    """V8LANE: overtake / return decision rule ('default' = lane_decision; 'sample' = always draw
    the calibrated hazard; 'budget' / 'budget_prop' = population-matched count, top-hazard /
    hazard-proportional car).  Merges keep lane_decision."""
    lane_conflict_resolve: bool = False
    """V6INTEG: resolve simultaneous lane-change conflicts during rollout."""
    courtesy_inputs: bool = False
    """V10COURTESY: add the ramp car's distance-to-lane-end, speed / slow / waiting
    time and the merge head's current probability to the residual inputs.  Default
    OFF: the residual sees kinematics only and old checkpoints load unchanged.  The
    new adapters are zero-initialised, so enabling it is an exact no-op until trained."""
    residual_head_rescale: bool = False
    """R5MERGE: divide the residual head's pre-tanh activation by a learned positive
    scalar exp(log_scale), log_scale initialised to 0.  The output is bit-identical at
    load (exp(0) = 1) and the scalar can shrink the activation back out of the tanh
    tail so the head regains gradient.  Default OFF: identity and no new parameter."""
    merge_negotiation: bool = False
    """V10NEGOT (default OFF): learned ramp-car / lane-1-follower merge-negotiation block
    (zero-initialised).  The follower gets a learned yield acceleration and the ramp car a
    merge-logit correction; trained through the rollout, merge-survival and matched
    non-yield negative losses.  Needs ``model.lane_model='hazard_v5'``."""

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
        require(self.lane_v5_mainline in ("default", "sample", "budget", "budget_prop"),
                "model.lane_v5_mainline must be default, sample, budget or budget_prop")
        if self.lane_v5_mainline != "default":
            require(self.lane_model == "hazard_v5",
                    "model.lane_v5_mainline needs model.lane_model='hazard_v5'")
        require(is_finite(self.grad_decay_per_step) and 0.0 < self.grad_decay_per_step <= 1.0,
                "model.grad_decay_per_step must be in (0, 1]")
        require(isinstance(self.courtesy_inputs, bool),
                "model.courtesy_inputs must be a boolean")
        require(isinstance(self.residual_head_rescale, bool),
                "model.residual_head_rescale must be a boolean")
        require(isinstance(self.merge_negotiation, bool),
                "model.merge_negotiation must be a boolean")
        if self.merge_negotiation:
            require(self.lane_model == "hazard_v5",
                    "model.merge_negotiation requires model.lane_model='hazard_v5'")
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
class BrakePersistConfig:
    """BRAKE_DIAG multi-step braking persistence terms (all default OFF)."""

    weight: float = 0.0
    quantile_weight: float = 0.0
    event_balance: float = 0.0
    window_s: float = 4.0

    def __post_init__(self) -> None:
        require(all(is_finite(getattr(self, name)) for name in (
            "weight", "quantile_weight", "event_balance", "window_s")),
            "loss.brake_persist values must be finite")
        require(self.weight >= 0.0 and self.quantile_weight >= 0.0,
                "loss.brake_persist weights must be >= 0 (0 = off)")
        require(0.0 <= self.event_balance <= 4.0,
                "loss.brake_persist.event_balance must be in [0, 4] (0 = off)")
        require(3.0 <= self.window_s <= 5.0,
                "loss.brake_persist.window_s must be in [3, 5] seconds")

    @property
    def enabled(self) -> bool:
        return self.weight > 0.0 or self.quantile_weight > 0.0


@config_dataclass
class BrakeWaveConfig:
    """WAVE onset-conditioned braking terms (all default OFF)."""

    weight: float = 0.0
    residual_weight: float = 0.0
    spurious_weight: float = 0.0
    window_s: float = 4.0
    onset_decel_mps2: float = 1.0
    spurious_decel_mps2: float = 1.5

    def __post_init__(self) -> None:
        require(all(is_finite(getattr(self, name)) for name in (
            "weight", "residual_weight", "spurious_weight", "window_s",
            "onset_decel_mps2", "spurious_decel_mps2")),
            "loss.brake_wave values must be finite")
        require(self.weight >= 0.0 and self.residual_weight >= 0.0
                and self.spurious_weight >= 0.0,
                "loss.brake_wave weights must be >= 0 (0 = off)")
        require(3.0 <= self.window_s <= 5.0,
                "loss.brake_wave.window_s must be in [3, 5] seconds")
        require(self.onset_decel_mps2 > 0.0 and self.spurious_decel_mps2 > 0.0,
                "loss.brake_wave decel thresholds must be positive")

    @property
    def enabled(self) -> bool:
        return (self.weight > 0.0 or self.residual_weight > 0.0
                or self.spurious_weight > 0.0)


@config_dataclass
class PostMergeConfig:
    """R5MERGE post-merge acceleration-consistency terms (all default OFF)."""

    weight: float = 0.0
    residual_weight: float = 0.0
    window_s: float = 10.0
    huber_delta_mps2: float = 0.5

    def __post_init__(self) -> None:
        require(all(is_finite(getattr(self, name)) for name in (
            "weight", "residual_weight", "window_s", "huber_delta_mps2")),
            "loss.post_merge values must be finite")
        require(self.weight >= 0.0 and self.residual_weight >= 0.0,
                "loss.post_merge weights must be >= 0 (0 = off)")
        require(2.0 <= self.window_s <= 20.0,
                "loss.post_merge.window_s must be in [2, 20] seconds")
        require(self.huber_delta_mps2 > 0.0,
                "loss.post_merge.huber_delta_mps2 must be positive")

    @property
    def enabled(self) -> bool:
        return self.weight > 0.0 or self.residual_weight > 0.0


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
class AggDelayConfig:
    """A5 long-horizon aggregate delay loss (``wms.train.losses.agg_horizon``; 0 = off).

    Matches WM-rollout aggregates to the recorded SUMO over the long window
    ``[start_s, horizon_s]`` (default 120-300 s, where ONROAD_DIAG localises 91 % of the
    native delay excess): the on-road team delay ``dt * relu(1 - v / 31.29)`` split by region
    (upstream / merge zone / ramp / downstream; bounds ``merge_s0_m`` / ``merge_s1_m`` /
    ``merge_lane``), the per-region queue length (region 2 = the ramp queue; ``queue_speed_mps``)
    and soft throughput crossings of the fixed ``sections_m`` plus the ramp merge count.
    ``paired_weight`` adds the paired DIFFERENCE of those aggregates on paired branch batches.
    Scales are ``delay_scale_veh_s`` (an extra coefficient), ``queue_scale_veh`` and
    ``throughput_scale_veh``; ``bin_s`` / ``region_temp_m`` / ``queue_temp_mps`` shape the bins
    and the soft memberships.
    """

    weight: float = 0.0
    paired_weight: float = 0.0
    start_s: float = 120.0
    horizon_s: float = 300.0
    fade_s: float = 0.0
    bin_s: float = 5.0
    merge_s0_m: float = 0.0
    merge_s1_m: float = 75.44
    merge_lane: int = 2
    road_end_m: float = 553.33
    sections_m: tuple[float, ...] = (-400.0, 0.0, 75.44, 400.0)
    queue_speed_mps: float = 5.0
    region_temp_m: float = 5.0
    queue_temp_mps: float = 1.0
    delay_scale_veh_s: float = 1.0
    queue_scale_veh: float = 1.0
    throughput_scale_veh: float = 1.0
    huber_delta: float = 1.0

    def __post_init__(self) -> None:
        require(all(is_finite(x) and x >= 0.0 for x in (
            self.weight, self.paired_weight, self.start_s, self.horizon_s, self.fade_s)),
            "loss.agg_delay weights/window must be finite and non-negative")
        require(all(is_finite(x) and x > 0.0 for x in (
            self.bin_s, self.region_temp_m, self.queue_temp_mps, self.delay_scale_veh_s,
            self.queue_scale_veh, self.throughput_scale_veh, self.huber_delta)),
            "loss.agg_delay bin/temperatures/scales must be finite and positive")
        require(is_finite(self.merge_s0_m) and is_finite(self.merge_s1_m)
                and self.merge_s1_m > self.merge_s0_m,
                "loss.agg_delay merge_s1_m must be finite and exceed merge_s0_m")
        require(self.merge_lane >= 1, "loss.agg_delay merge_lane must be >= 1")
        require(is_finite(self.road_end_m) and self.road_end_m > 0.0,
                "loss.agg_delay road_end_m must be finite and positive")
        require(len(self.sections_m) >= 1 and all(is_finite(float(x)) for x in self.sections_m),
                "loss.agg_delay sections_m must be non-empty finite positions")

    @property
    def enabled(self) -> bool:
        return self.weight > 0.0 or self.paired_weight > 0.0


@config_dataclass
class MergeLossConfig:
    """V7FT merge-zone calibration loss (``wms.train.losses.merge``; both weights 0 = off).

    ``speed_weight``: HDV mean speed per cross-section cell (edges ``cell_edges_m``, s = 0 =
    merge start) of the rollout vs the record, pooled over ``bin_s`` bins (Huber of the error
    in units of ``speed_scale_mps``).  ``discharge_weight``: cumulative soft count of HDVs
    crossing ``line_m`` since t0 (sigmoid width ``cross_temp_m``) vs the record (Huber in
    units of ``discharge_scale_veh``).  CAVs are teacher-forced and excluded.
    """

    speed_weight: float = 0.0
    discharge_weight: float = 0.0
    cell_edges_m: tuple[float, ...] = (-100.0, 0.0, 75.44, 300.0)
    line_m: float = 75.44
    bin_s: float = 5.0
    cell_temp_m: float = 2.0
    cross_temp_m: float = 5.0
    speed_scale_mps: float = 1.0
    discharge_scale_veh: float = 2.0
    speed_cells: Optional[tuple[int, ...]] = None
    """V8WM: indices of the cells (``cell_edges_m`` intervals) whose speed error is
    optimised; ``None`` (default) = every cell (v7 behaviour).  v8: ``[2]`` = only the
    downstream cell [75.44, 300) m.  Validation metrics still report every cell."""
    max_share: Optional[float] = None
    """V8WM: cap on the merge terms' (speed + discharge) share of each chunk's loss VALUE
    (all chunk terms; once-per-unroll terms such as loss.lane_event are not counted).  When
    exceeded both terms are scaled by one detached factor so the share equals the cap
    (``wms.train.losses.merge.cap_merge_share``).  ``None`` (default) = no cap."""

    def __post_init__(self) -> None:
        require(all(is_finite(x) and x >= 0.0 for x in (self.speed_weight,
                                                        self.discharge_weight)),
                "loss.merge weights must be finite and non-negative")
        n_cells = len(self.cell_edges_m) - 1
        require(self.speed_cells is None or (
            len(self.speed_cells) >= 1 and len(set(self.speed_cells)) == len(self.speed_cells)
            and all(0 <= int(c) < n_cells for c in self.speed_cells)),
            "loss.merge.speed_cells must be distinct cell indices in [0, len(cell_edges_m) - 1)")
        require(self.max_share is None or (is_finite(self.max_share)
                                           and 0.0 < self.max_share < 1.0),
                "loss.merge.max_share must be in (0, 1) when set")
        edges = [float(x) for x in self.cell_edges_m]
        require(len(edges) >= 2 and all(is_finite(x) for x in edges)
                and all(b > a for a, b in zip(edges[:-1], edges[1:])),
                "loss.merge.cell_edges_m must be >= 2 finite increasing values")
        require(is_finite(self.line_m), "loss.merge.line_m must be finite")
        require(all(is_finite(x) and x > 0.0 for x in (
            self.bin_s, self.cell_temp_m, self.cross_temp_m, self.speed_scale_mps,
            self.discharge_scale_veh)),
            "loss.merge bin/temperatures/scales must be finite and positive")

    @property
    def enabled(self) -> bool:
        return self.speed_weight > 0.0 or self.discharge_weight > 0.0


@config_dataclass
class MergeTimingConfig:
    """V8WM per-vehicle merge-timing (survival) loss (``wms.train.losses.merge_timing``).

    ``weight`` (0 = off): for every recorded HDV-step in the ramp lane the discrete-time
    survival NLL of the recorded lane-2 departure under the model's own per-step merge hazard
    on its free-running rollout state (physics-masked and forced-window steps excluded);
    normalised per recorded ramp vehicle.  ``metrics`` = compute the ramp / merge-timing
    validation metrics (stuck fraction, merge-time error, ramp-HDV |ds|) without the loss
    (needed by ``curriculum.selection_ramp_weight`` / ``selection_stuck_weight``).
    """

    weight: float = 0.0
    metrics: bool = False

    def __post_init__(self) -> None:
        require(is_finite(self.weight) and self.weight >= 0.0,
                "loss.merge_timing.weight must be finite and non-negative")

    @property
    def enabled(self) -> bool:
        return self.weight > 0.0 or bool(self.metrics)


@config_dataclass
class MergeSurvivalConfig:
    """V10NEGOT merge-survival loss (correctly censored discrete-time hazard).

    ``weight`` (0 = off): builds the per-vehicle merge-timing survival term with this weight
    (the recorded merge time is the event, a ramp car still in lane 2 at the end of the
    window / record is right-censored).  When the merge-negotiation block is enabled the same
    term trains its merge-logit correction; ``negative_weight`` additionally scores the
    matched non-yield negatives (a ramp car that stayed while its lane-1 follower did not
    decelerate).  ``metrics`` computes the stuck / merge-time validation metrics without the
    loss.
    """

    weight: float = 0.0
    negative_weight: float = 0.0
    metrics: bool = False

    def __post_init__(self) -> None:
        require(is_finite(self.weight) and self.weight >= 0.0,
                "loss.merge_survival.weight must be finite and non-negative")
        require(is_finite(self.negative_weight) and self.negative_weight >= 0.0,
                "loss.merge_survival.negative_weight must be finite and non-negative")

    @property
    def enabled(self) -> bool:
        return self.weight > 0.0 or self.negative_weight > 0.0 or bool(self.metrics)


@config_dataclass
class LaneFlowConfig:
    """V8LANE on-policy mainline lane-flow loss (``wms.train.losses.lane_flow``; 0 = off).

    ``weight``: pooled Poisson NLL of the expected overtake / return counts (sum of the
    hazard_v5 head probabilities over the rollout's own feasible rows) against the recorded
    counts of the same chunk, per window.  ``balance`` (0 = off): the lane-0 occupancy excess of
    the rollout over the record at the chunk start (HDV count) is added to the return target
    (deficit: to the overtake target) with this gain.
    """

    weight: float = 0.0
    balance: float = 0.0

    def __post_init__(self) -> None:
        require(is_finite(self.weight) and self.weight >= 0.0
                and is_finite(self.balance) and self.balance >= 0.0,
                "loss.lane_flow.weight / balance must be finite and non-negative")


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
    """INFRAV5 optional driver-estimate prior (weight 0 = off; owner decision pending).

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
    brake_follow_weight: float = 1.0
    """V10BRAKEWAVE: per-step position/speed multiplier for a follower whose same-lane
    recorded leader (<= 40 m) decelerates below -1 m/s^2 (1 = off, bit-identical)."""
    brake_follow_cav_weight: float = 0.0
    """V10CAVFOLLOW: extra per-step position/speed multiplier for an HDV follower whose
    same-lane recorded CAV leader (<= 40 m) decelerates below -1 m/s^2; the multiplier is
    ``1 + brake_follow_cav_weight`` on flagged steps (0 = off, bit-identical)."""
    step_weight_decay: float = 0.01
    braking: BrakingConfig = field(default_factory=BrakingConfig)
    brake_persist: BrakePersistConfig = field(default_factory=BrakePersistConfig)
    brake_wave: BrakeWaveConfig = field(default_factory=BrakeWaveConfig)
    post_merge: PostMergeConfig = field(default_factory=PostMergeConfig)
    residual: ResidualLossConfig = field(default_factory=ResidualLossConfig)
    lane_rate: LaneRateConfig = field(default_factory=LaneRateConfig)
    flip_rate: FlipRateConfig = field(default_factory=FlipRateConfig)
    lane_event: LaneEventConfig = field(default_factory=LaneEventConfig)
    traffic: TrafficLossConfig = field(default_factory=TrafficLossConfig)
    aggregate: AggregateLossConfig = field(default_factory=AggregateLossConfig)
    agg_delay: AggDelayConfig = field(default_factory=AggDelayConfig)
    paired_diff: PairedDiffLossConfig = field(default_factory=PairedDiffLossConfig)
    driver_prior: DriverPriorConfig = field(default_factory=DriverPriorConfig)
    merge: MergeLossConfig = field(default_factory=MergeLossConfig)
    merge_timing: MergeTimingConfig = field(default_factory=MergeTimingConfig)
    merge_survival: MergeSurvivalConfig = field(default_factory=MergeSurvivalConfig)
    lane_flow: LaneFlowConfig = field(default_factory=LaneFlowConfig)
    courtesy_weight: float = 1.0
    """V10COURTESY: per-step loss multiplier for lane-1 followers within 40 m behind a
    lane-2 car with distance-to-lane-end < 40 m.  1.0 (default) = off, bit-identical;
    values > 1 boost the courtesy events in the ordinary position/speed Huber terms."""
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
        require(is_finite(self.courtesy_weight) and float(self.courtesy_weight) >= 1.0,
                "loss.courtesy_weight must be finite and >= 1 (1 = off)")
        require(self.stability_probes >= 1, "loss.stability_probes must be positive")
        require(is_finite(self.brake_follow_weight) and self.brake_follow_weight >= 1.0,
                "loss.brake_follow_weight must be finite and >= 1 (1 = off)")
        require(is_finite(self.brake_follow_cav_weight)
                and self.brake_follow_cav_weight >= 0.0,
                "loss.brake_follow_cav_weight must be finite and >= 0 (0 = off)")


@config_dataclass
class CurriculumConfig:
    """Unroll-horizon curriculum."""

    unroll_s: tuple[float, ...] = (10.0, 30.0, 60.0, 120.0)
    stage_steps: Optional[tuple[int, ...]] = None
    truncation_steps: int = 20
    truncation_steps_per_stage: Optional[tuple[int, ...]] = None
    merge_teacher_frac_schedule: tuple[float, ...] = ()
    """V10NEGOT (default empty = off): per-curriculum-stage fraction of each window in which
    ramp cars follow the recorded lane (merge-decision teacher forcing).  The fraction decays
    linearly to 0 across each stage; the last value repeats for later stages."""
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
    selection_ramp_weight: float = 0.0
    """V8WM (``selection = "multi_horizon_v2"`` only; 0 = off): add
    ``w * mean_h(ramp |ds|_h / norm_h)`` over the validated horizons >= 60 s to the selection
    score (ramp-HDV position error, ``loss.merge_timing`` metrics; norm = the run's first value
    online, the best candidate in the final comparison)."""
    selection_stuck_weight: float = 0.0
    """V8WM (``multi_horizon_v2`` only; 0 = off): add ``w * mean_h |stuck_pred - stuck_rec|``
    (fraction of t0 ramp HDVs never leaving lane 2) over the validated horizons >= 60 s."""
    mixed_horizon_frac: float = 0.0
    mixed_horizon_s: tuple[float, ...] = (10.0, 20.0)
    mixed_horizon_min_stage_s: float = 30.0
    # V10HORIZON (2026-10-19, default OFF = fixed step budget, bit-identical): advance a
    # curriculum stage when its own-horizon validation error plateaus (the shorter horizon
    # stops improving) instead of after the fixed ``stage_steps`` budget.  Needs a validation
    # dataset; ``val_every`` sets the observation cadence and ``stage_steps`` stays the cap.
    plateau_advance: bool = False
    plateau_patience: int = 3
    """Number of consecutive non-improving validations before the stage advances."""
    plateau_min_delta: float = 0.01
    """Relative improvement of the best validation value that resets the patience counter."""
    plateau_min_steps: int = 2
    """Minimum number of validations observed before a plateau can advance the stage."""
    plateau_max_steps: int = 0
    """Observation cap forcing an advance (0 = none; the stage's step budget is the cap)."""

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
        require(all(is_finite(x) and x >= 0.0 for x in (self.selection_ramp_weight,
                                                        self.selection_stuck_weight)),
                "curriculum.selection_ramp_weight / selection_stuck_weight must be >= 0")
        require(is_finite(self.mixed_horizon_frac) and 0.0 <= self.mixed_horizon_frac <= 1.0
                and len(self.mixed_horizon_s) >= 1
                and all(is_finite(x) and x > 0 for x in self.mixed_horizon_s)
                and is_finite(self.mixed_horizon_min_stage_s),
                "curriculum.mixed_horizon_frac must be in [0,1] and mixed_horizon_s positive")
        require(isinstance(self.plateau_advance, bool)
                and self.plateau_patience >= 1 and self.plateau_min_steps >= 1
                and self.plateau_max_steps >= 0
                and is_finite(self.plateau_min_delta) and self.plateau_min_delta >= 0.0,
                "curriculum.plateau_* must be a bool plus positive patience/min_steps "
                "(max_steps >= 0) and a non-negative min_delta")
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
        require(all(is_finite(x) and 0.0 <= x <= 1.0
                    for x in self.merge_teacher_frac_schedule),
                "curriculum.merge_teacher_frac_schedule values must be in [0, 1]")


@config_dataclass
class OptimConfig:
    """Optimiser settings."""

    batch_size: int = 8
    batch_size_per_stage: Optional[tuple[int, ...]] = None
    """V10HORIZON: optional per-curriculum-stage batch size (long-horizon memory control);
    None (default) = every stage uses ``batch_size``."""
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
        if self.batch_size_per_stage is not None:
            require(len(self.batch_size_per_stage) >= 1
                    and all(int(x) >= 1 for x in self.batch_size_per_stage),
                    "optim.batch_size_per_stage needs positive batch sizes")


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
        if self.curriculum.selection_ramp_weight > 0.0 or self.curriculum.selection_stuck_weight > 0.0:
            require(self.curriculum.selection == "multi_horizon_v2"
                    and self.loss.merge_timing.enabled,
                    "curriculum.selection_ramp_weight / selection_stuck_weight need "
                    "curriculum.selection = 'multi_horizon_v2' and loss.merge_timing "
                    "(weight > 0 or metrics = true)")
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
                              ("brake_persist", self.loss.brake_persist.weight > 0.0),
                              ("brake_persist_quantile",
                               self.loss.brake_persist.quantile_weight > 0.0),
                              ("brake_wave", self.loss.brake_wave.weight > 0.0),
                              ("brake_wave_residual",
                               self.loss.brake_wave.residual_weight > 0.0),
                              ("brake_wave_spurious",
                               self.loss.brake_wave.spurious_weight > 0.0),
                              ("post_merge", self.loss.post_merge.weight > 0.0),
                              ("post_merge_residual",
                               self.loss.post_merge.residual_weight > 0.0),
                              ("stability", self.loss.stability_weight > 0.0),
                              ("conservation", self.loss.conservation_weight > 0.0)):
            if enabled:
                terms.append(name)
        agg = self.loss.aggregate
        if agg.enabled:
            terms.extend(name for name, w in (("agg_field", agg.field_weight),
                                              ("agg_reward", agg.reward_weight),
                                              ("agg_flow", agg.flow_weight)) if w > 0.0)
        delay = self.loss.agg_delay
        if delay.weight > 0.0:
            terms.extend(("agg_delay_region", "agg_queue", "agg_throughput"))
        if delay.paired_weight > 0.0:
            terms.append("agg_pair")
        pair = self.loss.paired_diff
        if pair.enabled:
            terms.extend(name for name, w in (("pair_pos", pair.position_weight),
                                              ("pair_speed", pair.speed_weight),
                                              ("pair_agg", pair.aggregate_weight),
                                              ("pair_return", pair.return_weight),
                                              ("pair_rank", pair.rank_weight)) if w > 0.0)
        merge = self.loss.merge
        if merge.enabled:
            terms.extend(name for name, w in (("merge_speed", merge.speed_weight),
                                              ("merge_discharge", merge.discharge_weight))
                         if w > 0.0)
        if self.loss.merge_timing.weight > 0.0 or self.loss.merge_survival.weight > 0.0:
            terms.append("merge_timing")
        if (self.loss.merge_survival.negative_weight > 0.0
                and self.model.merge_negotiation):
            terms.append("merge_negotiation")
        if self.loss.lane_flow.weight > 0.0:
            terms.append("lane_flow")
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
