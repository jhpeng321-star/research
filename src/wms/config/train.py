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
    init_from: Optional[str] = None
    allow_rejected_hard_cap: bool = False

    def __post_init__(self) -> None:
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
    traffic: TrafficLossConfig = field(default_factory=TrafficLossConfig)
    stability_weight: float = 0.0
    stability_margin: float = 1.0
    stability_probes: int = 1
    conservation_enabled: bool = False
    conservation_weight: float = 0.0
    loss_share_logging: bool = False

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

    def __post_init__(self) -> None:
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

    def __post_init__(self) -> None:
        require(self.val_every >= 1 and self.val_batches >= 1,
                "selection.val_every and val_batches must be positive")


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

    def __post_init__(self) -> None:
        require(self.schema_version == 2, f"unsupported schema_version {self.schema_version}")
        if self.data.history_s is not None:
            require(self.model.kind == "structured", "long history requires structured WM-S")
        if self.model.kind != "structured":
            require(not self.loss.braking.any_on and self.model.residual_cap_neg_mps2 is None,
                    "enabled braking terms require model.kind='structured'")

    # -- training-time (stricter than construction) ---------------------
    def validate_for_training(self) -> None:
        """Checks that gate a training run but must not block reading old receipts."""
        require(self.runtime.num_workers == 0,
                "runtime.num_workers must be 0 (lazy closures are not picklable)")
        require(not self.model.has_hard_cap or self.model.allow_rejected_hard_cap,
                "hard residual caps were REJECTED (change-log #3); "
                "set model.allow_rejected_hard_cap only to re-score old runs")
        if self.selection.select_by_long_horizon:
            require(self.data.windows.val is not None or self.data.windows.val_holdout_frac > 0.0,
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
