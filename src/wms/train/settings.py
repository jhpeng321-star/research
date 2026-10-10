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
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    position_huber_delta_m: float = 5.0
    speed_huber_delta_mps: float = 1.0
    lane_weight: float = 1.0
    position_weight: float = 1.0
    speed_weight: float = 1.0
    step_weight_decay: float = 0.01
    truncation_steps: int = 20
    history_features: int = 6
    history_s: Optional[float] = None
    history_stride_s: float = 1.0
    device: str = "cpu"
    threads: int = 6
    state_noise_std_m: float = 0.0
    state_noise_std_mps: float = 0.0
    rollout_mode: str = "residual"
    lane_head: str = "off"
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

    def __post_init__(self) -> None:
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
        _require(self.val_every >= 1, "val_every must be positive")

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
    def long_horizon_on(self) -> bool:
        """Any of the opt-in long-horizon guards (soft bound / penalties / selection)."""
        return bool(self.residual_soft_bound_mps2 or self.residual_mag_weight
                    or self.lane_rate_weight or self.select_by_long_horizon)
