"""Architecture and bounding options of :class:`~wms.model.wms.WmSModel`."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .controllers import N_CONTROLLERS
from .history import HISTORY_F, HISTORY_H
from .residual_bound import validate_bounds

ROLLOUT_MODES = ("residual", "driver_params", "hybrid")
LANE_HEADS = ("off", "history")
LANE_RULES = ("mobil", "forced_merge_only", "learned_discretionary")
#: Lane-decision model (WMLANE A): the gated residual lane delta (default) or the
#: dedicated per-HDV lane-change hazard model (``wms.model.lane_hazard``).
LANE_MODELS = ("residual", "hazard", "hazard_v5")
#: In-rollout driver-setting update from observed motion (WMLANE C).
DRIVER_UPDATES = ("off", "ema")
#: LANEFIX (hazard mode only): rollout lane decision, hazard feature set, merge ownership.
LANE_DECISIONS = ("argmax", "sample", "gated")
LANE_HAZARD_FEATURES = ("v1", "v2")
MERGE_HAZARD_ZONES = ("forced_late", "full")


@dataclass(frozen=True)
class ModelSpec:
    """Every option that determines the structure or bounding of the WM-S model.

    ``black_box`` is the controller-id-conditioned variant: CAVs become learned
    movers.  ``rollout_mode`` selects how the driver identity enters (``residual``:
    none; ``driver_params``: inferred IDM/MOBIL parameters without residual;
    ``hybrid``: inferred parameters plus gated residual conditioned on the driver
    embedding).  ``history_s`` (seconds, optional) enables the long driver history.
    ``lane_rule`` records the C4 lane-change policy of the training config; the
    runtime switch itself lives on :class:`~wms.physics.config.PhysicsConfig`
    (like ``merge_cal`` / ``sumo_fidelity``), which is what the model consults.

    The residual bounds are mutually exclusive: either the legacy tanh caps
    (``residual_cap_*``) or the soft bound (``residual_soft_bound_*``).
    """

    black_box: bool = False
    rollout_mode: str = "residual"
    lane_head: str = "off"
    lane_rule: str = "mobil"
    history_features: int = HISTORY_F
    history_len: int = HISTORY_H
    history_hidden: int = 64
    k_hidden: int = 32
    n_controllers: int = N_CONTROLLERS
    controller_embed_dim: int = 16
    checkpoint_residual: bool = False
    residual_cap_mps2: Optional[float] = None
    residual_cap_neg_mps2: Optional[float] = None
    residual_soft_bound_mps2: Optional[float] = None
    residual_soft_bound_neg_mps2: Optional[float] = None
    lane_logit_cap: Optional[float] = None
    history_s: Optional[float] = None
    history_stride_s: float = 1.0
    gate_kind: str = "cumsum"
    gate_g_max_init: float = 0.30
    gate_tau_steps_init: float = 50.0
    lane_gate_kind: str = "shared"
    lane_gate_g_max_init: float = 0.16
    lane_gate_tau_steps_init: float = 30.0
    checkpoint_step: bool = False
    grad_decay_per_step: float = 1.0
    # WMLANE (default OFF; created modules are appended after every existing one so
    # older checkpoints keep their key order and load unchanged).
    lane_model: str = "residual"
    lane_hazard_hidden: int = 64
    lane_hazard_cap: float = 10.0
    driver_update: str = "off"
    driver_update_rate: float = 0.3
    # LANEFIX (default OFF; hazard mode only; the v2 features widen the hazard input layer).
    lane_decision: str = "argmax"
    lane_logit_offsets: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lane_hazard_features: str = "v1"
    merge_hazard_zone: str = "forced_late"
    # LANEV5 (lane_model='hazard_v5' only): gated-decision gates and configured head APs.
    lane_v5_gates: tuple[float, float, float] = (0.25, 0.10, 0.5)
    lane_v5_head_ap: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    # V6: default values preserve the legacy encoder and algebraic EMA updates.
    driver_param_mode: str = "legacy"
    driver_ema_update: bool = True
    # V6INTEG: opt-in resolution of simultaneous lane-change conflicts.
    lane_conflict_resolve: bool = False

    def __post_init__(self) -> None:
        if self.driver_param_mode not in ("legacy", "physical"):
            raise ValueError("driver_param_mode must be legacy or physical")
        if not isinstance(self.driver_ema_update, bool):
            raise ValueError("driver_ema_update must be a boolean")
        if self.residual_cap_mps2 is not None and float(self.residual_cap_mps2) <= 0.0:
            raise ValueError("residual_cap_mps2 must be > 0 when set")
        if self.lane_logit_cap is not None and float(self.lane_logit_cap) <= 0.0:
            raise ValueError("lane_logit_cap must be > 0 when set")
        if self.residual_cap_neg_mps2 is not None and (
                not math.isfinite(float(self.residual_cap_neg_mps2))
                or float(self.residual_cap_neg_mps2) <= 0.0):
            raise ValueError("residual_cap_neg_mps2 must be finite and > 0 when set")
        validate_bounds(self.residual_soft_bound_mps2, self.residual_soft_bound_neg_mps2,
                        self.residual_cap_mps2, self.residual_cap_neg_mps2)
        if self.rollout_mode not in ROLLOUT_MODES:
            raise ValueError("rollout_mode must be residual, driver_params or hybrid")
        if self.lane_head not in LANE_HEADS:
            raise ValueError("lane_head must be off or history")
        if self.lane_rule not in LANE_RULES:
            raise ValueError(f"lane_rule must be one of {LANE_RULES}")
        if self.gate_kind not in ("cumsum", "saturating"):
            raise ValueError("gate_kind must be cumsum or saturating")
        if not 0.0 < self.gate_g_max_init < 1.0:
            raise ValueError("gate_g_max_init must be in (0, 1)")
        if not math.isfinite(self.gate_tau_steps_init) or self.gate_tau_steps_init <= 0.0:
            raise ValueError("gate_tau_steps_init must be finite and positive")
        if self.lane_gate_kind not in ("shared", "saturating"):
            raise ValueError("lane_gate_kind must be shared or saturating")
        if not 0.0 < self.lane_gate_g_max_init < 1.0:
            raise ValueError("lane_gate_g_max_init must be in (0, 1)")
        if not math.isfinite(self.lane_gate_tau_steps_init) or self.lane_gate_tau_steps_init <= 0.0:
            raise ValueError("lane_gate_tau_steps_init must be finite and positive")
        if self.lane_head == "history" and self.rollout_mode == "residual":
            raise ValueError("history lane head requires driver_params or hybrid rollout")
        if self.lane_model not in LANE_MODELS:
            raise ValueError(f"lane_model must be one of {LANE_MODELS}")
        if self.driver_update not in DRIVER_UPDATES:
            raise ValueError(f"driver_update must be one of {DRIVER_UPDATES}")
        if (self.lane_model in ("hazard", "hazard_v5") or self.driver_update != "off")                 and (self.rollout_mode == "residual" or self.black_box):
            raise ValueError("lane_model='hazard' / driver_update need the structured "
                             "driver_params or hybrid rollout (driver embedding/settings)")
        if self.lane_model in ("hazard", "hazard_v5") and self.lane_head != "off":
            raise ValueError("lane_model='hazard' and the history lane head are exclusive")
        if self.lane_hazard_hidden < 1 or not (math.isfinite(self.lane_hazard_cap)
                                               and self.lane_hazard_cap > 0.0):
            raise ValueError("lane_hazard_hidden must be >= 1 and lane_hazard_cap > 0")
        if not 0.0 < self.driver_update_rate < 1.0:
            raise ValueError("driver_update_rate must be in (0, 1)")
        offsets = tuple(float(x) for x in self.lane_logit_offsets)
        object.__setattr__(self, "lane_logit_offsets", offsets)
        if len(offsets) != 3 or not all(math.isfinite(x) for x in offsets):
            raise ValueError("lane_logit_offsets must be three finite (overtake, return, merge)")
        if self.lane_decision not in LANE_DECISIONS:
            raise ValueError(f"lane_decision must be one of {LANE_DECISIONS}")
        if self.lane_hazard_features not in LANE_HAZARD_FEATURES:
            raise ValueError(f"lane_hazard_features must be one of {LANE_HAZARD_FEATURES}")
        if self.merge_hazard_zone not in MERGE_HAZARD_ZONES:
            raise ValueError(f"merge_hazard_zone must be one of {MERGE_HAZARD_ZONES}")
        gates = tuple(float(x) for x in self.lane_v5_gates)
        head_ap = tuple(float(x) for x in self.lane_v5_head_ap)
        object.__setattr__(self, "lane_v5_gates", gates)
        object.__setattr__(self, "lane_v5_head_ap", head_ap)
        if len(gates) != 3 or len(head_ap) != 3 \
                or not all(math.isfinite(x) for x in (*gates, *head_ap)):
            raise ValueError("lane_v5_gates / lane_v5_head_ap must be three finite values")
        if self.lane_model != "hazard_v5" and (
                self.lane_decision == "gated" or gates != (0.25, 0.10, 0.5)
                or head_ap != (-1.0, -1.0, -1.0)):
            raise ValueError("lane_decision='gated' / lane_v5_* need lane_model='hazard_v5'")
        if self.lane_model not in ("hazard", "hazard_v5") and (
                self.lane_decision != "argmax" or any(offsets)
                or self.lane_hazard_features != "v1" or self.merge_hazard_zone != "forced_late"):
            raise ValueError("lane_decision / lane_logit_offsets / lane_hazard_features / "
                             "merge_hazard_zone need lane_model='hazard'")
        if self.history_features < 0:
            raise ValueError("history_features must be >= 0")
        if self.long_history and (self.black_box or self.history_features != HISTORY_F):
            raise ValueError("long history requires structured WM-S with six rolling features")
        if self.black_box and self.n_controllers < 1:
            raise ValueError("black_box needs at least one controller id")
        if self.black_box and self.controller_embed_dim < 1:
            raise ValueError("controller_embed_dim must be >= 1 in black_box")

    @property
    def long_history(self) -> bool:
        return self.history_s is not None

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ModelSpec":
        """Spec from a flat legacy training config (``checkpoint["config"]``).

        Keys absent from ``config`` keep their default, which reproduces the
        checkpoints written before the corresponding option existed.
        """
        return cls(**{key: config[key] for key in cls.__dataclass_fields__
                      if config.get(key) is not None})
