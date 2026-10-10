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


@dataclass(frozen=True)
class ModelSpec:
    """Every option that determines the structure or bounding of the WM-S model.

    ``black_box`` is the controller-id-conditioned variant: CAVs become learned
    movers.  ``rollout_mode`` selects how the driver identity enters (``residual``:
    none; ``driver_params``: inferred IDM/MOBIL parameters without residual;
    ``hybrid``: inferred parameters plus gated residual conditioned on the driver
    embedding).  ``history_s`` (seconds, optional) enables the long driver history.

    The residual bounds are mutually exclusive: either the legacy tanh caps
    (``residual_cap_*``) or the soft bound (``residual_soft_bound_*``).
    """

    black_box: bool = False
    rollout_mode: str = "residual"
    lane_head: str = "off"
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

    def __post_init__(self) -> None:
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
        if self.gate_kind not in ("cumsum", "saturating"):
            raise ValueError("gate_kind must be cumsum or saturating")
        if not 0.0 < self.gate_g_max_init < 1.0:
            raise ValueError("gate_g_max_init must be in (0, 1)")
        if not math.isfinite(self.gate_tau_steps_init) or self.gate_tau_steps_init <= 0.0:
            raise ValueError("gate_tau_steps_init must be finite and positive")
        if self.lane_head == "history" and self.rollout_mode == "residual":
            raise ValueError("history lane head requires driver_params or hybrid rollout")
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
