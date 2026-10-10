"""P2 HDV residual network configuration (Idea A step (b)).

The residual net consumes NEIGHBOUR KINEMATICS ONLY.  The main model must not
receive a CAV flag; ``use_cav_flag`` is an ablation switch that appends one
extra input column and is OFF in the headline model (amendment A3.1).
"""
from __future__ import annotations

from dataclasses import dataclass

# Kinematics-only input fields, in order.  Index into the P9 ``veh_kin``
# channel; CAV flags and action one-hots are deliberately absent.
KIN_FIELDS = (
    "s_down_m", "lane_main_left", "lane_main_right", "lane_accel",
    "v_mps", "a_mps2", "length_m")
KIN_F = len(KIN_FIELDS)

# The ablation-only extra column.
CAV_FLAG_FIELD = "is_cav"


@dataclass
class HdvConfig:
    """Residual attention net shape + output scales."""

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    ffn: int = 512
    radius_m: float = 150.0
    dropout: float = 0.0
    use_cav_flag: bool = False          # ablation only; not the headline model
    accel_delta_scale_mps2: float = 1.5
    lane_logit_scale: float = 1.0
    # Pre-flight gate G1: the residual is identically zero, so the baseline
    # IDM/MOBIL output is reproduced exactly (no-op control).
    zero_residual: bool = False
    # Hybrid rollout: the per-driver embedding inferred from pre-t0 history is
    # added to the embedded kinematics as an extra conditioning input.  ``0``
    # (default) leaves the residual net byte-identical to the WM-S baseline: no
    # conditioning module is built and ``forward`` ignores ``conditioning``.
    conditioning_dim: int = 0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if int(self.conditioning_dim) < 0:
            raise ValueError("conditioning_dim must be non-negative")

    @property
    def in_features(self) -> int:
        return KIN_F + (1 if self.use_cav_flag else 0)
