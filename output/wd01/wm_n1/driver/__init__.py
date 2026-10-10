"""WM-S T3: RSU-observable per-driver inputs (desired speed + motion history).

Two independent, default-off ingredients for the WM-S physics base + learned
correction (docs/WM_STATE_PLAN.md, P1/P2):

* :mod:`driver.desired_speed` -- estimate each vehicle's desired speed ``v0_i``
  (and optionally its time headway ``T_i``) from observed recent motion before
  ``t0``, shrunk to a population/vType prior when data is scarce.
* :mod:`driver.history` -- fixed-length per-vehicle motion-history tensors
  ``[K, H, F]`` (``H`` = last 5 s at 0.5 s) built from the same pack data the
  residual uses and from a rollout rolling buffer at inference.

Both modules consume only RSU-observable quantities (observed speeds / gaps /
lane occupancy).  No driver type, hidden vType, penetration or global metric is
an input.
"""
from __future__ import annotations

from .desired_speed import (DT_S, FREE_GAP_M, POP_SPEED_FACTOR,
                            DesiredSpeedEstimate, estimate_desired_speed)
from .history import (HISTORY_F, HISTORY_FIELDS, HISTORY_H, RollingHistoryBuffer,
                      build_history, pack_history_tensor)
from .rollout_flag import apply_desired_speed_v0

__all__ = [
    "DT_S", "FREE_GAP_M", "POP_SPEED_FACTOR", "DesiredSpeedEstimate",
    "estimate_desired_speed", "HISTORY_F", "HISTORY_FIELDS", "HISTORY_H",
    "RollingHistoryBuffer", "build_history", "pack_history_tensor",
    "apply_desired_speed_v0",
]
