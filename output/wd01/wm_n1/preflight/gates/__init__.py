"""G1..G6 gate modules."""
from __future__ import annotations

from . import g1_zero_residual, g2_ctrl_parity, g3_label_leak
from . import g4_overfit, g5_e2e_smoke, g6_heldout_dryrun

ALL = (g1_zero_residual, g2_ctrl_parity, g3_label_leak, g4_overfit,
       g5_e2e_smoke, g6_heldout_dryrun)

__all__ = ["ALL", "g1_zero_residual", "g2_ctrl_parity", "g3_label_leak",
           "g4_overfit", "g5_e2e_smoke", "g6_heldout_dryrun"]
