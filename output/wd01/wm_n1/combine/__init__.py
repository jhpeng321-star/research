"""WM-N1 combination layer (design section 5): tree OOF, CAL-only
stacking, conformal recalibration, the disagreement signal and the
planner-variant rule.

All modules are pure CPU / numpy + pandas + scipy; no SUMO, no training.
The only data roots this package ever touches are the ones a caller
passes in; HOLDOUT and chunk-5 TEST are sealed and must never be opened.
"""
from __future__ import annotations

__version__ = "0.1.0"

from . import (crps, disagreement, guards, oof_tree, select_variant,
               stacking, synthetic)

__all__ = ["crps", "disagreement", "guards", "oof_tree", "select_variant",
           "stacking", "synthetic"]
