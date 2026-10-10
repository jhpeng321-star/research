"""Long-horizon validation schedule and checkpoint-selection policy.

Selection compares validation ``mean_position`` only at the longest horizon reached
(>= 60 s, from a stage >= 60 s): a longer eligible horizon always wins, ties are
broken by the lowest mean position.  The final checkpoint policy stays
``last-optimizer-step``; the selected snapshot is written separately.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

#: Monitor keys averaged in the validation record next to the loss terms.
MONITORS = ("predicted_lane_change_rate", "raw_residual_abs_mean_mps2",
            "raw_residual_abs_p95_mps2")
MIN_SELECTION_HORIZON_S = 60.0


def validation_horizons(stage_s: float) -> tuple[float, ...]:
    """Horizons validated at the end of a stage: 60 s plus the stage length when >= 60 s."""
    return tuple(dict.fromkeys((60.0, float(stage_s)))) if stage_s >= 60 else (float(stage_s),)


@dataclass
class LongHorizonSelection:
    """Scores are comparable only at the longest eligible horizon reached."""

    horizon_s: float = 0.0
    score: float = float("inf")
    best: Optional[dict] = None

    def consider(self, record: dict) -> bool:
        """Offer a validation record; True when it becomes the selected checkpoint."""
        horizon = float(record["horizon_s"])
        score = float(record["mean_position"])
        stage = float(record["stage_s"])
        if (not math.isfinite(stage) or not math.isfinite(horizon)
                or stage < MIN_SELECTION_HORIZON_S or horizon < MIN_SELECTION_HORIZON_S
                or not math.isfinite(score) or record.get("valid_targets", 0) <= 0):
            return False
        if horizon > self.horizon_s or (horizon == self.horizon_s and score < self.score):
            self.horizon_s, self.score, self.best = horizon, score, dict(record)
            return True
        return False
