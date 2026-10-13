"""Long-horizon validation schedule and checkpoint-selection policy.

Selection compares validation ``mean_position`` only at the longest horizon reached
(>= 60 s, from a stage >= 60 s): a longer eligible horizon always wins, ties are
broken by the lowest mean position.  The final checkpoint policy stays
``last-optimizer-step``; the selected snapshot is written separately.

SCHED (2026-10-14, default off): ``validation_horizons(..., all_horizons=True)`` validates
every selection horizon up to the stage, and :class:`MultiHorizonSelection` scores a whole
validation event by the mean reference-normalised ``mean_position`` over its horizons, so
the final (longest) stage no longer wins by construction (SCHED_NOTES.md).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

#: Monitor keys averaged in the validation record next to the loss terms.
MONITORS = ("predicted_lane_change_rate", "raw_residual_abs_mean_mps2",
            "raw_residual_abs_p95_mps2")
MIN_SELECTION_HORIZON_S = 60.0


#: Horizons scored by the multi-horizon selection (and validated with ``val_all_horizons``).
SELECTION_HORIZONS_S = (10.0, 30.0, 60.0, 120.0)


def validation_horizons(stage_s: float, *, all_horizons: bool = False,
                        horizons: Sequence[float] = SELECTION_HORIZONS_S) -> tuple[float, ...]:
    """Horizons validated at the end of a stage: 60 s plus the stage length when >= 60 s.

    ``all_horizons`` (SCHED): every horizon of ``horizons`` up to the stage length plus the
    stage length itself, ascending (10 s stage -> 10; 30 s -> 10, 30; 120 s -> 10..120).
    """
    base = (tuple(dict.fromkeys((60.0, float(stage_s)))) if stage_s >= 60
            else (float(stage_s),))
    if not all_horizons:
        return base
    extra = [float(h) for h in horizons if float(h) <= float(stage_s)]
    return tuple(sorted(dict.fromkeys([*extra, *base])))


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


def _match(horizon: float, horizons: Sequence[float]) -> Optional[int]:
    for i, h in enumerate(horizons):
        if abs(float(horizon) - float(h)) < 1e-6:
            return i
    return None


@dataclass
class MultiHorizonSelection:
    """Best validation event by mean reference-normalised position over its horizons.

    score = mean_h(mean_position_h / ref_h) [+ w_agg * mean_h(mean_agg_reward_h / own_h)]
    over the selection horizons present (finite, valid targets) in the event.  ``ref_h`` is
    the configured reference (default: the C4 main WM's best validation value at h); a
    reference <= 0 normalises by this run's first finite validation at h (any stage).  The
    agg-reward term always uses the run's own first value.  Only events from stages
    ``>= eligible_stage_s`` compete; the lowest score wins (ties keep the earlier event).
    """

    horizons: tuple = SELECTION_HORIZONS_S
    references: tuple = (7.0356, 49.4889, 128.7179, 0.0)
    eligible_stage_s: float = 30.0
    agg_reward_weight: float = 0.0
    score: float = float("inf")
    horizon_s: float = 0.0
    best: Optional[dict] = None
    table: list = field(default_factory=list)
    own_position: dict = field(default_factory=dict)
    own_agg: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.horizons = tuple(float(h) for h in self.horizons)
        self.references = tuple(float(r) for r in self.references)
        if len(self.references) != len(self.horizons):
            raise ValueError("one selection reference per horizon")

    def _usable(self, row: dict) -> Optional[int]:
        i = _match(float(row["horizon_s"]), self.horizons)
        if i is None or row.get("valid_targets", 0) <= 0:
            return None
        return i if math.isfinite(float(row.get("mean_position", float("nan")))) else None

    def score_event(self, rows: Sequence[dict]) -> tuple:
        """Score of one validation event (updates the own-first normalisers)."""
        ratios: dict = {}
        agg: dict = {}
        for row in rows:
            i = self._usable(row)
            if i is None:
                continue
            h = self.horizons[i]
            pos = float(row["mean_position"])
            self.own_position.setdefault(h, pos)
            ref = self.references[i] if self.references[i] > 0 else self.own_position[h]
            if ref > 0 and math.isfinite(ref):
                ratios[f"{h:g}"] = pos / ref
            value = row.get("mean_agg_reward")
            if value is not None and math.isfinite(float(value)):
                self.own_agg.setdefault(h, float(value))
                if self.own_agg[h] > 0:
                    agg[f"{h:g}"] = float(value) / self.own_agg[h]
        if not ratios:
            return float("inf"), {"position_ratio": ratios, "agg_reward_ratio": agg}
        score = sum(ratios.values()) / len(ratios)
        if self.agg_reward_weight > 0 and agg:
            score += self.agg_reward_weight * sum(agg.values()) / len(agg)
        return score, {"position_ratio": ratios, "agg_reward_ratio": agg}

    def consider_event(self, rows: Sequence[dict], *, stage_s: float, step: int) -> bool:
        """Offer one validation event (all its horizons); True when it becomes the best."""
        score, parts = self.score_event(rows)
        eligible = bool(float(stage_s) >= self.eligible_stage_s and math.isfinite(score))
        entry = {"step": int(step), "stage_s": float(stage_s), "score": score,
                 "eligible": eligible, "selected": False,
                 "horizons_s": sorted(float(r["horizon_s"]) for r in rows), **parts}
        chosen = eligible and score < self.score
        if chosen:
            for old in self.table:
                old["selected"] = False
            entry["selected"] = True
            usable = [float(r["horizon_s"]) for r in rows if self._usable(r) is not None]
            self.score, self.horizon_s = score, max(usable)
            self.best = {**entry, "rows": [dict(r) for r in rows]}
        self.table.append(entry)
        return chosen

    def summary(self) -> dict:
        """Selection table + settings (written to the receipt and RUN_SUMMARY)."""
        return {"mode": "multi_horizon", "horizons_s": list(self.horizons),
                "references": list(self.references),
                "reference_note": "reference <= 0: run's own first validation at that horizon",
                "own_first_position": {f"{k:g}": v for k, v in self.own_position.items()},
                "eligible_stage_s": self.eligible_stage_s,
                "agg_reward_weight": self.agg_reward_weight, "best_score": self.score,
                "best_step": None if self.best is None else self.best["step"],
                "best_stage_s": None if self.best is None else self.best["stage_s"],
                "table": [dict(e) for e in self.table]}


def eligible_stage_s(stages_s: Sequence[float], min_stage_s: float) -> float:
    """Horizon of the first curriculum stage whose horizon is >= ``min_stage_s`` (else last)."""
    for s in stages_s:
        if float(s) >= float(min_stage_s):
            return float(s)
    return float(stages_s[-1])


class MixedHorizonSchedule:
    """Draws a short training horizon for a fraction of batches in long stages.

    Uses its own ``random.Random`` stream (the torch/python training RNG is untouched), so
    ``frac == 0`` changes nothing.  Only stages ``>= min_stage_s`` mix, and only horizons
    strictly shorter than the stage are drawn.
    """

    def __init__(self, frac: float, horizons_s: Sequence[float], min_stage_s: float,
                 seed: int) -> None:
        self.frac = float(frac)
        self.horizons_s = tuple(float(h) for h in horizons_s)
        self.min_stage_s = float(min_stage_s)
        self.rng = random.Random(int(seed) * 1_000_003 + 4242)

    def draw(self, stage_s: float) -> Optional[float]:
        """Short horizon (s) for the next batch, or None for the stage's own unroll."""
        if self.frac <= 0.0 or float(stage_s) < self.min_stage_s:
            return None
        options = [h for h in self.horizons_s if h < float(stage_s)]
        if not options or self.rng.random() >= self.frac:
            return None
        return self.rng.choice(options)
