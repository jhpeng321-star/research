"""Long-horizon validation schedule and checkpoint-selection policy.

Selection compares validation ``mean_position`` only at the longest horizon reached
(>= 60 s, from a stage >= 60 s): a longer eligible horizon always wins, ties are
broken by the lowest mean position.  The final checkpoint policy stays
``last-optimizer-step``; the selected snapshot is written separately.

SCHED (2026-10-14, default off): ``validation_horizons(..., all_horizons=True)`` validates
every selection horizon up to the stage, and :class:`MultiHorizonSelection` scores a whole
validation event by the mean reference-normalised ``mean_position`` over its horizons, so
the final (longest) stage no longer wins by construction (SCHED_NOTES.md).

INFRAV5 (2026-10-14, default off): :class:`SameHorizonSelection` (``multi_horizon_v2``)
compares candidates only on one common horizon set and never normalises by the run's own
first validation (INFRAV5_NOTES.md).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

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


@dataclass
class SameHorizonSelection:
    """INFRAV5 ``multi_horizon_v2``: candidates are only ever compared on the SAME horizons.

    v4's ``multi_horizon`` compared a 60 s-stage event (scored on 10/30/60 s) with the final
    120 s-stage event (scored also on 120 s, normalised by the run's own first 120 s
    validation = that very event, ratio 1.0) and picked the 60 s-stage checkpoint, although the
    final one was far better on unseen walls (RUN_SUMMARY of c4_wm_v4_cont).  Here:

    * ONLINE (every validation event): score on the common horizons -- selection horizons
      <= the first eligible stage, validated by every eligible event -- with the fixed
      references only (a reference <= 0 is skipped online).  This tracks the best event of each
      eligible stage (a candidate) and a provisional best (written as ``wm_s_best.pt`` while the
      run is going, so a run cut short still has one).
    * FINAL (end of the run): the candidates (best and last event of every eligible stage) are
      completed on the full horizon set F (the horizons of the last eligible event) by
      re-validating the missing horizons on the same fixed validation subset, then scored as
      mean_h(position_h / norm_h) over F, norm_h = fixed reference when > 0, else the best
      candidate's position at h (never the run's own first validation).  Lowest score wins;
      ties keep the earlier candidate.  The full comparison table goes to the receipt and
      RUN_SUMMARY.
    """

    horizons: tuple = SELECTION_HORIZONS_S
    references: tuple = (7.0356, 49.4889, 128.7179, 0.0)
    eligible_stage_s: float = 30.0
    score: float = float("inf")
    horizon_s: float = 0.0
    best: Optional[dict] = None
    table: list = field(default_factory=list)
    stage_best: dict = field(default_factory=dict)
    stage_last: dict = field(default_factory=dict)
    final: Optional[dict] = None

    def __post_init__(self) -> None:
        self.horizons = tuple(float(h) for h in self.horizons)
        self.references = tuple(float(r) for r in self.references)
        if len(self.references) != len(self.horizons):
            raise ValueError("one selection reference per horizon")

    @property
    def common_horizons(self) -> tuple:
        """Horizons every eligible event validates (selection horizons <= first eligible stage)."""
        return tuple(h for h in self.horizons if h <= self.eligible_stage_s + 1e-9)

    def _reference(self, horizon: float) -> float:
        return self.references[self.horizons.index(horizon)]

    def _positions(self, rows: Sequence[dict]) -> dict:
        out: dict = {}
        for row in rows:
            i = _match(float(row["horizon_s"]), self.horizons)
            if i is None or row.get("valid_targets", 0) <= 0:
                continue
            pos = float(row.get("mean_position", float("nan")))
            if math.isfinite(pos):
                out[self.horizons[i]] = pos
        return out

    def online_score(self, rows: Sequence[dict]) -> tuple:
        """Mean ``position / reference`` over the common horizons with a fixed reference.

        Without any fixed reference among the common horizons, the raw mean position over
        the common horizons is used (every eligible event validates them, so it is still a
        same-horizon comparison)."""
        pos = self._positions(rows)
        needed = [h for h in self.common_horizons if self._reference(h) > 0]
        if not needed:
            raw = {f"{h:g}": pos[h] for h in self.common_horizons if h in pos}
            if not self.common_horizons or len(raw) < len(self.common_horizons):
                return float("inf"), raw
            return sum(raw.values()) / len(raw), raw
        ratios = {f"{h:g}": pos[h] / self._reference(h) for h in needed if h in pos}
        if len(ratios) < len(needed):
            return float("inf"), ratios
        return sum(ratios.values()) / len(ratios), ratios

    def consider_event(self, rows: Sequence[dict], *, stage_s: float, step: int,
                       stage_end: bool = False) -> dict:
        """Offer one validation event; returns ``{"global_best", "stage_best", "stage_last"}``."""
        score, ratios = self.online_score(rows)
        eligible = bool(float(stage_s) >= self.eligible_stage_s and math.isfinite(score))
        entry = {"step": int(step), "stage_s": float(stage_s), "score": score,
                 "eligible": eligible, "selected": False, "stage_end": bool(stage_end),
                 "online_horizons_s": [float(h) for h in self.common_horizons],
                 "horizons_s": sorted(float(r["horizon_s"]) for r in rows),
                 "position_ratio": ratios,
                 "positions": {f"{h:g}": v for h, v in self._positions(rows).items()}}
        flags = {"global_best": False, "stage_best": False, "stage_last": False}
        if eligible:
            key = float(stage_s)
            event = {**entry, "rows": [dict(r) for r in rows]}
            if key not in self.stage_best or score < self.stage_best[key]["score"]:
                self.stage_best[key] = event
                flags["stage_best"] = True
            if stage_end:
                self.stage_last[key] = event
                flags["stage_last"] = True
            if score < self.score:
                for old in self.table:
                    old["selected"] = False
                entry["selected"] = True
                event["selected"] = True
                self.score, self.horizon_s = score, max(self.common_horizons)
                self.best = event
                flags["global_best"] = True
        self.table.append(entry)
        return flags

    def candidates(self) -> list:
        """Best and last event of each eligible stage, deduplicated by step, ascending."""
        found: dict = {}
        for kind, events in (("stage_best", self.stage_best), ("stage_last", self.stage_last)):
            for stage_s, event in events.items():
                cand = found.setdefault(event["step"], {"step": event["step"],
                                                        "stage_s": stage_s, "kinds": [],
                                                        "rows": list(event["rows"])})
                cand["kinds"].append(kind)
        return [found[k] for k in sorted(found)]

    def full_horizons(self) -> tuple:
        """Horizon set F of the final comparison: the horizons of the last eligible event."""
        events = list(self.stage_last.values()) + list(self.stage_best.values())
        if not events:
            return ()
        last = max(events, key=lambda e: e["step"])
        return tuple(sorted(self._positions(last["rows"])))

    def finalize(self, revalidate: Callable) -> Optional[dict]:
        """Final same-horizon comparison; ``revalidate(step, horizons) -> rows``."""
        cands = self.candidates()
        full = self.full_horizons()
        if not cands or not full:
            self.final = None
            return None
        for cand in cands:
            have = self._positions(cand["rows"])
            missing = [h for h in full if h not in have]
            cand["revalidated_s"] = missing
            if missing:
                cand["rows"] = list(cand["rows"]) + list(revalidate(cand["step"], missing))
            cand["positions"] = {h: v for h, v in self._positions(cand["rows"]).items()
                                 if h in full}
        norms = {}
        for h in full:
            values = [c["positions"][h] for c in cands if h in c["positions"]]
            norms[h] = (self._reference(h) if self._reference(h) > 0
                        else (min(values) if values else float("nan")))
        best = None
        for cand in cands:
            ratios = {f"{h:g}": cand["positions"][h] / norms[h] for h in full
                      if h in cand["positions"] and norms[h] > 0}
            cand["complete"] = len(ratios) == len(full)
            cand["position_ratio"] = ratios
            cand["score"] = (sum(ratios.values()) / len(ratios) if cand["complete"]
                             else float("inf"))
            if math.isfinite(cand["score"]) and (best is None or cand["score"] < best["score"]):
                best = cand
        table = [{"step": c["step"], "stage_s": c["stage_s"], "kinds": c["kinds"],
                  "positions": {f"{h:g}": v for h, v in c["positions"].items()},
                  "position_ratio": c["position_ratio"], "score": c["score"],
                  "complete": c["complete"], "revalidated_s": c["revalidated_s"],
                  "selected": best is not None and c["step"] == best["step"]} for c in cands]
        self.final = {"horizons_s": list(full),
                      "normalisers": {f"{h:g}": v for h, v in norms.items()},
                      "normaliser_note": "fixed reference when > 0, else the best candidate",
                      "table": table,
                      "best_step": None if best is None else best["step"],
                      "best_stage_s": None if best is None else best["stage_s"],
                      "best_score": None if best is None else best["score"]}
        if best is not None:
            self.score, self.horizon_s = best["score"], max(full)
            self.best = {"step": best["step"], "stage_s": best["stage_s"],
                         "score": best["score"], "selected": True, "final_comparison": True,
                         "rows": [dict(r) for r in best["rows"]]}
        return self.final

    def summary(self) -> dict:
        """Online table + final comparison (receipt and RUN_SUMMARY)."""
        return {"mode": "multi_horizon_v2", "horizons_s": list(self.horizons),
                "references": list(self.references),
                "reference_note": ("reference <= 0: skipped online; best candidate of the "
                                   "final same-horizon comparison"),
                "online_horizons_s": list(self.common_horizons),
                "eligible_stage_s": self.eligible_stage_s, "best_score": self.score,
                "best_step": None if self.best is None else self.best["step"],
                "best_stage_s": None if self.best is None else self.best["stage_s"],
                "table": [dict(e) for e in self.table], "final_comparison": self.final}


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
