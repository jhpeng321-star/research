"""Candidate vs previous best - the only comparison the evaluation produces.

Design rule: a new checkpoint is judged against exactly one reference, the
previous best, on the unseen-controller sets (``wall_same``, ``wall_pilot``) and the seen
controls, at 60 s and 120 s, together with the blow-up rate.  There is deliberately no
N-way leaderboard: :func:`compare` takes two :class:`ModelResult` values.

Verdict rule (per set and horizon, paired by state, whole worlds resampled; negative
difference favours the candidate):

* ``better`` if the 95% CI of the median paired ``|ds|`` difference lies below 0,
  ``worse`` if it lies above 0, otherwise ``inconclusive``;
* overall ``improved`` iff no cell is ``worse``, no blow-up rate increased, and at least
  one cell is ``better``; ``regressed`` iff any cell is ``worse`` or any blow-up rate
  increased; otherwise ``inconclusive``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .errors import ComparisonError
from .ranking import RANK_HORIZONS_S, ranking
from .report import (blowup_rate, fmt, fmt_ci, horizon_summary, normalize_seen, paired_bootstrap,
                     tested_arm, utc_now)
from .stats import DEFAULT_N_BOOT, DEFAULT_SEED

COMPARE_SCHEMA = "wms_eval_comparison/1"
DEFAULT_SETS = ("wall_same", "wall_pilot", "seen_controls")
DEFAULT_HORIZONS_S = (60.0, 120.0)
RANKED_SETS = ("wall_same", "wall_pilot")


@dataclass(frozen=True)
class ModelResult:
    """Scored states of one model: ``{set_name: [state result, ...]}`` (+ optional extras)."""

    name: str
    states_by_set: Mapping[str, Sequence[dict]]
    checkpoint_sha256: str | None = None
    #: ``{set_name: sliced_fingerprint(...)}`` from :mod:`wms.eval.fingerprints`
    fingerprints: Mapping[str, Mapping] | None = None

    def states(self, set_name: str) -> list[dict]:
        raw = self.states_by_set.get(set_name)
        if raw is None:
            raise ComparisonError(f"model {self.name!r} has no scored set {set_name!r}")
        return normalize_seen(raw) if set_name.startswith("seen") else list(raw)


def classify(block: Mapping) -> str:
    """``better`` / ``worse`` / ``inconclusive`` from a paired-bootstrap CI."""
    lo, hi = block.get("lo"), block.get("hi")
    if lo is None or hi is None:
        return "inconclusive"
    if hi < 0:
        return "better"
    if lo > 0:
        return "worse"
    return "inconclusive"


def compare(new: ModelResult, previous_best: ModelResult,
            sets: Sequence[str] = DEFAULT_SETS, horizons: Sequence[float] = DEFAULT_HORIZONS_S,
            *, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED) -> dict:
    """Compare ``new`` with ``previous_best`` on ``sets`` at ``horizons`` (seconds)."""
    if new.name == previous_best.name:
        raise ComparisonError("candidate and previous best must be different models")
    cells: list[dict] = []
    per_set: dict[str, dict] = {}
    for set_name in sets:
        a_states, b_states = new.states(set_name), previous_best.states(set_name)
        arms_a = sorted({a for s in a_states for a in (s.get("arms") or {})})
        arms_b = sorted({a for s in b_states for a in (s.get("arms") or {})})
        arm = tested_arm(set_name, arms_a)
        if arm is None or arm != tested_arm(set_name, arms_b):
            raise ComparisonError(f"set {set_name!r}: models do not share a tested arm")
        block: dict = {"tested_arm": arm, "horizons": {}}
        for horizon in horizons:
            key = f"{horizon:g}"
            paired = paired_bootstrap(a_states, b_states, arm, horizon,
                                      n_boot=n_boot, seed=seed)
            outcome = classify(paired)
            block["horizons"][key] = {
                "new": horizon_summary(a_states, arm, horizon),
                "previous_best": horizon_summary(b_states, arm, horizon),
                "paired_new_minus_previous": paired,
                "outcome": outcome,
            }
            cells.append({"set": set_name, "horizon_s": float(horizon), "outcome": outcome})
        block["blowup"] = {"new": blowup_rate(a_states, arm),
                           "previous_best": blowup_rate(b_states, arm)}
        if set_name in RANKED_SETS:
            block["ranking"] = {"new": ranking(a_states, horizons=RANK_HORIZONS_S, seed=seed),
                                "previous_best": ranking(b_states, horizons=RANK_HORIZONS_S,
                                                         seed=seed)}
        per_set[set_name] = block
    blowup_up = [s for s, b in per_set.items()
                 if (b["blowup"]["new"]["rate"] or 0.0) > (b["blowup"]["previous_best"]["rate"] or 0.0)]
    worse = [c for c in cells if c["outcome"] == "worse"]
    better = [c for c in cells if c["outcome"] == "better"]
    if worse or blowup_up:
        verdict = "regressed"
    elif better:
        verdict = "improved"
    else:
        verdict = "inconclusive"
    return {
        "schema": COMPARE_SCHEMA, "generated_utc": utc_now(),
        "rule": "candidate vs previous best only; negative difference favours the candidate",
        "new": {"name": new.name, "checkpoint_sha256": new.checkpoint_sha256},
        "previous_best": {"name": previous_best.name,
                          "checkpoint_sha256": previous_best.checkpoint_sha256},
        "horizons_s": [float(h) for h in horizons], "n_boot": n_boot, "seed": seed,
        "sets": per_set, "cells": cells,
        "blowup_increased_on": blowup_up, "verdict": verdict,
        "fingerprints": {new.name: new.fingerprints, previous_best.name: previous_best.fingerprints},
    }


def markdown(comparison: Mapping) -> str:
    new, prev = comparison["new"]["name"], comparison["previous_best"]["name"]
    lines = [f"# Comparison: `{new}` vs previous best `{prev}`", "",
             f"Verdict: **{comparison['verdict']}**.  Rule: {comparison['rule']}.", "",
             "## HDV |ds| (m), per-state median across states", "",
             f"| set | arm | h | {new} median [IQR] | {prev} median [IQR] | "
             "paired diff [95% CI over worlds] | worlds | outcome |",
             "|---|---|---|---|---|---|---:|---|"]
    for set_name, block in comparison["sets"].items():
        for horizon, cell in block["horizons"].items():
            a, b, paired = cell["new"], cell["previous_best"], cell["paired_new_minus_previous"]
            lines.append(
                f"| {set_name} | {block['tested_arm']} | {horizon}s | "
                f"{fmt(a['median'])} [{fmt(a['iqr'])}] | {fmt(b['median'])} [{fmt(b['iqr'])}] | "
                f"{fmt_ci(paired)} | {paired['n_worlds']} | {cell['outcome']} |")
    lines += ["", "## Blow-up rate", "", f"| set | {new} | {prev} |", "|---|---:|---:|"]
    for set_name, block in comparison["sets"].items():
        a, b = block["blowup"]["new"], block["blowup"]["previous_best"]
        lines.append(f"| {set_name} | {fmt(a['rate'], 3)} ({a['n_blowup']}/{a['n_eval']}) | "
                     f"{fmt(b['rate'], 3)} ({b['n_blowup']}/{b['n_eval']}) |")
    ranked = [(s, b["ranking"]) for s, b in comparison["sets"].items() if "ranking" in b]
    if ranked:
        lines += ["", "## Wall-vs-native decision regret (s; lower is better)", "",
                  f"| set | h | n | {new} regret | {prev} regret | always-wall | always-native |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for set_name, rk in ranked:
            for horizon in RANK_HORIZONS_S:
                key = str(horizon)
                if key not in rk["new"] or key not in rk["previous_best"]:
                    continue
                a, b = rk["new"][key], rk["previous_best"][key]
                lines.append(f"| {set_name} | {horizon:g}s | {a['n']} | "
                             f"{fmt(a['model_mean_regret_s'])} | {fmt(b['model_mean_regret_s'])} | "
                             f"{fmt(a['always_wall_regret_s'])} | "
                             f"{fmt(a['always_native_regret_s'])} |")
    lines.append("")
    return "\n".join(lines)


def write_comparison(comparison: Mapping, out_dir: Path) -> tuple[Path, Path]:
    """Write ``COMPARISON.md`` and ``COMPARISON.json`` into a new or empty ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path, json_path = out_dir / "COMPARISON.md", out_dir / "COMPARISON.json"
    if md_path.exists() or json_path.exists():
        raise ComparisonError(f"refusing to overwrite an existing comparison in {out_dir}")
    json_path.write_text(json.dumps(comparison, indent=1), encoding="utf-8")
    md_path.write_text(markdown(comparison), encoding="utf-8")
    return md_path, json_path
