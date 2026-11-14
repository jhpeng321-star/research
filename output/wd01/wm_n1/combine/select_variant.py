"""Planner-variant choice on CAL (WM-N1 design section 5; amendment 3 A3.3).

Among the four variants {T, A, C, N} pick the one with the lowest CAL mean
CRPS of the policy contrasts dV, over the gating contrasts and tau in
{60, 120, 300}, subject to the overall central 10-90 % coverage lying in
[0.75, 0.85].  A tie within 2 % goes to the simpler model: T < A < C < N
(amendment 3 A3.3; A is the controller-in-the-loop headline model).
The choice is written to PLANNER_VARIANT.json with a self-describing
sha256, so it can be frozen before TEST is opened.

The test/holdout split never enters this module: it consumes CAL scores
and coverage only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

VARIANTS = ("T", "A", "C", "N")
COMPLEXITY_RANK = {"T": 0, "A": 1, "C": 2, "N": 3}
DEFAULT_TAUS = (60.0, 120.0, 300.0)
DEFAULT_CONTRASTS = ("damping-native", "pi-native", "wall-native")
DEFAULT_COV_BOUNDS = (0.75, 0.85)
DEFAULT_TIE_REL = 0.02


def _tau_key(crps_variant, tau):
    if tau in crps_variant:
        return tau
    return str(float(tau)) if str(float(tau)) in crps_variant else None


def mean_dv_crps(crps, variant: str, taus=DEFAULT_TAUS,
                 contrasts=DEFAULT_CONTRASTS) -> float:
    """Mean CAL dV CRPS over the gating contrasts and taus for a variant."""
    per_tau = crps[variant]
    vals = []
    for tau in taus:
        key = _tau_key(per_tau, tau)
        if key is None:
            raise KeyError(f"variant {variant!r} has no tau {tau}")
        row = per_tau[key]
        for c in contrasts:
            if c not in row:
                raise KeyError(f"variant {variant!r} tau {tau} has no "
                               f"contrast {c!r}")
            vals.append(float(row[c]))
    return float(np.mean(vals))


def variant_coverage(coverage, variant: str, taus=DEFAULT_TAUS) -> float:
    """Overall 10-90 coverage for a variant (scalar or per-tau dict)."""
    cov = coverage[variant]
    if isinstance(cov, dict):
        vals = []
        for tau in taus:
            key = tau if tau in cov else (
                str(float(tau)) if str(float(tau)) in cov else None)
            if key is None:
                continue
            vals.append(float(cov[key]))
        return float(np.mean(vals)) if vals else float("nan")
    return float(cov)


def select_planner_variant(*, crps, coverage, taus=DEFAULT_TAUS,
                           contrasts=DEFAULT_CONTRASTS,
                           tie_rel: float = DEFAULT_TIE_REL,
                           cov_bounds=DEFAULT_COV_BOUNDS,
                           worlds=None, cal_half: str = None,
                           cal_split_path=None) -> dict:
    """Apply the design section 5 rule; returns a JSON-ready record.

    When ``cal_half`` is given (A4.2), the variant choice is pinned to that
    half of ``CAL_SPLIT_A4.json``: every ``worlds`` entry must belong to it or
    :class:`evalh.cal_split.CalSplitError` is raised.
    """
    if cal_half is not None:
        from evalh import cal_split

        if worlds is None:
            raise ValueError(
                f"cal_half={cal_half!r} needs the scored 'worlds'")
        cal_split.require_worlds(worlds, cal_half, path=cal_split_path,
                                 context="variant choice")
    missing = [v for v in VARIANTS if v not in crps or v not in coverage]
    if missing:
        raise KeyError(f"missing variant score/coverage for {missing}")
    scores = {v: mean_dv_crps(crps, v, taus, contrasts) for v in VARIANTS}
    covs = {v: variant_coverage(coverage, v, taus) for v in VARIANTS}
    lo, hi = float(cov_bounds[0]), float(cov_bounds[1])
    eligible = [v for v in VARIANTS if lo <= covs[v] <= hi]
    if not eligible:
        return dict(chosen=None, reason="no variant meets the coverage "
                    f"constraint [{lo}, {hi}]", scores=scores,
                    coverages=covs, eligible=[], taus=list(taus),
                    contrasts=list(contrasts), tie_rel=float(tie_rel),
                    cov_bounds=[lo, hi])
    best = min(scores[v] for v in eligible)
    band = best * (1.0 + float(tie_rel)) + 1e-12
    tied = [v for v in eligible if scores[v] <= band]
    chosen = min(tied, key=lambda v: COMPLEXITY_RANK[v])
    return dict(chosen=chosen, reason="lowest CAL mean dV CRPS subject to "
                "coverage; 2% tie -> simpler (T<A<C<N)",
                scores=scores, coverages=covs, eligible=eligible,
                tied=tied, taus=[float(t) for t in taus],
                contrasts=list(contrasts), tie_rel=float(tie_rel),
                cov_bounds=[lo, hi])


def planner_variant_sha(record: dict) -> str:
    """sha256 of the canonical JSON of ``record`` (its sha256 field aside)."""
    payload = {k: v for k, v in record.items() if k != "sha256"}
    blob = json.dumps(payload, sort_keys=True, indent=1)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def write_planner_variant(path, record: dict) -> str:
    """Write PLANNER_VARIANT.json (with sha256) and return the sha."""
    out = {k: v for k, v in record.items() if k != "sha256"}
    out["sha256"] = planner_variant_sha(record)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, sort_keys=True, indent=1),
                          encoding="utf-8")
    return out["sha256"]


def verify_planner_variant(path) -> str:
    """Recompute the sha of a written PLANNER_VARIANT.json; raises on drift."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    got = planner_variant_sha(data)
    if got != data.get("sha256"):
        raise ValueError(f"PLANNER_VARIANT.json sha mismatch: {got} != "
                         f"{data.get('sha256')}")
    return got
