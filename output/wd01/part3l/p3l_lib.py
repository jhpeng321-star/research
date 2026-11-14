"""WD-01 part 3l shared library: splits, fit-window helpers, spread
calibration math (pure functions, unit-tested).

Splits (part 3l):
  fit roots   = first 4 sorted TRAIN roots      (the p3j/p3k convention)
  calibration = NEXT 4 sorted TRAIN roots       (TRAIN-internal, held out
                from every fit in part 3l; validation never touched)
  validation / holdout = p3lib.roots_by_split() (holdout stays blocked)

Spread calibration (TASK C, mechanism A): per (output, horizon) factor
s_out >= 1 widening each window's member ensemble around its ensemble
MEAN,  x'_i = mean + s * (x_i - mean),  so the ensemble MEAN is exactly
invariant for any s (the brief's mean-invariance requirement); the
median-centred variant is computed alongside as a check.  s_out is the
smallest grid factor whose widened 10-90 % band covers the truth in
>= 80 % of the CALIBRATION windows.

ZERO simulation: no SUMO/TraCI import anywhere in part 3l; the only
dynamics is the part-3 rollout engine (pure torch).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P3K = HERE.parent / "part3k"
for _p in (str(P3K),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import decompose_p3k as D  # noqa: E402  (delay model + rollout helpers)
import p3lib as P3  # noqa: E402

N_FIT_ROOTS = 4
N_CAL_ROOTS = 4
HORIZONS = (30, 60, 90, 120)
CAL_T0S = tuple(range(120, 481, 60))     # 7 t0 x 4 calibration roots = 28

LC_GRID = (0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.003)
TAU_GRID = (1.9, 2.4, 3.0, 3.6)
B_GRID = (1.5, 2.0, 3.0)
DAWDLE_GRID = (0.8, 1.0, 1.25, 1.5)
P_COOP_GRID = (0.0, 0.25)                # narrow: p3k selected 0.0
COVERAGE_TARGET = 0.80
S_GRID = tuple(np.round(np.geomspace(1.0, 16.0, 31), 4))

# Canonical engine merge_stats keys (part 3l TASK A instrumentation;
# unit test asserts Rollout.merge_stats uses exactly these spellings --
# import these constants instead of re-typing the strings).
MS_CHECKS = "checks"
MS_ACCEPTED = "accepted"
MS_REJ_LEAD = "rej_lead"          # lead-gap term rejected, follower ok
MS_REJ_FOLL = "rej_foll"          # follower term rejected, lead ok
MS_REJ_BOTH = "rej_both"          # both terms rejected
MS_WITH_FOLL = "with_foll"        # checks with a follower present
MS_BREQ_HIST = "breq_hist"        # b_req histogram over follower checks
MS_BREQ_FOLL_REJ = "breq_foll_rej"  # b_req values at follower rejections


def splits_3l():
    """dict(fit=[4 roots], calibration=[4 roots], val=[...], holdout=...).
    The holdout entry is carried only to prove it is never passed to a
    rollout; every script asserts it stays disjoint from what it uses."""
    sp = P3.roots_by_split()
    train = sorted(sp["train"])
    return dict(fit=train[:N_FIT_ROOTS],
                calibration=train[N_FIT_ROOTS:N_FIT_ROOTS + N_CAL_ROOTS],
                val=sorted(sp["val"]), holdout=sorted(sp["holdout"]))


def lc_count_between(snaps, j0, j1):
    """Pred LC count between snapshot ticks j0 (exclusive of its own prior
    count) and j1 inclusive: counter difference snaps[j1]-snaps[j0]."""
    return int(snaps[j1]["lc"]) - int(snaps[j0]["lc"])


def lc_truth_between(ra, k0, k1):
    return int(((ra.lc_tick > k0) & (ra.lc_tick <= k1)).sum())


# ------------------------------------------------------------- spread ----

def widen_ensemble(vals, s, centre="mean"):
    """Widen one window's member ensemble by factor s around its centre.
    centre='mean': the ensemble mean is EXACTLY invariant.  centre='median'
    (brief's wording): median preserved, mean shifts -- reported as a
    check only.  Singleton ensembles are returned unchanged."""
    x = np.asarray(vals, dtype=np.float64)
    if centre == "mean":
        c = float(x.mean())
    else:
        c = float(np.median(x))
    return c + s * (x - c)


def band_10_90(vals):
    return float(np.percentile(vals, 10)), float(np.percentile(vals, 90))


def coverage_frac(samples, truths):
    """Fraction of windows whose (member) 10-90 % band covers the truth."""
    hit = [band_10_90(x)[0] <= y <= band_10_90(x)[1]
           for x, y in zip(samples, truths)]
    return float(np.mean(hit)) if hit else float("nan")


def s_out_for(samples, truths, target=COVERAGE_TARGET, grid=S_GRID):
    """Smallest grid factor s whose widened 10-90 % band reaches the
    coverage target on the calibration windows; None if the grid tops
    out below target (degenerate windows are counted as misses)."""
    best = None
    for s in grid:
        cov = coverage_frac([widen_ensemble(x, s) for x in samples], truths)
        if cov >= target:
            best = float(s)
            break
    return best


def crps_ensemble(vals, truth):
    """Empirical CRPS: mean|x_i - y| - 0.5 * mean_ij |x_i - x_j|."""
    x = np.asarray(vals, dtype=np.float64)
    if len(x) < 2:
        return float(abs(x[0] - truth)) if len(x) else float("nan")
    return (float(np.mean(np.abs(x - truth)))
            - 0.5 * float(np.mean(np.abs(x[:, None] - x[None, :]))))


def load_fit(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
