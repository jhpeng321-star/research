"""CAL-only stacking and conformal recalibration (WM-N1 design section 5a).

Per tau and target, pool the two predictive CDFs F_N (neural) and F_T
(tree): F_C = w*F_N + (1-w)*F_T with w in [0, 1].  On the quantile
representation the pool is exact in CDF space but NOT the average of the
quantile values, so the weight is fitted by minimising the trapezoid CRPS
of the pooled CDF on a dense common grid (``crps.crps_cdf``), and mixture
quantiles are read back by inverting that CDF (``crps.mixture_quantiles``).

The weight uses WORLD-GROUPED 2-FOLD CROSS-FITTING on CAL rows only: each
half is scored with a weight fitted on the other half, and the shipped
weight is the one fitted on all CAL rows.  The conformal recalibration is
a per-level shift (the q-quantile of the CAL residuals y - F_C^{-1}(q)).
Both steps refuse any non-CALIBRATION row (``guards.guard_cal_only``), so
TEST/HOLDOUT can never influence the pool.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .crps import (QuantilePred, build_cdf_grid, crps_cdf,
                   mixture_quantiles, pooled_crps_series)
from .guards import guard_cal_only

DEFAULT_QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def world_folds(worlds, folds: int = 2) -> dict:
    """Deterministic world-grouped fold assignment ``world -> fold``.

    Mirrors ``oof_tree.partition_worlds`` semantics (sorted worlds,
    round-robin) so the two layers agree on what "disjoint worlds" means.
    """
    uw = sorted({str(w) for w in worlds})
    if not uw:
        raise ValueError("no worlds to fold")
    k = max(1, min(int(folds), len(uw)))
    return {w: i % k for i, w in enumerate(uw)}


def pool_crps_curve(fn: QuantilePred, ft: QuantilePred, y, w_grid,
                    x_grid=None):
    """Mean CRPS of F_C = w*F_N + (1-w)*F_T for each w in ``w_grid``."""
    y = np.asarray(y, dtype=np.float64)
    w_grid = np.asarray(w_grid, dtype=np.float64)
    if x_grid is None:
        x_grid = build_cdf_grid([fn, ft], y)
    x_grid = np.asarray(x_grid, dtype=np.float64)
    series = pooled_crps_series(fn, ft, y, x_grid, w_grid)
    return w_grid, series.mean(axis=1), x_grid


def fit_pool_weight(fn: QuantilePred, ft: QuantilePred, y, *,
                    w_step: float = 0.01, refine: int = 2, x_grid=None) -> dict:
    """Minimise the pooled-CDF CRPS over w in [0, 1] by grid search."""
    y = np.asarray(y, dtype=np.float64)
    if x_grid is None:
        x_grid = build_cdf_grid([fn, ft], y)
    x_grid = np.asarray(x_grid, dtype=np.float64)

    def mean_curve(wg):
        return pooled_crps_series(fn, ft, y, x_grid, wg).mean(axis=1)

    wg = np.arange(0.0, 1.0 + 1e-12, w_step)
    curve = mean_curve(wg)
    k = int(np.argmin(curve))
    best_w, best_c = float(wg[k]), float(curve[k])
    step = float(w_step)
    for _ in range(int(refine)):
        lo, hi = max(0.0, best_w - step), min(1.0, best_w + step)
        wg = np.linspace(lo, hi, 21)
        c = mean_curve(wg)
        j = int(np.argmin(c))
        if c[j] < best_c - 1e-12:
            best_w, best_c = float(wg[j]), float(c[j])
        step /= 10.0
    coarse_w = np.arange(0.0, 1.0 + 1e-12, 0.05)
    coarse = mean_curve(coarse_w)
    return dict(weight=best_w, crps=best_c,
                curve=[[float(w), float(v)]
                       for w, v in zip(coarse_w, coarse)])


def cross_fit_pool(fn: QuantilePred, ft: QuantilePred, y, worlds, splits, *,
                   folds: int = 2, cal_half: str = None,
                   cal_split_path=None, **fit_kw) -> dict:
    """World-grouped cross-fitted pool weight on CAL rows only.

    Each fold's rows are scored with the weight fitted on the OTHER fold;
    the shipped weight is fitted on all CAL rows.  Returns the per-row
    cross-fitted CRPS and the fold bookkeeping.  ``cal_half`` (A4.2) pins the
    rows to CAL-cal before the weight is fitted.
    """
    guard_cal_only(splits, "cross_fit_pool", worlds=worlds, half=cal_half,
                   cal_split_path=cal_split_path)
    y = np.asarray(y, dtype=np.float64)
    assign = world_folds(worlds, folds)
    fold = np.array([assign[str(w)] for w in worlds])
    k = len(set(assign.values()))
    oof = np.full(len(y), np.nan)
    records = []
    for f in range(k):
        te = fold == f
        tr = ~te
        if not te.any() or not tr.any():
            continue
        rec = fit_pool_weight(fn.take(tr), ft.take(tr), y[tr], **fit_kw)
        xg = build_cdf_grid([fn.take(te), ft.take(te)], y[te])
        oof[te] = crps_cdf(fn.take(te), ft.take(te), rec["weight"],
                           y[te], xg)
        records.append(dict(
            fold=f, weight=rec["weight"],
            test_worlds=sorted({str(w) for w, fl in zip(worlds, fold)
                                if fl == f}),
            train_worlds=sorted({str(w) for w, fl in zip(worlds, fold)
                                 if fl != f})))
    final = fit_pool_weight(fn, ft, y, **fit_kw)
    return dict(final_weight=final["weight"], final_crps=final["crps"],
                folds=records, fold_of=assign, n_folds=k,
                oof_crps=oof,
                mean_oof_crps=float(np.nanmean(oof)) if len(y) else float("nan"))


def conformal_shifts(levels, qvals, y) -> np.ndarray:
    """Per-level conformal shift = q-quantile of the CAL residuals."""
    lv = np.asarray(levels, dtype=np.float64)
    qv = np.asarray(qvals, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    res = y[:, None] - qv
    return np.array([float(np.quantile(res[:, j], lv[j]))
                     for j in range(lv.size)])


@dataclass
class ConformalCalibration:
    levels: np.ndarray
    shifts: np.ndarray

    def apply(self, qvals) -> np.ndarray:
        out = np.asarray(qvals, dtype=np.float64) + self.shifts[None, :]
        return np.maximum.accumulate(out, axis=1)

    def to_dict(self) -> dict:
        return dict(levels=[float(q) for q in self.levels],
                    shifts=[float(s) for s in self.shifts])


def fit_conformal_pool(fn: QuantilePred, ft: QuantilePred, weight: float, y,
                       splits, levels=DEFAULT_QUANTILES, *, worlds=None,
                       cal_half: str = None, cal_split_path=None
                       ) -> ConformalCalibration:
    """Fit per-level conformal shifts of the pooled CDF on CAL rows only."""
    guard_cal_only(splits, "fit_conformal_pool", worlds=worlds,
                   half=cal_half, cal_split_path=cal_split_path)
    lv = np.asarray(sorted(float(q) for q in levels), dtype=np.float64)
    qvals = mixture_quantiles(fn, ft, float(weight), lv)
    return ConformalCalibration(lv, conformal_shifts(lv, qvals, y))


@dataclass
class StackModel:
    weight: float
    cal: ConformalCalibration

    def pool_quantiles(self, fn: QuantilePred, ft: QuantilePred,
                       probs=None) -> np.ndarray:
        p = self.cal.levels if probs is None else probs
        return mixture_quantiles(fn, ft, float(self.weight), p)

    def predict(self, fn: QuantilePred, ft: QuantilePred) -> np.ndarray:
        """Recalibrated pooled quantiles at the fitted levels."""
        return self.cal.apply(self.pool_quantiles(fn, ft))

    def to_dict(self) -> dict:
        return dict(weight=float(self.weight), cal=self.cal.to_dict())


def fit_stack(fn: QuantilePred, ft: QuantilePred, y, worlds, splits, *,
              levels=DEFAULT_QUANTILES, folds: int = 2,
              cal_half: str = None, cal_split_path=None,
              **fit_kw) -> tuple[StackModel, dict]:
    """Cross-fitted weight + CAL conformal recalibration -> a StackModel."""
    lv = np.asarray(sorted(float(q) for q in levels), dtype=np.float64)
    xf = cross_fit_pool(fn, ft, y, worlds, splits, folds=folds,
                        cal_half=cal_half, cal_split_path=cal_split_path,
                        **fit_kw)
    cal = fit_conformal_pool(fn, ft, xf["final_weight"], y, splits, lv,
                             worlds=worlds, cal_half=cal_half,
                             cal_split_path=cal_split_path)
    return StackModel(weight=xf["final_weight"], cal=cal), xf


def interval_coverage(qvals, y, lo_q: float = 0.1,
                      hi_q: float = 0.9, levels=DEFAULT_QUANTILES) -> float:
    """Empirical coverage of the [lo_q, hi_q] central interval."""
    lv = np.asarray(sorted(float(q) for q in levels), dtype=np.float64)
    qv = np.asarray(qvals, dtype=np.float64)
    lo = qv[:, int(np.argmin(np.abs(lv - lo_q)))]
    hi = qv[:, int(np.argmin(np.abs(lv - hi_q)))]
    return float(np.mean((y >= lo) & (y <= hi)))
