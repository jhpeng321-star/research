"""Disagreement signal between the neural and tree predictive CDFs.

D_tau = CRPS-distance(F_N, F_T) / scale_tau, the integrated squared CDF
difference (``crps.cdf_distance``) normalised by a per-tau scale.  It is
exposed to the planner as an uncertainty / abstain input: rows where the
two models disagree are the rows where a combined prediction is least
trustworthy.  The reporting helpers give the Spearman correlation of D
with the realised absolute error and an abstain curve (threshold -> kept
fraction and mean error of the kept rows).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .crps import QuantilePred, build_cdf_grid, cdf_distance


def disagreement(fn: QuantilePred, ft: QuantilePred, *, x_grid=None,
                 scale=1.0, y=None) -> np.ndarray:
    """Per-row CRPS-distance between F_N and F_T, normalised by ``scale``."""
    if x_grid is None:
        ref = build_cdf_grid([fn, ft],
                             np.asarray(y) if y is not None else
                             np.concatenate([fn.qvals[:, 0],
                                             ft.qvals[:, -1]]))
        x_grid = ref
    d = cdf_distance(fn, ft, x_grid)
    return d / float(scale)


def tau_scale(distances, *, floor: float = 1e-9) -> float:
    """A per-tau normaliser for D: the median absolute distance."""
    d = np.asarray(distances, dtype=np.float64)
    med = float(np.median(np.abs(d))) if d.size else 0.0
    return max(med, float(floor))


def spearman_abs_error(d, abs_error) -> dict:
    """Spearman(D, |error|): does disagreement track realised error?"""
    d = np.asarray(d, dtype=np.float64)
    e = np.asarray(abs_error, dtype=np.float64)
    mask = np.isfinite(d) & np.isfinite(e)
    if mask.sum() < 3 or np.allclose(d[mask], d[mask][0]):
        return dict(rho=float("nan"), pvalue=float("nan"),
                    n=int(mask.sum()))
    from scipy.stats import spearmanr
    rho, p = spearmanr(d[mask], e[mask])
    return dict(rho=float(rho), pvalue=float(p), n=int(mask.sum()))


def abstain_curve(d, abs_error, thresholds=None) -> pd.DataFrame:
    """For each disagreement threshold: kept fraction + mean error kept."""
    d = np.asarray(d, dtype=np.float64)
    e = np.asarray(abs_error, dtype=np.float64)
    if thresholds is None:
        qs = np.linspace(0.1, 1.0, 10)
        thresholds = np.quantile(d, qs)
    rows = []
    for t in np.asarray(thresholds, dtype=np.float64):
        keep = d <= t
        rows.append(dict(
            threshold=float(t),
            frac_kept=float(keep.mean()) if d.size else float("nan"),
            mean_abs_error=(float(e[keep].mean()) if keep.any()
                            else float("nan")),
            n_kept=int(keep.sum())))
    return pd.DataFrame(rows)
