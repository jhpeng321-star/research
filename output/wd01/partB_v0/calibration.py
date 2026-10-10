"""Post-hoc per-tau calibration of the v0.3 net (BRIEF_B3_V03.md D14).

Fits, per tau, a quantile SHIFT map (default; conformal-style: for each
level l, s_l = the l-quantile of the OOB residual y_norm - q_pred_l) or an
ISOTONIC map (monotone in the predicted quantile) on the pooled
OUT-OF-BAG TRAIN predictions.  The maps are saved (calibration.json,
ships with the weights) and applied to the predict-time quantiles; CAL
(the mean absolute coverage error over the central levels) is REPORTED
before and after on the CALIBRATION split and is never used for
selection (B6).

All quantities are in per-tau-NORMALISED units (A1); reporting multiplies
by the TRAIN medians to get real units.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CAL_SCHEMA = "wd01_partB_v03_calibration/1"
# central levels scored by the CAL metric (D14/D18: 3-decimal reporting)
CAL_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# levels fitted and shipped in the map
FIT_LEVELS = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 0.05..0.95


def quantile_matrix(grid, probs: np.ndarray,
                    levels=FIT_LEVELS) -> np.ndarray:
    """Predicted quantiles [N, L] from pmfs [N, K] (normalised units)."""
    return np.column_stack([grid.quantile(probs, q) for q in levels])


def coverage(y: np.ndarray, qmat: np.ndarray,
             levels=CAL_LEVELS) -> np.ndarray:
    """Empirical coverage per level; qmat columns align with `levels`."""
    y = np.asarray(y, dtype=np.float64)[:, None]
    return (y <= qmat).mean(axis=0)


def cal_metric(y: np.ndarray, qmat: np.ndarray,
               levels=CAL_LEVELS, fit_levels=FIT_LEVELS) -> float:
    """CAL = mean |coverage - nominal| over the central levels."""
    col = [list(fit_levels).index(lv) for lv in levels]
    cov = coverage(y, qmat[:, col], levels)
    return float(np.mean(np.abs(cov - np.asarray(levels))))


def pit_from_quantiles(y: np.ndarray, qmat: np.ndarray,
                       levels=FIT_LEVELS) -> np.ndarray:
    """PIT of each observation inside the calibrated quantile lattice
    (piecewise-linear inverse of the quantile function)."""
    y = np.asarray(y, dtype=np.float64)
    lv = np.asarray(levels, dtype=np.float64)
    n = y.shape[0]
    out = np.empty(n)
    for i in range(n):
        out[i] = np.interp(y[i], qmat[i], lv, left=0.0, right=1.0)
    return np.clip(out, 1e-6, 1.0 - 1e-6)


class CalMap:
    """Per-tau quantile maps: {tau: {"levels": [...], "shift": [...]}} or
    {"tau": {"levels": [...], "iso": [{"knots_x": [...], "knots_y":...}]}}.
    """

    def __init__(self, per_tau: dict, method: str, levels=FIT_LEVELS):
        self.per_tau = per_tau
        self.method = method
        self.levels = list(levels)

    def apply(self, qmat: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Calibrate predicted quantiles [N, L] per row's tau."""
        out = qmat.copy()
        for t, mp in self.per_tau.items():
            m = np.isclose(tau, float(t))
            if not m.any():
                continue
            if self.method == "shift":
                out[m] = qmat[m] + np.asarray(mp["shift"])[None, :]
            else:                                    # isotonic per level
                for j, iso in enumerate(mp["iso"]):
                    out[m, j] = np.interp(
                        qmat[m, j],
                        np.asarray(iso["knots_x"], dtype=np.float64),
                        np.asarray(iso["knots_y"], dtype=np.float64))
        return out

    def to_json(self) -> dict:
        return dict(schema=CAL_SCHEMA, method=self.method,
                    levels=[float(v) for v in self.levels],
                    per_tau=self.per_tau)

    @classmethod
    def from_json(cls, d: dict) -> "CalMap":
        return cls(d.get("per_tau", {}), d.get("method", "shift"),
                   d.get("levels", FIT_LEVELS))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CalMap":
        return cls.from_json(json.loads(
            Path(path).read_text(encoding="utf-8")))


def fit_per_tau(qmat: np.ndarray, y: np.ndarray, tau: np.ndarray,
                method: str = "shift", levels=FIT_LEVELS,
                min_rows: int = 30) -> CalMap:
    """Fit the per-tau maps on (pooled OOB) predictions + observations.

    shift:    s_l = l-quantile of (y - q_pred_l) per tau, so
              P(y <= q_pred_l + s_l) ~= l on the fitting rows.
    isotonic: per level, a monotone map q_pred_l -> y (binned empirical
              l-quantiles, monotone by accumulation; np.interp clamps
              outside the fitted range).
    """
    per_tau: dict = {}
    lv = np.asarray(levels, dtype=np.float64)
    for t in np.unique(tau):
        m = np.isclose(tau, float(t))
        if int(m.sum()) < min_rows:
            continue                      # too thin: leave this tau raw
        qt, yt = qmat[m], y[m]
        entry: dict = {"n": int(m.sum())}
        if method == "isotonic":
            entry["iso"] = []
            n = int(m.sum())
            nbins = int(max(5, min(20, n // 40)))
            for j in range(qt.shape[1]):
                edges = np.quantile(qt[:, j], np.linspace(0, 1, nbins + 1))
                edges[0] -= 1e-9
                edges[-1] += 1e-9
                centers = 0.5 * (edges[:-1] + edges[1:])
                ys = np.array([
                    np.quantile(yt[(qt[:, j] >= edges[b])
                                   & (qt[:, j] <= edges[b + 1])], lv[j])
                    for b in range(nbins)])
                ys = np.maximum.accumulate(ys)
                entry["iso"].append(dict(
                    knots_x=[float(x) for x in centers],
                    knots_y=[float(v) for v in ys]))
        else:
            shift = [float(np.quantile(yt - qt[:, j], lv[j]))
                     for j in range(qt.shape[1])]
            entry["shift"] = shift
        per_tau[str(float(t))] = entry
    return CalMap(per_tau, "shift" if method != "isotonic" else "isotonic",
                  levels)
