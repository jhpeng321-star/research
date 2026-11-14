"""CRPS and CDF utilities on quantile / CDF representations.

Part B predicts a delay distribution as a small set of quantiles
(``partB_v0.gbm.QUANTILES``: 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95).  The
combination layer needs to (a) score one predictive distribution against
a realised value, (b) pool two predictive CDFs, and (c) measure their
distance.  Two documented approximations are used:

``crps_pinball``
    CRPS ~= 2 * [q0*QL(q0) + trapz(QL, levels) + (1-q1)*QL(q1)], where
    QL_q is the pinball loss (x - pred_q) * (q - 1{x < pred_q}).  This is
    EXACTLY ``partB_v0.gbm.crps_from_quantiles`` and is used when only a
    quantile grid is available.

``crps_cdf``
    CRPS(F, y) = Integral (F(x) - 1{x >= y})^2 dx, evaluated by the
    trapezoid rule on a COMMON x grid (``build_cdf_grid``).  Pooling two
    CDFs is exact in this representation (F_C = w F_N + (1-w) F_T), so the
    CDF form is the one the stack and the disagreement signal use.  The
    grid is dense (default 400 points) and is built per tau from the union
    of both predictive quantiles plus the observed values, padded 25 % on
    each side, so the approximation error is far below the spread of the
    distribution.

A ``QuantilePred`` turns the quantile grid into a piecewise-linear CDF:
the CDF passes through (pred_q, q) for each fitted level and is flat 0/1
outside [pred_min, pred_max].  Quantiles at an arbitrary probability are
read back by inverting that same piecewise-linear CDF.  Duplicate quantile
nodes are separated by a negligible span-relative jitter so the linear
interpolation is well defined; the jitter is 1e-12 of the row spread.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_JITTER = 1e-12


@dataclass
class QuantilePred:
    """A per-row quantile predictive distribution.

    levels: [L] strictly increasing probabilities in (0, 1)
    qvals:  [n, L] quantile values (monotone along the level axis)
    """

    levels: np.ndarray
    qvals: np.ndarray

    def __post_init__(self):
        lv = np.asarray(self.levels, dtype=np.float64).ravel()
        qv = np.asarray(self.qvals, dtype=np.float64)
        if qv.ndim == 1:
            qv = qv[None, :]
        if qv.shape[1] != lv.size:
            raise ValueError(
                f"levels ({lv.size}) and qvals ({qv.shape[1]}) disagree")
        order = np.argsort(lv)
        self.levels = lv[order]
        if np.any(np.diff(self.levels) <= 0):
            raise ValueError("levels must be strictly increasing")
        self.qvals = np.maximum.accumulate(qv[:, order], axis=1)

    @property
    def n_rows(self) -> int:
        return self.qvals.shape[0]

    @property
    def n_levels(self) -> int:
        return self.levels.size

    def take(self, idx) -> "QuantilePred":
        return QuantilePred(self.levels, self.qvals[np.asarray(idx)])

    def _nodes_all(self) -> np.ndarray:
        nx = self.qvals.copy()
        if nx.shape[1] > 1:
            span = np.maximum(nx[:, -1] - nx[:, 0], 1.0)
            nx = nx + np.arange(nx.shape[1])[None, :] * (_JITTER * span[:, None])
        return nx

    def _nodes(self, i: int) -> np.ndarray:
        return self._nodes_all()[i]

    def cdf_row(self, i: int, x: np.ndarray) -> np.ndarray:
        nx = self._nodes(i)
        out = np.interp(x, nx, self.levels)
        out[x < nx[0]] = 0.0
        out[x > nx[-1]] = 1.0
        return out

    def cdf_at(self, x) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        nx = self._nodes_all()
        n, L = nx.shape
        ge = x[None, :, None] >= nx[:, None, :]
        j = np.clip(ge.sum(axis=2) - 1, 0, L - 2)
        row = np.arange(n)[:, None]
        x0 = nx[row, j]
        x1 = nx[row, j + 1]
        y0 = self.levels[j]
        y1 = self.levels[j + 1]
        span = np.where(x1 > x0, x1 - x0, 1.0)
        t = np.clip((x[None, :] - x0) / span, 0.0, 1.0)
        out = y0 + (y1 - y0) * t
        out[x[None, :] < nx[:, 0][:, None]] = 0.0
        out[x[None, :] > nx[:, -1][:, None]] = 1.0
        return out

    def quantiles_at(self, probs) -> np.ndarray:
        """Invert the piecewise-linear CDF at each probability; clamps."""
        p = np.atleast_1d(np.asarray(probs, dtype=np.float64))
        out = np.empty((self.n_rows, p.size), dtype=np.float64)
        for i in range(self.n_rows):
            nx = self._nodes(i)
            out[i] = np.interp(p, self.levels, nx)
        return out


def crps_pinball(levels, qvals, y) -> np.ndarray:
    """Pinball-average CRPS (identical to partB_v0.gbm.crps_from_quantiles)."""
    lv = np.sort(np.asarray(levels, dtype=np.float64))
    qv = np.asarray(qvals, dtype=np.float64)
    if qv.ndim == 1:
        qv = qv[None, :]
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    ql = np.maximum(lv[None, :] * (y - qv), (lv[None, :] - 1.0) * (y - qv))
    total = lv[0] * ql[:, 0] + (1.0 - lv[-1]) * ql[:, -1]
    for a, b, fa, fb in zip(lv[:-1], lv[1:], ql[:, :-1].T, ql[:, 1:].T):
        total = total + 0.5 * (b - a) * (fa + fb)
    return 2.0 * total


def build_cdf_grid(preds, y, n: int = 400) -> np.ndarray:
    """A common, per-tau dense x grid covering both predictives and y."""
    parts = [np.asarray(p.qvals, dtype=np.float64).ravel() for p in preds]
    parts.append(np.asarray(y, dtype=np.float64).ravel())
    vals = np.concatenate(parts)
    lo, hi = float(np.min(vals)), float(np.max(vals))
    pad = 0.25 * (hi - lo) + 1e-9
    return np.linspace(lo - pad, hi + pad, int(n))


def crps_cdf(fn: QuantilePred, ft: QuantilePred, w: float, y,
             x_grid) -> np.ndarray:
    """Trapezoid CRPS of the pooled CDF F_C = w*F_N + (1-w)*F_T."""
    x = np.asarray(x_grid, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    fc = w * fn.cdf_at(x) + (1.0 - w) * ft.cdf_at(x)
    h = (x[None, :] >= yv[:, None]).astype(np.float64)
    return np.trapezoid((fc - h) ** 2, x, axis=1)


def pooled_crps_series(fn: QuantilePred, ft: QuantilePred, y, x_grid,
                       w_grid) -> np.ndarray:
    """[n_w, n_rows] CRPS of F_C for every w, computing the two CDFs once."""
    x = np.asarray(x_grid, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    fn_cdf = fn.cdf_at(x)
    ft_cdf = ft.cdf_at(x)
    h = (x[None, :] >= yv[:, None]).astype(np.float64)
    out = np.empty((len(w_grid), len(yv)), dtype=np.float64)
    for i, w in enumerate(np.asarray(w_grid, dtype=np.float64)):
        fc = w * fn_cdf + (1.0 - w) * ft_cdf
        out[i] = np.trapezoid((fc - h) ** 2, x, axis=1)
    return out


def cdf_distance(fn: QuantilePred, ft: QuantilePred, x_grid) -> np.ndarray:
    """Integrated squared CDF difference (a CRPS-style distributional
    distance): Integral (F_N(x) - F_T(x))^2 dx."""
    x = np.asarray(x_grid, dtype=np.float64)
    d = fn.cdf_at(x) - ft.cdf_at(x)
    return np.trapezoid(d ** 2, x, axis=1)


def mixture_quantiles(fn: QuantilePred, ft: QuantilePred, w: float,
                      probs) -> np.ndarray:
    """Quantiles of F_C = w*F_N + (1-w)*F_T at ``probs`` (per row)."""
    p = np.atleast_1d(np.asarray(probs, dtype=np.float64))
    out = np.empty((fn.n_rows, p.size), dtype=np.float64)
    for i in range(fn.n_rows):
        nx = np.unique(np.concatenate([fn._nodes(i), ft._nodes(i)]))
        fc = w * fn.cdf_row(i, nx) + (1.0 - w) * ft.cdf_row(i, nx)
        fc = np.maximum.accumulate(fc) + np.arange(nx.size) * 1e-15
        out[i] = np.interp(p, fc, nx)
    return out
