"""The v0.4b direct-difference quantile-GBM (registration s.5).

ONE quantile-GBM (HistGradientBoosting, quantile loss) per
(tau x sensing variant) on dY DIRECTLY -- no level head, no delta head,
no pairing of two predictions; the v0.4a hybrid ridge-residual machinery
is dropped (registered simplification).  Quantile levels = partB_v0
QUANTILES (0.05 ... 0.95); the predictive dY distribution is the model's
own 7-quantile function sampled on the SAME K = 40 midpoint grid as
partB_v0.acceptance_v04a (identical CRPS code path for model and
baselines).

Tuning: world-grouped CV on TRAIN ONLY (GroupKFold over worlds; no CAL;
aux rows train but never select), over the registered fixed grid of 6
candidates, selected per tau by mean pinball loss over the 7 levels on
the delay head of the HEADLINE (wall-native) rows.  Conformal shifts for
the dY quantiles are fitted per (tau, head, level) on TRAIN
world-grouped OOB residuals of the headline rows only (the v0.4a CalMap
method: auxiliary rows never enter selection or calibration), then
frozen; shifted quantiles are made monotone (cum-max) at predict time.

Reported reference: ridge-on-dY on the SAME inputs (never gating).
Auxiliary rows ARE training rows (owner ruling s.11.1); the wall-only
ablation refits without them and is reported with its own verdict.

CPU only; fits run thread-limited (1) so a concurrent featurization job
keeps the box.  Zero SUMO.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np

from . import _paths  # noqa: F401
from . import reg

try:
    from threadpoolctl import threadpool_limits
except ImportError:                                    # pragma: no cover
    threadpool_limits = None


@contextmanager
def _limit(n: int = 1):
    if threadpool_limits is not None:
        with threadpool_limits(limits=n):
            yield
    else:                                              # pragma: no cover
        yield


def pinball(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    """Mean pinball (quantile) loss at level q."""
    d = np.asarray(y, float) - np.asarray(pred, float)
    return float(np.mean(np.maximum(q * d, (q - 1.0) * d)))


class DirectDiffGBM:
    """Per-(tau, head, q) quantile regressors on dY + conformal shifts."""

    def __init__(self, taus=reg.TAUS, quantiles=reg.QUANTILES,
                 heads=reg.HEADS, cv_folds: int = reg.CV_FOLDS,
                 grid=reg.GRID, seed: int = 20261001, threads: int = 1):
        self.taus = tuple(float(t) for t in taus)
        self.quantiles = tuple(sorted(float(q) for q in quantiles))
        self.heads = tuple(heads)
        self.cv_folds = int(cv_folds)
        self.grid = tuple(grid)
        self.seed = int(seed)
        self.threads = int(threads)
        self.models: dict = {}        # (tau, head, q) -> regressor
        self.shifts: dict = {}        # (tau, head, q) -> float
        self.ridge: dict = {}         # (tau, head) -> Ridge (reference)
        self.chosen: dict = {}        # tau -> grid candidate tuple
        self.selection: list = []     # per-tau tuning records
        self.oof: dict = {}           # (tau, head) -> {q: oof preds}
        self.fit_seconds: dict = {}

    # ------------------------------------------------------------- folds --
    def world_folds(self, groups: np.ndarray) -> list:
        """World-grouped GroupKFold folds (no world on both sides); a
        slice with fewer than 2 worlds is refused -- never silently
        row-split (the v0.3.2 #10 rule)."""
        uw = np.unique(groups)
        f = int(min(self.cv_folds, len(uw)))
        if f < 2:
            raise ValueError(
                f"world-grouped CV needs >= 2 worlds, got {len(uw)}")
        from sklearn.model_selection import GroupKFold
        return list(GroupKFold(n_splits=f).split(
            np.zeros((len(groups), 1)), groups=groups))

    # ---------------------------------------------------------- fitting --
    def _hgb(self, cand, q: float):
        from sklearn.ensemble import HistGradientBoostingRegressor
        max_iter, lr, leaves, min_leaf = cand
        return HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=int(max_iter),
            learning_rate=float(lr), max_leaf_nodes=int(leaves),
            min_samples_leaf=int(min_leaf), random_state=self.seed,
            early_stopping=False)

    def _oof_q(self, cand, q, X, y, folds) -> np.ndarray:
        oof = np.full(len(y), np.nan)
        for fit_i, cal_i in folds:
            m = self._hgb(cand, q).fit(X[fit_i], y[fit_i])
            oof[cal_i] = m.predict(X[cal_i])
        return oof

    def fit(self, X: np.ndarray, y: dict, tau: float, groups: np.ndarray,
            headline: np.ndarray, log=print) -> "DirectDiffGBM":
        """Fit one tau.  X: [n, F] design; y: {head: [n] dY};
        groups: world ids (world-grouped CV); headline: bool mask of the
        gating (wall-native) rows used for selection + conformal."""
        tau = float(tau)
        t0 = time.perf_counter()
        folds = self.world_folds(groups)
        yd = np.asarray(y[reg.GATING_HEAD], float)
        hl = np.asarray(headline, bool)
        with _limit(self.threads):        # the WHOLE fit (incl. tuning):
            # unbounded HGB on a loaded box thrashes OpenMP (v0.4a #10
            # precedent: limited fits stay fast while a featurization
            # job saturates the cores)
            # ---- tuning over the FIXED grid (headline delay rows) ----
            scored = []
            for cand in self.grid:
                oof_q = {q: self._oof_q(cand, q, X, yd, folds)
                         for q in self.quantiles}
                loss = float(np.mean(
                    [pinball(yd[hl], oof_q[q][hl], q)
                     for q in self.quantiles]))
                scored.append((loss, cand, oof_q))
            scored.sort(key=lambda t: (t[0], self.grid.index(t[1])))
            best_loss, best_cand, best_oof = scored[0]
            self.chosen[tau] = best_cand
            self.selection.append(dict(
                tau=tau, chosen=list(best_cand), pinball_delay=best_loss,
                grid=[dict(cand=list(c), pinball=float(l))
                       for l, c, _o in scored]))
            # ---- final fits on ALL TRAIN rows (aux included) ----
            for h in self.heads:
                yh = np.asarray(y[h], float)
                for q in self.quantiles:
                    self.models[(tau, h, q)] = self._hgb(
                        best_cand, q).fit(X, yh)
                # conformal shifts: OOB residuals of the HEADLINE rows
                oof = ({q: best_oof[q] for q in self.quantiles}
                       if h == reg.GATING_HEAD else
                       {q: self._oof_q(best_cand, q, X, yh, folds)
                        for q in self.quantiles})
                self.oof[(tau, h)] = oof
                res = (yh - oof[q])[hl]
                res = res[np.isfinite(res)]
                for q in self.quantiles:
                    self.shifts[(tau, h, q)] = \
                        float(np.quantile(res, q)) if len(res) else 0.0
                # ridge-on-dY reference on the SAME inputs (reported only)
                from sklearn.linear_model import Ridge
                self.ridge[(tau, h)] = Ridge(alpha=1.0).fit(X, yh)
        self.fit_seconds[tau] = round(time.perf_counter() - t0, 1)
        log(f"v04b[fit]: tau={tau:.0f} chosen {best_cand} "
            f"pinball={best_loss:.2f} n={len(yd)} "
            f"(headline {int(hl.sum())}) in "
            f"{self.fit_seconds[tau]:.1f} s")
        return self

    # --------------------------------------------------------- predict --
    def predict_quantiles(self, X: np.ndarray, tau: float,
                          head: str = reg.GATING_HEAD,
                          calibrated: bool = True) -> dict:
        """{q: [n]} shifted + monotone quantile predictions."""
        tau = float(tau)
        raw = {q: self.models[(tau, head, q)].predict(X)
               for q in self.quantiles}
        if not calibrated:
            raw = self._monotone(raw)
            return raw
        out = {q: raw[q] + self.shifts.get((tau, head, q), 0.0)
               for q in raw}
        return self._monotone(out)

    def predict_median(self, X: np.ndarray, tau: float,
                       head: str = reg.GATING_HEAD) -> np.ndarray:
        return self.predict_quantiles(X, tau, head)[0.5]

    def ridge_median(self, X: np.ndarray, tau: float,
                     head: str = reg.GATING_HEAD) -> np.ndarray:
        return self.ridge[(float(tau), head)].predict(X)

    @staticmethod
    def _monotone(preds: dict) -> dict:
        out, run = {}, None
        for q in sorted(preds):
            run = preds[q] if run is None else np.maximum(run, preds[q])
            out[q] = run
        return out

    # ------------------------------------------------------------ ablation --
    def fit_wall_only(self, X: np.ndarray, y: dict, tau: float,
                      groups: np.ndarray, headline: np.ndarray,
                      log=print) -> "DirectDiffGBM":
        """The registered wall-only (no-aux) ablation: TRAIN restricted to
        the headline rows; same grid, same rule.  Returns a SEPARATE
        model (labelled ABLATION; never merged into the shipped one)."""
        hl = np.asarray(headline, bool)
        m = DirectDiffGBM(taus=self.taus, quantiles=self.quantiles,
                          heads=self.heads, cv_folds=self.cv_folds,
                          seed=self.seed, threads=self.threads)
        return m.fit(X[hl], {h: np.asarray(v)[hl] for h, v in y.items()},
                     tau, np.asarray(groups)[hl], np.ones(int(hl.sum()), bool),
                     log=log)

