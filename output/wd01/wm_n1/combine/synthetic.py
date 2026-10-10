"""Synthetic fixtures for the combination layer (no real data, no SUMO).

Used by the CLI synthetic smoke and by the unit tests.  ``synthetic_rows``
builds a row table with a stable key, world id, split label and the
tau / target / policy conditioning; ``SyntheticQuantileModel`` is a
``GbmValueModel``-compatible stand-in (per-(target, tau, policy) mean plus
a tau-scaled Gaussian spread) that records the worlds and splits it was
fitted on, so leakage can be asserted directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .crps import QuantilePred
from .oof_tree import DEFAULT_QUANTILES

TAU_GRID = (60.0, 120.0, 300.0)
POLICIES = ("damping_fleet", "switching_mix")


def synthetic_rows(n_train: int = 8, n_cal: int = 4, n_test: int = 0,
                   *, taus=TAU_GRID, policies=POLICIES, seed: int = 0,
                   target: str = "delay") -> pd.DataFrame:
    """A deterministic synthetic row table with stable ``row_id``s."""
    rng = np.random.default_rng(seed)
    worlds = ([f"chunk1/w{i:04d}t2600" for i in range(n_train)]
              + [f"chunk2/c{i:04d}t2400" for i in range(n_cal)]
              + [f"chunk5/x{i:04d}t2400" for i in range(n_test)])
    splits = (["TRAIN"] * n_train + ["CALIBRATION"] * n_cal
              + ["TEST"] * n_test)
    rows = []
    for w, sp in zip(worlds, splits):
        off = float(rng.normal(0, 50))
        for tau in taus:
            for pol in policies:
                bump = 120.0 if pol == "damping_fleet" else 0.0
                y = 10.0 * float(tau) + off + bump + float(rng.normal(0, 20))
                rows.append(dict(world=w, split=sp, tau=float(tau),
                                 target=target, policy=pol, y=y))
    frame = pd.DataFrame(rows)
    frame.insert(0, "row_id", [f"r{i:05d}" for i in range(len(frame))])
    return frame


class SyntheticQuantileModel:
    """``fit(ds)`` + ``predict_quantiles_rows(ds, idx, target, ...)``."""

    def __init__(self, quantiles=DEFAULT_QUANTILES, noise: float = 1.0):
        self.quantiles = tuple(sorted(float(q) for q in quantiles))
        self.noise = float(noise)
        self.fitted_worlds: list[str] = []
        self.fitted_splits: list[str] = []
        self.means: dict = {}
        self.global_mean = 0.0

    def fit(self, ds):
        r = ds.rows
        self.fitted_worlds = sorted(set(r["world"].astype(str)))
        self.fitted_splits = sorted(set(r["split"].astype(str)))
        if {"target", "y"} <= set(r.columns):
            g = r.groupby([r["target"].astype(str),
                           r["tau"].astype(float),
                           r["policy"].astype(str)])["y"].mean()
            self.means = {tuple(k): float(v) for k, v in g.items()}
            self.global_mean = float(r["y"].mean())
        return self

    def predict_quantiles_rows(self, ds, idx, target: str = "delay",
                               calibrated: bool = True) -> dict:
        from scipy.stats import norm
        r = ds.rows.iloc[np.asarray(idx, dtype=int)]
        z = norm.ppf(np.asarray(self.quantiles))
        base = np.array([
            self.means.get((str(target), float(t), str(p)), self.global_mean)
            for t, p in zip(r["tau"].astype(float),
                            r["policy"].astype(str))])
        sigma = self.noise * np.maximum(r["tau"].astype(float), 1.0)
        return {q: base + zi * sigma for q, zi in zip(self.quantiles, z)}


def gaussian_pred(levels, loc, scale, seed: int = 0) -> QuantilePred:
    """A ``QuantilePred`` whose quantiles are Gaussian(loc, scale)."""
    from scipy.stats import norm
    lv = np.asarray(levels, dtype=np.float64)
    z = norm.ppf(lv)
    loc = np.atleast_1d(np.asarray(loc, dtype=np.float64))
    scale = np.atleast_1d(np.asarray(scale, dtype=np.float64))
    q = loc[:, None] + scale[:, None] * z[None, :]
    return QuantilePred(lv, q)
