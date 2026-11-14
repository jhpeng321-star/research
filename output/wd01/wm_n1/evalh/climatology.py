"""Trivial reference predictor: per-(tau, policy) empirical climatology.

The registration compares every variant against the marginal climatology
(design:199-200).  This class is the trivial reference the tests use; it is
also a valid concrete ``Predictor`` so it doubles as a shape/contract check.

Outcome quantiles are the empirical quantiles of the delay grouped by
``(tau, policy)``.  Rollouts are a persistence forecast from the observed t0
state (``true_pos``/``true_speed`` column 0), the cheapest possible reference.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .protocol import (HORIZONS_S, QUANTILE_LEVELS, OutcomeRow, PredictorBase,
                       RolloutBatch, RolloutRow)


class ClimatologyPredictor(PredictorBase):
    """Empirical-quantile climatology for the outcome heads."""

    name = "climatology"

    def __init__(self, fits: dict | None = None, default: np.ndarray | None = None,
                 levels: np.ndarray | None = None):
        self.quantile_levels = (np.asarray(levels, dtype=float)
                                if levels is not None
                                else QUANTILE_LEVELS)
        self.fits = {self._key(k): self._as_quantiles(v)
                     for k, v in (fits or {}).items()}
        self.default = self._as_quantiles(default) if default is not None \
            else np.zeros(self.quantile_levels.size)

    def _key(self, key) -> tuple:
        tau, policy = key
        return (round(float(tau), 6), str(policy))

    def _as_quantiles(self, values) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        if v.size == self.quantile_levels.size:
            return np.sort(v)
        return np.quantile(v, self.quantile_levels)

    @classmethod
    def fit(cls, rows: Sequence[OutcomeRow],
            levels: np.ndarray | None = None) -> "ClimatologyPredictor":
        lv = QUANTILE_LEVELS if levels is None else np.asarray(levels, float)
        groups: dict = {}
        for r in rows:
            groups.setdefault((round(float(r.tau_s), 6), r.policy), []).append(
                float(r.y_delay_s))
        fits = {k: np.quantile(np.asarray(v, float), lv)
                for k, v in groups.items()}
        everything = np.asarray([float(r.y_delay_s) for r in rows], float)
        default = (np.quantile(everything, lv) if everything.size
                   else np.zeros(lv.size))
        return cls(fits=fits, default=default, levels=lv)

    def predict_quantiles(self, rows: Sequence[OutcomeRow]) -> np.ndarray:
        out = np.empty((len(rows), self.quantile_levels.size), float)
        for i, r in enumerate(rows):
            key = (round(float(r.tau_s), 6), r.policy)
            out[i] = self.fits.get(key, self.default)
        return out

    def sample_rollouts(self, rows: Sequence[RolloutRow],
                        horizons_s: tuple = HORIZONS_S) -> RolloutBatch:
        h = len(horizons_s)
        n = len(rows)
        if n == 0:
            return RolloutBatch(horizons_s, np.zeros((0, h, 0)),
                                np.zeros((0, h, 0)))
        v = rows[0].true_pos.shape[1]
        pos = np.zeros((n, h, v), float)
        speed = np.zeros((n, h, v), float)
        for i, r in enumerate(rows):
            pos[i] = np.repeat(r.true_pos[:1, :], h, axis=0)
            speed[i] = np.repeat(r.true_speed[:1, :], h, axis=0)
        mask = np.ones_like(pos)
        return RolloutBatch(horizons_s, pos, speed, mask=mask,
                            collision=np.zeros(n), ttc_lt3_s=np.zeros(n),
                            hard_brake_s=np.zeros(n),
                            intervention=np.zeros(n))
