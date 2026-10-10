"""Synthetic predictors for the evalh CPU tests (no real data, no SUMO).

``OraclePredictor`` centres the outcome quantiles on the source's noiseless
``truth_center`` with a Gaussian spread, and rolls out the true trajectory
plus a configurable bias.  It is a test double, not a model.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np

from evalh.protocol import (HORIZONS_S, QUANTILE_LEVELS, PredictorBase,
                            RolloutBatch)


class OraclePredictor(PredictorBase):
    def __init__(self, name: str = "oracle", sigma: float = 1.0,
                 bias: float = 0.0, rollout_bias: float = 0.0,
                 rollout_speed_bias: float = 0.0, collision: float = 0.0,
                 policy_bias: dict | None = None, levels=None):
        self.name = name
        self.sigma = float(sigma)
        self.bias = float(bias)
        self.rollout_bias = float(rollout_bias)
        self.rollout_speed_bias = float(rollout_speed_bias)
        self.collision = float(collision)
        self.policy_bias = dict(policy_bias or {})
        self.quantile_levels = (np.asarray(levels, float)
                                if levels is not None else QUANTILE_LEVELS)
        nd = NormalDist()
        self._z = np.asarray([nd.inv_cdf(float(lv))
                              for lv in self.quantile_levels], float)

    def _center(self, row) -> float:
        base = (float(row.features["truth_center"])
                if "truth_center" in row.features else float(row.y_delay_s))
        return base + self.policy_bias.get(row.policy, 0.0)

    def predict_quantiles(self, rows) -> np.ndarray:
        c = np.asarray([self._center(r) for r in rows], float) + self.bias
        return c[:, None] + self.sigma * self._z[None, :]

    def sample_rollouts(self, rows, horizons_s=HORIZONS_S) -> RolloutBatch:
        if not rows:
            return RolloutBatch(tuple(horizons_s), np.zeros((0, len(horizons_s), 0)),
                                np.zeros((0, len(horizons_s), 0)))
        pos = np.stack([np.asarray(r.true_pos, float) for r in rows])
        speed = np.stack([np.asarray(r.true_speed, float) for r in rows])
        pos = pos + self.rollout_bias
        speed = speed + self.rollout_speed_bias
        n = len(rows)
        return RolloutBatch(
            tuple(horizons_s), pos, speed,
            mask=np.ones_like(pos),
            collision=np.full(n, self.collision),
            ttc_lt3_s=np.zeros(n), hard_brake_s=np.zeros(n),
            intervention=np.zeros(n))


def variant() -> OraclePredictor:
    """A good planner variant for the end-to-end test."""
    return OraclePredictor(name="variant", sigma=0.5, bias=0.0,
                           rollout_bias=0.0)


def hybrid() -> OraclePredictor:
    """A weaker hybrid reference for the end-to-end test.

    The per-policy ``native`` bias breaks the damping-native / pi-native /
    wall-native contrasts (a uniform bias would cancel in the difference).
    """
    return OraclePredictor(name="hybrid", sigma=2.0,
                           policy_bias={"native": 4.0},
                           rollout_bias=2.0, collision=0.0)
