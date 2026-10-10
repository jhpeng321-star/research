"""Synthetic quantification of paired-dV interval coverage by construction.

The scorer's historical paired-dV distribution for a per-arm marginal predictor
is the comonotone ``q_a - q_b``.  :mod:`evalh.scorer.metrics_loop` now also
offers the ``"independent"`` construction (sample each marginal, take the
difference quantiles).  This module measures, on synthetic data with a KNOWN
joint, how much 10-90 coverage each construction actually attains.

Two regimes:

* ``rho = 0`` -- the true arms are independent, which is exactly the modelling
  assumption of the independent construction.  Comonotone subtraction
  under-covers badly; the sampled construction is close to nominal.
* ``rho > 0`` -- the true arms are positively correlated.  Comonotone becomes
  better and the independent construction is deliberately conservative (wider
  intervals), which is the price of not having a joint from the model.

No data or checkpoint is read: this is pure numerics.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np

from evalh.metrics import crps_of_quantiles
from evalh.protocol import QUANTILE_LEVELS
from evalh.scorer.metrics_loop import (DV_METHOD_COMONOTONE,
                                       DV_METHOD_INDEPENDENT,
                                       independent_difference_quantiles)

_NORM = NormalDist()


def _normal_marginal(mu: float, sigma: float, levels) -> np.ndarray:
    return np.asarray([mu + sigma * _NORM.inv_cdf(float(v)) for v in levels])


def coverage_experiment(*, n_pairs: int = 4000, seed: int = 7, rho: float = 0.0,
                        n_samples: int = 4096, levels=QUANTILE_LEVELS) -> dict:
    """Coverage/CRPS of comonotone vs independent paired-dV on a known joint."""
    lv = np.asarray(levels, dtype=float)
    lo_i, hi_i = int(np.argmin(np.abs(lv - 0.10))), int(
        np.argmin(np.abs(lv - 0.90)))
    rng = np.random.default_rng(seed)
    cov = {DV_METHOD_COMONOTONE: [], DV_METHOD_INDEPENDENT: []}
    crps = {DV_METHOD_COMONOTONE: [], DV_METHOD_INDEPENDENT: []}
    width = {DV_METHOD_COMONOTONE: [], DV_METHOD_INDEPENDENT: []}
    for i in range(int(n_pairs)):
        mu_a, mu_b = rng.normal(100.0, 30.0), rng.normal(100.0, 30.0)
        sa, sb = float(rng.uniform(5.0, 25.0)), float(rng.uniform(5.0, 25.0))
        qa = _normal_marginal(mu_a, sa, lv)
        qb = _normal_marginal(mu_b, sb, lv)
        z, w = rng.normal(), rng.normal()
        # joint draw; the marginals are exactly qa/qb by construction.
        a = mu_a + sa * z
        b = mu_b + sb * (float(rho) * z + np.sqrt(1.0 - float(rho) ** 2) * w)
        dy = float(a - b)
        dv_com = qa - qb
        dv_ind = independent_difference_quantiles(
            qa, qb, levels=lv, n_samples=n_samples, seed=0, key=i)
        for name, dv in ((DV_METHOD_COMONOTONE, dv_com),
                         (DV_METHOD_INDEPENDENT, dv_ind)):
            cov[name].append(float(dv[lo_i] <= dy <= dv[hi_i]))
            crps[name].append(float(crps_of_quantiles(dv, dy, lv)))
            width[name].append(float(dv[hi_i] - dv[lo_i]))
    out = dict(
        n_pairs=int(n_pairs), seed=int(seed), rho=float(rho),
        n_samples=int(n_samples), levels=[float(v) for v in lv],
        methods=list((DV_METHOD_COMONOTONE, DV_METHOD_INDEPENDENT)),
        coverage={k: float(np.mean(v)) for k, v in cov.items()},
        crps={k: float(np.mean(v)) for k, v in crps.items()},
        mean_interval_width={k: float(np.mean(v)) for k, v in width.items()})
    return out


def run(report_path=None, **kwargs) -> dict:
    rho0 = coverage_experiment(rho=0.0, **kwargs)
    rho5 = coverage_experiment(rho=0.5, **kwargs)
    doc = dict(schema="wm_n1_dv_method_coverage/1", independent=rho0,
               correlated_rho_0p5=rho5)
    if report_path is not None:
        import json
        from pathlib import Path

        Path(report_path).write_text(
            json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return doc


if __name__ == "__main__":                                  # pragma: no cover
    import json

    print(json.dumps(run(), indent=1, sort_keys=True))
