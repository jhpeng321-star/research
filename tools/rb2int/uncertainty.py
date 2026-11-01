"""Paired fixed-cell world uncertainty with explicit replication/support guards.

Saved-ensemble scores condition on K8. Forecast-mean quantities can additionally
resample future indices, shared across every arm/model/metric in the world.
Never let finite-K variation rescue an unreplicated factual stratum.
"""
from collections import Counter, defaultdict

import numpy as np


class PairedWorldBootstrap:
    def __init__(self, metadata, *, B=10000, seed=20261101):
        self.metadata = list(metadata)
        self.worlds = [m['seed'] for m in self.metadata]
        if len(set(self.worlds)) != len(self.worlds):
            raise ValueError('a world is one factual cluster, never an arm/future row')
        self.B = B
        self.cells = [(m['demand'], m['penetration']) for m in self.metadata]
        counts = Counter(self.cells)
        self.replicated = bool(counts) and all(n >= 2 for n in counts.values())
        self.stratum_counts = [dict(demand=d, penetration=p, worlds=n)
                               for (d, p), n in sorted(counts.items())]
        self.weights = np.zeros((B, len(metadata)))
        rng = np.random.default_rng(seed)
        groups = defaultdict(list)
        for i, cell in enumerate(self.cells):
            groups[cell].append(i)
        for ix in groups.values():
            self.weights[:, ix] = rng.multinomial(
                len(ix), np.full(len(ix), 1/len(ix)), size=B)
        self.future_weights = rng.multinomial(
            8, np.full(8, 1/8), size=(B, len(metadata)))/8

    def draws(self, values, *, futures=None):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[0] != len(self.worlds):
            raise ValueError('one value vector per registered scored world required')
        if not len(self.worlds):
            return np.full(values.shape[1], np.nan), np.full((self.B, values.shape[1]), np.nan)
        point = values.mean(0)
        draws = self.weights @ values/len(self.worlds)
        if futures is not None:
            futures = np.asarray(futures, dtype=float)
            if futures.shape != (len(self.worlds), 8, values.shape[1]):
                raise ValueError('future index coupling must retain [world,8,feature]')
            if not np.allclose(futures.mean(1), values, rtol=1e-10, atol=1e-10, equal_nan=True):
                raise ValueError('future means differ from reported forecast values')
            for i in range(len(self.worlds)):
                delta = self.future_weights[:, i] @ (futures[i]-values[i])
                draws += delta*self.weights[:, i:i+1]/len(self.worlds)
        return point, draws

    def bands(self, values, *, futures=None, transform=None, alpha=.05):
        point, draws = self.draws(values, futures=futures)
        if transform is not None:
            point = np.asarray(transform(point[None]), float)[0]
            draws = np.asarray(transform(draws), float)
        if point.ndim != 1 or draws.shape != (self.B, len(point)):
            raise ValueError('transform must preserve [draw,feature] convention')
        finite = np.isfinite(draws).all(0) & np.isfinite(point)
        sd = np.std(draws, axis=0, ddof=1)
        valid = finite & (sd > 1e-14) & self.replicated
        # Center at each estimate; the simultaneous family uses one paired draw
        # for every column. Nonlinear transformations retain their paired shape.
        critical = None
        if valid.any():
            maxstat = np.max(np.abs((draws[:, valid]-point[valid])/sd[valid]), axis=1)
            critical = float(np.quantile(maxstat, 1-alpha))
        result = []
        for j, value in enumerate(point):
            reason = ('unreplicated_factual_stratum' if not self.replicated else
                      'missing_or_nonfinite_world_support' if not finite[j] else
                      'zero_estimable_variance' if not valid[j] else None)
            result.append(dict(mean=float(value) if np.isfinite(value) else None,
                low=float(value-critical*sd[j]) if valid[j] else None,
                high=float(value+critical*sd[j]) if valid[j] else None,
                uncertainty_available=bool(valid[j]), unavailable_reason=reason,
                worlds=len(self.worlds), family_size=len(point),
                maxstat_critical=critical,
                uncertainty=('paired world + within-world future bootstrap'
                             if futures is not None else 'paired world; conditional on saved K8')))
        return result
