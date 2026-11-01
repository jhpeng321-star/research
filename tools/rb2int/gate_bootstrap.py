"""Strict feature-family adapter for the registered distributional gate scorer."""
from collections import Counter, defaultdict

import numpy as np

from tools.rb2int.uncertainty import PairedWorldBootstrap


class GateBootstrap(PairedWorldBootstrap):
    seed = 20261101
    method = 'scored-world-only demand/penetration paired max-stat; singleton and missing-support guarded'

    def __init__(self, real, B=10000):
        meta = {r['seed']: dict(seed=r['seed'], demand=r['demand'], penetration=r['penetration']) for r in real}
        super().__init__([meta[s] for s in sorted(meta)], B=B, seed=self.seed)
        self.seeds = self.worlds
        self.index = {s: i for i, s in enumerate(self.seeds)}

    def supports(self, worlds):
        selected = set(worlds)
        by = defaultdict(set)
        for seed, cell in zip(self.seeds, self.cells):
            by[cell].add(seed)
        used = {cell for seed, cell in zip(self.seeds, self.cells) if seed in selected}
        return bool(used) and all(len(by[cell]) >= 2 and by[cell] <= selected for cell in used)

    def bands(self, features, alpha=.05, chunk_size=128):
        if not features:
            return []
        estimates = np.full(len(features), np.nan)
        sd = np.zeros(len(features))
        valid = np.zeros(len(features), bool)
        counts = np.zeros(len(features), int)
        reasons = ['missing_or_nonfinite_world_support']*len(features)
        maximum = np.zeros(self.B)
        for start in range(0, len(features), chunk_size):
            stop = min(start+chunk_size, len(features))
            matrix = np.zeros((len(self.seeds), stop-start))
            present = np.zeros_like(matrix, dtype=bool)
            deviations = defaultdict(dict)
            for local, pairs in enumerate(features[start:stop]):
                j = start+local
                by = defaultdict(list)
                support = True
                for pair in pairs:
                    if pair[0] not in self.index:
                        raise ValueError('unscored world cannot contribute bootstrap weights')
                    value = pair[1]
                    if value is None or not np.isfinite(value):
                        support = False
                    by[pair[0]].append(pair)
                counts[j] = len(by)
                cells = Counter(self.cells[self.index[s]] for s in by)
                replicated = bool(cells) and all(n >= 2 for n in cells.values())
                complete_strata = self.supports(by)
                if not replicated:
                    reasons[j] = 'unreplicated_factual_stratum'
                elif not complete_strata:
                    reasons[j] = 'incomplete_scored_stratum_support'
                elif support:
                    reasons[j] = 'zero_estimable_variance'
                values = []
                for seed, rows in by.items():
                    i = self.index[seed]
                    vals = [p[1] for p in rows if p[1] is not None and np.isfinite(p[1])]
                    if len(vals) != len(rows):
                        continue
                    value = float(np.mean(vals))
                    matrix[i, local] = value
                    present[i, local] = True
                    values.append(value)
                    future_rows = [np.asarray(p[2], float) for p in rows if len(p) == 3]
                    if future_rows:
                        if len(future_rows) != len(rows) or any(v.shape != (8,) or not np.isfinite(v).all() for v in future_rows):
                            raise ValueError('incomplete paired K8 feature support')
                        future = np.mean(future_rows, axis=0)
                        if not np.isclose(future.mean(), value):
                            raise ValueError('future feature mean differs from point')
                        deviations[i][local] = future-value
                estimates[j] = float(np.mean(values)) if values else np.nan
                valid[j] = support and replicated and complete_strata and bool(values)
            denominator = self.weights @ present.astype(float)
            draws = np.divide(self.weights @ matrix, denominator,
                out=np.full((self.B, stop-start), np.nan), where=denominator > 0)
            for i, columns in deviations.items():
                delta = np.zeros((8, stop-start))
                for local, value in columns.items():
                    delta[:, local] = value
                draws += np.divide((self.future_weights[:, i] @ delta)*self.weights[:, i:i+1],
                    denominator, out=np.zeros_like(draws), where=denominator > 0)
            std = np.nanstd(draws-estimates[None,start:stop], axis=0, ddof=1)
            sd[start:stop] = std
            tolerance = 64*np.finfo(float).eps*np.maximum(1.,np.abs(estimates[start:stop]))
            good = valid[start:stop] & np.isfinite(draws).all(0) & (std > tolerance)
            valid[start:stop] = good
            if good.any():
                maximum = np.maximum(maximum, np.max(np.abs(
                    (draws[:, good]-estimates[start:stop][good])/std[good]), axis=1))
        critical = float(np.quantile(maximum, 1-alpha)) if valid.any() else None
        return [dict(mean=float(estimates[j]) if np.isfinite(estimates[j]) else None,
            worlds=int(counts[j]), low=float(estimates[j]-critical*sd[j]) if valid[j] else None,
            high=float(estimates[j]+critical*sd[j]) if valid[j] else None,
            uncertainty_available=bool(valid[j]), unavailable_reason=None if valid[j] else reasons[j],
            maxstat_critical=critical) for j in range(len(features))]
