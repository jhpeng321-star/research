"""Proper scores and forecast calibration; no fitting or behavioural thresholds."""

from collections import defaultdict

import numpy as np


def _sample(values):
    x = np.asarray(values, float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite one-dimensional ensemble")
    return x


def crps(values, outcome, *, fair=False):
    """Empirical CRPS; fair form estimates the latent-distribution score.

    Fair CRPS assumes exchangeable independent draws within an action. Common
    random numbers across actions do not violate that within-action assumption.
    """
    x = np.sort(_sample(values))
    if not np.isfinite(outcome):
        raise ValueError("Outcome must be finite")
    n = len(x)
    if fair and n < 2:
        return None
    pair_half = float(np.dot(2 * np.arange(n) - n + 1, x))
    return float(np.mean(np.abs(x - outcome)) - pair_half / (n * (n-1) if fair else n*n))


def brier(probability, outcome):
    if not np.isfinite(probability) or not 0 <= probability <= 1 or outcome not in (0, 1):
        raise ValueError("Brier score requires a probability and a binary outcome")
    return float((probability - outcome) ** 2)


def interval_coverage(values, outcome, levels=(.5, .8, .9)):
    """Central interpolated empirical intervals plus inclusive ensemble range.

    Nominal central levels describe empirical quantiles, not exact finite-K
    exchangeability coverage. Only continuous-outcome range coverage has the
    exact (K-1)/(K+1) reference; ties can increase inclusive coverage.
    """
    x = _sample(values)
    if not np.isfinite(outcome):
        raise ValueError("Outcome must be finite")
    rows = []
    for level in levels:
        if not 0 < level < 1:
            raise ValueError("Interval levels must be in (0,1)")
        low, high = np.quantile(x, [(1-level)/2, (1+level)/2])
        rows.append(dict(interval="central", nominal_level=float(level), low=float(low),
                         high=float(high), covered=int(low <= outcome <= high),
                         width=float(high-low), continuous_exchangeable_reference=None))
    low, high = float(x.min()), float(x.max())
    rows.append(dict(interval="range", nominal_level=None, low=low, high=high,
                     covered=int(low <= outcome <= high), width=high-low,
                     continuous_exchangeable_reference=(len(x)-1)/(len(x)+1)))
    return rows


class WorldIntervals:
    """Pointwise percentile CIs; all arms, pairs and futures stay in each world.

    Composition is fixed by split/demand/penetration. These are descriptive
    whole-world CIs, NOT simultaneous qualification bands or model-error bounds.
    No uncertainty is asserted for a single world or zero bootstrap variation.
    """

    def __init__(self, metadata, repetitions=10000, seed=20261031):
        if repetitions < 2:
            raise ValueError("At least two bootstrap repetitions are required")
        self.worlds = sorted(metadata)
        self.index = {w: i for i, w in enumerate(self.worlds)}
        self.weights = np.zeros((repetitions, len(self.worlds)), dtype=float)
        strata = defaultdict(list)
        for w, meta in metadata.items():
            strata[tuple(meta)].append(self.index[w])
        rng = np.random.default_rng(seed)
        for ids in strata.values():
            self.weights[:, ids] = rng.multinomial(len(ids), [1/len(ids)] * len(ids),
                                                 size=repetitions)

    def mean(self, rows, field):
        """Row-weighted mean with cluster resampling and available-case counts."""
        total = np.zeros(len(self.worlds))
        count = np.zeros(len(self.worlds))
        for row in rows:
            v = row.get(field)
            if v is not None and np.isfinite(v):
                i = self.index[row["seed"]]
                total[i] += v
                count[i] += 1
        worlds = int(np.sum(count > 0))
        result = dict(mean=float(total.sum()/count.sum()) if count.sum() else None,
                      count=int(count.sum()), worlds=worlds, low=None, high=None,
                      uncertainty_available=False)
        if worlds < 2:
            return result
        denominator = self.weights @ count
        samples = np.divide(self.weights @ total, denominator,
                            out=np.full(len(denominator), np.nan), where=denominator > 0)
        # Empty-bin resamples are omitted and explicitly counted, never set to zero.
        samples = samples[np.isfinite(samples)]
        result["valid_bootstrap_repetitions"] = len(samples)
        numerical_resolution = 32 * np.finfo(float).eps * max(1., float(np.max(np.abs(samples)))) \
            if len(samples) else 0.
        if len(samples) > 1 and np.ptp(samples) > numerical_resolution:
            result.update(low=float(np.quantile(samples, .025)),
                          high=float(np.quantile(samples, .975)), uncertainty_available=True)
        return result


def reliability(rows, intervals, bins=10):
    """Fixed equal-width probability bins; 1.0 goes in the last bin.

    Each row needs seed, probability, outcome. Unknowns must be counted by caller
    as unscorable, never recoded as negative labels. Empty bins remain explicit.
    """
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("Positive integer bin count required")
    grouped = defaultdict(list)
    for row in rows:
        brier(row["probability"], row["outcome"])
        grouped[min(int(row["probability"] * bins), bins-1)].append(row)
    result = []
    for i in range(bins):
        subset = grouped[i]
        forecast = intervals.mean(subset, "probability")
        observed = intervals.mean(subset, "outcome")
        result.append(dict(bin=i, bin_low=i/bins, bin_high=(i+1)/bins,
            count=observed["count"], worlds=observed["worlds"],
            mean_probability=forecast["mean"], observed_rate=observed["mean"],
            observed_low=observed["low"], observed_high=observed["high"],
            uncertainty_available=observed["uncertainty_available"],
            valid_bootstrap_repetitions=observed.get("valid_bootstrap_repetitions")))
    return result
