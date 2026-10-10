"""Small statistics helpers shared by the report, comparison and ranking modules."""

from __future__ import annotations

import math
import random
import statistics
from typing import Callable, Mapping, Sequence

DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 0


def quantile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated quantile ``q`` in [0, 1] of unsorted ``values``."""
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


def summarize(values: Sequence[float]) -> dict:
    """Median / IQR / p90 / population sd of ``values``."""
    xs = [float(v) for v in values]
    if not xs:
        return {"n": 0, "median": None, "q1": None, "q3": None, "iqr": None,
                "p90": None, "sd": None}
    q1, med, q3 = (quantile(xs, q) for q in (0.25, 0.5, 0.75))
    mean = sum(xs) / len(xs)
    sd = math.sqrt(sum((v - mean) ** 2 for v in xs) / len(xs))
    return {"n": len(xs), "median": med, "q1": q1, "q3": q3, "iqr": q3 - q1,
            "p90": quantile(xs, 0.9), "sd": sd}


def bootstrap_ci(rows: Sequence, statistic: Callable[[Sequence], float], *,
                 n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED) -> dict:
    """Percentile 95% CI of ``statistic`` over iid resamples of ``rows``."""
    point = statistic(rows)
    rng = random.Random(seed)
    samples = [statistic([rows[rng.randrange(len(rows))] for _ in range(len(rows))])
               for _ in range(n_boot)]
    return {"point": point, "lo": quantile(samples, 0.025), "hi": quantile(samples, 0.975)}


def world_bootstrap_median(grouped: Mapping[str, Sequence[float]], *,
                           n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED) -> dict:
    """Median of the pooled per-world values with a CI from resampling whole worlds."""
    worlds = sorted(grouped)
    pooled = [d for world in worlds for d in grouped[world]]
    point = statistics.median(pooled) if pooled else None
    if not pooled or n_boot <= 0:
        return {"point": point, "lo": None, "hi": None, "n_worlds": len(worlds),
                "n_pairs": len(pooled), "n_boot": 0}
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        pick = rng.choices(worlds, k=len(worlds))
        values = [d for world in pick for d in grouped[world]]
        if values:
            samples.append(statistics.median(values))
    return {"point": point, "lo": quantile(samples, 0.025), "hi": quantile(samples, 0.975),
            "n_worlds": len(worlds), "n_pairs": len(pooled), "n_boot": len(samples)}
