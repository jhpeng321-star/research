"""Breakdown rule at the lane drop (registered in WM_QUAL_GATE.md section 3 before any result).

Rule (inherited from the S-PAIRCRIT/LRDIAG detector family, adapted to wz3to2): fit two ordered
Gaussian regimes to log approach-zone speed (10 s bins, zone [drop-300 m, drop], all lanes) on
the FITTING tapes only, deterministic initialisation at the 25th/75th percentiles, fixed EM
iterations; the boundary is the equal-posterior point between the two means.  The fitted
constants are written (and hashed) before any WM prediction is scored.  No manual speed cutoff.

Event: breakdown occurs at the first bin b such that bins b .. b+P-1 (P = 3, i.e. 30 s) are all
below the boundary (an empty zone counts as free flow).  ``occurred(H)``: onset bin starts and its
P-bin persistence window ends within [t0, t0+H].  A start whose last P OBSERVED pre-t0 bins are
below the boundary is ``broken_at_start`` (shared by truth and WM) and reported separately with
the recovery event instead (it cannot "break down").
The rule is degenerate (=> breakdown criteria INCONCLUSIVE) when the regimes are not separated:
|mu_hi - mu_lo| < 0.15 (log m/s) or a weight < 0.03.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np

PERSIST_BINS = 3
BIN_S = 10.0
EM_ITERS = 200
MIN_SEPARATION = 0.15
MIN_WEIGHT = 0.03
SPEED_FLOOR = 0.1          # m/s, log(max(v, floor))


@dataclass(frozen=True)
class BreakdownRule:
    mu_lo: float
    sd_lo: float
    w_lo: float
    mu_hi: float
    sd_hi: float
    w_hi: float
    boundary_log: float
    boundary_mps: float
    n_fit: int
    degenerate: bool
    source: str = ""

    def record(self) -> dict:
        d = asdict(self)
        d["sha256"] = hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()
        return d


def _norm_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


def fit_rule(speeds_mps, *, source: str = "") -> BreakdownRule:
    v = np.asarray(speeds_mps, np.float64)
    v = v[np.isfinite(v)]
    x = np.log(np.maximum(v, SPEED_FLOOR))
    if x.size < 20:
        return BreakdownRule(*(float("nan"),) * 8, n_fit=int(x.size), degenerate=True, source=source)
    q25, q75 = np.percentile(x, [25, 75])
    sd0 = max(float(np.std(x)) / 2.0, 1e-3)
    mu = np.array([q25, q75], np.float64)
    sd = np.array([sd0, sd0])
    w = np.array([0.5, 0.5])
    for _ in range(EM_ITERS):
        p = np.stack([w[j] * _norm_pdf(x, mu[j], sd[j]) for j in range(2)])
        tot = np.maximum(p.sum(0), 1e-300)
        r = p / tot
        nk = np.maximum(r.sum(1), 1e-12)
        w = nk / x.size
        mu = (r * x).sum(1) / nk
        sd = np.sqrt(np.maximum((r * (x - mu[:, None]) ** 2).sum(1) / nk, 1e-6))
    lo, hi = (0, 1) if mu[0] <= mu[1] else (1, 0)
    boundary = _equal_posterior(mu[lo], sd[lo], w[lo], mu[hi], sd[hi], w[hi])
    degenerate = bool(abs(mu[hi] - mu[lo]) < MIN_SEPARATION or min(w) < MIN_WEIGHT
                      or not math.isfinite(boundary))
    return BreakdownRule(float(mu[lo]), float(sd[lo]), float(w[lo]), float(mu[hi]), float(sd[hi]),
                         float(w[hi]), float(boundary), float(math.exp(boundary)) if math.isfinite(boundary)
                         else float("nan"), int(x.size), degenerate, source)


def _equal_posterior(m1, s1, w1, m2, s2, w2) -> float:
    """Point between m1 < m2 where w1 N(m1,s1) = w2 N(m2,s2)."""
    a = 1 / (2 * s2 ** 2) - 1 / (2 * s1 ** 2)
    b = m1 / s1 ** 2 - m2 / s2 ** 2
    c = (m2 ** 2) / (2 * s2 ** 2) - (m1 ** 2) / (2 * s1 ** 2) + math.log((w1 * s2) / (w2 * s1))
    # w1 N1 = w2 N2  <=>  a x^2 + b x + c = 0
    if abs(a) < 1e-12:
        roots = [-c / b] if abs(b) > 1e-12 else []
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            return float("nan")
        r = math.sqrt(disc)
        roots = [(-b - r) / (2 * a), (-b + r) / (2 * a)]
    inside = [z for z in roots if m1 <= z <= m2]
    if inside:
        return float(inside[0])
    return float(min(roots, key=lambda z: abs(z - 0.5 * (m1 + m2)))) if roots else float("nan")


@dataclass(frozen=True)
class Event:
    broken_at_start: bool
    onset_s: float          # seconds after t0 of the onset bin start (inf = none within data)
    n_bins: int

    def occurred(self, horizon_s: float) -> bool:
        if self.broken_at_start:
            return False
        return bool(self.onset_s + PERSIST_BINS * BIN_S <= horizon_s + 1e-9)


def _low(bins, rule):
    z = np.asarray(bins, np.float64)
    return np.where(np.isfinite(z), np.log(np.maximum(z, SPEED_FLOOR)) < rule.boundary_log, False)


def event(zone_bins: np.ndarray, rule: BreakdownRule, *, broken_at_start: bool = False) -> Event:
    """Onset of the first P-bin low run after t0.  ``broken_at_start`` comes from the shared,
    observed pre-t0 bins (:func:`start_class`), never from the predicted future."""
    low = _low(zone_bins, rule)
    n = len(low)
    for b in range(0, n - PERSIST_BINS + 1):
        if low[b:b + PERSIST_BINS].all():
            return Event(bool(broken_at_start), b * BIN_S, n)
    return Event(bool(broken_at_start), float("inf"), n)


def start_class(pre_bins, rule: BreakdownRule) -> bool:
    """True = broken at start: the last P observed 10 s bins before t0 are all below the boundary."""
    if pre_bins is None or len(pre_bins) < PERSIST_BINS:
        return False
    return bool(_low(np.asarray(pre_bins)[-PERSIST_BINS:], rule).all())


def recovered(zone_bins: np.ndarray, rule: BreakdownRule, horizon_s: float) -> bool:
    """For broken starts: a P-bin run at/above the boundary completes within the horizon."""
    high = ~_low(zone_bins, rule)          # an empty zone counts as free flow
    n =min(len(high), int(round(horizon_s / BIN_S)))
    return any(high[b:b + PERSIST_BINS].all() for b in range(0, n - PERSIST_BINS + 1))
