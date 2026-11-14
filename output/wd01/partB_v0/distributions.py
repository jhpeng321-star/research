"""NumPy distribution metrics for the twohot value heads (evaluation).

The ensemble's predictive distribution is the MIXTURE of its members'
twohot probabilities (mean probability per bin).  All metrics here treat
that pmf over the real-space bin centers as the predictive distribution:

  * crps: CRPS(F, y) = E|X - y| - 0.5 E|X - X'| (exact for a discrete
    distribution; the pairwise term is a fixed K x K matrix);
  * randomized PIT for discrete predictive distributions (u ~ U(0,1)
    within the bin containing the observation; the u-stream is seeded so
    reports are reproducible);
  * quantiles by linear interpolation of the bin CDF, and the central
    10-90% coverage they induce;
  * crps_empirical: NRG-estimator CRPS of an empirical climatology.
"""
from __future__ import annotations

import numpy as np
import torch

from .model import N_BINS, SYM_MAX, inv_symlog


def bin_centers_np() -> np.ndarray:
    """Real-space bin centers [K]."""
    grid = torch.linspace(-SYM_MAX, SYM_MAX, N_BINS, dtype=torch.float64)
    return inv_symlog(grid).numpy().astype(np.float64)


class DistMetrics:
    """Precomputed bin geometry + pairwise-distance matrix (share one)."""

    def __init__(self):
        grid = torch.linspace(-SYM_MAX, SYM_MAX, N_BINS, dtype=torch.float64)
        step = float(grid[1] - grid[0])
        edges_sym = torch.cat(
            [grid - step / 2, torch.tensor([float(grid[-1]) + step / 2])])
        self.centers = inv_symlog(grid).numpy().astype(np.float64)
        self.edges = inv_symlog(edges_sym).numpy().astype(np.float64)
        self.pair = np.abs(self.centers[:, None] - self.centers[None, :])

    def mean_value(self, probs: np.ndarray) -> np.ndarray:
        """Posterior mean per row: probs [N, K] -> [N]."""
        return probs.astype(np.float64) @ self.centers

    def crps(self, probs: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Exact CRPS of the row pmfs against observations y; [N, K],[N] -> [N]."""
        p = probs.astype(np.float64)
        e_abs = np.abs(self.centers[None, :] - np.asarray(y)[:, None])
        term1 = (e_abs * p).sum(1)
        term2 = 0.5 * np.einsum("nk,kl,nl->n", p, self.pair, p)
        return term1 - term2

    def pit(self, probs: np.ndarray, y: np.ndarray,
            seed: int = 20260928) -> np.ndarray:
        """Randomized PIT values in (0, 1) for discrete predictives."""
        p = probs.astype(np.float64)
        cdf = np.cumsum(p, axis=1)
        rows = np.arange(p.shape[0])
        k = np.clip(np.searchsorted(self.centers, y, side="right"), 0,
                    N_BINS - 1)
        lo = np.where(k > 0, cdf[rows, np.clip(k - 1, 0, N_BINS - 1)], 0.0)
        hi = cdf[rows, k]
        u = np.random.default_rng(seed).uniform(0.0, 1.0, size=p.shape[0])
        return np.clip(lo + u * (hi - lo), 1e-6, 1.0 - 1e-6)

    def quantile(self, probs: np.ndarray, q: float) -> np.ndarray:
        """Quantile via linear interpolation of the bin CDF; [N, K] -> [N]."""
        cdf = np.cumsum(probs.astype(np.float64), axis=1)
        out = np.empty(cdf.shape[0])
        for i in range(cdf.shape[0]):
            j = int(np.searchsorted(cdf[i], q, side="left"))
            j = min(max(j, 1), N_BINS - 1)
            f_lo, f_hi = cdf[i, j - 1], cdf[i, j]
            frac = 0.0 if f_hi <= f_lo else (q - f_lo) / (f_hi - f_lo)
            out[i] = self.centers[j - 1] + frac * (self.centers[j]
                                                   - self.centers[j - 1])
        return out

    def central_coverage(self, probs: np.ndarray, y: np.ndarray,
                         lo: float = 0.1, hi: float = 0.9) -> np.ndarray:
        """Boolean: y inside the central (lo, hi) credible interval."""
        return ((y >= self.quantile(probs, lo))
                & (y <= self.quantile(probs, hi)))


def crps_empirical(samples: np.ndarray, y: np.ndarray,
                   block: int = 1024) -> np.ndarray:
    """CRPS of an empirical climatology against y (NRG estimator).

    samples [S,] one climatology group; y [N,] observations.  With S < 2
    it degenerates to mean |x - y| (the caller flags the group as thin).

    Both pairwise-difference terms are accumulated in `block`-row chunks:
    the OOB acceptance table calls this with N = S ~ 20715 TRAIN rows,
    where the naive [N, S] / [S, S] matrices would need ~3.2 GiB each
    (the v0.3.1 auto run OOMed exactly there); chunking is exact (same
    sum, same float64 dtype) and bounded at block*S*8 bytes.
    """
    s = np.asarray(samples, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    out = np.empty(len(y))
    for i in range(0, len(y), block):
        out[i:i + block] = np.abs(
            s[None, :] - y[i:i + block, None]).mean(axis=1)
    if len(s) < 2:
        return out
    acc = 0.0
    for i in range(0, len(s), block):
        acc += float(np.abs(s[i:i + block, None] - s[None, :]).sum())
    return out - 0.5 * acc / (len(s) * len(s))


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) with ties split, as scipy.stats.rankdata."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation without scipy (NaN when degenerate,
    including a constant input)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2:
        return float("nan")
    ra = _average_ranks(a)
    rb = _average_ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")
