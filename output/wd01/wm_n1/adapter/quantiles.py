"""Quantile-curve re-gridding for the adapter (INTERFACES Mismatch 1).

The neural outcome heads emit 19 quantile levels
(``model.constants.QUANTILE_LEVELS`` = 0.05, 0.10, ..., 0.95) while the
combination layer pools at the 7 ``combine.stacking.DEFAULT_QUANTILES``
(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95).  The C variant cannot pool until
the neural curve is mapped onto those 7 levels.

The map is a per-row **linear interpolation of the quantile function**
``Q(p)`` on the source levels, evaluated at the target levels (``numpy.interp``
per row).  It is *not* a resample of quantile values and *not* an average of
levels:

* at a target level that is also a source level the value is returned
  **exactly** (``numpy.interp`` is exact at its nodes), so the 7 shared levels
  are exact;
* between source levels the value is the linear interpolant of the curve;
* outside the source range the curve is clamped to its end values;
* monotonicity is preserved (the source curve is accumulated and the
  interpolant of a non-decreasing function is non-decreasing).
"""
from __future__ import annotations

import numpy as np

from model import constants as C

#: The 19 neural levels, as a sorted float array ``[19]``.
NEURAL_LEVELS = np.asarray(C.QUANTILE_LEVELS, dtype=np.float64)

#: The 7 pooling levels (mirrors ``combine.stacking.DEFAULT_QUANTILES``).
POOL_LEVELS = np.asarray(
    (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95), dtype=np.float64)


def map_quantiles(src_levels, qvals, dst_levels) -> np.ndarray:
    """Map a quantile curve onto ``dst_levels`` by curve interpolation.

    ``src_levels`` is ``[L]``; ``qvals`` is ``[N, L]`` (or ``[L]`` for one
    row); the result is ``[N, len(dst_levels)]``.  Exact at shared levels,
    monotone, and clamped outside the source range.
    """
    src = np.asarray(src_levels, dtype=np.float64).ravel()
    dst = np.asarray(dst_levels, dtype=np.float64).ravel()
    q = np.asarray(qvals, dtype=np.float64)
    if q.ndim == 1:
        q = q[None, :]
    if q.shape[1] != src.size:
        raise ValueError(
            f"map_quantiles: {q.shape[1]} quantile columns vs {src.size} "
            "source levels")
    order = np.argsort(src)
    src, q = src[order], q[:, order]
    q = np.maximum.accumulate(q, axis=1)
    out = np.empty((q.shape[0], dst.size), dtype=np.float64)
    for i in range(q.shape[0]):
        out[i] = np.interp(dst, src, q[i])
    return np.maximum.accumulate(out, axis=1)


def to_pool_levels(q19, levels=POOL_LEVELS) -> np.ndarray:
    """Convenience wrapper: map the 19 neural levels onto ``levels``."""
    return map_quantiles(NEURAL_LEVELS, q19, levels)


def shared_levels(src_levels, dst_levels) -> np.ndarray:
    """Sorted levels present in both grids (used by the mapping tests)."""
    src = set(np.round(np.asarray(src_levels, float), 9).tolist())
    dst = np.asarray(dst_levels, float)
    return np.asarray([d for d in dst if round(float(d), 9) in src],
                      dtype=np.float64)
