"""Shared observable-only, fixed-grid secondary estimands (no simulator)."""
import numpy as np

EDGES = np.arange(-1000., 501., 100.)


def merge_queue(s, v, present):
    return ((s >= -200.) & (s <= 75.44) & (v < 5.) & present).sum(-1)


def fields(s, v, present):
    """[arm,time,slot] -> [arm,10s-bin,100m-cell]; left boundaries."""
    assert s.shape == v.shape == present.shape and s.shape[1] % 20 == 0
    counts, sums = [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        mask = present & (s >= lo) & (s < hi)
        counts.append(mask.sum(-1).reshape(s.shape[0], -1, 20).sum(-1))
        sums.append(np.where(mask, v, 0.).sum(-1).reshape(s.shape[0], -1, 20).sum(-1))
    counts, sums = np.stack(counts, -1), np.stack(sums, -1)
    speed = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts > 0)
    return dict(density=(counts / 20. * 10.).tolist(), speed=speed.tolist(),
                occupied=(counts > 0).tolist())
