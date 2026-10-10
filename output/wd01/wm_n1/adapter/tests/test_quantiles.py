"""19 -> 7 quantile-curve mapping (INTERFACES Mismatch 1)."""
from __future__ import annotations

import numpy as np

from adapter.quantiles import (NEURAL_LEVELS, POOL_LEVELS, map_quantiles,
                               shared_levels)
from combine.stacking import DEFAULT_QUANTILES


def test_levels_are_the_frozen_grids():
    assert NEURAL_LEVELS.size == 19
    assert np.allclose(NEURAL_LEVELS, np.arange(1, 20) / 20.0)
    assert tuple(POOL_LEVELS) == tuple(DEFAULT_QUANTILES)
    assert shared_levels(NEURAL_LEVELS, POOL_LEVELS).size == 7


def test_exact_at_shared_levels_and_monotone():
    rng = np.random.default_rng(0)
    q = np.cumsum(rng.random((5, 19)), axis=1)
    out = map_quantiles(NEURAL_LEVELS, q, POOL_LEVELS)
    assert out.shape == (5, 7)
    assert np.all(np.diff(out, axis=1) >= -1e-12)
    idx = [int(np.argmin(np.abs(NEURAL_LEVELS - lv))) for lv in POOL_LEVELS]
    assert np.allclose(out, q[:, idx])


def test_linear_interpolation_between_levels():
    src = np.array([0.0, 1.0])
    q = np.array([[0.0, 10.0]])
    out = map_quantiles(src, q, np.array([0.25, 0.5, 0.75]))
    assert np.allclose(out[0], [2.5, 5.0, 7.5])


def test_unsorted_levels_and_clamping():
    src = np.array([0.9, 0.1, 0.5])
    q = np.array([[9.0, 1.0, 5.0]])
    out = map_quantiles(src, q, np.array([0.0, 0.3, 1.0]))
    assert np.allclose(out[0], [1.0, 3.0, 9.0])


def test_shape_guard():
    import pytest
    with pytest.raises(ValueError):
        map_quantiles(np.arange(3) / 4.0, np.zeros((2, 5)), np.array([0.5]))
