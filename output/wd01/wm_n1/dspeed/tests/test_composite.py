"""A9 step 5: ConcatArray slicing across pack boundaries."""
import numpy as np

from dspeed.composite import ConcatArray


def test_int_and_slice_indexing():
    a = np.arange(6, dtype=np.float32).reshape(3, 2)
    b = (np.arange(6) + 100).astype(np.float32).reshape(3, 2)
    c = ConcatArray([a, b], [0, 3])
    assert c.shape == (6, 2)
    assert c.dtype == np.float32
    assert np.array_equal(c[0], a[0])
    assert np.array_equal(c[3], b[0])
    assert np.array_equal(c[2], a[2])
    assert np.array_equal(c[2:4], np.vstack([a[2:], b[:1]]))


def test_empty_and_full_slices():
    a = np.ones((2, 2), np.float32)
    b = np.full((2, 2), 2.0, np.float32)
    c = ConcatArray([a, b], [0, 2])
    assert c[0:0].shape == (0, 2)
    assert np.array_equal(c[:], np.vstack([a, b]))


def test_tuple_indexing_across_boundary():
    a = np.arange(6, dtype=np.float32).reshape(3, 2)
    b = (np.arange(6) + 100).astype(np.float32).reshape(3, 2)
    c = ConcatArray([a, b], [0, 3])
    full = np.vstack([a, b])
    assert np.array_equal(c[2:4, 1], full[2:4, 1])
    assert c[2:4, 0:1].shape == (2, 1)
    assert c[0, 1] == full[0, 1]
    assert c[2:2, 1].shape == (0,)
    assert np.array_equal(c[2:4, ...], full[2:4, ...])
