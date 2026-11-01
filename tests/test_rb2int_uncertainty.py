import numpy as np
import pytest

from tools.rb2int.uncertainty import PairedWorldBootstrap


def metadata(cells):
    return [dict(seed=900+i, demand=d, penetration=.2) for i, d in enumerate(cells)]


def test_singleton_factual_cell_cannot_be_rescued_by_future_monte_carlo():
    boot = PairedWorldBootstrap(metadata([1900, 1900, 2400]), B=500)
    futures = np.arange(24, dtype=float).reshape(3, 8, 1)
    row = boot.bands(futures.mean(1), futures=futures)[0]
    assert not row['uncertainty_available']
    assert row['low'] is row['high'] is None
    assert row['unavailable_reason'] == 'unreplicated_factual_stratum'


def test_scored_world_composition_stays_fixed_in_every_draw():
    boot = PairedWorldBootstrap(metadata([1900, 1900, 2400, 2400, 2400]), B=500)
    assert np.array_equal(boot.weights[:, :2].sum(1), np.full(500, 2))
    assert np.array_equal(boot.weights[:, 2:].sum(1), np.full(500, 3))
    rows = boot.bands(np.array([[1, -1], [2, -2], [3, -3], [4, -4], [5, -5]]))
    assert all(r['uncertainty_available'] for r in rows)
    assert rows[0]['low'] == -rows[1]['high']
    assert rows[0]['high'] == -rows[1]['low']


def test_missing_world_is_not_silently_replaced_by_an_unscored_peer():
    boot = PairedWorldBootstrap(metadata([1900, 1900, 2400, 2400]), B=500)
    values = np.array([[1, 1], [2, 2], [3, 3], [4, np.nan]])
    rows = boot.bands(values)
    assert rows[0]['uncertainty_available']
    assert not rows[1]['uncertainty_available']
    assert rows[1]['unavailable_reason'] == 'missing_or_nonfinite_world_support'


def test_identical_paired_predictions_do_not_create_independent_arm_noise():
    boot = PairedWorldBootstrap(metadata([1900, 1900]), B=500)
    futures = np.tile(np.arange(8, dtype=float)[None, :, None], (2, 1, 2))
    rows = boot.bands(futures.mean(1), futures=futures,
                     transform=lambda x: x[:, :1]-x[:, 1:])
    assert rows[0]['mean'] == 0
    assert rows[0]['unavailable_reason'] == 'zero_estimable_variance'


def test_paired_nonlinear_rate_distance_uses_difference_of_group_distances():
    boot = PairedWorldBootstrap(metadata([1900, 1900]), B=500)
    # candidate, baseline, real; mean distance difference = |.5-.5|-|1-.5|.
    values = np.array([[0, 1, 0], [1, 1, 1]])
    row = boot.bands(values, transform=lambda x:
                     np.abs(x[:, :1]-x[:, 2:])-np.abs(x[:, 1:2]-x[:, 2:]))[0]
    assert row['mean'] == -.5


def test_repeated_arm_rows_and_inconsistent_future_means_are_rejected():
    with pytest.raises(ValueError, match='one factual cluster'):
        PairedWorldBootstrap([metadata([1900])[0]]*2, B=500)
    boot = PairedWorldBootstrap(metadata([1900, 1900]), B=500)
    with pytest.raises(ValueError, match='future means differ'):
        boot.bands(np.zeros((2, 1)), futures=np.ones((2, 8, 1)))
