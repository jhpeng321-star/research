"""Check paired whole-world estimands and exposure weighting, not detector refits."""
import numpy as np

from tools.autonomy_e5.summarize import draw_plan, physical, pooled_rate, result


def test_world_draws_preserve_twelve_registered_cells():
    seeds, groups, samples = draw_plan()
    assert len(seeds) == 228 and len(samples) == 12
    for c, (g, sample) in enumerate(zip(groups, samples)):
        assert sample.shape == (5000, 19)
        assert np.all((seeds[g]-973000) % 12 == c)
        assert np.isin(sample, g).all()
    assert all(np.array_equal(a, b) for a, b in zip(samples, draw_plan()[2]))


def test_paired_constant_effect_cancels_world_variation_and_reverses_ci():
    _, groups, samples = draw_plan()
    baseline = np.arange(228, dtype=float)**2
    effect = baseline-37-baseline
    value, boot = physical(effect, groups, samples)
    assert value == -37 and np.all(boot == -37)
    forward = result('physical_contrast', 'delay', 'ON', 'OFF', 600, 'All', 'All', 228, value, boot, 'veh-s')
    reverse = result('saved_delay', 'delay', 'ON', 'OFF', 600, 'All', 'All', 228, -value, -boot, 'veh-s')
    assert reverse['value'] == -forward['value']
    assert reverse['ci_low'] == -forward['ci_high']
    assert reverse['ci_high'] == -forward['ci_low']


def test_rates_pool_counts_and_resample_exposure_with_world():
    indices = [np.array([0, 1])]
    draws = [np.array([[0, 0], [0, 1], [1, 1]])]
    value, boot = pooled_rate(np.array([1, 9]), np.array([1., 3.]), indices, draws)
    assert value == 2.5  # Not the arithmetic mean of rates (1 + 3)/2.
    np.testing.assert_allclose(boot, [1, 2.5, 3])
