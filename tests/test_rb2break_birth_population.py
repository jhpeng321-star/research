import numpy as np

from tools.rb2break.birth_population import acceleration_crps, acceleration_support, births, fit_weights


def test_observed_births_not_junction_reappearance_or_owner_assignment():
    kin = np.zeros((3, 2, 7), dtype=np.float32)
    kin[..., 1] = 1
    kin[1:, 0, 0] = -2640.57
    kin[1:, 1, 0] = -10  # same clock/lane, but a reappearing road car
    history = dict(time_s=np.array([180., 180.5, 181.]), kin=kin,
        vid_index=np.array([[-1, -1], [17, 18], [17, 18]]),
        mask=np.array([[False, False], [True, True], [True, True]]))
    start = dict(past_entries=[dict(time=180.5, lane=0, s_m=-2640.57, is_cav=False)])
    result, skipped = births(start, history)
    assert set(result) == {17} and not result[17]['is_cav'] and skipped == 1


def test_mixture_fit_recovers_frequency_on_identifiable_support():
    likelihood = np.array([[0., -1000.]]*3+[[-1000., 0.]])
    weights, _ = fit_weights(likelihood, np.array([.5, .5]))
    np.testing.assert_allclose(weights, [.75, .25], rtol=0, atol=1e-12)


def test_weighted_crps_matches_independent_pairwise_definition():
    predictions = np.array([[3., -1., 2.], [2., 2., 8.]])
    target = np.array([4., 3.])
    weights = np.array([.2, .3, .5])
    actual = acceleration_crps(predictions, target, weights)
    expected = np.abs(predictions-target[:, None])@weights-.5*(
        np.abs(predictions[:, :, None]-predictions[:, None, :])*weights[None, :, None]
        *weights[None, None, :]).sum((1, 2))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_acceleration_envelope_rejects_impossible_profiles_but_allows_roundoff():
    valid, lower = acceleration_support(np.array([0., 10.]), np.array([.5, 10.5]),
        np.array([.5, .5]), np.array([.8, 1., 1.2]))
    assert 0.999 < lower <= 1.
    assert valid.tolist() == [False, True, True]
