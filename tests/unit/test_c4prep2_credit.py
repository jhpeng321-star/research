"""Registered-baseline guard prevents real coop erasure and negative net own credit."""
import pytest

from wms.control.policy14 import guarded_real_credit_weights


@pytest.mark.parametrize('individual', [0., .40108381909949953])
def test_nonzero_guard_uses_registered_baseline(individual):
    registered = {'individual': 8.9, 'coop': 8.9, 'group': 9.9, 'team': 10.3}
    mult = {'individual': individual, 'coop': 1., 'group': 1., 'team': 1.}
    effective, receipt = guarded_real_credit_weights(registered, mult, individual_floor=True)
    assert effective == registered
    assert receipt['raw']['individual'] == 8.9 * individual
    assert receipt['effective']['individual'] - receipt['effective']['coop'] >= 0
    assert receipt['registered'] == registered


def test_legacy_guard_and_repaired_coop_never_amplifies():
    weights = {'individual': 8.9, 'coop': 8.9, 'group': 9.9, 'team': 10.3}
    mult = {'individual': 0., 'coop': .25, 'group': .5, 'team': .3}
    legacy, _ = guarded_real_credit_weights(weights, mult)
    repaired, _ = guarded_real_credit_weights(weights, mult, individual_floor=True)
    assert legacy['individual'] == legacy['coop'] == 0
    assert repaired['individual'] == weights['individual']
    assert repaired['coop'] == weights['coop'] * mult['coop']
    assert repaired['group'] == legacy['group'] and repaired['team'] == legacy['team']


def test_no_registered_individual_baseline_does_not_invent_one():
    result, _ = guarded_real_credit_weights({'individual': 0., 'coop': 8.9},
                                            {'individual': 0., 'coop': 1.}, individual_floor=True)
    assert result['individual'] == result['coop'] == 0.
