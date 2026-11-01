import pytest

from tools.rb2int.gate_bootstrap import GateBootstrap


def bootstrap():
    return GateBootstrap([dict(seed=s, demand=d, penetration=.2)
        for s, d in ((1, 1900), (2, 1900), (3, 2400), (4, 2400), (5, 2400))], B=500)


def test_feature_cannot_borrow_unscored_same_cell_peer_weights():
    boot = bootstrap()
    rows = boot.bands([[(3, 1.), (4, 3.)], [(3, 1.), (4, 3.), (5, 5.)]])
    assert not rows[0]['uncertainty_available']
    assert rows[0]['unavailable_reason'] == 'incomplete_scored_stratum_support'
    assert rows[1]['uncertainty_available']


def test_complete_cell_and_full_mixture_features_can_share_joint_family():
    boot = bootstrap()
    rows = boot.bands([[(1, 1.), (2, 4.)], [(i, float(i)) for i in range(1, 6)]])
    assert all(r['uncertainty_available'] for r in rows)
    assert rows[0]['maxstat_critical'] == rows[1]['maxstat_critical']


def test_singleton_and_missing_factual_values_remain_na():
    boot = bootstrap()
    rows = boot.bands([[(1, 1.)], [(1, 1.), (2, None)]])
    assert all(r['low'] is r['high'] is None for r in rows)
    assert rows[0]['unavailable_reason'] == 'unreplicated_factual_stratum'


def test_unscored_world_or_wrong_finite_k_mean_is_rejected():
    boot = bootstrap()
    with pytest.raises(ValueError, match='unscored world'):
        boot.bands([[(99, 1.)]])
    with pytest.raises(ValueError, match='mean differs'):
        boot.bands([[(1, 1., [0.]*8), (2, 2., [2.]*8)]])
