import pytest


def test_world_cluster_bootstrap_reselects_fixed_and_keeps_world_weights():
    pytest.importorskip('numpy')
    from tools.c1switch.analyze import primary,rank
    # Controller0/1 trade wins across worlds; correct bootstrap reselects which
    # fixed controller is best, instead of leaking the full-panel identity.
    result = primary([[0,100,200,200,200],[100,0,200,200,200]],
                     [[2,1],[4,1]],[[0,10,20,20,20],[10,0,20,20,20]])
    assert result['wm_minus_bb_regret60']['mean'] == 2
    assert result['wm_minus_best_fixed_regret60']['mean'] == 5
    assert result['wm_minus_best_fixed_regret60']['ci95'] == [0,5]
    assert rank([5,1,1,10]) == [3.,1.5,1.5,4.]
