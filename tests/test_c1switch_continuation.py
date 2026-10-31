import pytest

from tools.c1switch.continuation import imported_ledger, reuse_decision


def test_superseded_compute_still_counts_without_consuming_new_scientific_arm():
    rows = [dict(id='retained_step0_47010',sim_s=47010,stage='step0'),
            dict(id='wm_s969112_episode',sim_s=900,stage='main'),
            dict(id='wm_s969112_branches',sim_s=11550,stage='main'),
            dict(id='bb_s969112_episode',sim_s=900,stage='main'),
            dict(id='bb_s969112_branches',sim_s=11550,stage='main')]
    result = imported_ledger(rows)
    assert sum(r['sim_s'] for r in result)==71910
    assert sum(r['sim_s'] for r in result if r['id'].startswith('wm_'))==0
    assert sum(r['sim_s'] for r in result if r['id'].startswith('bb_'))==12450
    assert rows[1]['id']=='wm_s969112_episode'
    with pytest.raises(ValueError):imported_ledger(rows[:-1])


def test_bb_reuse_requires_same_incumbent_aware_tie_pick():
    old = dict(incumbent='pi',pick='pi')
    costs = dict(native=1.,ctrl0=1.,mappo=1.,pi=1.,circles=2.)
    assert reuse_decision(old,costs)=='pi'
    costs['native']=0.
    with pytest.raises(ValueError,match='reuse forbidden'):reuse_decision(old,costs)
