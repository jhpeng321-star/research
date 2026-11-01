import copy

import pytest

from tools.rb2int.prepare_scoring import join_endpoints


def test_original_and_new_scoring_endpoints_are_disjoint_and_never_model_inputs():
    def data(worlds):
        return dict(schema='wmphys-factual-endpoints/1',horizon_s=60,t0_s=300,
            episodes=[dict(seed=s,arm=a) for s in worlds for a in ('native','ctrl0','mappo','pi','circles')])
    old,new = data(range(969420,969440)),data(range(969550,969600))
    result = join_endpoints(old,new)
    assert len(result['episodes']) == 350
    assert result['model_inputs'] is False and result['training'] is False
    bad = copy.deepcopy(new)
    bad['episodes'][0] = copy.deepcopy(old['episodes'][0])
    with pytest.raises(ValueError,match='unique endpoint'):
        join_endpoints(old,bad)
