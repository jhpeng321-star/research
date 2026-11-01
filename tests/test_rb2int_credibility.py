from tools.rb2int.credibility import ratio_bands
from tools.rb2int.uncertainty import PairedWorldBootstrap


def boot():
    return PairedWorldBootstrap([dict(seed=s,demand=d,penetration=.2)
        for s,d in ((1,1900),(2,1900),(3,2400),(4,2400))],B=1000)


def test_nonselected_scored_worlds_remain_zero_opportunities_not_fake_independent_calls():
    rows = [dict(seed=1,outcome=1),dict(seed=2,outcome=0)]
    result = ratio_bands(boot(),[(rows,'outcome')])[0]
    assert result['mean'] == .5 and result['count'] == 2
    assert result['contributing_worlds'] == 2
    assert result['uncertainty_available']


def test_empty_selected_resamples_and_single_world_do_not_certify_reliability():
    rows = [dict(seed=1,outcome=1)]
    result = ratio_bands(boot(),[(rows,'outcome')])[0]
    assert result['mean'] == 1
    assert result['low'] is result['high'] is None
    assert not result['uncertainty_available']
