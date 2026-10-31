import numpy as np

from tools.ensprobe.analyze import load_detector, spread, variance_parts


def test_frozen_detector_onset_and_preexisting_are_distinct():
    detect,receipt=load_detector()
    assert 'bic<null' in receipt['source']
    assert detect([25.]*66)['event'] is False
    event=detect([25.]*15+[12.]*51)
    assert event['event'] is True and event['onset_s']==90.
    existing=detect([25.]*4+[12.]*62)
    assert existing['event'] is False and existing['preexisting'] is True and existing['onset_s']==-20.


def test_variance_components_do_not_conflate_state_and_seed():
    same=np.repeat(np.array([[10.],[20.],[30.]]),8,axis=1)
    result=variance_parts(same)
    assert result['within_seed_variance']==0.
    assert result['between_state_variance']==100.
    repeated=np.tile(np.arange(8.),(3,1))
    result=variance_parts(repeated)
    assert result['state_mean_variance']==0 and result['within_seed_variance']==6.
    assert result['between_state_variance']==0 and result['between_state_variance_untruncated']==-.75


def test_spread_uses_sample_sd():
    result=spread([1.,3.])
    assert result['mean']==2. and result['sd']==np.sqrt(2.)
    assert result['minimum']==1. and result['maximum']==3.
