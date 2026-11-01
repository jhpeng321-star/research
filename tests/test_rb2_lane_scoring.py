import copy

from tools.rblane.round2_free_finalize import exclude_internal_initialization


def test_internal_to_normal_transition_is_not_a_lane_change_and_raw_record_is_preserved():
    raw=dict(world=969420,sample=0,variant='fitted_lc',
        receipt=dict(fitted_artifacts={'posterior':'fixed'},demand_rate_draws=[[.5,.1]]),
        cells={'merge/mainline/0':dict(count=3,veh_time_s=20.)})
    audit=dict(world=969420,sample=0,variant='fitted_lc',receipt=copy.deepcopy(raw['receipt']),
        excluded_LC_counts={'merge/mainline/0':1})
    scored=copy.deepcopy(raw)
    assert exclude_internal_initialization(scored,audit)==1
    assert raw['cells']['merge/mainline/0']['count']==3
    assert scored['cells']['merge/mainline/0']==dict(count=2,veh_time_s=20.)


def test_scoring_audit_cannot_be_transferred_to_a_different_future():
    import pytest
    raw=dict(world=969420,sample=0,variant='fitted_lc')
    audit=dict(world=969420,sample=1,variant='fitted_lc')
    with pytest.raises(AssertionError):exclude_internal_initialization(raw,audit)
