from .ls_saved_regrade import eligible,select


def sample(key='a',status='failed',sim=0,branches=None,reason='FileNotFoundError(2)'):
    return dict(search_id=key,world_id=986000,verification=dict(status=status,actual_sim_s=sim,branches=branches,reason=reason))


def test_only_prephysics_missing_dependency_failures_are_retryable():
    assert eligible(sample())
    assert not eligible(sample(sim=1))
    assert not eligible(sample(branches=[dict(status='failed')]))
    assert not eligible(sample(reason='prefix mismatch'))
    assert not eligible(sample(status='rejected'))


def test_completed_saved_search_is_never_regraded():
    assert select([sample(),sample(status='accepted')])==[]
    assert select([sample(),sample(status='rejected')])==[]


def test_selection_is_world_key_order_not_outcome_order():
    a=sample('z');b=sample('b');b['world_id']=986001
    assert select([b,a])==[a,b]
