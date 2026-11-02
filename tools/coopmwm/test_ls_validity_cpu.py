import pytest
from .ls_validity import contract, feasible, exclusion_counts


def test_bound_is_from_artifact_and_covers_whole_registered_search():
    from .mppi_search import REGISTERED
    a = contract(dict(schema='wm_phys.observed_arrivals.v1', observed_phase_end_s=1200), REGISTERED)
    assert a['required_horizon_s'] == 420
    assert a['screen_horizon_s'] == 300
    assert feasible(780, a) and not feasible(790, a) and not feasible(860, a)
    b = contract(dict(schema='wm_phys.observed_arrivals.v1', observed_phase_end_s=1150), REGISTERED)
    assert feasible(730, b) and not feasible(740, b)


def test_counts_retain_late_snapshots_but_never_admit_them():
    a = dict(required_horizon_s=420, observed_phase_end_s=1200)
    pending = {i:dict(time_s=t) for i,t in enumerate((300,600,780,790,800,860,900))}
    assert exclusion_counts(pending, a) == {'700':1,'800':2,'900':1}
    assert len(pending) == 7
    with pytest.raises(ValueError):feasible(float('nan'), a)


def test_arrival_bound_error_is_one_query_decline_not_worker_death():
    from .query_errors import execute
    r = dict(state_id='fixture')
    def fail():raise ValueError('forecast exceeds observed/documented fitted demand phase')
    out = execute(r, fail)
    assert not out['accepted'] and not out['query_error']['inference_retry']
    assert 'fitted demand phase' in out['query_error']['traceback']
    assert execute(r, lambda:dict(accepted=True)) == dict(accepted=True)


def test_observed_ssh_transport_variants_retry_but_code_faults_do_not():
    from .warm_transport import connection_error
    for text in ('ssh_dispatch_run_fatal: Connection to 1.15.42.207 port 22: Connection timed out',
                 'Connection timed out during banner exchange\nConnection to 1.15.42.207 port 22 timed out'):
        assert connection_error(RuntimeError(text))
        assert not connection_error(RuntimeError('Traceback\n'+text))
