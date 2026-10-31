import copy

import numpy as np
import pytest

from tools.rbint.score_endpoints import endpoint_map, summarize, world_errors
from tools.rbeval.core import ARMS
from tools.rbeval.score import WorldBootstrap


def example(seed, arm, c=12., b=15.):
    factual = dict(seed=seed, arm=arm, initial_sha256='initial', start_sha256='start',
        history_sha256='history', vehicles=[
            dict(vehicle_id='hdv', is_hdv=True, s_m=10., censoring=None),
            dict(vehicle_id='gone', is_hdv=True, s_m=None, censoring='absent_at60'),
            dict(vehicle_id='cav', is_hdv=False, s_m=10., censoring=None)])
    def prediction(value):
        return dict(seed=seed, arm=arm, initial_sha256='initial', endpoint60=dict(
            schema='wmphys-predicted-endpoints/1', horizon_s=60, factual_future_used=False,
            start_sha256='start', history_sha256='history',
            vehicle_ids=['cav', 'gone', 'hdv'], is_hdv=[False, True, True],
            survived=[True, True, True], s_m=[9999., 100., value]))
    return factual, [prediction(c) for _ in range(8)], [prediction(b) for _ in range(8)]


def test_world_metric_joins_original_id_and_excludes_cav_factual_exits():
    factual, candidate, baseline = [], {}, {}
    for arm in ARMS:
        f, c, b = example(969432, arm)
        factual.append(f); candidate[969432, arm] = c; baseline[969432, arm] = b
    row = world_errors(candidate, baseline, factual, [969432])[0]
    assert row['candidate_median_m'] == 2.
    assert row['baseline_median_m'] == 5.
    assert row['median_difference_m'] == -3.
    assert row['counts'] == dict(t0_hdv=10, factual_absent_at60=5, common_hdv=5)
    candidate[969432, ARMS[0]][3]['endpoint60']['survived'][-1] = False
    candidate[969432, ARMS[0]][3]['endpoint60']['s_m'][-1] = None
    changed = world_errors(candidate, baseline, factual, [969432])[0]
    assert changed['counts']['common_hdv'] == 4
    assert changed['counts']['prediction_union_censored'] == 1


@pytest.mark.parametrize('key,value', [('factual_future_used', True), ('s_m', [1., 1., np.nan]),
    ('vehicle_ids', ['cav', 'gone', 'newborn']), ('start_sha256', 'different')])
def test_endpoint_contract_rejects_leakage_nonfinite_newborn_and_changed_start(key, value):
    factual, candidate, _ = example(969432, ARMS[0])
    candidate[0]['endpoint60'][key] = value
    with pytest.raises(AssertionError):
        endpoint_map(candidate[0], factual)


def test_singleton_strata_and_empty_cohort_cannot_certify_nonregression():
    real = [dict(seed=969432, demand=2400, penetration=.2)]
    bootstrap = WorldBootstrap(real, 10000)
    row = dict(seed=969432, counts=dict(common_hdv=5), median_difference_m=-2., mae_difference_m=-3.)
    summary = summarize([row], bootstrap)
    assert summary['point_nonregression']
    assert not summary['supported_nonregression']
    assert summary['primary']['low'] is summary['primary']['high'] is None
    empty = copy.deepcopy(row)
    empty.update(median_difference_m=None, mae_difference_m=None)
    assert not summarize([empty], bootstrap)['point_nonregression']
