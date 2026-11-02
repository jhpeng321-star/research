import pytest
from .search_timing import summarize


def row(host='gpuA', n=2, wall=100, end=1000):
    return dict(host=host, group_size=n, wall_s=wall, unix=end, top8_count=8,
        search_config=dict(name='it2_K2', overrides=dict(iterations=2, screen_K=2),
            full_population=True, screen_H=300, final_K=8))


def test_variant_filter_and_aggregate_rate_keep_unfinished_time():
    rows = [row(), row(wall=200, end=1200), row('gpuR', 6, 300, 1300),
            dict(search_config=dict(name='plain'))]
    result = summarize(rows, 1800)
    assert result['completed'] == 3
    assert result['window_start_unix'] == 900
    assert result['observed_completed_per_hour'] == 12
    assert result['by_host_group_size']['gpuA/n2']['mean_query_wall_s'] == 150
    assert result['by_host_group_size']['gpuR/n6']['completed'] == 1
    assert result['new_sumo_sim_s'] == result['new_wm_searches'] == 0


def test_empty_is_not_zero_throughput():
    result = summarize([], 100)
    assert result['observed_completed_per_hour'] is None
    assert result['window_start_unix'] is None


@pytest.mark.parametrize('field,value', [('screen_H', 120), ('final_K', 4)])
def test_reject_horizon_or_final_k_drift(field, value):
    r = row()
    r['search_config'][field] = value
    with pytest.raises(AssertionError):
        summarize([r], 1800)
