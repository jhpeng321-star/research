from types import SimpleNamespace

import numpy as np
import pytest

from tools.rbint.compare import paired_features, summarize_features
from tools.rbeval.core import ARMS, HS


def episode(seed, arm, delay, event):
    return dict(seed=seed, arm=arm,
        clocks={str(h):dict(team_delay_vs=delay) for h in HS},
        bins=dict(upstream=dict(events={str(h):dict(event=event) for h in HS})))


def test_paired_crps_is_world_clustered_and_rate_distance_is_after_pooling():
    worlds = [969432, 969433]
    candidate, baseline, real = {}, {}, []
    for index, seed in enumerate(worlds):
        for arm in ARMS:
            real.append(episode(seed, arm, 0., bool(index)))
            candidate[seed, arm] = [episode(seed, arm, 2.+index, bool(index))]*8
            baseline[seed, arm] = [episode(seed, arm, 5.+index, not bool(index))]*8
    features = paired_features(candidate, baseline, real, worlds)
    np.testing.assert_array_equal(features[0], np.full((2, 3), -3.))
    bootstrap = SimpleNamespace(index={s:i for i,s in enumerate(worlds)},
        weights=np.tile(np.array([[2., 0.], [0., 2.], [1., 1.]]), (20, 1)))
    result = summarize_features(features, worlds, bootstrap)
    rates = [r for r in result['comparisons'] if r['metric'].startswith('upstream')]
    assert all(r['mean'] == 0. for r in rates)  # pooled rates both1/2, real1/2
    assert all(r['candidate_rate'] == r['real_rate'] == r['baseline_rate'] == .5 for r in rates)
    assert result['all_three_crps_point_improvements']
    assert result['switching_eligibility'] == 'NOT_ESTABLISHED'  # endpoints still missing


def test_one_world_has_no_invented_sampling_ci_or_eligibility():
    features = [np.full((1, 3), x) for x in (-2., .75, 1., 1.)]
    bootstrap = SimpleNamespace(index={969432:0}, weights=np.ones((100, 1)))
    result = summarize_features(features, [969432], bootstrap)
    assert all(row['low'] is None and row['high'] is None for row in result['comparisons'])
    assert result['hdv60_position_error']['mean'] is None


def test_missing_event_support_and_empty_complete_ensemble_are_explicit():
    c, b, real = {}, {}, []
    for arm in ARMS:
        real.append(episode(969432, arm, 0., False))
        c[969432, arm] = [episode(969432, arm, 0., None)]*8
        b[969432, arm] = [episode(969432, arm, 0., False)]*8
    with pytest.raises(ValueError, match='event support'):
        paired_features(c, b, real, [969432])
    assert summarize_features([np.zeros((0, 3))]*4, [], None)['status'] == 'NO_COMPLETE_PAIRED_K8_WORLD'
