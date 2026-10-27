"""Paired world-cluster bootstrap of absolute signed-bias reduction."""
import re

import numpy as np


def compare(reference, candidate):
    r = {row['pair_key']: row for row in reference['rows']}
    c = {row['pair_key']: row for row in candidate['rows']}
    assert len(r) == len(reference['rows']) and len(c) == len(candidate['rows'])
    assert set(r) == set(c), 'scoring panel changed'
    worlds = {}
    for key in sorted(r):
        match = re.match(r'w(\d+)m', key)
        assert match, key
        a, b = np.asarray(r[key]['dJ']['300']), np.asarray(c[key]['dJ']['300'])
        assert a.shape == b.shape and np.array_equal(a[:, 1], b[:, 1]), 'recorded targets changed'
        assert np.isfinite(a).all() and np.isfinite(b).all()
        worlds.setdefault(int(match[1]), []).extend(zip(a[:, 0]-a[:, 1], b[:, 0]-b[:, 1]))
    arrays = [np.asarray(v) for v in worlds.values()]
    def delta(rows):
        means = np.concatenate(rows).mean(0)
        return float(abs(means[0])-abs(means[1]))
    rng = np.random.default_rng(0)
    draws = [delta([arrays[i] for i in rng.integers(len(arrays), size=len(arrays))])
             for _ in range(2000)]
    ci = np.quantile(draws, [.025, .975]).tolist()
    threshold = candidate['threshold']
    assert threshold == reference['threshold']
    short = all(candidate['horizons'][h]['sign_agreement'] is not None and
                candidate['horizons'][h]['sign_agreement'] >= threshold for h in ('10', '30', '60'))
    point = delta(arrays)
    return {'verdict': 'helps' if point > 0 and ci[0] > 0 and short else 'does_not_help',
            'absolute_300s_bias_reduction': point, 'reduction_ci95': ci,
            'n_worlds': len(worlds), 'n_pair_groups': len(r), 'short_signs_pass': short,
            'bootstrap': {'unit': 'world_seed', 'draws': 2000, 'seed': 0}}


def bias(doc):
    out = {}
    for h in ('120', '300'):
        a = np.asarray([v for row in doc['rows'] for v in row['dJ'][h]], float)
        assert np.isfinite(a).all()
        e = a[:, 0]-a[:, 1]
        out[h] = {'mean_signed_error': float(e.mean()), 'error_sd': float(e.std()),
                  'reference_dJ_sd': float(a[:, 1].std()), 'wm_dJ_sd': float(a[:, 0].std()),
                  'mae': float(abs(e).mean()), 'n_pairs': len(e)}
    return out
