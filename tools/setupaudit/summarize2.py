"""Paired equal-world panel estimates and shared WORLD-cluster percentile CIs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def safe(path):
    assert not re.search(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype',
                         str(path), re.I)
    return Path(path)


def cluster_plan(worlds, draws=5000, seed=20261030):
    unique = sorted(set(worlds))
    plan = np.random.default_rng(seed).integers(len(unique), size=(draws, len(unique)))
    weights = np.zeros((draws, len(unique)))
    np.add.at(weights, (np.arange(draws)[:, None], plan), 1/len(unique))
    return unique, weights


def estimate(values, worlds, unique, weights):
    values = np.asarray(values, dtype=float)
    clusters = np.stack([values[np.array(worlds) == world].mean(0) for world in unique])
    original_shape = clusters.shape[1:]
    boot = (weights @ clusters.reshape(len(unique), -1)).reshape(len(weights), *original_shape)
    return {'mean': clusters.mean(0).tolist(),
            'lo': np.quantile(boot, .025, axis=0).tolist(),
            'hi': np.quantile(boot, .975, axis=0).tolist()}


def analyze(document, *, allow_incomplete=False):
    starts = sorted(set(r['start'] for r in document['rows']))
    assert allow_incomplete or starts == list(range(24)), 'need all 24 starts'
    records = []
    for start in starts:
        rows = sorted((r for r in document['rows'] if r['start'] == start), key=lambda r: r['future'])
        futures = [r['future'] for r in rows]
        assert len(set(futures)) == len(rows), 'duplicate futures'
        assert allow_incomplete or futures == list(range(document['K'])), 'missing futures'
        assert all(r['metadata'] == rows[0]['metadata'] for r in rows)
        for row in rows:
            assert row['costs']['births'][0] == row['costs']['births'][-1]
            assert row['costs']['terminal_queue_by_source'][0] == row['costs']['terminal_queue_by_source'][-1]
            for h in ('90', '120', '300', '600'):
                c = row['costs'][h]
                for total, pieces in [('J', ['road', 'source']), ('road', ['CAV', 'HDV']),
                                      ('road', ['entrant', 'initial']),
                                      ('J', ['initial', 'entrant_total']),
                                      ('real_J', ['real_road', 'real_source']),
                                      ('road', ['ramp', 'mainline', 'merge_zone']),
                                      ('source', ['source_mainline', 'source_ramp'])]:
                    np.testing.assert_allclose(c[total], np.sum([c[k] for k in pieces], axis=0),
                                               rtol=0, atol=.05)
                assert c['source'][0] == c['source'][-1], 'frozen source differs from CTRL0'
                assert c['real_source'][0] == c['real_source'][-1]
        item = {'start': start, 'metadata': rows[0]['metadata'], 'n_futures': len(rows),
                'horizons': {}, 'futures': {}}
        for h in ('90', '120', '300', '600'):
            item['horizons'][h] = {k: np.mean([r['costs'][h][k] for r in rows], axis=0).tolist()
                                   for k in rows[0]['costs'][h]}
            item['futures'][h] = {k: [r['costs'][h][k] for r in rows] for k in ('J', 'road', 'source')}
        item['time_bins_30s'] = {k: np.mean([r['costs']['time_bins_30s'][k] for r in rows], axis=0).tolist()
                                 for k in rows[0]['costs']['time_bins_30s']}
        item['lane_requests'] = np.mean([r['lane_requests'] for r in rows], axis=0).tolist()
        item['births'] = np.mean([r['costs']['births'] for r in rows], axis=0).tolist()
        records.append(item)
    worlds = [r['metadata']['world'] for r in records]
    unique, weights = cluster_plan(worlds)
    output = {'schema': 'setupaudit2/summary1', 'complete': len(records) == 24 and
              all(r['n_futures'] == document['K'] for r in records),
              'n_starts': len(records), 'n_worlds': len(unique), 'K': document['K'],
              'actors': document['actors'], 'bootstrap': {'draws': len(weights), 'seed': 20261030,
              'unit': 'world; average futures and starts within world before resampling',
              'estimand': 'equal-world mean in the restored TRAIN sampler panel'},
              'horizons': {}, 'per_start': records, 'source_sha256': document.get('source_sha256')}
    for h in ('90', '120', '300', '600'):
        stats = {}
        for metric in records[0]['horizons'][h]:
            values = np.array([r['horizons'][h][metric] for r in records])
            stats[metric] = {'arms': estimate(values, worlds, unique, weights),
                            'vs_CTRL0': estimate(values-values[:, :1], worlds, unique, weights)}
        j = np.array([r['horizons'][h]['J'] for r in records])
        stats['contrasts'] = {name: estimate(values, worlds, unique, weights) for name, values in {
            'lane_intervention': j[:, 2]-j[:, 1],
            'admission_intervention': j[:, 5]-j[:, 1],
            'lane_longitudinal_interaction': j[:, 1]-j[:, 2]-j[:, 3]+j[:, 0],
            'lane_main_effect': .5*(j[:, 1]-j[:, 2]+j[:, 3]-j[:, 0]),
            'longitudinal_main_effect': .5*(j[:, 1]-j[:, 3]+j[:, 2]-j[:, 0]),
        }.items()}
        output['horizons'][h] = stats
    output['time_bins_30s'] = {}
    for metric in ('J', 'road', 'source', 'real_J', 'real_road', 'real_source'):
        v = np.array([r['time_bins_30s'][metric] for r in records])
        output['time_bins_30s'][metric] = estimate(v-v[:, :1], worlds, unique, weights)
    output['lane_requests'] = estimate([r['lane_requests'] for r in records], worlds, unique, weights)
    output['births'] = estimate([r['births'] for r in records], worlds, unique, weights)
    output['intervals'] = {}
    for lo, hi in ((0, 90), (90, 300), (300, 600)):
        stats = {}
        for metric in ('J', 'road', 'source', 'real_J', 'real_road', 'real_source'):
            upper = np.array([r['horizons'][str(hi)][metric] for r in records])
            lower = np.array([r['horizons'][str(lo)][metric] for r in records]) if lo else 0
            v = upper-lower
            stats[metric] = estimate(v-v[:, :1], worlds, unique, weights)
        output['intervals'][f'{lo}-{hi}'] = stats
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input')
    p.add_argument('--out')
    p.add_argument('--csv')
    p.add_argument('--allow-incomplete', action='store_true')
    a = p.parse_args()
    path = safe(a.input)
    doc = json.loads(path.read_text())
    result = analyze(doc, allow_incomplete=a.allow_incomplete)
    result['input_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    if a.out:
        out = safe(a.out)
        assert not out.exists()
        out.write_text(json.dumps(result, indent=2)+'\n')
    if a.csv:
        out = safe(a.csv)
        assert not out.exists()
        with out.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['start', 'world', 't0_s', 'K', 'horizon_s', 'actor', 'metric', 'value', 'minus_CTRL0'])
            for row in result['per_start']:
                for h, metrics in row['horizons'].items():
                    for metric, values in metrics.items():
                        for actor, value in zip(result['actors'], values):
                            writer.writerow([row['start'], row['metadata']['world'], row['metadata']['t0_s'],
                                             row['n_futures'], h, actor, metric, value, value-values[0]])
    print('complete', result['complete'], 'starts/worlds/K', result['n_starts'], result['n_worlds'], result['K'])
    for h, metrics in result['horizons'].items():
        for actor in range(1, len(result['actors'])):
            s = metrics['J']['vs_CTRL0']
            print(h, result['actors'][actor], *(round(s[k][actor], 2) for k in ('mean', 'lo', 'hi')))


if __name__ == '__main__':
    main()
