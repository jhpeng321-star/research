"""Independent arithmetic/identity checks for the completed FIT attribution."""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np


def cdf_crps(draws, target):
    draws = np.asarray(draws, dtype=float)
    x = np.unique(np.r_[draws, target])
    midpoint = (x[1:] + x[:-1]) / 2
    cdf = (draws[:, None] <= midpoint[None]).mean(0)
    return float(np.sum(np.diff(x) * (cdf - (midpoint >= target)) ** 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--targets', type=Path, default=Path('E:/agent-work/rb-shared/targets/clocks_world.csv'))
    args = p.parse_args()
    records, summary = [json.loads(p.read_text()) for p in (args.input, args.summary)]
    assert hashlib.sha256(args.input.read_bytes()).hexdigest() == summary['input_sha256']
    assert len(records) == 32 and sorted(r['receipt']['world'] for r in records) == list(range(969400,969432))
    with args.targets.open(newline='') as f:
        target = {(int(r['seed']), int(r['horizon_s'])): r for r in csv.DictReader(f) if r['arm'] == 'native'}
    checked = 0
    for key in ('team_delay_vs', 'road_delay_vs', 'source_wait_vs'):
        independent_crps, independent_sd, factual = [], [], []
        for r in records:
            values = np.array([f[key] for f in r['modes']['both']])
            assert values.shape == (8,) and np.isfinite(values).all()
            variance = np.sum((values[:, None] - values[None])**2)/(2*8*7)
            independent_sd.append(float(np.sqrt(variance)))
            independent_crps.append(cdf_crps(values, r['real'][key]))
            if key == 'team_delay_vs':
                assert abs(float(target[r['receipt']['world'],120][key])-r['real'][key]) < 1e-6
            factual.append(r['real'][key])
            checked += 1
        s = summary['scores'][key]
        assert np.isclose(np.mean(independent_sd), s['mean_within_K_sd'], atol=1e-8, rtol=0)
        assert np.isclose(np.mean(independent_crps), s['mean_crps'], atol=1e-8, rtol=0)
        assert np.isclose(np.std(factual, ddof=1), s['real_across_world_sd'], atol=1e-8, rtol=0)
    first = next(r for r in records if r['receipt']['world'] == 969400)
    fixed = first['modes']['fixed']
    assert len({r['tape_sha256'] for r in fixed}) == 1
    assert first['requests']['both'] == first['requests']['arrivals']
    assert first['requests']['drivers'] == first['requests']['fixed']
    assert all(f == first['requests']['fixed'][0] for f in first['requests']['fixed'])
    assert first['driver_sd']['both'] == first['driver_sd']['drivers']
    assert all(v == 0 for v in first['driver_sd']['arrivals'].values())
    assert all(not r['receipt']['both']['physics_noise'] and not r['receipt']['both']['embedding_noise'] for r in records)
    result = dict(status='PASS', metric_world_cells=checked, worlds=32, futures=280,
        crps='independent empirical-CDF integral', sd='independent pairwise variance',
        real_target_equality=True, paired_source_invariants=True, fixed_tape_hash_equality=True,
        input_sha256=summary['input_sha256'],
        summary_sha256=hashlib.sha256(args.summary.read_bytes()).hexdigest())
    assert args.out.parent.is_dir()
    with args.out.open('x') as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
