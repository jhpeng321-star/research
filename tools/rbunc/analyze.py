"""Descriptive FIT uncertainty attribution; never tunes the sampler to outcomes."""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np


def sd(values):
    return float(np.std(values, ddof=1))


def crps(draws, observed):
    a = np.array(draws, dtype=float)
    return float(np.abs(a-observed).mean() - .5*np.abs(a[:, None]-a[None]).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--targets', type=Path, default=Path('E:/agent-work/rb-shared/targets/clocks_world.csv'))
    args = parser.parse_args()
    assert args.out.parent.is_dir()
    records = json.loads(args.input.read_text())
    assert sorted(r['receipt']['world'] for r in records) == list(range(969400, 969432))
    assert len({r['sampler_sha256'] for r in records}) == 1
    assert len({r['source_sha256'] for r in records}) == 1
    assert all(r['horizon'] == 120 and len(r['modes']['both']) == 8 for r in records)
    assert all(f['no_slot'] == 0 for r in records for mode in r['modes'].values() for f in mode)
    assert all(r['receipt']['both']['physics_noise'] is False for r in records)
    with args.targets.open(newline='') as f:
        targets = [row for row in csv.DictReader(f) if row['arm'] == 'native']
    assert {int(row['seed']) for row in targets} == set(range(969400, 969432))
    summary = dict(schema='rbunc_attribution/1', worlds=32, K=8, horizon=120,
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        factual_target_sha256=hashlib.sha256(args.targets.read_bytes()).hexdigest(),
        sampler_sha256=records[0]['sampler_sha256'], probe_sha256=records[0]['source_sha256'],
        physics=records[0]['receipt']['physics'], checkpoint_sha256=records[0]['receipt']['checkpoint_sha256'],
        diagnostic_only=True, scores={}, source_ablation={}, request_spread={}, driver_sd={})
    by_world = {r['receipt']['world']: r for r in records}
    for key in ('team_delay_vs', 'road_delay_vs', 'source_wait_vs'):
        within = np.array([sd([f[key] for f in r['modes']['both']]) for r in records])
        real = np.array([r['real'][key] for r in records])
        mean = np.array([np.mean([f[key] for f in r['modes']['both']]) for r in records])
        rng = np.random.default_rng(20261031)
        boot = within[rng.integers(0, 32, (10000, 32))].mean(1)
        score = dict(mean_within_K_sd=float(within.mean()), median_within_K_sd=float(np.median(within)),
            rms_within_K_sd=float(np.sqrt(np.mean(within**2))),
            min_within_K_sd=float(within.min()), max_within_K_sd=float(within.max()),
            mean_within_K_sd_world_bootstrap95=np.quantile(boot, [.025, .975]).tolist(),
            real_across_world_sd=sd(real), real_mean=float(real.mean()),
            predicted_mean=float(mean.mean()), mean_bias=float((mean-real).mean()),
            mean_crps=float(np.mean([crps([f[key] for f in r['modes']['both']], r['real'][key])
                                    for r in records])),
            range_coverage=float(np.mean([min(f[key] for f in r['modes']['both']) <= r['real'][key] <=
                max(f[key] for f in r['modes']['both']) for r in records])))
        score['mean_within_K_sd_over_real_across_world_sd'] = score['mean_within_K_sd']/score['real_across_world_sd']
        summary['scores'][key] = score
    first = by_world[969400]
    assert set(first['modes']) == {'both', 'drivers', 'arrivals', 'fixed'}
    for name, futures in first['modes'].items():
        summary['source_ablation'][name] = dict(
            team_delay_sd=sd([f['team_delay_vs'] for f in futures]),
            road_delay_sd=sd([f['road_delay_vs'] for f in futures]),
            source_wait_sd=sd([f['source_wait_vs'] for f in futures]),
            unique_physical_tapes=len({f['tape_sha256'] for f in futures}),
            driver_v0_mean_sd=first['driver_sd'][name]['v0'],
            arrival_count_sd=sd([sum(f['arrivals']) for f in futures]))
    assert summary['source_ablation']['fixed']['unique_physical_tapes'] == 1
    assert summary['source_ablation']['fixed']['team_delay_sd'] == 0
    assert summary['source_ablation']['drivers']['arrival_count_sd'] == 0
    assert summary['source_ablation']['arrivals']['driver_v0_mean_sd'] == 0
    for horizon in (120, 300, 600):
        spread = [sd([sum(f[str(horizon)]) for f in r['requests']['both']]) for r in records]
        factual = [float(row['arrivals']) for row in targets if int(row['horizon_s']) == horizon]
        assert len(factual) == 32
        summary['request_spread'][str(horizon)] = dict(mean_within_K_sd=float(np.mean(spread)),
            rms_within_K_sd=float(np.sqrt(np.mean(np.square(spread)))),
            mean_poisson_theoretical_sd=float(np.mean([np.sqrt(sum(r['receipt']['demand_rates'])*horizon)
                                                       for r in records])),
            recorded_destination_exit_sd=sd(factual),
            recorded_destination_exit_mean=float(np.mean(factual)),
            comparison='recorder arrivals are destination exits, NOT entrance requests; not comparable')
    for name in first['driver_sd']['both']:
        summary['driver_sd'][name] = float(np.mean([r['driver_sd']['both'][name] for r in records]))
    diagnostics = [d for r in records for d in r['receipt']['both']['likelihood']]
    summary['likelihood'] = dict(drivers=len(diagnostics),
        residual_mse_quantiles=np.quantile([d['residual_mse'] for d in diagnostics], [0,.5,.9,.99,1]).tolist(),
        particle_ess_quantiles=np.quantile([d['particle_ess'] for d in diagnostics], [0,.5,.9,1]).tolist())
    summary['verification'] = dict(worlds_complete=True, no_slot_or_dropped_requests=True,
        fixed_latents_bit_identical=True, driver_only_arrivals_identical=True,
        arrivals_only_present_drivers_identical=True,
        physics_rng_unchanged='asserted on every probe rollout',
        queue_conservation='asserted on every probe rollout',
        physical_futures=sum(len(f) for r in records for f in r['modes'].values()))
    with args.out.open('x') as f:
        json.dump(summary, f, indent=1, allow_nan=False)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
