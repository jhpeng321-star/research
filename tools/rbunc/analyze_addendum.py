"""Complete-panel-only independent metric/source checks for the fitted variant."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'docs/paper_results/rbunc'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=OUT/'fitted_source_probe.json')
    parser.add_argument('--out', type=Path, default=OUT/'fitted_summary.json')
    parser.add_argument('--compare-raw', type=Path)
    args = parser.parse_args()
    raw = args.raw
    bundle = json.loads(raw.read_text())
    records = bundle['records']
    assert [r['world'] for r in records] == list(range(969400, 969432))
    assert all(r['rc'] == 0 for r in bundle['receipts']['probe_TERMINAL.json']['results'])
    arrivals = json.loads((OUT/'observed_arrivals.json').read_text())
    summary = dict(schema='rbunc_fitted_summary/1', worlds=32, K=8, physical_horizon_s=120,
                   physical_futures=sum(len(v) for r in records for v in r['modes'].values()),
                   raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(), metrics={},
                   qualification='source plumbing in deterministic legacy IDM/MOBIL reference only')
    rng = np.random.default_rng(20261031)
    indices = rng.integers(0, 32, size=(10000, 32))
    for metric in ('team_delay_vs', 'road_delay_vs', 'source_wait_vs'):
        forecasts = np.array([[f[metric] for f in r['modes']['both']] for r in records])
        real = np.array([r['real'][metric] for r in records])
        sd = forecasts.std(1, ddof=1)
        pairvar = ((forecasts[:, :, None]-forecasts[:, None, :])**2).sum((1,2))/(2*8*7)
        np.testing.assert_allclose(sd**2, pairvar, rtol=1e-12, atol=1e-8)
        crps = np.abs(forecasts-real[:, None]).mean(1) - .5*np.abs(
            forecasts[:, :, None]-forecasts[:, None, :]).mean((1,2))
        independent = []
        for row, target in zip(forecasts, real):
            edges = np.unique(np.r_[row, target])
            mid = (edges[:-1]+edges[1:])/2
            cdf = (row[:, None] <= mid).mean(0)
            independent.append(np.sum(np.diff(edges)*(cdf-(mid >= target))**2))
        np.testing.assert_allclose(crps, independent, rtol=1e-12, atol=1e-8)
        summary['metrics'][metric] = dict(mean_within_K_sd=float(sd.mean()),
            mean_within_K_sd_ci95=np.quantile(sd[indices].mean(1), [.025,.975]).tolist(),
            rms_within_K_sd=float(np.sqrt((sd**2).mean())), real_across_world_sd=float(real.std(ddof=1)),
            mean_bias=float((forecasts.mean(1)-real).mean()), empirical_crps=float(crps.mean()),
            range_coverage=float(((real>=forecasts.min(1)) & (real<=forecasts.max(1))).mean()))
    first = records[0]
    summary['first_state_team_sd'] = {name:float(np.std([f['team_delay_vs'] for f in rows], ddof=1))
                                     for name, rows in first['modes'].items()}
    assert summary['first_state_team_sd']['fixed'] == 0
    assert len({r['tape_sha256'] for r in first['modes']['fixed']}) == 1
    assert all(v==0 for v in first['driver_sd']['arrivals'].values())
    assert all(v==0 for v in first['driver_sd']['fixed'].values())
    assert all(r==first['requests']['drivers'][0] for r in first['requests']['drivers'])
    assert first['requests']['drivers'] == first['requests']['fixed']
    old = json.loads((OUT/'fit_h120_fit_probe_v2.json').read_text())
    for r in records:
        assert r['identity']['exact_observable_correspondence']
        assert r['paired_hidden_inputs_exact'] and r['global_rng_unchanged']
        for rows in r['modes'].values():
            assert len(rows)==8 and all(f['no_slot']==0 for f in rows)
            assert all(np.isfinite(f['team_delay_vs']) for f in rows)
        match = next(v for v in old if v['receipt']['world']==r['world'])
        for metric in summary['metrics']:
            assert r['real'][metric] == match['real'][metric]
    summary['requests'] = {}
    for horizon in (120, 300, 600):
        sd = [np.std([sum(f[str(horizon)]) for f in r['requests']['both']], ddof=1) for r in records]
        theory = [np.sqrt((sum(arrivals['worlds'][str(r['world'])]['total'])+1)*
                  (horizon/300+(horizon/300)**2)) for r in records]
        summary['requests'][str(horizon)] = dict(mean_within_K_sd=float(np.mean(sd)),
            mean_gamma_poisson_predictive_sd=float(np.mean(theory)))
    summary['mean_driver_input_sd'] = {n:float(np.mean([r['driver_sd']['both'][n] for r in records]))
                                       for n in first['driver_sd']['both']}
    summary['mean_optional_lc_input_sd'] = {n:float(np.mean([r['optional_lc_input_sd'][n] for r in records]))
                                            for n in first['optional_lc_input_sd']}
    summary['checks'] = dict(pairwise_variance=True, independent_cdf_crps=True,
        original_factual_targets_exact=True, source_ablations=True, physical_replay=True,
        exact_numeric_identity=True, paired_hidden_inputs=True, metric_world_cells=96)
    if args.compare_raw is not None:
        previous = json.loads(args.compare_raw.read_text())['records']
        assert [r['world'] for r in previous] == [r['world'] for r in records]
        for a, b in zip(records, previous):
            assert a['requests'] == b['requests'], 'arrival CRN changed'
            assert a['real'] == b['real'] and a['identity'] == b['identity']
            sa, sb = a['source_receipts']['both'], b['source_receipts']['both']
            assert sa['learned_arrival_sha256'] == sb['learned_arrival_sha256']
            assert sa['demand_rate_draws'] == sb['demand_rate_draws']
            assert sa['fitted_artifacts']['idm_population_sha256'] == sb['fitted_artifacts']['idm_population_sha256']
            assert sa['fitted_artifacts']['idm_posterior_sha256'] != sb['fitted_artifacts']['idm_posterior_sha256']
        comparison = dict(reference_sha256=hashlib.sha256(args.compare_raw.read_bytes()).hexdigest(),
            arrivals_requests_rates_population_identity_targets_exact=True, metrics={})
        for metric in summary['metrics']:
            def scores(rows):
                x = np.array([[f[metric] for f in r['modes']['both']] for r in rows])
                y = np.array([r['real'][metric] for r in rows])
                return (np.abs(x-y[:, None]).mean(1) - .5*np.abs(x[:, :, None]-x[:, None, :]).mean((1,2)),
                        x.mean(1)-y)
            new_crps, new_bias = scores(records)
            old_crps, old_bias = scores(previous)
            delta = new_crps-old_crps
            comparison['metrics'][metric] = dict(crps_change=float(delta.mean()),
                crps_change_ci95=np.quantile(delta[indices].mean(1), [.025,.975]).tolist(),
                bias_change=float((new_bias-old_bias).mean()), worlds_crps_lower=int((delta<0).sum()))
        summary['paired_posterior_comparison'] = comparison
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=1, allow_nan=False)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
