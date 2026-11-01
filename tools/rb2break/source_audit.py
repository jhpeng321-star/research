"""Separate finite-K arrival error from admitted-flow physics using DEV receipts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tools.rb2break.summarize import tasks
from tools.rbeval.core import digest, load, save_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--targets', type=Path)
    args = parser.parse_args()
    data = tasks(Path(args.baseline))
    rows = []
    insertions = None
    if args.targets is not None:
        with (args.targets/'insertions_10s.csv').open() as stream:
            insertions = [r for r in csv.DictReader(stream)
                          if int(r['seed']) in range(969420, 969432) and r['arm'] == 'native']
    for seed in sorted({seed for seed, _ in data}):
        futures = [data[seed, f] for f in range(8)]
        draws = np.array([task['future_sampling']['demand_rate_draws'][0][0] for task in futures])
        obs = futures[0]['future_sampling']['learned_arrival_observations'][0]
        counts = np.array(obs['total'], dtype=float)
        exposure = obs['exposure_s']
        due = np.array([t['future_sampling']['population_audit']['due'][0] for t in futures])
        initial = np.array([t['future_sampling']['population_audit']['initial_backlog'][0] for t in futures])
        inserted = np.array([t['future_sampling']['population_audit']['admitted'][0] for t in futures])
        pending = np.array([t['future_sampling']['population_audit']['final_pending'][0] for t in futures])
        assert np.array_equal(due+initial, inserted+pending)
        rows.append(dict(seed=seed, observed_rate=counts.tolist(), exposure_s=exposure,
            mle_veh_s=(counts/exposure).tolist(), posterior_mean_veh_s=((counts+.5)/exposure).tolist(),
            k8_mean_draw_veh_s=draws.mean(0).tolist(), k8_draw_sd=draws.std(0, ddof=1).tolist(),
            excess_requests600_vs_mle=(600*(draws.mean(0)-counts/exposure)).tolist(),
            poisson_realization_mean_due=due.mean(0).tolist(),
            native_mean_admitted=inserted.mean(0).tolist(), native_final_pending=pending.mean(0).tolist(),
            initial_backlog=initial[0].tolist()))
        if insertions is not None:
            actual = np.array([sum(int(r['count']) for r in insertions
                if int(r['seed']) == seed and r['entrance'] == entrance)
                for entrance in ('mainline', 'ramp')])
            rows[-1].update(recorded_native_inserted600=actual.tolist(),
                native_admission_bias600=(inserted.mean(0)-actual).tolist())
    excess = np.array([r['excess_requests600_vs_mle'] for r in rows])
    result = dict(worlds=[r['seed'] for r in rows], rows=rows,
        mean_excess_requests600_vs_mle=excess.mean(0).tolist(),
        jeffreys_prior_expected_extra_requests600=[1., 1.],
        note='Mle counter rate is observed, not future demand truth. Finite-K sampling is not a fitted outcome correction.',
        source_sha256=digest(__file__))
    if insertions is not None:
        result.update(targets_sha256=digest(args.targets/'insertions_10s.csv'),
            mean_native_admission_bias600=np.array([r['native_admission_bias600'] for r in rows]).mean(0).tolist())
    save_new(Path(args.output), result)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
