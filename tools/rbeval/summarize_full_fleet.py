"""Verify immutable FIT full-fleet receipts and summarize distributions by world."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.rbeval.full_fleet import safe, stats


def load_panel(directory):
    directory = safe(directory)
    collection = json.loads((directory/'COLLECTION.json').read_text())
    for name, expected in collection.items():
        path = safe(directory/name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'collection hash differs: {name}')
    complete = json.loads((directory/'COMPLETE.json').read_text())
    rows = complete['rows']
    if sorted(r['seed'] for r in rows) != list(range(969400, 969432)):
        raise ValueError('exact FIT32 complete panel required')
    horizons = {r['horizon_s'] for r in rows}
    if len(horizons) != 1 or any(r['arm'] != 'native' or r['horizon_s'] not in (120, 300, 600) for r in rows):
        raise ValueError('different arm or horizon')
    for row in rows:
        if json.loads((directory/f"{row['seed']}.json").read_text()) != row:
            raise ValueError('per-world receipt differs from COMPLETE')
        demand = row['demand']
        if not np.array_equal(np.array(demand['inserted'])+demand['pending'], demand['due']):
            raise ValueError('source conservation differs')
        if not demand['conserved'] or demand['no_slot'] or demand['dropped']:
            raise ValueError('invalid source admission')
        for e in row['merges']['events']:
            if not 300 < e['time_s'] <= 300+row['horizon_s']:
                raise ValueError('merge event outside supported clock')
    return sorted(rows, key=lambda r: r['seed'])


def analyze(rows, targets):
    horizon = rows[0]['horizon_s']
    end = 300+horizon
    with safe(targets/'ramp_merges.csv').open(newline='') as f:
        real_merges = list(csv.DictReader(f))
    with safe(targets/'insertions_10s.csv').open(newline='') as f:
        real_insertions = list(csv.DictReader(f))
    with safe(targets/'ramp_delay_world.csv').open(newline='') as f:
        real_delays = {(int(r['seed']), int(float(r['horizon_s']))): float(r['ramp_delay_vs'])
                       for r in csv.DictReader(f) if r['arm'] == 'native'}
    predicted_delay = np.array([r['delay']['predicted_ramp_veh_s'] for r in rows])
    recorded_delay = np.array([float(r['delay']['recorded']['ramp_delay_vs']) for r in rows])
    time_errors, position_errors = [], []
    missed = extra = recorded_cohort = predicted_cohort = 0
    insertion = []
    for row in rows:
        seed = row['seed']
        initial = {int(i) for i in row['road_delay_per_identity'] if int(i) >= 0}
        rm = {int(r['vid']): r for r in real_merges if int(r['seed']) == seed and r['arm'] == 'native'
              and int(r['vid']) in initial and 300 < float(r['t_merge_s']) <= end}
        pm = {r['identity']: r for r in row['merges']['events'] if r['identity'] >= 0}
        common = rm.keys() & pm.keys()
        recorded_cohort += len(rm)
        predicted_cohort += len(pm)
        missed += len(rm.keys()-pm.keys())
        extra += len(pm.keys()-rm.keys())
        time_errors.extend(pm[i]['time_s']-float(rm[i]['t_merge_s']) for i in common)
        position_errors.extend(pm[i]['position_m']-float(rm[i]['s_merge_m']) for i in common)
        actual = [sum(int(r['count']) for r in real_insertions if int(r['seed']) == seed
                      and r['arm'] == 'native' and r['entrance'] == e and float(r['t_start_s']) < end)
                  for e in ('mainline', 'ramp')]
        insertion.append(dict(seed=seed, recorded=actual, predicted=row['demand']['inserted']))
    upper = float(np.nextafter(np.float32(75.44), np.float32(np.inf)))
    out_of_zone = [dict(seed=r['seed'], **e) for r in rows for e in r['merges']['events']
                   if not 0 <= e['position_m'] <= upper]
    checkpoint_scores = {}
    for h in (120, 300, 600):
        if all(str(h) in r.get('checkpoints', {}) for r in rows):
            x = np.array([r['checkpoints'][str(h)]['ramp_total'] for r in rows])
            y = np.array([real_delays[r['seed'], h] for r in rows])
            checkpoint_scores[str(h)] = dict(predicted=stats(x), recorded=stats(y),
                mean_bias=float((x-y).mean()), marginal_wasserstein1=float(np.abs(np.sort(x)-np.sort(y)).mean()))
    return dict(worlds=len(rows), horizons_s=[horizon], arms=['native'],
        source_versions=dict(cf=sorted({r['cf_model'] for r in rows}),
                             native_model=sorted({r.get('native_model', 'head') for r in rows}),
                             settings=sorted({r.get('settings', 'unspecified') for r in rows}),
                             entry_phase=sorted({r.get('entry_phase', 'before') for r in rows}),
                             lc_memory=sorted({r['lc_memory'] for r in rows})),
        ramp_origin_full_fleet_delay_veh_s=dict(recorded=stats(recorded_delay), predicted=stats(predicted_delay),
            mean_bias=float((predicted_delay-recorded_delay).mean()),
            marginal_wasserstein1=float(np.abs(np.sort(predicted_delay)-np.sort(recorded_delay)).mean()),
            secondary_paired_mae=float(np.abs(predicted_delay-recorded_delay).mean()),
            predicted_source=stats([r['delay']['ramp_source_veh_s'] for r in rows])),
        t0_merge_events=dict(recorded=recorded_cohort, predicted=predicted_cohort, matched=len(time_errors),
            missed=missed, extra=extra, time_error_s=stats(time_errors), position_error_m=stats(position_errors),
            timing_mae_s=float(np.abs(time_errors).mean()) if time_errors else None,
            position_mae_m=float(np.abs(position_errors).mean()) if position_errors else None),
        full_fleet_merge_counts=dict(recorded=stats([r['merges']['recorded'] for r in rows]),
                                     predicted=stats([r['merges']['predicted'] for r in rows])),
        insertions=dict(per_world=insertion,
            recorded_total=np.array([r['recorded'] for r in insertion]).sum(0).tolist(),
            predicted_total=np.array([r['predicted'] for r in insertion]).sum(0).tolist()),
        geometric_overlap=dict(available=all('body_overlap' in r for r in rows),
            pair_samples=sum(r.get('body_overlap', {}).get('pair_samples', 0) for r in rows),
            worlds_with_overlap=[r['seed'] for r in rows if r.get('body_overlap', {}).get('pair_samples', 0)]),
        merge_geometry_violations=out_of_zone,
        source_conservation=dict(worlds_passed=len(rows), dropped=0, no_slot=0),
        checkpoint_ramp_delays=checkpoint_scores,
        not_qualified=True,
        limitations=['One future per world; marginal distributions do not identify same-state variability.',
            'Full-fleet insertion counts use autonomous stochastic arrivals, not replayed future demand.',
            'Historical 562.4 veh-s is a different panel/contract; compare recorded per-world values.',
            'No breakdown, controller-shift, or distributional qualification gate is claimed.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', required=True)
    parser.add_argument('--targets', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = load_panel(Path(args.panel))
    targets = safe(args.targets)
    result = analyze(rows, targets)
    result['inputs_sha256'] = {n: hashlib.sha256((targets/n).read_bytes()).hexdigest()
                              for n in ('ramp_merges.csv', 'insertions_10s.csv', 'ramp_delay_world.csv')}
    result['collection_sha256'] = hashlib.sha256((Path(args.panel)/'COLLECTION.json').read_bytes()).hexdigest()
    result['scorer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    safe(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', newline='\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('insertions',)}))


if __name__ == '__main__':
    main()
