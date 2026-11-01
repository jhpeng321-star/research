"""Matched development-only one-factor readout; actual K, no false qualification CI."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from tools.rbeval.core import ARMS, HS, REGIONS, digest, load, save_new
from tools.rb2break.run import DEV


def crps(values, real):
    values = np.array(values, dtype=float)
    return float(np.abs(values-real).mean() - .5*np.abs(values[:, None]-values).mean())


def tasks(path):
    result = {}
    for seed in DEV:
        for future in range(8):
            files = list(path.glob(f'**/w{seed}_k{future}.json'))
            if len(files) > 1:
                raise ValueError('duplicate task')
            if files:
                task = load(files[0])
                assert task['seed'] == seed and task['future'] == future
                assert task['future_sampling']['population_audit']['conserved']
                result[seed, future] = task
    return result


def onset(event):
    if event.get('preexisting'):
        return -1  # category BEFORE all post-t0 onset bins, not a fake exact onset
    if event.get('event'):
        return event['onset_s']
    return float('inf')


def summarize(candidate, baseline, real, *, full_baseline=False):
    keys = sorted(candidate)
    assert keys and set(keys) <= set(baseline)
    by_world = defaultdict(list)
    for seed, future in keys:
        by_world[seed].append(future)
    rows = []
    for seed, futures in sorted(by_world.items()):
        for arm in ARMS:
            factual = real[seed, arm]
            samples = [next(e for e in candidate[seed, f]['episodes'] if e['arm'] == arm) for f in futures]
            parents = [next(e for e in baseline[seed, f]['episodes'] if e['arm'] == arm) for f in futures]
            assert all(e['initial_sha256'] == factual['initial_sha256'] for e in samples+parents)
            for h in HS:
                for label, ensemble in (('candidate', samples), ('baseline', parents)):
                    clock = factual['clocks'][str(h)]
                    row = dict(seed=seed, arm=arm, horizon=h, model=label,
                               cell=[factual['demand'], factual['penetration']], K=len(futures))
                    for component in ('team_delay_vs', 'road_delay_vs', 'source_wait_vs'):
                        values = [e['clocks'][str(h)][component] for e in ensemble]
                        row[component] = dict(mean=float(np.mean(values)), real=clock[component],
                            bias=float(np.mean(values)-clock[component]), crps=crps(values, clock[component]))
                    for region in REGIONS:
                        real_event = factual['bins'][region]['events'][str(h)]
                        events = [e['bins'][region]['events'][str(h)] for e in ensemble]
                        r_onset = onset(real_event)
                        grid = np.r_[-1., np.arange(0, h+1, 10.)]
                        p = np.array([np.mean([onset(e) <= t for e in events]) for t in grid])
                        speed_crps = []
                        speed_bias = []
                        missing = 0
                        for i in range(6, 6+h//10):
                            observed = factual['bins'][region]['speed'][i]
                            values = [e['bins'][region]['speed'][i] for e in ensemble]
                            if observed is None or any(v is None for v in values):
                                missing += 1
                                continue
                            speed_crps.append(crps(values, observed))
                            speed_bias.append(float(np.mean(values)-observed))
                        row[region] = dict(rate=float(np.mean([bool(e['event']) for e in events])),
                            real_rate=float(bool(real_event['event'])),
                            preexisting=float(np.mean([bool(e.get('preexisting')) for e in events])),
                            real_preexisting=float(bool(real_event.get('preexisting'))),
                            onset_cdf_brier=float(np.mean((p-(grid >= r_onset))**2)),
                            speed_crps=float(np.mean(speed_crps)) if speed_crps else None,
                            speed_bias=float(np.mean(speed_bias)) if speed_bias else None,
                            speed_supported_bins=len(speed_crps), speed_missing_bins=missing)
                    rows.append(row)
    pooled = []
    cells = []
    for model in ('candidate', 'baseline'):
        for h in HS:
            selected = [r for r in rows if r['model'] == model and r['horizon'] == h]
            def average(rs, cell=None, arm=None):
                out = dict(model=model, horizon=h, worlds=len({r['seed'] for r in rs}),
                           K_by_world={str(r['seed']): r['K'] for r in rs}, cell=cell, arm=arm)
                for component in ('team_delay_vs', 'road_delay_vs', 'source_wait_vs'):
                    out[component] = {k:float(np.mean([r[component][k] for r in rs]))
                                      for k in ('mean', 'real', 'bias', 'crps')}
                for region in REGIONS:
                    out[region] = {k:float(np.mean([r[region][k] for r in rs if r[region][k] is not None]))
                        if any(r[region][k] is not None for r in rs) else None
                        for k in ('rate', 'real_rate', 'preexisting', 'real_preexisting',
                                  'onset_cdf_brier', 'speed_crps', 'speed_bias')}
                out['CI'] = None
                out['CI_note'] = 'descriptive development screen; cells may be singleton; no qualification claim'
                return out
            pooled.append(average(selected))
            groups = defaultdict(list)
            for row in selected:
                groups[json.dumps(row['cell'], sort_keys=True), row['arm']].append(row)
            for (cell, arm), rs in sorted(groups.items()):
                cells.append(average(rs, json.loads(cell), arm))
    return dict(schema='rb2break-development/1', worlds=sorted(by_world),
                futures={str(k):v for k,v in by_world.items()}, matched_task_count=len(keys),
                qualification=False, rows=rows, pooled=pooled, cells=cells)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    real_path = Path(__file__).resolve().parents[1]/'rbeval/real_observables.json'
    # This pre-existing mixed-split container is parsed only to extract allowlisted
    # records; no reserved record is used by any computation or output.
    real = {(e['seed'], e['arm']):e for e in load(real_path)['episodes'] if e['seed'] in DEV}
    result = summarize(tasks(Path(args.candidate)), tasks(Path(args.baseline)), real)
    result.update(candidate=str(args.candidate), baseline=str(args.baseline),
                  real_receipt_sha256=digest(real_path), source_sha256=digest(__file__))
    save_new(Path(args.output), result)
    print(json.dumps(dict(worlds=result['worlds'], matched_tasks=result['matched_task_count'],
        pooled=[dict(model=r['model'], horizon=r['horizon'],
            bias=r['team_delay_vs']['bias'], source_bias=r['source_wait_vs']['bias'],
            road_bias=r['road_delay_vs']['bias'], crps=r['team_delay_vs']['crps'],
            rate=r['upstream']['rate'], real_rate=r['upstream']['real_rate'],
            speed_crps=r['upstream']['speed_crps'], onset_brier=r['upstream']['onset_cdf_brier'])
            for r in result['pooled']]), indent=1))


if __name__ == '__main__':
    main()
