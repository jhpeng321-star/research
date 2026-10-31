"""Frozen HDV60 common-cohort comparison; factual endpoints are scoring-only.

Contract: docs/rbint/HDV60_DESIGN.md. No inference, fitting or simulator imports.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from tools.rbeval.core import ARMS, HERE, SEEDS, digest, load, safe, save_new
from tools.rbeval.score import WorldBootstrap, read_predictions


def read_short_baseline(inputs, real):
    targets = {(r['seed'], r['arm']): r for r in real}
    tasks, receipts, contract = {}, [], None
    for root in inputs:
        for path in sorted(safe(root).rglob('w*_k*.json')):
            task = load(path)
            identity_path = path.parent/'IDENTITY.json'
            identity = load(identity_path)
            assert task['schema'] == identity['schema'] == 'wmphys-hdv60/1'
            assert task['identity_sha256'] == digest(identity_path)
            assert identity['model'] == 'wm2_hdv60'
            assert identity['wm_sha256'] == 'b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436'
            assert identity['t0'] == 300 and identity['horizon'] == 60 and identity['K'] == 8
            assert identity['arms'] == task['arms'] == list(ARMS)
            assert identity['real_receipt_sha256'] == digest(HERE/'real_observables.json')
            selected = {k: v for k, v in identity.items() if k not in ('shard', 'shards')}
            if contract is None:
                contract = selected
            assert contract == selected, 'mixed short-baseline identities'
            key = task['seed'], task['future']
            assert key[0] in SEEDS and key[1] in range(8) and key not in tasks
            assert len(task['episodes']) == 5 and {e['arm'] for e in task['episodes']} == set(ARMS)
            for episode in task['episodes']:
                assert (episode['seed'], episode['future']) == key
                assert episode['initial_sha256'] == targets[key[0], episode['arm']]['initial_sha256']
            tasks[key] = task
            receipts.append(dict(path=str(path), sha256=digest(path), identity_sha256=digest(identity_path)))
    complete = [s for s in SEEDS if all((s, k) in tasks for k in range(8))]
    pred = {(s, arm): [next(e for e in tasks[s, k]['episodes'] if e['arm'] == arm)
                       for k in range(8)] for s in complete for arm in ARMS}
    return pred, dict(tasks=len(tasks), complete_worlds=complete, contract=contract, inputs=receipts)


def endpoint_map(episode, factual):
    assert episode['initial_sha256'] == factual['initial_sha256']
    end = episode['endpoint60']
    assert end['schema'] == 'wmphys-predicted-endpoints/1' and end['horizon_s'] == 60
    assert end['factual_future_used'] is False
    assert end['start_sha256'] == factual['start_sha256']
    assert end['history_sha256'] == factual['history_sha256']
    names = end['vehicle_ids']
    assert len(set(names)) == len(names)
    assert all(len(end[k]) == len(names) for k in ('is_hdv', 'survived', 's_m'))
    observed = {v['vehicle_id']: v for v in factual['vehicles']}
    assert set(names) == set(observed), 'different original-ID endpoint cohort'
    result = {}
    for name, hdv, alive, value in zip(names, end['is_hdv'], end['survived'], end['s_m']):
        assert isinstance(hdv, bool) and hdv == observed[name]['is_hdv']
        assert isinstance(alive, bool) and (value is not None) == alive
        assert value is None or np.isfinite(value)
        result[name] = value
    return result


def world_errors(candidate, baseline, factual, worlds):
    target = {(e['seed'], e['arm']): e for e in factual}
    rows = []
    for seed in worlds:
        c_errors, b_errors, counts, arms = [], [], Counter(), []
        for arm in ARMS:
            truth = target[seed, arm]
            c = [endpoint_map(e, truth) for e in candidate[seed, arm]]
            b = [endpoint_map(e, truth) for e in baseline[seed, arm]]
            attrition = Counter()
            for vehicle in truth['vehicles']:
                if not vehicle['is_hdv']:
                    continue
                name, value = vehicle['vehicle_id'], vehicle['s_m']
                attrition['t0_hdv'] += 1
                if value is None:
                    attrition['factual_'+str(vehicle['censoring'])] += 1
                    continue
                assert np.isfinite(value) and vehicle['censoring'] is None
                cv, bv = [e[name] for e in c], [e[name] for e in b]
                c_ok, b_ok = all(v is not None for v in cv), all(v is not None for v in bv)
                if not c_ok:
                    attrition['candidate_not_allK8'] += 1
                if not b_ok:
                    attrition['baseline_not_allK8'] += 1
                if not c_ok or not b_ok:
                    attrition['prediction_union_censored'] += 1
                    continue
                c_errors.append(abs(float(np.median(cv))-value))
                b_errors.append(abs(float(np.median(bv))-value))
                attrition['common_hdv'] += 1
            counts.update(attrition)
            arms.append(dict(arm=arm, **attrition))
        valid = bool(c_errors)
        rows.append(dict(seed=seed, counts=dict(counts), arms=arms,
            candidate_median_m=float(np.median(c_errors)) if valid else None,
            baseline_median_m=float(np.median(b_errors)) if valid else None,
            median_difference_m=float(np.median(c_errors)-np.median(b_errors)) if valid else None,
            candidate_mae_m=float(np.mean(c_errors)) if valid else None,
            baseline_mae_m=float(np.mean(b_errors)) if valid else None,
            mae_difference_m=float(np.mean(c_errors)-np.mean(b_errors)) if valid else None))
    return rows


def summarize(rows, bootstrap):
    features = [[(r['seed'], r[key]) for r in rows]
                for key in ('median_difference_m', 'mae_difference_m')]
    bands = bootstrap.bands(features)
    primary, sensitivity = bands
    counts = Counter()
    for row in rows:
        counts.update(row['counts'])
    return dict(worlds=[r['seed'] for r in rows], counts=dict(counts),
        primary=primary, sensitivity=sensitivity,
        point_nonregression=primary['mean'] is not None and primary['mean'] <= 0,
        supported_nonregression=primary['high'] is not None and primary['high'] <= 0,
        status='AVAILABLE_CASE_DIAGNOSTIC', per_world=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', nargs='+', required=True)
    p.add_argument('--baseline', nargs='+', required=True)
    p.add_argument('--factual', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = safe(args.out)
    if out.exists() or not out.parent.is_dir():
        raise ValueError('new report in an existing parent required')
    real = load(HERE/'real_observables.json')['episodes']
    factual = load(safe(args.factual))
    assert factual['schema'] == 'wmphys-factual-endpoints/1' and factual['horizon_s'] == 60
    assert factual['t0_s'] == 300
    assert len(factual['episodes']) == len({(e['seed'], e['arm']) for e in factual['episodes']})
    candidate, ca = read_predictions(args.candidate, real, partial=True)
    baseline, ba = read_short_baseline(args.baseline, real)
    common = sorted(set(ca['complete_worlds']) & set(ba['complete_worlds']))
    bootstrap = WorldBootstrap(real, 10000)
    summaries = {}
    for split, worlds in (('development12', [s for s in common if s < 969432]),
                           ('reserved8', [s for s in common if s >= 969432])):
        summaries[split] = summarize(world_errors(candidate, baseline, factual['episodes'], worlds), bootstrap)
    out.mkdir()
    report = dict(schema='wmphys-hdv60-comparison/1', summaries=summaries,
        candidate_audit=ca, baseline_audit=ba, factual_sha256=digest(args.factual),
        source_sha256=digest(Path(__file__)), design_sha256=digest(Path(__file__).resolve().parents[2]/'docs/rbint/HDV60_DESIGN.md'),
        bootstrap=10000, seed=20261031, stratified='demand x penetration x split',
        uncertainty='paired whole-world maxstat family2; conditional on common cohort and saved K8',
        qualification='endpoint diagnostic only; delay/breakdown/credibility gates separate')
    save_new(out/'HDV60_COMPARISON.json', report)
    print(json.dumps({k: {n: v[n] for n in ('worlds', 'primary', 'point_nonregression', 'supported_nonregression')}
                      for k, v in summaries.items()}))


if __name__ == '__main__':
    main()
