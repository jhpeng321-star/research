"""Registered whole-world analysis of fixed-K factorial replay receipts."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'tools/longloss')]
from tools.longloss import amended
from tools.lrdiag.summarize import clean, interval, metrics as base_metrics, write_csv
from tools.coopw import horizon_gate

VARIANTS = ('baseline', 'ramp_adapt_off', 'entrant_on', 'entrant_on_ramp_off')
HORIZONS = (10, 30, 60, 90, 120, 180, 240, 300)


def metrics(a):
    a = np.asarray(a, float)
    return dict(base_metrics(a), error_sd=float((a[:, 0]-a[:, 1]).std()))


def load(raw, panel, variants, partial=False):
    bins = defaultdict(dict)
    hashes = {}
    for host in ('cpuP', 'cpuQ'):
        for path in sorted((raw/host/'results_v4').glob(f'{panel}_*_s600.json')):
            doc = json.loads(path.read_text())
            if doc['variant'] not in variants:
                continue
            key = (doc['pair_key'], doc['variant'])
            assert doc['draw'] not in bins[key], ('duplicate result', key, doc['draw'])
            assert doc['panel'] == panel and doc['steps'] == 600
            assert doc['entrant_control'] == doc['variant'].startswith('entrant_on')
            bins[key][doc['draw']] = doc
            hashes[str(path.relative_to(raw))] = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = 36 if panel == 'e3' else 48
    groups = sorted({key for key, _ in bins})
    complete = [key for key in groups if all(set(bins.get((key, v), {})) == set(range(8)) for v in variants)]
    coverage = dict(panel=panel, expected_groups=expected, complete_groups=len(complete),
                    seen_groups=len(groups), expected_results=expected*len(variants)*8,
                    results=sum(map(len, bins.values())), K=8, variants=variants)
    if not partial:
        assert len(complete) == expected, coverage
    return bins, complete, coverage, hashes


def make_gates(bins, groups, variants, threshold):
    gates = {}
    paired_rows = []
    breakdown_rows = []
    for variant in variants:
        rows = []
        for key in groups:
            draws = [bins[key, variant][k] for k in range(8)]
            names = list(draws[0]['arms'])
            ref = names[0]
            assert ref == 'native' or ref.endswith('_native'), ref
            pairs = {}
            for h in HORIZONS:
                pairs[str(h)] = []
                for arm in names[1:]:
                    def delay(doc, name):
                        a = doc['arms'][name]
                        if h in (120, 300):
                            return np.asarray(a['delay'][str(h)])
                        # Same mathematical estimand/reduction for old and new trajectories.
                        wm = math.fsum(r[3] for r in a['timeline'] if r[1] == 'HDV' and r[2] <= h)
                        return np.asarray([wm, a['delay'][str(h)][1]])
                    arr = np.asarray([delay(d, arm)-delay(d, ref) for d in draws])
                    assert np.ptp(arr[:, 1]) == 0
                    pair = arr.mean(0).tolist()
                    pairs[str(h)].append(pair)
                    paired_rows.append(dict(world=key.split('@')[0], pair_key=key, variant=variant,
                        arm=arm, horizon_s=h, wm_dJ=pair[0], real_dJ=pair[1],
                        error=pair[0]-pair[1], wm_sd_K=float(arr[:, 0].std(ddof=1))))
            rows.append(dict(pair_key=key, arm_names=names[1:], arms=len(names)-1, dJ=pairs))
            for arm in names:
                for region in ('upstream', 'merge'):
                    sensors = [d['arms'][arm]['breakdown'][region] for d in draws]
                    real = sensors[0]['real']
                    assert all(s['real'] == real for s in sensors)
                    events = [s['wm']['event'] for s in sensors]
                    probability = float(np.mean(events)) if all(e is not None for e in events) else None
                    times = [s['wm']['onset_s'] for s in sensors if s['wm']['event']]
                    majority = probability > .5 if probability is not None else None
                    onset = float(np.median(times)) if majority and times else None
                    error = (onset-real['onset_s'] if onset is not None and real['event'] else None)
                    breakdown_rows.append(dict(world=key.split('@')[0], pair_key=key,
                        variant=variant, arm=arm, region=region, real_event=real['event'],
                        wm_event_probability=probability, wm_event_majority=majority,
                        real_onset_s=real['onset_s'], wm_onset_s=onset, onset_error_s=error))
        horizons = {}
        for h in HORIZONS:
            per_group = [r['dJ'][str(h)] for r in rows]
            sign, ci = horizon_gate.bootstrap(per_group, horizon_gate.sign_agreement, seed=h)
            corr, corr_ci = horizon_gate.bootstrap(per_group, horizon_gate.pearson, seed=h+1)
            horizons[str(h)] = dict(sign_agreement=sign, sign_ci95=ci,
                pearson=corr, pearson_ci95=corr_ci, n_groups=len(rows), n_pairs=sum(map(len, per_group)),
                verdict='BASE' if h == 10 else ('PASS' if sign is not None and sign >= threshold else 'FAIL'))
        gates[variant] = dict(schema='entrant_horizon_gate/1', threshold=threshold,
            rule=horizon_gate.__doc__, panel_seed=20261026, groups=len(groups), samples=8,
            horizons=horizons, rows=rows)
    return gates, paired_rows, breakdown_rows


def analyze_rows(rows, variants):
    summary = []
    for variant in variants:
        arms = (['mappo', 'ours_c1', 'pooled'] if rows and rows[0]['arm'] in ('mappo', 'ours_c1')
                else ['pooled'])
        for arm in arms:
            for h in HORIZONS:
                select = [r for r in rows if r['variant'] == variant and r['horizon_s'] == h
                          and (arm == 'pooled' or r['arm'] == arm)]
                worlds = sorted({r['world'] for r in select})
                arrays = [np.asarray([[r['wm_dJ'], r['real_dJ']] for r in select if r['world'] == w])
                          for w in worlds]
                if not arrays:
                    continue
                point = metrics(np.concatenate(arrays))
                rng = np.random.default_rng(0)
                boot = [metrics(np.concatenate([arrays[i] for i in rng.integers(len(worlds), size=len(worlds))]))
                        for _ in range(2000)]
                row = dict(variant=variant, arm=arm, horizon_s=h, n_worlds=len(worlds),
                           n_pairs=sum(map(len, arrays)), **point)
                for k in point:
                    row[k+'_lo95'], row[k+'_hi95'] = interval([b[k] for b in boot])
                summary.append(row)
    return summary


def regressions(reference, candidate):
    """Paired whole-world intervals: all improvements are oriented positive."""
    rw, cw = amended.by_world(reference), amended.by_world(candidate)
    def stat(which, fn):
        def delta(r, c):
            a = metrics(np.concatenate([amended.arrays(x, '300') for x in r]))
            b = metrics(np.concatenate([amended.arrays(x, '300') for x in c]))
            return fn(a[which])-fn(b[which])
        point, ci = amended.boot(rw, cw, delta)
        return dict(improvement=point, ci95=ci, significant_regression=ci[1] < 0)
    return {name: stat(key, fn) for name, key, fn in (
        ('absolute_bias', 'bias', abs), ('MAE', 'mae', float), ('RMSE', 'rmse', float),
        ('error_sd', 'error_sd', float), ('sign', 'sign', lambda x: -x),
        ('correlation', 'pearson', lambda x: -x), ('slope_alignment', 'slope', lambda x: abs(x-1)))}


def breakdown_summary(rows):
    out = []
    for variant in sorted({r['variant'] for r in rows}):
        for arm in ('native', 'mappo', 'ours_c1', 'pooled'):
            for region in ('upstream', 'merge'):
                rs = [r for r in rows if r['variant'] == variant and r['region'] == region
                      and (arm == 'pooled' or r['arm'] == arm)
                      and r['real_event'] is not None and r['wm_event_probability'] is not None]
                if not rs:
                    continue
                errors = [r['onset_error_s'] for r in rs if r['onset_error_s'] is not None]
                row = dict(variant=variant, arm=arm, region=region, n=len(rs),
                    real_occurrence=float(np.mean([r['real_event'] for r in rs])),
                    wm_expected_occurrence=float(np.mean([r['wm_event_probability'] for r in rs])),
                    wm_majority_occurrence=float(np.mean([r['wm_event_majority'] for r in rs])),
                    onset_MAE_s=float(np.mean(np.abs(errors))) if errors else None)
                for a, b, label in ((True, True, 'TP'), (True, False, 'FN'),
                                    (False, True, 'FP'), (False, False, 'TN')):
                    row[label] = sum(r['real_event'] == a and r['wm_event_majority'] == b for r in rs)
                out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=ROOT/'evidence_entrant/raw')
    parser.add_argument('--out', type=Path, default=ROOT/'evidence_entrant')
    parser.add_argument('--panel', choices=('e3', 'pairs300'), default='e3')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    bins, groups, coverage, hashes = load(args.raw, args.panel, args.variants, args.partial)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/f'{args.panel}_COVERAGE.json').write_text(json.dumps(coverage, indent=1))
    if not groups:
        print(json.dumps(coverage));return
    threshold_path = args.raw/'cpuP/runtime/configs/coopw_horizon_gate_13E_b2.json'
    threshold_doc = json.loads(threshold_path.read_text())
    gates, rows, sensors = make_gates(bins, groups, args.variants, threshold_doc['threshold'])
    summary = analyze_rows(rows, args.variants)
    write_csv(args.out/f'{args.panel}_paired_worlds.csv', rows)
    write_csv(args.out/f'{args.panel}_metrics.csv', summary)
    write_csv(args.out/f'{args.panel}_breakdown_worlds.csv', sensors)
    write_csv(args.out/f'{args.panel}_breakdown_summary.csv', breakdown_summary(sensors))
    verdicts = {}
    for variant, gate in gates.items():
        (args.out/f'{args.panel}_{variant}_GATE.json').write_text(json.dumps(clean(gate), indent=1))
        if variant != 'baseline' and 'baseline' in gates:
            verdicts[variant] = amended.evaluate(gates['baseline'], gate)
            verdicts[variant]['no_300s_regression'] = regressions(gates['baseline'], gate)
    (args.out/f'{args.panel}_VERDICTS.json').write_text(json.dumps(clean(verdicts), indent=1))
    (args.out/f'{args.panel}_RESULT_SHA256.json').write_text(json.dumps(hashes, sort_keys=True, indent=1))
    print(json.dumps(coverage))
    print(json.dumps(clean({v: d['amended_rule'] for v, d in verdicts.items()})))


if __name__ == '__main__':
    main()
