"""Once-only complete frozen A-E test; no estimator fitting or selection API."""
import argparse
import json

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .audit import ARMS, digest, safe
from .core import TIMES, evaluate, finite

FROZEN_SHA = 'ef3e7bced5737c921a65596c566c865152990874be1d9901b8bbb6fb53735d8b'
WM_SHA = 'b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436'


def predictions(tasks, data, models):
    expected = {(s, t, f) for s in range(969432, 969440) for t in TIMES for f in range(8)}
    if set(tasks) != expected:
        raise ValueError('require all 320 immutable K8 five-arm tasks')
    real_cells = list(zip(data['seed'].tolist(), data['time_s'].tolist(), data['arm'].tolist()))
    cells = {(s, t, a) for s in range(969432, 969440) for t in TIMES for a in ARMS}
    if len(real_cells) != 200 or set(real_cells) != cells:
        raise ValueError('require exactly all 200 reserved real cells')
    finite(data['x'], (200, 111))
    finite(data['actions'], (200, 6))
    finite(data['y'], (200,))
    source_ref = None
    terminal_x, window_j, long_j = [], [], []
    order = sorted(tasks)
    for key in order:
        d = tasks[key]
        if ((d['seed'], d['t0'], d['future']) != key or d['start_time'] != key[1]
                or d['end_time'] != 900 or d['arms'] != list(ARMS) or d['K'] != 8 or not d['mix']
                or d['real_outcomes_opened'] or d['wm'][0]['checkpoint_sha256'] != WM_SHA):
            raise ValueError('task identity/binding/clock mismatch')
        if source_ref is None:
            source_ref = d['source_sha256']
        elif d['source_sha256'] != source_ref:
            raise ValueError('runtime sources differ across tasks/backends')
        r = d['results']
        if r['steps'] != 2*(900-key[1]) or set(r['windows']) != {'60', '90'}:
            raise ValueError('missing long or bootstrap clocks')
        long_j.append(finite(r['J'], (5,)))
        for h in (60, 90):
            w = r['windows'][str(h)]
            x = finite(w['x'], (5, 111))
            if not np.all(x[:, -1] == 900-key[1]-h):
                raise ValueError('terminal feature remaining-time disagreement')
            terminal_x.append(x)
            window_j.append(finite(w['J'], (5,)))
    with threadpool_limits(limits=1):
        v = models['V'].model.predict(np.concatenate(terminal_x)).reshape(320, 2, 5)
        direct = models['D'].model.predict(np.c_[data['x'], data['actions']])
    short = np.stack(window_j).reshape(320, 2, 5)+v
    long = np.stack(long_j)
    joined = {key: i for i, key in enumerate(real_cells)}
    indexes = {key: i for i, key in enumerate(order)}
    rows = []
    for s in range(969432, 969440):
        for t in TIMES:
            indices = [indexes[s, t, f] for f in range(8)]
            for ai, arm in enumerate(ARMS):
                i = joined[s, t, arm]
                rows.append(dict(seed=s, t0=t, arm=arm, real=float(data['y'][i]),
                    A=float(long[indices, ai].mean()), B=float(short[indices, 0, ai].mean()),
                    C=float(short[indices, 1, ai].mean()), D=float(direct[i]), E=models['E'][arm, t]))
    return rows


def markdown(result):
    def interval(d, percent=False):
        scale = 100 if percent else 1
        return f"{scale*d['value']:.2f} [{scale*d['ci95'][0]:.2f}, {scale*d['ci95'][1]:.2f}]"
    lines = ['# S-WMVALUE: frozen decisive offline test (2026-10-31)', '',
             f"**VERDICT: {result['verdict']}**; {result['n_worlds']} reserved FIT worlds, "
             f"{result['n_pairs']} paired differences; no final-evaluation claim.",
             'Real target: remaining full fleet+source delay to absolute t900; starts300..540.',
             'V/D/E frozen before reserved outcomes; 111 prefix-only sensed inputs, no source queue feature.',
             'All10 arm pairs; K8 common futures; world bootstrap10,000,seed20261031,95% intervals.', '',
             '| Method | Sign accuracy % [CI] | Spearman [CI] | MAE veh-s [CI] |',
             '|---|---:|---:|---:|']
    names = dict(A='A: WM to900', B='B: WM60 + V', C='C: WM90 + V',
                 D='D: V + present commands', E='E: FIT arm/time mean')
    for m in ('A', 'B', 'C', 'D', 'E'):
        d = result['metrics'][m]
        lines.append(f"| {names[m]} | {interval(d['sign_accuracy'], True)} | "
                     f"{interval(d['spearman'])} | {interval(d['mae'])} |")
    lines += ['', '| Contrast | Sign gain percentage points [CI] | MAE reduction veh-s [CI] |',
              '|---|---:|---:|']
    for key, d in result['contrasts'].items():
        sci, mci = d['sign_ci95'], d['mae_ci95']
        lines.append(f"| {key} | {100*d['sign_gain']:.2f} [{100*sci[0]:.2f}, {100*sci[1]:.2f}] | "
                     f"{d['mae_reduction']:.2f} [{mci[0]:.2f}, {mci[1]:.2f}] |")
    lines += ['', f"Real zero pairs: {result['real_zero_pairs']}; predicted zeros: "+
              ', '.join(f"{m}={result['metrics'][m]['predicted_zero_pairs']}" for m in names)+'.',
              'PASS requires one short method to have strictly positive lower CI bounds for',
              'both improvements against EACH A,D,E; no post-test horizon selection or refit.', '']
    if result['verdict'] == 'PASS':
        lines += ['Recommendation: proceed to a separately registered WM-primary policy pilot.',
                  'Replace policy14 long ES credit by WM60/90 imagination + frozen observable V;',
                  'cooperation credit = counterfactual group WM cost + terminal-V differences.',
                  'Keep real SUMO as the verifier and oracle teacher/reference. Register exact',
                  'policy budget only after the measured trainer-specific GPU-hour estimate.']
    else:
        lines += ['Recommendation: do not plug this frozen V into policy training as a validated fix.',
                  'The registered short-WM+V gate failed; keep WM-primary as the research goal,',
                  'retain the SUMO oracle as teacher/reference, and diagnose on FIT worlds only.',
                  'These eight exposed worlds may not be reused to select a revised value/adapter.']
    lines += ['', 'Adapter: native121-frame history; original PI/CIRCLES memory replay; closed-loop entrants.',
              'V terminals use real sensing projection plus imagined boundary flow; no actual future state.',
              'WM source prior uses recorded warmup queue and prefix flow only, never late true backlog.',
              'Cached A aggregates failed exit-clock/CRN certification; preserved, not silently mixed.',
              'Evidence: tools/wmvalue/evidence/frozen_test.json and arm_predictions.json.',
              'No SUMO, eval/prohibited data, refit, policy training, peer signals or push in this test.']
    assert len(lines) <= 60
    return '\n'.join(lines)+'\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tasks', required=True)
    p.add_argument('--real', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--report', required=True)
    a = p.parse_args()
    out, report = safe(a.out), safe(a.report)
    if out.exists() or not out.parent.is_dir() or not report.parent.is_dir():
        raise ValueError('new result receipt with existing parents required')
    assert digest(safe(a.model)) == FROZEN_SHA, 'frozen model changed; never refit'
    root = safe(a.tasks)
    paths = [root/f's{s}_t{t}_k{f}.json' for s in range(969432, 969440) for t in TIMES for f in range(8)]
    assert all(path.is_file() for path in paths), 'incomplete predictions: no partial test'
    tasks = {}
    for path in paths:
        d = json.loads(path.read_text())
        tasks[d['seed'], d['t0'], d['future']] = d
    with np.load(safe(a.real), allow_pickle=False) as z:
        data = {k: z[k] for k in ('x', 'y', 'actions', 'seed', 'time_s', 'arm')}
    models = joblib.load(safe(a.model))
    assert models['V'].fit_seeds == models['D'].fit_seeds
    assert not set(range(969432, 969440)) & set(models['V'].fit_seeds)
    rows = predictions(tasks, data, models)
    result = evaluate(rows)
    result['model_sha256'] = FROZEN_SHA
    result['real_arrays_sha256'] = digest(a.real)
    result['task_sha256'] = {p.name: digest(p) for p in paths}
    result['policy_training_authorized'] = False
    with out.open('x') as stream:
        json.dump(result, stream, indent=1, allow_nan=False)
    with out.with_name('arm_predictions.json').open('x') as stream:
        json.dump(rows, stream, indent=1, allow_nan=False)
    # The requested report is a generated deliverable; the old report is preserved
    # in its prior git commits and the pipeline never rewrites result receipts.
    report.write_text(markdown(result), encoding='utf-8')
    print(json.dumps(dict(verdict=result['verdict'], winners=result['winners'], pairs=400)), flush=True)


if __name__ == '__main__':
    main()
