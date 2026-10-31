"""Write the amended necessary-condition test exactly once; never fit models."""
import argparse
import json
from pathlib import Path
import subprocess

import joblib
import numpy as np

from .audit import digest, safe
from .core import TIMES
from .evaluate_frozen import FROZEN_SHA, predictions
from .evaluate_short import evaluate_short
from .orchestrate import save_new, task_key

OLD_WORKER = '13517d8dbb632cd7ff3b378e4243a9cefafd097805629d9382e33202b207c02a'


def markdown(result):
    def ci(d, scale=1):
        return f"{scale*d['value']:.2f} [{scale*d['ci95'][0]:.2f}, {scale*d['ci95'][1]:.2f}]"
    lines = ['# S-WMVALUE frozen decisive offline test', '',
        f"**VERDICT: {result['verdict']}** (MAIN21:12 short-prefix-first amendment).",
        'Eight reserved FIT worlds969432..969439;200arm cells;400paired differences.',
        'Starts300/360/420/480/540; full remaining fleet+source delay to absolute900s.',
        'V/D/E unchanged SHAef3e7bce;111causal sensed features; no source queue input.',
        'K8 WM2+mix; identical long stochastic plans truncated only after90s.',
        'World-cluster bootstrap10,000,seed20261031;95% percentile intervals.', '',
        '| Method | Sign accuracy % [CI] | Spearman [CI] | MAE veh-s [CI] |',
        '|---|---:|---:|---:|',
        '| A: WM to900 | NA | NA | NA |']
    for m, name in [('B', 'WM60 + V'), ('C', 'WM90 + V'), ('D', 'V + present commands'),
                    ('E', 'FIT arm/time mean')]:
        d = result['metrics'][m]
        lines.append(f"| {m}: {name} | {ci(d['sign_accuracy'], 100)} | {ci(d['spearman'])} | {ci(d['mae'])} |")
    lines += ['', '| Contrast | Sign gain pp [CI] | MAE reduction veh-s [CI] |', '|---|---:|---:|']
    for key, d in result['contrasts'].items():
        lo, hi = d['sign_ci95']
        ml, mh = d['mae_ci95']
        lines.append(f"| {key} | {100*d['sign_gain']:.2f} [{100*lo:.2f}, {100*hi:.2f}] | "
                     f"{d['mae_reduction']:.2f} [{ml:.2f}, {mh:.2f}] |")
    lines += ['', 'Positive strictly-above-zero lower bounds are required for BOTH sign and MAE',
        'against EACH D/E, then A. Short methods surviving D/E: '+str(result['survivors'])+'.']
    if result['verdict'] == 'FAIL':
        lines += ['Neither short method clears the necessary D/E gate; the full A/D/E',
            'conjunction therefore FAILS regardless of A. Long-A completion is unnecessary.',
            'Recommendation: do not plug this frozen V into policy training as a validated fix.',
            'Keep WM-primary as the research goal; diagnose value/rollout mismatch on FIT only.',
            'Preserve SUMO oracle as teacher/reference. These exposed worlds cannot tune a revision.']
    else:
        lines += ['The D/E condition passed; this is NOT a full PASS. Complete matched A next.',
                  'No policy integration or training is authorized before the full gate passes.']
    lines += ['', 'Real/imagined sensing parity, prefix causality, history, entries/exits, clocks and CRN',
        'passed CPU gates before reserved inference. Only registered worker-driver versions differ.',
        'Existing long outputs supply their own short checkpoints; active jobs were not duplicated.',
        'A is NA because the complete320long-task test has not run; no partial A efficacy claim.',
        'Evidence: tools/wmvalue/evidence/short_test.json and short_arm_predictions.json.',
        'No SUMO, eval/prohibited paths, refit, peer PID signals, policy training or push.']
    assert len(lines) <= 60
    return '\n'.join(lines)+'\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--short', required=True)
    p.add_argument('--original', required=True)
    p.add_argument('--real', required=True)
    p.add_argument('--model', required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    evidence = root/'tools/wmvalue/evidence'
    out = evidence/'short_test.json'
    assert not out.exists(), 'once-only amended test already written'
    assert not (evidence/'frozen_test.json').exists(), 'full test already written'
    assert digest(a.model) == FROZEN_SHA, 'never refit'
    assert digest(a.real) == 'a61d1846534c1e89c238d70eb11c001a01528e6b940bc9b21b78e8c03469d386'
    short, original = safe(a.short), safe(a.original)
    ack = json.loads((short/'ACK.json').read_text())
    tasks, paths = {}, []
    bindings = None
    for s in range(969432, 969440):
        for t in TIMES:
            for f in range(8):
                key = task_key((s, t, f))
                old, new = original/'tasks'/(key+'.json'), short/'tasks'/(key+'.json')
                assert not (old.exists() and new.exists()), 'duplicate inference needs ownership audit'
                path = old if old.exists() else new
                d = json.loads(path.read_text())
                assert d['gate_sha256'] == ack['gate_sha256']
                binding = {k: d[k] for k in ('prefixes_sha256', 'candidate_sha256', 'export_sha256')}
                binding['wm'] = [{k: w[k] for k in ('checkpoint_sha256', 'config_sha256')} for w in d['wm']]
                if bindings is None:
                    bindings = binding
                assert binding == bindings, 'cross-task frozen runtime/input binding mismatch'
                tasks[s, t, f] = d
                paths.append(path)
    with np.load(safe(a.real), allow_pickle=False) as z:
        data = {k: z[k] for k in ('x', 'y', 'actions', 'seed', 'time_s', 'arm')}
    models = joblib.load(safe(a.model))
    assert models['V'].fit_seeds == models['D'].fit_seeds
    assert not set(range(969432, 969440)) & set(models['V'].fit_seeds)
    rows = predictions(tasks, data, models, short_only=True,
                       allowed_workers={OLD_WORKER, ack['worker_sha256']})
    result = evaluate_short(rows)
    result.update(model_sha256=FROZEN_SHA, real_arrays_sha256=digest(a.real),
                  task_sha256={str(p): digest(p) for p in paths},
                  source_sha256={n: digest(Path(__file__).with_name(n)) for n in
                      ('evaluate_short.py', 'evaluate_frozen.py', 'core.py', 'finalize_short.py')})
    save_new(out, result)
    save_new(evidence/'short_arm_predictions.json', rows)
    report = root/'docs/paper_results/WMVALUE.md'
    report.write_text(markdown(result), encoding='utf-8')
    files = [out, evidence/'short_arm_predictions.json', report]
    if result['verdict'] == 'FAIL':
        line = ('- S-WMVALUE FROZEN DECISIVE VERDICT FAIL (MAIN21:12 necessary-condition stop; '
            'c4/wmvalue;8reservedFITworlds969432..969439,200cells,400pairs,K8): '
            'B/C60/90+frozenV fail to clear BOTH sign and MAE positive95% lower bounds against '
            'EACH D/E; full A/D/E conjunction FAILS regardless of A, long-A completion waived. '
            'Same10000world-bootstrap/seed20261031; V/D/E SHAef3e7bce unchanged,no refit. '
            'A NA,not a partial-test estimate. Table/contrasts docs/paper_results/WMVALUE.md; '
            f'resultSHA{digest(out)}. WM-primary research goal retained; no validated policy plug-in. '
            'No SUMO/eval/prohibited paths/peer signals/policy training/push.\n')
        registration = safe('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
        with registration.open('a', encoding='utf-8') as stream:
            stream.write(line)
        receipt = evidence/'final_verdict_registration.md'
        with receipt.open('x') as stream:
            stream.write(line)
        files.append(receipt)
    if subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=root, text=True).strip():
        raise RuntimeError('unrelated staged work: verdict preserved, automatic commit paused')
    for cmd in [['git', 'status', '--short'], ['git', 'diff', '--', str(report.relative_to(root))],
                ['git', 'log', '--oneline', '-10']]:
        subprocess.run(cmd, cwd=root, check=True)
    subprocess.run(['git', 'add', *[str(p.relative_to(root)) for p in files]], cwd=root, check=True)
    subprocess.run(['git', 'diff', '--cached', '--check'], cwd=root, check=True)
    subprocess.run(['git', 'commit', '-m', 'WMVALUE: report frozen short-first '+result['verdict']+' gate'],
                   cwd=root, check=True)
    print(json.dumps(dict(verdict=result['verdict'], survivors=result['survivors'])), flush=True)


if __name__ == '__main__':
    main()
