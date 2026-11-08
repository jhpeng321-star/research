"""c4/learnfix demo-BC extraction / featurizer parity (host, CPU, zero SUMO, read-only). See tools/coopmwm/demo_bc.py.

extract: python ops/demo_extract.py extract --run-root <CLEAN5C root (CONFIG, u0 checkpoint, B_track2_000)>
            --guidance GUIDANCE.json --ponset P1.jsonl ... --bank BANK1.jsonl ... --verify-root ROOT1 ... --out DEMO.json
parity:  python ops/demo_extract.py parity --run-root <root> --job <jobs/t2c_warm_u001_B_x> --out PARITY.json
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get('LF_THREADS', '2')))


def _runtime(run_root, guidance=None):
    from tools.coopmwm import clean4_restart as c4, launch_production as production
    from tools.coopmwm.launch_training import read
    from tools.coopmwm.v2v_runtime import load_checkpoint
    cfg = read(Path(run_root)/'CONFIG.json')
    budget = cfg.get('clean4_budget') or {}
    if budget.get('run') in c4.APPROVED_BY_RUN:
        c4.APPROVED_BY_RUN[budget['run']] = budget['approved_sim_s']
    native = production.backend_config(cfg)
    if guidance is not None:
        from tools.coopmwm.bank_growth import apply_guidance
        native = apply_guidance(native, guidance)
    base = load_checkpoint(os.path.join(run_root, 'B_checkpoint_000.pt'), config=native)[0]
    state = torch.load(os.path.join(run_root, 'B_track2_000.pt'), map_location='cpu', weights_only=False)['worker']
    return native, base, state


def _activations(job):
    return json.load(gzip.open(os.path.join(job, 'SUMMARY.json.gz'), 'rt')).get('activations') or []


def extract(a):
    from tools.coopmwm import demo_bc
    guidance = json.loads(Path(a.guidance).read_text(encoding='utf-8'))
    admitted = [p for p in guidance['situational']['pairs'] if p['admitted'] and p['family'].startswith('mut_')]
    bins = {}
    for p in admitted:
        bins.setdefault(p['family'], []).append(p['bin'])
    onset = {}
    for f in a.ponset:
        for line in open(f):
            r = json.loads(line)
            if r.get('mode') == 'privileged':
                onset[(r['root'], r['cell'], int(r['seed']), r['arm'], int(r['t']))] = r
    effect = {}
    for f in a.bank:
        for line in open(f):
            r = json.loads(line)
            effect[(r['root'].rstrip('/').split('/')[-1], r['cell'], int(r['seed']), r['family'])] = float(r['diff_vs_incumbent'])
    native, base, state = _runtime(a.run_root, guidance)
    plans, skipped = [], []
    for (root, cell, seed, arm, t), r in sorted(onset.items()):
        if arm not in bins or r['bin'] not in bins[arm]:
            continue
        d = effect.get((root, cell, seed, arm))
        if d is None:
            skipped.append(dict(root=root, cell=cell, seed=seed, family=arm, t=t, reason='no per-seed bank effect'))
            continue
        plans.append(dict(root=root, cell=cell, seed=seed, family=arm, t=t, holder=r['holder'], diff=d))
    demo_bc.assert_seeds_disjoint({p['seed'] for p in plans})
    help_max = max([-p['diff'] for p in plans if p['diff'] < 0] or [1.])
    hurt_max = max([p['diff'] for p in plans if p['diff'] > 0] or [1.])
    roots = {Path(r).name: r for r in a.verify_root}
    items, cache = [], {}
    for p in plans:
        jobs = roots[p['root']]+'/jobs/'
        plan_job = jobs+f"{p['cell']}_s{p['seed']}_{p['family']}"
        acts = [x for x in _activations(plan_job) if int(x['t']) == p['t']]
        members = list(acts[0]['members']) if acts else [p['holder']]
        tapes = ([(plan_job, 'act', -p['diff']/help_max)] if p['diff'] < 0 else
                 [(jobs+f"{p['cell']}_s{p['seed']}_{arm}", 'none', p['diff']/hurt_max) for arm in ('wall_g_solo2', 'u0')]
                 if p['diff'] > 0 else [])
        for job, kind, w in tapes:
            key = (job, p['t'])
            if key not in cache:
                seconds = demo_bc.tape_seconds(job+'/plant/observable.jsonl.gz')
                cache[key] = demo_bc.featurize(native, base, state, seconds, {p['t']})[p['t']]
            trig = cache[key]
            hit = None if trig is None else demo_bc.plan_slot(trig, p['family'], members)
            if hit is None:
                skipped.append(dict(job=Path(job).name, t=p['t'], family=p['family'], kind=kind, reason='plan not offered on the slate'))
                continue
            items.append(dict(kind=kind, weight=float(w), row=hit[0], slot=hit[1], trigger=trig,
                              meta=dict(job=Path(job).name, root=p['root'], cell=p['cell'], seed=p['seed'], t=p['t'],
                                        family=p['family'], diff_vs_incumbent=p['diff'],
                                        member=trig['context']['members'][hit[0]])))
    doc = dict(schema=demo_bc.SCHEMA, rule=demo_bc.RULE, guidance_file_sha256=hashlib.sha256(Path(a.guidance).read_bytes()).hexdigest(),
               items=items, skipped=skipped,
               summary=dict(rows=len(items), act=sum(x['kind'] == 'act' for x in items),
                            none=sum(x['kind'] == 'none' for x in items),
                            seeds=sorted({x['meta']['seed'] for x in items}),
                            families=sorted({x['meta']['family'] for x in items})))
    data = (json.dumps(doc, sort_keys=True)+'\n').encode('utf-8')
    Path(a.out).write_bytes(data)
    print(json.dumps(dict(out=a.out, sha256=hashlib.sha256(data).hexdigest(), **doc['summary'], skipped=len(skipped))))


def parity(a):
    """Featurize a recorded TRAINING episode's tape and compare to its recorded trigger records at every tick
    before the episode's first non-none trigger action (the featurizer never acts)."""
    from tools.coopmwm import demo_bc
    native, base, state = _runtime(a.run_root)
    res = json.load(gzip.open(os.path.join(a.job, 'RESULT.json.gz'), 'rt'))
    recorded = {s['time_s']: s['sample'].get('track2_trigger') for s in res['steps']}
    first = min([t for t, tr in recorded.items() if tr and any(x > 0 for x in tr['actions'])] or [max(recorded)+1])
    want = {t for t, tr in recorded.items() if tr and t < first}
    seconds = demo_bc.tape_seconds(os.path.join(a.job, 'observable.jsonl.gz'))
    got = demo_bc.featurize(native, base, state, seconds, want, seed=0)
    keys = ('context', 'tables', 'descriptors', 'mask')
    diffs = [t for t in sorted(want) if any(json.dumps(got[t][k], sort_keys=True) != json.dumps(recorded[t][k], sort_keys=True)
                                          for k in keys)]
    out = dict(job=Path(a.job).name, ticks=len(want), first_action_s=first, mismatched_ticks=diffs[:20], n_mismatch=len(diffs),
               keys=list(keys))
    Path(a.out).write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps(out))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['extract', 'parity'])
    ap.add_argument('--run-root', required=True)
    ap.add_argument('--guidance')
    ap.add_argument('--ponset', nargs='*', default=())
    ap.add_argument('--bank', nargs='*', default=())
    ap.add_argument('--verify-root', nargs='*', default=())
    ap.add_argument('--job')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    (extract if args.cmd == 'extract' else parity)(args)
