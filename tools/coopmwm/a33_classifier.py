"""AMENDMENT 33 classifier: the EXACT LEARNED_TRIGGER recipe (tools.coopmwm.learned_trigger_train) on
  (a) the original WM-plan rows: CANDS b2fc52f4 (features) joined with LABELED a110af44 (kind/y/w as labelled by
      learned_trigger_train.label_rows from PLAN 777cab92; the refit of these rows reproduces model 8e0ddd3e exactly),
      plan one-hot widened from the 5 WM families to the 6 COMBINED families (duo12_v20 = 0 on every original row);
  (b) duo12_v20 verification executions on fresh TRAIN-type seeds (a33t root): candidates collected with
      learned_trigger.collect_tape over the COMBINED plans (6 families), labelled by learned_trigger_train.label_rows
      (unconditional row: diff = episode J(duo12_v20) - J(u0), ref_arms ['u0'], lookback 10 s).
Background weight is re-applied over the union (BG_SHARE x total harm weight, uniform per bg row), as the recipe does
over its full row set. CV / C grid / threshold / refit / model doc = learned_trigger_train.main.

  collect (host):  python -m tools.coopmwm.a33_classifier collect --tapes DIR --acts ACTS.json --out NEWCANDS.jsonl.gz
  train (laptop):  python -m tools.coopmwm.a33_classifier train --orig-cands .. --orig-labeled .. --new-cands ..
                   --acts ACTS.json --out DIR
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from . import a33_combined as C
from . import learned_trigger as L
from . import learned_trigger_train as T

ORIG_CANDS_SHA256 = 'b2fc52f4d0fd45cba2137f0d8a725633e37856515b490138097255b1b1607c8d'
ORIG_LABELED_SHA256 = 'a110af443a6686c30e1216148127359e4d76d72226fad9f028b468c085e278a7'
ORIG_MODEL_SHA256 = '8e0ddd3e241ce0172736909133f0b745c58b6bf795e06ca5364e5755f7e3b720'
ROOT = 'a33t'
CELL = 'D2800_p20'
TAPE_ARMS = (C.DUO_FAMILY, 'u0', T.INCUMBENT)


def widen(f, old_families, families=C.FAMILIES):
    """Re-map the plan one-hot block of an original feature vector to ``families``."""
    n0 = len(f) - len(old_families)
    onehot = dict(zip(old_families, f[n0:]))
    return list(f[:n0]) + [float(onehot.get(x, 0.0)) for x in families]


def orig_rows(cands_path, labeled_path, model_path):
    for p, h in ((cands_path, ORIG_CANDS_SHA256), (labeled_path, ORIG_LABELED_SHA256), (model_path, ORIG_MODEL_SHA256)):
        if T.sha(p) != h:
            raise ValueError(f'{p}: sha256 differs from the registered original')
    old = json.loads(Path(model_path).read_bytes())['families']
    cands = {(c['tape'], c['t'], c['family']): c for c in T.load_cands(cands_path)}
    rows = []
    with gzip.open(L.guard_path(labeled_path), 'rt', encoding='utf-8') as fh:
        for line in fh:
            r = json.loads(line)
            c = cands[(r['tape'], r['t'], r['family'])]
            rows.append(dict(c, y=r['y'], kind=r['kind'], w=r['w'], f=widen(c['f'], old)))
    return rows


def tape_key(seed, arm):
    return f'{ROOT}__{CELL}_s{int(seed)}_{arm}'


def acts_doc(records):
    """ACTS.json from runner records {(seed, arm): record}: every duo12_v20 activation of a DUO_HOLD episode with the
    episode-level diff J(duo) - J(u0) on that seed (unconditional row protocol)."""
    acts = []
    seeds = sorted({s for s, _ in records})
    for s in seeds:
        duo, u0 = records.get((s, 'DUO_HOLD')), records.get((s, 'U0'))
        if duo is None or u0 is None:
            continue
        d = float(duo['J_veh_s']) - float(u0['J_veh_s'])
        for a in duo['a32']['acts']:
            if a['family'] != C.DUO_FAMILY:
                raise ValueError('DUO_HOLD record with a non-duo activation')
            acts.append(dict(root=ROOT, cell=CELL, seed=int(s), family=C.DUO_FAMILY, t=int(a['t']), diff=d,
                             lookback=C.UNC_LOOKBACK_S, ref_arms=['u0'], admission='unconditional',
                             members=list(a['members'])))
    L.assert_seeds({a['seed'] for a in acts})
    return acts


def cmd_collect(a):
    """Host step: candidate rows (6 COMBINED families) of every a33t tape -> one jsonl.gz (+ meta)."""
    from concurrent.futures import ProcessPoolExecutor
    wm = json.loads(Path(L.guard_path(a.wm_bank)).read_bytes())
    un = json.loads(Path(L.guard_path(a.duo_bank)).read_bytes())
    if T.sha(a.wm_bank) != C.WM_BANK_SHA256 or T.sha(a.duo_bank) != C.DUO_BANK_SHA256:
        raise ValueError('bank docs differ from the registered fd385a2b / 39f384a2')
    plans, refused = C.ordered_plans(C.combined_bank(wm, un, CELL), CELL)
    if sorted(p['family'] for p in plans) != list(C.FAMILIES) or refused:
        raise ValueError(f'COMBINED plans {[p["family"] for p in plans]} refused {refused}')
    acts = json.loads(Path(a.acts).read_bytes())
    ticks = {}
    for x in acts:
        for arm in TAPE_ARMS:
            ticks.setdefault(tape_key(x['seed'], arm), set()).update(
                range(max(300, x['t'] - x['lookback']), x['t'] + 1))
    jobs = []
    for p in sorted(Path(a.tapes).glob(f'{ROOT}__*.jsonl.gz')):
        key = p.name[:-len('.jsonl.gz')]
        jobs.append((str(p), key, plans, list(C.FAMILIES), sorted(ticks.get(key, ())), a.every))
    L.assert_seeds([L.tape_seed_arm(k)[2] for _, k, *_ in jobs])
    n = 0
    with gzip.open(Path(a.out), 'wt', encoding='utf-8') as fh, ProcessPoolExecutor(max_workers=a.workers) as pool:
        for key, rows in pool.map(L._collect_one, jobs, chunksize=1):
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + '\n')
            n += len(rows)
            print(json.dumps(dict(tape=key, rows=len(rows))), flush=True)
    meta = dict(schema=C.SCHEMA + '.collect', acts_sha256=T.sha(a.acts), wm_bank_sha256=C.WM_BANK_SHA256,
                duo_bank_sha256=C.DUO_BANK_SHA256, cell=CELL, families=list(C.FAMILIES),
                strategy_keys={p['family']: p['key'] for p in plans}, every=a.every, tapes=len(jobs), rows=n,
                digest=L.learned_trigger_digest(), a33_digest=C.a33_digest())
    Path(str(a.out) + '.meta.json').write_text(json.dumps(meta, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(meta))


def build_rows(orig, new_cands, acts):
    new, counts = T.label_rows(new_cands, acts, list(C.FAMILIES))
    rows = orig + new
    harm_w = sum(r['w'] for r in rows if r['kind'] == 'harm')
    nbg = sum(r['kind'] == 'bg' for r in rows)
    for r in rows:
        if r['kind'] == 'bg':
            r['w'] = T.BG_SHARE * harm_w / max(1, nbg)
    rows.sort(key=lambda r: (r['tape'], r['t'], r['family']))
    return rows, counts


def cmd_train(a):
    import numpy as np
    meta = json.loads(Path(a.new_cands + '.meta.json').read_text(encoding='utf-8'))
    if meta['families'] != list(C.FAMILIES) or meta['acts_sha256'] != T.sha(a.acts):
        raise ValueError('new candidates were not collected for these families / ACTS')
    acts = json.loads(Path(a.acts).read_bytes())
    orig = orig_rows(a.orig_cands, a.orig_labeled, a.orig_model)
    rows, new_counts = build_rows(orig, T.load_cands(a.new_cands), acts)
    L.assert_seeds({r['seed'] for r in rows})
    names = L.feature_names(list(C.FAMILIES))
    if any(len(r['f']) != len(names) for r in rows):
        raise ValueError('feature length differs from feature_names')
    cv = {}
    for Cc in T.C_GRID:
        cv[str(Cc)] = dict(T.metrics(rows, T.oof(rows, Cc)))
    best_c = max(T.C_GRID, key=lambda Cc: (cv[str(Cc)]['auc_onset_pos_vs_harm'] or 0, -Cc))
    p = T.oof(rows, best_c)
    curve = T.threshold_curve(rows, p)
    best = max(curve, key=lambda c: (round(c['net_veh_s'], 6), c['tau']))
    X = np.array([r['f'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows])
    w = np.array([r['w'] for r in rows], dtype=float)
    sc, m = T.fit(X, y, w / w.mean(), best_c)
    kinds = {k: sum(r['kind'] == k for r in rows) for k in ('pos', 'harm', 'bg')}
    model = dict(schema=L.SCHEMA, families=list(C.FAMILIES), strategy_keys=meta['strategy_keys'], feature_names=names,
                 mean=[float(v) for v in sc.mean_], scale=[float(v) for v in sc.scale_],
                 coef=[float(v) for v in m.coef_[0]], intercept=float(m.intercept_[0]),
                 threshold=float(best['tau']),
                 threshold_rule='argmax out-of-fold (seed-grouped 5-fold) net verified benefit over onset rows; '
                                'ties -> higher threshold',
                 C=best_c, cv=dict(folds=T.FOLDS, groups='seed', grid=cv, chosen=T.metrics(rows, p),
                                   per_family=T.per_family_auc(rows, p, list(C.FAMILIES)),
                                   at_threshold=best, calibration=T.calibration(rows, p)),
                 data=dict(orig_cands_sha256=ORIG_CANDS_SHA256, orig_labeled_sha256=ORIG_LABELED_SHA256,
                           orig_model_sha256=ORIG_MODEL_SHA256, new_cands_sha256=T.sha(a.new_cands),
                           acts_sha256=T.sha(a.acts), cell=CELL, new_counts=new_counts, kinds=kinds,
                           seeds=sorted({r['seed'] for r in rows}), every=meta['every'], excl_s=T.EXCL_S,
                           bg_share=T.BG_SHARE, collect_digest=meta['digest'], amendment=33),
                 code_digest=L.learned_trigger_digest(), a33_digest=C.a33_digest())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(model, indent=1, sort_keys=True) + '\n').encode('utf-8')
    (out / 'COMBINED_MODEL.json').write_bytes(raw)
    (out / 'COMBINED_MODEL.json.sha256').write_text(hashlib.sha256(raw).hexdigest() + '\n', encoding='utf-8')
    (out / 'THRESHOLD_CURVE.json').write_text(json.dumps(curve, indent=1) + '\n', encoding='utf-8')
    with gzip.open(out / 'LABELED.jsonl.gz', 'wt', encoding='utf-8') as fh:
        for r, q in zip(rows, p):
            fh.write(json.dumps(dict({k: r[k] for k in ('tape', 'seed', 'arm', 't', 'family', 'kind', 'y', 'w',
                                                        'members')}, oof=float(q)), sort_keys=True) + '\n')
    print(json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest(), C=best_c, kinds=kinds, new_counts=new_counts,
                          chosen=model['cv']['chosen'], per_family=model['cv']['per_family'], at_threshold=best),
                     indent=1))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('collect')
    p.add_argument('--tapes', required=True)
    p.add_argument('--acts', required=True)
    p.add_argument('--wm-bank', required=True)
    p.add_argument('--duo-bank', required=True)
    p.add_argument('--every', type=int, default=5)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--out', required=True)
    p = sub.add_parser('train')
    p.add_argument('--orig-cands', required=True)
    p.add_argument('--orig-labeled', required=True)
    p.add_argument('--orig-model', required=True)
    p.add_argument('--new-cands', required=True)
    p.add_argument('--acts', required=True)
    p.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    (cmd_collect if a.cmd == 'collect' else cmd_train)(a)


if __name__ == '__main__':
    main()
