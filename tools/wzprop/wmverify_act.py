"""S-WMVERIFY calibrated: per-activation analysis conditioned on the observable onset situation, pooled across
penetration (owner direction 2026-11-07 23:16 / MAIN 03:13; penetration is a label only, never a conditioning input).

Reuses the S-MULTICRIT-ACT method verbatim (runs/active/C4-V7FT-20261017/multicrit/act: reduce_act.py runs on the host
next to the job dirs; act_analyze.activations / cboot / sit_bins here): effect of activation k = change in the paired
(arm - u0) delay gap over [t_k, min(next activation, t_k + 300 s, 1200 s)) from 10-s binned observed delay (u0 labels by
vehicle id); lane-2 cost = the same for origin-lane-2 vehicles; cluster bootstrap over (root, cell, seed), 4,000
resamples, RNG 20261107. Holder = the activation's SUMO-lane-2 member (else its first member) for the onset situation.
Generalised from the gentle hold arms to every verified strategy arm (one table per arm and pooled per pick source).

    python -m tools.wzprop.wmverify_act --act-dir runs/active/C4-V7FT-20261017/multicrit/act \
        --reduced act_D2800.jsonl[,act_D3200.jsonl] --arms ARMS.json[,ARMS2.json] --out ACT_WMVERIFY.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

FACTORS = ('dist', 'cong', 'gap', 'partner', 'lane_diff')


def load_act(act_dir):
    spec = importlib.util.spec_from_file_location('act_analyze_multicrit', Path(act_dir) / 'act_analyze.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sources(arms_paths):
    out = {}
    for p in arms_paths:
        doc = json.loads(Path(p).read_text(encoding='utf-8'))
        for cell, d in doc.items():
            if not isinstance(d, dict):
                continue
            for grp, v in d.items():
                items = v if isinstance(v, list) else ([v] if isinstance(v, dict) and 'roles' in v else [])
                for s in items:
                    if isinstance(s, dict) and s.get('name'):
                        out.setdefault(s['name'], grp)
            if isinstance(d.get('incumbent'), dict):
                out['wall_g_solo2'] = 'incumbent'
    return out


def analyze(act, reduced, arms_paths):
    eps = act.load(reduced)
    rows = [r for r in act.activations(eps) if r['sit'] is not None]
    for r in rows:
        r['b'] = act.sit_bins(r['sit'])
    src = sources(arms_paths)
    res = dict(method='S-MULTICRIT-ACT activations/cboot/sit_bins (verbatim), all strategy arms, pooled across '
                      'penetration (label only)', n_activations=len(rows), by_arm={}, by_source={}, situation={})
    for arm in sorted({r['arm'] for r in rows}):
        rs = [r for r in rows if r['arm'] == arm]
        res['by_arm'][arm] = dict(
            source=src.get(arm), eff=act.cboot(rs), l2=act.cboot(rs, lambda r: r['l2']),
            others=act.cboot(rs, lambda r: r['others']), eff300=act.cboot(rs, lambda r: r['eff300']),
            by_pen_label={p: act.cboot([r for r in rs if r['pen'] == p]) for p in sorted({r['pen'] for r in rs})},
            situation={fac: {lv: dict(eff=act.cboot(sub), l2=act.cboot(sub, lambda r: r['l2']))
                             for lv in sorted({r['b'][fac] for r in rs}) for sub in [[r for r in rs if r['b'][fac] == lv]]}
                       for fac in FACTORS})
    for s in sorted({src.get(r['arm'], 'unknown') for r in rows}):
        rs = [r for r in rows if src.get(r['arm'], 'unknown') == s]
        res['by_source'][s] = dict(arms=sorted({r['arm'] for r in rs}), eff=act.cboot(rs),
                                   l2=act.cboot(rs, lambda r: r['l2']))
    for fac in FACTORS:
        res['situation'][fac] = {lv: dict(eff=act.cboot(sub), l2=act.cboot(sub, lambda r: r['l2']))
                                 for lv in sorted({r['b'][fac] for r in rows})
                                 for sub in [[r for r in rows if r['b'][fac] == lv and r['arm'] != 'wall_g_solo2']]}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m tools.wzprop.wmverify_act')
    ap.add_argument('--act-dir', required=True)
    ap.add_argument('--reduced', required=True)
    ap.add_argument('--arms', required=True)
    ap.add_argument('--out', required=True)
    x = ap.parse_args(argv)
    act = load_act(x.act_dir)
    res = analyze(act, x.reduced.split(','), x.arms.split(','))
    with Path(x.out).open('x', encoding='utf-8') as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({a: (v['source'], v['eff']) for a, v in res['by_arm'].items()}, default=str, indent=1))


if __name__ == '__main__':
    main()
