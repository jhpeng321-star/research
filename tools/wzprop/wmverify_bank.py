"""S-WMVERIFY verified-plan BANK for later training (clean5c; owner 2026-11-07 "start the collecting now and have the
training start later"). Zero physics: reads a fail-closed `wmverify analyze` output directory (VERIFY.json with
analysis_written=True, ANALYSIS_ALLOWED.json) and the run roots' job SUMMARY files.

One JSONL row per (cell, seed, family, activation) of every fairness-ALLOWED family (the incumbent included when it
is allowed); never the banned comparator, never a banned spec, never the parity arm. Each row carries: the executed
activation (second t, members in role order, phase, SUMO lanes, x, release reasons), the family spec / strategy key /
executor / LCEXEC slot template (+ phase templates) / offer condition, and the verified episode outcome of that seed
(J of the arm, u0 and the incumbent over [300, 1200) s; diff = arm - reference, negative = better) plus the cell-level
fairness verdict (labels from u0, N6 amendment 13).

    python -m tools.wzprop.wmverify_bank --analysis DIR --root A [B ...] --out BANK.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = 'wmverify.bank.v1'


def _read(path):
    import gzip
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    return json.loads(path.read_text(encoding='utf-8'))


def build(analysis_dir, roots):
    from tools.coopmwm import regimeu0 as R
    from . import strategy_search as SS
    from . import wmverify as V
    adir = Path(analysis_dir)
    ver = _read(adir / 'VERIFY.json')
    if not ver.get('analysis_written') or not ver['integrity']['ok']:
        raise ValueError('analysis is not integrity-clean (fail closed): no bank')
    allowed_doc = _read(adir / 'ANALYSIS_ALLOWED.json')
    sha_allowed = hashlib.sha256((adir / 'ANALYSIS_ALLOWED.json').read_bytes()).hexdigest()
    if sha_allowed != ver.get('analysis_allowed_sha256'):
        raise ValueError('ANALYSIS_ALLOWED.json differs from the VERIFY record')
    banned = set(ver.get('banned_arms') or ()) | set(V.NOT_FAMILIES)
    rows = []
    for root in [Path(r) for r in roots]:
        body = _read(root / 'PLAN.json')
        if body.get('pilot'):
            raise ValueError(f'{root} is an ungated PILOT root: never a bank source (never pooled)')
        for g in body['groups']:
            if g['stage'] != 'paired':
                continue
            w = g['jobs'][0]['world']
            cell = w['cell_key']
            allowed = {a for a in (allowed_doc['cells'].get(cell) or {}) if a != V.BASE_ARM and 'minus' not in a}
            sums = {}
            for j in g['jobs']:
                p = root / 'jobs' / j['id'] / 'SUMMARY.json.gz'
                s = _read(p) if p.exists() else None
                sums[j['arm']] = (j, s)
            ref = {a: (R.control_j(sums[a][1]) if a in sums and sums[a][1] and not sums[a][1].get('error') else None)
                   for a in (V.BASE_ARM, V.INCUMBENT)}
            for arm, (j, s) in sums.items():
                if arm in banned or arm not in allowed or s is None or s.get('error'):
                    continue
                spec = j.get('strategy_spec')
                if spec is not None and SS.banned(spec):
                    continue
                fam = (ver.get('families') or {}).get(arm) or {}
                jv = R.control_j(s)
                fair = ((ver.get('fairness') or {}).get(cell) or {}).get(arm) or {}
                for k, act in enumerate(s.get('activations') or ()):
                    rows.append(dict(
                        schema=SCHEMA, cell=cell, seed=w['seed'], world=w, family=arm, activation=k, t=act.get('t'),
                        members=act.get('members'), phase=act.get('phase', 0), sumo_lanes=act.get('sumo_lanes'),
                        x=act.get('x') or act.get('s'), release=act.get('release'),
                        executor=j['executor'], strategy_key=fam.get('strategy_key'), spec=spec or fam.get('spec'),
                        template=fam.get('template'), template_sha256=fam.get('template_sha256'),
                        phase_templates=fam.get('phase_templates'), offer_condition=fam.get('offer_condition'),
                        lcexec_parity=fam.get('lcexec_parity'), n2_parity_verdict=(fam.get('n2_parity') or {}).get(
                            'verdict'),
                        J=jv, J_u0=ref[V.BASE_ARM], J_incumbent=ref[V.INCUMBENT],
                        diff_vs_u0=None if ref[V.BASE_ARM] is None else jv - ref[V.BASE_ARM],
                        diff_vs_incumbent=None if ref[V.INCUMBENT] is None or arm == V.INCUMBENT
                        else jv - ref[V.INCUMBENT],
                        cell_allowed=fair.get('allowed'), cell_groups_vs_u0=fair.get('groups_vs_u0'),
                        job_id=j['id'], root=str(root), analysis_allowed_sha256=sha_allowed))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m tools.wzprop.wmverify_bank')
    ap.add_argument('--analysis', required=True)
    ap.add_argument('--root', required=True, nargs='+')
    ap.add_argument('--out', required=True)
    x = ap.parse_args(argv)
    rows = build(x.analysis, x.root)
    with Path(x.out).open('x', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, default=str) + '\n')
    fams = sorted({r['family'] for r in rows})
    print(json.dumps(dict(out=x.out, rows=len(rows), families=fams,
                          seeds=sorted({r['seed'] for r in rows}))))


if __name__ == '__main__':
    main()
