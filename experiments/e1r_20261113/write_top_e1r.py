"""E1R TOP file writer (zero SUMO, zero WM): the a75ss.main TOP_OFFER_CONDITIONS row builder (a75ss cae54dc1,
functions offer_condition / is_extended imported unchanged) applied to the 12 randomly drawn plans.

Self-check first: the same builder applied to the 12 pinned WM plans (their `final` records) must reproduce the real
TOP_OFFER_CONDITIONS_D2800_p60.json byte for byte.

WM fields of a random row (no WM number is used by any SUMO rule of the A80 driver; carried for the writer's row keys):
  score            the plan's screen score from the search archives listing if the plan sits in an archive, else null
  gain_vs_base     screened record (18 screen states, K=4)
  gain_vs_base_lcb null (not stored for screened-only plans)
  feasible         screened record
usage: python -I write_top_e1r.py SEARCH_JSON TOP_JSON DRAW_JSON OUT_TOP
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'src_ref' / 'tree'))
import a75ss as A  # noqa: E402

A75SS_SHA = 'cae54dc11be5f5220f19eb7399efac2699c7cdf363a90b7fa202766d5b906bc0'


def rows_text(rows):
    return json.dumps(rows, indent=1)


def row(strategy, key, score, gain, lcb, feasible):
    return dict(family=strategy['name'], strategy_key=key, score=score, gain_vs_base=gain, gain_vs_base_lcb=lcb,
                feasible=feasible, extended=A.is_extended(strategy), offer_condition=A.offer_condition(strategy))


def main(search, top, draw, out):
    assert hashlib.sha256(Path(A.__file__).read_bytes()).hexdigest() == A75SS_SHA
    S = json.loads(Path(search).read_text())
    real = Path(top).read_bytes().decode("utf-8")
    fin = {r['key']: r for r in S['final']}
    pinned = json.loads(real)
    again = [row(fin[t['strategy_key']]['strategy'], t['strategy_key'], fin[t['strategy_key']]['score'],
                 fin[t['strategy_key']]['gain_vs_base'], fin[t['strategy_key']]['gain_vs_base_lcb'],
                 fin[t['strategy_key']]['feasible']) for t in pinned]
    if rows_text(again) != real:
        raise SystemExit('writer self-check FAILED: cannot reproduce the real TOP file')
    print('writer self-check: real TOP file reproduced byte-identically')
    sc = {r['key']: r for r in S['screened']}
    arch = {}
    for lst in S['archives'].values():
        for r in lst:
            arch.setdefault(r['key'], r['score'])
    D = json.loads(Path(draw).read_text())
    rows = []
    for d in D['drawn']:
        s, k = d['strategy'], d['key']
        assert A.strategy_key(s) == k and s['name'] == sc[k]['name']
        rows.append(row(s, k, arch.get(k), sc[k]['gain_vs_base'], None, sc[k]['feasible']))
    txt = rows_text(rows)
    p = Path(out)
    if p.exists():
        raise SystemExit(f'REFUSE: {p} exists')
    p.write_bytes(txt.encode("utf-8"))   # LF bytes, as the Linux writer
    print('wrote', p, hashlib.sha256(txt.encode()).hexdigest())


if __name__ == '__main__':
    main(*sys.argv[1:5])
