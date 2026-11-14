"""E1R random-12 draw (run ONCE; no outcome of any plan is read). Zero SUMO, zero WM.

Pool = all 607 screened S2b plans (full definitions from POOL_REPLAY_ALL607.json, key-verified replay)
       minus the 12 pinned WM plans (TOP_OFFER_CONDITIONS_D2800_p60.json), minus the hand-written duo12_v20 /
       duo12_v18, minus any comparator (solo2, wall_g_solo2; not among the screened keys) -- sorted by key.
Draw   = numpy default_rng(20261113).choice(len(pool), 12, replace=False) over the key-sorted pool.
Refuses to overwrite an existing DRAW file (the draw is made once).

usage: python -I draw_e1r.py SEARCH_JSON TOP_JSON REPLAY_JSON OUTDIR
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SEED = 20261113
N_DRAW = 12
HAND = ('duo12_v20', 'duo12_v18')


def canon(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'))


def main(search, top, replay, outdir):
    out = Path(outdir)
    if (out / 'E1R_DRAW.json').exists():
        raise SystemExit('REFUSE: E1R_DRAW.json exists (draw is made once)')
    S = json.loads(Path(search).read_text())
    T = json.loads(Path(top).read_text())
    R = json.loads(Path(replay).read_text())['strategies']
    screened = {r['key']: r for r in S['screened']}
    strat = {r['key']: r['strategy'] for r in R}
    assert set(strat) == set(screened) and len(strat) == 607
    pinned = {t['strategy_key'] for t in T}
    comp = {f['key'] for f in S['final'] if f['comparator']}
    hand = {k for k, s in strat.items() if s['name'] in HAND}
    excl = pinned | comp | hand
    pool = sorted(k for k in strat if k not in excl)
    pool_doc = [dict(key=k, name=strat[k]['name'], strategy=strat[k]) for k in pool]
    pool_txt = canon(pool_doc)
    pool_sha = hashlib.sha256(pool_txt.encode()).hexdigest()
    (out / 'E1R_POOL.json').write_text(json.dumps(dict(
        schema='e1r.pool.v1', n=len(pool), sha256_canonical=pool_sha,
        definition='607 screened S2b plans - 12 pinned TOP plans - {duo12_v20, duo12_v18} - comparators; sorted by key',
        excluded=dict(pinned=sorted(pinned), comparators=sorted(comp), hand=sorted(hand)),
        pool=pool_doc), indent=1))
    rng = np.random.default_rng(SEED)
    idx = [int(i) for i in rng.choice(len(pool), N_DRAW, replace=False)]
    drawn = [dict(draw_order=j, pool_index=i, key=pool[i], name=strat[pool[i]]['name'], strategy=strat[pool[i]])
             for j, i in enumerate(idx)]
    (out / 'E1R_DRAW.json').write_text(json.dumps(dict(
        schema='e1r.draw.v1', seed=SEED, method='numpy.random.default_rng(seed).choice(len(pool), 12, replace=False)',
        numpy=np.__version__, pool_n=len(pool), pool_sha256_canonical=pool_sha, indices=idx, drawn=drawn), indent=1))
    print('pool', len(pool), pool_sha)
    for d in drawn:
        print(d['draw_order'], d['pool_index'], d['key'], d['name'])


if __name__ == '__main__':
    main(*sys.argv[1:5])
