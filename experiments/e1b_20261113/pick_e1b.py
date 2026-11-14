"""E1B bottom-12 pick (rule-based, made once; zero SUMO, zero WM; no SUMO outcome of any plan is read).

Pool  = the E1R pool (607 screened S2b plans - 12 pinned WM top plans - comparators; 595, key-sorted).
Rule  = keep plans with WM screen feasible > 0 (the real TOP writer's `activated` filter), sort by WM screen
        gain_vs_base ascending (lowest predicted gain first), ties by key; take the first 12.
Output in the E1R draw schema so write_top_e1r.py / analyse_e1r.py are reused unchanged.
usage: python -I pick_e1b.py SEARCH_JSON E1R_POOL_JSON E1R_DRAW_JSON OUTDIR
"""
import hashlib, json, sys
from pathlib import Path

def main(search, pool, e1r, outdir):
    out = Path(outdir) / 'E1B_DRAW.json'
    if out.exists():
        raise SystemExit('REFUSE: E1B_DRAW.json exists')
    S = json.loads(Path(search).read_text())
    P = json.loads(Path(pool).read_text())
    R = {d['key'] for d in json.loads(Path(e1r).read_text())['drawn']}
    sc = {r['key']: r for r in S['screened']}
    assert P['n'] == 595 == len(P['pool'])
    elig = [p for p in P['pool'] if sc[p['key']]['feasible'] > 0]
    elig.sort(key=lambda p: (sc[p['key']]['gain_vs_base'], p['key']))
    pick = elig[:12]
    drawn = [dict(draw_order=j, key=p['key'], name=p['name'], strategy=p['strategy'],
                  wm_screen_gain_vs_base=sc[p['key']]['gain_vs_base'], wm_screen_feasible=sc[p['key']]['feasible'],
                  also_in_e1r_random_draw=p['key'] in R) for j, p in enumerate(pick)]
    out.write_text(json.dumps(dict(schema='e1b.pick.v1', rule=__doc__, pool_n=595, eligible_n=len(elig),
                                   pool_sha256_canonical=P['sha256_canonical'], drawn=drawn), indent=1))
    print('eligible', len(elig), 'sha', hashlib.sha256(out.read_bytes()).hexdigest())
    for d in drawn:
        s = d['strategy']
        print(d['draw_order'], d['key'], d['name'], d['wm_screen_gain_vs_base'], d['wm_screen_feasible'],
              s.get('trigger'), s.get('zone'), len(s['roles']), 'inE1R' if d['also_in_e1r_random_draw'] else '')


if __name__ == '__main__':
    main(*sys.argv[1:5])
