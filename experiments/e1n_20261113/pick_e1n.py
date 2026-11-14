"""E1N 'unchosen, unlike-known' batch (rule-based, made once; zero SUMO/WM; no SUMO outcome read).
Rule: S2b finalists (63) that are not comparators, not among the 12 pinned top plans, searched (name rand_/mut_),
WM feasible > 0, and (a role in lane 0, the closing lane, OR >= 3 roles); in the order of the final list.
Plans already drawn in the E1R random arm are listed but not re-run (same worlds, same rule: result reused).
Remaining plans are split in final-list order into parts of <= 12 (driver limit).
usage: python -I pick_e1n.py SEARCH TOP E1R_DRAW OUTDIR
"""
import hashlib, json, sys
from pathlib import Path

def main(search, top, e1r, outdir):
    o = Path(outdir)
    if (o / 'E1N_LIST.json').exists():
        raise SystemExit('REFUSE: exists')
    S = json.loads(Path(search).read_text())
    T = {t['strategy_key'] for t in json.loads(Path(top).read_text())}
    R = {d['key'] for d in json.loads(Path(e1r).read_text())['drawn']}
    sel = []
    for r in S['final']:
        s = r['strategy']
        if r['comparator'] or r['key'] in T or not s['name'].startswith(('rand_', 'mut_')) or r['feasible'] <= 0:
            continue
        lanes = sorted(x['lane'] for x in s['roles'])
        if 0 in lanes or len(lanes) >= 3:
            sel.append(dict(key=r['key'], name=s['name'], strategy=s, lanes=lanes, wm_score=r['score'],
                            wm_gain_vs_base=r['gain_vs_base'], wm_feasible=r['feasible'], in_e1r=r['key'] in R))
    run = [d for d in sel if not d['in_e1r']]
    parts = [run[i:i + 12] for i in range(0, len(run), 12)]
    (o / 'E1N_LIST.json').write_text(json.dumps(dict(schema='e1n.list.v1', rule=__doc__, n=len(sel), n_run=len(run), plans=sel), indent=1))
    for j, p in enumerate(parts, 1):
        (o / f'E1N_DRAW_part{j}.json').write_text(json.dumps(dict(schema='e1n.part.v1', part=j,
            drawn=[dict(draw_order=i, key=d['key'], name=d['name'], strategy=d['strategy']) for i, d in enumerate(p)]), indent=1))
    print(len(sel), len(run), [len(p) for p in parts], [d['name'] for d in sel if d['in_e1r']])


if __name__ == '__main__':
    main(*sys.argv[1:5])
