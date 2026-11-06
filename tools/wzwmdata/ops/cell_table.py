"""WM-B report table: per cell/arm data volume and descriptive outcomes (zero physics)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

roots = [Path(r) for r in sys.argv[1:]]
rows = defaultdict(lambda: defaultdict(list))
for root in roots:
    for res in sorted(root.rglob('RESULT.json')):
        r = json.loads(res.read_text())
        if r.get('status') != 'ok' or r.get('block') not in (None, 'WMDATA'):
            continue
        if 'wmqual' in str(res):
            continue
        cell = r['world']['cell_key']
        arm = r['arm'].replace('_u0', '')
        t = r['totals']
        rows[cell][arm].append(dict(seed=r['world']['seed'], J=t['team_delay_vs'], ttc=t['ttc_below_3_vs'],
                                    brake=t['hard_brake'], present=r['max_present'], sim_s=r['actual_sim_s'],
                                    bytes=r.get('episode_npz_bytes', 0),
                                    moves=(r.get('coop') or {}).get('n_moves', 0)))
out = {}
for cell in sorted(rows):
    d = {}
    for arm, xs in sorted(rows[cell].items()):
        n = len(xs)
        d[arm] = dict(n=n, sim_s=sum(x['sim_s'] for x in xs), mean_J_vs=round(sum(x['J'] for x in xs) / n, 1),
                      mean_ttc3_vs=round(sum(x['ttc'] for x in xs) / n, 1),
                      max_present=max(x['present'] for x in xs), moves=sum(x['moves'] for x in xs),
                      mb=round(sum(x['bytes'] for x in xs) / 1e6, 1))
    nat = {x['seed']: x['J'] for x in rows[cell].get('native', [])}
    for arm in ('mappo', 'coop_mappo', 'coop_native'):
        pairs = [(x['J'], nat[x['seed']]) for x in rows[cell].get(arm, []) if x['seed'] in nat]
        if pairs:
            d[arm]['paired_ratio_vs_native'] = round(sum(a for a, _ in pairs) / sum(b for _, b in pairs), 3)
    out[cell] = d
print(json.dumps(out, indent=1))
