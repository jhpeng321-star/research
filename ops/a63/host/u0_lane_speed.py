"""A63 optional descriptive (zero simulation): mean speed per SUMO lane in the 'far' control zone under the stored A42 U0
runs (96 confirmation worlds), from observable.jsonl.gz (read-only). Zone 'far' = x in [X_B-1800, X_B-1000] with
x = s_m + S_OFF (oracle_plan constants); SUMO lane = 2 - WM lane. All logged rows (every vehicle, every logged step)."""
import glob
import gzip
import json
import sys

import numpy as np

X_B, DROP_S = 2696.0, 75.44
S_OFF = X_B - DROP_S
LO, HI = X_B - 1800.0, X_B - 1000.0
root = sys.argv[1] if len(sys.argv) > 1 else '/root/a42v/root/evaluation'
dirs = sorted(glob.glob(f'{root}/a42c_u0_00[0-9][0-9]'))
assert len(dirs) == 96, len(dirs)
per_world = {k: [] for k in (0, 1, 2)}
pooled = {k: [] for k in (0, 1, 2)}
for d in dirs:
    vs = {k: [] for k in (0, 1, 2)}
    with gzip.open(f'{d}/observable.jsonl.gz', 'rt') as f:
        for line in f:
            o = json.loads(line)
            for r in o['vehicles']:
                x = float(r['s_m']) + S_OFF
                if LO <= x <= HI:
                    vs[2 - int(r['lane'])].append(float(r['v_mps']))
    for k in vs:
        v = np.asarray(vs[k], float)
        if v.size:
            per_world[k].append(float(v.mean()))
            pooled[k].append(v)
out = dict(zone_x=[LO, HI], worlds=len(dirs), convention='SUMO lane; speeds m/s; all logged rows in the far zone')
for k in (0, 1, 2):
    v = np.concatenate(pooled[k])
    out[f'lane{k}'] = dict(rows=int(v.size), mean_pooled=float(v.mean()), mean_of_world_means=float(np.mean(per_world[k])),
                           worlds=len(per_world[k]), p10=float(np.quantile(v, .1)), median=float(np.median(v)),
                           p90=float(np.quantile(v, .9)), share_above_18=float((v > 18).mean()),
                           share_above_22=float((v > 22).mean()))
print(json.dumps(out, indent=1))
