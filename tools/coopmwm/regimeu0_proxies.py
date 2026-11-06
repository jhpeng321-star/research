"""S-REGIMEU0 amendment 4: observable congestion proxies at 100-s window starts (zero SUMO, stdlib only).

Reads every u0 job's plant/RESULT.json.gz (accel_plant controller events: the controller's own observable
rows = CAV self reports + vehicles detected by CAVs; no truth) and writes <root>/PROXIES.json:
{job_id: {t_w: {blk_c, blk_n, blk_v, slow_field}}} for t_w in 300, 400, ..., 1100.
  blk_v  = mean speed of rows in WM lane 2 (closing lane), s in [-100, 75.44] (REGIME / N1 RSU block)
  blk_c  = 31.29 - blk_v (0 when no row is in the block)
  slow_field = rows in WM lane 2, s in [-378.66, 75.44] (N1 RSU field) with v < 5 m/s
Usage: python3 regimeu0_proxies.py ROOT [ROOT ...]
"""
import gzip
import json
import sys
from pathlib import Path

V_MAX = 31.29
BLOCK = (-100.0, 75.44)
FIELD = (-378.66, 75.44)
CLOSING = 2
WINDOWS = tuple(range(300, 1200, 100))


def window_proxies(steps):
    by_t = {int(e['time_s']): e.get('rows') or [] for e in steps if int(e['time_s']) in WINDOWS}
    out = {}
    for t in WINDOWS:
        rows = by_t.get(t)
        if rows is None:
            continue
        blk = [float(r['v_mps']) for r in rows if int(r.get('lane', -1)) == CLOSING
               and BLOCK[0] <= float(r['s_m']) <= BLOCK[1]]
        slow = sum(1 for r in rows if int(r.get('lane', -1)) == CLOSING and FIELD[0] <= float(r['s_m']) <= FIELD[1]
                   and float(r['v_mps']) < 5.0)
        v = sum(blk)/len(blk) if blk else None
        out[str(t)] = dict(blk_v=v, blk_n=len(blk), blk_c=(V_MAX-v) if v is not None else 0.0, slow_field=slow)
    return out


def main(roots):
    for root in map(Path, roots):
        plan = json.loads((root/'PLAN.json').read_text())
        res = {}
        for g in plan['groups']:
            for j in g['jobs']:
                if j['arm'] != 'u0' or g['stage'] != 'paired':
                    continue
                p = root/'jobs'/j['id']/'plant'/'RESULT.json.gz'
                if not p.exists():
                    continue
                with gzip.open(p, 'rt') as f:
                    steps = json.load(f)['steps']
                res[j['id']] = dict(seed=j['world']['seed'], cell=j['world']['cell_key'], windows=window_proxies(steps))
        out = root/'PROXIES.json'
        if out.exists():
            raise SystemExit(f'{out} exists (never overwritten)')
        out.write_text(json.dumps(dict(rule=__doc__.split('\n')[0], proxies=res), sort_keys=True))
        print(root, len(res))


if __name__ == '__main__':
    main(sys.argv[1:])
