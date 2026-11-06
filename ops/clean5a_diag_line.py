"""clean5a per-update report-only diagnostic line (MAIN 2026-11-06 ~23:4x; owner: earliest confirmation).

Reads ONLY finished logs (no SUMO, no checkpoint loads):
  <root>/{A,B,R}_UPDATE_<u>.json  (physical[].id/world/totals.team_delay_vs; learning.fullpol valsplit receipt)
  <root>/jobs/<episode id>/RESULT.json.gz + observable.jsonl.gz (trigger tables/actions, observed ego lane)
By cell (D3200/D2800/D2400 x p60/p40):
  (1) B and R hold rate per decision (selected group_hold / trigger decisions), ego observed lane 2 (closing) vs 0/1;
  (2) paired team-delay difference B-A and R-A on the same TRAIN worlds (A = frozen control), veh-s, mean and
      t-95% CI over the update's paired worlds (negative = treatment has less delay);
  (3) EV-BC held-back seed-split receipt (B): ev_act_auc after and EV-target NLL before -> after.
Usage: python ops/clean5a_diag_line.py <root> <update> [--wait] [--out FILE]
"""
import argparse
import gzip
import json
import math
from pathlib import Path
from statistics import fmean, stdev
import time

CELLS = ('D3200_p60', 'D3200_p40', 'D2800_p60', 'D2800_p40', 'D2400_p60', 'D2400_p40')
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 20: 2.086, 30: 2.042}


def t975(df):
    return T975.get(df) or T975[max(k for k in T975 if k <= df)] if df >= 1 else math.nan


def ci(values):
    if not values:
        return None
    m = fmean(values)
    if len(values) < 2:
        return (m, None, None, len(values))
    h = t975(len(values)-1)*stdev(values)/math.sqrt(len(values))
    return (m, m-h, m+h, len(values))


def episode_holds(job):
    """{lane_class: [decisions, holds]} from one finished episode."""
    lanes = {}
    with gzip.open(job/'observable.jsonl.gz', 'rt') as f:
        for line in f:
            row = json.loads(line)
            lanes[row['time_s']] = {v['vid']: v['lane'] for v in row['vehicles']}
    out = {'lane2': [0, 0], 'lane01': [0, 0]}
    result = json.load(gzip.open(job/'RESULT.json.gz', 'rt'))
    for step in result['steps']:
        trig = step['sample'].get('track2_trigger')
        if not trig:
            continue
        acts = trig['actions']
        for i, vid in enumerate(trig['context']['members']):
            table = trig['tables'][i]
            if not any(table):
                continue                       # no offered move: not a decision with a hold option
            lane = lanes.get(step['time_s'], {}).get(vid)
            key = 'lane2' if lane == 2 else 'lane01'
            a = int(acts[i])
            move = table[a] if 0 <= a < len(table) else None
            out[key][0] += 1
            out[key][1] += int(bool(move) and move.get('family') == 'group_hold')
    return out


def line(root, u):
    root = Path(root)
    ups = {arm: json.loads((root/f'{arm}_UPDATE_{u:03}.json').read_text()) for arm in ('A', 'B', 'R')}
    delay = {arm: {p['world']['seed']: (p['world']['cell_key'], p['totals']['team_delay_vs'])
                   for p in ups[arm]['physical']} for arm in ups}
    jobs = {arm: {p['world']['seed']: p['id'] for p in ups[arm]['physical']} for arm in ('B', 'R')}
    parts = []
    for cell in CELLS:
        cell_bits = []
        for arm in ('B', 'R'):
            dec = {'lane2': [0, 0], 'lane01': [0, 0]}
            for seed, eid in jobs[arm].items():
                if delay[arm][seed][0] != cell:
                    continue
                for k, (n, h) in episode_holds(root/'jobs'/eid).items():
                    dec[k][0] += n
                    dec[k][1] += h
            rates = '/'.join(f"{(h/n if n else float('nan')):.3f}" for n, h in (dec['lane2'], dec['lane01']))
            diffs = [delay[arm][s][1]-delay['A'][s][1] for s in delay[arm]
                     if delay[arm][s][0] == cell and s in delay['A']]
            c = ci(diffs)
            d = ('n/a' if c is None else f"{c[0]:+.0f}" + ('' if c[1] is None else f" [{c[1]:+.0f},{c[2]:+.0f}]")
                 + f" n{c[3]}")
            cell_bits.append(f"{arm} hold L2/L01={rates} (dec {dec['lane2'][0]}/{dec['lane01'][0]}) {arm}-A={d}")
        parts.append(cell+': '+'; '.join(cell_bits))
    pooled = []
    for arm in ('B', 'R'):
        c = ci([delay[arm][s][1]-delay['A'][s][1] for s in delay[arm] if s in delay['A']])
        pooled.append(f"{arm}-A all={c[0]:+.0f}" + ('' if c[1] is None else f" [{c[1]:+.0f},{c[2]:+.0f}]")
                      + f" n{c[3]}" if c else f'{arm}-A all=n/a')
    vs = ((ups['B']['learning'].get('fullpol') or {}).get('report_valsplit_auc') or {})
    folds = [f for f in vs.get('folds', []) if 'heldback_after' in f]
    nll_b = [f['heldback_before'].get('ev_target_nll') for f in folds if f['heldback_before'].get('ev_target_nll') is not None]
    nll_a = [f['heldback_after'].get('ev_target_nll') for f in folds if f['heldback_after'].get('ev_target_nll') is not None]
    auc = vs.get('ev_act_auc')
    ev = (f"EV-BC valsplit(B): ev_act_auc={'n/a' if auc is None else f'{auc:.3f}'}, target NLL "
          f"{'n/a' if not nll_b else f'{fmean(nll_b):.3f}'}->{'n/a' if not nll_a else f'{fmean(nll_a):.3f}'}")
    return (f"CLEAN5A diag u{u} (report-only, logs only; team delay veh-s, negative = less delay than frozen A): "
            + ' | '.join(parts)+' | '+'; '.join(pooled)+' | '+ev)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root'); p.add_argument('update', type=int)
    p.add_argument('--wait', action='store_true'); p.add_argument('--out')
    a = p.parse_args()
    root = Path(a.root)
    while a.wait and not all((root/f'{arm}_UPDATE_{a.update:03}.json').is_file() for arm in ('A', 'B', 'R')):
        time.sleep(60)
    text = line(root, a.update)
    if a.out:
        with open(a.out, 'a', encoding='utf-8') as f:
            f.write(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
