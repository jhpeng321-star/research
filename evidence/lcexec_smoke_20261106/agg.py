import json, glob
from collections import Counter, defaultdict
base = '/root/autodl-tmp/lcexec_smoke_20261106'
res = []
for f in sorted(glob.glob(f'{base}/a*/jobs/*/RESULT.json')):
    res.append(json.load(open(f)))
fails = sorted(f.split('/')[-2] for f in glob.glob(f'{base}/a*/jobs/*/FAILED.json'))
tot = Counter()
fam = defaultdict(Counter)
moves_list = []
cls_all = Counter()
maxdev = 0.
for r in res:
    tot.update(r['totals'])
    tot['actual_sim_s'] += r['actual_sim_s']
    for m in r['moves']:
        f = m.get('family') or 'infeasible'
        c = fam[f]
        c['moves'] += 1
        if m.get('infeasible'):
            continue
        if m.get('refused'):
            c['refused'] += 1
            moves_list.append((r['id'], m['t0'], f, 'REFUSED', m['refused'][:200]))
            continue
        c['executed'] += 1
        c['parity_exact'] += int(m['parity_exact'])
        c['matched_calls'] += m['matched']
        for k, v in m['mismatch_classes'].items():
            c['mm_'+k] += v
            cls_all[k] += v
        c['closing_members'] += len(m['closing_lane_members'])
        c['closing_entered'] += len(m['closing_lane_entered'])
        maxdev = max(maxdev, m['max_target_dev'])
        for v, pm in m['per_member'].items():
            if pm['lane_target_wm'] is not None and pm['lane_target_wm'] != pm['lane_t0_wm']:
                c['lane_members'] += 1
                c['lane_reached'] += int(bool(pm['lane_reached'] or pm['held_seconds']))
                c['gate_blocked_s'] += pm['gate_blocked_seconds']
                c['guard_s'] += pm['guard_seconds']
        for k in ('hard_brake_vs', 'ttc_below_3_vs'):
            c['win_'+k] += m['window'].get(k, 0.)
        moves_list.append((r['id'], m['t0'], f, 'exact' if m['parity_exact'] else m['mismatch_classes'],
                           f"closing {len(m['closing_lane_entered'])}/{len(m['closing_lane_members'])}"))
out = dict(jobs_ok=len(res), jobs_failed=fails, totals=dict(tot), families={k: dict(v) for k, v in fam.items()},
           mismatch_classes=dict(cls_all), max_target_dev=maxdev)
print(json.dumps(out, indent=1))
for x in moves_list:
    print(x)
json.dump(dict(out, moves=moves_list), open(f'{base}/SMOKE_SUMMARY.json', 'w'), indent=1)
