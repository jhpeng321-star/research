"""S-WMVERIFY fairness-tail report (MAIN 2026-11-08: report worst group and CVaR90 prominently, D2400 first).

Zero-simulation reader of an S-WMVERIFY root's eval-only vehicle logs (<root>/vehicle_logs/<job>/VEHICLES.jsonl.gz),
using the registered per-vehicle convention (wmverify_fair.per_vehicle: window [300, 1200) road delay incl. depart
delay, N6 amendment 13) and labels from u0 by vehicle id (wmverify_fair._labels). Per cell and arm, paired by seed:
- worst group: the co-primary group (origin:L0/L1/L2, class:hdv, class:cav) with the largest mean per-vehicle delay
  increase vs the reference; its value and paired 95% bootstrap CI;
- CVaR90 = mean of the worst ceil(0.1 n) per-vehicle delays (pervehicle.cvar90), arm - reference, paired CI;
- p90 and Gini changes (pervehicle);
- per-group within-group CVaR90 change; worst_group_cvar90 = the group with the largest mean change (paired CI).
References: u0 and the incumbent wall_g_solo2. Convention: diff = arm - ref; positive = worse.

Usage: python -m tools.wzprop.wmverify_fairx --root ROOT [ROOT ...] --out FAIRX.json [--first D2400]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tools.wzprop import pervehicle as PV
from tools.wzprop import wmverify_fair as WF

ID_RE = re.compile(r'^(?:(D\d+_p\d+)_)?s(\d+)_(.+)$')
REFS = ('u0', 'wall_g_solo2')


def episode_stats(pv, labels):
    per = [d for d, _, _ in pv.values()]
    gm = WF._group_means(pv, labels)
    members = {g: [] for g in WF.CO_PRIMARY_GROUPS}
    for vid, (d, _, _) in pv.items():
        for g in labels.get(vid, ()):
            if g in members:
                members[g].append(d)
    return dict(groups={g: gm.get(g) for g in WF.CO_PRIMARY_GROUPS}, cvar90=PV.cvar90(per), p90=PV.p90(per),
                gini=PV.gini(per), n=len(per),
                group_cvar90={g: (PV.cvar90(v) if v else None) for g, v in members.items()})


def collect(roots):
    """{cell: {seed: {arm: per_vehicle}}} plus problems."""
    data, problems = {}, []
    for root in roots:
        vd = Path(root) / 'vehicle_logs'
        for jd in sorted(vd.iterdir()) if vd.is_dir() else []:
            m = ID_RE.match(jd.name)
            p = jd / WF.VEHICLE_LOG
            if not m or not p.exists():
                continue
            pv, prob = WF.per_vehicle(p)
            if prob:
                problems.append(dict(job=jd.name, problems=prob))
            data.setdefault(m.group(1) or '?', {}).setdefault(int(m.group(2)), {})[m.group(3)] = pv
    return data, problems


def report(data, n_boot=10_000):
    out = {}
    for cell, seeds in sorted(data.items()):
        stats = {}
        for s, arms in seeds.items():
            if 'u0' not in arms:
                continue
            lab = WF._labels(arms['u0'])
            stats[s] = {a: episode_stats(pv, lab) for a, pv in arms.items()}
        arms = sorted({a for d in stats.values() for a in d})
        cell_out = {}
        for arm in arms:
            row = dict(arm=arm, n_seeds=sum(1 for d in stats.values() if arm in d))
            for ref in REFS:
                if arm == ref:
                    continue
                ss = [s for s, d in stats.items() if arm in d and ref in d]
                if not ss:
                    continue
                gd = {}
                for g in WF.CO_PRIMARY_GROUPS:
                    d = [stats[s][arm]['groups'][g] - stats[s][ref]['groups'][g] for s in ss
                         if stats[s][arm]['groups'][g] is not None and stats[s][ref]['groups'][g] is not None]
                    if d:
                        gd[g] = dict(mean=sum(d) / len(d), ci95=PV.boot_ci(d, n_boot), n=len(d))
                r = dict(n=len(ss), groups=gd)
                if gd:
                    wg = max(gd, key=lambda g: gd[g]['mean'])
                    r.update(worst_group=wg, worst_group_inc=gd[wg]['mean'], worst_group_ci95=gd[wg]['ci95'])
                gc = {}
                for g in WF.CO_PRIMARY_GROUPS:     # within-group CVaR90 (worst ceil(0.1 n_g) vehicles of group g)
                    d = [stats[s][arm]['group_cvar90'][g] - stats[s][ref]['group_cvar90'][g] for s in ss
                         if stats[s][arm]['group_cvar90'][g] is not None and stats[s][ref]['group_cvar90'][g] is not None]
                    if d:
                        gc[g] = dict(mean=sum(d) / len(d), ci95=PV.boot_ci(d, n_boot), n=len(d))
                r['group_cvar90'] = gc
                if gc:
                    wc = max(gc, key=lambda g: gc[g]['mean'])
                    r.update(worst_group_cvar90=wc, worst_group_cvar90_diff=gc[wc]['mean'],
                             worst_group_cvar90_ci95=gc[wc]['ci95'])
                for k in ('cvar90', 'p90', 'gini'):
                    d = [stats[s][arm][k] - stats[s][ref][k] for s in ss
                         if stats[s][arm][k] is not None and stats[s][ref][k] is not None]
                    if d:
                        r[f'{k}_diff'] = sum(d) / len(d)
                        r[f'{k}_ci95'] = PV.boot_ci(d, n_boot)
                row[f'vs_{ref}'] = r
            row['cvar90_abs_mean'] = (sum(d[arm]['cvar90'] for d in stats.values() if arm in d)
                                      / max(1, row['n_seeds']))
            cell_out[arm] = row
        out[cell] = cell_out
    return out


def headline(rep, first=None):
    """Compact lines: cell arm | worst group vs u0 | CVaR90 diff vs u0 | CVaR90 diff vs incumbent."""
    lines = []
    cells = sorted(rep, key=lambda c: (not (first and c.startswith(first)), c))
    for c in cells:
        for arm, r in rep[c].items():
            if arm == 'u0':
                continue
            u, i = r.get('vs_u0', {}), r.get('vs_wall_g_solo2', {})
            lines.append(f"{c} {arm}: worst {u.get('worst_group')} {u.get('worst_group_inc', float('nan')):+.1f} s/veh; "
                         f"CVaR90 vs u0 {u.get('cvar90_diff', float('nan')):+.1f}; "
                         f"vs inc {i.get('cvar90_diff', float('nan')):+.1f}")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--first', default='D2400')
    a = ap.parse_args(argv)
    data, problems = collect(a.root)
    rep = report(data)
    doc = dict(schema='wmverify.fairx.v1', convention='diff = arm - ref, positive = worse; s/veh window [300,1200)',
               roots=[str(r) for r in a.root], problems=problems, cells=rep, headline=headline(rep, a.first))
    Path(a.out).write_bytes((json.dumps(doc, indent=1) + '\n').encode())
    print('\n'.join(doc['headline']))


if __name__ == '__main__':
    main()
