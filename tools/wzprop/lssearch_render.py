"""Markdown tables for S-LOWSHARE-SEARCH analyses (zero physics): python -m tools.wzprop.lssearch_render A.json"""
from __future__ import annotations

import json
import sys


def _k(v):
    return 'n/a' if v is None else f'{v:+,.0f}'


def _ci(c):
    return 'n/a' if not c else f'[{c[0]:+,.0f}, {c[1]:+,.0f}]'


def _f(v, d=3):
    return 'n/a' if v is None else f'{v:+.{d}f}'


def render(rep):
    out = []
    for cell, arms in sorted(rep['cells'].items()):
        u0 = arms.get('u0', {}).get('mean_J')
        out.append(f'### {cell} ({rep["stage"]}; u0 mean J {u0:,.0f} veh-s)\n' if u0 else f'### {cell}\n')
        out.append('| arm | n | vs u0 mean [95% CI] | rel | vs solo2 mean [95% CI] | Bonf-6 CI | beats solo2 | '
                   'acts | members | coll/tele | hard brake | TTC<3 |')
        out.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
        order = [a for a in arms if 'minus' not in a and a != 'u0']
        order.sort(key=lambda a: (a != 'wall_h_solo2', (arms[a].get('vs_solo2') or {}).get('mean', 0)))
        for a in order:
            v = arms[a]
            s2 = v.get('vs_solo2') or {}
            beats = f"{round(s2['beats_fraction'] * s2['n'])}/{s2['n']}" if s2 else ''
            out.append(
                f"| {a} | {v.get('n_paired', v.get('n'))} | {_k(v.get('mean_diff'))} {_ci(v.get('ci95'))} | "
                f"{(v.get('rel_diff') or 0) * 100:+.1f}% | {_k(s2.get('mean'))} {_ci(s2.get('ci95'))} | "
                f"{_ci(s2.get('ci_bonferroni6'))} | "
                f"{beats} | "
                f"{(v.get('mean_activations') or 0):.1f} | {(v.get('mean_members') or 0):.2f} | "
                f"{v.get('collisions'):.0f}/{v.get('teleports'):.0f} | {(v.get('hard_brake_mean') or 0):.1f} "
                f"(u0 {(arms['u0'].get('hard_brake_mean') or 0):.1f}) | {(v.get('ttc_below_3_vs_mean') or 0):.0f} "
                f"(u0 {(arms['u0'].get('ttc_below_3_vs_mean') or 0):.0f}) |")
        out.append('')
        out.append('Fairness vs u0 (paired per-seed diffs of mean per-vehicle window delay, s/veh; * = 95% CI > 0, '
                   'i.e. significantly worse off):\n')
        out.append('| arm | FAIR | FAIR_point | lane0 | lane1 | lane2 | HDV | CAV | Gini diff | Jain diff | worst-10% diff |')
        out.append('|---|---|---|---|---|---|---|---|---|---|---|')
        for a in order:
            f = arms[a].get('fairness')
            if not f:
                continue
            g = f['groups']

            def cell_of(name):
                x = g.get(name)
                if not x:
                    return 'n/a'
                return f"{x['mean_per_vehicle_diff']:+.1f} {_ci([round(c, 1) for c in x['ci95']])}{'*' if x['worse_sig'] else ''}"
            out.append(f"| {a} | {'yes' if f['FAIR'] else 'no'} | {'yes' if f['FAIR_point'] else 'no'} | "
                       + ' | '.join(cell_of(n) for n in ('lane0', 'lane1', 'lane2', 'hdv', 'cav')) + ' | '
                       + ' | '.join(_f((f.get(k) or {}).get('mean_diff'), 4) for k in ('gini', 'jain'))
                       + f" | {_f((f.get('worst10') or {}).get('mean_diff'), 1)} |")
        out.append('')
    if rep.get('selection'):
        out.append('Selection (pre-registered top 3 by screen mean of group - solo2): '
                   + '; '.join(f'{c}: {", ".join(v)}' for c, v in sorted(rep['selection'].items())))
    return '\n'.join(out).replace('{', '{').replace('[+', '[+')


if __name__ == '__main__':
    print(render(json.loads(open(sys.argv[1], encoding='utf-8').read())))
