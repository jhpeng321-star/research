"""S-REGIMEU0: markdown tables from regimeu0.analyze / amendment4 JSON (zero SUMO).

Usage: python -m tools.coopmwm.regimeu0_render ANALYSIS.json AMEND4.json > tables.md
"""
import json
import sys

ORDER = ('D3200_p60', 'D3200_p40', 'D2800_p60', 'D2800_p40', 'D2400_p60', 'D2400_p40')
ARM_ORDER = ('wall_h_solo2', 'wall_h', 'wall_h_solo0')


def f(x, nd=0):
    if x is None:
        return '-'
    return f'{x:,.{nd}f}' if isinstance(x, (int, float)) else str(x)


def pct(x):
    return '-' if x is None else f'{100*x:+.1f}%'


def ci(c, nd=0):
    return '-' if not c or c[0] is None else f'[{f(c[0], nd)}, {f(c[1], nd)}]'


def cells_sorted(d):
    return sorted(d, key=lambda c: ORDER.index(c) if c in ORDER else 99)


def main(analysis, amend4):
    rep = json.load(open(analysis))
    a4 = json.load(open(amend4)) if amend4 else {}
    out = []
    out.append('### Average effect per cell (J over 300-1200 s; diff = strategy - u0, negative = better; 95% seed '
               'bootstrap)')
    out.append('')
    out.append('| cell | seeds | u0 mean J | strategy | mean diff | 95% CI | relative | benefit / harm seeds | '
               'TTC<3 s per ep. (u0 -> arm) | hard brake per ep. (u0 -> arm) | collisions / teleports | CRN ok |')
    out.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for cell in cells_sorted(rep['cells']):
        arms = rep['cells'][cell]
        u0 = arms.get('u0', {})
        for arm in ARM_ORDER:
            a = arms.get(arm)
            if not a or a.get('mean_diff') is None:
                continue
            out.append('| {c} | {n} | {j} | {arm} | {d} | {ci} | {r} | {b}/{h} | {t0} -> {t1} | {h0} -> {h1} | {col}/{tel} | '
                       '{crn}/{n} |'.format(
                           c=cell, n=a['n_paired'], j=f(u0.get('mean_J')), arm=arm, d=f(a['mean_diff']), ci=ci(a['ci95']),
                           r=pct(a.get('rel_diff')), b=round(a['benefit_rate']*a['n_paired']),
                           h=round(a['harm_rate']*a['n_paired']), t0=f(u0.get('ttc_below_3_vs_mean')),
                           t1=f(a.get('ttc_below_3_vs_mean')), h0=f(u0.get('hard_brake_mean'), 1),
                           h1=f(a.get('hard_brake_mean'), 1), col=f(a.get('collisions')), tel=f(a.get('teleports')),
                           crn=a.get('crn_ok')))
    out.append('')
    out.append('### Group (wall_h) minus best solo strategy (best solo re-chosen per bootstrap resample)')
    out.append('')
    out.append('| cell | solos compared | group - best solo | 95% CI |')
    out.append('|---|---|---|---|')
    for cell in cells_sorted(rep['cells']):
        g = rep['cells'][cell].get('group_minus_best_solo')
        if g:
            out.append(f'| {cell} | {", ".join(g["solos"])} | {f(g["point"])} | {ci(g["ci95"])} |')
    if a4:
        out.append('')
        out.append('### Amendment 4: per-seed distribution (episode diff, veh-s)')
        out.append('')
        out.append('| cell | strategy | benefit | harm | 25% | median | 75% | sorted per-seed diffs |')
        out.append('|---|---|---|---|---|---|---|---|')
        for cell in cells_sorted(a4):
            for arm in ARM_ORDER:
                r = a4[cell]['strategies'].get(arm)
                if not r:
                    continue
                q = r['quartiles']
                vals = ', '.join(f'{v/1000:+.1f}k' for v in sorted(r['per_seed'].values()))
                out.append(f'| {cell} | {arm} | {r["benefit"]:.2f} | {r["harm"]:.2f} | {f(q[0])} | {f(q[1])} | {f(q[2])} | '
                           f'{vals} |')
        out.append('')
        out.append('### Amendment 4: conditional effect by observable closing-lane congestion at window start '
                   '(u0 arm; mean window diff per 100 s, seed-cluster 95% CI)')
        out.append('')
        out.append('| cell | strategy | primary index terciles (c = 31.29 - block speed) | t1 (least congested) | t2 | '
                   't3 (most congested) | secondary (slow count) t1 / t2 / t3 |')
        out.append('|---|---|---|---|---|---|---|')
        for cell in cells_sorted(a4):
            co = a4[cell]
            for arm in ARM_ORDER:
                r = co['strategies'].get(arm)
                if not r:
                    continue
                p, s = r['by_primary_tercile'], r['by_secondary_tercile']
                cell_txt = 'cuts {}, {} (c = 0 share {:.2f})'.format(f(co['primary_cuts'][0], 1), f(co['primary_cuts'][1], 1),
                                                                    co['primary_zero_share'] or 0)

                def b(x):
                    return f'{f(x["mean_dw"])} {ci(x["ci95"])} (n={x["n_windows"]})'
                out.append(f'| {cell} | {arm} | {cell_txt} | {b(p["t1"])} | {b(p["t2"])} | {b(p["t3"])} | '
                           f'{f(s["t1"]["mean_dw"])} / {f(s["t2"]["mean_dw"])} / {f(s["t3"]["mean_dw"])} |')
        out.append('')
        out.append('### Amendment 4: split-seed gate (theta chosen on seeds 1-12, evaluated on seeds 13-24)')
        out.append('')
        out.append('| cell | strategy | theta (c >=) | gated share of test windows | gated test mean | 95% CI | '
                   'benefit / harm | always-apply test mean | 95% CI |')
        out.append('|---|---|---|---|---|---|---|---|---|')
        for cell in cells_sorted(a4):
            for arm in ARM_ORDER:
                r = a4[cell]['strategies'].get(arm)
                if not r or not r.get('split_gate'):
                    continue
                g = r['split_gate']
                out.append(f'| {cell} | {arm} | {f(g["theta"], 2)} | {g["gated_fraction_test"]:.2f} | '
                           f'{f(g["gated_test_mean"])} | {ci(g["gated_test_ci95"])} | {g["gated_benefit"]:.2f} / '
                           f'{g["gated_harm"]:.2f} | {f(g["always_test_mean"])} | {ci(g["always_test_ci95"])} |')
    print('\n'.join(out))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
