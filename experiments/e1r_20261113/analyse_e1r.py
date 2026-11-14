"""E1R analysis (ZERO SUMO): WM-ranked 12 (A80) vs random 12 (E1R) from the same 607-plan S2b pool, admitted on the
same 24 A68 admission worlds with the same ADMIT_RULE (a80_s2b 48c5c448 / a75_d2forms 8a2533ef).

Inputs: the two sealed admission receipts A80_ADMISSION_D2800_p60.json (schema coopmwm.a75d2.admission.v1; field
`verdicts` = one a37_cellbank.unc_verdict row per tested plan: family, PASS, mean = mean over 24 worlds of
(plan - MAPPO60C) total delay veh-s, ci95, per_seed) and the E1R draw file (to count drawn plans that never reached
admission, e.g. a pin-top mapping refusal: those count as NOT admitted and are listed).

Primary   : admitted count WM-12 vs random-12; two-sided Fisher exact p on the 2x2 admitted / not admitted table.
Secondary : per-plan mean (plan - MAPPO60C) total delay (veh-s, negative = less delay); Mann-Whitney U two-sided
            (exact when no ties, else normal approximation with tie correction) WM vs random.
The hand-written duo12_v20 / duo12_v18 (dense rule seeds, not search products) are reported both included in and
excluded from the WM arm.

usage: python -I analyse_e1r.py A80_ADMISSION.json E1R_ADMISSION.json E1R_DRAW.json [--out OUT.json]
       python -I analyse_e1r.py --selftest          (synthetic inputs only)
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

HAND = ('duo12_v20', 'duo12_v18')


# ------------------------------------------------------------------------------------------------ statistics (pure)
def _hyper_p(a, r1, c1, n):
    return math.comb(c1, a) * math.comb(n - c1, r1 - a) / math.comb(n, r1)


def fisher_two_sided(a, b, c, d):
    """2x2 [[a, b], [c, d]]; two-sided p = sum of tables with probability <= observed (scipy convention)."""
    r1, c1, n = a + b, a + c, a + b + c + d
    lo, hi = max(0, r1 + c1 - n), min(r1, c1)
    p0 = _hyper_p(a, r1, c1, n)
    return min(1.0, sum(p for x in range(lo, hi + 1) if (p := _hyper_p(x, r1, c1, n)) <= p0 * (1 + 1e-7)))


def _ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def mann_whitney(x, y):
    """Two-sided Mann-Whitney U for samples x, y. Returns dict(U_x, p, method)."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return dict(U_x=None, p=None, method='empty sample')
    allv = list(x) + list(y)
    r = _ranks(allv)
    u_x = sum(r[:nx]) - nx * (nx + 1) / 2.0
    ties = len(set(allv)) != len(allv)
    mu = nx * ny / 2.0
    if not ties and nx + ny <= 30:
        # exact null distribution of U by counting subsets (rank sums of nx ranks among 1..N)
        N = nx + ny
        counts = {}
        for comb in itertools.combinations(range(1, N + 1), nx) if math.comb(N, nx) <= 3_000_000 else ():
            u = sum(comb) - nx * (nx + 1) / 2.0
            counts[u] = counts.get(u, 0) + 1
        if counts:
            tot = sum(counts.values())
            dev = abs(u_x - mu)
            p = sum(v for u, v in counts.items() if abs(u - mu) >= dev - 1e-9) / tot
            return dict(U_x=u_x, p=min(1.0, p), method='exact')
    N = nx + ny
    tie_term = 0.0
    for v in set(allv):
        t = allv.count(v)
        tie_term += t ** 3 - t
    var = nx * ny / 12.0 * ((N + 1) - tie_term / (N * (N - 1)))
    if var <= 0:
        return dict(U_x=u_x, p=1.0, method='normal (all tied)')
    z = (abs(u_x - mu) - 0.5) / math.sqrt(var)
    p = math.erfc(max(z, 0.0) / math.sqrt(2.0))
    return dict(U_x=u_x, p=min(1.0, p), method='normal, tie-corrected, continuity 0.5')


# ------------------------------------------------------------------------------------------------ receipts
def arm_rows(receipt):
    rows = {}
    for v in receipt['verdicts']:
        rows[v['family']] = dict(family=v['family'], admitted=bool(v['PASS']), mean_diff_veh_s=float(v['mean']),
                                 ci95=v['ci95'], delay_test_pass=bool(v.get('delay_test_pass', v['ci95'][1] < 0)),
                                 group_rule_pass=bool(v.get('group_rule_pass', v.get('allowed'))))
    return rows


def summarise(wm, rnd, rnd_missing):
    a = sum(r['admitted'] for r in wm)
    c = sum(r['admitted'] for r in rnd)
    nw, nr = len(wm), len(rnd) + len(rnd_missing)
    xs, ys = [r['mean_diff_veh_s'] for r in wm], [r['mean_diff_veh_s'] for r in rnd]
    return dict(
        n_wm=nw, n_random=nr, admitted_wm=a, admitted_random=c,
        table=[[a, nw - a], [c, nr - c]], fisher_two_sided_p=fisher_two_sided(a, nw - a, c, nr - c),
        random_not_tested=sorted(rnd_missing),
        delay_secondary=dict(
            convention='mean over 24 worlds of (plan - MAPPO60C) total delay, veh-s; negative = less delay',
            wm_median=_median(xs), random_median=_median(ys), wm_mean=_mean(xs), random_mean=_mean(ys),
            mann_whitney=mann_whitney(xs, ys),
            note='random plans never tested (refused) have no delay value and are absent from this comparison'))


def _median(v):
    if not v:
        return None
    s = sorted(v)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def _mean(v):
    return sum(v) / len(v) if v else None


def analyse(a80, e1r, draw):
    wm = arm_rows(a80)
    rnd = arm_rows(e1r)
    drawn = [d['name'] for d in draw['drawn']]
    extra = sorted(set(rnd) - set(drawn))
    if extra:
        raise SystemExit(f'E1R receipt has plans that were not drawn: {extra}')
    if set(wm) & set(rnd):
        raise SystemExit(f'overlap between arms: {sorted(set(wm) & set(rnd))}')
    for k in ('panel_sha256', 'rule', 'reference', 'seeds'):
        if a80.get(k) != e1r.get(k):
            raise SystemExit(f'receipts differ in {k}: not the same admission test')
    missing = [n for n in drawn if n not in rnd]
    wm_all = list(wm.values())
    wm_search = [r for r in wm_all if r['family'] not in HAND]
    return dict(
        schema='e1r.analysis.v1', primary='admitted count WM-12 vs random-12, Fisher exact two-sided',
        same_test=dict(panel_sha256=a80.get('panel_sha256'), reference=a80.get('reference'), n_seeds=len(a80['seeds'])),
        receipts=dict(a80=a80.get('receipt_sha256'), e1r=e1r.get('receipt_sha256')),
        hand_written_in_wm=[f for f in HAND if f in wm],
        including_hand_written=summarise(wm_all, list(rnd.values()), missing),
        excluding_hand_written=summarise(wm_search, list(rnd.values()), missing),
        per_plan=dict(wm=wm_all, random=list(rnd.values())))


# ------------------------------------------------------------------------------------------------ self-test
def _synthetic(names, passes, means, seeds):
    return dict(panel_sha256='p' * 64, rule='R', reference='A68V_MAPPO60C', seeds=seeds, receipt_sha256='x',
                verdicts=[dict(family=n, PASS=p, mean=m, ci95=[m - 50, m + (-1 if p else 30)], allowed=True)
                          for n, p, m in zip(names, passes, means)])


def selftest():
    import random
    rng = random.Random(1)
    seeds = list(range(992500, 992524))
    wm_names = ['duo12_v20', 'duo12_v18'] + [f'mut_wm{i}' for i in range(10)]
    rd_names = [f'rand_r{i}' for i in range(12)]
    a80 = _synthetic(wm_names, [True] * 7 + [False] * 5, [rng.uniform(-300, 50) for _ in wm_names], seeds)
    e1r = _synthetic(rd_names[:11], [True] * 2 + [False] * 9, [rng.uniform(-100, 200) for _ in range(11)], seeds)
    draw = dict(drawn=[dict(name=n) for n in rd_names])
    out = analyse(a80, e1r, draw)
    inc, exc = out['including_hand_written'], out['excluding_hand_written']
    assert inc['admitted_wm'] == 7 and inc['admitted_random'] == 2 and inc['n_random'] == 12
    assert inc['random_not_tested'] == ['rand_r11'] and exc['n_wm'] == 10 and exc['admitted_wm'] == 5
    try:
        from scipy import stats
        for t in ([[7, 5], [2, 10]], [[5, 5], [2, 10]], [[0, 12], [0, 12]], [[12, 0], [0, 12]], [[3, 9], [4, 8]]):
            mine = fisher_two_sided(t[0][0], t[0][1], t[1][0], t[1][1])
            ref = stats.fisher_exact(t, alternative='two-sided')[1]
            assert abs(mine - ref) < 1e-9, (t, mine, ref)
        xs = [r['mean_diff_veh_s'] for r in out['per_plan']['wm']]
        ys = [r['mean_diff_veh_s'] for r in out['per_plan']['random']]
        m = mann_whitney(xs, ys)
        ref = stats.mannwhitneyu(xs, ys, alternative='two-sided', method='exact')
        assert abs(m['p'] - ref.pvalue) < 1e-9 and m['method'] == 'exact', (m, ref)
        xt, yt = [1, 2, 2, 3, 5, 5, 7], [2, 3, 3, 4, 6, 8, 8, 9]
        m2 = mann_whitney(xt, yt)
        ref2 = stats.mannwhitneyu(xt, yt, alternative='two-sided', method='asymptotic', use_continuity=True)
        assert abs(m2['p'] - ref2.pvalue) < 1e-9, (m2, ref2)
        print('scipy cross-check ok')
    except ImportError:
        print('scipy not available: cross-check skipped')
    bad = dict(e1r, seeds=seeds[:-1])
    try:
        analyse(a80, bad, draw)
        raise AssertionError('seed mismatch not detected')
    except SystemExit:
        pass
    print(json.dumps(dict(inc=dict((k, inc[k]) for k in ('admitted_wm', 'admitted_random', 'table',
                                                         'fisher_two_sided_p')),
                          exc=dict((k, exc[k]) for k in ('admitted_wm', 'admitted_random', 'table',
                                                         'fisher_two_sided_p')),
                          mw=inc['delay_secondary']['mann_whitney']), indent=1))
    print('SELFTEST OK (synthetic data only)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a80', nargs='?')
    ap.add_argument('e1r', nargs='?')
    ap.add_argument('draw', nargs='?')
    ap.add_argument('--out')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    rd = [json.loads(Path(p).read_text(encoding='utf-8')) for p in (a.a80, a.e1r, a.draw)]
    out = analyse(*rd)
    txt = json.dumps(out, indent=1)
    if a.out:
        if Path(a.out).exists():
            raise SystemExit(f'REFUSE: {a.out} exists')
        Path(a.out).write_text(txt, encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    main()
