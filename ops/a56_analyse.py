"""AMENDMENT 56 (S-RANDCTRL60) analysis, exactly as registered (zero physics; laptop; numpy + stdlib only).

Input: A56_REDUCE_D2800_p60.json (sealed box reduction: per-world J_veh_s / safety / activations of U0, the six A42
WM top-ranked plans and the 12 random plans on the 24 A42 verification worlds; UNC admission per plan computed on the
box with tools.coopmwm.a37_cellbank.unc_verdict + nm_finaleval._bank_compare, i.e. the A42 admission code).

PRIMARY (registered): per world, mean total delay over the six WM plans minus mean over the 12 random plans; paired
percentile bootstrap over the 24 worlds, 10,000 resamples, seed 20261104 (planner_eval._paired_ci verbatim: numpy
default_rng(seed).integers whole-world resampling, quantiles 2.5 / 97.5 %). SUPPORTED iff CI95 upper < 0.
Secondary (descriptive): admission counts random k/12 vs WM 4/6 (Fisher exact two-sided), per-plan plan - U0 with CI,
activations, group-limit results, hard brakes, TTC, collisions, teleports; A45 shown as a separate, non-pooled row.

    python ops/a56_analyse.py --reduce <A56_REDUCE json> [--a45 <ADMISSION_A45 json>] --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np

REPS, SEED, LEVEL = 10_000, 20261104, .95
WM_ADMITTED_A42 = 4


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_seal(doc, field):
    body = {k: v for k, v in doc.items() if k != field}
    canon = json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False)   # tools.coopnative canonical
    return hashlib.sha256(canon.encode()).hexdigest() == doc[field]


def paired_ci(deltas, n_boot=REPS, boot_seed=SEED, alpha=1-LEVEL):
    """tools/coopmwm/planner_eval._paired_ci verbatim."""
    d = np.asarray(deltas, float)
    idx = np.random.default_rng(boot_seed).integers(0, len(d), (n_boot, len(d)))
    boot = d[idx].mean(1)
    lo, hi = np.quantile(boot, [alpha/2, 1-alpha/2])
    p = min(1., 2*min(np.sum(boot <= 0)+1, np.sum(boot >= 0)+1)/(n_boot+1))
    return dict(n_worlds=len(d), mean_delta_vs=float(d.mean()), ci95=[float(lo), float(hi)],
                p_two_sided=float(p), worlds_improved=int(np.sum(d < 0)))


def fisher_two_sided(a, b, c, d):
    """2x2 [[a, b], [c, d]]; two-sided p = sum of tables (fixed margins) with probability <= observed."""
    r1, c1, n = a+b, a+c, a+b+c+d

    def pr(x):
        return comb(c1, x)*comb(n-c1, r1-x)/comb(n, r1)
    p0 = pr(a)
    lo, hi = max(0, r1+c1-n), min(r1, c1)
    return float(min(1.0, sum(pr(x) for x in range(lo, hi+1) if pr(x) <= p0*(1+1e-7))))


def analyse(red, a45=None):
    worlds = red['worlds']
    wm, rnd = red['wm_top6'], red['random12']
    if len(wm) != 6 or len(rnd) != 12 or len(worlds) != 24:
        raise SystemExit('reduce file does not have 6 WM plans, 12 random plans, 24 worlds')
    J = {a: np.array([red['per_world'][a][w]['J_veh_s'] for w in worlds], float) for a in ['U0']+wm+rnd}
    mean_wm = np.mean([J[a] for a in wm], 0)
    mean_rnd = np.mean([J[a] for a in rnd], 0)
    prim = paired_ci(mean_wm-mean_rnd)
    supported = bool(prim['ci95'][1] < 0)
    V = red['verdicts']
    k_rnd = sum(bool(V[f]['PASS']) for f in rnd)
    k_wm = sum(bool(V[f]['PASS']) for f in wm)
    if k_wm != WM_ADMITTED_A42 or not all(V[f].get('matches_a42_admission') for f in wm):
        raise SystemExit('the recomputed WM admission does not reproduce the A42 receipt')
    fisher = fisher_two_sided(k_wm, 6-k_wm, k_rnd, 12-k_rnd)

    def row(f, src):
        v = V[f]
        g = v['groups']
        return dict(family=f, source=src, PASS=bool(v['PASS']), mean_minus_u0=v['mean'], ci95_unc=v['ci95'],
                    secondary_ci95_panel=v['total_minus_u0_secondary']['ci95'], benefit_share=v['benefit'],
                    allowed=bool(v['allowed']), group_all_pass=g.get('all_pass'), worst_group=v['worst_group'],
                    worst_group_inc_s=v['worst_group_inc'], activations_per_ep=v['activations_per_ep'],
                    episodes_with_activation=v['episodes_with_activation'], safety_totals=v['safety_totals'],
                    cvar90_diff_s=(v.get('cvar90_vs_u0') or {}).get('diff_mean_s'))
    rows = [row(f, 'WM_A42') for f in wm]+[row(f, 'RANDOM_A56') for f in rnd]
    safety = {grp: {k: int(sum(V[f]['safety_totals'][k] for f in fams)) for k in V[rnd[0]]['safety_totals']}
              for grp, fams in (('WM6', wm), ('RANDOM12', rnd))}
    safety['U0'] = {k: int(sum(red['per_world']['U0'][w]['safety'][k] for w in worlds))
                    for k in red['per_world']['U0'][worlds[0]]['safety']}
    out = dict(
        schema='coopmwm.a56.randctrl.results.v1', amendment='56', cell=red['cell'], label='S-RANDCTRL60',
        reduce_analysis_sha256=red['analysis_sha256'], plans_sha256=red['plans_sha256'],
        convention='diff = treatment - reference (veh-s total delay; negative = less delay)',
        primary=dict(contrast='mean(six WM top-ranked A42 plans) - mean(12 random plans), per world',
                     bootstrap=dict(reps=REPS, seed=SEED, level=LEVEL, method='paired percentile (planner_eval._paired_ci)'),
                     mean_wm6_J=float(mean_wm.mean()), mean_random12_J=float(mean_rnd.mean()),
                     mean_u0_J=float(J['U0'].mean()), **prim,
                     rule='SUPPORTED ("WM proposals better than random") iff CI95 upper < 0',
                     verdict='SUPPORTED' if supported else 'NOT SUPPORTED'),
        admission=dict(rule=red['unc_rule'], random_k=k_rnd, random_n=12, wm_k=k_wm, wm_n=6,
                       fisher_exact_two_sided_p=fisher,
                       random_admitted=[f for f in rnd if V[f]['PASS']]),
        secondary_rows_not_pooled={}, plans=rows, safety_totals=safety,
        per_world=dict(worlds=worlds, u0=J['U0'].tolist(), mean_wm6=mean_wm.tolist(), mean_random12=mean_rnd.tolist()),
        charged_sim_s=red['charged_sim_s'])
    if a45 is not None:
        out['secondary_rows_not_pooled']['A45_diverse_lower_ranked'] = dict(
            k=sum(bool(v['PASS']) for v in a45['verdicts']), n=len(a45['verdicts']),
            receipt_sha256=a45['receipt_sha256'])
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--reduce', required=True)
    p.add_argument('--a45')
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    red = json.loads(Path(a.reduce).read_text(encoding='utf-8'))
    if not check_seal(red, 'analysis_sha256'):
        raise SystemExit('reduce file seal does not verify')
    a45 = json.loads(Path(a.a45).read_text(encoding='utf-8')) if a.a45 else None
    res = analyse(red, a45)
    res['inputs'] = dict(reduce_file_sha256=sha_file(a.reduce), a45_file_sha256=sha_file(a.a45) if a.a45 else None)
    out = Path(a.out)/'RANDCTRL_RESULTS_D2800_p60.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(out=str(out), sha256=sha_file(out), primary=res['primary']['verdict'],
                          ci95=res['primary']['ci95'], mean=res['primary']['mean_delta_vs'],
                          admission=res['admission'])))


if __name__ == '__main__':
    main()
