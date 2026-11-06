"""Matched-effort comparison: WM-proposed vs structured-search vs random proposals on the same moments.

    python -m tools.wzwm compare --summary REF/PG_SUMMARY.json --summary WMROOT/results/WMC_SUMMARY.json \
        [--proposals PROPOSALS.jsonl] [--cap SIM_S] [--salt S] --out COMPARE.json

Arms come from the verdicts of N3/PLANGEN/LOWCAV summaries (arm B -> 'structured', R -> 'random') and of
WMC summaries (field 'proposer': wm / random / structured). Only moments screened by EVERY arm count.
Matched effort = the same moments, the same per-moment screening count (WM k = the structured count, or
fixed k), the same funnel rules, and a standalone SUMO cost per arm and moment from the funnel stages
reached (shared base runs are charged to every arm that needed them):
    300 (base r0) + 300 per screened group + 300 per solo member (solo sets run)
    + rep1: 300 base + per replicated candidate 300 group + 300 per solo member
    + rep2: 300 base + 120 base extension + per candidate 300 group + 300 per solo member + 120 extension
Reported per arm: verified plans, verified plans per 1,000 sim-s (moment-bootstrap 95% CI), admission
rate per screened candidate and per moment (Wilson 95%), funnel counts; paired WM-minus-other
differences (moment bootstrap); a fixed-budget view (moments in an outcome-blind sha256 order, each arm
stops at the first moment that no longer fits the cap); WM ranking diagnostics and a post hoc
WM-abstention view (zero extra SUMO). Diagnostic evidence, not a gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import geometry as G

H_MAIN, H_REP, H_EXT = 300.0, 300.0, 120.0
REP1 = {'r1', 'r2_missing', 'r2', 'r0ext_missing', 'r0ext', 'admitted'}
REP2 = {'r2', 'r0ext_missing', 'r0ext', 'admitted'}
SYN = REP1 | {'r1_missing', 'r0_not_picked'}
B_BOOT, RNG = 2000, 20261106
COST_MODEL = ('standalone per arm and moment: 300 base r0 + 300 per screened group + 300 per solo member; rep1 '
              '300 base + per candidate 300 group + 300 per solo member; rep2 300 base + 120 base ext + per '
              'candidate 300 group + 300 per solo member + 120 group ext (shared base runs charged to every arm)')


def proposer(vd):
    if vd.get('proposer'):
        return vd['proposer']
    return 'random' if vd.get('arm') == 'R' or vd.get('family') == 'R_random' else 'structured'


def moment_cost(vds):
    """Standalone SUMO sim-s of one arm's funnel at one moment (see module docstring)."""
    c = H_MAIN * (1 + len(vds))
    c += sum(H_MAIN * len(v['members']) for v in vds if v.get('S_r0') is not None)
    r1 = [v for v in vds if v['stage'] in REP1]
    if r1:
        c += H_REP + sum(H_REP * (1 + len(v['members'])) for v in r1)
    r2 = [v for v in vds if v['stage'] in REP2]
    if r2:
        c += H_REP + H_EXT + sum(H_REP * (1 + len(v['members'])) + H_EXT for v in r2)
    return c


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def ranks(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind='mergesort')
    r = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return round(float(np.corrcoef(ra, rb)[0, 1]), 4)


def load_verdicts(paths):
    """arm -> sid -> verdicts. An arm name already taken by an earlier summary gets the suffix @<index>
    (e.g. the count-matched random arm of a --with-random WM root becomes 'random@1')."""
    by = defaultdict(lambda: defaultdict(list))
    sources = {}
    for i, p in enumerate(paths):
        doc = json.loads(G.check_path(p).read_text())
        names = {}
        for vd in doc['verdicts']:
            G.check_seed(vd['seed'], where=f'({p})')
            arm = proposer(vd)
            if arm not in names:
                names[arm] = arm if arm not in sources else f'{arm}@{i}'
                sources[names[arm]] = str(p)
            by[names[arm]][vd['sid']].append(vd)
    return by, sources


def _boot_ratio(adm, cost, rng):
    n = len(adm)
    idx = rng.integers(0, n, (B_BOOT, n))
    return 1000.0 * adm[idx].sum(1) / np.maximum(cost[idx].sum(1), 1e-9)


def arm_metrics(vmap, sids, rng):
    adm = np.array([float(any(v['admitted'] for v in vmap[s])) for s in sids])
    adm_var = np.array([float(sum(bool(v['admitted']) for v in vmap[s])) for s in sids])
    cost = np.array([moment_cost(vmap[s]) for s in sids])
    vds = [v for s in sids for v in vmap[s]]
    n_c = len(vds)
    boot = _boot_ratio(adm_var, cost, rng) if len(sids) else np.zeros(1)
    return dict(
        n_moments=len(sids), n_candidates=n_c, candidates_per_moment=round(n_c / max(1, len(sids)), 3),
        n_solo_sets=sum(v.get('S_r0') is not None for v in vds), n_r0_synergy=sum(v['stage'] in SYN for v in vds),
        n_replicated=sum(v['stage'] in REP1 for v in vds), n_rep2=sum(v['stage'] in REP2 for v in vds),
        admitted_variants=int(adm_var.sum()), admitted_moments=int(adm.sum()), sim_s=float(cost.sum()),
        verified_per_sim_s=float(adm_var.sum() / max(cost.sum(), 1e-9)),
        verified_per_ksim=round(1000.0 * adm_var.sum() / max(cost.sum(), 1e-9), 5),
        verified_per_ksim_ci95=[round(float(np.quantile(boot, .025)), 5), round(float(np.quantile(boot, .975)), 5)],
        admission_rate_candidate=round(adm_var.sum() / max(1, n_c), 4),
        admission_rate_candidate_wilson95=wilson(int(adm_var.sum()), n_c),
        admission_rate_moment=round(adm.sum() / max(1, len(sids)), 4),
        admission_rate_moment_wilson95=wilson(int(adm.sum()), len(sids)),
        stages=dict(sorted(_count(v['stage'] for v in vds).items()))), adm_var, cost


def _count(xs):
    out = defaultdict(int)
    for x in xs:
        out[x] += 1
    return out


def paired(a_adm, a_cost, b_adm, b_cost, rng):
    n = len(a_adm)
    if n == 0:
        return None
    idx = rng.integers(0, n, (B_BOOT, n))
    ra = 1000.0 * a_adm[idx].sum(1) / np.maximum(a_cost[idx].sum(1), 1e-9)
    rb = 1000.0 * b_adm[idx].sum(1) / np.maximum(b_cost[idx].sum(1), 1e-9)
    d = ra - rb
    dm = a_adm[idx].sum(1) - b_adm[idx].sum(1)
    return dict(diff_verified_per_ksim=round(float(1000.0 * a_adm.sum() / max(a_cost.sum(), 1e-9)
                                                    - 1000.0 * b_adm.sum() / max(b_cost.sum(), 1e-9)), 5),
                diff_verified_per_ksim_ci95=[round(float(np.quantile(d, .025)), 5), round(float(np.quantile(d, .975)), 5)],
                diff_admitted=int(a_adm.sum() - b_adm.sum()),
                diff_admitted_ci95=[float(np.quantile(dm, .025)), float(np.quantile(dm, .975))],
                only_first=int(((a_adm > 0) & (b_adm == 0)).sum()), only_second=int(((a_adm == 0) & (b_adm > 0)).sum()),
                diff_sim_s=float(a_cost.sum() - b_cost.sum()))


def fixed_budget(order, per_arm, cap):
    out = {}
    for arm, (adm, cost) in per_arm.items():
        used = got = n = 0
        for i in order:
            if used + cost[i] > cap + 1e-6:
                break
            used += cost[i]
            got += adm[i]
            n += 1
        out[arm] = dict(cap=cap, moments=n, sim_s=used, verified=int(got),
                        verified_per_ksim=round(1000.0 * got / max(used, 1e-9), 5))
    return out


def wm_diagnostics(vmap, sids):
    rows = [v for s in sids for v in vmap[s] if v.get('wm_prediction')]
    if not rows:
        return None
    t_pred = [v['wm_prediction']['T']['mean'] for v in rows]
    t_meas = [v.get('T_r0') for v in rows]
    keep = [i for i, x in enumerate(t_meas) if x is not None]
    s_rows = [v for v in rows if v.get('S_r0') is not None and (v['wm_prediction'].get('S') or {}).get('mean') is not None]
    top_best, n_top = 0, 0
    for s in sids:
        vs = [v for v in vmap[s] if v.get('wm_prediction') and v.get('T_r0') is not None]
        if len(vs) >= 2:
            n_top += 1
            top = min(vs, key=lambda v: v['wm_prediction']['rank'])
            top_best += top['T_r0'] >= max(v['T_r0'] for v in vs)
    sign = defaultdict(int)
    for v in rows:
        if v.get('T_lb_r0') is None or not v['wm_prediction'].get('M'):
            continue
        sign[f"pred_{'pos' if v['wm_prediction']['M']['lcb'] > 0 else 'nonpos'}"
             f"__sumo_T_{'pos' if v['T_lb_r0'] > 0 else 'nonpos'}"] += 1
    return dict(n=len(rows), spearman_T=spearman([t_pred[i] for i in keep], [t_meas[i] for i in keep]),
                spearman_S=spearman([v['wm_prediction']['S']['mean'] for v in s_rows], [v['S_r0'] for v in s_rows]),
                n_S=len(s_rows), top1_has_best_sumo_T=dict(k=int(top_best), n=n_top),
                sign_table=dict(sign))


def abstention(vmap, sids, props, thresholds):
    if not props:
        return None
    out = []
    for thr in thresholds:
        keep = [s for s in sids if props.get(s, {}).get('best_score') is not None and props[s]['best_score'] > thr]
        adm = sum(sum(bool(v['admitted']) for v in vmap[s]) for s in keep)
        cost = sum(moment_cost(vmap[s]) for s in keep)
        lost = sum(sum(bool(v['admitted']) for v in vmap[s]) for s in sids if s not in set(keep))
        out.append(dict(threshold=thr, moments=len(keep), verified=int(adm), sim_s=cost, lost_verified=int(lost),
                        verified_per_ksim=round(1000.0 * adm / max(cost, 1e-9), 5)))
    return out


def compare(summaries, proposals=None, cap=None, salt='wmc-compare-v1', thresholds=(0.0, 100.0, 250.0, 500.0),
            arms=None):
    by, sources = load_verdicts(summaries)
    arms = list(arms or sorted(by))
    missing = [a for a in arms if a not in by]
    if missing:
        raise ValueError(f'arms {missing} not found in the summaries')
    common = sorted(set.intersection(*(set(by[a]) for a in arms)))
    order_sids = sorted(common, key=lambda s: hashlib.sha256(f'{salt}:{s}'.encode()).hexdigest())
    rng = np.random.default_rng(RNG)
    metrics, arrays = {}, {}
    for a in arms:
        m, adm, cost = arm_metrics(by[a], order_sids, rng)
        metrics[a] = m
        arrays[a] = (adm, cost)
    pairs = {}
    if 'wm' in arms:
        for b in arms:
            if b != 'wm':
                pairs[f'wm_minus_{b}'] = paired(*arrays['wm'], *arrays[b], np.random.default_rng(RNG))
    cap = float(cap) if cap is not None else (min(arrays[a][1].sum() for a in arms) if order_sids else 0.0)
    props = {}
    if proposals:
        for line in G.check_path(proposals).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                props[r['sid']] = r
    out = dict(schema='wmc-compare-v1', summaries=sources, arms=arms, n_common_moments=len(common),
               moments_per_arm={a: len(by[a]) for a in by}, order_salt=salt, metrics=metrics, paired=pairs,
               fixed_budget=fixed_budget(range(len(order_sids)), arrays, cap),
               wm_diagnostics=wm_diagnostics(by['wm'], order_sids) if 'wm' in by else None,
               wm_abstention_post_hoc=abstention(by['wm'], order_sids, props, thresholds) if 'wm' in by else None,
               cost_model=COST_MODEL,
               interpretation='diagnostic evidence for the WM-guided claim; SUMO admission is the only label; '
                              'no gate, no retuning on these outcomes')
    return out


def auc(pos, neg):
    if not pos or not neg:
        return None
    r = ranks(list(pos) + list(neg))
    return round(float((r[:len(pos)].sum() + len(pos) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 4)


def read_scores(path):
    out = {}
    for line in G.check_path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r['sid'], r['variant'])] = r['wm']
    return out


def rank_check(summaries, scores_path):
    """Zero-SUMO WM ranking check on variants SUMO already verified (WM scores from proposer.score_moments):
    Spearman of WM-predicted vs SUMO-measured T (all screened) and S (solos run), AUC of the WM M-LCB for
    admitted vs not admitted, and per moment whether the WM's top M-LCB variant was an admitted one."""
    by, sources = load_verdicts(summaries)
    sc = read_scores(scores_path)
    rows = [(arm, sid, vd, sc.get((sid, vd['variant']))) for arm, sm in by.items() for sid, vds in sm.items()
            for vd in vds]
    have = [x for x in rows if x[3] is not None]

    def block(xs):
        t = [(w['T']['mean'], vd['T_r0']) for _, _, vd, w in xs if vd.get('T_r0') is not None]
        s = [(w['S']['mean'], vd['S_r0']) for _, _, vd, w in xs if vd.get('S_r0') is not None and w.get('S')]
        pos = [w['M']['lcb'] for _, _, vd, w in xs if vd['admitted'] and w.get('M')]
        neg = [w['M']['lcb'] for _, _, vd, w in xs if not vd['admitted'] and w.get('M')]
        return dict(n=len(xs), spearman_T=spearman([a for a, _ in t], [b for _, b in t]), n_T=len(t),
                    spearman_S=spearman([a for a, _ in s], [b for _, b in s]), n_S=len(s),
                    auc_admitted_by_M_lcb=auc(pos, neg), n_admitted=len(pos))
    per_moment = defaultdict(list)
    for x in have:
        per_moment[x[1]].append(x)
    hit = n_mom = 0
    for sid, xs in per_moment.items():
        if any(vd['admitted'] for _, _, vd, _ in xs):
            n_mom += 1
            top = max(xs, key=lambda x: (x[3]['M']['lcb'], x[3]['M']['mean']))
            hit += bool(top[2]['admitted'])
    return dict(schema='wmc-rank-check-v1', summaries=sources, scores=str(scores_path), n_verdicts=len(rows),
                n_scored=len(have), all=block(have),
                per_arm={a: block([x for x in have if x[0] == a]) for a in sorted({x[0] for x in have})},
                top1_admitted_given_any=dict(k=hit, n=n_mom),
                interpretation='zero-SUMO check of the WM ranking on existing SUMO labels; diagnostic only')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m tools.wzwm compare')
    ap.add_argument('--summary', action='append', required=True)
    ap.add_argument('--proposals')
    ap.add_argument('--cap', type=float)
    ap.add_argument('--salt', default='wmc-compare-v1')
    ap.add_argument('--arms', help='comma list (default: every arm found)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    res = compare(a.summary, a.proposals, a.cap, a.salt, arms=a.arms.split(',') if a.arms else None)
    out = G.check_path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with Path(out).open('x') as fh:
        fh.write(json.dumps(res, indent=1, default=float))
    for arm, m in res['metrics'].items():
        print(f"{arm:10s} moments {m['n_moments']:3d} cand {m['n_candidates']:4d} verified {m['admitted_variants']:3d} "
              f"sim-s {m['sim_s']:9.0f} per-ksim {m['verified_per_ksim']:.4f} {m['verified_per_ksim_ci95']} "
              f"adm/cand {m['admission_rate_candidate']:.3f}")
    return res


if __name__ == '__main__':
    main()
