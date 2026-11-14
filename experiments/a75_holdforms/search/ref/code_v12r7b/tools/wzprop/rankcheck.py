"""Zero-SUMO WM ranking check against existing SUMO verdicts, with world-cluster bootstrap CIs.

Inputs: funnel summaries (PG/NM2/LC/WMC; verdict rows with T_r0, S_r0, admitted, stage) and WM scores
(proposer.score_moments: per (sid, variant) the WM's T, S, M = min(base, best solo) - group per draw).
Sign convention (veh-s): T = J(base) - J(group), S = J(best solo) - J(group); positive = group better.

Metrics (all with a 2,000-draw bootstrap that resamples WORLD seeds with all their moments, RNG 20261106):
  spearman_T            pooled Spearman, WM mean T vs SUMO r0 T, every screened candidate
  spearman_T_within     mean within-moment Spearman (moments with >= 3 candidates)
  spearman_S            pooled Spearman, WM mean S vs SUMO r0 S (candidates whose solos SUMO ran)
  precision_T_pos       P(SUMO T > 0 | WM mean T > 0)  (merge-map reference 0.41); strict: SUMO LB95(T) > 0
  precision_at_k_T      pooled top-k by WM mean T: share with SUMO T > 0 (k = 10, 20, 40)
  precision_at_k_S      top-k by WM mean S among S-labelled: share with SUMO S > 0 (k = 10 if >= 30 else ceil(n/3))
  top1_T_pos            per moment, the WM's top-T candidate has SUMO T > 0
  admitted_hit_at_k     per moment with >= 1 admitted candidate (complete funnels only): an admitted candidate
                        is in the WM top-k by the proposer score (M-LCB) and by mean T (k = 1, 2, 3)
Every metric is reported with its random-ranking expectation (base rate) for reference.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import compare as C
from . import geometry as G

B_BOOT, RNG = 2000, 20261106
KS_T = (10, 20, 40)
KS_HIT = (1, 2, 3)


def join(summaries, scores_path):
    by, sources = C.load_verdicts(summaries)
    sc = C.read_scores(scores_path)
    rows, missing = [], 0
    for arm, sm in by.items():
        for sid, vds in sm.items():
            stranded = any(v['stage'].endswith('_missing') for v in vds)
            for vd in vds:
                w = sc.get((sid, vd['variant']))
                if w is None:
                    missing += 1
                    continue
                rows.append(dict(arm=arm, sid=sid, seed=int(vd['seed']), variant=vd['variant'], family=vd['family'],
                                 pT=w['T']['mean'], pS=(w.get('S') or {}).get('mean'),
                                 pM=(w.get('M') or {}).get('mean'), pM_lcb=(w.get('M') or {}).get('lcb'),
                                 T=vd.get('T_r0'), T_lb=vd.get('T_lb_r0'), S=vd.get('S_r0'), S_lb=vd.get('S_lb_r0'),
                                 admitted=bool(vd['admitted']), stage=vd['stage'], stranded=stranded))
    return rows, sources, missing


def _sp(a, b):
    return C.spearman(list(a), list(b))


def m_spearman_T(rows):
    r = [x for x in rows if x['T'] is not None]
    return _sp([x['pT'] for x in r], [x['T'] for x in r]), len(r)


def m_spearman_S(rows):
    r = [x for x in rows if x['S'] is not None and x['pS'] is not None]
    return _sp([x['pS'] for x in r], [x['S'] for x in r]), len(r)


def by_moment(rows):
    out = defaultdict(list)
    for x in rows:
        out[x['sid']].append(x)
    return out


def m_spearman_T_within(rows):
    vals = []
    for xs in by_moment([x for x in rows if x['T'] is not None]).values():
        if len(xs) >= 3:
            s = _sp([x['pT'] for x in xs], [x['T'] for x in xs])
            if s is not None:
                vals.append(s)
    return (float(np.mean(vals)) if vals else None), len(vals)


def m_precision_T(rows, strict=False):
    r = [x for x in rows if x['T'] is not None and x['pT'] > 0]
    if not r:
        return None, 0
    ok = [(x['T_lb'] or -1) > 0 if strict else x['T'] > 0 for x in r]
    return float(np.mean(ok)), len(r)


def base_rate_T(rows, strict=False):
    r = [x for x in rows if x['T'] is not None]
    if not r:
        return None, 0
    return float(np.mean([(x['T_lb'] or -1) > 0 if strict else x['T'] > 0 for x in r])), len(r)


def m_precision_at_k_T(rows, k):
    r = sorted([x for x in rows if x['T'] is not None], key=lambda x: -x['pT'])
    if len(r) < k:
        return None, len(r)
    return float(np.mean([x['T'] > 0 for x in r[:k]])), k


def k_for_S(n):
    return 10 if n >= 30 else max(1, math.ceil(n / 3))


def m_precision_at_k_S(rows, k=None):
    r = sorted([x for x in rows if x['S'] is not None and x['pS'] is not None], key=lambda x: -x['pS'])
    if not r:
        return None, 0
    k = k or k_for_S(len(r))
    k = min(k, len(r))
    return float(np.mean([x['S'] > 0 for x in r[:k]])), k


def base_rate_S(rows):
    r = [x for x in rows if x['S'] is not None]
    return (float(np.mean([x['S'] > 0 for x in r])) if r else None), len(r)


def m_top1_T(rows):
    vals = []
    for xs in by_moment([x for x in rows if x['T'] is not None]).values():
        top = max(xs, key=lambda x: (x['pT'], x['variant']))
        vals.append(top['T'] > 0)
    return (float(np.mean(vals)) if vals else None), len(vals)


def _score(x, key):
    if key == 'M_lcb':
        return (x['pM_lcb'] if x['pM_lcb'] is not None else -math.inf, x['pT'])
    return (x['pT'], x['pM_lcb'] if x['pM_lcb'] is not None else -math.inf)


def m_admitted_hit(rows, k, key='M_lcb'):
    hits = []
    for xs in by_moment([x for x in rows if not x['stranded']]).values():
        if not any(x['admitted'] for x in xs):
            continue
        if key == 'M_lcb' and any(x['pM_lcb'] is None for x in xs):
            continue                                   # M needs every candidate's solos (score --solos all)
        top = sorted(xs, key=lambda x: _score(x, key), reverse=True)[:k]
        hits.append(any(x['admitted'] for x in top))
    return (float(np.mean(hits)) if hits else None), len(hits)


def random_admitted_hit(rows, k):
    vals = []
    for xs in by_moment([x for x in rows if not x['stranded']]).values():
        a, n = sum(x['admitted'] for x in xs), len(xs)
        if a == 0:
            continue
        kk = min(k, n)
        vals.append(1.0 - math.comb(n - a, kk) / math.comb(n, kk))
    return (float(np.mean(vals)) if vals else None), len(vals)


def bootstrap(rows, fn, b=B_BOOT, seed=RNG):
    """World-cluster bootstrap 95% CI of fn(rows)[0] (seeds resampled with all their rows)."""
    seeds = sorted({x['seed'] for x in rows})
    if len(seeds) < 2:
        return None
    by = defaultdict(list)
    for x in rows:
        by[x['seed']].append(x)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        pick = rng.choice(len(seeds), len(seeds), replace=True)
        rr = []
        for j, i in enumerate(pick):        # relabel resampled worlds/moments so duplicates stay distinct
            for x in by[seeds[i]]:
                rr.append(dict(x, sid=f"{x['sid']}#{j}", seed=j))
        v = fn(rr)[0]
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if len(vals) < b // 2:
        return None
    return [round(float(np.quantile(vals, .025)), 4), round(float(np.quantile(vals, .975)), 4)]


def metric(rows, fn, ci=True):
    v, n = fn(rows)
    return dict(value=None if v is None else round(float(v), 4), n=n, ci95=bootstrap(rows, fn) if ci and v is not None else None)


def run(summaries, scores_path, *, label='WM', checkpoint=None):
    rows, sources, missing = join(summaries, scores_path)
    if not rows:
        raise ValueError('no scored verdicts')
    for x in rows:
        G.check_seed(x['seed'])
    out = dict(schema='wmc-rankcheck-v2', label=label, checkpoint=checkpoint, summaries=sources,
               scores=str(scores_path), n_rows=len(rows), n_unscored=missing,
               n_moments=len({x['sid'] for x in rows}), worlds=sorted({x['seed'] for x in rows}),
               n_admitted=sum(x['admitted'] for x in rows))
    out['spearman_T'] = metric(rows, m_spearman_T)
    out['spearman_T_within'] = metric(rows, m_spearman_T_within)
    out['spearman_S'] = metric(rows, m_spearman_S)
    out['precision_T_pos'] = metric(rows, m_precision_T)
    out['precision_T_pos_strict'] = metric(rows, lambda r: m_precision_T(r, strict=True))
    out['base_rate_T_pos'] = metric(rows, base_rate_T, ci=False)
    out['base_rate_T_pos_strict'] = metric(rows, lambda r: base_rate_T(r, strict=True), ci=False)
    out['precision_at_k_T'] = {str(k): metric(rows, lambda r, k=k: m_precision_at_k_T(r, k)) for k in KS_T}
    out['precision_at_k_S'] = metric(rows, m_precision_at_k_S)
    out['base_rate_S_pos'] = metric(rows, base_rate_S, ci=False)
    out['top1_T_pos'] = metric(rows, m_top1_T)
    out['admitted_hit_at_k'] = {
        key: {str(k): metric(rows, lambda r, k=k, key=key: m_admitted_hit(r, k, key)) for k in KS_HIT}
        for key in ('M_lcb', 'T_mean')}
    out['admitted_hit_random'] = {str(k): metric(rows, lambda r, k=k: random_admitted_hit(r, k), ci=False)
                                  for k in KS_HIT}
    out['per_arm'] = {a: dict(spearman_T=metric([x for x in rows if x['arm'] == a], m_spearman_T),
                              precision_T_pos=metric([x for x in rows if x['arm'] == a], m_precision_T),
                              base_rate_T_pos=metric([x for x in rows if x['arm'] == a], base_rate_T, ci=False))
                      for a in sorted({x['arm'] for x in rows})}
    out['interpretation'] = ('zero-SUMO diagnostic of the WM ranking on existing SUMO labels; no gate (the registered '
                             'gate is WM-D WM_QUAL_GATE); SUMO r0 labels are themselves noisy (WM-D: r0 vs r1 '
                             'T Spearman 0.68, S Spearman 0.37)')
    return out


def _fmt(m):
    if m is None or m.get('value') is None:
        return 'n/a'
    ci = m.get('ci95')
    return f"{m['value']:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else '') + f" (n={m['n']})"


def markdown(res):
    L = [f"| metric | {res['label']} | random-ranking reference |", '|---|---|---|']
    L.append(f"| Spearman, WM T vs SUMO T (pooled) | {_fmt(res['spearman_T'])} | 0 |")
    L.append(f"| Spearman within moment (mean) | {_fmt(res['spearman_T_within'])} | 0 |")
    L.append(f"| Spearman, WM S vs SUMO S | {_fmt(res['spearman_S'])} | 0 |")
    L.append(f"| precision of WM-positive T: P(SUMO T > 0 / WM T > 0) | {_fmt(res['precision_T_pos'])} | "
             f"{_fmt(res['base_rate_T_pos'])} |")
    L.append(f"| same, strict (SUMO LB95(T) > 0) | {_fmt(res['precision_T_pos_strict'])} | "
             f"{_fmt(res['base_rate_T_pos_strict'])} |")
    for k, m in res['precision_at_k_T'].items():
        L.append(f"| precision@{k} by WM T: SUMO T > 0 | {_fmt(m)} | {_fmt(res['base_rate_T_pos'])} |")
    L.append(f"| synergy precision@k by WM S: SUMO S > 0 | {_fmt(res['precision_at_k_S'])} | "
             f"{_fmt(res['base_rate_S_pos'])} |")
    L.append(f"| WM top-1 per moment has SUMO T > 0 | {_fmt(res['top1_T_pos'])} | {_fmt(res['base_rate_T_pos'])} |")
    for key, name in (('M_lcb', 'proposer score M-LCB'), ('T_mean', 'WM mean T')):
        for k, m in res['admitted_hit_at_k'][key].items():
            L.append(f"| admitted move in WM top-{k} ({name}) | {_fmt(m)} | {_fmt(res['admitted_hit_random'][k])} |")
    for a, d in res['per_arm'].items():
        L.append(f"| [{a}] Spearman T / precision T>0 | {_fmt(d['spearman_T'])} / {_fmt(d['precision_T_pos'])} | "
                 f"base {_fmt(d['base_rate_T_pos'])} |")
    return '\n'.join(L)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='python -m tools.wzprop rankmetrics')
    ap.add_argument('--summary', action='append', required=True)
    ap.add_argument('--scores', required=True)
    ap.add_argument('--label', default='WM')
    ap.add_argument('--checkpoint')
    ap.add_argument('--out', required=True, help='new JSON; a .md table is written next to it')
    a = ap.parse_args(argv)
    res = run(a.summary, a.scores, label=a.label, checkpoint=a.checkpoint)
    out = G.check_path(a.out)
    with Path(out).open('x') as fh:
        fh.write(json.dumps(res, indent=1))
    with Path(out).with_suffix('.md').open('x') as fh:
        fh.write(markdown(res) + '\n')
    print(markdown(res))
    return res
