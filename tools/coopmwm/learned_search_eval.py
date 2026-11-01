"""Offline 'evaluations saved' estimate for the learned search (CPU, no WM, no SUMO).

Question: how many WM candidate evaluations does each search mode need to reach
the plain search's final elite quality?

Modes (all run the real ``mppi_search.search`` code, same seed and budget):
  plain  - registered policy-centred search (no proposer)
  seeds  - + ``plan_proposer.proposal_seeds`` as ``extra_seeds``
  warm   - proposer ``warm_start`` replaces the initial distribution
  both   - warm start + seeds

There is no WM offline, so each held-out search is replayed on a SURROGATE of
its own WM screen scores. The surrogate is kernel ridge regression (RBF,
median-distance bandwidth) fitted to that search's FULL screened pool, recorded
by ``plan_proposer.RecordingEvaluator``: perturbation -> predicted saving vs
'policy continues'. The surrogate is noise-free and only trustworthy near the
recorded pool. Its leave-one-out R^2 is reported per search. Replays work in
perturbation space (zero policy trajectory, no clipping, no MAPPO seed; the same
for every mode).

Target = plain's best surrogate value over its whole screen budget. For each mode,
evals-to-target is the first screened candidate (in evaluation order) whose
surrogate value reaches the target. Modes that never reach it are censored at
budget+1. Final K8 re-scoring is identical for all modes and is not counted.
World-cluster bootstrap 95% CIs are computed on mode - plain.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import mppi_search as ms
from . import plan_proposer as pp

MODES = ('plain', 'seeds', 'warm', 'both')
REGISTERED = dict(n_seeds=4, ridge=1e-2, min_pool=16, bootstraps=2000, seed=20261102,
                  curve_fracs=(.1, .25, .5, .75, 1.))
WIDE = (-1e6, 1e6)       # perturbation-space replay: no plant clipping


class Surrogate:
    """RBF kernel ridge regression of WM screen saving on (perturbation, duration)."""

    def __init__(self, Z, D, y, ridge=REGISTERED['ridge']):
        X = self._feat(Z, D); y = np.asarray(y, float)
        self.X, self.mu, self.sd = X, float(y.mean()), float(y.std()) or 1.
        d2 = ((X[:, None]-X[None])**2).sum(-1)
        off = d2[np.triu_indices(len(X), 1)]
        self.h2 = float(np.median(off)) if off.size and np.median(off) > 0 else 1.
        K = np.exp(-d2/(2*self.h2)); A = K+ridge*np.eye(len(X))
        self.alpha = np.linalg.solve(A, (y-self.mu)/self.sd)
        H = K@np.linalg.inv(A); r = ((y-self.mu)/self.sd-H@((y-self.mu)/self.sd))/np.clip(1-np.diag(H), 1e-9, None)
        yt = (y-self.mu)/self.sd
        self.loo_r2 = float(1-(r**2).sum()/max(((yt-yt.mean())**2).sum(), 1e-12))

    @staticmethod
    def _feat(Z, D):
        Z = np.asarray(Z, float).reshape(len(D), -1)/pp.Z_SCALE
        return np.concatenate([Z, np.asarray(D, float)[:, None]/100.], 1)

    def __call__(self, Z, D):
        X = self._feat(Z, D)
        k = np.exp(-((X[:, None]-self.X[None])**2).sum(-1)/(2*self.h2))
        return self.mu+self.sd*(k@self.alpha)


def surrogate_for(receipt, cfg=REGISTERED):
    pool = [c for c in receipt.get('candidates') or [] if c.get('stage', 'screen') == 'screen'
            and c.get('saving_vs_policy') is not None and c.get('perturbation_canonical') is not None]
    m = len(receipt['members']); n = ms.canonical_knots()
    pool = [c for c in pool if np.shape(c['perturbation_canonical']) == (m, n)
            and np.isfinite(c['saving_vs_policy']) and np.isfinite(c['perturbation_canonical']).all()]
    if len(pool) < cfg['min_pool']:
        return None
    return Surrogate([c['perturbation_canonical'] for c in pool], [c['duration_s'] for c in pool],
                     [c['saving_vs_policy'] for c in pool], cfg['ridge'])


def _group(receipt):
    by = {r['vid']: r for r in receipt['rows']}
    rows = [dict(vid=v, s_m=float(by[v]['s_m']), v_mps=float(by[v]['v_mps']), lane=int(by[v]['lane']),
                 is_cav=True) for v in receipt['members']]
    return dict(id='G'+ms._digest(receipt['members'])[:12], members=list(receipt['members']), rows=rows,
                lanes=sorted({r['lane'] for r in rows}), spread_m=rows[0]['s_m']-rows[-1]['s_m'],
                leader_s_m=rows[0]['s_m'])


def replay(receipt, surrogate, mode, model=None, cfg=REGISTERED, config=None):
    """Run the real search for one mode on the surrogate; screened values in evaluation order."""
    group = _group(receipt); m = len(group['members'])
    policy = np.zeros((m, ms.REGISTERED['max_duration_s']))
    regime = receipt.get('regime') or 'unknown'
    record = dict(state_id=str(receipt.get('search_id')), rows=receipt['rows'],
                  policy_residuals={v: 0. for v in group['members']})
    calls = []

    def evaluate(rec, actions, K, H):
        out, vals = {}, []
        cands = [a for a in actions if 'perturbation_canonical' in a]
        y = surrogate([a['perturbation_canonical'] for a in cands], [a['duration_s'] for a in cands]) if cands else []
        val = {a['id']: float(v) for a, v in zip(cands, y)}
        for a in actions:
            c = -val.get(a['id'], 0.)
            out[a['id']] = [c]*K
            if a['id'] in val:
                vals.append(val[a['id']])
        calls.append(vals)
        return out

    kw = {}
    if mode in ('seeds', 'both'):
        kw['extra_seeds'] = pp.proposal_seeds(record, group, policy, cfg['n_seeds'], model, WIDE, regime=regime)
    if mode in ('warm', 'both'):
        cov0 = (receipt.get('initial') or {}).get('covariance')
        kw['warm_start'] = pp.warm_start(record, group, model, regime, cov0=cov0)
    seed = int(ms._digest([cfg['seed'], receipt.get('search_id')])[:15], 16)
    ms.search(record, group, policy, WIDE, evaluate, regime, seed=seed, config=config, **kw)
    return [v for c in calls[:-1] for v in c]          # screen calls only (last call = final re-score)


def _first_reach(values, target):
    best = np.maximum.accumulate(np.asarray(values, float))
    hit = np.nonzero(best >= target-1e-9*max(1., abs(target)))[0]
    return int(hit[0])+1 if hit.size else len(values)+1, best


def _ci(vals, worlds, B, seed):
    return pp._bootstrap_ci(vals, worlds, B, seed)


def evaluations_curve(model, receipts, modes=MODES, cfg=None, config=None):
    """Quality-vs-evaluations comparison on held-out FULL-pool receipts."""
    cfg = dict(REGISTERED, **(cfg or {}))
    rows, skipped = [], []
    for r in receipts:
        sur = surrogate_for(r, cfg)
        if sur is None:
            skipped.append((r.get('search_id'), 'no full screened pool')); continue
        plain = replay(r, sur, 'plain', model, cfg, config)
        target = max(plain)
        if not target > 0:
            skipped.append((r.get('search_id'), 'plain search found no positive surrogate saving')); continue
        row = dict(id=r.get('search_id'), world=str(r.get('world_id')), m=len(r['members']), budget=len(plain),
                   target=target, surrogate_loo_r2=sur.loo_r2, modes={})
        for mode in modes:
            vals = plain if mode == 'plain' else replay(r, sur, mode, model, cfg, config)
            n, best = _first_reach(vals, target)
            pts = {str(f): float(best[max(0, int(math.ceil(f*len(best)))-1)]/target) for f in cfg['curve_fracs']}
            row['modes'][mode] = dict(evals_to_target=n, reached=n <= len(vals), quality_at_frac=pts)
        rows.append(row)
    if not rows:
        return dict(searches=0, skipped=skipped)
    worlds = [x['world'] for x in rows]
    summary = {}
    for mode in modes:
        ev = [x['modes'][mode]['evals_to_target'] for x in rows]
        rel = [x['modes'][mode]['evals_to_target']/x['budget'] for x in rows]
        d = [x['modes'][mode]['evals_to_target']-x['modes']['plain']['evals_to_target'] for x in rows]
        summary[mode] = dict(
            mean_evals_to_target=float(np.mean(ev)), median_evals_to_target=float(np.median(ev)),
            mean_fraction_of_budget=float(np.mean(rel)),
            reached_rate=float(np.mean([x['modes'][mode]['reached'] for x in rows])),
            minus_plain=dict(zip(('mean', 'ci_lo', 'ci_hi'), _ci(d, worlds, cfg['bootstraps'], cfg['seed']))),
            quality_curve={str(f): float(np.mean([x['modes'][mode]['quality_at_frac'][str(f)] for x in rows]))
                           for f in cfg['curve_fracs']})
    return dict(searches=len(rows), worlds=len(set(worlds)), mean_budget=float(np.mean([x['budget'] for x in rows])),
                median_surrogate_loo_r2=float(np.median([x['surrogate_loo_r2'] for x in rows])),
                summary=summary, per_search=rows, skipped=skipped, cfg=cfg,
                note='surrogate replay (noise-free KRR on each search\'s own WM screen pool); censored at budget+1')


def main(argv=None):
    ap = argparse.ArgumentParser(description='evaluations-to-quality curve on full-pool receipts')
    ap.add_argument('checkpoint'); ap.add_argument('receipts', nargs='+'); ap.add_argument('--out')
    a = ap.parse_args(argv)
    model, meta = pp.load_checkpoint(a.checkpoint)
    rec, _ = pp.load_receipts(a.receipts)
    val = set(meta.get('validation_worlds') or [])
    held = [r for r in rec if str(r.get('world_id')) in val] if val else rec
    res = evaluations_curve(model, held)
    text = json.dumps({k: v for k, v in res.items() if k != 'per_search'}, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
