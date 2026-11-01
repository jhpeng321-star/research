"""Offline 'evaluations saved' estimate for the learned search (CPU, no WM, no SUMO).

Question: how many WM candidate evaluations does each search mode need to reach
the plain search's final elite quality?

Modes (all run the real ``mppi_search.search`` code, same seed and budget):
  plain  - registered policy-centred search (no proposer)
  seeds  - + ``plan_proposer.proposal_seeds`` as ``extra_seeds``
  warm   - proposer ``warm_start`` replaces the initial distribution
  both   - warm start + seeds
  warm_it1 / warm_it2 - warm start with only 1 / 2 CEM iterations (fewer candidates),
           SAME look-ahead: screen K4/H300 and final K8/H_final are unchanged
           (owner 2026-11-02: never shorten horizons; savings only from fewer
           candidates/iterations). The 'plain' reference is the registered 3-iteration search.
Every warm mode keeps the registered generic policy-centred share
(``mppi_search.WARM_GENERIC_SHARE`` = 0.2 of each iteration's sampled population).

There is no WM offline, so each held-out search is replayed on a SURROGATE of
its own WM screen scores. The surrogate is kernel ridge regression (RBF,
median-distance bandwidth) fitted to that search's FULL screened pool, recorded
by ``plan_proposer.RecordingEvaluator``: perturbation -> predicted saving vs
'policy continues'. The surrogate is noise-free and only trustworthy near the
recorded pool. Its leave-one-out R^2 is reported per search. Replays work in
perturbation space (zero policy trajectory, no clipping, no MAPPO seed; the same
for every mode).

PRIMARY ('elite') quality is the mean of the best E candidates found so far, with
E = ceil(N/4), the search's own elite count. Its target is the plain search's final
elite mean, and it is normalised so that 0 = the 'centre' seed (unchanged policy)
and 1 = the plain final elite mean. A secondary, strict metric uses the plain
search's single best value (max) as the target. For each mode,
evals-to-target is the first screened candidate (in evaluation order) whose
surrogate value reaches the target. Modes that never reach it are censored at
budget+1. Final K8 re-scoring is identical for all modes and is not counted.
World-cluster bootstrap 95% CIs are computed on mode - plain. ELITE COVERAGE asks
whether a mode still finds what the plain search found. It is the fraction of the
receipt's recorded real-WM pool elites that the mode's candidates approach at
least as closely (RMS) as the plain replay's candidates do.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import mppi_search as ms
from . import plan_proposer as pp

MODES = ('plain', 'seeds', 'warm', 'both', 'warm_it1', 'warm_it2')
ITERATIONS = dict(warm_it1=1, warm_it2=2)
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
    """Run the real search for one mode on the surrogate.

    Returns (values, perturbations m x n in m/s^2) of the SCREENED candidates in
    evaluation order (the last call, the final K8 re-score, is excluded).
    """
    group = _group(receipt); m = len(group['members'])
    policy = np.zeros((m, ms.REGISTERED['max_duration_s']))
    regime = receipt.get('regime') or 'unknown'
    record = dict(state_id=str(receipt.get('search_id')), rows=receipt['rows'],
                  policy_residuals={v: 0. for v in group['members']})
    calls = []

    def evaluate(rec, actions, K, H):
        out, vals, zs = {}, [], []
        cands = [a for a in actions if 'perturbation_canonical' in a]
        y = surrogate([a['perturbation_canonical'] for a in cands], [a['duration_s'] for a in cands]) if cands else []
        val = {a['id']: float(v) for a, v in zip(cands, y)}
        for a in actions:
            c = -val.get(a['id'], 0.)
            out[a['id']] = [c]*K
            if a['id'] in val:
                vals.append(val[a['id']]); zs.append(np.asarray(a['perturbation_canonical'], float))
        calls.append((vals, zs))
        return out

    kw = {}
    if mode in ITERATIONS:          # only the iteration count changes; K/H rules untouched
        config = dict(config or {}, iterations=ITERATIONS[mode])
    if mode in ('seeds', 'both'):
        kw['extra_seeds'] = pp.proposal_seeds(record, group, policy, cfg['n_seeds'], model, WIDE, regime=regime)
    if mode in ('warm', 'both') or mode in ITERATIONS:
        cov0 = (receipt.get('initial') or {}).get('covariance')
        kw['warm_start'] = pp.warm_start(record, group, model, regime, cov0=cov0)
    seed = int(ms._digest([cfg['seed'], receipt.get('search_id')])[:15], 16)
    ms.search(record, group, policy, WIDE, evaluate, regime, seed=seed, config=config, **kw)
    return [v for c in calls[:-1] for v in c[0]], [z for c in calls[:-1] for z in c[1]]


def _first_reach(values, target, budget):
    """First evaluation reaching target (censored at the PLAIN budget+1); best-so-far
    extended to the plain budget (a mode with fewer candidates stops evaluating)."""
    best = np.maximum.accumulate(np.asarray(values, float))
    hit = np.nonzero(best >= target-1e-9*max(1., abs(target)))[0]
    ext = np.concatenate([best, np.full(max(0, budget-len(best)), best[-1])])
    return int(hit[0])+1 if hit.size else budget+1, ext


def _running_elite_mean(values, E, budget):
    """Mean of the best E values found so far (NaN before E evaluations), extended to budget."""
    import bisect
    top, out = [], []
    for v in values:
        bisect.insort(top, -float(v)); top = top[:E]
        out.append(-float(np.mean(top)) if len(top) >= E else np.nan)
    out += [out[-1]]*max(0, budget-len(out))
    return np.asarray(out)


def _mean_or_none(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


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
        plain, plain_z = replay(r, sur, 'plain', model, cfg, config)
        # quality is normalised between the 'centre' seed (unchanged policy, first
        # plain candidate) and the plain search's final best (= target, 1.0)
        base, target = plain[0], max(plain)
        if not target > base:
            skipped.append((r.get('search_id'), 'plain search found nothing better than the centre')); continue
        s, _ = pp.build_search(r)
        elites = [e['z']*pp.Z_SCALE for e in (s['elites'] if s else [])]     # recorded (real WM) pool elites

        def nearest(Z):
            Z = np.stack(Z)
            return [float(np.sqrt(((Z-e)**2).mean((1, 2))).min()) for e in elites]
        d_plain = nearest(plain_z) if elites else []
        # ELITE quality (primary): mean of the best E (= the search's per-iteration elite
        # count ceil(N/4)) values found so far; target = plain's final elite mean.
        E = ms.scaling(len(r['members']), ms.canonical_knots())['elites']
        plain_e = _running_elite_mean(plain, E, len(plain)); target_e = float(plain_e[-1])
        row = dict(id=r.get('search_id'), world=str(r.get('world_id')), m=len(r['members']), budget=len(plain),
                   target=target, target_elite=target_e, elites_E=E, centre=base, surrogate_loo_r2=sur.loo_r2,
                   recorded_elites=len(elites), modes={})
        for mode in modes:
            vals, zs = (plain, plain_z) if mode == 'plain' else replay(r, sur, mode, model, cfg, config)
            n, best = _first_reach(vals, target, len(plain))
            q = (best-base)/(target-base)
            pts = {str(f): float(q[max(0, int(math.ceil(f*len(plain)))-1)]) for f in cfg['curve_fracs']}
            d = nearest(zs) if elites else []
            em = _running_elite_mean(vals, E, len(plain))
            hit = np.nonzero(em[:len(vals)] >= target_e-1e-9*max(1., abs(target_e)))[0]
            ne = int(hit[0])+1 if hit.size else len(plain)+1
            eq = (em-base)/(target_e-base) if target_e > base else em*np.nan
            row['modes'][mode] = dict(evals_to_elite=ne, elite_reached=ne <= len(vals),
                                      final_elite_quality=float(eq[len(vals)-1]),
                                      elite_quality_at_frac={str(f): float(eq[max(0, int(math.ceil(f*len(plain)))-1)])
                                                             for f in cfg['curve_fracs']},
                                      evals_to_target=n, reached=n <= len(vals), quality_at_frac=pts,
                                      evaluations=len(vals), final_quality=float(q[len(vals)-1]),
                                      # elite coverage: comes at least as close to each recorded elite as plain did
                                      elite_coverage=float(np.mean([a <= b+1e-12 for a, b in zip(d, d_plain)])) if d else None,
                                      elite_distance=float(np.mean(d)) if d else None)
        rows.append(row)
    if not rows:
        return dict(searches=0, skipped=skipped)
    worlds = [x['world'] for x in rows]
    summary = {}
    for mode in modes:
        ev = [x['modes'][mode]['evals_to_target'] for x in rows]
        rel = [x['modes'][mode]['evals_to_target']/x['budget'] for x in rows]
        d = [x['modes'][mode]['evals_to_target']-x['modes']['plain']['evals_to_target'] for x in rows]
        de = [x['modes'][mode]['evals_to_elite']-x['modes']['plain']['evals_to_elite'] for x in rows]
        summary[mode] = dict(
            elite=dict(mean_evals_to_plain_elite_quality=float(np.mean([x['modes'][mode]['evals_to_elite'] for x in rows])),
                       mean_fraction_of_plain_budget=float(np.mean([x['modes'][mode]['evals_to_elite']/x['budget'] for x in rows])),
                       reached_rate=float(np.mean([x['modes'][mode]['elite_reached'] for x in rows])),
                       minus_plain=dict(zip(('mean', 'ci_lo', 'ci_hi'), _ci(de, worlds, cfg['bootstraps'], cfg['seed']+2))),
                       final_quality=dict(zip(('mean', 'ci_lo', 'ci_hi'), _ci(
                           [x['modes'][mode]['final_elite_quality'] for x in rows], worlds, cfg['bootstraps'], cfg['seed']+3))),
                       quality_curve={str(f): float(np.nanmean([x['modes'][mode]['elite_quality_at_frac'][str(f)]
                                                                for x in rows])) for f in cfg['curve_fracs']}),
            mean_evals_to_target=float(np.mean(ev)), median_evals_to_target=float(np.median(ev)),
            mean_fraction_of_budget=float(np.mean(rel)),
            reached_rate=float(np.mean([x['modes'][mode]['reached'] for x in rows])),
            evaluations_fraction=float(np.mean([x['modes'][mode]['evaluations']/x['budget'] for x in rows])),
            elite_coverage=_mean_or_none([x['modes'][mode]['elite_coverage'] for x in rows]),
            elite_distance=_mean_or_none([x['modes'][mode]['elite_distance'] for x in rows]),
            final_quality=dict(zip(('mean', 'ci_lo', 'ci_hi'), _ci(
                [x['modes'][mode]['final_quality'] for x in rows], worlds, cfg['bootstraps'], cfg['seed']+1))),
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
