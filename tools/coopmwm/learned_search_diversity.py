"""Diversity / collapse metrics for the learned search (owner 2026-11-02: does it keep trying new moves?).

The metrics compare warm vs plain vs plain_alt on held-out full-pool searches, in
the same surrogate replay as ``learned_search_eval``, plus direct proposer samples.
Distances are RMS over the canonical members x 10 perturbation (m/s^2).

(1) Spread: mean pairwise distance and per-dimension std of 64 proposer samples,
    relative to 64 samples of the plain initial distribution N(0, Sigma_0). The
    same mean pairwise distance is also computed for each mode's screened pool,
    relative to plain.
(2) Distinct members: number of distinct member patterns and clusters among each
    mode's top-8 and its elites (top-E, E = ceil(N/4)).
    - Member pattern: the dominant member (largest mean |perturbation| over the
      plan, leader-first index) and its sign, or 'none' when that mean is below
      0.1 m/s^2 (0.2 sigma).
    - Clusters: connected components with linkage distance <= the median
      plain-pool nearest-neighbour spacing of that search (no hand-set threshold).
(3) New regions: the share of learned-mixture samples (no defensive component)
    farther than that spacing from EVERY training-set elite of the same group size.
    The plain initial distribution is given as the reference.
(4) Repetition: across searches, the share of the most common top-1 member pattern
    and the normalised entropy of top-1 patterns, for each mode.

The live post-hoc family labels of executed B plans are reported separately by the
report (``posthoc_label_counts``). Replays cannot carry them because the family
prototypes need live commands.
"""
import math
from collections import Counter

import numpy as np
import torch

from . import learned_search_eval as le
from . import mppi_search as ms
from . import plan_proposer as pp

MODES = ('plain', 'plain_alt', 'warm')
SMALL_MOVE = .1          # m/s^2 = 0.2 x the registered 0.5 sigma: below this a plan is 'none'
N_SAMPLES = 64


def rms(a, b):
    return float(np.sqrt(((np.asarray(a)-np.asarray(b))**2).mean()))


def mean_pairwise(Z):
    Z = np.stack(Z).reshape(len(Z), -1)
    if len(Z) < 2:
        return float('nan')
    sq = (Z**2).sum(1); d2 = np.clip(sq[:, None]+sq[None]-2*Z@Z.T, 0, None)/Z.shape[1]
    iu = np.triu_indices(len(Z), 1)
    return float(np.sqrt(d2[iu]).mean())


def nn_spacing(Z):
    Z = np.stack(Z).reshape(len(Z), -1)
    sq = (Z**2).sum(1); d2 = np.clip(sq[:, None]+sq[None]-2*Z@Z.T, 0, None)/Z.shape[1]
    np.fill_diagonal(d2, np.inf)
    return float(np.median(np.sqrt(d2.min(1))))


def member_pattern(z, D):
    z = np.asarray(z, float); keep = pp.knot_mask(D)
    means = z[:, keep].mean(1)
    i = int(np.argmax(np.abs(means)))
    return 'none' if abs(means[i]) < SMALL_MOVE else f"m{i}{'-' if means[i] < 0 else '+'}"


def clusters(Z, eps):
    n = len(Z); parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(n):
        for j in range(i+1, n):
            if rms(Z[i], Z[j]) <= eps:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(n)})


def _norm_entropy(labels):
    c = Counter(labels); n = sum(c.values())
    if n == 0 or len(c) < 2:
        return 0.
    h = -sum(v/n*math.log(v/n) for v in c.values())
    return float(h/math.log(len(c)))


@torch.no_grad()
def learned_samples(model, x, n, rng):
    """Samples from the learned mixture only (defensive component excluded)."""
    eps = model.cfg.get('defensive', 0.)
    model.cfg['defensive'] = 0.
    try:
        return pp.sample(model, x, n, rng, modes=False)
    finally:
        model.cfg['defensive'] = eps


def diversity(model, heldout, train_searches, cfg=None, n_samples=N_SAMPLES):
    """Per-search and aggregate diversity/collapse metrics (see module docstring)."""
    cfg = dict(le.REGISTERED, **(cfg or {}))
    elites_by_m = {}
    for s in train_searches:
        for e in s['elites']:
            elites_by_m.setdefault(s['m'], []).append(e['z']*pp.Z_SCALE)
    rows, top1 = [], {m: [] for m in MODES}
    for r in heldout:
        sur = le.surrogate_for(r, cfg)
        s, _ = pp.build_search(r)
        if sur is None or s is None:
            continue
        rng = np.random.default_rng(int(pp._digest([cfg['seed'], 'div', r.get('search_id')])[:12], 16))
        pools = {}
        for mode in MODES:
            Ds = []
            vals, zs = le.replay(r, sur, mode, model, cfg, durations=Ds)
            pools[mode] = (np.asarray(vals), zs, Ds[:len(vals)])
        spacing = nn_spacing(pools['plain'][1])
        E = ms.scaling(s['m'], ms.canonical_knots())['elites']
        row = dict(id=r.get('search_id'), world=str(r.get('world_id')), m=s['m'], spacing=spacing, modes={})
        # (1) direct proposer spread vs the plain initial distribution
        P = [z for z, _ in pp.sample(model, s['x'], n_samples, rng, modes=False)]
        I = [z for z, _ in pp._initial_samples(s, n_samples, rng)]
        Ps, Is = np.stack(P).reshape(n_samples, -1), np.stack(I).reshape(n_samples, -1)
        keep = Is.std(0) > 0
        row['proposer_spread_ratio'] = mean_pairwise(P)/mean_pairwise(I)
        row['proposer_std_ratio'] = float((Ps.std(0)[keep]/Is.std(0)[keep]).mean())
        # (3) new regions vs all training-set elites of the same size
        ref = elites_by_m.get(s['m'], [])
        L = [z for z, _ in learned_samples(model, s['x'], n_samples, rng)]
        if ref:
            R = np.stack(ref)
            far = lambda Z: float(np.mean([np.sqrt(((R-z)**2).mean((1, 2))).min() > spacing for z in Z]))
            row['new_region_share_learned'] = far(L); row['new_region_share_initial'] = far(I)
        plain_pool = mean_pairwise(pools['plain'][1])
        for mode, (vals, zs, Ds) in pools.items():
            order = np.argsort(-vals, kind='stable')
            t8, te = order[:8], order[:E]
            row['modes'][mode] = dict(
                pool_spread_ratio=mean_pairwise(zs)/plain_pool,
                top8_patterns=len({member_pattern(zs[i], Ds[i]) for i in t8}),
                top8_clusters=clusters([zs[i] for i in t8], spacing),
                elite_patterns=len({member_pattern(zs[i], Ds[i]) for i in te}),
                elite_clusters=clusters([zs[i] for i in te], spacing),
                top1_pattern=member_pattern(zs[order[0]], Ds[order[0]]))
            top1[mode].append(row['modes'][mode]['top1_pattern'])
        rows.append(row)
    if not rows:
        return dict(searches=0)
    worlds = [x['world'] for x in rows]
    ci = lambda v, k: dict(zip(('mean', 'ci_lo', 'ci_hi'), pp._bootstrap_ci(v, worlds, cfg['bootstraps'], cfg['seed']+20+k)))
    out = dict(searches=len(rows), worlds=len(set(worlds)),
               proposer_spread_ratio=ci([x['proposer_spread_ratio'] for x in rows], 0),
               proposer_std_ratio=ci([x['proposer_std_ratio'] for x in rows], 1))
    nr = [x for x in rows if 'new_region_share_learned' in x]
    if nr:
        wn = [x['world'] for x in nr]
        out['new_region_share_learned'] = dict(zip(('mean', 'ci_lo', 'ci_hi'), pp._bootstrap_ci(
            [x['new_region_share_learned'] for x in nr], wn, cfg['bootstraps'], cfg['seed']+30)))
        out['new_region_share_initial'] = float(np.mean([x['new_region_share_initial'] for x in nr]))
    out['modes'] = {}
    for k, mode in enumerate(MODES):
        g = lambda key: [x['modes'][mode][key] for x in rows]
        c = Counter(top1[mode])
        out['modes'][mode] = dict(
            pool_spread_ratio=ci(g('pool_spread_ratio'), 40+k),
            top8_patterns=float(np.mean(g('top8_patterns'))), top8_clusters=float(np.mean(g('top8_clusters'))),
            elite_patterns=float(np.mean(g('elite_patterns'))), elite_clusters=float(np.mean(g('elite_clusters'))),
            top1_most_common=dict(c.most_common(3)), top1_most_common_share=c.most_common(1)[0][1]/len(rows),
            top1_distinct=len(c), top1_norm_entropy=_norm_entropy(top1[mode]))
    out['note'] = ('ratios vs plain initial distribution (1) or plain pool; clusters linked at the median plain-pool '
                   'NN spacing; member pattern = dominant member index + sign (|mean| >= 0.1 m/s^2)')
    return out
