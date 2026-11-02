"""TRAIN-only inner cross-validation for learned-screen design (never touches held-out worlds).

The held-out worlds are FIXED for v2 and later (LEARNED_SCREEN_PREREG_V2.md): the 18 of
the 72 TRAIN worlds (986000..986071) with the lowest SHA256 rank of (20261102, world),
which is the same order ``plan_proposer.world_split`` uses. Development uses only the
other 54 worlds, split into 4 world-grouped folds.

Per inner-held-out search, on the REAL recorded WM screen pool:
- spearman: rank correlation with the actual screen saving;
- q16: (best actual among the model's top-16 - centre) / (best actual - centre);
- ndcg16: top-weighted ranking quality;
- prefilter_f: quality kept when the model drops the bottom f of candidates and the
  full WM screen picks among the rest. This is a pool-level proxy for the low-risk
  hybrid; the speed gain on the screen is 1/(1-f).

Each is compared with a random ranking of the same pool.
"""
import json
import math
import sys
import time

import numpy as np

from . import learned_screen as ls
from . import mppi_search as ms
from . import plan_proposer as pp

SEED = 20261102
ALL_TRAIN = [str(w) for w in range(986000, 986072)]
HELD72 = sorted(sorted(ALL_TRAIN, key=lambda w: (pp._digest([SEED, w]), w))[:18])
FOLDS = 4
DUR = ms.allowed_durations()
FRACS = (.25, .5)


def dev_units(units):
    return [u for u in units if u['world'] not in set(HELD72)]


def folds(units):
    ws = sorted({u['world'] for u in units}, key=lambda w: (pp._digest(['inner', SEED, w]), w))
    return [[u for u in units if u['world'] in set(ws[k::FOLDS])] for k in range(FOLDS)]


# --------------------------------------------------------------------------- handcrafted features

def plan_features(u, idx=None):
    """Fixed-slot (leader-first, 6 slots) plan x state features for each candidate of unit u."""
    idx = np.arange(len(u['y'])) if idx is None else idx
    x = u['x']; m = u['m']; out = []
    for i in idx:
        z = u['z'][i]/pp.Z_SCALE; D = int(u['D'][i]); km = pp.knot_mask(D); kk = int(km.sum())
        f = []
        for j in range(6):
            if j < m:
                row = z[j, km]
                third = max(1, kk//3)
                st = x[j]; lane = int(np.argmax(st[6:9])); o = 9+lane*8     # same-lane ahead (4) then behind (4)
                f += [row.mean(), row[:third].mean(), row[-third:].mean(), row.min(), row.max(), np.abs(row).mean(),
                      st[0], st[5], lane/2., st[o], st[o+1], st[o+3], st[o+4], st[o+5]]
            else:
                f += [0.]*14
        f += [float(D == d) for d in DUR]+[D/300., m/6., x[0, -6], x[0, -5], x[0, -4], x[0, -3], x[0, -1],
                                          float(np.abs(z[:, km]).mean()), float(z[:, km].mean())]
        out.append(f)
    return np.asarray(out, np.float32)


# --------------------------------------------------------------------------- metrics

def _ndcg(scores, actual, k=16):
    a = actual-actual.min(); order = np.argsort(-scores, kind='stable')[:k]
    disc = 1/np.log2(np.arange(2, k+2))
    ideal = (np.sort(a)[::-1][:k]*disc[:min(k, len(a))]).sum()
    return float((a[order]*disc[:len(order)]).sum()/ideal) if ideal > 0 else 1.


def pool_metrics(scores, u, rng, subsets=200):
    act = u['saving']; centre = next((a for a, o in zip(act, u['origin']) if o == 'centre'), 0.)
    top = act.max()
    if top <= centre:
        return None
    q = lambda sel: float((act[sel].max()-centre)/(top-centre))
    order = np.argsort(-scores, kind='stable'); n = len(act)
    ra = np.argsort(np.argsort(scores)); rb = np.argsort(np.argsort(act))
    out = dict(spearman=float(np.corrcoef(ra, rb)[0, 1]), q16=q(order[:16]), ndcg16=_ndcg(scores, act))
    rnd = [rng.permutation(n) for _ in range(subsets)]
    out['q16_random'] = float(np.mean([q(p[:16]) for p in rnd]))
    out['ndcg16_random'] = float(np.mean([_ndcg(rng.standard_normal(n), act) for _ in range(20)]))
    for f in FRACS:
        keep = max(1, int(math.ceil((1-f)*n)))
        out[f'prefilter_{f}'] = q(order[:keep])
        out[f'prefilter_{f}_random'] = float(np.mean([q(p[:keep]) for p in rnd]))
    return out


# --------------------------------------------------------------------------- models

def fit_predict(kind, train, test, seed=SEED):
    """Return {unit id: scores} for test units."""
    if kind.startswith('net'):
        cfg = dict(steps=int(kind.split(':')[1]) if ':' in kind else 2000, eval_every=100)
        if 'noes' in kind:
            cfg['inner_min_worlds'] = 10**9
        model, _ = ls.train(train, cfg)
        return {u['id']: ls.predict(model, u['x'], list(u['z']), list(u['D'])) for u in test}
    X = np.concatenate([plan_features(u) for u in train]); y = np.concatenate([u['y'] for u in train])
    if kind == 'ridge':
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.)).fit(X, y)
    elif kind.startswith('hgb'):
        from sklearn.ensemble import HistGradientBoostingRegressor
        # 'hgb_top': top-weighted (candidates in the top quarter of their own search x4)
        w = None
        if kind == 'hgb_top':
            w = np.concatenate([np.where(u['y'] >= np.quantile(u['y'], .75), 4., 1.) for u in train])
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, max_leaf_nodes=31,
                                              min_samples_leaf=40, l2_regularization=1., random_state=seed)
        model.fit(X, y, sample_weight=w)
    elif kind == 'hgb_cls':
        from sklearn.ensemble import HistGradientBoostingClassifier
        yc = np.concatenate([(u['y'] >= np.quantile(u['y'], .75)).astype(int) for u in train])
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, min_samples_leaf=40,
                                               l2_regularization=1., random_state=seed).fit(X, yc)
        return {u['id']: model.predict_proba(plan_features(u))[:, 1] for u in test}
    else:
        raise ValueError(kind)
    return {u['id']: model.predict(plan_features(u)) for u in test}


def cv(units, kinds, seed=SEED):
    dev = dev_units(units); F = folds(dev); rng = np.random.default_rng(seed); res = {}
    for kind in kinds:
        rows = []
        for k in range(FOLDS):
            test = F[k]; train = [u for j in range(FOLDS) if j != k for u in F[j]]
            if not test or not train:
                continue
            pred = fit_predict(kind, train, test)
            for u in test:
                m = pool_metrics(pred[u['id']], u, rng)
                if m:
                    rows.append(dict(m, world=u['world']))
        worlds = [r['world'] for r in rows]
        agg = {}
        for key in rows[0]:
            if key == 'world':
                continue
            v = [r[key] for r in rows]
            agg[key] = dict(zip(('mean', 'ci_lo', 'ci_hi'), pp._bootstrap_ci(v, worlds, 1000, seed)))
        res[kind] = dict(searches=len(rows), worlds=len(set(worlds)), **agg)
        print(kind, json.dumps({k: (round(v['mean'], 3) if isinstance(v, dict) else v) for k, v in res[kind].items()}),
              flush=True)
    return res


if __name__ == '__main__':
    from . import learned_search_report as lr
    t = time.time()
    rec, _ = lr.load_cache('E:/agent-work/_plan_proposer_cache')
    U = ls.build_units(rec)
    kinds = sys.argv[1].split(',') if len(sys.argv) > 1 else ['ridge', 'hgb', 'hgb_top', 'hgb_cls', 'net:2000']
    out = dict(held_out_fixed=HELD72, dev_searches=len(dev_units(U)), dev_worlds=len({u['world'] for u in dev_units(U)}),
               results=cv(U, kinds), wall_s=None)
    out['wall_s'] = time.time()-t
    path = sys.argv[2] if len(sys.argv) > 2 else 'E:/agent-work/_plan_proposer_cache/ls/cv.json'
    json.dump(out, open(path, 'w'), indent=1)
