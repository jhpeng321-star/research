"""LEARNED_TRIGGER offline training (TRIGCLF 2026-11-09; CPU, zero SUMO, light): labels, seed-grouped CV, frozen
threshold, frozen model doc. Never imported by the hook (numpy / sklearn here only).

Labels (per candidate row of learned_trigger collect = feasible (tape, second, family, members)):
  pos  : helped activation (PLAN activation, paired diff < 0; admitted bin, or the unconditional row) - the row of its
         family on the plan's own tape at the latest feasible second in [t - lookback, t] (normally t itself);
         weight = -diff / (activations of that seed+family), i.e. the seed's verified benefit split over its onsets.
  harm : hurt activation (diff > 0) - its family's row on each reference tape (incumbent wall_g_solo2 + u0; u0 only
         for the unconditional row) at the latest feasible second in the same window; weight = diff / n_acts / n_refs.
  bg   : every other feasible row on a NON-plan tape (reference arms and other plans' tapes) at t % 5 == 0 that is
         more than EXCL_S from any activation onset (helped or hurt, any arm) of that family on that seed; rows on
         the plan's own tape are never background (the plan is executing / cooling down there). Total bg weight =
         BG_SHARE x total harm weight, uniform per row.
CV: GroupKFold by seed (5 folds). Model: StandardScaler + L2 logistic regression; C chosen by mean fold AUC on
onset rows (pos vs harm). Threshold: maximises out-of-fold net verified benefit over onset rows
(sum pos weight - sum harm weight with p >= tau); ties -> higher tau. Calibration: OOF probability deciles.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from . import learned_trigger as L

INCUMBENT = 'wall_g_solo2'
EXCL_S = 30
BG_SHARE = 0.5
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
FOLDS = 5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cands(path):
    with gzip.open(L.guard_path(path), 'rt', encoding='utf-8') as fh:
        return [json.loads(line) for line in fh]


def label_rows(cands, acts, families):
    """(rows with label/weight/kind, counts)."""
    acts = [a for a in acts if a['family'] in families]
    L.assert_seeds([a['seed'] for a in acts] + [c['seed'] for c in cands])
    idx = defaultdict(dict)                     # (tape, family) -> {t: row}
    for c in cands:
        idx[(c['tape'], c['family'])][c['t']] = c
    n_sf = defaultdict(int)
    for a in acts:
        n_sf[(a['seed'], a['family'])] += 1
    onsets = defaultdict(list)                  # (seed, family) -> onset seconds
    for a in acts:
        onsets[(a['seed'], a['family'])].append(int(a['t']))
    labeled, counts = {}, defaultdict(int)

    def pick(tape, fam, t, lb):
        rows = idx.get((tape, fam), {})
        for s in range(int(t), int(t) - int(lb) - 1, -1):
            if s in rows:
                return rows[s]
        return None

    for a in acts:
        lb = int(a.get('lookback', 8))
        job = f"{a['cell']}_s{a['seed']}_"
        if a['diff'] < 0:
            r = pick(f"{a['root']}__{job}{a['family']}", a['family'], a['t'], lb)
            if r is None:
                counts['pos_not_feasible_or_missing'] += 1
                continue
            key = (r['tape'], r['t'], r['family'])
            w = -a['diff'] / n_sf[(a['seed'], a['family'])]
            prev = labeled.get(key)
            labeled[key] = dict(r, y=1, kind='pos', w=w + (prev['w'] if prev else 0.), diff=a['diff'],
                                act_t=a['t'], lag=a['t'] - r['t'])
            counts['pos'] += 1
        elif a['diff'] > 0:
            refs = [x for x in (a.get('ref_arms') or (INCUMBENT, 'u0')) if x != a['family']]
            for arm in refs:
                r = pick(f"{a['root']}__{job}{arm}", a['family'], a['t'], lb)
                if r is None:
                    counts['harm_not_feasible_or_missing'] += 1
                    continue
                key = (r['tape'], r['t'], r['family'])
                if key in labeled and labeled[key]['y'] == 1:
                    counts['harm_conflict_skipped'] += 1
                    continue
                w = a['diff'] / n_sf[(a['seed'], a['family'])] / len(refs)
                prev = labeled.get(key)
                labeled[key] = dict(r, y=0, kind='harm', w=w + (prev['w'] if prev else 0.), diff=a['diff'],
                                    act_t=a['t'], lag=a['t'] - r['t'])
                counts['harm'] += 1
    bg = []
    for c in cands:
        key = (c['tape'], c['t'], c['family'])
        if key in labeled or c['arm'] == c['family'] or c['t'] % 5:
            continue
        if any(abs(c['t'] - t) <= EXCL_S for t in onsets.get((c['seed'], c['family']), ())):
            counts['bg_excluded_near_onset'] += 1
            continue
        bg.append(dict(c, y=0, kind='bg', w=None))
    harm_w = sum(r['w'] for r in labeled.values() if r['kind'] == 'harm')
    for r in bg:
        r['w'] = BG_SHARE * harm_w / max(1, len(bg))
    rows = list(labeled.values()) + bg
    rows.sort(key=lambda r: (r['tape'], r['t'], r['family']))
    counts.update(pos_rows=sum(r['kind'] == 'pos' for r in rows), harm_rows=sum(r['kind'] == 'harm' for r in rows),
                  bg_rows=len(bg), seeds=len({r['seed'] for r in rows}))
    return rows, dict(counts)


def auc(y, s, w=None):
    from sklearn.metrics import roc_auc_score
    import numpy as np
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, s, sample_weight=w))


def fit(X, y, w, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    sc.scale_[sc.scale_ < 1e-9] = 1.0
    m = LogisticRegression(C=C, max_iter=5000, solver='lbfgs')
    m.fit(sc.transform(X), y, sample_weight=w)
    return sc, m


def oof(rows, C):
    import numpy as np
    from sklearn.model_selection import GroupKFold
    X = np.array([r['f'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows])
    w = np.array([r['w'] for r in rows], dtype=float)
    w = w / w.mean()
    g = np.array([r['seed'] for r in rows])
    p = np.zeros(len(rows))
    for tr, te in GroupKFold(n_splits=FOLDS).split(X, y, g):
        sc, m = fit(X[tr], y[tr], w[tr], C)
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def metrics(rows, p):
    import numpy as np
    k = np.array([r['kind'] for r in rows])
    y = np.array([r['y'] for r in rows])
    w = np.array([r['w'] for r in rows])
    on = k != 'bg'
    return dict(auc_onset_pos_vs_harm=auc(y[on], p[on]),
                auc_onset_pos_vs_harm_weighted=auc(y[on], p[on], w[on]),
                auc_pos_vs_all_neg=auc(y, p),
                auc_pos_vs_bg=auc(y[k != 'harm'], p[k != 'harm']))


def per_family_auc(rows, p, families):
    import numpy as np
    out = {}
    for f in families:
        sel = np.array([r['family'] == f and r['kind'] != 'bg' for r in rows])
        y = np.array([r['y'] for r in rows])[sel]
        out[f] = dict(n_pos=int(y.sum()), n_harm=int((1 - y).sum()), auc=auc(y, p[sel]) if sel.any() else None)
    return out


def calibration(rows, p, bins=10):
    import numpy as np
    y = np.array([r['y'] for r in rows])
    w = np.array([r['w'] for r in rows], dtype=float)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    out = []
    for i in range(bins):
        sel = (p >= edges[i]) & ((p <= edges[i + 1]) if i == bins - 1 else (p < edges[i + 1]))
        if sel.any():
            out.append(dict(lo=round(float(edges[i]), 4), hi=round(float(edges[i + 1]), 4), n=int(sel.sum()),
                            mean_pred=round(float(p[sel].mean()), 4), frac_pos=round(float(y[sel].mean()), 4),
                            weighted_frac_pos=round(float((w[sel] * y[sel]).sum() / w[sel].sum()), 4)))
    return out


def threshold_curve(rows, p):
    """Out-of-fold net verified benefit over onset rows at each candidate threshold (veh-s per CV episode-split)."""
    import numpy as np
    k = np.array([r['kind'] for r in rows])
    w = np.array([r['w'] for r in rows])
    on = k != 'bg'
    taus = sorted(set(np.round(np.quantile(p[on], np.linspace(0, 1, 101)), 6).tolist()))
    taus = [t for t in taus if 0 < t < 1]
    curve = []
    for t in taus:
        fire = p >= t
        helped = float(w[fire & (k == 'pos')].sum())
        hurt = float(w[fire & (k == 'harm')].sum())
        npos, nharm = int((fire & (k == 'pos')).sum()), int((fire & (k == 'harm')).sum())
        curve.append(dict(tau=t, net_veh_s=helped - hurt, helped_veh_s=helped, hurt_veh_s=hurt, pos_fired=npos,
                          harm_fired=nharm, precision=npos / max(1, npos + nharm),
                          recall=npos / max(1, int((k == 'pos').sum())),
                          bg_fire_rate=float(fire[k == 'bg'].mean()) if (k == 'bg').any() else None))
    return curve


def main(argv=None):
    import argparse
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument('--cands', required=True)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    meta = json.loads(Path(a.cands + '.meta.json').read_text(encoding='utf-8'))
    plan = json.loads(Path(L.guard_path(a.plan)).read_bytes())
    if sha(a.plan) != meta['plan_sha256']:
        raise ValueError('PLAN differs from the collect PLAN')
    families = meta['families']
    rows, counts = label_rows(load_cands(a.cands), plan['activations'], families)
    L.assert_seeds({r['seed'] for r in rows})
    names = L.feature_names(families)
    if any(len(r['f']) != len(names) for r in rows):
        raise ValueError('feature length differs from feature_names')
    cv = {}
    for C in C_GRID:
        p = oof(rows, C)
        cv[str(C)] = dict(metrics(rows, p))
    best_c = max(C_GRID, key=lambda C: (cv[str(C)]['auc_onset_pos_vs_harm'] or 0, -C))
    p = oof(rows, best_c)
    curve = threshold_curve(rows, p)
    best = max(curve, key=lambda c: (round(c['net_veh_s'], 6), c['tau']))
    X = np.array([r['f'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows])
    w = np.array([r['w'] for r in rows], dtype=float)
    sc, m = fit(X, y, w / w.mean(), best_c)
    model = dict(schema=L.SCHEMA, families=families, strategy_keys=meta['strategy_keys'], feature_names=names,
                 mean=[float(v) for v in sc.mean_], scale=[float(v) for v in sc.scale_],
                 coef=[float(v) for v in m.coef_[0]], intercept=float(m.intercept_[0]),
                 threshold=float(best['tau']),
                 threshold_rule='argmax out-of-fold (seed-grouped 5-fold) net verified benefit over onset rows; '
                                'ties -> higher threshold',
                 C=best_c, cv=dict(folds=FOLDS, groups='seed', grid=cv, chosen=metrics(rows, p),
                                   per_family=per_family_auc(rows, p, families),
                                   at_threshold=best, calibration=calibration(rows, p)),
                 data=dict(cands_sha256=sha(a.cands), plan_sha256=meta['plan_sha256'],
                           guidance_sha256=meta['guidance_sha256'], cell=meta['cell'], counts=counts,
                           seeds=sorted({r['seed'] for r in rows}), every=meta['every'], excl_s=EXCL_S,
                           bg_share=BG_SHARE, collect_digest=meta['digest']),
                 code_digest=L.learned_trigger_digest())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(model, indent=1, sort_keys=True) + '\n').encode('utf-8')
    (out / 'LEARNED_TRIGGER_MODEL.json').write_bytes(raw)
    (out / 'LEARNED_TRIGGER_MODEL.json.sha256').write_text(hashlib.sha256(raw).hexdigest() + '\n', encoding='utf-8')
    (out / 'THRESHOLD_CURVE.json').write_text(json.dumps(curve, indent=1) + '\n', encoding='utf-8')
    with gzip.open(out / 'LABELED.jsonl.gz', 'wt', encoding='utf-8') as fh:
        for r, q in zip(rows, p):
            fh.write(json.dumps(dict({k: r[k] for k in ('tape', 'seed', 'arm', 't', 'family', 'kind', 'y', 'w',
                                                        'members')}, oof=float(q)), sort_keys=True) + '\n')
    print(json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest(), C=best_c, counts=counts, chosen=model['cv']['chosen'],
                          at_threshold=best), indent=1))


if __name__ == '__main__':
    main()
