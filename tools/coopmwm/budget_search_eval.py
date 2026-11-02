"""Budget-reduced PLAIN search: offline non-inferiority check (registered: BUDGET_SEARCH_PREREG.md).

The question is whether the registered plain MPPI search can use fewer candidates or
iterations (horizons unchanged: screen K4/H300, final K8/H_final) without losing the
quality that matters downstream. Exact SUMO picks among the WM final top-8, so the
quality that matters is the best of the final top-8 (primary) and the top-1.

NOISY surrogate replay. Each held-out search is replayed through the real
``mppi_search`` code:
- truth = RBF kernel-ridge surrogate of the search's own recorded WM screen pool;
- WM screen score = truth + N(0, s_s^2), with s_s = sqrt(1 - LOO R^2) x the sd of
  the recorded pool (the surrogate's out-of-sample residual = K4 screen noise);
- WM final score = truth + N(0, s_s^2 / 2), since K8 has half the K4 variance.

The noise model is validated against the observed screen-vs-final agreement among
finalists (Spearman 0.44, top-1 agreement 0.51 on TRAIN dev searches).
"""
import json
import math

import numpy as np

from . import learned_search_diversity as ld
from . import learned_search_eval as le
from . import mppi_search as ms
from . import plan_proposer as pp

CONFIGS = {
    'plain': None,
    'plain_alt': None,
    'pop50': dict(pop_min=16, pop_per_dim=1),                 # population x0.5, 3 iterations
    'it2': dict(iterations=2),                                # population x1, 2 iterations
    'pop50_it2': dict(pop_min=16, pop_per_dim=1, iterations=2),
}
SEARCH_WALL_S = {2: 559.567, 3: 794.766, 4: 1029.964, 5: 964.120, 6: 898.275}   # COOPMWM.md measured
SEED = 20261102


def futures(m, config):
    c = dict(ms.REGISTERED, **(config or {}))
    sc = ms.scaling(m, ms.canonical_knots(), c['top_k'], c)
    return (sc['population']+2)*c['screen_K']*c['iterations']+(sc['finalists']+2)*c['final_K']


def noisy_replay(receipt, sur, sigma, config, tag):
    group = le._group(receipt); m = len(group['members'])
    policy = np.zeros((m, ms.REGISTERED['max_duration_s'])); regime = receipt.get('regime') or 'unknown'
    record = dict(state_id=str(receipt.get('search_id')), rows=receipt['rows'],
                  policy_residuals={v: 0. for v in group['members']})
    seed = int(ms._digest([SEED, receipt.get('search_id'), tag])[:15], 16)
    rng = np.random.default_rng(seed+1)
    screened, final = [], []

    def screen(rec, grp, cands):
        t = sur([c['perturbation_canonical'] for c in cands], [c['duration_s'] for c in cands])
        noisy = t+sigma*rng.standard_normal(len(t))
        screened.extend(dict(id=c['id'], truth=float(a), screen=float(b), z=np.asarray(c['perturbation_canonical'], float),
                             D=int(c['duration_s'])) for c, a, b in zip(cands, t, noisy))
        return -noisy

    def evaluate(rec, actions, K, H):
        cands = [a for a in actions if 'perturbation_canonical' in a]
        t = sur([a['perturbation_canonical'] for a in cands], [a['duration_s'] for a in cands]) if cands else []
        noisy = np.asarray(t)+sigma/math.sqrt(2)*rng.standard_normal(len(cands))
        final.extend(dict(id=a['id'], truth=float(x), final=float(y)) for a, x, y in zip(cands, t, noisy))
        val = {a['id']: float(y) for a, y in zip(cands, noisy)}
        return {a['id']: [-val.get(a['id'], 0.)]*K for a in actions}
    ms.search(record, group, policy, le.WIDE, evaluate, regime, seed=seed, config=config, screen=screen)
    return screened, final


def evaluate_configs(heldout, configs=CONFIGS, bootstraps=2000, collapse_ratio=.75):
    rows, agree = [], []
    for r in heldout:
        sur = le.surrogate_for(r)
        if sur is None:
            continue
        rec_sav = np.array([c['saving_vs_policy'] for c in r['candidates'] if c.get('stage', 'screen') == 'screen'
                            and c.get('saving_vs_policy') is not None], float)
        sigma = math.sqrt(max(0., 1-sur.loo_r2))*float(rec_sav.std())
        m = len(r['members']); base = float(sur([np.zeros((m, ms.canonical_knots()))], [300])[0])
        out = {}
        for name, cfg in configs.items():
            out[name] = noisy_replay(r, sur, sigma, cfg, name)
        sc0, fi0 = out['plain']
        byid = {s['id']: s['screen'] for s in sc0}
        fs = [(byid[f['id']], f['final']) for f in fi0 if f['id'] in byid]
        if len(fs) >= 6:
            a, b = np.array(fs).T
            agree.append((float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]),
                          float(np.argmax(a) == np.argmax(b))))

        def best8_top1(final):
            fin = sorted(final, key=lambda f: -f['final'])[:8]
            return max(f['truth'] for f in fin), fin[0]['truth']
        b8p, _ = best8_top1(out['plain'][1])
        scale = b8p-base
        if not scale > 0:
            continue
        spacing = ld.nn_spacing([s['z'] for s in out['plain'][0]])
        E = ms.scaling(m, ms.canonical_knots())['elites']
        row = dict(id=r.get('search_id'), world=str(r.get('world_id')), m=m, sigma=sigma, modes={})
        for name, (sc, fi) in out.items():
            b8, t1 = best8_top1(fi)
            order = sorted(range(len(sc)), key=lambda i: -sc[i]['truth'])
            row['modes'][name] = dict(q8=(b8-base)/scale, q1=(t1-base)/scale, futures=futures(m, configs[name]),
                                      elite_clusters=ld.clusters([sc[i]['z'] for i in order[:E]], spacing),
                                      top8_patterns=len({ld.member_pattern(sc[i]['z'], sc[i]['D']) for i in order[:8]}),
                                      top1_pattern=ld.member_pattern(sc[order[0]]['z'], sc[order[0]]['D']),
                                      spread=ld.mean_pairwise([s['z'] for s in sc]))
        rows.append(row)
    if not rows:
        return dict(searches=0)
    worlds = [x['world'] for x in rows]
    ci = lambda v, k: dict(zip(('mean', 'ci_lo', 'ci_hi'), pp._bootstrap_ci(v, worlds, bootstraps, SEED+k)))
    g = lambda name, key: [x['modes'][name][key] for x in rows]
    d8 = float(np.mean(np.abs(np.array(g('plain', 'q8'))-np.array(g('plain_alt', 'q8')))))
    d1 = float(np.mean(np.abs(np.array(g('plain', 'q1'))-np.array(g('plain_alt', 'q1')))))
    from collections import Counter
    res = dict(searches=len(rows), worlds=len(set(worlds)), margin_q8=d8, margin_q1=d1,
               noise_model_check=dict(spearman_screen_final_finalists=float(np.mean([a for a, _ in agree])) if agree else None,
                                      top1_agreement=float(np.mean([b for _, b in agree])) if agree else None,
                                      observed_dev=dict(spearman=.44, top1_agreement=.51)),
               configs={})
    ec_alt = float(np.mean(g('plain_alt', 'elite_clusters')))
    for k, name in enumerate(configs):
        q8 = ci([a-b for a, b in zip(g(name, 'q8'), g('plain_alt', 'q8'))], 10+k)
        q1 = ci([a-b for a, b in zip(g(name, 'q1'), g('plain_alt', 'q1'))], 30+k)
        ec = float(np.mean(g(name, 'elite_clusters')))
        fut = float(np.mean([x['modes'][name]['futures']/x['modes']['plain']['futures'] for x in rows]))
        top1 = Counter(g(name, 'top1_pattern'))
        res['configs'][name] = dict(
            q8_minus_plain_alt=q8, q1_minus_plain_alt=q1, q8=float(np.mean(g(name, 'q8'))), q1=float(np.mean(g(name, 'q1'))),
            futures_ratio=fut, wall_s=float(np.mean([SEARCH_WALL_S[x['m']]*x['modes'][name]['futures']/x['modes']['plain']['futures']
                                                     for x in rows])),
            elite_clusters=ec, top8_patterns=float(np.mean(g(name, 'top8_patterns'))),
            spread_ratio=float(np.mean([x['modes'][name]['spread']/x['modes']['plain']['spread'] for x in rows])),
            top1_most_common_share=top1.most_common(1)[0][1]/len(rows), top1_distinct=len(top1),
            noninferior=bool(q8['ci_lo'] > -d8 and q1['ci_lo'] > -d1 and ec >= collapse_ratio*ec_alt))
    ok = {n: v for n, v in res['configs'].items() if n not in ('plain', 'plain_alt') and v['noninferior']}
    res['recommended'] = min(ok, key=lambda n: ok[n]['futures_ratio']) if ok else None
    res['per_search'] = rows
    return res
