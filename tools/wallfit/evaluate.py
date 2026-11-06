"""W3 evaluation: rank correlation and precision@k for admitted synergy on TEST world seeds, head vs
the CURRENT WM ranking (the WM's own mean predicted synergy). No other comparator is trained.

Units: one (situation, joint move) pair. Spearman uses the pair's mean SUMO synergy over its measured
CRN seeds; precision@k uses `admitted_any` (some measured seed admitted, LB95 > 0) with expected
precision under random tie-breaking. The WM can only rank pairs it predicted, so the head is also
reported on that common subset (`head_common` vs `wm_common`).

run(): (1) fixed split -- head trained on TRAIN-split seeds, scored on the test seeds;
       (2) grouped cross-validation by world seed -- out-of-fold scores for every pair;
       (3) HEAD.pt trained on every labelled seed with the same config (the deployable head).
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from . import dataset as ds
from . import guard
from . import head as hd

KS = (1, 3, 5, 10)


def average_ranks(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.
        i = j + 1
    return ranks


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = average_ranks(x), average_ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def precision_at_k(scores, positive, k):
    """Expected fraction of positives among the top-k by score (ties broken uniformly at random)."""
    s = np.asarray(scores, float)
    pos = np.asarray(positive, float)
    n = len(s)
    if n == 0:
        return None
    k = min(int(k), n)
    above = s > np.sort(s)[::-1][k - 1]
    tied = s == np.sort(s)[::-1][k - 1]
    need = k - above.sum()
    expected = pos[above].sum() + need * (pos[tied].mean() if tied.any() else 0.)
    return float(expected / k)


def ranking_metrics(scores, pairs, ks=KS):
    syn = [(s, p['labels']['synergy']) for s, p in zip(scores, pairs) if p['labels']['synergy'] is not None]
    adm = [(s, float(bool(p['labels']['admitted_any']))) for s, p in zip(scores, pairs)
           if p['labels']['admitted_any'] is not None]
    out = dict(n=len(pairs), n_synergy=len(syn), n_admit_determined=len(adm),
               n_admitted=int(sum(a for _, a in adm)),
               base_rate=(sum(a for _, a in adm) / len(adm)) if adm else None,
               spearman=spearman([s for s, _ in syn], [y for _, y in syn]) if syn else None)
    for k in ks:
        out[f'precision_at_{k}'] = precision_at_k([s for s, _ in adm], [a for _, a in adm], k) if adm else None
    return out


def wm_score(pair):
    wm = pair.get('wm')
    return None if not wm or wm.get('mean_synergy') is None else float(wm['mean_synergy'])


def compare(pairs, head_scores):
    """head_all (every pair) + head_common / wm_common on the pairs the current WM ranked."""
    common = [i for i, p in enumerate(pairs) if wm_score(p) is not None]
    return dict(head_all=ranking_metrics(head_scores, pairs),
                head_common=ranking_metrics([head_scores[i] for i in common], [pairs[i] for i in common]),
                wm_common=ranking_metrics([wm_score(pairs[i]) for i in common], [pairs[i] for i in common]),
                n_common=len(common))


def cross_validate(pairs, split, latents=None, config=None):
    """Out-of-fold head scores by world-seed fold."""
    fold = {int(k): v for k, v in split['fold_of'].items()}
    oof = [None] * len(pairs)
    fits = []
    for f in range(split['folds']):
        test = [i for i, p in enumerate(pairs) if fold[int(p['world_seed'])] == f]
        train = [p for i, p in enumerate(pairs) if fold[int(p['world_seed'])] != f]
        if not test or not train:
            continue
        b = hd.fit(train, latents, config)
        fits.append(dict(fold=f, n_train=len(train), n_test=len(test), fit_wall_s=b['fit_wall_s']))
        for i, pr in zip(test, hd.predict(b, [pairs[i] for i in test], latents)):
            oof[i] = pr
    return oof, fits


def _load_latents(paths):
    if not paths:
        return None
    import torch
    out = {}
    for p in paths:
        d = torch.load(str(guard.check_path(p)), map_location='cpu', weights_only=True)
        if d.get('schema') != 'wallfit-latent-v1':
            raise ValueError(f'not a wallfit latent file: {p}')
        for k, v in d['items'].items():
            out.setdefault(k, {kk: (vv.numpy() if hasattr(vv, 'numpy') else vv) for kk, vv in v.items()})
    return out


def run(labels, out, *, latent=(), config=None, folds=5, test_seeds=None, salt=ds.SPLIT_SALT):
    begin = time.time()
    out = guard.check_path(out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = ds.load_pairs([guard.check_path(p) for p in labels])
    if not pairs:
        raise ValueError('no labelled pairs')
    latents = _load_latents(latent)
    split = ds.split_by_seed([p['world_seed'] for p in pairs], folds, salt, test_seeds)
    test_set = set(split['test_seeds'])
    train = [p for p in pairs if p['world_seed'] not in test_set]
    test = [p for p in pairs if p['world_seed'] in test_set]
    result = dict(schema='wallfit-eval-v1', n_pairs=len(pairs), split=split, config=dict(hd.DEFAULT_CONFIG, **(config or {})),
                  latent_files=[str(p) for p in latent], latent_items=len(latents or {}),
                  pairs_with_latent=sum(hd.latent_for(p, latents) is not None for p in pairs),
                  pairs_with_wm=sum(wm_score(p) is not None for p in pairs))
    preds = {p['pair_id']: dict(pair_id=p['pair_id'], world_seed=p['world_seed'],
                                fold=split['fold_of'][str(p['world_seed'])],
                                role='test' if p['world_seed'] in test_set else 'train', move=p['move'],
                                labels={k: p['labels'][k] for k in ('synergy', 'admitted_any', 'admitted_rate',
                                                                    'n_measured', 'group_adv', 'best_solo_adv')},
                                wm_score=wm_score(p)) for p in pairs}
    if train and test:
        b = hd.fit(train, latents, config)
        hd.save(b, out / 'HEAD_SPLIT.pt')
        tp = hd.predict(b, test, latents)
        result['split_eval'] = compare(test, [x['score'] for x in tp])
        for p, x in zip(test, tp):
            preds[p['pair_id']]['head_split'] = x
    else:
        result['split_eval'] = None
    oof, fits = cross_validate(pairs, split, latents, config)
    have = [i for i, x in enumerate(oof) if x is not None]
    result['cv_eval'] = compare([pairs[i] for i in have], [oof[i]['score'] for i in have]) if have else None
    result['cv_fits'] = fits
    for i in have:
        preds[pairs[i]['pair_id']]['head_cv'] = oof[i]
    final = hd.fit(pairs, latents, config)
    hd.save(final, out / 'HEAD.pt')
    for p, x in zip(pairs, hd.predict(final, pairs, latents)):
        preds[p['pair_id']]['head_all_in_sample'] = x
    result.update(wall_s=time.time() - begin, new_sumo_sim_s=0, wm_retrained=False,
                  interpretation='head = residual on the frozen WM prediction; compared ONLY with the current WM '
                                 'ranking (wm_mean_synergy) on test world seeds and out-of-fold; HEAD.pt is trained '
                                 'on every labelled TRAIN seed with the same config')
    _write(out / 'PREDICTIONS.jsonl', ''.join(json.dumps(v, sort_keys=True) + '\n' for v in preds.values()))
    _write(out / 'RESULT.json', json.dumps(result, indent=1, sort_keys=True, default=float))
    return result


def _write(p, text):
    p = guard.check_path(p)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, p)


def summary(result):
    def pick(m):
        return None if m is None else {k: m[k] for k in ('n', 'n_admitted', 'spearman', 'precision_at_1',
                                                         'precision_at_3', 'precision_at_5')}
    out = {}
    for name in ('split_eval', 'cv_eval'):
        r = result.get(name)
        out[name] = None if r is None else dict(head_all=pick(r['head_all']), head_common=pick(r['head_common']),
                                                 wm_common=pick(r['wm_common']))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m tools.wallfit run', description='train + evaluate the W3 head')
    ap.add_argument('--labels', action='append', required=True)
    ap.add_argument('--latent', action='append', default=[])
    ap.add_argument('--out', required=True)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--test-seed', action='append', type=int, default=None)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--epochs', type=int, default=hd.DEFAULT_CONFIG['epochs'])
    ap.add_argument('--ensemble', type=int, default=hd.DEFAULT_CONFIG['ensemble'])
    a = ap.parse_args(argv)
    r = run(a.labels, Path(a.out), latent=a.latent, folds=a.folds, test_seeds=a.test_seed,
            config=dict(device=a.device, epochs=a.epochs, ensemble=a.ensemble))
    print(json.dumps(summary(r), indent=1, sort_keys=True))
    return r
