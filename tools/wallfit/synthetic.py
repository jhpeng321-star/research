"""Synthetic LABELS-schema pairs (fakes) for CPU tests and the operator smoke run. No SUMO, no WM.

Hidden rule: SUMO-like synergy grows with the observable ramp queue and the wall size; the fake
'WM' prediction is anti-correlated with it (the mis-ranking W3 must correct).
"""
import json
from pathlib import Path

import numpy as np

from . import dataset as ds
from .features import RAMP_END_S, move_key


def _situation(rng, w, j):
    rows, n = [], 0

    def add(lane, s, v, cav, prefix='mainline'):
        nonlocal n
        n += 1
        rows.append(dict(vid=f'{prefix}.{w % 100}{j:02d}{n:03d}', s_m=float(s), v_mps=float(v), lane=int(lane),
                         is_cav=bool(cav), length_m=5.))
    for lane in (0, 1):
        s = -500. + rng.uniform(0, 20)
        while s < 150.:
            add(lane, s, rng.uniform(8, 28), rng.random() < .4)
            s += rng.uniform(15., 45.)
    queue = int(rng.integers(0, 7))
    for q in range(queue):
        add(2, RAMP_END_S - 2. - 8. * q, rng.uniform(0., 1.5), rng.random() < .4, 'ramp')
    for _ in range(int(rng.integers(0, 3))):
        add(2, rng.uniform(-300., RAMP_END_S - 8. * queue - 10.), rng.uniform(8., 20.), False, 'ramp')
    return rows, queue


def synthetic_pairs(n_worlds=12, per_world=8, seed=0, start=986000, latent_dim=4):
    """(pairs, latents) in the dataset/latent schemas; world seeds start..start+n_worlds-1."""
    rng = np.random.default_rng(seed)
    pairs, latents = [], {}
    for w in range(n_worlds):
        world = start + w
        for j in range(per_world):
            rows, queue = _situation(rng, world, j)
            mains = sorted([r for r in rows if r['lane'] in (0, 1) and -300. <= r['s_m'] <= RAMP_END_S],
                           key=lambda r: -r['s_m'])
            k = int(rng.integers(3, 5))
            members = [r['vid'] for r in mains[:k]]
            for r in rows:
                if r['vid'] in members:
                    r['is_cav'] = True
            move = dict(family='wall', members=members, onset=0, hold=30, depth=1.)
            truth = 120. * queue - 350. + 60. * (len(members) - 3)
            sid = f'syn_R_train_{world}_t{300 + 5 * j}'
            seeds = []
            for s_i in range(int(rng.integers(1, 3))):
                syn = truth + rng.normal(0., 30.)
                lb = syn - 60.
                seeds.append(ds._seed_row(source='probe', source_path='synthetic', state_id=sid, world_seed=world,
                                          t0=300 + 5 * j, setting='wall', arm='R', plan_id='g_wall_' + '_'.join(members),
                                          move=move, seed_id='crn1' if s_i == 0 else f'salt:syn#{sid}#r{s_i + 1}',
                                          seed_index=s_i + 1, tail_s=120, group_adv=syn + 50., best_solo_adv=50.,
                                          synergy=syn, synergy_lb95=lb, admitted=bool(lb > 0),
                                          status='admitted' if lb > 0 else 'not_admitted'))
            qid = f'wall__{sid}__E'
            wm = dict(mean_synergy=float(-.3 * truth + rng.normal(0., 40.)), saving_vs_policy=float(rng.normal(0, 30)),
                      best_solo_saving=float(rng.normal(0, 10)), draws=None, horizon_s=120, source='rank')
            pairs.append(dict(schema=ds.SCHEMA, pair_id=f'{sid}#{move_key(move)}', state_id=sid, world_seed=world,
                              t0=300 + 5 * j, settings=['wall'], move=move, plan_ids=[seeds[0]['plan_id']],
                              situation=dict(source='query', query_id=qid, rows=[dict(r) for r in rows]), wm=wm,
                              labels=ds.aggregate(seeds), seeds=seeds))
            if latent_dim:
                vids = [r['vid'] for r in rows]
                latents[qid] = dict(vids=vids, embedding=rng.normal(size=(len(vids), latent_dim)).astype(np.float32),
                                    values=rng.normal(size=(len(vids), 9)).astype(np.float32),
                                    uncertainty=np.abs(rng.normal(size=(len(vids), 9))).astype(np.float32))
    return pairs, latents


def write_labels(pairs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(p, sort_keys=True) + '\n' for p in pairs))
    return path


def write_latents(latents, path):
    import torch
    items = {k: {kk: (torch.as_tensor(vv) if kk != 'vids' else list(vv)) for kk, vv in v.items()}
             for k, v in latents.items()}
    torch.save(dict(schema='wallfit-latent-v1', items=items, errors={}, n_queries=len(items)), str(path))
    return Path(path)
