"""Independent recorded-only WMDATA labels. Run on cpuQ; never imported by WM.

Only observable trajectories, identity registry and public departure RNG are
read. The public RNG reconstructs offline service labels, never model inputs.
"""
from pathlib import Path
import hashlib
import json
import random
import sys

import numpy as np
import torch
from aggregates import fields, merge_queue

ROOT = Path('/home/ubuntu/wmdata20261031')
HORIZONS = (10, 30, 60, 90, 120, 180, 300)
ARMS = ('native', 'ctrl0', 'mappo', 'pi', 'circles')


def departures(seed, rate, stream):
    rng = random.Random(f'damping_requests_v1:{seed}:{stream}')
    t, result = 0., []
    while True:
        t += rng.expovariate(rate/3600.)
        if t >= 1200:
            return np.asarray(result)
        result.append(round(t, 6))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def one(world, arm):
    seed = int(world['seed'])
    assert 969400 <= seed <= 969439
    root = ROOT/'jobs'/f's{seed}_{arm}'
    result = json.loads((root/'result.json').read_text())
    assert result['ok']
    st = torch.load(root/'start.pt', map_location='cpu', weights_only=False)
    with np.load(root/'start_history.npz', allow_pickle=False) as x:
        indices = x['vid_index'][-1]
        keep = st['state']['present'][0].numpy().astype(bool)
        original_ids = indices[keep].astype(int)
        initial_kin = x['kin'][-1, keep]
    tape = root/'traj_base.npz'
    assert digest(tape) == result['output_hashes']['traj_base.npz']
    with np.load(tape, allow_pickle=False) as x:
        t, vid, s, v, lane = (x[k] for k in ('t','vid','s','v','lane'))
    assert float(t.min()) == 300.5
    # The post-step tape begins at t0+.5; the genuine observed start supplies
    # the first left integration boundary, never a fabricated future state.
    t = np.r_[np.full(len(original_ids),300.),t]
    vid = np.r_[original_ids,vid]
    s = np.r_[initial_kin[:,0],s]
    v = np.r_[initial_kin[:,4],v]
    lane = np.r_[initial_kin[:,1:4].argmax(-1),lane]
    times = 300.+np.arange(601)*.5
    rows = [np.flatnonzero(np.isclose(t, at, atol=1e-7, rtol=0)) for at in times]
    initial = rows[0]
    assert set(vid[initial]) == set(original_ids), dict(seed=seed,arm=arm,
        first_times=np.unique(t)[:5].tolist(),n_t300=len(initial),n_start=len(original_ids),
        missing=sorted(set(original_ids)-set(vid[initial]))[:10],
        extra=sorted(set(vid[initial])-set(original_ids))[:10])
    for identity, kin in zip(original_ids, initial_kin):
        j = initial[vid[initial] == identity]
        assert len(j) == 1
        assert abs(float(s[j[0]])-float(kin[0])) < .01
        assert abs(float(v[j[0]])-float(kin[4])) < .01
    due = sum(np.searchsorted(departures(seed, rate, stream), times, side='right')
              for rate, stream in ((world['demand'],'mainline'), (400,'ramp')))
    latest = len(st['inserted'])-1
    assert max(vid[initial]) == latest
    inserted = []
    for row in rows:
        if len(row):
            latest = max(latest, int(vid[row].max()))
        inserted.append(latest+1)
    queues = due-np.asarray(inserted)
    assert (queues >= 0).all() and queues[0] == st['queue']
    road = np.zeros((600,3))
    cross = np.zeros(600, int)
    # Constant observable ID registry size, with absent slots masked.
    slots = int(vid.max())+1
    gs = np.zeros((1, 600, slots), dtype=np.float32)
    gv = np.zeros_like(gs)
    gm = np.zeros_like(gs, dtype=bool)
    for i in range(600):
        row = rows[i]
        if not len(row):
            continue
        cost = .5*np.maximum(0., 1.-v[row]/31.29)
        ix = vid[row].astype(int)
        gs[0, i, ix], gv[0, i, ix], gm[0, i, ix] = s[row], v[row], True
        region = np.where(s[row] < 0, 0, np.where(s[row] <= 75.44, 1, 2))
        road[i] = np.bincount(region, weights=cost, minlength=3)
        after = {int(vid[j]): int(j) for j in rows[i+1]}
        for j in row:
            k = after.get(int(vid[j]))
            if k is not None and lane[j] >= 0 and lane[k] in (0,1) and s[j] < 75.44 <= s[k]:
                cross[i] += 1
    values = {}
    grid = fields(gs, gv, gm)
    mq = merge_queue(gs, gv, gm)[0]
    for h in HORIZONS:
        n = h*2
        r = road[:n].sum(0)
        source = float(.5*queues[:n].sum())
        row = rows[n]
        values[str(h)] = dict(J=float(r.sum()+source), source=source,
            upstream=float(r[0]), merge=float(r[1]), discharge=float(r[2]),
            crossings=int(cross[:n].sum()), queue=int(queues[n]),
            merge_queue=int(merge_queue(s[row], v[row], np.ones(len(row), dtype=bool))),
            mean_merge_queue=float(mq[:n].mean()),
            fields={k:np.asarray(value)[0, :h//10].tolist() for k,value in grid.items()},
            endpoints=dict(ids=vid[row].astype(int).tolist(), s=s[row].tolist(), v=v[row].tolist()))
    return dict(seed=seed, arm=arm, values=values, original_ids=original_ids.tolist(),
        initial_kin=initial_kin.tolist(), original_cav=st['state']['is_cav'][0,keep].tolist(),
        initial_state_sha256=st['initial_state_sha256'],
        hashes={name:digest(root/name) for name in ('result.json','traj_base.npz','start.pt','start_history.npz')},
        convention='left-boundary full-present road speed deficit + due-minus-inserted source rectangles')


def main():
    design = json.loads((ROOT/'design.json').read_text())
    rows = []
    for world in design['worlds']:
        for arm in ARMS:
            rows.append(one(world, arm))
        assert len({r['initial_state_sha256'] for r in rows[-len(ARMS):]}) == 1
    print(json.dumps(dict(schema='c1closed_labels/1', source_sha256=digest(Path(__file__)),
        design_sha256=digest(ROOT/'design.json'), rows=rows), allow_nan=False))


if __name__ == '__main__':
    main()
