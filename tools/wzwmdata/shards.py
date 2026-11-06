"""Incremental WM-data shards for the WM fine-tune (WM-B -> WM-E), zero physics.

    python -m tools.wzwmdata.shards publish --run-root <run> --out <shards dir> [--min-worlds 24] [--final]
        [--canary-root <cpuV canary run copy>]

A shard is a set of COMPLETE WMDATA worlds (all four arms finished with RESULT.json, 0 collisions /
teleports / dropped slot-steps) not contained in an earlier shard. ``SHARD_<k>.json`` lists every
episode directory (the c4_sumo_episode/1 tape dir ``<job>/episode``) with its seed, cell, arm and the
world-grouped split. The split is registered before collection and does not depend on timing:
VAL = in every cell, the seed with the smallest sha256('wzwmdata_val_v1:<seed>') (12 worlds);
TRAIN = the other 88 WMDATA seeds. WMQUAL tapes are never listed in a shard (WM-D hold-out).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from . import registry as reg

ARMS_NEEDED = 4
VAL_KEY = 'wzwmdata_val_v1'


def val_seeds():
    per = {}
    for s, cell in reg.wmdata_assignment().items():
        per.setdefault(cell, []).append(s)
    return sorted(min(seeds, key=lambda x: hashlib.sha256(f'{VAL_KEY}:{x}'.encode()).hexdigest())
                  for seeds in per.values())


def split_of(seed):
    if any(lo <= seed <= hi for lo, hi in reg.WMD_RESERVED) or reg.WMQUAL_RANGE[0] <= seed <= reg.WMQUAL_RANGE[1]:
        raise ValueError(f'{seed} is a WM qualification hold-out; never train/val')
    return 'val' if seed in set(val_seeds()) else 'train'


def _worlds(run_root, canary_root=None):
    """{seed: {arm: entry}} of finished WMDATA jobs (RESULT.json present)."""
    out = {}
    roots = [Path(run_root)] + ([Path(canary_root)] if canary_root else [])
    for root in roots:
        plan = json.loads((root / 'PLAN.json').read_text(encoding='utf-8'))
        jobs = list(plan['jobs'])
        for amend in sorted(root.glob('PLAN_AMEND_*.json')):
            jobs.extend(json.loads(amend.read_text(encoding='utf-8'))['jobs'])
        for job in jobs:
            if job['block'] != 'WMDATA':
                continue
            p = Path(job['path'])
            if canary_root and root == Path(canary_root):         # cpuV paths -> local copy
                p = Path(canary_root) / Path(job['path']).relative_to(plan_root(plan))
            res = p / 'RESULT.json'
            if not res.exists():
                continue
            r = json.loads(res.read_text(encoding='utf-8'))
            if r.get('status') != 'ok':
                continue
            t = r['totals']
            safe = not any(t.get(k, 0) for k in ('collisions', 'teleports', 'dropped_vehicle_steps'))
            arm = job['arm'].replace('_u0', '')
            seed = int(job['world']['seed'])
            out.setdefault(seed, {})[arm] = dict(
                episode_dir=str(p / 'episode'), job=job['id'], arm_tag=job['arm'], seed=seed,
                cell_key=job['world']['cell_key'], demand=job['world']['demand'],
                penetration=job['world']['penetration'], safe=safe,
                episode_npz_sha256=r.get('episode_npz_sha256'), host_root=str(root),
                canary=bool(canary_root and root == Path(canary_root)))
    return out


def plan_root(plan):
    first = Path(plan['jobs'][0]['path'])
    for parent in first.parents:
        if parent.name in ('wmdata', 'wmqual'):
            return parent.parent
    raise ValueError('cannot locate the plan data root')


def publish(run_root, out, *, min_worlds=24, final=False, canary_root=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / 'SHARDS_INDEX.json'
    index = json.loads(index_path.read_text(encoding='utf-8')) if index_path.exists() else dict(shards=[])
    taken = {s for sh in index['shards'] for s in sh['seeds']}
    worlds = _worlds(run_root, canary_root)
    complete = sorted(s for s, arms in worlds.items()
                      if len(arms) == ARMS_NEEDED and all(e['safe'] for e in arms.values()) and s not in taken)
    if not complete or (len(complete) < min_worlds and not final):
        return dict(published=None, complete_new=len(complete), taken=len(taken))
    k = len(index['shards']) + 1
    episodes = []
    for s in complete:
        for arm, e in sorted(worlds[s].items()):
            episodes.append(dict(e, split=split_of(s)))
    body = dict(schema='wzwmdata.shard.v1', shard=k, created_unix=time.time(), map=reg.MAP,
                seeds=complete, n_worlds=len(complete), n_episodes=len(episodes),
                n_train_worlds=sum(split_of(s) == 'train' for s in complete),
                n_val_worlds=sum(split_of(s) == 'val' for s in complete),
                split_rule=f"world-grouped by seed: VAL = per cell min sha256('{VAL_KEY}:<seed>') "
                           "(12 of 100 WMDATA seeds), registered before collection",
                val_seeds_all=val_seeds(), episodes=episodes,
                notes='canary episodes (cpuV, seed 989103) have arm tags mappo/coop_mappo and map identity '
                      'only in manifest.json (meta_json lacks map); all others carry map in meta_json.')
    body['shard_sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    path = out / f'SHARD_{k:02d}.json'
    with path.open('x', encoding='utf-8') as f:
        f.write(json.dumps(body, indent=1, sort_keys=True))
        f.flush()
        os.fsync(f.fileno())
    index['shards'].append(dict(shard=k, path=str(path), seeds=complete, n_episodes=len(episodes),
                                sha256=body['shard_sha256'], unix=body['created_unix']))
    tmp = index_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(index, indent=1, sort_keys=True), encoding='utf-8')
    tmp.replace(index_path)
    return dict(published=str(path), n_worlds=len(complete), n_episodes=len(episodes))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m tools.wzwmdata.shards')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('publish')
    p.add_argument('--run-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--min-worlds', type=int, default=24)
    p.add_argument('--final', action='store_true')
    p.add_argument('--canary-root')
    a = ap.parse_args(argv)
    print(json.dumps(publish(a.run_root, a.out, min_worlds=a.min_worlds, final=a.final,
                             canary_root=a.canary_root)))


if __name__ == '__main__':
    main()
