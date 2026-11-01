"""Evaluation-only recorded60s HDV targets for the additional50 worlds."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get('RB2INT_CONTROL_ROOT',str(Path(__file__).resolve().parents[2])))
sys.path[:0] = [str(ROOT),str(ROOT/'src')]

import numpy as np
import torch

from tools.rbint.endpoint_targets import (ARMS, causal_lane_carry, match_targets_with_carry,
                                          safe, sha)


def extract(views, starts, worlds):
    views, starts = safe(views), safe(starts)
    episodes = []
    for seed in worlds:
        if not 969550 <= seed <= 969599:
            raise ValueError('additional registered50 only; original targets are reused')
        manifest_path = views/f'world_{seed}.json'
        manifest = json.loads(safe(manifest_path).read_text())
        job = starts/f's{seed}_ctrl0'
        receipt = json.loads(safe(job/'result.json').read_text())
        for name in ('start.pt','start_history.npz'):
            if sha(job/name) != receipt['output_hashes'][name]:
                raise ValueError('causal start hash differs')
        start = torch.load(safe(job/'start.pt'),map_location='cpu',weights_only=False)
        if start['seed'] != seed or start['t0'] != 300 or start['initial_state_sha256'] != receipt['metrics']['initial_state_sha256']:
            raise ValueError('factual target world/start differs')
        with np.load(safe(job/'start_history.npz'),allow_pickle=False) as history:
            ids = np.array(history['vid_index'][-1],copy=True)
            carry = causal_lane_carry(history['kin'],history['mask'].astype(bool),history['vid_index'])
        start = {**start,'observed_lane_carry':carry}
        for arm in ARMS:
            name = manifest[arm]['episode']
            if not name.startswith(f'w{seed}m') or not name.endswith('__'+arm) or '/' in name or '\\' in name:
                raise ValueError('unexpected converted observable episode name')
            ep = safe(views/'episodes'/name)
            meta = json.loads(safe(ep/'episode.json').read_text())
            if meta['seed'] != seed or meta['arm'] != arm:
                raise ValueError('factual converted episode identity differs')
            files = {n:safe(ep/(n+'.npy')) for n in ('time_s','veh_kin','veh_mask','vid','is_cav')}
            arrays = {n:np.load(p,mmap_mode='r',allow_pickle=False) for n,p in files.items()}
            ticks = [np.flatnonzero(arrays['time_s'] == t) for t in (300.,360.)]
            if any(len(t) != 1 for t in ticks):
                raise ValueError('exact300/360s factual frames required')
            a,b = (int(t[0]) for t in ticks)
            vid = arrays['vid']
            observed_carry = causal_lane_carry(arrays['veh_kin'][:a+1],arrays['veh_mask'][:a+1].astype(bool),
                vid[:a+1] if vid.ndim == 2 else vid)
            rows = match_targets_with_carry(start,ids,arrays['veh_kin'][a],arrays['veh_mask'][a].astype(bool),
                vid[a] if vid.ndim == 2 else vid,arrays['is_cav'][a].astype(bool),
                arrays['veh_kin'][b],arrays['veh_mask'][b].astype(bool),
                vid[b] if vid.ndim == 2 else vid,arrays['is_cav'][b].astype(bool),observed_carry)
            episodes.append(dict(seed=seed,arm=arm,initial_sha256=start['initial_state_sha256'],
                start_sha256=sha(job/'start.pt'),history_sha256=sha(job/'start_history.npz'),
                manifest_sha256=sha(manifest_path),metadata_sha256=sha(ep/'episode.json'),
                files_sha256={n:sha(p) for n,p in files.items()},vehicles=rows))
        print('FACTUAL_WORLD',seed,flush=True)
    return dict(schema='wmphys-factual-endpoints/1',horizon_s=60,t0_s=300,episodes=episodes,
        evaluation_only=True,model_inputs=False,training=False,sumo_runs=0,
        matching='unique exact observed kinematics/ownership/junction carry then persistent converted VID')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--views',default='/root/wmdata2/views')
    parser.add_argument('--starts',required=True)
    parser.add_argument('--out',required=True)
    args = parser.parse_args()
    out = safe(args.out)
    if not out.parent.is_dir() or out.exists():
        raise ValueError('exclusive factual target output required')
    result = extract(args.views,args.starts,range(969550,969600))
    result.update(source_sha256=sha(__file__),helper_sha256=sha(ROOT/'tools/rbint/endpoint_targets.py'))
    with out.open('x') as stream:
        json.dump(result,stream,indent=1,allow_nan=False)
    print('FACTUAL50 COMPLETE',len(result['episodes']),flush=True)


if __name__ == '__main__':
    main()
