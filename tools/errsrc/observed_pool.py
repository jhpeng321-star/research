"""Encode every identity observed in the supplied pre-t0 60s history, including exits.

Only observational arrays are accessed. A cutoff wrapper prevents future-array
access. Each identity contributes ONCE, at its last observed historical tick.
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import torch


def pool(model,raw,cutoff,cache_root):
    from wms.data.pack_history import prior_history
    from wms.model.driver_encoder import PARAMETER_NAMES
    from tools.lrdiag.replay import WM_SHA
    raw=Path(raw);assert not any(x in str(raw).lower() for x in
        ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth'))
    key=hashlib.sha256(f'errsrc-observed-pool-v1:{WM_SHA}:{raw}:{cutoff}'.encode()).hexdigest()
    cache_root=Path(cache_root);cache_root.mkdir(exist_ok=True)
    cache=cache_root/(key+'.npy')
    if cache.is_file():return torch.from_numpy(np.load(cache,allow_pickle=False).copy())
    names=('veh_kin','veh_mask','target_mask','vid')
    arrays={name:np.load(raw/(name+'.npy'),mmap_mode='r')[:cutoff+(name!='target_mask')]
            for name in names}
    lower=max(0,cutoff-120)
    class Prefix:
        def array(self,name):return arrays[name]
        def run_start(self,tick):return lower
    prefix=Prefix();last={}
    for tick in range(lower,cutoff+1):
        valid=arrays['veh_mask'][tick].astype(bool)&(arrays['veh_kin'][tick,:,1:4].sum(-1)>0)
        for slot in np.flatnonzero(valid):
            identity=int(arrays['vid'][tick,slot]);assert identity>=0
            last[identity]=(tick,int(slot))
    groups={}
    for identity,(tick,slot) in last.items():groups.setdefault(tick,[]).append((identity,slot))
    values={}
    for tick,members in sorted(groups.items()):
        kin=torch.from_numpy(np.array(arrays['veh_kin'][tick],copy=True))[None]
        present=arrays['veh_mask'][tick].astype(bool)&(arrays['veh_kin'][tick,:,1:4].sum(-1)>0)
        hist,mask=prior_history(prefix,tick,present,history_s=60.,history_stride_s=1.)
        with torch.no_grad():
            _,_,params=model.encode_drivers({'veh_kin':kin,'present':torch.from_numpy(present)[None]},
                driver_history=torch.from_numpy(hist)[None],driver_history_mask=torch.from_numpy(mask)[None])
        assert params is not None
        for identity,slot in members:
            values[identity]=torch.stack([params[n][0,slot] for n in PARAMETER_NAMES]).detach()
    assert values,'no observable pre-t0 identities'
    result=torch.stack([values[i] for i in sorted(values)])
    temporary=cache.with_suffix(f'.{os.getpid()}.tmp')
    with temporary.open('wb') as out:np.save(out,result.numpy(),allow_pickle=False)
    temporary.replace(cache)
    arrays.clear()  # Release mmap references retained by the local adapter class.
    return result


def group_pools(model,home,group,panel):
    import json
    root=home/('fitdata' if panel=='fit' else 'e3data')
    manifest=json.loads((root/'C4_EPISODES_MANIFEST.json').read_text())
    keys=sorted({e['pair_key'] for e in manifest['episodes']})
    episodes=[e for e in manifest['episodes'] if e['pair_key']==keys[group]]
    order={'native':0,'mappo':1,'ours_c1':2}
    episodes.sort(key=lambda e:order[e['arm']])
    return [pool(model,root/e['dir'],e['t0_index'],home/'pool_cache') for e in episodes]
