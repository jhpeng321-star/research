"""Reproduce old target through its canonical loader, CPU/read-only arrays."""
from pathlib import Path
import sys
import json
import numpy as np

p=Path('/root/autodl-tmp/c4/wmrequal20261030')
rt=Path('/root/autodl-tmp/c4/restart20261026/runtime_longint')
sys.path[:0]=[str(p),str(p/'tools/wmrequal'),str(p/'src'),str(rt)]
import torch
torch.set_num_threads(1)
from real_panel import entries_for
from score import source_for,bootstrap_runtime
e3=bootstrap_runtime()
real=json.loads((p/'REAL.json').read_text())['panels']
out={}
initial_counts=[]
for panel in ('validation','pairs300','calibration','panel120'):
    source,indexes=source_for(panel,real[panel])
    rows=[]
    for gi in indexes:
        key,ref,arms=e3.group_rows(source,gi)
        for arm,row in [('native',ref),*arms]:
            rec=source.dataset[row]
            states=np.asarray(rec['states'])
            present=np.asarray(rec['present']).astype(bool)
            cav=np.asarray(rec['is_cav']).astype(bool)
            ep_index,tick=source.dataset.positions[row]
            ep=source.dataset._loader.episode(ep_index)
            raw_count=int(np.asarray(ep.array('veh_mask')[tick]).astype(bool).sum())
            initial_counts.append(int(present[0].sum())-raw_count)
            sel=present[0] & ~cav[0]
            values={str(h):float(.5*np.clip(1-states[1:h*2+1,:,4]/31.29,0,None)[
                present[1:h*2+1] & sel[None]].sum()) for h in (60,90,120,180,300) if h*2 < len(states)}
            rows.append(dict(pair_key=key,arm=arm,values=values))
    out[panel]=rows
    print(panel,len(rows),flush=True)
(p/'LEGACY_REAL.json').write_text(json.dumps(out,indent=1))
old=json.loads((p.parent/'cct2mix20261029/WM2mix_validation.json').read_text())
lookup={(r['pair_key'],r['arm']):r for r in out['validation']}
errors=[]
for r in old['rows']:
 for h in ('60','90','120','180','300'):
  for arm,pair in zip(r['arm_names'],r['dJ'][h]):
   recomputed=lookup[r['pair_key'],arm]['values'][h]-lookup[r['pair_key'],'native']['values'][h]
   errors.append(abs(recomputed-pair[1]))
print('LEGACY_REAL_MAX_ERROR',max(errors))
print('INITIAL_POPULATION_COUNT_DIFFERENCE_RANGE',min(initial_counts),max(initial_counts))
