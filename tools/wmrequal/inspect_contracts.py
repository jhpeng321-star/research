from pathlib import Path
import json
import numpy as np
base = Path('/root/autodl-tmp/c4')
for rel in ('overnight20261028/W5_COMBINED/code/score.py',
            'cct2mix20261029/score_mix.py', 'cct2mix20261029/RECEIPT.json',
            'overnight20261028/W5_COMBINED/DESIGN.json'):
    p=base/rel
    print('FILE', p, p.is_file())
    if p.is_file():
        text=p.read_text()
        if p.name == 'DESIGN.json':
            d=json.loads(text);print('CONTROLLERS',d.get('controllers'))
        else:print(text[:22000])
for root in (base/'e3data', base/'panel120/package', base/'wmdeep20261027/score_pairs'):
    d=json.loads((root/'C4_EPISODES_MANIFEST.json').read_text())
    es=[e for e in d['episodes'] if (e.get('e3data_split') in ('validation','calibration')
        or root.name in ('package','score_pairs'))]
    es=es[:1] if root.name=='score_pairs' else es
    rows=[]
    for e in es:
        ep=root/e['dir'];ts=np.load(ep/'time_s.npy',mmap_mode='r')
        t=int(np.searchsorted(ts,e['warmup_s']))
        vid=np.load(ep/'vid.npy',mmap_mode='r');m=np.load(ep/'veh_mask.npy',mmap_mode='r').astype(bool)
        c=np.load(ep/'counter.npy',mmap_mode='r')[t].sum(-1)
        maximum=int(vid[:t+1][m[:t+1]].max())+1
        rows.append([e['episode'],maximum,c.tolist(),float(c.sum()-maximum)])
    print('COUNTER_REGISTRY',root,'n',len(rows),'minmax_diff',
          (min(r[3] for r in rows),max(r[3] for r in rows)), 'examples',rows[:3])
