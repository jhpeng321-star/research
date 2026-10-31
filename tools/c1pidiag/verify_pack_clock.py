"""Check nominal pack origins against actual observed snapshot kinematics."""
from pathlib import Path
import json
import numpy as np
from operate import ROOT, ssh, allowed


def main():
    panel=json.loads((ROOT/'docs/paper_results/c1pidiag/PANEL.json').read_text())['states']
    sample=[panel[0],next(e for e in panel if e['cohort']=='wall_same')]
    code=f'''from pathlib import Path
import gzip,json
p=Path('/root/autodl-tmp/c4/c1refresh/states/states')
out={{}}
for sid in {[e['sid'] for e in sample]!r}:
 with gzip.open(p/(sid+'.json.gz'),'rt') as f:r=json.load(f)
 out[sid]=[(v['s'],v['v']) for v in r['initial_state']['vehicles'] if v.get('present',True) and v['lane']>=0]
print(json.dumps(out))
'''
    observed=json.loads(ssh(code))
    cache=Path('E:/research-dense/data_cache/pack_v3')
    idx=json.loads((cache/'INDEX.json').read_text())
    items=next(v for v in idx.values() if isinstance(v,list) and v and isinstance(v[0],dict) and 'tick_start' in v[0])
    by_key={r['key']:r for r in items}
    kin=np.load(cache/'veh_kin.npy',mmap_mode='r',allow_pickle=False)
    mask=np.load(cache/'veh_mask.npy',mmap_mode='r',allow_pickle=False)
    out=[]
    for e in sample:
        allowed(e['key']);item=by_key[e['key']]
        snapshot=set(map(tuple,observed[e['sid']]))
        values=[]
        for offset in (0,1,9,10,120,600):
            if offset>=item['n_ticks']:continue
            k=item['tick_start']+offset
            pairs={(float(v[0]),float(v[4])) for v in kin[k][mask[k].astype(bool)]}
            values.append(dict(offset=offset,n=len(pairs),matches=len(snapshot & pairs),snapshot_n=len(snapshot)))
        out.append(dict(sid=e['sid'],key=e['key'],candidates=values))
    with (ROOT/'docs/paper_results/c1pidiag/PACK_CLOCK_CHECK_v2.json').open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
