"""Registered pairs300 panel: diagnostic only, never E3 verdict evidence."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from summarize import clean,interval,metrics,write_csv,VARIANTS


def main(root,out):
    bins=defaultdict(list)
    for f in sorted((root/'results').glob('p300_*_s600.json')):
        d=json.loads(f.read_text());assert d['panel']=='pairs300' and d['role']=='diagnostic'
        bins[d['pair_key'],d['variant']].append(d)
    assert len(bins)==48*6,len(bins)
    rows=[]
    for (key,v),ds in sorted(bins.items()):
        assert len(ds)==8 and {d['draw'] for d in ds}==set(range(8))
        # JSON preserves native-reference-first insertion order.
        ref=next(iter(ds[0]['arms']))
        for arm in ds[0]['arms']:
            if arm==ref:continue
            for h in ('120','300'):
                pairs=np.array([np.array(d['arms'][arm]['delay'][h])-d['arms'][ref]['delay'][h] for d in ds])
                assert np.ptp(pairs[:,1])<1e-8
                y,x=pairs.mean(0)
                rows.append({'scope':'pairs300_non_verdict','world':key.split('@')[0],'pair_key':key,
                    'variant':v,'arm':arm,'horizon_s':int(h),'wm_dJ':y,'real_dJ':x,'paired_bias':y-x,
                    'wm_sd_K':pairs[:,0].std(ddof=1),'sign_agreement':float(np.sign(y)==np.sign(x)) if x!=0 else None})
    summary=[];within=[]
    for v in VARIANTS:
        for h in (120,300):
            sel=[r for r in rows if r['variant']==v and r['horizon_s']==h]
            worlds=sorted({r['world'] for r in sel})
            groups=[np.array([[r['wm_dJ'],r['real_dJ']] for r in sel if r['world']==w]) for w in worlds]
            point=metrics(np.concatenate(groups));rng=np.random.default_rng(20261028)
            bootstrap=[metrics(np.concatenate([groups[i] for i in rng.integers(0,len(worlds),len(worlds))])) for _ in range(2000)]
            row={'scope':'pairs300_non_verdict','variant':v,'horizon_s':h,'n_groups':48,'n_worlds':len(worlds),'K':8,**point}
            for k in point:
                lo,hi=interval([r[k] for r in bootstrap]);row[k+'_lo95']=lo;row[k+'_hi95']=hi
            summary.append(row)
            for w,a in zip(worlds,groups):
                x=a[:,1];y=a[:,0]
                within.append({'world':w,'variant':v,'horizon_s':h,'n_pairs':len(a),
                    'through_origin_slope':float(x@y/(x@x)) if x@x else None})
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'pairs300_worlds.csv',rows);write_csv(out/'pairs300_summary.csv',summary)
    write_csv(out/'pairs300_within_world_slopes.csv',within)
    (out/'PAIRS_COVERAGE.json').write_text(json.dumps({'scope':'non_verdict','n_group_variants':len(bins),
        'results':sum(map(len,bins.values())),'K':8,'expected':48*6*8},indent=1))
    print('PASS pairs300:',len(rows),'paired arm-horizon-variant rows')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();main(a.root,a.out)
