"""Common-HDV pairing, world bootstrap and attribution exports; no inference."""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def ci(rows):
    if not rows:return dict(n=0,worlds=0,point=None,lo=None,hi=None)
    worlds=sorted({w for w,v in rows})
    groups=[np.asarray([v for w,v in rows if w==world],dtype=float) for world in worlds]
    values=np.concatenate(groups)
    assert np.isfinite(values).all()
    rng=np.random.default_rng(0)
    padded=np.full((len(groups),max(map(len,groups))),np.nan)
    for i,g in enumerate(groups):padded[i,:len(g)]=g
    draws=rng.integers(0,len(groups),size=(2000,len(groups)))
    boot=np.nanmedian(padded[draws].reshape(2000,-1),axis=1)
    lo,hi=np.quantile(boot,[.025,.975])
    return dict(n=len(values),worlds=len(worlds),point=float(np.median(values)),lo=float(lo),hi=float(hi),
        equal_world_mean=float(np.mean([g.mean() for g in groups])))


def config(r):
    return '__'.join((r['tag'],r['history'],'pop' if r['population'] else 'inf',
        'A1' if r['residual'] else 'A0',r['lanes'],r.get('component','baseline')))


def baseline_config(r):
    hist='h5' if r['cohort']=='wall_same' else 'snapshot'
    if r['_panel']=='aux':hist='snapshot'
    return '__'.join((r['tag'],hist,'inf','A1','original','baseline'))


def median_error(row,h,ids=None):
    entries=row['endpoints'][str(h)]['ids']
    ids=set(entries) if ids is None else ids
    return float(np.median([abs(entries[i]['ds']) for i in ids])) if ids else None


def paired(a,b,h,anchor=None):
    ids=set(a['endpoints'][str(h)]['ids']) & set(b['endpoints'][str(h)]['ids'])
    if anchor is not None:ids &= set(anchor['endpoints'][str(h)]['ids'])
    if not ids:return None
    av,bv=median_error(a,h,ids),median_error(b,h,ids)
    return dict(a=av,b=bv,delta=av-bv,n_ids=len(ids))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',required=True);ap.add_argument('--out',default='docs/paper_results/c1pidiag');ap.add_argument('--quiet',action='store_true');args=ap.parse_args()
    path=Path(args.raw);out=ROOT/args.out;assert out.is_dir()
    with gzip.open(path,'rt') as f:rows=[json.loads(line) for line in f]
    assert not any(r['error'] for r in rows)
    keyed={};bb={};groups={}
    for r in rows:
        k=(r['_panel'],r['mode'],r['sid'],config(r));assert k not in keyed,k
        keyed[k]=r
        groups.setdefault((r['_panel'],r['mode'],r['cohort'],config(r)),[]).append(r)
        if r['tag']=='bbtraj':bb[(r['_panel'],r['mode'],r['sid'])]=r
    summary=[];states=[];contrasts=[]
    for (panel,mode,cohort,cfg),rs in sorted(groups.items()):
        result=dict(panel=panel,mode=mode,cohort=cohort,config=cfg,tag=rs[0]['tag'],history=rs[0]['history'],
                    population=rs[0]['population'],residual=rs[0]['residual'],lanes=rs[0]['lanes'],component=rs[0].get('component','baseline'),horizons={})
        for h in (60,120):
            raw=[];common_a=[];common_b=[];delta=[];drift=[];vsbase=[];vssnapshot=[];vsc1=[]
            counts=[]
            for r in rs:
                ref=bb[(panel,mode,r['sid'])]
                value=median_error(r,h)
                if value is not None:raw.append((r['world'],value))
                p=paired(r,ref,h)
                s=dict(panel=panel,mode=mode,cohort=cohort,config=cfg,sid=r['sid'],world=r['world'],horizon=h,
                    own_n=r['endpoints'][str(h)]['n'],truth_n=r['endpoints'][str(h)]['n_truth'],own_error=value,
                    common_n=p['n_ids'] if p else 0,common_error=p['a'] if p else None,
                    bb_common_error=p['b'] if p else None,paired_delta=p['delta'] if p else None)
                states.append(s)
                if p:
                    counts.append(p['n_ids']);common_a.append((r['world'],p['a']));common_b.append((r['world'],p['b']));delta.append((r['world'],p['delta']))
                    common_ids=set(r['endpoints'][str(h)]['ids']) & set(ref['endpoints'][str(h)]['ids'])
                    drift.append((r['world'],float(np.median([r['endpoints'][str(h)]['ids'][i]['ds'] for i in common_ids]))))
                base=keyed.get((panel,mode,r['sid'],baseline_config(r)))
                if base:
                    d=paired(r,base,h,ref)
                    if d:vsbase.append((r['world'],d['delta']))
                snap='__'.join((r['tag'],'snapshot','pop' if r['population'] else 'inf',
                    'A1' if r['residual'] else 'A0',r['lanes'],r.get('component','baseline')))
                base=keyed.get((panel,mode,r['sid'],snap))
                if base:
                    d=paired(r,base,h,ref)
                    if d:vssnapshot.append((r['world'],d['delta']))
                c1hist='h60' if panel=='aux' else ('h5' if cohort=='wall_same' else 'snapshot')
                c1key='__'.join(('c1struct',c1hist,'inf','A1','original','baseline'))
                c1=keyed.get((panel,mode,r['sid'],c1key))
                if c1:
                    d=paired(r,c1,h,ref)
                    if d:vsc1.append((r['world'],d['delta']))
            result['horizons'][str(h)]=dict(own=ci(raw),common_wm=ci(common_a),common_bb=ci(common_b),
                paired_wm_minus_bb=ci(delta),paired_vs_model_baseline=ci(vsbase),paired_vs_snapshot=ci(vssnapshot),
                paired_vs_c1=ci(vsc1),signed_position_drift=ci(drift),
                matched_ids_total=sum(counts),matched_ids_min=min(counts) if counts else 0,
                matched_ids_median=float(np.median(counts)) if counts else None)
            for kind,values in [('vs_baseline',vsbase),('vs_snapshot',vssnapshot),('vs_c1',vsc1)]:
                contrasts.extend(dict(panel=panel,mode=mode,cohort=cohort,config=cfg,horizon=h,contrast=kind,world=w,delta=v) for w,v in values)
        summary.append(result)
    value=dict(schema='c1pidiag_analysis/1',raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        units=len(rows),estimand='median of per-start HDV medians; paired differences formed within start on common HDV IDs',
        bootstrap=dict(draws=2000,seed=0,cluster='world',interval='percentile95'),summary=summary)
    with (out/'SUMMARY.json').open('x') as f:json.dump(value,f,indent=1,allow_nan=False)
    for name,data in [('PER_STATE.csv',states),('ATTRIBUTION_PER_STATE.csv',contrasts)]:
        with (out/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print('units',len(rows),'config groups',len(summary),'state/horizon rows',len(states))
    if args.quiet:return
    for r in summary:
        if r['mode']=='clh' or (r['mode']=='tf' and r['history']==('h5' if r['cohort']=='wall_same' else 'snapshot') and not r['population']) or (r['mode']=='aux_tf' and not r['population']):
            print(r['panel'],r['cohort'],r['mode'],r['config'],
                {h:{k:round(v['point'],2) if v['point'] is not None else None for k,v in r['horizons'][h].items() if isinstance(v,dict) and 'point' in v}
                 for h in ('60','120')})


if __name__=='__main__':main()
