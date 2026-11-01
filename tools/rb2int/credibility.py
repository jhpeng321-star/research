"""Registered decision/coverage definitions with strict world-cluster ratio bands."""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.rb2int.predictions import read_model
from tools.rb2int.uncertainty import PairedWorldBootstrap
from tools.rbeval.core import ARMS, HS, REGIONS, load, digest, save_new
from tools.wmdecide.rbeval_adapter import observations


def ratio_bands(bootstrap, features):
    """features=(rows,field): preserve zero-selected scored worlds in denominators."""
    totals = np.zeros((len(bootstrap.worlds),len(features)))
    counts = np.zeros_like(totals)
    index = {s:i for i,s in enumerate(bootstrap.worlds)}
    for j,(rows,field) in enumerate(features):
        for row in rows:
            value = row.get(field)
            if value is not None and np.isfinite(value):
                totals[index[row['seed']],j] += value
                counts[index[row['seed']],j] += 1
    denom = bootstrap.weights@counts
    draws = np.divide(bootstrap.weights@totals,denom,
        out=np.full_like(denom,np.nan),where=denom>0)
    n = counts.sum(0)
    point = np.divide(totals.sum(0),n,out=np.full(len(features),np.nan),where=n>0)
    sd = np.nanstd(draws,axis=0,ddof=1)
    valid = np.isfinite(draws).all(0) & np.isfinite(point) & (sd>1e-14) & bootstrap.replicated & ((counts>0).sum(0)>=2)
    critical = float(np.quantile(np.abs((draws[:,valid]-point[valid])/sd[valid]).max(1),.95)) if valid.any() else None
    return [dict(mean=float(point[j]) if n[j] else None,count=int(n[j]),
        contributing_worlds=int((counts[:,j]>0).sum()),low=float(point[j]-critical*sd[j]) if valid[j] else None,
        high=float(point[j]+critical*sd[j]) if valid[j] else None,uncertainty_available=bool(valid[j]),
        family_size=len(features),maxstat_critical=critical,
        uncertainty='fixed-cell paired whole-world row-ratio bootstrap; conditional on stored K8') for j in range(len(features))]


def summarize(data,metadata):
    boot = PairedWorldBootstrap(metadata)
    features,labels = [],[]
    for h in HS:
        selected = [r for r in data['events'] if r['target']=='selected_preference' and r['horizon_s']==h]
        for r in selected:
            r['calibration_gap'] = r['outcome']-r['probability']
        for field in ('probability','outcome','calibration_gap','brier','real_tie'):
            features.append((selected,field));labels.append(dict(target='selected_preference',horizon_s=h,metric=field))
        for target in ('preference','breakdown'):
            rows = [r for r in data['events'] if r['target']==target and r['horizon_s']==h and
                    (target!='breakdown' or r['region']=='upstream')]
            features.append((rows,'brier'));labels.append(dict(target=target,horizon_s=h,metric='brier'))
        for target in ('delay','paired_delay'):
            for level in (.5,.8,.9,None):
                rows = [r for r in data['coverage'] if r['target']==target and r['horizon_s']==h and r['nominal_level']==level]
                features.append((rows,'covered'));labels.append(dict(target=target,horizon_s=h,metric='covered',
                    interval='range' if level is None else 'central',nominal_level=level,
                    continuous_range_reference=7/9 if level is None else None))
        choices = [r for r in data['decisions'] if r['horizon_s']==h]
        features.append((choices,'switched'));labels.append(dict(target='decision',horizon_s=h,metric='switch_fraction'))
    summaries = [dict(label,**band) for label,band in zip(labels,ratio_bands(boot,features))]
    for row in summaries:
        if row['low'] is not None:
            lower,upper = (-1.,1.) if row['metric']=='calibration_gap' else (0.,1.)
            row['low'],row['high'] = max(lower,row['low']),min(upper,row['high'])
    reliability = []
    for h in HS:
        for target in ('preference','selected_preference'):
            rows = [r for r in data['events'] if r['target']==target and r['horizon_s']==h]
            for b in range(10):
                subset = [r for r in rows if min(int(r['probability']*10),9)==b]
                reliability.append(dict(target=target,horizon_s=h,bin=b,count=len(subset),
                    worlds=len({r['seed'] for r in subset}),
                    forecast=float(np.mean([r['probability'] for r in subset])) if subset else None,
                    observed=float(np.mean([r['outcome'] for r in subset])) if subset else None,
                    qualification='point diagram only; no independent-row or sparse-bin CI'))
    return summaries,reliability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',required=True)
    parser.add_argument('--real',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--scope',choices=('primary54','all58','reserved8'),default='primary54')
    parser.add_argument('--out',required=True)
    args = parser.parse_args()
    manifest = load(args.runs);registration=load(manifest['registration_path'])
    worlds = registration[dict(primary54='replicated_primary_worlds',all58='final_comparison_worlds',reserved8='reserved8')[args.scope]]
    real = load(args.real)['episodes']
    pred,audit = read_model(args.runs,args.model,real,worlds,partial=False)
    data = observations(pred,real,ARMS,HS,REGIONS)
    for rows in data.values():
        for r in rows: r['split']=args.scope
    by = {r['seed']:dict(seed=r['seed'],demand=r['demand'],penetration=r['penetration']) for r in real}
    summary,reliability = summarize(data,[by[s] for s in worlds])
    out = Path(args.out)
    if not out.parent.is_dir() or out.exists(): raise ValueError('exclusive credibility output root required')
    out.mkdir()
    save_new(out/'CREDIBILITY.json',dict(model=args.model,scope=args.scope,worlds=worlds,
        summary=summary,reliability=reliability,input_audit=audit,real_sha256=digest(args.real),
        bootstrap=10000,bootstrap_seed=20261101,source_sha256=digest(__file__),
        policy='strict paired probability>=.8 plus lower expected delay; no probability-map transplant'))
    save_new(out/'OBSERVATIONS.json',data)
    print(json.dumps(dict(out=str(out),worlds=len(worlds),model=args.model)))


if __name__ == '__main__':
    main()
