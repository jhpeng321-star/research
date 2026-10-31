"""K3 diagnostic using S-RB-EVAL's exact S-WMBREAKDOWN detector/reductions.

Sparse-cell outputs are descriptive, never a simultaneous qualification PASS.
No future outcomes are read by the simulation worker.
"""
from collections import defaultdict
from pathlib import Path
import csv
import json
import numpy as np
from eval_core import (crps, reduce_trajectory,attach_events,onset_vector,breakdown,digest,wilson)
from admission import amendments,effective

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/ubuntu/sumocore20261031')
HS=(120,300,600)
REGIONS=('upstream','merge','ramp')


def avg(x):
    a=[v for v in x if v is not None and np.isfinite(v)]
    return float(np.mean(a)) if a else None


def save(path,data):
    with path.open('x') as f:json.dump(data,f,indent=1,allow_nan=False)


def csvout(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)


def bootstrap_family(features,B=10000):
    """Common stratified whole-world resamples + family standardized max-stat.

    Descriptive aggregate bands only; singleton demand cells remain unqualified.
    Finite K Monte Carlo uncertainty is not separately identified by these bands.
    """
    worlds=sorted({w for f in features for w in f})
    rng=np.random.default_rng(20261031)
    weights=np.zeros((B,len(worlds)))
    for reserved in (False,True):
        ix=[i for i,w in enumerate(worlds) if (w>=969432)==reserved]
        if ix:weights[:,ix]=rng.multinomial(len(ix),np.ones(len(ix))/len(ix),size=B)
    matrix=np.array([[f.get(w,np.nan) for f in features] for w in worlds])
    present=np.isfinite(matrix);denom=weights@present
    draws=np.divide(weights@np.nan_to_num(matrix),denom,out=np.full_like(denom,np.nan),where=denom>0)
    means=np.nanmean(matrix,axis=0);sd=np.nanstd(draws,axis=0,ddof=1)
    valid=(present.sum(0)>=2)&(sd>1e-12)&np.isfinite(draws).all(0)
    critical=float(np.quantile(np.max(abs((draws[:,valid]-means[valid])/sd[valid]),axis=1),.95)) if valid.any() else None
    return [dict(mean=float(means[j]),worlds=int(present[:,j].sum()),
                 low=float(means[j]-critical*sd[j]) if valid[j] else None,
                 high=float(means[j]+critical*sd[j]) if valid[j] else None,
                 simultaneous_family_critical=critical,world_bootstrap_samples=B,
                 finite_K_uncertainty_included=False) for j in range(len(features))]


def by_world(rows,fn):
    grouped=defaultdict(list)
    for r in rows:
        v=fn(r)
        if v is not None and np.isfinite(v):grouped[r['world']].append(v)
    return {w:float(np.mean(v)) for w,v in grouped.items()}


def main():
    plan=json.loads((ROOT/'plan.json').read_text())
    # Full panel only. Neither a failed attempt nor a partial K is promoted.
    assert len(plan['jobs'])==144
    real=json.loads((HERE/'real_observables.json').read_text())['episodes']
    lookup={(r['seed'],r['arm']):r for r in real}
    forecasts=defaultdict(list)
    attempts=[]
    mapping=amendments()
    for original in plan['jobs']:
        job=effective(original,mapping)
        d=ROOT/'jobs'/job['id']
        result_path=d/'result.json' if (d/'result.json').exists() else d/'recovered_result.json'
        result=json.loads(result_path.read_text())
        assert result['ok'] and result['actual_sim_s']==600
        assert result['job']==job
        r=lookup[job['world'],job['arm']]
        with np.load(d/'trajectory.npz',allow_pickle=False) as z:
            bins=reduce_trajectory(z['t'],z['s'],z['v'],z['lane'],np.ones_like(z['t'],bool))
        bins=attach_events(bins,r)
        f=dict(job=job,bins=bins,cumulative=result['cumulative'])
        forecasts[job['world'],job['variant'],job['arm']].append(f)
        attempts.append(dict(id=job['id'],actual_sim_s=result['actual_sim_s'],
                             scientific_valid=result.get('scientific_valid',True),
                             result_sha256=digest(result_path),trajectory_sha256=digest(d/'trajectory.npz')))
    rows=[];speed_rows=[];event_rows=[];effects=[]
    for (world,variant,arm),fs in forecasts.items():
        fs.sort(key=lambda f:f['job']['rng']);assert len(fs)==3
        r=lookup[world,arm]
        split='development6' if world<969432 else 'reserved2'
        pools=[e for e in real if e['seed']<969432 and e['seed']!=world and e['arm']==arm]
        cells=[e for e in pools if (e['demand'],e['penetration'])==(r['demand'],r['penetration'])]
        for h in HS:
            common=dict(world=world,split=split,variant=variant,arm=arm,horizon=h,
                        demand=r['demand'],penetration=r['penetration'])
            y=r['clocks'][str(h)]['team_delay_vs']
            xs=[f['cumulative'][str(h)]['team'] for f in fs]
            rank_rng=np.random.default_rng(20261031+world+h+sum(ord(c) for c in variant+arm))
            pit=(sum(x<y for x in xs)+float(rank_rng.random())*(1+sum(x==y for x in xs)))/4
            rows.append(dict(common,real=y,predicted_mean=avg(xs),bias=avg(xs)-y,
                crps=crps(xs,y),fair_crps=crps(xs,y,True),range_covered=int(min(xs)<=y<=max(xs)),
                randomized_rank_pit=pit,
                future_sd=float(np.std(xs,ddof=1)),
                pooled_climatology=crps([e['clocks'][str(h)]['team_delay_vs'] for e in pools],y),
                cell_climatology=crps([e['clocks'][str(h)]['team_delay_vs'] for e in cells],y),
                cell_baseline_worlds=len(cells),
                predicted_source=avg([f['cumulative'][str(h)]['source'] for f in fs]),
                real_source=r['clocks'][str(h)]['source_wait_vs'],
                predicted_road=avg([f['cumulative'][str(h)]['road'] for f in fs]),
                real_road=r['clocks'][str(h)]['road_delay_vs']))
            for reg in REGIONS:
                real_event=breakdown(r['bins'][reg]['speed'][:6+h//10])
                ev=[f['bins'][reg]['events'][str(h)] for f in fs]
                pred=[onset_vector(e,h) for e in ev]
                target=onset_vector(real_event,h)
                good=target is not None and all(x is not None for x in pred)
                both=[abs(e['onset_s']-real_event['onset_s']) for e in ev
                      if e['event'] and real_event['event']]
                event_rows.append(dict(common,region=reg,real_event=real_event['event'],
                    event_probability=avg([int(e['event']) if e['event'] is not None else None for e in ev]),
                    real_preexisting=bool(real_event.get('preexisting',False)),
                    preexisting_probability=avg([int(e.get('preexisting',False)) for e in ev]),
                    real_onset=real_event['onset_s'],both_event_onset_mae=avg(both),
                    onset_ibs=float(np.mean((np.mean(pred,axis=0)-target)**2)) if good else None))
                scores=[];fair=[];pooled=[];cell=[];coverage=[];missing=0
                for b in range(h//10):
                    t=6+b;y_speed=r['bins'][reg]['speed'][t]
                    x_speed=[f['bins'][reg]['speed'][t] for f in fs]
                    if y_speed is None or any(x is None for x in x_speed):missing+=1;continue
                    scores.append(crps(x_speed,y_speed));fair.append(crps(x_speed,y_speed,True))
                    pooled.append(crps([e['bins'][reg]['speed'][t] for e in pools if e['bins'][reg]['speed'][t] is not None],y_speed))
                    cell.append(crps([e['bins'][reg]['speed'][t] for e in cells if e['bins'][reg]['speed'][t] is not None],y_speed))
                    coverage.append(int(min(x_speed)<=y_speed<=max(x_speed)))
                speed_rows.append(dict(common,region=reg,crps=avg(scores),fair_crps=avg(fair),
                    pooled_climatology=avg(pooled),cell_climatology=avg(cell),range_covered=avg(coverage),
                    scored_bins=len(scores),missing_bins=missing))
            if arm!='native':
                native=forecasts[world,variant,'native'];native.sort(key=lambda f:f['job']['rng'])
                ys= y-lookup[world,'native']['clocks'][str(h)]['team_delay_vs']
                shift=[f['cumulative'][str(h)]['team']-n['cumulative'][str(h)]['team'] for f,n in zip(fs,native)]
                effects.append(dict(common,real_shift=ys,expected_shift=avg(shift),
                    effect_crps=crps(shift,ys),fair_effect_crps=crps(shift,ys,True),
                    upstream_real_probability_shift=int(breakdown(r['bins']['upstream']['speed'][:6+h//10])['event'])-
                        int(breakdown(lookup[world,'native']['bins']['upstream']['speed'][:6+h//10])['event']),
                    upstream_predicted_probability_shift=avg([int(f['bins']['upstream']['events'][str(h)]['event'])-
                        int(n['bins']['upstream']['events'][str(h)]['event']) for f,n in zip(fs,native)])))
    out=ROOT/'scores';out.mkdir(exist_ok=False)
    save(out/'forecasts.json',dict(schema='sumocore_observable_forecasts/1',K=3,
        episodes=[f for group in forecasts.values() for f in group]))
    for name,data in [('delay',rows),('speed',speed_rows),('events',event_rows),('effects',effects)]:
        csvout(out/(name+'.csv'),data)
    summary=[];families=defaultdict(list);bindings=defaultdict(list)
    for variant in ('default','inferred'):
        for split in ('development6','reserved2','all8'):
            for arm in ('native','mappo','pi','all'):
                for h in HS:
                    select=lambda data:[r for r in data if r['variant']==variant and r['horizon']==h
                        and (split=='all8' or r['split']==split) and (arm=='all' or r['arm']==arm)]
                    d=select(rows);e=[r for r in select(event_rows) if r['region']=='upstream']
                    s=[r for r in select(speed_rows) if r['region']=='upstream']
                    shift=select(effects)
                    record=dict(variant=variant,split=split,arm=arm,horizon=h,worlds=len({r['world'] for r in d}),
                        delay_crps=avg([r['crps'] for r in d]),fair_delay_crps=avg([r['fair_crps'] for r in d]),
                        delay_bias=avg([r['bias'] for r in d]),range_coverage=avg([r['range_covered'] for r in d]),
                        cell_climatology_crps=avg([r['cell_climatology'] for r in d]),
                        pooled_climatology_crps=avg([r['pooled_climatology'] for r in d]),
                        upstream_real_rate=avg([r['real_event'] for r in e]),upstream_predicted_rate=avg([r['event_probability'] for r in e]),
                        upstream_speed_crps=avg([r['crps'] for r in s]),
                        upstream_fair_speed_crps=avg([r['fair_crps'] for r in s]),
                        upstream_speed_range_coverage=avg([r['range_covered'] for r in s]),
                        upstream_speed_cell_climatology_crps=avg([r['cell_climatology'] for r in s]),
                        upstream_onset_ibs=avg([r['onset_ibs'] for r in e]),
                        upstream_both_event_onset_mae=avg([r['both_event_onset_mae'] for r in e]),
                        upstream_real_preexisting=avg([r['real_preexisting'] for r in e]),
                        upstream_predicted_preexisting=avg([r['preexisting_probability'] for r in e]),
                        expected_delay_shift=avg([r['expected_shift'] for r in shift]),real_delay_shift=avg([r['real_shift'] for r in shift]),
                        expected_event_shift=avg([r['upstream_predicted_probability_shift'] for r in shift]),
                        real_event_shift=avg([r['upstream_real_probability_shift'] for r in shift]))
                    if arm!='all':record['real_rate_wilson95']=wilson(sum(int(r['real_event']) for r in e),len(e))
                    summary.append(record)
                    for family,data,fn in (
                        ('delay_cell_difference',d,lambda r: r['crps']-r['cell_climatology'] if r['cell_climatology'] is not None else None),
                        ('delay_bias',d,lambda r:r['bias']),
                        ('speed_cell_difference',s,lambda r:r['crps']-r['cell_climatology'] if r['crps'] is not None and r['cell_climatology'] is not None else None),
                        ('upstream_rate_difference',e,lambda r:r['event_probability']-r['real_event']),
                        ('expected_delay_shift_error',shift,lambda r:r['expected_shift']-r['real_shift'])):
                        feature=by_world(data,fn)
                        if feature:families[family].append(feature);bindings[family].append(record)
    for family,features in families.items():
        bands=bootstrap_family(features)
        for record,band in zip(bindings[family],bands):record[family+'_band']=band
    # Exact subset WM2 comparison exists only for MAPPO/PI (no saved native WM2).
    wm=defaultdict(list)
    with (HERE/'wm2_forecast_draws.csv').open() as f:
        for r in csv.DictReader(f):
            if r['arm'] in ('mappo','pi') and r['target']=='absolute' and r['clock']=='team':
                key=int(r['seed']),r['arm'],int(r['horizon_s'])
                if key[0] in {j['world'] for j in plan['jobs']}:wm[key].append((float(r['predicted']),float(r['real'])))
    comparisons=[]
    for h in HS:
        records=[(key,v) for key,v in wm.items() if key[2]==h]
        comparisons.append(dict(horizon=h,world_arm_cells=len(records),
            wm2_crps=avg([crps([x[0] for x in v],v[0][1]) for _,v in records]),
            wm2_bias=avg([avg([x[0] for x in v])-v[0][1] for _,v in records]),
            default_crps=avg([r['crps'] for r in rows if r['horizon']==h and r['variant']=='default' and r['arm']!='native']),
            inferred_crps=avg([r['crps'] for r in rows if r['horizon']==h and r['variant']=='inferred' and r['arm']!='native'])))
    save(out/'summary.json',dict(summary=summary,wm2_matched=comparisons,attempts=attempts,
        K=3,exchangeable_range_reference=.5,
        references=dict(core_sha256=digest(HERE/'eval_core.py'),detector_bundle_sha256=digest(HERE/'references.json'),
                        real_sha256=digest(HERE/'real_observables.json'),wm2_sha256=digest(HERE/'wm2_forecast_draws.csv')),
        collision_flagged_attempts=[a['id'] for a in attempts if not a['scientific_valid']],
        qualification='NOT QUALIFIED: collision-flagged futures retained; sparse geometry-supported8-world diagnostic, K3, no simultaneous powered gate'))
    print(json.dumps(dict(jobs=len(attempts),actual_sim_s=sum(r['actual_sim_s'] for r in attempts),wm2_matched=comparisons)))


if __name__=='__main__':main()
