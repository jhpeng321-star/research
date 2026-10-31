"""Analyze existing observable reductions and canonical K8 aggregates only."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'docs/paper_results/wmbreakdown'
CAL=Path('E:/agent-work/c4-wmcalib/tools/wmcalib')
ARMS=('ctrl0','mappo','pi','circles')
HS=(120,300,600)
BAD=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vehicle_params','driver_params','vtype')

def read(p):
    assert not any(x in str(p).lower() for x in BAD),p
    return Path(p).read_bytes()

def csvout(name,rows):
    assert rows,name
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def mean(x):
    x=[v for v in x if v is not None]
    return float(np.mean(x)) if x else None

def corr(x,y):
    return float(np.corrcoef(x,y)[0,1]) if len(x)>2 and np.std(x)>0 and np.std(y)>0 else None

def wilson(k,n):
    z=1.959963984540054; p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d; s=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return float(c-s),float(c+s)

def groups(rows,keys):
    out=defaultdict(list)
    for r in rows: out[tuple(r[k] for k in keys)].append(r)
    return out

def boot_world(rows,key):
    by=groups(rows,['seed']); vals=np.array([mean(r[key] for r in v) for v in by.values()])
    if len(vals)<2:return [None,None]
    rng=np.random.default_rng(20261031)
    draws=vals[rng.integers(len(vals),size=(2000,len(vals)))].mean(1)
    return np.quantile(draws,[.025,.975]).tolist()

def breakdowns(episodes):
    events=[]
    for e in episodes:
        for reg,b in e['bins'].items():
            for h in HS:
                d=b['events'][str(h)]
                events.append({k:e[k] for k in ('corpus','split','seed','arm','demand','penetration')}|
                    dict(region=reg,horizon_s=h,event=d['event'],onset_s=d['onset_s'],
                         preexisting=d.get('preexisting',False),drop_mps=d.get('drop_mps'),
                         bic_improvement=d.get('bic_improvement'),team_delay_vs=e['clocks'][str(h)]['team_delay_vs']))
    assert len(events)==4500
    csvout('real_events',events)
    summary=[]
    for keys in (['corpus','split','region','horizon_s','arm'],
                 ['corpus','split','region','horizon_s','demand','penetration','arm']):
        for group,rs in groups(events,keys).items():
            k=sum(r['event'] is True for r in rs); n=len(rs); ci=wilson(k,n)
            on=[r['onset_s'] for r in rs if r['event']]
            summary.append(dict(zip(keys,group))|dict(level='cell' if 'demand' in keys else 'arm',
                n=n,events=k,probability=k/n,wilson_low=ci[0],wilson_high=ci[1],
                preexisting=sum(r['preexisting'] for r in rs),onset_mean_s=mean(on),
                onset_median_s=float(np.median(on)) if on else None,
                onset_p10_s=float(np.quantile(on,.1)) if on else None,
                onset_p90_s=float(np.quantile(on,.9)) if on else None))
    csvout('real_probability',summary)
    lookup={(r['seed'],r['arm'],r['region'],r['horizon_s']):r for r in events}
    pairs=[]
    for r in events:
        for ref in ('native','ctrl0'):
            if ref==r['arm']:continue
            b=lookup[r['seed'],ref,r['region'],r['horizon_s']]
            re,ae=b['event'],r['event']; category='excluded_preexisting'
            both=bool(re and ae)
            dt=r['onset_s']-b['onset_s'] if both else None
            if not r['preexisting'] and not b['preexisting']:
                category=('avoided' if re and not ae else 'induced' if ae and not re else
                          'neither' if not ae and not re else 'delayed' if dt>0 else
                          'advanced' if dt<0 else 'same_onset')
            pairs.append({k:r[k] for k in ('corpus','split','seed','arm','demand','penetration','region','horizon_s')}|
                dict(reference=ref,probability_shift=int(ae)-int(re),both_events=both,
                     both_onset_shift_s=dt,category=category,
                     gain_veh_s=b['team_delay_vs']-r['team_delay_vs'],
                     censored_onset_shift_s=(r['onset_s'] if ae else r['horizon_s']-30)-
                        (b['onset_s'] if re else r['horizon_s']-30)
                        if category!='excluded_preexisting' else None))
    csvout('real_paired',pairs)
    shifts=[]
    for keys in (['corpus','split','region','horizon_s','reference','arm'],
                 ['corpus','split','region','horizon_s','reference','demand','penetration','arm']):
        for group,rs in groups(pairs,keys).items():
            dt=[r['both_onset_shift_s'] for r in rs if r['both_events']]
            ci=boot_world(rs,'probability_shift')
            shifts.append(dict(zip(keys,group))|dict(level='cell' if 'demand' in keys else 'arm',n=len(rs),
                probability_shift=mean(r['probability_shift'] for r in rs),shift_low=ci[0],shift_high=ci[1],
                both_events_n=len(dt),both_onset_shift_s=mean(dt),gain_veh_s=mean(r['gain_veh_s'] for r in rs)))
    csvout('real_shifts',shifts)
    association=[]
    focus=[r for r in pairs if r['region']=='upstream' and r['horizon_s']==600 and r['arm']!='native']
    for group,rs in groups(focus,['corpus','split','reference','category']).items():
        association.append(dict(zip(('corpus','split','reference','category'),group))|
            dict(n=len(rs),gain_veh_s=mean(r['gain_veh_s'] for r in rs)))
    csvout('gain_by_breakdown',association)
    return events,pairs,summary,shifts,association

def load_forecasts(episodes):
    by={(e['seed'],e['arm']):e for e in episodes}
    manifest_path=CAL/'evidence/canonical_manifest.json'
    manifest=json.loads(read(manifest_path)); selected=manifest['selected']
    assert len(selected)==320
    tasks={}; audit=[]; dirs=set()
    for r in selected:
        p=CAL/r['path']; data=read(p)
        assert hashlib.sha256(data).hexdigest()==r['sha256']
        t=json.loads(data); seed=r['seed']; k=r['future']
        assert t['metadata']['seed']==seed and t['future']==k and t['arms']==list(ARMS)
        assert t['metadata']['initial_state_sha256']==by[seed,'ctrl0']['initial_sha256']
        assert all(by[seed,a]['initial_sha256']==by[seed,'ctrl0']['initial_sha256'] for a in ARMS)
        assert (seed,k) not in tasks
        tasks[seed,k]=t; dirs.add(p.parent)
        audit.append(dict(seed=seed,future=k,path=str(p),sha256=r['sha256']))
    assert set(tasks)=={(s,k) for s in range(969400,969440) for k in range(8)}
    inventories=[]
    for d in sorted(dirs):
        identity=d/'IDENTITY.json'
        shard_identity_available=identity.exists()
        if not shard_identity_available:
            identity=CAL/'data/wm40/wm_gpu_shard0/IDENTITY.json'
        ident=json.loads(read(identity))
        assert ident['wm_sha256'].startswith('b8797242') and ident['mix'] is True
        assert ident['batch_arms']==list(ARMS)
        files=[p.name for p in d.iterdir() if not any(x in str(p).lower() for x in BAD)]
        inventories.append(dict(directory=str(d),files=files,identity_path=str(identity),
            shard_identity_available=shard_identity_available,
            identity_sha256=hashlib.sha256(read(identity)).hexdigest()))
    producer=CAL/'wm_gpu_wmcalib.py'
    scorer=Path('E:/agent-work/c4-speedhead/tools/setupaudit/panel2.py')
    audit_doc=dict(tasks=audit,directories=inventories,canonical_manifest_sha256=hashlib.sha256(read(manifest_path)).hexdigest(),
        saved_task_keys=sorted(set(k for t in tasks.values() for k in t)),
        saved_cost_keys=sorted(set(k for t in tasks.values() for k in t['costs'])),
        inspected_sources={str(p):hashlib.sha256(read(p)).hexdigest() for p in (producer,scorer)},
        trajectory_available=False,wm_native_available=False,
        explanation='Producer lines 277-286 save costs/stats JSON then delete roll; scorer lines 217-224 retain cumulative and 30s cost sums, not speed/position traces. No imagined BIC detection is possible. Missing per-shard identities are explicit; shared shard0 identity is context, canonical manifest hashes and per-task start identities bind every task.')
    (OUT/'forecast_audit.json').write_text(json.dumps(audit_doc,indent=1))
    draws=[]; forecasts=[]
    for seed in range(969400,969440):
        e=by[seed,'ctrl0']
        for target,arms in (('absolute',ARMS),('effect',ARMS[1:])):
            for arm in arms:
                i=ARMS.index(arm)
                for h in HS:
                    for clock,key,realkey in (('team','real_J','team_delay_vs'),('objective','J','objective_J')):
                        vals=np.array([tasks[seed,k]['costs'][str(h)][key][i] for k in range(8)],float)
                        y=by[seed,arm]['clocks'][str(h)][realkey]
                        if target=='effect':
                            vals-=np.array([tasks[seed,k]['costs'][str(h)][key][0] for k in range(8)])
                            y-=by[seed,'ctrl0']['clocks'][str(h)][realkey]
                        meta={k:e[k] for k in ('seed','split','demand','penetration')}|dict(arm=arm,target=target,horizon_s=h,clock=clock)
                        forecasts.append(meta|dict(real=float(y),samples=vals))
                        for k,v in enumerate(vals):draws.append(meta|dict(future=k,real=float(y),predicted=float(v)))
    csvout('forecast_draws',draws)
    return forecasts

def crps(x,y,fair=False):
    x=np.asarray(x,float); n=len(x)
    d=float(np.abs(x[:,None]-x).sum())
    return float(np.abs(x-y).mean()-.5*d/(n*(n-1) if fair and n>1 else n*n))

def distributions(forecasts):
    rows=[]
    by=groups(forecasts,['target','horizon_s','clock'])
    for key,allrs in by.items():
        dev=[r for r in allrs if r['split']=='dev32']
        for r in allrs:
            peers=[p for p in dev if p['seed']!=r['seed']]
            same=[p['real'] for p in peers if p['arm']==r['arm']]
            cell=[p['real'] for p in peers if p['arm']==r['arm'] and p['demand']==r['demand'] and p['penetration']==r['penetration']]
            assert same and cell
            x=r['samples']; y=r['real']; less=int((x<y).sum()); equal=int((x==y).sum())
            pit=(less+.5*equal+.5)/9
            row={k:v for k,v in r.items() if k!='samples'}
            wm=crps(x,y)
            row|=dict(wm_mean=float(x.mean()),wm_sd=float(x.std(ddof=1)),error=float(x.mean()-y),
                wm_crps=wm,wm_fair_crps=crps(x,y,True),
                climatology_crps=crps([p['real'] for p in peers],y),
                arm_climatology_crps=crps(same,y),arm_mean_crps=abs(float(np.mean(same))-y),
                cell_arm_climatology_crps=crps(cell,y),cell_baseline_n=len(cell),
                minmax_coverage=int(x.min()<=y<=x.max()),
                central50_coverage=int(np.quantile(x,.25)<=y<=np.quantile(x,.75)),
                central80_coverage=int(np.quantile(x,.1)<=y<=np.quantile(x,.9)),
                rank=less+.5*equal,rank_pit=pit,
                below_all=int(y<x.min()),above_all=int(y>x.max()),baseline_worlds=31 if r['split']=='dev32' else 32)
            for base in ('climatology','arm_climatology','arm_mean','cell_arm_climatology'):
                row['wm_minus_'+base]=wm-row[base+'_crps']
            rows.append(row)
    csvout('distribution_worlds',rows)
    summaries=[]
    for keys in (['split','target','horizon_s','clock'],['split','target','horizon_s','clock','arm'],
                 ['split','target','horizon_s','clock','demand','penetration','arm']):
        for group,rs in groups(rows,keys).items():
            s=dict(zip(keys,group))|dict(arm=rs[0]['arm'] if 'arm' in keys else 'pooled',n=len(rs),worlds=len(set(r['seed'] for r in rs)))
            for k in ('wm_crps','wm_fair_crps','climatology_crps','arm_climatology_crps','arm_mean_crps',
                      'cell_arm_climatology_crps','minmax_coverage','central50_coverage','central80_coverage',
                      'wm_sd','error','below_all','above_all'):
                s[k]=mean(r[k] for r in rs)
            s['real_sd_across_worlds']=float(np.std([r['real'] for r in rs],ddof=1)) if len(rs)>1 else None
            s['wm_mae']=mean(abs(r['error']) for r in rs)
            for base in ('climatology','arm_climatology','arm_mean','cell_arm_climatology'):
                k='wm_minus_'+base; ci=boot_world(rs,k)
                s[k]=mean(r[k] for r in rs); s[k+'_low']=ci[0]; s[k+'_high']=ci[1]
            summaries.append(s)
    csvout('distribution_summary',summaries)
    hist=[]
    for group,rs in groups(rows,['split','target','horizon_s','clock']).items():
        counts=np.histogram([r['rank_pit'] for r in rs],np.linspace(0,1,10))[0]
        for i,n in enumerate(counts):hist.append(dict(zip(('split','target','horizon_s','clock'),group))|
            dict(rank_bin=i,count=int(n),fraction=float(n/len(rs)),expected_fraction=1/9))
    csvout('pit_histogram',hist)
    return rows,summaries,hist

def expected_effects(rows):
    """Expected arm-minus-CTRL0 delay, with uncertainty across observed worlds."""
    effects=[r for r in rows if r['target']=='effect' and r['clock']=='team']
    out=[]
    for keys in (['split','horizon_s','arm'],['split','horizon_s','demand','penetration','arm']):
        for group,rs in groups(effects,keys).items():
            real=mean(r['real'] for r in rs); wm=mean(r['wm_mean'] for r in rs)
            ci=boot_world(rs,'error')
            realci=boot_world(rs,'real')
            out.append(dict(zip(keys,group))|dict(level='cell' if 'demand' in keys else 'arm',
                n=len(rs),real_effect_veh_s=real,wm_expected_effect_veh_s=wm,
                real_low=realci[0],real_high=realci[1],bias=wm-real,bias_low=ci[0],bias_high=ci[1],
                sign_agrees=bool(np.sign(real)==np.sign(wm)),
                real_sign_resolved=bool(realci[0] is not None and (realci[0]>0 or realci[1]<0))))
    csvout('expected_effects',out)
    ranks=[]
    for group,rs in groups([r for r in out if r['level']=='cell'],['split','horizon_s']).items():
        concordant=[]
        for _,cell in groups(rs,['demand','penetration']).items():
            for i,a in enumerate(cell):
                for b in cell[i+1:]:
                    concordant.append(np.sign(a['real_effect_veh_s']-b['real_effect_veh_s'])==
                                      np.sign(a['wm_expected_effect_veh_s']-b['wm_expected_effect_veh_s']))
        resolved=[r for r in rs if r['real_sign_resolved']]
        ranks.append(dict(zip(('split','horizon_s'),group))|dict(cell_arm_groups=len(rs),
            expected_sign_agreement=mean(r['sign_agrees'] for r in rs),
            resolved_groups=len(resolved),resolved_sign_agreement=mean(r['sign_agrees'] for r in resolved),
            within_cell_arm_pairs=len(concordant),within_cell_order_agreement=mean(concordant)))
    csvout('expected_effect_skill',ranks)
    return out,ranks

def figures(events,pairs,rows):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,3,figsize=(13,7.4),constrained_layout=True)
    for j,split in enumerate(('dev32','reserved8','replication50')):
        ax=axs[0,j]
        rs=[r for r in events if r['split']==split and r['region']=='upstream']
        arms=['native',*ARMS]+(['generic_random'] if split=='replication50' else [])
        for arm in arms:
            a=[r for r in rs if r['arm']==arm]
            ax.plot(HS,[mean(r['event'] for r in a if r['horizon_s']==h) for h in HS],marker='o',label=arm)
        ax.set(title=split+': upstream events',xlabel='Detector horizon (s)',ylabel='Detected occurrence fraction',ylim=(0,1.05),xticks=HS)
        if j in (0,2):ax.legend(fontsize=8,ncol=2)
        ax=axs[1,j]
        rs=[r for r in pairs if r['split']==split and r['region']=='upstream' and r['horizon_s']==600 and r['reference']=='ctrl0' and r['arm']!='native' and r['category']!='excluded_preexisting']
        cats=('avoided','delayed','same_onset','advanced','induced','neither')
        for i,c in enumerate(cats):
            a=[r['gain_veh_s'] for r in rs if r['category']==c]
            if a:
                ax.scatter(np.full(len(a),i),a,s=12,alpha=.45,color='#536878')
                ax.plot([i-.23,i+.23],[np.mean(a)]*2,color='black',lw=2)
                ax.text(i,.98,str(len(a)),fontsize=8,ha='center',va='top',transform=ax.get_xaxis_transform())
        ax.axhline(0,color='gray',lw=.8)
        ax.set(title=split+': event category vs gain',ylabel='CTRL0 − controller team delay (veh-s)',xlabel='600 s detected category; black = mean',xticks=range(6),xticklabels=cats)
        ax.tick_params(axis='x',rotation=35)
    fig.suptitle('Recorded traffic: registered BIC step+trend detector, no speed cutoff\nWMDATA 969400–439 / WMDATA2 969550–599; pre-existing pairs excluded from gain categories',fontsize=12)
    fig.savefig(OUT/'real_breakdown.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(12,7.3),constrained_layout=True)
    for i,split in enumerate(('dev32','reserved8')):
        for j,h in enumerate(HS):
            ax=axs[i,j]; rs=[r for r in rows if r['split']==split and r['clock']=='team' and r['target']=='absolute' and r['horizon_s']==h]
            ax.hist([r['rank_pit'] for r in rs],bins=np.linspace(0,1,10),weights=np.full(len(rs),1/len(rs)),color='#536878',rwidth=.86)
            ax.axhline(1/9,color='black',ls='--',label='Exchangeable K8 expectation')
            ax.set(title=f'{split}, {h} s (n={len(rs)})\nMin–max coverage: {mean(r["minmax_coverage"] for r in rs):.1%}',
                xlabel='K8 rank PIT of recorded team delay',ylabel='Fraction of world×arm outcomes',xlim=(0,1),ylim=(0,1))
            if i==0 and j==0:ax.legend(fontsize=8)
    fig.suptitle('WM2+mix saved K8 futures: rank calibration of absolute team delay\nCanonical 320 tasks; 4 arms; cumulative clock; uniformity is necessary, not sufficient',fontsize=12)
    fig.savefig(OUT/'wm_distribution.png',dpi=170);plt.close(fig)

def main():
    episodes=[]
    for corpus,n in (('wmdata',200),('wmdata2',300)):
        d=json.loads(read(OUT/(corpus+'_observables.json'))); assert len(d['episodes'])==n
        episodes+=d['episodes']
    ev,pa,pr,sh,assoc=breakdowns(episodes)
    forecasts=load_forecasts(episodes)
    rows,summary,hist=distributions(forecasts)
    effects,ranks=expected_effects(rows)
    figures(ev,pa,rows)
    digest=dict(real_episodes=len(episodes),event_rows=len(ev),forecasts=len(forecasts),
        primary_summary=[r for r in summary if r['clock']=='team' and r['arm']=='pooled'],
        real_upstream=[r for r in pr if r['region']=='upstream' and r['level']=='arm'],
        upstream_shifts=[r for r in sh if r['region']=='upstream' and r['horizon_s']==600 and r['level']=='arm'],
        gain_categories=assoc,expected_effects=[r for r in effects if r['level']=='arm'],
        expected_effect_skill=ranks)
    (OUT/'summary.json').write_text(json.dumps(digest,indent=1,allow_nan=False))
    print('Verified:',len(episodes),'recorded episodes;',len(forecasts),'K8 distributions;',len(rows),'score rows')
    for r in digest['primary_summary']:
        print(r['split'],r['target'],r['horizon_s'],'CRPS WM/climate/armClimate/armMean',
              *(round(r[k]) for k in ('wm_crps','climatology_crps','arm_climatology_crps','arm_mean_crps')),
              'rangeCoverage',round(r['minmax_coverage'],3),'bias',round(r['error']))

if __name__=='__main__':main()
