"""Aggregate worlds, never treating K draws as independent evidence."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

VARIANTS=('baseline','residual_off','population','courtesy_off','yield_off','ramp_adapt_off')
ARMS=('mappo','ours_c1')


def clean(obj):
    if isinstance(obj,dict):return {k:clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [clean(v) for v in obj]
    if isinstance(obj,(float,np.floating)):return float(obj) if np.isfinite(obj) else None
    if isinstance(obj,np.integer):return int(obj)
    return obj


def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(clean(rows))


def slope(x,y):
    if len(x)<3 or np.std(x)==0:return np.nan
    return float(np.polyfit(x,y,1)[0])


def metrics(a):
    a=np.asarray(a,float); y,x=a[:,0],a[:,1]; e=y-x
    keep=x!=0
    return {'bias':float(e.mean()),'mae':float(abs(e).mean()),
            'real_mean':float(x.mean()),'wm_mean':float(y.mean()),
            'sign':float(np.mean(np.sign(y[keep])==np.sign(x[keep]))) if keep.any() else np.nan,
            'slope':slope(x,y)}


def interval(vals):
    a=np.asarray(vals,float);a=a[np.isfinite(a)]
    return np.quantile(a,[.025,.975]).tolist() if len(a) else [None,None]


def aggregate(root,out,allow_partial=False):
    out.mkdir(parents=True,exist_ok=True)
    docs=[]
    for f in sorted((root/'results').glob('*_s600.json')):
        d=json.loads(f.read_text())
        if d.get('panel','e3')!='e3':continue
        assert d['role']=='validation' and d['steps']==600
        docs.append(d)
    bins=defaultdict(list)
    for d in docs:bins[d['pair_key'],d['variant']].append(d)
    worlds=sorted({d['pair_key'] for d in docs})
    complete=[w for w in worlds if all(len(bins[w,v])==8 for v in VARIANTS)]
    if not allow_partial:
        assert len(worlds)==36 and len(complete)==36,(len(worlds),len(complete),len(docs))
        assert len(docs)==36*6*8
    coverage={'worlds_seen':len(worlds),'complete_worlds':len(complete),'n_results':len(docs),
              'expected':36*6*8,'partial':len(complete)!=36,
              'counts':{v:sum(d['variant']==v for d in docs) for v in VARIANTS}}
    (out/'COVERAGE.json').write_text(json.dumps(coverage,indent=1))
    rows=[]; draws=[]; timeline=[]; raw_breakdown=[]; world_breakdown=[]; confusion=[]
    # The production training loader exposes a t0 cohort, NOT full future traffic.
    # Breakdown sensors must read the full converted recordings including arrivals.
    # Recompute both detectors here, also standardizing the detector implementation
    # across analysis plumbing revisions without rerunning model trajectories.
    from replay import safe,regions,breakdown
    manifest=json.loads(safe(root/'e3data/C4_EPISODES_MANIFEST.json').read_text())
    sensor_cache={}
    def sensor(world,arm):
        key=(world,arm)
        if key in sensor_cache:return sensor_cache[key]
        e=next(e for e in manifest['episodes'] if e['world']==world.split('@')[0] and e['arm']==arm)
        assert e['e3data_split']=='validation'
        p=root/'e3data'/e['dir']
        t=np.load(safe(p/'time_s.npy'),mmap_mode='r')
        t0=int(np.searchsorted(t,e['warmup_s']));assert t0>=120
        a=np.load(safe(p/'veh_kin.npy'),mmap_mode='r')[t0-120:t0+601]
        m=np.load(safe(p/'veh_mask.npy'),mmap_mode='r')[t0-120:t0+601].astype(bool)
        lane=a[...,1:4].argmax(-1);m=m&(a[...,1:4].sum(-1)>0)
        result={}
        for region,reg in regions(a[...,0],lane).items():
            if region not in ('upstream','merge'):continue
            vals=[]
            for start in range(0,720,20):
                keep=(m&reg)[start:start+20]
                vals.append(float(a[start:start+20,:,4][keep].mean()) if keep.any() else None)
            result[region]=vals
        sensor_cache[key]=result
        return result
    for (world,v),group in sorted(bins.items()):
        if len(group)!=8:continue
        assert {d['draw'] for d in group}==set(range(8))
        group=sorted(group,key=lambda d:d['draw'])
        for arm in ARMS:
            for h in ('120','300'):
                pairs=[]
                for d in group:
                    a=np.array(d['arms'][arm]['delay'][h]); n=np.array(d['arms']['native']['delay'][h]); p=a-n
                    pairs.append(p)
                    draws.append({'world':world,'variant':v,'arm':arm,'horizon_s':int(h),
                        'draw':d['draw'],'wm_dJ':p[0],'real_dJ':p[1],'error':p[0]-p[1]})
                a=np.array(pairs)
                assert np.ptp(a[:,1])<1e-8
                y,x=a.mean(0)
                rows.append({'world':world,'variant':v,'arm':arm,'horizon_s':int(h),'K':8,
                    'wm_dJ':y,'real_dJ':x,'paired_bias':y-x,'wm_sd_K':a[:,0].std(ddof=1),
                    'sign_agreement':float(np.sign(y)==np.sign(x)) if x!=0 else None,
                    'paired_ratio_not_regression':y/x if x!=0 else None})
            if v=='baseline':
                for idx,(region,cls,end,*_) in enumerate(group[0]['arms'][arm]['timeline']):
                    a=[]
                    for d in group:
                        p=np.array(d['arms'][arm]['timeline'][idx][3:5])
                        n=np.array(d['arms']['native']['timeline'][idx][3:5]);a.append(p-n)
                    a=np.array(a); y,x=a.mean(0)
                    timeline.append({'world':world,'arm':arm,'region':region,'class':cls,
                        'bin_end_s':end,'wm_dJ_bin':y,'real_dJ_bin':x,'paired_error_bin':y-x,
                        'wm_sd_K':a[:,0].std(ddof=1)})
        if v=='baseline':
            for arm in ('native',*ARMS):
                for region in ('upstream','merge'):
                    observed=sensor(world,arm)[region]
                    real_detector=breakdown(observed)
                    records=[{'real':real_detector,'wm':breakdown(observed[:6]+d['arms'][arm]['speed_bins_wm'][region+'/all']['speed'])}
                             for d in group]
                    real=records[0]['real']
                    assert all(r['real']==real for r in records)
                    valid=[r['wm']['event'] for r in records if r['wm']['event'] is not None]
                    p=float(np.mean(valid)) if len(valid)==8 else None
                    pred=(p>.5) if p is not None else None
                    times=[r['wm']['onset_s'] for r in records if r['wm']['event']]
                    onset=float(np.median(times)) if pred and times else None
                    rt=real['onset_s'] if real['event'] else None
                    err=onset-rt if onset is not None and rt is not None else None
                    event_steps=[r['wm']['drop_mps'] for r in records if r['wm']['event']]
                    world_breakdown.append({'world':world,'arm':arm,'region':region,
                        'real_event':real['event'],'wm_event_majority':pred,'wm_event_probability':p,
                        'real_onset_s':rt,'wm_onset_s':onset,'onset_error_s':err,
                        'real_step_mps':real.get('drop_mps'),'real_BIC_improvement':real.get('bic_improvement'),
                        'wm_event_step_mean_mps':float(np.mean(event_steps)) if event_steps else None,
                        'real_preexisting':real.get('preexisting',False)})
                    for d,r in zip(group,records):
                        raw_breakdown.append({'world':world,'arm':arm,'region':region,'draw':d['draw'],
                            **{'real_'+k:v for k,v in r['real'].items()},
                            **{'wm_'+k:v for k,v in r['wm'].items()}})
    # Identifiable two-contrast within-world gain; ratios above are NOT regressions.
    within=[]
    for w in worlds:
        for v in VARIANTS:
            for h in (120,300):
                a=[r for r in rows if r['world']==w and r['variant']==v and r['horizon_s']==h]
                if len(a)!=2:continue
                x=np.array([r['real_dJ'] for r in a]); y=np.array([r['wm_dJ'] for r in a])
                within.append({'world':w,'variant':v,'horizon_s':h,'n_contrasts':2,
                    'through_origin_slope':float(x@y/(x@x)) if x@x>0 else None})
    pooled=[]
    rng=np.random.default_rng(20261028)
    draws_idx=rng.integers(0,36,(2000,36))
    for v in VARIANTS:
        for h in (120,300):
            for arm in (*ARMS,'pooled'):
                selection=[r for r in rows if r['variant']==v and r['horizon_s']==h
                           and (arm=='pooled' or r['arm']==arm)]
                ws=sorted({r['world'] for r in selection})
                if len(ws)!=36:continue
                groups=[np.array([[r['wm_dJ'],r['real_dJ']] for r in selection if r['world']==w]) for w in ws]
                a=np.concatenate(groups); point=metrics(a)
                boot=[metrics(np.concatenate([groups[i] for i in ix])) for ix in draws_idx]
                row={'variant':v,'arm':arm,'horizon_s':h,'n_worlds':36,'n_pairs':len(a),**point}
                for key in point:
                    lo,hi=interval([r[key] for r in boot]);row[key+'_lo95']=lo;row[key+'_hi95']=hi
                if v!='baseline':
                    base=[r for r in rows if r['variant']=='baseline' and r['horizon_s']==h
                          and (arm=='pooled' or r['arm']==arm)]
                    if len(base)==len(selection):
                        bg=[np.array([[r['wm_dJ'],r['real_dJ']] for r in base if r['world']==w]) for w in ws]
                        bm=metrics(np.concatenate(bg)); bboot=[metrics(np.concatenate([bg[i] for i in ix])) for ix in draws_idx]
                        for label,key,func in [('slope_toward_one','slope',lambda z:abs(z-1)),
                                               ('abs_bias_reduction','bias',abs),
                                               ('mae_reduction','mae',lambda z:z)]:
                            row[label]=func(bm[key])-func(point[key])
                            lo,hi=interval([func(b[key])-func(c[key]) for b,c in zip(bboot,boot)])
                            row[label+'_lo95']=lo;row[label+'_hi95']=hi
                pooled.append(row)
    for arm in ('native',*ARMS,'pooled'):
        for region in ('upstream','merge'):
            a=[r for r in world_breakdown if r['region']==region and (arm=='pooled' or r['arm']==arm)]
            known=[r for r in a if r['real_event'] is not None and r['wm_event_majority'] is not None]
            if not known:continue
            row={'arm':arm,'region':region,'n':len(known),'unknown':len(a)-len(known)}
            for re,pred,label in [(True,True,'TP'),(True,False,'FN'),(False,True,'FP'),(False,False,'TN')]:
                row[label]=sum(r['real_event']==re and r['wm_event_majority']==pred for r in known)
            es=[r['onset_error_s'] for r in known if r['onset_error_s'] is not None]
            row.update(onset_n=len(es),onset_MAE_s=float(np.mean(np.abs(es))) if es else None,
                       onset_bias_s=float(np.mean(es)) if es else None,
                       real_occurrence=sum(r['real_event'] for r in known)/len(known),
                       wm_occurrence=sum(r['wm_event_majority'] for r in known)/len(known))
            steps=[r['real_step_mps'] for r in known if r['real_event']]
            wsteps=[r['wm_event_step_mean_mps'] for r in known if r['wm_event_majority']]
            row['real_event_step_mean_mps']=float(np.mean(steps)) if steps else None
            row['wm_event_step_mean_mps']=float(np.mean(wsteps)) if wsteps else None
            confusion.append(row)
    tl_summary=[]
    for arm in ARMS:
        for region in ('upstream','merge','ramp','downstream'):
            for cls in ('CAV','HDV'):
                cum={}
                for end in range(10,301,10):
                    a=[r for r in timeline if r['arm']==arm and r['region']==region and r['class']==cls and r['bin_end_s']==end]
                    if not a:continue
                    for r in a:cum[r['world']]=cum.get(r['world'],0)+r['paired_error_bin']
                    tl_summary.append({'arm':arm,'region':region,'class':cls,'bin_end_s':end,'n_worlds':len(a),
                        'wm_dJ_bin':np.mean([r['wm_dJ_bin'] for r in a]),
                        'real_dJ_bin':np.mean([r['real_dJ_bin'] for r in a]),
                        'paired_error_bin':np.mean([r['paired_error_bin'] for r in a]),
                        'cumulative_error':np.mean(list(cum.values()))})
    regional=[];horizon_curve=[]
    for arm in ARMS:
        for region in ('upstream','merge','ramp','downstream'):
            for cls in ('CAV','HDV'):
                for h in (120,300):
                    groups=[]
                    for w in worlds:
                        rs=[r for r in timeline if r['world']==w and r['arm']==arm and r['region']==region
                            and r['class']==cls and r['bin_end_s']<=h]
                        if len(rs)==h//10:groups.append([sum(r['wm_dJ_bin'] for r in rs),sum(r['real_dJ_bin'] for r in rs)])
                    if len(groups)!=36:continue
                    a=np.array(groups);err=a[:,0]-a[:,1]
                    boots=err[draws_idx].mean(1);lo,hi=interval(boots)
                    regional.append({'arm':arm,'region':region,'class':cls,'horizon_s':h,'n_worlds':36,
                        'wm_dJ':a[:,0].mean(),'real_dJ':a[:,1].mean(),'paired_error':err.mean(),
                        'paired_error_lo95':lo,'paired_error_hi95':hi})
        for h in range(10,301,10):
            pairs=[]
            for w in worlds:
                rs=[r for r in timeline if r['world']==w and r['arm']==arm and r['class']=='HDV' and r['bin_end_s']<=h]
                if len(rs)==4*h//10:pairs.append([sum(r['wm_dJ_bin'] for r in rs),sum(r['real_dJ_bin'] for r in rs)])
            if len(pairs)!=36:continue
            a=np.array(pairs);point=metrics(a)
            row={'arm':arm,'horizon_s':h,'n_worlds':36,**point}
            # Error timeline is descriptive; avoid selecting a significance-based
            # onset across the 30 correlated time points.
            lo,hi=interval((a[:,0]-a[:,1])[draws_idx].mean(1))
            row['bias_lo95']=lo;row['bias_hi95']=hi
            horizon_curve.append(row)
    for name,data in [('ablation_worlds',rows),('ablation_draws',draws),('ablation_summary',pooled),
        ('within_world_slopes',within),('timeline_worlds',timeline),('timeline_summary',tl_summary),
        ('breakdown_worlds',world_breakdown),('breakdown_draws',raw_breakdown),('breakdown_confusion',confusion),
        ('regional_summary',regional),('horizon_curve',horizon_curve)]:
        write_csv(out/(name+'.csv'),data)
    # Numerical accounting check: regional HDV errors recover the gate objective.
    accounting=[]
    for r in rows:
        if r['variant']!='baseline' or r['horizon_s']!=300:continue
        tl=[t for t in timeline if t['world']==r['world'] and t['arm']==r['arm'] and t['class']=='HDV']
        diff=sum(t['paired_error_bin'] for t in tl)-r['paired_bias']
        accounting.append(abs(diff))
    assert not accounting or max(accounting)<.002,max(accounting)
    summary={'coverage':coverage,'ablation_summary':pooled,'breakdown_confusion':confusion,
             'max_delay_accounting_error':max(accounting) if accounting else None,
             'scope':'E3 validation ONLY; K means before world statistics; OLS includes intercept',
             'detector':'BIC downward step+trend vs constant/linear; no speed threshold; majority > 0.5',
             'breakdown_sensor':'full recorded traffic including arrivals, valid mapped road lanes; not the training t0-cohort mask',
             'timeline':'t0 cohort, dynamic trajectory regions; absent/exited vehicles have zero delay'}
    (out/'SUMMARY.json').write_text(json.dumps(clean(summary),indent=1,allow_nan=False))
    return summary


def waves(root,out):
    """Paired-controller signal growth: descriptive, not an isolated pulse gain."""
    data=[]
    for p in sorted((root/'traces').glob('g*_k*.npz')):
        g=int(p.name[1:3]);k=int(p.stem.rsplit('k',1)[1])
        result=root/'results'/f'g{g:02d}_baseline_k{k}_s600.json'
        if not result.exists():continue
        d=json.loads(result.read_text());a=np.load(p,allow_pickle=False)
        if a['speed'].shape[1]!=601:continue
        for b,arm in enumerate(ARMS,1):
            cohort=a['m0'][b]&a['m0'][0]&~a['cav0'][b]&~a['cav0'][0]
            assert np.allclose(a['real_speed'][b,0,cohort],a['real_speed'][0,0,cohort])
            for start in range(0,600,20):
                row={'world':d['pair_key'],'arm':arm,'draw':k,'bin_end_s':(start//20+1)*10}
                for label,vkey,mkey in [('wm','speed','mask'),('real','real_speed','real_mask')]:
                    v=a[vkey][:,start+1:start+21];m=a[mkey][:,start+1:start+21]
                    common=m[b]&m[0]&cohort
                    dv=v[b]-v[0]
                    row[label+'_rms_dv_mps']=float(np.sqrt(np.mean(dv[common]**2))) if common.any() else None
                    row[label+'_n_vehicle_ticks']=int(common.sum())
                    # Fixed-cohort speed-deficit signal: exits imputed at free speed.
                    z=np.where(m,v,31.29)
                    row[label+'_cohort_rms_dv_mps']=float(np.sqrt(np.mean((z[b,:,cohort]-z[0,:,cohort])**2))) if cohort.any() else None
                data.append(row)
    write_csv(out/'paired_signal_draws.csv',data)
    grouped=defaultdict(list)
    for r in data:grouped[r['world'],r['arm'],r['bin_end_s']].append(r)
    rows=[]
    for (w,arm,t),rs in sorted(grouped.items()):
        if len(rs)!=8:continue
        row={'world':w,'arm':arm,'bin_end_s':t,'K':8}
        for label in ('wm','real'):
            for key in ('rms_dv_mps','cohort_rms_dv_mps','n_vehicle_ticks'):
                vals=[r[label+'_'+key] for r in rs if r[label+'_'+key] is not None]
                row[label+'_'+key]=float(np.mean(vals)) if vals else None
        rows.append(row)
    write_csv(out/'paired_signal_worlds.csv',rows)
    summary=[]
    for arm in ARMS:
        for t in range(10,301,10):
            rs=[r for r in rows if r['arm']==arm and r['bin_end_s']==t]
            if not rs:continue
            row={'arm':arm,'bin_end_s':t,'n_worlds':len(rs)}
            for k in ('wm_rms_dv_mps','real_rms_dv_mps','wm_cohort_rms_dv_mps','real_cohort_rms_dv_mps'):
                vals=[r[k] for r in rs if r[k] is not None];row[k]=np.mean(vals) if vals else None
            summary.append(row)
    write_csv(out/'paired_signal_summary.csv',summary)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--partial',action='store_true');a=ap.parse_args()
    result=aggregate(a.root,a.out,a.partial);waves(a.root,a.out)
    print(json.dumps(result['coverage']))


if __name__=='__main__':main()
