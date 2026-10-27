"""Full-road region/class paired timelines, distinct from the t0 gate cohort."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def evaluate(root,out):
    from summarize import write_csv,interval
    bins=defaultdict(list)
    for f in sorted((root/'results').glob('g*_baseline_k*_s600.json')):
        d=json.loads(f.read_text());assert d['role']=='validation'
        if d.get('fullfield_version',0)>=2:bins[d['pair_key']].append(d)
    complete={w:sorted(ds,key=lambda d:d['draw']) for w,ds in bins.items()
              if len(ds)==8 and {d['draw'] for d in ds}==set(range(8))}
    coverage={'scope':'full_onroad_traffic','worlds_complete':len(complete),
              'n_results':sum(map(len,bins.values())),'expected':288,'K':8,
              'status':'complete' if len(complete)==36 else 'pending_class_output_upgrade',
              'bins':'half-open 10 s bins, 0..300 s; valid mapped road lanes; source waiting excluded'}
    (out/'FULLFIELD_COVERAGE.json').write_text(json.dumps(coverage,indent=1))
    if len(complete)!=36:return
    rows=[];max_class_accounting=0.
    for world,ds in sorted(complete.items()):
        for d in ds:
            for arm in ('native','mappo','ours_c1'):
                for source in ('wm','real'):
                    b=d['arms'][arm]['speed_bins_'+source]
                    for region in ('upstream','merge','ramp','downstream'):
                        all_=b[region+'/all'];cav=b[region+'/CAV'];hdv=b[region+'/HDV']
                        assert np.array_equal(np.array(all_['count']),np.array(cav['count'])+hdv['count'])
                        delta=np.array(all_['delay_veh_s'])-np.array(cav['delay_veh_s'])-hdv['delay_veh_s']
                        max_class_accounting=max(max_class_accounting,float(abs(delta).max()))
        for arm in ('mappo','ours_c1'):
            for region in ('upstream','merge','ramp','downstream'):
                for cls in ('CAV','HDV'):
                    key=region+'/'+cls
                    for i in range(30):
                        wdelay=[];rspeed=[];wspeed=[];rdelay=[];counts=[]
                        for d in ds:
                            wa=d['arms'][arm]['speed_bins_wm'][key];wn=d['arms']['native']['speed_bins_wm'][key]
                            ra=d['arms'][arm]['speed_bins_real'][key];rn=d['arms']['native']['speed_bins_real'][key]
                            wdelay.append(wa['delay_veh_s'][i]-wn['delay_veh_s'][i])
                            rdelay.append(ra['delay_veh_s'][i]-rn['delay_veh_s'][i])
                            if wa['speed'][i] is not None and wn['speed'][i] is not None:wspeed.append(wa['speed'][i]-wn['speed'][i])
                            if ra['speed'][i] is not None and rn['speed'][i] is not None:rspeed.append(ra['speed'][i]-rn['speed'][i])
                            counts.append([wa['count'][i],wn['count'][i],ra['count'][i],rn['count'][i]])
                        assert np.ptp(rdelay)<1e-8
                        wm=float(np.mean(wdelay));real=float(np.mean(rdelay))
                        wv=float(np.mean(wspeed)) if wspeed else None;rv=float(np.mean(rspeed)) if rspeed else None
                        count=np.mean(counts,0)*.5
                        rows.append({'scope':'full_onroad_traffic','world':world,'arm':arm,'region':region,'class':cls,
                            'bin_end_s':10*(i+1),'K':8,'wm_dJ_bin':wm,'real_dJ_bin':real,'paired_error_bin':wm-real,
                            'wm_delta_speed_mps':wv,'real_delta_speed_mps':rv,
                            'paired_speed_error_mps':wv-rv if wv is not None and rv is not None else None,
                            'n_wm_speed_draws':len(wspeed),'wm_arm_vehicle_s':count[0],'wm_native_vehicle_s':count[1],
                            'real_arm_vehicle_s':count[2],'real_native_vehicle_s':count[3]})
    assert max_class_accounting<.005,max_class_accounting
    write_csv(out/'fullfield_timeline_worlds.csv',rows)
    summary=[];rng=np.random.default_rng(20261028);picks=rng.integers(0,36,(2000,36))
    for arm in ('mappo','ours_c1'):
        for region in ('upstream','merge','ramp','downstream'):
            for cls in ('CAV','HDV'):
                cumulative={}
                for t in range(10,301,10):
                    rs=[r for r in rows if r['arm']==arm and r['region']==region and r['class']==cls and r['bin_end_s']==t]
                    assert len(rs)==36
                    e=np.array([r['paired_error_bin'] for r in rs]);lo,hi=interval(e[picks].mean(1))
                    for r in rs:cumulative[r['world']]=cumulative.get(r['world'],0)+r['paired_error_bin']
                    speeds=[r for r in rs if r['paired_speed_error_mps'] is not None]
                    summary.append({'scope':'full_onroad_traffic','arm':arm,'region':region,'class':cls,'bin_end_s':t,
                        'n_worlds_delay':36,'wm_dJ_bin':np.mean([r['wm_dJ_bin'] for r in rs]),
                        'real_dJ_bin':np.mean([r['real_dJ_bin'] for r in rs]),'paired_error_bin':e.mean(),
                        'paired_error_lo95':lo,'paired_error_hi95':hi,'cumulative_error':np.mean(list(cumulative.values())),
                        'n_worlds_speed':len(speeds),
                        'wm_delta_speed_mps':np.mean([r['wm_delta_speed_mps'] for r in speeds]) if speeds else None,
                        'real_delta_speed_mps':np.mean([r['real_delta_speed_mps'] for r in speeds]) if speeds else None})
    write_csv(out/'fullfield_timeline_summary.csv',summary)
    coverage['max_class_delay_accounting_error']=max_class_accounting
    (out/'FULLFIELD_COVERAGE.json').write_text(json.dumps(coverage,indent=1))
