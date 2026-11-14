"""MAIN09:57 TRAIN-only safety diagnosis and fixed composite release rule; zero SUMO."""
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .accel_storage import read
from .storage import write


def interval(values):
    a=np.asarray(values,float)
    if not len(a):return None
    idx=np.random.default_rng(20261101).integers(0,len(a),(10000,len(a)))
    return list(map(float,np.quantile(a[idx].mean(1),[.025,.975])))


def comparison(rows):
    out=dict(n=len(rows),worlds=[x['seed'] for x in rows],metrics={})
    if len(rows)<2:return dict(out,release=False,reason='insufficient same-world pairs')
    for metric in ('collisions','teleports','hard_brake','ttc_below_1p5_events','critical_total'):
        d=[r['right'][metric]-r['A'][metric] for r in rows]
        out['metrics'][metric]=dict(mean_delta=float(np.mean(d)),ci95=interval(d),
            mean_A=float(np.mean([r['A'][metric] for r in rows])),
            mean_right=float(np.mean([r['right'][metric] for r in rows])))
    hard=all(r['right'][k]<=r['A'][k] for r in rows for k in ('collisions','teleports'))
    composite=out['metrics']['critical_total']['ci95'][0]<=0
    out.update(collision_teleport_hard_constraints=hard,critical_total_not_significantly_higher=composite,
        release=bool(hard and composite),rule='MAIN09:57; no collision/teleport increase; total critical 95% CI lower<=0')
    return out


def episode(root,ident,with_ttc=False,cache=None):
    p=Path(root)/'jobs'/ident
    if not (p/'RESULT.json').exists():return None
    e=read(p/'RESULT.json');m=e['metrics']['900']
    out=dict(seed=e['world']['seed'],world=e['world'],id=ident,
        collisions=float(m['collisions']),teleports=float(m['teleports']),hard_brake=float(m['hard_brake']))
    if with_ttc:
        from .replay_bank import ttc_events
        marker=hashlib.sha256((p/'RESULT.json').read_bytes()).hexdigest()
        saved=Path(cache)/(ident+'_'+marker+'.json')
        if saved.exists():ttc=read(saved)['ttc_below_1p5_events']
        else:
            ttc=ttc_events(p/'observable.jsonl.gz',300,1200)
            write(saved,dict(ttc_below_1p5_events=ttc,source_result_sha256=marker,
                code='replay_bank.ttc_events/closing_ttc/threshold1.5/inherited1s trace cadence'))
        out.update(ttc_below_1p5_events=ttc,critical_total=out['hard_brake']+ttc)
    return out


def paired(root,arm,u,cache,with_ttc=True):
    rows=[];missing=[]
    for p in sorted((Path(root)/'jobs').glob(f'replay0735_warm_u{u:03}_{arm}_*')):
        if not (p/'RESULT.json').exists():continue
        right=episode(root,p.name,with_ttc,cache)
        A=episode(root,f'warm_u{u:03}_A_{right["seed"]}',with_ttc,cache)
        if A is None:missing.append(right['seed']);continue
        assert A['world']==right['world'],'not the same registered TRAIN world'
        rows.append(dict(seed=right['seed'],world=right['world'],A=A,right=right))
    assert len({r['seed'] for r in rows})==len(rows),'duplicate worlds in paired diagnosis'
    return rows,missing


def windows(path):
    """Continuous deceleration/acceleration proxies; no new behaviour thresholds."""
    bins={};previous={}
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            x=json.loads(line);t=int(x['time_s']);bucket=(t//10)*10
            b=bins.setdefault(bucket,dict(deceleration_variation=0.,positive_CAV_command=0.,
                command_sum=0.,command_count=0,road_position_min=None,road_position_max=None))
            current={v['vid']:v for v in x['vehicles']}
            for vid,v in current.items():
                if vid in previous:b['deceleration_variation']+=max(0.,previous[vid]['v_mps']-v['v_mps'])
                a=v.get('issued_acceleration_mps2')
                if v['is_cav'] and a is not None:
                    b['positive_CAV_command']+=max(0.,float(a));b['command_sum']+=float(a);b['command_count']+=1
                    s=float(v['s_m']);b['road_position_min']=s if b['road_position_min'] is None else min(s,b['road_position_min'])
                    b['road_position_max']=s if b['road_position_max'] is None else max(s,b['road_position_max'])
            previous=current
    for b in bins.values():
        n=b['command_count'];b['mean_CAV_command']=b['command_sum']/n if n else None
        b['mean_positive_CAV_command']=b['positive_CAV_command']/n if n else None
    return bins


def diagnose(root,control):
    import os,fcntl,torch
    root=Path(root);control=Path(control);torch.set_num_threads(1);os.nice(19)
    lock=(control/'DIAGNOSIS.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (control/'COMPLETE.json').exists():return
    cache=control/'ttc_cache';cache.mkdir(exist_ok=True)
    pause=read(root/'PAUSE_SAFETY.json')
    B,missingB=paired(root,'B',6,cache)
    R,missingR=paired(root,'R',5,cache)
    bcmp=comparison(B);rcmp=comparison(R)
    histories={}
    for arm in ('B','R'):
        for u in (4,5,6):
            rows,missing=paired(root,arm,u,cache,with_ttc=False)
            if rows:
                d=[x['right']['hard_brake']-x['A']['hard_brake'] for x in rows]
                histories[f'{arm}{u}_minus_A{u}']=dict(n=len(rows),mean_delta=float(np.mean(d)),ci95=interval(d),missing=missing)
    av={}
    for u in (5,6,7):
        es=[episode(root,p.name) for p in sorted((root/'jobs').glob(f'warm_u{u:03}_A_*')) if (p/'RESULT.json').exists()]
        av[str(u)]=dict(n=len(es),mean=float(np.mean([e['hard_brake'] for e in es])) if es else None,
            sd=float(np.std([e['hard_brake'] for e in es],ddof=1)) if len(es)>1 else None,
            by_world={str(e['seed']):e['hard_brake'] for e in es})
    changes={}
    for left,right in ((5,6),(6,7)):
        a=av[str(left)];b=av[str(right)];common=sorted(set(a['by_world'])&set(b['by_world']))
        d=[b['by_world'][w]-a['by_world'][w] for w in common]
        changes[f'A{right}_minus_A{left}']=dict(n_common=len(common),paired_mean=float(np.mean(d)) if d else None,
            paired_ci95=interval(d),unpaired_panel_mean_change=b['mean']-a['mean'],
            interpretation='panels/world mix may differ; paired subset and full-panel means both disclosed')
    top=sorted(B,key=lambda r:r['right']['hard_brake']-r['A']['hard_brake'],reverse=True)
    from .replay_bank import bank_decisions,confirmations
    from .plans import acceleration_at
    source=read(root/'B_REPLAY_START_0835_V4.json')['collection_root'];confirmed=confirmations(root,'B')
    bank_profiles=[]
    for r in bank_decisions(root,'B',sources=[source]):
        if not confirmed.get(r['verification_id'],{}).get('admitted'):continue
        plan=r['chosen_plan'];values=[a for t in range(min(10,int(plan['duration_s']))) for a in acceleration_at(plan,t).values()]
        bank_profiles.append(dict(verification_id=r['verification_id'],members=len(plan['members']),
            requested_mean_first10s=float(np.mean(values)),duration_s=plan['duration_s']))
    situations=[];all_proxy=[]
    for row in top:
        bw=windows(root/'jobs'/row['right']['id']/'observable.jsonl.gz')
        aw=windows(root/'jobs'/row['A']['id']/'observable.jsonl.gz')
        ws=[]
        for t in sorted(set(bw)&set(aw)):
            x=bw[t];a=aw[t]
            d=x['deceleration_variation']-a['deceleration_variation']
            accel=(x['mean_positive_CAV_command'] or 0.)-(a['mean_positive_CAV_command'] or 0.)
            all_proxy.append([d,accel]);ws.append(dict(time_s=t,delta_deceleration_variation=d,
                delta_mean_positive_CAV_command=accel,B=x,A=a))
            if bank_profiles and x['mean_CAV_command'] is not None:
                nearest=min(bank_profiles,key=lambda p:abs(p['requested_mean_first10s']-x['mean_CAV_command']))
                ws[-1]['nearest_admitted_bank_acceleration_mean']=dict(nearest,
                    absolute_mean_difference=abs(nearest['requested_mean_first10s']-x['mean_CAV_command']),
                    interpretation='continuous requested-bank vs issued-CAV mean proxy only; not a discovered-behaviour detector')
        situations.append(dict(seed=row['seed'],world=row['world'],hard_brake_delta=row['right']['hard_brake']-row['A']['hard_brake'],
            top_inherited_10s_windows=sorted(ws,key=lambda x:x['delta_deceleration_variation'],reverse=True)[:3]))
    proxy=np.asarray(all_proxy,float);correlation=None
    if len(proxy)>1 and np.all(proxy.std(0)>0):correlation=float(np.corrcoef(proxy.T)[0,1])
    exposure={}
    for u in (4,5):
        up=read(root/f'B_UPDATE_{u:03}.json');lesson=read(root/f'LESSON_CHECK_u{u:03}_B.json')
        exposure[str(u)]=dict(imitation_action_error_mps2=lesson.get('imitation_action_error_mean_mps2'),
            optimizer_batches_with_imitation=sum(m.get('exact_imitation_steps',0)>0 for m in up['ppo']),
            imitation_pairs_seen=sum(m.get('exact_imitation_steps',0) for m in up['ppo']))
    out=dict(unix=time.time(),authority='MAIN2026-11-03 09:57',new_sim_s=0,training_resumed=False,
        original_pause_sha256=hashlib.sha256((root/'PAUSE_SAFETY.json').read_bytes()).hexdigest(),
        original_pause=pause,B_u6_vs_A_u6=bcmp,R_u5_vs_A_u5=rcmp,
        R_u6_collected=False,R_rule='same composite guard before the first new R-u6 PPO; R5 is last completed comparison',
        missing=dict(B=missingB,R=missingR),paired_world_rows=dict(B=B,R=R),
        A_variability=av,A_update_changes=changes,braking_history=histories,imitation_exposure=exposure,
        situations=situations,admitted_bank_profile_reference=bank_profiles,acceleration_braking_proxy=dict(n_windows=len(proxy),pearson=correlation,
            rule='continuous negative speed variation vs positive issued CAV acceleration; inherited10s windows; descriptive, not a hard-brake event locator or causal detector'),
        attribution='u4/u5 combine PPO, tiny bank imitation and independent exploration; episode data alone do not identify an imitation-only causal braking effect',
        release_eligible=bool(bcmp['release'] and rcmp['release'] and not missingB and not missingR))
    write(control/'SAFETY_DIAGNOSIS_u6.json',out)
    write(control/'COMPLETE.json',dict(unix=time.time(),release_eligible=out['release_eligible'],
        B=bcmp,R=rcmp,new_sim_s=0))


if __name__=='__main__':
    import sys
    diagnose(sys.argv[1],sys.argv[2])
