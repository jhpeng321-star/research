"""Registered complete-panel, whole-world inference; never fits any model."""
from pathlib import Path
import argparse
import csv
import hashlib
import itertools
import json
import warnings

import numpy as np

HERE = Path(__file__).resolve().parent
HS = (10,30,60,90,120,180,300)
ARMS = ('native','ctrl0','mappo','pi','circles')
COMP = ('source','upstream','merge','discharge')
WM_HASH = 'b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436'
MODEL_HASHES=dict(wm2=WM_HASH,
    bbtraj='afaaa32cb15079014d1a9e08f78b26e74fd341298d1aa0292a24009d9f783742',
    bbid='e12c81c008e3bb29e366db5597061754e0fa56092dbe5a995a9baa1fbfcce61d',
    historical='87ed02e0707e5e4eefdd2dbb0fd282f834c1288776f7f395b41ab4d13826aaef')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',newline='',encoding='utf-8') as f:
        out=csv.DictWriter(f,fieldnames=keys);out.writeheader();out.writerows(rows)


def load(root):
    labels=json.loads((root/'labels.json').read_text())
    truth={(r['seed']-969400,r['arm']):r for r in labels['rows']}
    assert len(truth)==200 and set(w for w,a in truth)==set(range(40))
    tasks={}
    closure=None
    allowed=set(roster())
    for p in sorted((root/'panel').glob('*_h300.json')):
        x=json.loads(p.read_text())
        key=(x['model'],x['plant'],x['world'],x['future'])
        assert key in allowed
        assert key not in tasks and x['seed']==969400+x['world']
        assert x['steps']==600 and x['seconds']==300
        assert set(map(int,x['values']))==set(HS)
        assert x['arms']==(list(ARMS) if x['model']!='bbid' else ['native','pi','circles'])
        for arm in x['arms']:
            real=truth[x['world'],arm]
            assert x['identity']['provenance']['metadata']['initial_state_sha256']==real['initial_state_sha256']
            assert x['original_ids']==real['original_ids']
            assert x['original_cav']==real['original_cav']
        assert x['identity']['model_sha256']==MODEL_HASHES[x['model']]
        current={k:x['identity'][k] for k in ('source_sha256','aggregates_sha256','runtime_closure',
                 'candidate_sha256','mappo_sha256','wm_config_sha256','execution_driver_sha256')}
        if closure is None: closure=current
        assert current==closure,'runtime/model execution changed within panel'
        for h,v in x['values'].items():
            assert np.allclose(v['J'],np.asarray([v[c] for c in COMP]).sum(0),atol=1e-4)
            assert np.asarray(v['fields']['density']).shape==(len(x['arms']),int(h)//10,15)
        tasks[key]=x
    return truth,tasks,closure


def roster():
    return [(model,plant,w,k) for w in range(40)
            for model,plant,n in [('wm2','common',8),('wm2','production',1),
                ('bbtraj','common',1),('bbid','common',1),('historical','common',1)]
            for k in range(n)]


def forecast(rows,arm,h):
    ix=[r['arms'].index(arm) for r in rows]
    v=[r['values'][str(h)] for r in rows]
    # Define J by its registered additive components in float64. The inference
    # writer's three-road-component float32 sum can differ by a few ULPs; this
    # arithmetic normalization is exported, never fitted to recorded outcomes.
    j=np.asarray([sum(float(value[c][i]) for c in COMP) for value,i in zip(v,ix)],dtype=float)
    order=np.argsort(j,kind='stable')
    middle=order[(len(order)-1)//2:len(order)//2+1]
    result={c:float(np.mean([v[k][c][ix[k]] for k in middle])) for c in COMP}
    result['J']=sum(result[c] for c in COMP)
    result['J_rounding_correction']=result['J']-float(np.median([value['J'][i] for value,i in zip(v,ix)]))
    result['horizon_s']=h
    assert np.isclose(sum(result[c] for c in COMP),result['J'],atol=1e-4)
    for metric in ('crossings','merge_queue','mean_merge_queue'):
        result[metric]=float(np.median([value[metric][i] for value,i in zip(v,ix)]))
    result['fields']={k:np.median([np.asarray(value['fields'][k])[i] for value,i in zip(v,ix)],axis=0)
                      for k in ('density','speed')}
    result['fields']['occupied']=result['fields']['density']>0
    result['endpoints']=[value['endpoints'][i] for value,i in zip(v,ix)]
    return result


def endpoint(pred,real,hdv,cohort=None):
    actual={i:(s,v) for i,s,v in zip(real['ids'],real['s'],real['v'])}
    maps=[{i:(s,v) for i,s,v,m in zip(e['ids'],e['s'],e['v'],e['present']) if m}
          for e in pred['endpoints']]
    common=set(hdv)&set(actual)
    for m in maps: common &= set(m)
    if cohort is not None: common &= set(cohort)
    ids=sorted(common)
    if not ids: return dict(position=None,speed=None,n_hdv=0), common
    errors=np.asarray([np.median(np.abs(np.asarray([m[i] for i in ids])-
                         np.asarray([actual[i] for i in ids])),axis=0) for m in maps])
    a=np.median(errors,axis=0)
    return dict(position=float(a[0]),speed=float(a[1]),n_hdv=len(ids)),common


def errors(pred,real):
    out={f'{c}_bias':pred[c]-real[c] for c in (*COMP,'J')}
    out.update({f'{c}_ae':abs(pred[c]-real[c]) for c in (*COMP,'J','crossings','merge_queue','mean_merge_queue')})
    out.update({f'{c}_{kind}':float(row[c]) for c in (*COMP,'J','crossings','merge_queue','mean_merge_queue')
                for kind,row in [('pred',pred),('real',real)]})
    out['throughput_ae']=out['crossings_ae']/pred['horizon_s']*3600.
    out['J_rounding_correction']=pred['J_rounding_correction']
    for k in ('density','speed'):
        out[k+'_field_ae']=float(np.abs(pred['fields'][k]-np.asarray(real['fields'][k])).mean())
    out['field_occupancy_mismatch']=float(np.mean(pred['fields']['occupied']!=np.asarray(real['fields']['occupied'])))
    return out


def decision(pred,real):
    # Exactly-zero real effects are excluded only from direction denominators.
    delta=np.asarray(real)-real[0]
    dhat=np.asarray(pred)-pred[0]
    pairs=[(i,j) for i,j in itertools.combinations(range(len(real)),2) if real[i]!=real[j]]
    result=dict(pairwise_accuracy=float(np.mean([np.sign(pred[i]-pred[j])==np.sign(real[i]-real[j])
                      for i,j in pairs])) if pairs else None,
                top_choice_correct=float(real[int(np.argmin(pred))]==min(real)),
                regret=float(real[int(np.argmin(pred))]-min(real)))
    result['direction_accuracy']=float(np.mean(np.sign(dhat[delta!=0])==np.sign(delta[delta!=0]))) if (delta!=0).any() else None
    return result


def records(truth,tasks):
    raw,pairs=[],[]
    cache={}
    for model,plant in [('wm2','common'),('wm2','production'),('bbtraj','common'),('bbid','common'),('historical','common')]:
        arms=ARMS if model!='bbid' else ('native','pi','circles')
        for w in range(40):
            rows=[tasks[model,plant,w,k] for k in range(8 if (model,plant)==('wm2','common') else 1)]
            for h in HS:
                for arm in arms:
                    pred=forecast(rows,arm,h);cache[model,plant,w,h,arm]=pred
                    real=truth[w,arm]['values'][str(h)]
                    hdv=[i for i,c in zip(rows[0]['original_ids'],rows[0]['original_cav']) if not c]
                    ep,_=endpoint(pred,real['endpoints'],hdv)
                    r=dict(model=model,plant=plant,world=w,seed=969400+w,arm=arm,horizon_s=h,
                           **errors(pred,real),**ep,initial_hdv=len(hdv))
                    delta=real['J']-truth[w,'native']['values'][str(h)]['J']
                    pdelta=pred['J']-forecast(rows,'native',h)['J']
                    r['direction_accuracy']=float(np.sign(delta)==np.sign(pdelta)) if delta else None
                    raw.append(r)
                costs=[cache[model,plant,w,h,a]['J'] for a in arms]
                actual=[truth[w,a]['values'][str(h)]['J'] for a in arms]
                raw.append(dict(model=model,plant=plant,world=w,seed=969400+w,
                    arm='choices_'+('_'.join(arms)),horizon_s=h,**decision(costs,actual)))
    for other in ('bbtraj','bbid','historical'):
        arms=ARMS if other!='bbid' else ('native','pi','circles')
        for w in range(40):
            for h in HS:
                for arm in arms:
                    wm=cache['wm2','common',w,h,arm];bb=cache[other,'common',w,h,arm]
                    real=truth[w,arm]['values'][str(h)]
                    hdv=[i for i,c in zip(truth[w,arm]['original_ids'],truth[w,arm]['original_cav']) if not c]
                    _,wm_ids=endpoint(wm,real['endpoints'],hdv)
                    _,bb_ids=endpoint(bb,real['endpoints'],hdv)
                    a,common=endpoint(wm,real['endpoints'],hdv,wm_ids&bb_ids)
                    b,_=endpoint(bb,real['endpoints'],hdv,common)
                    r=dict(model='wm2-minus-'+other,plant='common',world=w,seed=969400+w,
                        arm=arm,horizon_s=h,n_hdv=len(common))
                    for k in ('position','speed'):
                        r[k]=a[k]-b[k] if common else None
                        # Also export matched-cohort marginal curves, not just contrasts.
                        r['wm_'+k]=a[k];r['bb_'+k]=b[k]
                    ae,be=errors(wm,real),errors(bb,real)
                    r.update({k:ae[k]-be[k] for k in ae if not k.endswith(('_pred','_real'))})
                    delta=real['J']-truth[w,'native']['values'][str(h)]['J']
                    if delta:
                        r['direction_accuracy']=float(np.sign(wm['J']-cache['wm2','common',w,h,'native']['J'])==np.sign(delta))-float(
                            np.sign(bb['J']-cache[other,'common',w,h,'native']['J'])==np.sign(delta))
                    else: r['direction_accuracy']=None
                    pairs.append(r)
                actual=[truth[w,a]['values'][str(h)]['J'] for a in arms]
                da=decision([cache['wm2','common',w,h,a]['J'] for a in arms],actual)
                db=decision([cache[other,'common',w,h,a]['J'] for a in arms],actual)
                pairs.append(dict(model='wm2-minus-'+other,plant='common',world=w,seed=969400+w,
                    arm='choices_'+'_'.join(arms),horizon_s=h,
                    **{k:da[k]-db[k] if da[k] is not None and db[k] is not None else None for k in da}))
    return raw,pairs


def summarize(rows):
    rng=np.random.default_rng(0)
    draws=rng.integers(0,40,(2000,40))
    groups={}
    excluded={'model','plant','world','seed','arm','horizon_s'}
    for r in rows:
        for metric,val in r.items():
            if metric in excluded: continue
            key=(r['model'],r['plant'],r['arm'],r['horizon_s'],metric)
            a=groups.setdefault(key,np.full(40,np.nan))
            assert np.isnan(a[r['world']])
            if val is not None: a[r['world']]=float(val)
    result=[]
    for (model,plant,arm,h,metric),a in sorted(groups.items()):
        # Signed component means preserve additivity across worlds; accuracy and
        # regret are equal-world means. Endpoint/aggregate AEs use world medians.
        mean=metric.endswith('_bias') or metric.endswith('accuracy') or metric in ('regret','top_choice_correct')
        reducer=np.nanmean if mean else np.nanmedian
        n=int(np.isfinite(a).sum())
        if n:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                boot=reducer(a[draws],axis=1)
            point=float(reducer(a));lo,hi=np.nanpercentile(boot,[2.5,97.5])
        else: point=lo=hi=''
        result.append(dict(model=model,plant=plant,arm=arm,horizon_s=h,metric=metric,
            estimate=point,lo=lo,hi=hi,n_worlds=n,world_stat='mean' if mean else 'median',
            units=units(metric),bootstrap_draws=2000,bootstrap_seed=0,status='measured' if n else 'no_surviving_ids'))
    for h in HS:
        for arm in ('pi','mappo'):
            result.append(dict(model='SUMO_RNG_floor',plant='recorded',arm=arm,horizon_s=h,
                metric='position',estimate='',lo='',hi='',n_worlds=0,units='m',status='MAIN_input_pending'))
    return result


def units(metric):
    if metric.startswith(('wm_','bb_')):metric=metric[3:]
    if metric in ('position',):return 'm'
    if metric=='speed' or metric.startswith('speed_field'):return 'm/s'
    if metric.startswith('density_field'):return 'veh/km road'
    if metric.startswith('throughput'):return 'veh/h'
    if metric.startswith(('J','source','upstream','merge_bias','merge_ae','merge_pred','merge_real','discharge')) or metric=='regret':return 'veh-s'
    if metric in ('n_hdv','initial_hdv') or 'queue' in metric or metric.startswith('crossings'):return 'veh'
    return 'fraction'


def plot(summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(11,13),sharex=True,layout='constrained')
    metrics=[('position','HDV position absolute error (m)'),('speed','HDV speed absolute error (m/s)'),
             ('J_ae','Team-delay absolute error (veh-s)'),('density_field_ae','Density-field MAE (veh/km)')]
    for col,arm in enumerate(('pi','mappo')):
        for row,(metric,ylabel) in enumerate(metrics):
            ax=axes[row,col]
            for model,label,color in [('wm2','WM-S + mix','#176b87'),('bbtraj','BB-traj','#c66b32'),('bbid','BB-id (own ID contract)','#777777')]:
                if model=='bbid' and arm=='mappo': continue
                # Primary curves share WM/BB-traj surviving IDs across ALL K8
                # draws. BB-id has a separately labelled paired cohort.
                if row<2:
                    other='bbid' if model=='bbid' else 'bbtraj'
                    keymetric=('wm_' if model=='wm2' else 'bb_')+metric
                    selected=[r for r in summary if r['model']=='wm2-minus-'+other and r['plant']=='common'
                              and r['arm']==arm and r['metric']==keymetric]
                else:
                    selected=[r for r in summary if r['model']==model and r['plant']=='common'
                              and r['arm']==arm and r['metric']==metric]
                selected=sorted(selected,key=lambda r:r['horizon_s'])
                x=[r['horizon_s'] for r in selected]
                y=[float(r['estimate']) if r['estimate']!='' else np.nan for r in selected]
                lo=[float(r['lo']) if r['lo']!='' else np.nan for r in selected]
                hi=[float(r['hi']) if r['hi']!='' else np.nan for r in selected]
                ax.plot(x,y,'o-',label=label,color=color,markersize=3)
                ax.fill_between(x,lo,hi,color=color,alpha=.12)
            ax.set_ylabel(ylabel);ax.grid(alpha=.2);ax.set_xticks(HS)
            if row==0: ax.set_title('PI external-U' if arm=='pi' else 'MAPPO speed-only iter169')
            if row==3: ax.set_xlabel('Forecast horizon (s)')
            if row==0: ax.legend(fontsize=8)
    fig.suptitle('Closed-loop C1 — fixed common external CAV plant, WM2 b8797242\n40 recorded TRAIN worlds; world medians and 95% whole-world bootstrap CIs',fontsize=12)
    fig.text(.5,-.018,'Endpoint gaps = no common surviving initial HDVs; BB-id/MAPPO unavailable.\n'
             'SUMO RNG divergence ("irreducible" reference): MAIN input pending; no floor inferred.\n'
             'Density: fixed -1000..500 m, 100 m × 10 s cells; source: S-C1CLOSED registered panel.',
             ha='center',fontsize=9)
    fig.savefig(out/'C1CLOSED_error_horizon.png',dpi=180,bbox_inches='tight');plt.close(fig)


def report(summary,out):
    def cell(model,arm,h,metric):
        rows=[r for r in summary if (r['model'],r['plant'],r['arm'],r['horizon_s'],r['metric'])==(model,'common',arm,h,metric)]
        if not rows or rows[0]['estimate']=='': return 'unavailable'
        r=rows[0]
        return f"{float(r['estimate']):.3g} [{float(r['lo']):.3g}, {float(r['hi']):.3g}] (n={r['n_worlds']})"
    lines=['# S-C1CLOSED — frozen current-model closed-loop C1',
        '**COMPLETE recorded TRAIN/dev panel; scoped common-plant evidence, provisional algorithm novelty.**',
        'WM2 `b8797242` + learned entrant mix, K8; frozen BB-traj `afaaa32c`, BB-id `e12c81c0`, historical structured `87ed02e0`, K1.',
        '40 worlds 969400–969439, t300, genuine observed 60 s histories, 480 world/future tasks; no SUMO or fitting.',
        'Primary comparison uses shared external CAV longitudinal/gap-merge plant (no ordinary CAV lane changes); production-native WM2 is separate.',
        'Real PI/CIRCLES/MAPPO feedback observes predictions; autonomous HDV/entry/exit/source evolution, no original command tape or future motion inputs.',
        'S-C1GAP supports PI/MAPPO absence from audited WM2 corpora; pre-R16 initialization exclusion remains incomplete. Reused worlds are not unseen-world evidence.',
        'BBs have their original different/no-PI training corpus; BB-id learns CAV motion by ID and cannot execute feedback MAPPO. No baseline retraining.',
        'Wall feedback pending: 120 genuine-history starts survive, but zero executor snapshots/controller maps and only 120 s cached futures; causal restoration not established. CIRCLES/native/CTRL0 are diagnostics.',
        'Endpoint errors: median initial-HDV AE within world/draw, then median K8, then world median; paired models use the identical cohort surviving truth and every WM draw.',
        'All 10/30/60/90/120/180/300 s clocks are in CSV; surviving counts exported, no zero imputation; synthetic entrants have no matched identity.',
        '95% percentile CIs: 2,000 whole-world resamples, seed 0; differences formed within world before bootstrap. Negative WM−BB error differences favour WM.',
        '## Headline 60 s matched endpoints (m)',]
    for arm in ('pi','mappo'):
        lines.append(f"- {arm}: WM {cell('wm2-minus-bbtraj',arm,60,'wm_position')}; BB-traj {cell('wm2-minus-bbtraj',arm,60,'bb_position')}; paired WM−BB {cell('wm2-minus-bbtraj',arm,60,'position')}.")
    lines += ['## All-horizon paired position error WM−BB-traj (m)']
    for h in HS: lines.append(f"- {h} s: PI {cell('wm2-minus-bbtraj','pi',h,'position')}; MAPPO {cell('wm2-minus-bbtraj','mappo',h,'position')}.")
    lines += ['## Long-horizon aggregate fidelity (secondary)',
        'Merge queue: -200≤s≤75.44 m, v<5 m/s; endpoint and time-mean counts. Fixed field cells: 100 m×10 s, -1000..500 m, all lanes pooled, density veh/km road.',
        'Field speed is occupancy-weighted (empty=0); occupancy mismatch exported. Queue/fields/crossings use pointwise draw medians. These are traffic-level errors, not causal diagnoses.']
    for h in (60,120,180,300):
        lines.append(f"- {h} s J absolute-error difference (veh-s): PI {cell('wm2-minus-bbtraj','pi',h,'J_ae')}; MAPPO {cell('wm2-minus-bbtraj','mappo',h,'J_ae')}.")
    lines += ['## Delay accounting and controller effects',
        'J = source waiting + upstream(s<0) + merge(0..75.44) + discharge(s>75.44) speed-deficit integrals; left-boundary .5 s rectangles, vref31.29 m/s.',
        'Common total-J median-order draw weights preserve component additivity within world. Signed component biases use equal-world means (also additive); absolute errors use world medians.',]
    for arm in ('pi','mappo'):
        lines.append('- WM 300 s '+arm+' component biases source/upstream/merge/discharge (veh-s): '+
                     '; '.join(cell('wm2',arm,300,c+'_bias') for c in COMP)+'.')
        lines.append(f"- {arm} WM E2 direction accuracy 60/300 s: {cell('wm2',arm,60,'direction_accuracy')} / {cell('wm2',arm,300,'direction_accuracy')}.")
    choices='choices_'+'_'.join(ARMS)
    lines += [f"- Five-arm oracle regret 300 s: WM {cell('wm2',choices,300,'regret')}; BB-traj {cell('bbtraj',choices,300,'regret')}; includes exposed CTRL0 diagnostic.",
        'E2 ΔJ=treatment−native, negative improves; exactly-zero true effects excluded only from direction denominators. Pairwise rankings exclude true ties; native-first deterministic choice ties.',
        'WMFIXMERGE v2-DATA / WMFIX-ES secondary slot: no independently passing, hash-bound verdict was supplied; primary WM2 is unchanged.',
        'SUMO RNG divergence-floor slot: unavailable pending MAIN-supplied approved recordings; figure leaves it explicitly unmeasured, not zero or a proven lower bound.',
        'Files: `C1CLOSED_per_world.csv`, `C1CLOSED_paired.csv`, `C1CLOSED_bootstrap.csv`, `C1CLOSED_error_horizon.png`, `C1CLOSED_ANALYSIS.json`.',
        'Reproduce: `python -B tools/c1closed/sync.py`; `python -B tools/c1closed/analyze.py`; CPU/source/PID receipts in `tools/c1closed/evidence/`.',
        'No qualification threshold is borrowed from E2; relative BB advantages, absolute fidelity and controller selection are distinct claims.']
    assert len(lines)<=60
    (out/'C1CLOSED.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=HERE/'out')
    p.add_argument('--out',type=Path,default=HERE.parents[1]/'docs/paper_results')
    p.add_argument('--validate',action='store_true');a=p.parse_args()
    truth,tasks,closure=load(a.root)
    missing=set(roster())-tasks.keys()
    print(json.dumps(dict(validated_tasks=len(tasks),missing=len(missing),label_arms=len(truth))))
    if a.validate: return
    assert not missing,'no selective partial-panel result is exported'
    a.out.mkdir(parents=True,exist_ok=True)
    assert not (a.out/'C1CLOSED_ANALYSIS.json').exists(),'write-once analysis already exists'
    raw,paired=records(truth,tasks);summary=summarize(raw+paired)
    write_csv(a.out/'C1CLOSED_per_world.csv',raw);write_csv(a.out/'C1CLOSED_paired.csv',paired)
    write_csv(a.out/'C1CLOSED_bootstrap.csv',summary)
    plot(summary,a.out);report(summary,a.out)
    receipt=dict(schema='c1closed_analysis/1',tasks=len(tasks),worlds=40,K=8,
        raw_rows=len(raw),paired_rows=len(paired),summary_rows=len(summary),closure=closure,
        analyzer_sha256=digest(__file__),labels_sha256=digest(a.root/'labels.json'),
        inputs={q.name:digest(q) for q in sorted((a.root/'panel').glob('*_h300.json'))},
        outputs={q.name:digest(q) for q in a.out.glob('C1CLOSED*') if q.is_file()})
    with (a.out/'C1CLOSED_ANALYSIS.json').open('x') as f: json.dump(receipt,f,indent=1)
    print('COMPLETE',len(raw),len(paired),len(summary))


if __name__=='__main__': main()
