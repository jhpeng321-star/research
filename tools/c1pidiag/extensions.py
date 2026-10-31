"""Single built-in component switches after primary and genuine-history panels."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
import traceback
import worker as w
from feedback_support import init as init_feedback

OUT=w.ROOT/'extra_units'
EVIDENCE=w.ROOT/'extra_evidence'
SWITCHES={
    'residual_net_off':dict(residual_accel_off=True,diag_zero_temporal=True),
    'courtesy_off':dict(follower_courtesy_off=True),
    'yield_off':dict(mainline_merge_yield_learnable=False),
    'ramp_adapt_off':dict(ramp_speed_adapt_learnable=False),
}


def run(u):
    e,tag,switch,mode=u
    target=OUT/('__'.join((e['sid'],tag,switch,mode))+'.json')
    if target.exists():
        doc=json.loads(target.read_text());assert doc['unit']==list(u)
        return doc['error']
    hist='h5' if e['cohort']=='wall_same' else 'snapshot'
    value=dict(unit=u,name=target.stem,sid=e['sid'],world=e['world'],cohort=e['cohort'],
        tag=tag,history=hist,population=False,residual=True,lanes='original',component=switch,mode=mode,error=None,pid=os.getpid())
    begin=time.monotonic();previous={};m=None
    try:
        r=w.history(w.record(e),hist)
        pred,info=w.predictor(tag,False,True,'original');m=pred.model
        previous={k:getattr(m,k) for k in SWITCHES[switch]}
        for k,v in SWITCHES[switch].items():setattr(m,k,v)
        seed=w.legacy.unit_seed(r,e['arm'],0);w.legacy.seed_all(seed)
        if mode=='tf':frames=w.legacy.predict_teacher(pred,r,e['arm'],120.)
        else:frames,_=w.legacy.predict_closed(pred,r,e['arm'],120.,plant='head')
        value.update(endpoints=w.endpoints(r,frames,e['arm']),model=info,seed=seed,switches=SWITCHES[switch])
    except Exception:value['error']=traceback.format_exc()
    finally:
        for k,v in previous.items():setattr(m,k,v)
    value['seconds']=time.monotonic()-begin;w.write(target,value)
    return value['error']


def execute(units,pool,label):
    futures=[pool.submit(run,u) for u in units];errors=[]
    for i,f in enumerate(as_completed(futures),1):
        err=f.result()
        if err:errors.append(err)
        if i%25==0 or i==len(futures):print(label,i,'/',len(futures),'errors',len(errors),flush=True)
    return errors


def select(panel):
    bb={}
    for e in panel:
        hist='h5' if e['cohort']=='wall_same' else 'snapshot'
        p=w.UNITS/('__'.join((e['sid'],'bbtraj',hist,'inf','r1','original','tf'))+'.json')
        bb[e['sid']]=json.loads(p.read_text())
    groups={}
    for p in OUT.glob('*__tf.json'):
        r=json.loads(p.read_text());assert not r['error']
        ids=set(r['endpoints']['60']['ids']) & set(bb[r['sid']]['endpoints']['60']['ids'])
        if ids:
            groups.setdefault((r['cohort'],r['tag'],r['component']),[]).append(
                float(w.np.median([abs(r['endpoints']['60']['ids'][i]['ds']) for i in ids])))
    selected=[]
    for cohort in ('pi_u_same','wall_same'):
        for tag in ('wm13e','wm2'):
            n=sum(e['cohort']==cohort for e in panel)
            options=[(k,v) for k,v in groups.items() if k[:2]==(cohort,tag) and len(v)==n]
            if not options:continue
            k,v=min(options,key=lambda kv:(float(w.np.median(kv[1])),str(kv[0])))
            baseline=[]
            for e in panel:
                if e['cohort']!=cohort:continue
                hist='h5' if cohort=='wall_same' else 'snapshot'
                p=w.UNITS/('__'.join((e['sid'],tag,hist,'inf','r1','original','tf'))+'.json')
                r=json.loads(p.read_text())
                ids=set(r['endpoints']['60']['ids']) & set(bb[e['sid']]['endpoints']['60']['ids'])
                if ids:baseline.append(float(w.np.median([abs(r['endpoints']['60']['ids'][i]['ds']) for i in ids])))
            score=float(w.np.median(v));base=float(w.np.median(baseline))
            selected.append(dict(cohort=cohort,tag=tag,component=k[2],score60=score,baseline60=base,improves=score<base))
    w.write(EVIDENCE/'SELECTED.json',selected)
    return [(e,s['tag'],s['component'],'clh') for e in panel for s in selected if e['cohort']==s['cohort'] and s['improves']]


def main():
    assert os.getpriority(os.PRIO_PROCESS,0)==19
    begin=time.time()
    while not (w.ROOT/'aux_evidence/COMPLETE.json').exists():
        assert time.time()-begin<24*3600
        time.sleep(15)
    done=json.loads((w.ROOT/'aux_evidence/COMPLETE.json').read_text());assert not done['errors']
    assert os.getloadavg()[0]<100
    panel=json.loads((w.ROOT/'PANEL.json').read_text())['states']
    units=[(e,tag,s,'tf') for e in panel for tag in ('wm13e','wm2') for s in SWITCHES
           if s!='ramp_adapt_off' or tag=='wm2']
    sources={str(p):w.digest(p) for p in (Path(__file__),Path(w.__file__),
        w.frozen.FIX/'src/wms/model/wms.py',w.frozen.WM_CONFIG)}
    w.write(EVIDENCE/'DISPATCH.json',dict(units=len(units),workers=24,sources=sources,started=time.time()))
    begin=time.time()
    with ProcessPoolExecutor(max_workers=24,initializer=init_feedback) as pool:
        errors=execute(units,pool,'extra_tf')
        w.write(EVIDENCE/'TF_DONE.json',dict(units=len(units),errors=errors))
        if errors:return
        selected=select(panel)
        errors=execute(selected,pool,'extra_clh')
    assert all(w.digest(p)==h for p,h in sources.items())
    w.write(EVIDENCE/'COMPLETE.json',dict(tf=len(units),clh=len(selected),errors=errors,seconds=time.time()-begin,sources=sources))


if __name__=='__main__':main()
