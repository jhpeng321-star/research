"""Prospectively registered t0+60 s history panel; never changes original starts."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import copy
import json
import os
from pathlib import Path
import time
import traceback
import worker as w

OUT = w.ROOT/'aux_units'
EVIDENCE = w.ROOT/'aux_evidence'
SOURCE_RECORD = w.record
DERIVED = {}


def history_from_observed(frames, cutoff):
    """Consume only observed ticks, with stable transmitted IDs and causal lane carry."""
    frames = [f for f in frames if cutoff-60.-1e-8 <= float(f['time_s']) <= cutoff+1e-8]
    assert len(frames)==121
    assert all(abs(float(f['time_s'])-(cutoff-60.+.5*i))<1e-6 for i,f in enumerate(frames))
    ids=sorted({str(v['id']) for f in frames for v in f['vehicles'] if v.get('present',True)})
    col={vid:i for i,vid in enumerate(ids)}
    kin=w.np.zeros((121,len(ids),7),dtype=w.np.float32)
    mask=w.np.zeros((121,len(ids)),dtype=bool)
    last={}
    for ti,f in enumerate(frames):
        for v in f['vehicles']:
            if not v.get('present',True):continue
            vid=str(v['id']);lane=int(v['lane'])
            if lane<0:
                if vid not in last:continue
                lane=last[vid]
            else:last[vid]=lane
            k=col[vid]
            kin[ti,k,0]=float(v['s']);kin[ti,k,1+lane]=1.
            kin[ti,k,4]=float(v['v']);kin[ti,k,6]=5.
            mask[ti,k]=True
    target=mask[:-1]&mask[1:]
    return dict(ids=ids,kin=kin.tolist(),mask=mask.tolist(),target_mask=target.tolist(),
        history_s=60.,history_stride_s=1.,dt_s=.5,available_span_s=60.,run_start_limited=False,
        observed_min_s=cutoff-60.,observed_max_s=cutoff,
        note='same-arm recorded observations at the prospectively shifted start; not original-t0 inputs')


def derive(e):
    if e['sid'] in DERIVED:return DERIVED[e['sid']]
    r=SOURCE_RECORD(e)
    frames=w.recorded_for(r,e['arm'])
    t0=float(r['t0_s'])+60.
    initial=w.frame_at(frames,t0)
    assert abs(float(initial['time_s'])-t0)<1e-6
    hist=history_from_observed(frames,t0)
    # Explicit future-noninterference check: deleting every future frame cannot
    # change any history byte. Remaining future labels are used only for scoring.
    assert hist==history_from_observed([f for f in frames if float(f['time_s'])<=t0],t0)
    out=copy.deepcopy(r)
    out['t0_s']=t0
    out['initial_state']=copy.deepcopy(initial)
    out['history']=hist
    out['recorded']={e['arm']:[f for f in frames if float(f['time_s'])>=t0]}
    out['arms']=[e['arm']]
    out['recorded_horizon_steps']={e['arm']:len(out['recorded'][e['arm']])}
    # No stale original-start action timeline may overwrite the CAV teacher.
    out['cav_controllers']={'arm':e['arm']}
    end=max(float(f['time_s']) for f in frames)-t0
    clocks=[h for h in (60,120) if end>=h-.5-1e-6]
    assert 60 in clocks
    info=dict(sid=e['sid'],original_t0=r['t0_s'],t0=t0,history_min=t0-60.,
        history_max=t0,n_history_ticks=121,n_history_ids=len(hist['ids']),
        n_t0=len(initial['vehicles']),remaining_s=end,clocks=clocks,
        history_sha256=w.hashlib.sha256(json.dumps(hist,sort_keys=True).encode()).hexdigest(),
        future_noninterference=True,source_sha256=e['file_sha256'])
    DERIVED[e['sid']]=(out,info)
    return out,info


def run(u):
    e,tag,hist,pop=u
    name='__'.join((e['sid'],tag,hist,'pop' if pop else 'inf'))
    target=OUT/(name+'.json')
    if target.exists():
        old=json.loads(target.read_text());assert old['unit']==list(u)
        return old['error']
    value=dict(name=name,unit=u,sid=e['sid'],world=e['world'],cohort=e['cohort'],
        tag=tag,history=hist,population=pop,residual=True,lanes='original',mode='aux_tf',error=None,pid=os.getpid())
    begin=time.monotonic()
    try:
        r,info=derive(e)
        r=w.history(r,hist)
        pred,model=w.predictor(tag,pop,True,'original')
        seed=w.legacy.unit_seed(r,e['arm'],0);w.legacy.seed_all(seed)
        frames=w.legacy.predict_teacher(pred,r,e['arm'],float(max(info['clocks'])))
        ep=w.endpoints(r,frames,e['arm'])
        for h in (60,120):
            if h not in info['clocks']:
                ep[str(h)]=dict(ids={},n=0,n_truth=0,median=None,unavailable='no recorded target at this horizon')
        value.update(endpoints=ep,model=model,seed=seed,derived=info)
    except Exception:value['error']=traceback.format_exc()
    value['seconds']=time.monotonic()-begin
    w.write(target,value)
    return value['error']


def execute(units,pool,label):
    futures=[pool.submit(run,u) for u in units]
    errors=[]
    for i,f in enumerate(as_completed(futures),1):
        err=f.result()
        if err:errors.append(err)
        if i%25==0 or i==len(futures):print(label,i,'/',len(futures),'errors',len(errors),flush=True)
    return errors


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');args=ap.parse_args()
    assert os.getpriority(os.PRIO_PROCESS,0)==19
    if args.wait:
        start=time.time()
        while not (w.EVIDENCE/'COMPLETE.json').exists():
            assert time.time()-start<24*3600,'original campaign incomplete after 24 h'
            time.sleep(15)
    original=json.loads((w.EVIDENCE/'COMPLETE.json').read_text())
    assert not original['errors'],'original campaign requires inspection'
    assert os.getloadavg()[0]<100
    panel=json.loads((w.ROOT/'PANEL.json').read_text())['states']
    units=[(e,tag,hist,pop) for e in panel for tag in ('c1struct','wm13e','wm2')
           for hist in ('snapshot','h5','h60') for pop in (False,True)]
    units += [(e,'bbtraj','h60',False) for e in panel]
    sources={str(p):w.digest(p) for p in (Path(__file__),Path(w.__file__),Path(w.legacy.__file__),
        w.frozen.FIX/'src/wms/model/wms.py',w.frozen.WM_CONFIG)}
    begin=time.time()
    w.write(EVIDENCE/'DISPATCH.json',dict(units=len(units),workers=24,sources=sources,started=begin))
    with ProcessPoolExecutor(max_workers=24,initializer=w.init) as pool:
        checks=[(e,'wm13e','h60',False) for e in (panel[0],next(e for e in panel if e['cohort']=='wall_same'))]
        errors=execute(checks,pool,'aux_preflight')
        w.write(EVIDENCE/'PREFLIGHT.json',dict(errors=errors))
        if errors:return
        errors=execute(units,pool,'aux')
    assert all(w.digest(p)==h for p,h in sources.items())
    w.write(EVIDENCE/'COMPLETE.json',dict(units=len(units),errors=errors,seconds=time.time()-begin,sources=sources))


if __name__=='__main__':main()
