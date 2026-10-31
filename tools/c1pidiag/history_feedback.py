"""Fixed full-history conditional feedback on the previously registered shifted starts."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip
import json
from pathlib import Path
import time
import traceback
import worker as w
from auxiliary import derive
from feedback_support import init

OUT=w.ROOT/'aux_feedback_units'
EVIDENCE=w.ROOT/'aux_feedback_evidence'


def run(u):
    e,tag=u;target=OUT/(e['sid']+'__'+tag+'.json')
    if target.exists():return json.loads(target.read_text())['error']
    begin=time.monotonic()
    value=dict(unit=u,name=target.stem,sid=e['sid'],world=e['world'],cohort=e['cohort'],tag=tag,
        history='h60',population=False,residual=True,lanes='original',mode='aux_clh',error=None)
    try:
        r,info=derive(e);r=w.history(r,'h60');pred,model=w.predictor(tag,False,True,'original')
        seed=w.legacy.unit_seed(r,e['arm'],0);w.legacy.seed_all(seed)
        frames,wall=w.legacy.predict_closed(pred,r,e['arm'],float(max(info['clocks'])),plant='head')
        ep=w.endpoints(r,frames,e['arm'])
        if 120 not in info['clocks']:ep['120']=dict(ids={},n=0,n_truth=0,median=None,unavailable='no recorded target at this horizon')
        value.update(endpoints=ep,model=model,seed=seed,derived=info,wall=wall)
    except Exception:value['error']=traceback.format_exc()
    value['seconds']=time.monotonic()-begin;w.write(target,value)
    return value['error']


def main():
    assert not json.loads((w.ROOT/'legacy_evidence/COMPLETE.json').read_text())['errors']
    panel=json.loads((w.ROOT/'PANEL.json').read_text())['states']
    units=[(e,tag) for e in panel for tag in ('c1struct','wm13e','wm2','bbtraj')]
    sources={str(p):w.digest(p) for p in (Path(__file__),Path(w.__file__),w.ROOT/'auxiliary.py',w.ROOT/'feedback_support.py')}
    w.write(EVIDENCE/'DISPATCH.json',dict(units=len(units),workers=24,sources=sources))
    errors=[];begin=time.time()
    with ProcessPoolExecutor(max_workers=24,initializer=init) as pool:
        for i,f in enumerate(as_completed([pool.submit(run,u) for u in units]),1):
            error=f.result()
            if error:errors.append(error)
            if i%25==0 or i==len(units):print('history_feedback',i,'/',len(units),'errors',len(errors),flush=True)
    assert all(w.digest(p)==h for p,h in sources.items())
    w.write(EVIDENCE/'COMPLETE.json',dict(units=len(units),errors=errors,seconds=time.time()-begin,sources=sources))
    if errors:return
    target=w.ROOT/'AUX_FEEDBACK.jsonl.gz'
    with target.open('xb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as gz:
            for p in sorted(OUT.glob('*.json')):
                doc=json.loads(p.read_text());doc['_panel']='aux';doc['_source']=str(p)
                gz.write(json.dumps(doc,allow_nan=False,separators=(',',':')).encode()+b'\n')


if __name__=='__main__':main()
