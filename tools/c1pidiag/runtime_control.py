"""All fixed original baselines in the archived refresh source, same current CPU."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip
import hashlib
import json
import os
import sys
import time
import traceback

ROOT=Path('/root/autodl-tmp/c4/c1pidiag20261031')
REF=Path('/root/autodl-tmp/c4/c1refresh')
sys.path[:0]=[str(REF/'code/src'),str(REF/'code'),str(REF/'code/tools/c1refresh')]
for k in list(os.environ):
    if k.startswith('WMS_SPEED_') or k in ('C4_EXECUTION_PARITY','C4_SUMO_LEADER_RULE','C4_EXECUTION_DRIVER'):
        os.environ.pop(k)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import c1r_run as legacy
import numpy as np
import torch
from wms.eval.metrics import frame_at,present_by_id
from wms.eval.normalize import recorded_for

BAD=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vehicle_params','driver_params','vtype')
CACHE={}


def audit(event,args):
    if event=='subprocess.Popen':raise PermissionError('no child commands from inference')
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=os.fsdecode(args[0]).lower()
        if any(b in p for b in BAD) or p.endswith(('.xml','.xml.gz')):raise PermissionError('excluded path')


def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=1,allow_nan=False)


def init():
    torch.set_num_threads(1);sys.addaudithook(audit);legacy.install_pivot_diag_stub()


def run(u):
    e,tag=u;target=ROOT/'legacy_units'/(e['sid']+'__'+tag+'.json')
    if target.exists():return json.loads(target.read_text())['error']
    begin=time.monotonic()
    value=dict(unit=u,name=target.stem,sid=e['sid'],world=e['world'],cohort=e['cohort'],tag=tag,
        history='h5' if e['cohort']=='wall_same' else 'snapshot',population=False,residual=True,lanes='original',mode='tf',error=None)
    try:
        assert not any(b in str(e).lower() for b in BAD)
        path=REF/'states/states'/(e['sid']+'.json.gz')
        assert legacy.sha256_file(path)==e['file_sha256']
        with gzip.open(path,'rt') as f:r=json.load(f)
        seed=int(r.get('seed',0));assert not (964000<=seed<=964199 or 962000<=seed<=962999 or 977000<=seed<=977999)
        if tag not in CACHE:CACHE[tag]=legacy.load_predictor(REF,tag)
        pred=CACHE[tag];r=legacy.with_snapshot_history(pred,r)
        seed=legacy.unit_seed(r,e['arm'],0);legacy.seed_all(seed)
        frames=legacy.predict_teacher(pred,r,e['arm'],120.)
        rec=recorded_for(r,e['arm']);roster={v['id'] for v in r['initial_state']['vehicles'] if not v.get('is_cav',False)}
        ep={}
        for h in (60,120):
            p=present_by_id(frame_at(frames,r['t0_s']+h));t=present_by_id(frame_at(rec,r['t0_s']+h))
            ids=sorted(roster & set(p) & set(t))
            errors={i:dict(ds=float(p[i]['s'])-float(t[i]['s']),dv=float(p[i]['v'])-float(t[i]['v']),lane_match=p[i]['lane']==t[i]['lane']) for i in ids}
            ep[str(h)]=dict(ids=errors,n=len(ids),n_truth=len(roster & set(t)),median=float(np.median([abs(v['ds']) for v in errors.values()])) if ids else None)
        value.update(endpoints=ep,seed=seed,model=dict(sha256=legacy.MODELS[tag][2],runtime=str(REF/'code')))
    except Exception:value['error']=traceback.format_exc()
    value['seconds']=time.monotonic()-begin;write(target,value)
    return value['error']


def main():
    assert os.getpriority(os.PRIO_PROCESS,0)==19
    begin=time.time()
    while not (ROOT/'extra_evidence/COMPLETE.json').exists():
        assert time.time()-begin<24*3600;time.sleep(15)
    assert not json.loads((ROOT/'extra_evidence/COMPLETE.json').read_text())['errors']
    assert os.getloadavg()[0]<100
    panel=json.loads((ROOT/'PANEL.json').read_text())['states']
    units=[(e,t) for e in panel for t in ('c1struct','wm13e','bbtraj')]
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(legacy.__file__),REF/'code/src/wms/model/wms.py')}
    write(ROOT/'legacy_evidence/DISPATCH.json',dict(units=len(units),sources=sources,torch=torch.__version__))
    errors=[];begin=time.time()
    with ProcessPoolExecutor(max_workers=24,initializer=init) as pool:
        for i,f in enumerate(as_completed([pool.submit(run,u) for u in units]),1):
            error=f.result()
            if error:errors.append(error)
            if i%25==0 or i==len(units):print('legacy',i,'/',len(units),'errors',len(errors),flush=True)
    write(ROOT/'legacy_evidence/COMPLETE.json',dict(units=len(units),errors=errors,seconds=time.time()-begin,sources=sources))


if __name__=='__main__':main()
