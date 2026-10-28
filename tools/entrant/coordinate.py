"""Detached, resumable task-owned CPU queue; never signals any other process."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
import traceback

HOME=Path.home()/'entrant'


def atomic(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,allow_nan=False));tmp.replace(path)


def work(unit):
    import run
    group,draw,variant,steps,panel,repeat=unit
    target=HOME/'results_v3'/f'{panel}_g{group:02d}_{variant}_k{draw}_s{steps}.json'
    if target.exists():return {'reused':str(target)}
    try:
        result=run.run(group,draw,variant,steps,panel)
        if repeat:
            again=run.run(group,draw,variant,steps,panel)
            strip=lambda r:{k:v for k,v in r.items() if k!='seconds'}
            assert strip(result)==strip(again),'same weights/seeds gave different outputs'
            result['repeat_identical']=True
        atomic(target,result)
        return {'result':str(target),'seconds':result['seconds'], 'audit':result['controller_audit']}
    except Exception:
        error={'unit':unit,'error':traceback.format_exc()}
        atomic(HOME/'errors'/f'{panel}_g{group:02d}_{variant}_k{draw}_{time.time_ns()}.json',error)
        return error


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parity',type=int,choices=(0,1),required=True)
    p.add_argument('--workers',type=int,default=20)
    p.add_argument('--canary',action='store_true')
    p.add_argument('--panel',default='e3',choices=('e3','pairs300'))
    a=p.parse_args();assert 1<=a.workers<=24
    os.nice(10)
    import run
    run.setup()
    for d in ('results_v3','errors'):(HOME/d).mkdir(exist_ok=True)
    import fcntl
    with (HOME/('CANARY.lock' if a.canary else 'QUEUE.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pid=os.getpid();ticks=Path(f'/proc/{pid}/stat').read_text().split()[21]
        owner={'pid':pid,'start_ticks':ticks,'parity':a.parity,'workers':a.workers,
               'panel':a.panel,'canary':a.canary,'started':time.time()}
        atomic(HOME/('CANARY_OWNER.json' if a.canary else 'OWNER.json'),owner)
        units=([(1,0,'entrant_on',600,'e3',True), (1,0,'baseline',600,'e3',True)] if a.canary else
            [(g,k,v,600,a.panel,g==1 and k==0 and v=='entrant_on')
             for v in ('entrant_on','baseline','ramp_adapt_off','entrant_on_ramp_off')
             for g in range(36 if a.panel=='e3' else 48) if g%2==a.parity for k in range(8)])
        results=[]
        with ProcessPoolExecutor(max_workers=1 if a.canary else a.workers) as pool:
            for f in as_completed([pool.submit(work,u) for u in units]):
                r=f.result();results.append(r);print(json.dumps(r),flush=True)
                atomic(HOME/('CANARY_PROGRESS.json' if a.canary else 'PROGRESS.json'),
                       {'owner':owner,'done':len(results),'total':len(units),
                        'errors':sum('error' in x for x in results),'updated':time.time()})
        atomic(HOME/('CANARY_DONE.json' if a.canary else 'DONE.json'),results)


if __name__=='__main__':main()
