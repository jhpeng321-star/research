"""Resumable 24-child CPU coordinator. Signals only owned unreaped children."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path.home()/'lrdiag'
VARIANTS=('baseline','residual_off','population','courtesy_off','yield_off','ramp_adapt_off')


def write(name,doc):
    p=ROOT/name; tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(doc)); tmp.replace(p)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--parity',type=int,choices=(0,1),required=True)
    ap.add_argument('--workers',type=int,default=24)
    ap.add_argument('--panel',choices=('e3','pairs300'),default='e3'); a=ap.parse_args()
    assert 1<=a.workers<=24
    assert ROOT.is_dir()
    for sub in ('results','logs','traces'): (ROOT/sub).mkdir(exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
        WMS_SPEED_PIVOT_DIAG_OFF='1',WMS_SPEED_BATCH_REPLAY='1',WMS_SPEED_FAST_CAV_DRIVE='1',
        WMS_SPEED_FAST_LONG_HISTORY='1',WMS_LOOPLONG_SPEED='1')
    # Baseline first; one full panel before later stochastic draws.
    jobs=[(g,k,v) for v in VARIANTS for k in range(8) for g in range(a.parity,36 if a.panel=='e3' else 48,2)]
    live={}; failed=[]; complete=0; skip=0
    prefix='' if a.panel=='e3' else 'PAIRS_'
    write(prefix+'OWNER.json',{'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
                        'parity':a.parity,'workers':a.workers,'jobs':len(jobs)})
    try:
        while jobs or live:
            available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
            while jobs and len(live)<a.workers and available>5*1024**2 and not (ROOT/'YIELD').exists():
                g,k,v=jobs.pop(0); name=('' if a.panel=='e3' else 'p300_')+f'g{g:02d}_{v}_k{k}_s600'
                out=ROOT/'results'/(name+'.json')
                if out.exists():
                    doc=json.loads(out.read_text())
                    assert doc['steps']==600 and doc['role']==('validation' if a.panel=='e3' else 'diagnostic')
                    skip+=1; continue
                f=(ROOT/'logs'/(name+'.log')).open('a')
                p=subprocess.Popen(['nice','-n','10',sys.executable,str(ROOT/'replay.py'),
                    '--group',str(g),'--draw',str(k),'--variant',v,'--panel',a.panel],env=env,cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
                live[p.pid]=(p,f,name)
                available-=700*1024
            for pid,(p,f,name) in list(live.items()):
                rc=p.poll()
                if rc is None: continue
                f.close(); del live[pid]
                if rc: failed.append({'job':name,'returncode':rc})
                else: complete+=1
            write(prefix+'PROGRESS.json',{'complete':complete,'skipped':skip,'remaining':len(jobs),
                'live':[{'pid':pid,'job':v[2]} for pid,v in live.items()],'failed':failed,'time':time.time()})
            time.sleep(5)
    finally:
        for p,f,name in live.values():
            p.terminate()
        for p,f,name in live.values():
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            f.close()
    write(prefix+'DONE.json',{'complete':complete,'skipped':skip,'failed':failed,'time':time.time()})


if __name__=='__main__': main()
