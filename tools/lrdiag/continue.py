"""Own sequential continuation: wait for E3, pulse qualification, pairs300."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path.home()/'lrdiag'


def pulses():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
        WMS_SPEED_COMPACT_SLOTS='1',MALLOC_MMAP_THRESHOLD_='16777216',MALLOC_ARENA_MAX='2',WMS_SPEED_FAST_MERGE_LANE='1',
        WMS_SPEED_PIVOT_DIAG_OFF='1',WMS_SPEED_BATCH_REPLAY='1',WMS_SPEED_FAST_CAV_DRIVE='1',
        WMS_SPEED_FAST_LONG_HISTORY='1',WMS_LOOPLONG_SPEED='1')
    (ROOT/'pulse_results').mkdir(exist_ok=True);(ROOT/'pulse_logs').mkdir(exist_ok=True)
    jobs=[(s,k) for k in range(8) for s in range(969824,969836)]
    live={};fail=[]
    try:
        while jobs or live:
            available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
            while jobs and len(live)<24 and available>5*1024**2 and not (ROOT/'YIELD').exists():
                seed,k=jobs.pop(0);name=f's{seed}_k{k}'
                if (ROOT/'pulse_results'/(name+'.json')).exists():continue
                f=(ROOT/'pulse_logs'/(name+'.log')).open('a')
                p=subprocess.Popen(['nice','-n','10',sys.executable,str(ROOT/'pulses.py'),
                    '--seed',str(seed),'--draw',str(k)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                    stdout=f,stderr=subprocess.STDOUT)
                live[p.pid]=(p,f,name);available-=800*1024
            for pid,(p,f,name) in list(live.items()):
                rc=p.poll()
                if rc is None:continue
                f.close();del live[pid]
                if rc:fail.append({'job':name,'rc':rc})
            tmp=ROOT/'PULSE_PROGRESS.tmp'
            tmp.write_text(json.dumps({'remaining':len(jobs),'live':[{'pid':pid,'job':x[2]} for pid,x in live.items()],'failed':fail}))
            tmp.replace(ROOT/'PULSE_PROGRESS.json');time.sleep(5)
    finally:
        for p,f,name in live.values():p.terminate()
        for p,f,name in live.values():
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            f.close()
    (ROOT/'PULSE_DONE.json').write_text(json.dumps({'failed':fail}))
    return not fail


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parity',type=int,choices=(0,1),required=True);a=ap.parse_args()
    while not (ROOT/'DONE.json').exists():time.sleep(30)
    d=json.loads((ROOT/'DONE.json').read_text())
    if d['failed']:raise RuntimeError('E3 failed units need investigation before continuation')
    if a.parity==0:
        pulses()  # non-verdict diagnostic failures do not invalidate the E3 panel
    rc=subprocess.call([sys.executable,str(ROOT/'launch.py'),'--parity',str(a.parity),'--panel','pairs300','--workers','24'])
    if rc:raise RuntimeError(f'pairs300 launcher exit {rc}')
    (ROOT/'ALL_DONE.json').write_text(json.dumps({'parity':a.parity,'time':time.time()}))


if __name__=='__main__':main()
