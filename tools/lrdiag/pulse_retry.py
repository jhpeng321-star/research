"""Repair projection-check-only pulse failures serially inside an existing CPU slot."""
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT=Path.home()/'lrdiag'


def atomic(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d));tmp.replace(p)


def one():
    # A model worker calls this only AFTER its paired rollout is written. It
    # waits while the one pulse child computes: <=24 active workers, not 24+N.
    if not (ROOT/'PULSE_DONE.json').exists():return
    serial=(ROOT/'PULSE_RETRY_SERIAL.lock').open('a')
    try:fcntl.flock(serial,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:serial.close();return
    try:
        queue=ROOT/'PULSE_RETRY_QUEUE.json'
        if queue.exists():q=json.loads(queue.read_text())
        else:
            old=json.loads((ROOT/'PULSE_DONE.json').read_text())
            q={'reason':'initial alignment check included unmapped internal-lane coordinates; '
                        'mapped initial positions/speeds are exact; model dynamics unchanged',
               'pending':[r['job'] for r in old['failed']],'completed':[],'failed':[]}
        while q['pending']:
            name=q['pending'].pop(0)
            if (ROOT/'pulse_results'/(name+'.json')).exists():continue
            m=re.fullmatch(r's(9698[0-9]{2})_k([0-7])',name);assert m,name
            seed,k=int(m[1]),int(m[2]);assert 969824<=seed<=969835
            atomic(queue,q)
            log=ROOT/'pulse_logs'/(name+'.retry.log')
            with log.open('a') as f:
                child=subprocess.Popen([sys.executable,str(ROOT/'pulses.py'),'--seed',str(seed),'--draw',str(k)],
                    cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,
                    stdout=f,stderr=subprocess.STDOUT)
                token=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
                atomic(ROOT/'PULSE_RETRY_CHILD.json',{'pid':child.pid,'start_ticks':token,
                    'parent_owned_worker':os.getpid(),'job':name,'time':time.time()})
                try:rc=child.wait()
                except BaseException:
                    if child.poll() is None:
                        child.send_signal(signal.SIGCONT);child.terminate()
                        try:child.wait(timeout=30)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                    q['pending'].insert(0,name);atomic(queue,q)
                    raise
            q['completed' if rc==0 else 'failed'].append({'job':name,'returncode':rc})
            atomic(queue,q)
            break
        if not q['pending']:
            n=len(list((ROOT/'pulse_results').glob('s*_k*.json')))
            atomic(ROOT/'PULSE_RETRY_DONE.json',{'n_results':n,'expected':96,'failed':q['failed'],
                'status':'complete' if n==96 and not q['failed'] else 'needs_investigation'})
    finally:
        fcntl.flock(serial,fcntl.LOCK_UN);serial.close()
