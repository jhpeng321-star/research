"""Disjoint host assignments; frozen inference with exact owned-PID guards."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def write(path,value):
    with path.open('x') as f:
        json.dump(value,f,indent=1);f.flush();os.fsync(f.fileno())


def incarnation(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]
    except FileNotFoundError:return None


def memory():
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    own=set()
    for q in (ROOT/'evidence').glob('EXPAND_*_OWNED.json'):
        r=json.loads(q.read_text())
        if r.get('device')=='cuda' and incarnation(r['pid'])==r['start_ticks']:own.add(r['pid'])
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    total=0
    for line in raw.splitlines():
        pid,mib=[v.strip() for v in line.split(',')]
        if pid.isdigit() and int(pid) in own and mib.isdigit():total+=int(mib)
    return free,total


def one(task,device,assignment):
    model,world,future,plant=task
    name=f'{model}_{plant}_w{world}_k{future}_h300'
    output=ROOT/'panel'/(name+'.json')
    assert not output.exists(), 'assignment must exclude every completed task'
    while True:
        if (ROOT/'EXPAND_STOP').exists():return dict(task=name,status='stopped_before_launch')
        free,total=memory() if device=='cuda' else (0,0)
        if device=='cuda' and free>=5120 and total<assignment['budget_bytes']/2**20-3072 and os.getloadavg()[0]<80:break
        if device=='cpu' and os.getloadavg()[0]<18:break
        time.sleep(30)
    cmd=[sys.executable,'-B','-u',str(ROOT/'worker.py'),'--model',model,'--world',str(world),
         '--future',str(future),'--plant',plant,'--seconds','300','--out','panel','--device',device]
    with (ROOT/'evidence'/f'EXPAND_{name}.log').open('x') as log:
        child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,'PYTHONPATH':'/root/autodl-tmp/c4/speedhead20261030/pythondeps',
                 'CUDA_VISIBLE_DEVICES':'0' if device=='cuda' else '',
                 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        ticks=incarnation(child.pid)
        own=dict(pid=child.pid,start_ticks=ticks,device=device,task=name,command=cmd,started=time.time())
        write(ROOT/'evidence'/f'EXPAND_{name}_OWNED.json',own)
        peak=0;guard=False
        while child.poll() is None:
            time.sleep(5)
            if device=='cuda':
                free,total=memory();peak=max(peak,total)
                if total*2**20>assignment['budget_bytes'] or free<1024:
                    guard=True
                    if child.poll() is None and incarnation(child.pid)==ticks:
                        child.terminate();child.wait(timeout=120)
                    break
        code=child.wait()
    result=dict(**own,returncode=code,finished=time.time(),peak_host_owned_mib=peak,memory_guard_stop=guard,
                status='complete' if code==0 and output.exists() else 'failed')
    write(ROOT/'evidence'/f'EXPAND_{name}_attempt1.json',result)
    return result


def main():
    lock=(ROOT/'EXPAND.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    a=json.loads((ROOT/'EXPAND_ASSIGNMENT.json').read_text())
    assert hashlib.sha256((ROOT/'worker.py').read_bytes()).hexdigest()==a['worker_sha256']
    write(ROOT/'EXPAND_DISPATCH.json',dict(pid=os.getpid(),start_ticks=incarnation(os.getpid()),assignment=a,started=time.time()))
    results=[]
    with ThreadPoolExecutor(max_workers=a['gpu_lanes']) as gpu,ThreadPoolExecutor(max_workers=2) as cpu:
        futures=[gpu.submit(one,t,'cuda',a) for t in a['gpu_tasks']]+[cpu.submit(one,t,'cpu',a) for t in a['cpu_tasks']]
        for future in as_completed(futures):
            result=future.result();results.append(result)
            with (ROOT/'evidence/EXPAND_PROGRESS.jsonl').open('a') as f:
                f.write(json.dumps(result)+'\n');f.flush();os.fsync(f.fileno())
    write(ROOT/'EXPAND_COMPLETE.json',dict(results=results,success=all(x['status']=='complete' for x in results),finished=time.time()))


if __name__=='__main__':main()
