"""Bounded detached dispatcher; records only and manages only its own children."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def write(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())


def incarnation(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]
    except FileNotFoundError:
        return None


def gpu_room():
    text = subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    return int(text.strip().splitlines()[0])


def tasks():
    result = []
    # Entire 40-world panel. No outcome-selected ordering/subsetting.
    for world in range(40):
        for model in ('wm2','bbtraj','bbid','historical'):
            for future in range(8 if model == 'wm2' else 1):
                result.append((model,world,future,'common'))
        result.append(('wm2',world,0,'production'))
    return result


def one(task, lane):
    model,world,future,plant = task
    name = f'{model}_{plant}_w{world}_k{future}_h300'
    output = ROOT/'panel'/(name+'.json')
    if output.exists():
        x = json.loads(output.read_text())
        assert (x['model'],x['world'],x['future'],x['plant'],x['seconds']) == (*task,300)
        assert x['identity']['source_sha256'] == hashlib.sha256((ROOT/'worker.py').read_bytes()).hexdigest()
        return dict(task=name,status='reused')
    if (ROOT/'STOP').exists():
        return dict(task=name,status='stopped_before_launch')
    # Each GPU worker has a 3 GiB allocator; at most two workers; contexts and
    # total-owned usage must stay <=12 GB. CPU lanes use only spare quota cores.
    while True:
        if (ROOT/'STOP').exists():
            return dict(task=name,status='stopped_before_launch')
        load = os.getloadavg()[0]
        if lane == 'cuda' and gpu_room() >= 5120 and load < 32:
            break
        if lane == 'cpu' and load < 18:
            break
        time.sleep(60)
    attempts = 1
    while (ROOT/'evidence'/f'{name}_attempt{attempts}.json').exists():
        attempts += 1
    command = [sys.executable,'-B','-u',str(ROOT/'worker.py'),'--model',model,
        '--world',str(world),'--future',str(future),'--plant',plant,
        '--seconds','300','--out','panel','--device',lane]
    env = dict(os.environ,PYTHONPATH='/root/autodl-tmp/c4/speedhead20261030/pythondeps',
               CUDA_VISIBLE_DEVICES='0' if lane == 'cuda' else '')
    logpath = ROOT/'evidence'/f'{name}_attempt{attempts}.log'
    with logpath.open('x') as log:
        child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        start = incarnation(child.pid)
        owned = dict(task=name,command=command,pid=child.pid,start_ticks=start,device=lane,started=time.time())
        write(ROOT/'evidence'/f'{name}_attempt{attempts}_OWNED.json',owned)
        while child.poll() is None:
            time.sleep(15)
            if lane == 'cuda':
                raw = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
                    '--format=csv,noheader,nounits'],text=True)
                total = 0
                # Only this dispatcher's receipt-owned child incarnations.
                own = {}
                for p in (ROOT/'evidence').glob('*_OWNED.json'):
                    receipt = json.loads(p.read_text())
                    if receipt.get('device') == 'cuda' and incarnation(receipt['pid']) == receipt['start_ticks']:
                        own[receipt['pid']] = receipt['start_ticks']
                for line in raw.splitlines():
                    pid,mem = [x.strip() for x in line.split(',')]
                    if pid.isdigit() and int(pid) in own and mem.isdigit():
                        total += int(mem)
                if total*2**20 > 12_000_000_000 or gpu_room() < 1024:
                    if child.poll() is None and incarnation(child.pid) == start:
                        child.terminate()  # exact Popen child, never a peer PID
                        child.wait(timeout=120)
                    owned['memory_guard_stop'] = True
                    break
        code = child.wait()
    receipt = dict(owned,returncode=code,finished=time.time(),log=str(logpath),output=str(output))
    write(ROOT/'evidence'/f'{name}_attempt{attempts}.json',receipt)
    if code:
        # Preserve a resource/import failure for explicit coordinator review.
        return dict(task=name,status='failed',returncode=code)
    assert output.exists()
    return dict(task=name,status='complete')


def main():
    lock = (ROOT/'DISPATCH.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert (ROOT/'evidence/CPU_VERIFIED.json').exists()
    verified=json.loads((ROOT/'evidence/CPU_VERIFIED.json').read_text())
    assert verified['worker_sha256'] == hashlib.sha256((ROOT/'worker.py').read_bytes()).hexdigest()
    canary=json.loads((ROOT/'evidence/CANARY_DONE.json').read_text())
    assert canary['returncode'] == 0
    roster = tasks()
    write(ROOT/'DISPATCH.json',dict(pid=os.getpid(),start_ticks=incarnation(os.getpid()),
        task_count=len(roster),cpu_workers=2,gpu_workers=2,
        worker_sha256=hashlib.sha256((ROOT/'worker.py').read_bytes()).hexdigest(),
        K=8,started=time.time(),reason='registered complete 40-world panel'))
    gpu = [t for t in roster if t[0] == 'wm2']
    cpu = [t for t in roster if t[0] != 'wm2']
    results = []
    with ThreadPoolExecutor(max_workers=2) as gp, ThreadPoolExecutor(max_workers=2) as cp:
        futures = [gp.submit(one,t,'cuda') for t in gpu]+[cp.submit(one,t,'cpu') for t in cpu]
        for f in as_completed(futures):
            results.append(f.result())
            # Append-only progress survives a coordinator disconnect.
            with (ROOT/'evidence/PROGRESS.jsonl').open('a') as handle:
                handle.write(json.dumps(results[-1])+'\n'); handle.flush(); os.fsync(handle.fileno())
    write(ROOT/'COMPLETE.json',dict(finished=time.time(),results=results,
        success=all(x['status'] in ('complete','reused') for x in results),count=len(results)))


if __name__ == '__main__':
    main()
