"""Durable local dispatcher. CPU on gpuA; GPU on gpuR ONLY without peer compute.

Every task is claimed exclusively before dispatch; claims and remote outputs are
never deleted. Restart only collects existing claims; it never repeats them.
No process is signalled, no simulator/environment/training is launched.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from .audit import digest, safe
from .core import TIMES

LOCK = threading.Lock()
OWN_GPU_PIDS = set()


def task_key(task):
    s, t, f = task
    return f's{s}_t{t}_k{f}'


def save_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=1, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def gpu_free():
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'gpuR',
        'nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=30)
    if p.returncode:
        return False
    try:
        pids = {int(s.strip()) for s in p.stdout.splitlines() if s.strip()}
    except ValueError:
        return False
    with LOCK:
        return not (pids-OWN_GPU_PIDS)


def cpu_free(host='gpuA', reserve_cores=4, reserve_gib=8):
    # Read only cgroup counters, leaving a four-core and eight-GiB admission
    # margin. All four CPU lanes are nice19/single-thread on the gpuA host.
    code = '''
from pathlib import Path
import json,time
r=Path('/sys/fs/cgroup')
def usage():
    return int(dict(s.split() for s in (r/'cpu.stat').read_text().splitlines())['usage_usec'])
a=usage(); t=time.monotonic(); time.sleep(.5); b=usage(); elapsed=time.monotonic()-t
q,p=(r/'cpu.max').read_text().split()
quota=float(q)/float(p) if q!='max' else 20.
free=int((r/'memory.max').read_text())-int((r/'memory.current').read_text())
print(json.dumps(dict(cores=(b-a)/elapsed/1e6,quota=quota,free=free)))
'''
    try:
        p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host,
                            '/root/miniconda3/bin/python -B -'], input=code,
                           text=True, capture_output=True, timeout=30)
        d = json.loads(p.stdout)
        return (not p.returncode and d['cores'] <= d['quota']-reserve_cores
                and d['free'] >= reserve_gib*2**30)
    except (ValueError, KeyError, subprocess.TimeoutExpired):
        return False


def fetch(host, remote, local):
    if local.exists():
        return
    subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                    host+':'+remote, str(local)], check=True, timeout=120)


def task_path(root, key):
    receipt = root/'recovered_tasks'/(key+'.receipt.json')
    if receipt.exists():
        d = json.loads(receipt.read_text())
        path = root/d['path']
        assert path.parent.resolve() == (root/'recovered_tasks').resolve()
        assert digest(path) == d['sha256'], 'recovered transfer changed'
        return path
    return root/'tasks'/(key+'.json')


def run_task(root, tag, task, host, device, gate, reservation_pid=None, short=False):
    key = task_key(task)
    remote = '/root/wmvalue20261031/'+('short_cells_v1' if short else 'cells_v1')+'/'+key+'.json'
    claim_path = root/'claims'/(key+'.json')
    with LOCK:
        if (root/'tasks'/(key+'.json')).exists():
            return False
        if reservation_pid is None:
            if claim_path.exists():
                return False
            save_new(claim_path, dict(task=list(task), host=host, device=device, remote=remote,
                                      dispatcher_pid=os.getpid(), time=time.time()))
        else:
            # Reserved disjoint CPU-assist partition: a claim is a queued task,
            # not permission to repeat a previously started SSH/remote process.
            claim = json.loads(claim_path.read_text())
            assert claim['dispatcher_pid'] == reservation_pid == os.getpid()
            assert claim['host'] == host and claim['device'] == device
            assert claim['task'] == list(task) and claim['remote'] == remote
            assert not (root/'claims'/(key+'.ssh.json')).exists(), 'already dispatched'
    module = 'tools.wmvalue.worker'
    args = [module, '--prefixes', '/root/wmvalue20261031/reserved_prefix_v1.zip',
            '--gate', '/root/wmvalue20261031/cpu_audit_v3.json',
            '--seed', str(task[0]), '--t0', str(task[1]), '--future', str(task[2]),
             '--device', device, '--out', remote]
    if short:
        args.append('--short')
    code = ("import sys,runpy; sys.path[:0]=["+repr('/root/wmvalue20261031/'+tag)+','+
            repr('/root/autodl-tmp/c4/speedhead20261030/pythondeps')+
            ']; sys.argv='+repr(args)+"; runpy.run_module('tools.wmvalue.worker',run_name='__main__')")
    command = ('nice -n 19 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '+
               '/root/miniconda3/bin/python -B -u -')
    begun = time.monotonic()
    remote_pid = None
    done_sha256 = None
    with (root/'logs'/(key+'.log')).open('x', encoding='utf-8') as log:
        proc = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                                 '-o', 'ServerAliveInterval=30', host, command],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        save_new(root/'claims'/(key+'.ssh.json'), dict(pid=proc.pid, host=host, task=list(task)))
        proc.stdin.write(code)
        proc.stdin.close()
        for line in proc.stdout:
            log.write(line)
            log.flush()
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get('status') == 'START':
                remote_pid = int(d['pid'])
                save_new(root/'claims'/(key+'.remote.json'), d)
                if device == 'cuda':
                    with LOCK:
                        OWN_GPU_PIDS.add(remote_pid)
            if d.get('status') == 'DONE':
                done_sha256 = d['sha256']
        code = proc.wait()
    if device == 'cuda' and remote_pid is not None:
        with LOCK:
            OWN_GPU_PIDS.discard(remote_pid)
    if code:
        save_new(root/'claims'/(key+'.error.json'), dict(returncode=code, task=list(task),
                  seconds=time.monotonic()-begun, log=str(root/'logs'/(key+'.log'))))
        raise RuntimeError(f'{key} failed rc={code}; inspect preserved log, no automatic retry')
    target = root/'tasks'/(key+'.json')
    try:
        fetch(host, remote, target)
    except subprocess.CalledProcessError:
        # SCP can return nonzero after all bytes arrived. A worker-DONE hash
        # proves complete collection; never repeat inference for this case.
        if not target.is_file() or done_sha256 is None or digest(target) != done_sha256:
            from .recover_completed import recover
            target = recover(root, key)
    assert done_sha256 is not None and digest(target) == done_sha256, 'worker DONE/output hash mismatch'
    d = json.loads(target.read_text())
    assert (d['seed'], d['t0'], d['future']) == task
    assert d['end_time'] == (task[1]+90 if short else 900)
    assert d['gate_sha256'] == gate and set(d['results']['windows']) == {'60', '90'}
    save_new(root/'claims'/(key+'.complete.json'), dict(sha256=digest(target),
        host=host, seconds=time.monotonic()-begun, device=device))
    print(json.dumps(dict(status='COMPLETE', task=key, device=device,
                           seconds=round(time.monotonic()-begun, 2))), flush=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--gate', required=True)
    p.add_argument('--cpu-lanes', type=int, default=4, choices=range(1, 5))
    p.add_argument('--gpu-lanes', type=int, default=3, choices=range(1, 4))
    p.add_argument('--real')
    p.add_argument('--model')
    a = p.parse_args()
    if bool(a.real) != bool(a.model):
        raise ValueError('automatic finalization needs both real arrays and frozen model')
    root = safe(a.root)
    assert root.parent.is_dir()
    root.mkdir(exist_ok=True)
    for name in ('tasks', 'claims', 'logs'):
        (root/name).mkdir(exist_ok=True)
    # One local dispatcher, independent of chat/session lifetime.
    lock = (root/'DISPATCH.lock').open('a+b')
    import msvcrt
    lock.seek(0)
    if lock.read(1) == b'':
        lock.write(b'1')
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    gate = digest(safe(a.gate))
    tasks = [(s, t, f) for s in range(969432, 969440) for t in TIMES for f in range(8)]
    # Existing claims are never redispatched, even if the old SSH link vanished.
    # Collect remote-completed claims; leave live/failed ones for explicit audit.
    for task in tasks:
        key = task_key(task)
        claim = root/'claims'/(key+'.json')
        if claim.exists() and not (root/'tasks'/(key+'.json')).exists():
            d = json.loads(claim.read_text())
            try:
                fetch(d['host'], d['remote'], root/'tasks'/(key+'.json'))
            except subprocess.CalledProcessError:
                print('EXISTING CLAIM retained, not redispatched:', key, flush=True)

    failed = threading.Event()

    def lane(host, device):
        while not failed.is_set():
            with LOCK:
                pending = [task for task in tasks if not (root/'claims'/(task_key(task)+'.json')).exists()
                           and not (root/'tasks'/(task_key(task)+'.json')).exists()]
            if not pending:
                return
            if device == 'cuda' and not gpu_free():
                time.sleep(30)
                continue
            if device == 'cpu' and not cpu_free():
                time.sleep(30)
                continue
            try:
                run_task(root, a.tag, pending[0], host, device, gate)
            except Exception as exc:
                failed.set()
                print('DISPATCH PAUSED:', repr(exc), flush=True)
                return

    with ThreadPoolExecutor(max_workers=a.cpu_lanes+a.gpu_lanes) as pool:
        jobs = [pool.submit(lane, 'gpuA', 'cpu') for _ in range(a.cpu_lanes)]
        jobs += [pool.submit(lane, 'gpuR', 'cuda') for _ in range(a.gpu_lanes)]
        for job in jobs:
            job.result()
    complete = [task for task in tasks if (root/'tasks'/(task_key(task)+'.json')).exists()]
    if len(complete) != len(tasks):
        print(json.dumps(dict(status='INCOMPLETE', completed=len(complete), expected=len(tasks),
                              no_automatic_repeat=True)), flush=True)
        return
    save_new(root/'COMPLETE.json', dict(schema='wmvalue_inference_complete/1', tasks=320,
        hashes={task_key(t)+'.json': digest(root/'tasks'/(task_key(t)+'.json')) for t in tasks}))
    print('ALL INFERENCE COMPLETE; frozen A-E evaluation may run', flush=True)
    if a.real:
        import sys
        subprocess.run([sys.executable, '-B', '-m', 'tools.wmvalue.finalize',
            '--tasks', str(root/'tasks'), '--real', a.real, '--model', a.model, '--commit'],
            cwd=Path(__file__).resolve().parents[2], check=True)


if __name__ == '__main__':
    main()
