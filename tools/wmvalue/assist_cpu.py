"""Disjoint CPU partition after MAIN19:52 releases gpuR CPUs to C1/WM.

Uses the original task directory/worker snapshot, exclusive claims and source
bindings. It cannot repeat original workers or change the frozen experiment.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from .audit import digest, safe
from .core import TIMES
from .orchestrate import cpu_free, run_task, save_new, task_key


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--gate', required=True)
    p.add_argument('--real', required=True)
    p.add_argument('--model', required=True)
    a = p.parse_args()
    root = safe(a.root)
    assert (root/'LAUNCH.json').is_file() and (root/'claims').is_dir()
    # Separate one-time assistant lock, with a fixed disjoint highest-world
    # partition. Original dispatcher skips these exclusively reserved claims.
    import msvcrt
    lock = (root/'ASSIST_CPU_R.lock').open('a+b')
    lock.seek(0)
    if lock.read(1) == b'':
        lock.write(b'1')
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    assert not (root/'ASSIST_CPU_R_ACK.json').exists(), 'existing assist: never duplicate'
    # Initial admission reserves8 new single-thread lanes +2 cores spare.
    while not cpu_free('gpuR', reserve_cores=10, reserve_gib=40):
        print('WAIT gpuR CPU headroom after MAIN19:52', flush=True)
        time.sleep(30)
    tasks = [(969439, t, f) for t in reversed(TIMES) for f in range(8)]
    reserved = []
    for task in tasks:
        key = task_key(task)
        claim = root/'claims'/(key+'.json')
        if claim.exists() or (root/'tasks'/(key+'.json')).exists():
            continue
        try:
            save_new(claim, dict(task=list(task), host='gpuR', device='cpu',
                remote='/root/wmvalue20261031/cells_v1/'+key+'.json',
                dispatcher_pid=os.getpid(), time=time.time(), reservation='MAIN19:52 CPU release'))
        except FileExistsError:
            continue  # Another owner claimed it first: never run that task.
        reserved.append(task)
    save_new(root/'ASSIST_CPU_R_ACK.json', dict(pid=os.getpid(), host='gpuR', device='cpu',
        max_lanes=8, tasks=[list(t) for t in reserved], source_snapshot=a.tag,
        original_workers_preserved=True, initial_cpu_headroom_cores=10))
    print(json.dumps(dict(status='RESERVED', pid=os.getpid(), tasks=len(reserved))), flush=True)
    queue_lock = threading.Lock()
    failed = threading.Event()

    def lane():
        while not failed.is_set():
            if not cpu_free('gpuR', reserve_cores=2):
                time.sleep(30)
                continue
            with queue_lock:
                if not reserved:
                    return
                task = reserved.pop(0)
            try:
                run_task(root, a.tag, task, 'gpuR', 'cpu', digest(safe(a.gate)),
                         reservation_pid=os.getpid())
            except Exception as exc:
                failed.set()
                print('ASSIST PAUSED:', repr(exc), flush=True)
                return
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(lane) for _ in range(8)]
        for job in jobs:
            job.result()
    if failed.is_set():
        return
    all_tasks = [(s, t, f) for s in range(969432, 969440) for t in TIMES for f in range(8)]
    if not all((root/'tasks'/(task_key(t)+'.json')).is_file() for t in all_tasks):
        print('CPU assist finished; original dispatcher continues remaining tasks', flush=True)
        return
    # If this partition finishes last, exactly one COMPLETE-file creator owns
    # finalization. The original creator otherwise owns it; no repeated A-E test.
    try:
        save_new(root/'COMPLETE.json', dict(schema='wmvalue_inference_complete/1', tasks=320,
            hashes={task_key(t)+'.json': digest(root/'tasks'/(task_key(t)+'.json')) for t in all_tasks}))
    except FileExistsError:
        return
    subprocess.run([sys.executable, '-B', '-m', 'tools.wmvalue.finalize',
        '--tasks', str(root/'tasks'), '--real', a.real, '--model', a.model, '--commit'],
        cwd=Path(__file__).resolve().parents[2], check=True)


if __name__ == '__main__':
    main()
