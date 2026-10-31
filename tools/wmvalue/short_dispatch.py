"""Durable MAIN21:12 prefix dispatcher, preserving and collecting original jobs."""
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
from .orchestrate import cpu_free, fetch, run_task, save_new, task_key


def inventory(original, bucket='cells_v1', exclude=()):
    """Only explicit owned outputs and receipt PIDs; never signal processes."""
    claims = {}
    for path in (original/'claims').glob('*.json'):
        if path.name.count('.') != 1:
            continue
        d = json.loads(path.read_text())
        key = path.stem
        if key in exclude:
            continue
        if (original/'tasks'/(key+'.json')).exists():
            continue
        rp = path.with_name(key+'.remote.json')
        d['pid'] = json.loads(rp.read_text())['pid'] if rp.exists() else None
        d['started'] = path.with_name(key+'.ssh.json').exists()
        claims[key] = d
    status = {}
    for host in sorted({d['host'] for d in claims.values()}):
        selected = {k: d for k, d in claims.items() if d['host'] == host}
        code = ('from pathlib import Path; import json,os; claims='+repr(selected)+'\n'+
                "out={}\nfor k,d in claims.items():\n"
                " p=Path(d['remote']); assert str(p).startswith("+repr('/root/wmvalue20261031/'+bucket+'/')+")\n"
                " pid=d['pid']; live=bool(pid and Path('/proc/'+str(pid)).exists())\n"
                " out[k]=dict(output=p.is_file(),live=live,pid=pid)\n"
                "print(json.dumps(out))\n")
        p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host,
                            '/root/miniconda3/bin/python -B -'], input=code,
                           text=True, capture_output=True, timeout=30, check=True)
        for key, state in json.loads(p.stdout).items():
            state['host'] = host
            state['device'] = selected[key]['device']
            state['started'] = selected[key]['started']
            if state['output']:
                fetch(host, selected[key]['remote'], original/'tasks'/(key+'.json'))
            status[key] = state
    return status


def gpu_free(host, root):
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host,
        'nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=30)
    if p.returncode:
        return False
    owned = set()
    for path in (root/'claims').glob('*.remote.json'):
        claim = json.loads(path.with_name(path.name.replace('.remote.json', '.json')).read_text())
        if claim['host'] == host and claim['device'] == 'cuda':
            owned.add(int(json.loads(path.read_text())['pid']))
    try:
        return not ({int(s.strip()) for s in p.stdout.splitlines() if s.strip()}-owned)
    except ValueError:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--original', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--gate', required=True)
    p.add_argument('--real', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--collect-only', action='store_true')
    a = p.parse_args()
    root, original = safe(a.root), safe(a.original)
    assert root.parent.is_dir() and (original/'LAUNCH.json').is_file()
    if a.collect_only:
        print(json.dumps(inventory(original), indent=1), flush=True)
        return
    root.mkdir(exist_ok=True)
    for name in ('tasks', 'claims', 'logs'):
        (root/name).mkdir(exist_ok=True)
    import msvcrt
    lock = (root/'SHORT.lock').open('a+b')
    if not lock.read(1):
        lock.write(b'1')
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    if not (root/'ACK.json').exists():
        save_new(root/'ACK.json', dict(pid=os.getpid(), time=time.time(), tag=a.tag,
            gate_sha256=digest(a.gate), worker_sha256=digest(Path(__file__).with_name('worker.py')),
            original_launch_sha256=digest(original/'LAUNCH.json'), horizon=90,
            cpu_limits=dict(gpuA=4, gpuR=8), no_peer_signals=True))
    tasks = [(s, t, f) for s in range(969432, 969440) for t in TIMES for f in range(8)]
    running = {}
    failed = False
    pool = ThreadPoolExecutor(max_workers=18)
    try:
        while True:
            try:
                states = inventory(original)
                inherited = inventory(root, bucket='short_cells_v1', exclude=running)
            except (subprocess.SubprocessError, OSError) as exc:
                print('COLLECTION/OBSERVATION WAIT (no inference retry):', repr(exc), flush=True)
                time.sleep(30)
                continue
            # inventory(root) collects inherited claims without touching the
            # actively managed futures or attempting to repeat their inference.
            for key, (future, host, device) in list(running.items()):
                if future.done():
                    try:
                        future.result()
                    except Exception as exc:
                        failed = True
                        print('SHORT PAUSED:', key, repr(exc), flush=True)
                    del running[key]
            complete = [t for t in tasks if (root/'tasks'/(task_key(t)+'.json')).is_file()
                        or (original/'tasks'/(task_key(t)+'.json')).is_file()]
            print(json.dumps(dict(time=time.time(), complete=len(complete), expected=320,
                active_short=len(running), active_original=sum(s['live'] and not s['output']
                    for s in states.values()), paused=failed)), flush=True)
            if len(complete) == 320:
                break
            if failed:
                if not running:
                    return
                time.sleep(30)
                continue
            for host, device, limit in [('gpuA', 'cpu', 4), ('gpuR', 'cpu', 8),
                                        ('gpuA', 'cuda', 3), ('gpuR', 'cuda', 3)]:
                used = sum(h == host and d == device for _, h, d in running.values())
                used += sum(s['host'] == host and s['device'] == device and s['live'] and not s['output']
                            for s in [*states.values(), *inherited.values()])
                if used >= limit:
                    continue
                admitted = gpu_free(host, root) if device == 'cuda' else cpu_free(host, reserve_cores=2)
                if not admitted:
                    continue
                for task in tasks:
                    key = task_key(task)
                    if (root/'claims'/(key+'.json')).exists() or key in running:
                        continue
                    if (root/'tasks'/(key+'.json')).exists() or (original/'tasks'/(key+'.json')).exists():
                        continue
                    state = states.get(key)
                    if state and state['live']:
                        continue
                    # Previously started originals without output require explicit
                    # failure audit. k0's once-only retry is reserved, not spent here.
                    if state and (state['pid'] or state['started']):
                        continue
                    future = pool.submit(run_task, root, a.tag, task, host, device,
                                         digest(a.gate), short=True)
                    running[key] = future, host, device
                    used += 1
                    if used >= limit:
                        break
            time.sleep(30)
    finally:
        pool.shutdown(wait=True)
    save_new(root/'COMPLETE.json', dict(tasks=320, horizon=90, time=time.time()))
    subprocess.run([sys.executable, '-B', '-m', 'tools.wmvalue.finalize_short',
        '--short', str(root), '--original', str(original), '--real', a.real, '--model', a.model],
        cwd=Path(__file__).resolve().parents[2], check=True)


if __name__ == '__main__':
    main()
