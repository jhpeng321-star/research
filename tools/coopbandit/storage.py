"""Durable exclusive attempts and locked precharged accounting; no refunds."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from .protocol import CAP, DATA_CAP, LIVE_CAP, canonical, checked


def write(path, value):
    p = checked(path)
    with p.open('x', encoding='utf-8') as f:
        f.write(canonical(value)+'\n')
        f.flush()
        os.fsync(f.fileno())


def read(path):
    return json.loads(checked(path).read_text(encoding='utf-8'))


@contextmanager
def mutex(root):
    import fcntl
    with (checked(root)/'ledger.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def charge(root, ident, seconds, phase):
    if phase not in ('data', 'live') or type(seconds) is not int or seconds <= 0:
        raise ValueError('invalid reservation')
    root = checked(root)
    with mutex(root):
        plan_path = root/'ALLOCATION.json'
        if plan_path.exists():
            plan = read(plan_path)
            if plan['attempts'].get(ident) != dict(sim_s=seconds,phase=phase):
                raise ValueError('attempt outside immutable host allocation')
        p = root/'CHARGES.jsonl'
        rows = [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
        if any(r['id'] == ident for r in rows):
            raise ValueError('duplicate attempt; no automatic retry')
        if sum(r['sim_s'] for r in rows)+seconds > CAP:
            raise ValueError('global cap')
        if sum(r['sim_s'] for r in rows if r['phase'] == phase)+seconds > dict(data=DATA_CAP, live=LIVE_CAP)[phase]:
            raise ValueError('phase cap')
        with p.open('a', encoding='utf-8') as f:
            f.write(canonical(dict(id=ident,sim_s=seconds,phase=phase,unix=time.time()))+'\n')
            f.flush()
            os.fsync(f.fileno())


def stop(root, error):
    with mutex(root):
        if not (Path(root)/'STOP').exists():
            write(Path(root)/'STOP',dict(error=error,pid=os.getpid(),unix=time.time()))


def admission(root):
    import shutil
    while True:
        if (Path(root)/'STOP').exists():
            raise ValueError('durable STOP')
        limits = read(Path(root)/'BINDING.json').get('resources',{}) if (Path(root)/'BINDING.json').exists() else {}
        if (Path(root)/'YIELD').exists() or os.getloadavg()[0] > limits.get('load_max',24):
            time.sleep(30)
            continue
        available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                             if x.startswith('MemAvailable:')))*1024
        if shutil.disk_usage(root).free < 2*1024**3 or available < 8*1024**3:
            raise ValueError('disk/memory admission')
        return
