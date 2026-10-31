"""Exclusive attempt IDs, durable precharge, append-only failure evidence."""
import json
import os
from pathlib import Path
import time
from contextlib import contextmanager

from .protocol import checked, canonical, sha, budget, extension_budget, approval


def read(path):
    return json.loads(checked(path).read_text(encoding='utf-8'))


def write(path, value):
    path = checked(path)
    with path.open('x',encoding='utf-8') as f:
        f.write(canonical(value)+'\n')
        f.flush()
        os.fsync(f.fileno())


def verify(root):
    root = checked(root)
    b = read(root/'BINDING.json')
    a = read(root/'APPROVAL.json')
    approval(a['owner_line'])
    expected = extension_budget() if b.get('phase') == 'extension' else budget()
    if b['budget'] != expected or a['design_sha256'] != sha(root/'COOPPLAN_DESIGN.md'):
        raise ValueError('registered design/budget binding')
    if b['design_sha256'] != a['design_sha256']:
        raise ValueError('source design mismatch')
    for name, expected in b['sources'].items():
        if sha(root/'pytools'/name) != expected:
            raise ValueError('immutable source changed: '+name)
    for name, expected in b.get('artifacts',{}).items():
        if sha(root/name) != expected:
            raise ValueError('immutable artifact changed: '+name)
    return b


def admission(root):
    b = read(root/'BINDING.json')
    while True:
        if (root/'STOP.json').exists():
            raise RuntimeError('durable STOP: no automatic retry')
        available = not (root/'YIELD').exists() and os.getloadavg()[0] <= b['load_max']
        if available:
            import shutil
            mem = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
            available = int(mem['MemAvailable'].split()[0])*1024 >= 8*1024**3 and shutil.disk_usage(root).free >= 2*1024**3
        if available:
            return
        time.sleep(15)


@contextmanager
def ledger_lock(root):
    with (root/'ledger.lock').open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            if not lock.tell():
                lock.write(b'0');lock.flush()
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_LOCK,1)
            try:yield
            finally:
                lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX)
            yield


def charge(root, ident, seconds):
    verify(root)
    allocation = read(root/'ALLOCATION.json')
    with ledger_lock(root):
        ledger = root/'CHARGES.jsonl'
        records = [json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
        if any(r['id'] == ident for r in records) or (root/'jobs'/ident).exists():
            raise ValueError('attempt already charged; never retry or refund')
        reserve_key = ident
        if allocation.get('reserved') is not None:
            from .quota import outstanding, credit_key
            reserve_key = credit_key(ident) if '_solo' in ident else ident
            pending = outstanding(root,records)
            if pending.get(reserve_key,0) < seconds:
                raise ValueError('physics without complete prior reservation: '+ident)
            if sum(r['sim_s'] for r in records)+sum(pending.values()) > allocation['cap']:
                raise ValueError('reserved allocation exceeded')
        if sum(r['sim_s'] for r in records)+seconds > allocation['cap']:
            raise ValueError('host allocation exhausted')
        with ledger.open('a',encoding='utf-8') as f:
            f.write(canonical(dict(id=ident,sim_s=seconds,reserve_key=reserve_key,pid=os.getpid(),unix=time.time()))+'\n')
            f.flush()
            os.fsync(f.fileno())
        path = root/'jobs'/ident
        path.mkdir()
        return path


def stop(root, error):
    if not (root/'STOP.json').exists():
        try:
            write(root/'STOP.json',dict(error=str(error),pid=os.getpid(),unix=time.time()))
        except FileExistsError:
            pass
