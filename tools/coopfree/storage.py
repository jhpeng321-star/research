"""Durable exclusive receipts, locked accounting and polite one-worker admission."""
import json
import math
import os
import re
from pathlib import Path
import shutil
import time
from .protocol import CAPS, canonical, checked


def read(path):
    return json.loads(checked(path).read_text(encoding='utf-8'))


def write(path, value):
    path = checked(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        f.write(canonical(value)+'\n')
        f.flush()
        os.fsync(f.fileno())


def ledger(root):
    path = checked(root)/'CHARGES.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def reserve(root, attempts):
    """Entire batch is affordable BEFORE any member is dispatched, under one lock."""
    import fcntl
    root = checked(root)
    with (root/'charge.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        old = ledger(root)
        ids = [r['id'] for r in old]
        newids = [r['id'] for r in attempts]
        if len(set(newids)) != len(newids) or set(ids) & set(newids):
            raise ValueError('duplicate or previously charged attempt')
        for row in attempts:
            if not re.fullmatch(r'[A-Za-z0-9_-]+',row['id']) or row['phase'] not in CAPS or \
                    not math.isfinite(row['sim_s']) or row['sim_s'] <= 0:
                raise ValueError('invalid charge')
        for phase,cap in CAPS.items():
            if sum(r['sim_s'] for r in old+attempts if r['phase'] == phase) > cap:
                raise ValueError('whole-batch budget insufficient: '+phase)
        with (root/'CHARGES.jsonl').open('a',encoding='utf-8') as f:
            for row in attempts:
                f.write(canonical(row)+'\n')
            f.flush()
            os.fsync(f.fileno())


def admission(root, max_load=20.):
    """No signals; nice19 and one thread. User STOP/YIELD and spare-load guard."""
    root = checked(root)
    if os.getpriority(os.PRIO_PROCESS,0) != 19 or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('nice19 CPU-only execution required')
    while True:
        if (root/'STOP.json').exists():
            raise ValueError('durable STOP')
        mem = {k:int(v.split()[0]) for k,v in
               (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
        if (not (root/'YIELD').exists() and os.getloadavg()[0] <= max_load
                and mem['MemAvailable'] >= 8*1024**2 and shutil.disk_usage(root).free >= 2*1024**3):
            return
        time.sleep(5)


def status(root):
    rows = ledger(root)
    out = {}
    for phase,cap in CAPS.items():
        selected = [r for r in rows if r['phase'] == phase]
        actual,unclosed = 0.,0
        for r in selected:
            path = checked(root)/'jobs'/r['id']/'CLOSE.json'
            if path.exists():
                close = read(path)
                if close['charged_sim_s'] != r['sim_s'] or not 0 <= close['actual_sim_s'] <= r['sim_s']:
                    raise ValueError('accounting mismatch '+r['id'])
                actual += close['actual_sim_s']
            else:
                unclosed += 1
        out[phase] = dict(cap=cap,charged=sum(r['sim_s'] for r in selected),actual_closed=actual,unclosed=unclosed)
    return out


def completed(root, ident):
    path = checked(root)/'jobs'/ident
    if (path/'RESULT.json').exists() and (path/'CLOSE.json').exists():
        result, close = read(path/'RESULT.json'), read(path/'CLOSE.json')
        if not 0 <= close['actual_sim_s'] <= close['charged_sim_s'] or result['actual_sim_s'] != close['actual_sim_s']:
            raise ValueError('completed accounting mismatch '+ident)
        return result
    if path.exists():
        raise ValueError('unfinished attempt blocks resume; no retry '+ident)
    return None  # Reserved but not started is safe to start once.
