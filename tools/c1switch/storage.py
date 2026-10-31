"""Exclusive, crash-conservative accounting and atomic receipts."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time

from .design import CAP, STEP0_CAP, checked


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = checked(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@contextmanager
def exclusive(root):
    """A crash leaves the lock in place for explicit receipt/PID reconciliation."""
    root = checked(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / 'DISPATCH.lock'
    with lock.open('x', encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(), unix=time.time()), f)
        f.flush()
        os.fsync(f.fileno())
    try:
        yield
    finally:
        # No deletion: retain lock receipt under a unique terminal name.
        lock.rename(root / f'DISPATCH.closed.{time.time_ns()}.json')


class Ledger:
    """Single dispatcher under exclusive(); reservations are never refunded."""
    def __init__(self, root):
        self.root = checked(root)
        self.path = self.root / 'accounting.jsonl'

    def rows(self):
        return [json.loads(x) for x in self.path.read_text().splitlines()] if self.path.exists() else []

    def charge(self, ident, sim_s, stage):
        rows = self.rows()
        if stage not in ('step0', 'main') or sim_s <= 0:
            raise ValueError('invalid reservation')
        if any(r['id'] == ident for r in rows):
            raise ValueError('already reserved; refuse redispatch')
        if sum(r['sim_s'] for r in rows) + sim_s > CAP:
            raise ValueError('total cap')
        if stage == 'step0' and sum(r['sim_s'] for r in rows if r['stage'] == stage) + sim_s > STEP0_CAP:
            raise ValueError('Step0 cap')
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(id=ident, sim_s=sim_s, stage=stage, unix=time.time()))+'\n')
            f.flush()
            os.fsync(f.fileno())
