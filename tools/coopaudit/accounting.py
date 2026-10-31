"""Single-host durable precharge ledger. Failed reservations are never refunded."""
import json
import os
import time
from pathlib import Path
from tools.coopaudit.design import CAP


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, sort_keys=True, indent=1, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Ledger:
    def __init__(self, root, cap=CAP):
        self.root, self.cap = Path(root), cap
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root/'accounting.jsonl'

    def rows(self):
        return [json.loads(s) for s in self.path.read_text().splitlines()] if self.path.exists() else []

    def charge(self, job_id, sim_s):
        import fcntl
        if type(sim_s) is not int or sim_s <= 0:
            raise ValueError('positive exact-integer precharge required')
        with (self.root/'accounting.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            rows = self.rows()
            if len({r['id'] for r in rows}) != len(rows) or any(r['id'] == job_id for r in rows):
                raise ValueError('duplicate reservation; never retry')
            if sum(r['sim_s'] for r in rows)+sim_s > self.cap:
                raise ValueError('audit hard cap exceeded')
            with self.path.open('a') as f:
                f.write(json.dumps(dict(id=job_id, sim_s=sim_s, unix=time.time()))+'\n')
                f.flush()
                os.fsync(f.fileno())

    def admit_panel(self, charges):
        if sum(r['sim_s'] for r in self.rows())+sum(charges) > self.cap:
            raise ValueError('complete fixed panel does not fit; design amendment required')
