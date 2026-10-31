"""One-shot boundary gate: preserve in-flight physics; park legacy supervisor before dev."""
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = '/home/ubuntu/coopfree20261101_build5'


def gate(root=ROOT):
    r = Path(root)
    sys.path.insert(0, str(r/'pytools'))
    from tools.coopfree.storage import read, write
    from tools.coopfree_batch import identity
    dispatch = read(r/'DISPATCH.json')
    assert dispatch['pid'] == 339173 and dispatch['start_ticks'] == '47835664'
    identity(Path('/proc/339173/stat').read_text())
    write(r/'HOLD_GATE_OWNED.json', dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split()[21]))
    while not (r/'BATCH_COMPLETE.json').exists():
        if (r/'STOP.json').exists():
            raise RuntimeError('durable STOP; preserve all attempts')
        time.sleep(1)
    # The excluded original arm is still running; its existing plant is never signalled.
    write(r/'YIELD', dict(reason='OWNER01:48 HOLD credit migration; no new legacy physics'))
    while True:
        identity(Path('/proc/339173/stat').read_text())
        if (r/'STOP.json').exists():
            raise RuntimeError('durable STOP; inspect without retry')
        unfinished = []
        for p in (r/'jobs').iterdir():
            if p.is_dir() and (p/'START.json').exists() and not (p/'CLOSE.json').exists():
                unfinished.append(p.name)
        if not unfinished and (r/'UPDATE_001.json').exists():
            identity(Path('/proc/339173/stat').read_text())
            os.kill(339173, signal.SIGSTOP)
            write(r/'HOLD_GATE_PARKED.json', dict(supervisor_pid=339173,
                  supervisor_start_ticks='47835664', unfinished=[], unix=time.time(),
                  legacy_checkpoint_only=True, next='CPU-verified HOLD continuation; no retry'))
            print(json.dumps(dict(parked=True, unfinished=[])), flush=True)
            return
        time.sleep(1)


if __name__ == '__main__':
    gate()
