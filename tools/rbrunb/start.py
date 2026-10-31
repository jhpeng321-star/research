"""Detach the bounded overnight watcher and verify its first poll."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    here = Path(__file__).resolve().parent
    work = here.parents[1]
    root = work/'runs'/'rbrunb'
    assert root.is_dir(), 'Run watch.py --once first to verify the startup path'
    import msvcrt
    with (root/'WATCH.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
    stamp = time.time_ns()
    log_path = root/f'watch_{stamp}.log'
    with log_path.open('x', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, '-B', '-u', str(here/'watch.py'), *sys.argv[1:]],
                                 cwd=work, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT,
                                 creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                                 env=dict(os.environ, PYTHONUNBUFFERED='1'))
    time.sleep(3)
    assert child.poll() is None, f'Watcher startup failed; see {log_path}'
    status = json.loads((root/'STATUS.json').read_text(encoding='utf-8'))
    receipt = dict(pid=child.pid, log=str(log_path), status=status, unix=time.time())
    target = root/f'LAUNCH_{stamp}.json'
    with target.open('x', encoding='utf-8') as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(dict(receipt=str(target), **receipt)))


if __name__ == '__main__':
    main()
