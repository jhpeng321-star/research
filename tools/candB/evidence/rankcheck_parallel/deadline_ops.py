"""Freeze only this task's recorded process incarnations at the reporting cutoff."""
import ctypes
from datetime import datetime
import errno
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE = Path('/root/autodl-tmp/c4')
ROOT = BASE/'candB_rankcheck_2dcb6af5_host_20261030'
DEADLINE = datetime.fromisoformat('2026-10-30T21:20:00+08:00').timestamp()
HOST = sys.argv[-1]
assert HOST in ('A', 'R')
PATHS = ([ROOT/'measurement_CIRCLES',
          BASE/'candB_rankcheck_3cd6481e_plan_20261030/measurement_MAPPO_native_lanes+MAPPO_speed_partial']
         if HOST == 'A' else [ROOT/'measurement_PI'])


def ticks(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[19]
    except FileNotFoundError:
        return None


def write(path, doc):
    with path.open('x') as stream:
        json.dump(doc, stream, indent=1)


if '--worker' in sys.argv:
    owned = [(path, json.loads((path/'OWNERSHIP.json').read_text())) for path in PATHS]
    while time.time() < DEADLINE:
        time.sleep(max(0, min(30, DEADLINE-time.time())))
    events = []
    for path, doc in owned:
        pid = doc['pid']
        proc = Path(f'/proc/{pid}')
        if not proc.exists() or ticks(pid) != str(doc['start_ticks']):
            events.append(dict(pid=pid, status='already exited; no signal'))
            continue
        try:
            argv = (proc/'cmdline').read_bytes().split(b'\0')
        except FileNotFoundError:
            events.append(dict(pid=pid, status='exited before command check; no signal'))
            continue
        if not any(argv):
            events.append(dict(pid=pid, status='exited zombie; no signal'))
            continue
        assert b'tools.candB.rankcheck' in argv
        libc = ctypes.CDLL(None, use_errno=True)
        fd = libc.syscall(434, int(pid), 0)  # pidfd_open, Linux x86_64
        if fd < 0:
            if ctypes.get_errno() == errno.ESRCH:
                events.append(dict(pid=pid, status='exited before pidfd; no signal'))
                continue
            raise OSError(ctypes.get_errno(), 'pidfd_open')
        try:
            if ticks(pid) != str(doc['start_ticks']):
                events.append(dict(pid=pid, status='exited before signal; no signal'))
                continue
            event = dict(pid=pid, start_ticks=doc['start_ticks'], time_unix=time.time(),
                reason='MAIN21:30 report deadline: freeze completed arm receipts at21:20',
                no_optimizer_updates=True, no_sumo=True)
            write(path/'OWNED_DEADLINE_STOP.json', event)
            rc = libc.syscall(424, fd, int(signal.SIGTERM), 0, 0)  # pidfd_send_signal
            if rc != 0 and ctypes.get_errno() != errno.ESRCH:
                raise OSError(ctypes.get_errno(), 'pidfd_send_signal')
            events.append(dict(**event, status='SIGTERM to exact owned incarnation'))
        finally:
            os.close(fd)
    write(ROOT/f'DEADLINE_RESULT_{HOST}.json', dict(events=events, deadline_unix=DEADLINE))
else:
    receipt = ROOT/f'DEADLINE_OWNERSHIP_{HOST}.json'
    assert ROOT.is_dir() and not receipt.exists() and time.time() < DEADLINE
    with (ROOT/f'DEADLINE_{HOST}.log').open('x') as log:
        process = subprocess.Popen([sys.executable, __file__, '--worker', HOST], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(receipt, dict(pid=process.pid, start_ticks=ticks(process.pid),
        deadline_unix=DEADLINE, measurement_directories=[str(p) for p in PATHS]))
    print('Own measurement reporting cutoff supervisor', HOST, process.pid)
