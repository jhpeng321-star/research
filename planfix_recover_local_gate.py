"""Recover only the interrupted 16:31 PLANFIX local F6 reservation, no signals."""
import fcntl
import json
import os
from pathlib import Path
import time

root = Path('/tmp/c4-cpuv-1000')
with (root/'gate.lock').open('r+b') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    p = root/'active.json'
    value = json.loads(p.read_text())
    assert value['parent'] == 1680
    assert value['result'].endswith('/b54794a198af4831993d97c0a6482209.result')
    assert value['argv'][-1] == 'python3 -B -'
    assert not Path('/proc/1680').exists()
    assert time.monotonic() > value['deadline']
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError):
            continue
        # No supervisor or session for this shared pool may survive.
        assert not (b'_session' in argv and str(p).encode() in argv), proc.name
        assert not (argv and argv[0] == b'ssh' and b'-T' in argv and b'cpuV' in argv), proc.name
    target = root/'PLANFIX_INTERRUPTED_1631_ACTIVE.json'
    assert not target.exists()
    os.rename(p, target)
    print(json.dumps(dict(recovered_own_local_reservation=True, original=value,
                         archive=str(target),remote_signals=0,new_dispatches=0)))
