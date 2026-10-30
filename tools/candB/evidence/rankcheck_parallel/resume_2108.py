"""Read-only progress plus retirement of this task's superseded cutoff supervisors."""
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE = Path('/root/autodl-tmp/c4')
HOST_ROOT = BASE / 'candB_rankcheck_2dcb6af5_host_20261030'
PLAN_ROOT = BASE / 'candB_rankcheck_3cd6481e_plan_20261030'
TAIL_ROOT = BASE / 'candB_rankcheck_f396955a_tail_20261030'


def ticks(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[19]
    except FileNotFoundError:
        return None


def retire_cutoff(host):
    """Signal only our exact owned deadline supervisor, never a scoring child."""
    ownership = HOST_ROOT / f'DEADLINE_OWNERSHIP_{host}.json'
    receipt = HOST_ROOT / f'MAIN2108_CUTOFF_RETIRED_{host}.json'
    if receipt.exists():
        print(receipt.read_text())
        return
    owned = json.loads(ownership.read_text())
    pid = owned['pid']
    result = dict(ownership=owned, time_unix=time.time(),
                  reason='MAIN21:08: existing shards finish; supersedes21:20 cutoff')
    if ticks(pid) == str(owned['start_ticks']):
        argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        assert str(HOST_ROOT / 'deadline_ops.py').encode() in argv
        assert b'--worker' in argv and host.encode() in argv
        libc = ctypes.CDLL(None, use_errno=True)
        fd = libc.syscall(434, int(pid), 0)
        assert fd >= 0
        try:
            if ticks(pid) == str(owned['start_ticks']):
                assert libc.syscall(424, fd, int(signal.SIGTERM), 0, 0) == 0
                result['status'] = 'Retired exact owned cutoff supervisor only'
        finally:
            os.close(fd)
    else:
        result['status'] = 'Supervisor already absent; no signal'
    with receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


def inspect(host):
    paths = ([HOST_ROOT / 'measurement_CIRCLES',
              PLAN_ROOT / 'measurement_MAPPO_native_lanes+MAPPO_speed_partial',
              TAIL_ROOT / 'measurement_MAPPO_native_lanes_tail',
              TAIL_ROOT / 'measurement_MAPPO_speed_partial_tail']
             if host == 'A' else [HOST_ROOT / 'measurement_PI'])
    for path in paths:
        owned = json.loads((path / 'OWNERSHIP.json').read_text())
        result = dict(directory=str(path), ownership=owned,
                      live=ticks(owned['pid']) == str(owned['start_ticks']))
        raw = path / 'RAW.json'
        if raw.exists():
            doc = json.loads(raw.read_text())
            result.update(mtime=raw.stat().st_mtime,
                          complete=doc.get('complete'),
                          shard_finished=doc.get('shard_finished'),
                          rows=[dict(index=r.get('plan_index', i + doc.get('start_index', 0)),
                                     arms=list(r.get('arms', {})))
                                for i, r in enumerate(doc.get('rows', []))])
        result['events'] = {p.name: json.loads(p.read_text())
                            for p in path.glob('OWNED_*.json')}
        print(json.dumps(result))
    if host == 'A':
        for name in ('TAIL_GUARD_RESULT.json', 'TAIL_GUARD_OWNERSHIP.json'):
            path = TAIL_ROOT / name
            if path.exists():
                print(name, path.read_text())
        log = TAIL_ROOT / 'TAIL_GUARD.log'
        if log.exists():
            print('TAIL_GUARD.log', log.read_text()[-6000:])
    print(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                                  '--format=csv,noheader'], text=True).strip())


if __name__ == '__main__':
    action, host = sys.argv[1:]
    assert host in ('A', 'R')
    if action == 'retire-cutoff':
        retire_cutoff(host)
    elif action == 'inspect':
        inspect(host)
    else:
        raise ValueError(action)
