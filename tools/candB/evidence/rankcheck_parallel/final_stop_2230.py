"""MAIN22:30: stop only receipt-bound rankcheck incarnations; collect saved arms."""
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

BASE = Path('/root/autodl-tmp/c4')
PLAN = BASE / 'candB_rankcheck_3cd6481e_plan_20261030'
HOST = BASE / 'candB_rankcheck_2dcb6af5_host_20261030'
TAIL = BASE / 'candB_rankcheck_f396955a_tail_20261030'
RESUME = BASE / 'candB_rankcheck_82dbd7bc_resume_20261030'
BIAS = BASE / 'candB_rankcheck_4d56f2a8_bias_20261030'
PATHS = {
    'A': [('circles', HOST / 'measurement_CIRCLES'),
          ('mappo', PLAN / 'measurement_MAPPO_native_lanes+MAPPO_speed_partial'),
          ('mappo_proxy_tail', TAIL / 'measurement_MAPPO_native_lanes_tail'),
          ('mappo_speed_tail', TAIL / 'measurement_MAPPO_speed_partial_tail'),
          ('zero', PLAN / 'measurement_candB_zero')],
    'R': [('pi', HOST / 'measurement_PI'),
          ('pi_resume_gpu', RESUME / 'measurement_PI_native_cuda_12_16_a2'),
          ('pi_resume_cpu', RESUME / 'measurement_PI_native_cpu_11_12_a2'),
          *[(f'native_cpu_{a}_{b}', RESUME / f'measurement_native_cpu_{a}_{b}')
            for a, b in ((0, 3), (3, 6), (6, 9), (9, 11))],
          ('bias_large', BIAS / 'measurement_PI_native_CIRCLES_11_12'),
          ('circles_resume_gpu', BIAS / 'measurement_CIRCLES_12_16')],
}


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()
        argv = [s.decode() for s in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if s]
        return dict(start_ticks=fields[19], state=fields[0], argv=argv)
    except FileNotFoundError:
        return None


def write(path, doc):
    with path.open('x') as stream:
        json.dump(doc, stream, indent=1)


def stop(directory):
    ownership = directory / 'OWNERSHIP.json'
    if not ownership.exists():
        return dict(directory=str(directory), status='not dispatched; no ownership receipt')
    owned = json.loads(ownership.read_text())
    pid = int(owned['pid'])
    before = identity(pid)
    event = dict(directory=str(directory), ownership=owned, before=before, time_unix=time.time())
    if not before or before['start_ticks'] != str(owned['start_ticks']) or before['state'] == 'Z':
        return dict(event, status='already exited; no signal')
    assert before['argv'] == owned['argv'], 'owned argv mismatch: refuse signal'
    assert 'tools.candB.rankcheck' in before['argv'] and '--measure-only' in before['argv']
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(434, pid, 0)
    if fd < 0:
        if ctypes.get_errno() == errno.ESRCH:
            return dict(event, status='exited before pidfd; no signal')
        raise OSError(ctypes.get_errno(), 'pidfd_open')
    try:
        current = identity(pid)
        if not current or current['start_ticks'] != str(owned['start_ticks']) or current['state'] == 'Z':
            return dict(event, status='exited before signal; no signal')
        assert current['argv'] == owned['argv']
        write(directory / 'OWNED_MAIN2230_STOP.json', dict(event,
              reason='Owner-approved MAIN22:30: stop remaining rankcheck; preserve every saved arm'))
        rc = libc.syscall(424, fd, int(signal.SIGTERM), 0, 0)
        if rc and ctypes.get_errno() != errno.ESRCH:
            raise OSError(ctypes.get_errno(), 'pidfd_send_signal')
    finally:
        os.close(fd)
    for _ in range(100):
        after = identity(pid)
        if not after or after['start_ticks'] != str(owned['start_ticks']) or after['state'] == 'Z':
            return dict(event, status='SIGTERM exact owned incarnation; stopped', after=after)
        time.sleep(.1)
    raise RuntimeError(f'Owned process {pid} did not stop; no peer signalling permitted')


def main():
    host = sys.argv[1]
    assert host in PATHS
    root = PLAN if host == 'A' else BIAS
    receipt = root / f'MAIN2230_STOP_{host}.json'
    bundle = root / f'MAIN2230_COLLECT_{host}.tar'
    assert root.is_dir()
    if receipt.exists():
        print(receipt.read_text())
        return
    snapshot = subprocess.check_output(['nvidia-smi'], text=True)
    events = [stop(directory) for _, directory in PATHS[host]]
    files = []
    for label, directory in PATHS[host]:
        for name in ('RAW.json', 'OWNERSHIP.json', 'COMMAND.json', 'run.log',
                     'OWNED_MAIN2230_STOP.json'):
            path = directory / name
            if path.is_file():
                files.append((path, f'{label}/{name}'))
        raw = directory / 'RAW.json'
        if raw.is_file():
            doc = json.loads(raw.read_text())
            events_for_dir = next(e for e in events if e['directory'] == str(directory))
            events_for_dir['completed_arms'] = {arm: sum(arm in r['arms'] for r in doc['rows'])
                                               for arm in doc['arms']}
            events_for_dir['raw_sha256'] = hashlib.sha256(raw.read_bytes()).hexdigest()
    document = dict(reason='Owner-approved MAIN22:30 final stop; no new scorers',
                    host=host, time_unix=time.time(), events=events,
                    nvidia_smi_before=snapshot,
                    nvidia_smi_after=subprocess.check_output(['nvidia-smi'], text=True),
                    no_training=True, no_sumo=True, peer_signals=False)
    write(receipt, document)
    files.append((receipt, receipt.name))
    with tarfile.open(bundle, 'x') as archive:
        for path, name in files:
            archive.add(path, arcname=name, recursive=False)
    print(json.dumps(dict(receipt=str(receipt), bundle=str(bundle), events=events), indent=1))


if __name__ == '__main__':
    main()
