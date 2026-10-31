"""Explicit owner-side A2 monitor transfer. Never launches/replays inference.

Deploy beside a NEW copy of remote.py, outside the immutable candidate code.
Original scorer, identities, dispatch and workers remain unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

try:
    from . import remote
except ImportError:  # Standalone immutable gpuA staging.
    import remote

INTERIMS = (('04', 1793476800), ('08', 1793491200))


def living(receipt):
    if not remote.alive(receipt):
        return False
    return Path(f"/proc/{receipt['pid']}/stat").read_text().split()[2] != 'Z'


def snapshot(root):
    paths = [root/'spec.json', root/'DISPATCH.json', root/'WORKERS.json']
    paths += list((root/'code').iterdir())
    paths += [p for i in range(3) for p in (root/f'shard{i}').glob('w*_k*.json')]
    paths += [root/f'shard{i}/IDENTITY.json' for i in range(3)]
    return {str(p): remote.sha(p) for p in paths if p.is_file()}


def verify_snapshot(hashes):
    for path, expected in hashes.items():
        assert remote.sha(path) == expected, ('changed preserved artifact', path)


def transfer(root, expected):
    """Signal only the authorized own supervisor; never its process group."""
    assert json.loads((root/'DISPATCH.json').read_text()) == expected
    assert living(expected), 'original supervisor absent; no failed-job replay'
    assert not any((root/n).exists() for n in ('FAILED.json', 'RESULT.json'))
    workers = json.loads((root/'WORKERS.json').read_text())['workers']
    assert len(workers) == 3 and len({r['pid'] for r in workers}) == 3
    for i, receipt in enumerate(workers):
        assert str(root/f'shard{i}') in receipt['argv']
        assert living(receipt) or (root/f'shard{i}/COMPLETE.json').exists()
    cmd = Path(f"/proc/{expected['pid']}/cmdline").read_bytes().split(b'\0')
    assert str(root/'code/remote.py').encode() in cmd and b'supervise' in cmd
    hashes = snapshot(root)
    remote.save(root/'NOON_HANDOVER_AUTHORIZED.json', dict(
        owner='S-RB-RUNA', authority='owner02:08 / registration5736',
        original=expected, successor=remote.incarnation(os.getpid()), workers=workers,
        preserved_hashes=hashes, deadline=remote.DEADLINE, unix=time.time()))
    # Recheck the exact incarnation immediately before the single parent-only signal.
    assert living(expected)
    os.kill(expected['pid'], signal.SIGTERM)
    until = time.monotonic()+15
    while living(expected) and time.monotonic() < until:
        time.sleep(.1)
    assert not living(expected), 'old supervisor did not yield; no second monitor'
    import fcntl
    lock = (root/'SUPERVISOR.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_snapshot(hashes)
    assert all(living(r) or (root/f'shard{i}/COMPLETE.json').exists()
               for i, r in enumerate(workers)), 'worker changed during transfer'
    remote.save(root/'NOON_HANDOVER_ACCEPTED.json', dict(
        monitor=remote.incarnation(os.getpid()), original=expected,
        workers=workers, deadline=remote.DEADLINE, unix=time.time()))
    return lock, workers, hashes


def own_usage(workers):
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits'], text=True)
    owned = {r['pid'] for r in workers if living(r)}
    return sum(int(row.split(',')[1]) for row in raw.splitlines()
               if row.strip() and int(row.split(',')[0]) in owned)


def stop_workers(root, workers):
    for i in range(3):
        p = root/f'shard{i}/STOP'
        if not p.exists():
            p.touch(exist_ok=False)
    until = time.monotonic()+20
    while any(living(r) for r in workers) and time.monotonic() < until:
        time.sleep(1)
    for receipt in workers:
        if living(receipt):
            os.kill(receipt['pid'], signal.SIGTERM)


def score_start(root, label, partial, env):
    reports = root/'reports'
    reports.mkdir(exist_ok=True)
    out = reports/label
    marker = root/f'NOON_SCORE_{label}.json'
    assert not marker.exists() and not out.exists(), 'never replay scoring'
    argv = ['nice', '-n', '19', '/root/miniconda3/bin/python', '-B', '-u',
            str(root/'code/score.py'), '--inputs',
            *[str(root/f'shard{i}') for i in range(3)], '--out', str(out)]
    if partial:
        argv.append('--allow-partial')
    with (root/f'noon-score-{label}.log').open('x') as log:
        p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env,
                             cwd=root, start_new_session=True)
    remote.save(marker, dict(**remote.incarnation(p.pid), argv=argv))
    return p, out


def monitor(root, expected):
    # Transfer failures before acceptance never signal or stop any workers.
    lock, workers, hashes = transfer(root, expected)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    peak = 0
    active = None
    attempted = set()
    next_status = 0
    try:
        while True:
            now = time.time()
            peak = max(peak, own_usage(workers))
            assert peak < 18500, f'own GPU guard: {peak}MiB'
            if now >= next_status:
                verify_snapshot(hashes)
                remote.save(root/f'NOON_STATUS_{time.time_ns()}.json', dict(
                    unix=now, deadline=remote.DEADLINE, peak_own_gpu_mib=peak,
                    live=[r['pid'] for r in workers if living(r)],
                    tasks=[len(list((root/f'shard{i}').glob('w*_k*.json'))) for i in range(3)]))
                next_status = now+600
            complete = remote.completed(root)
            for i, r in enumerate(workers):
                assert living(r) or (root/f'shard{i}/COMPLETE.json').exists(), 'worker exited incomplete; no retry'
            if active and active[0].poll() is not None:
                p, out, label = active
                assert p.returncode == 0, f'scorer {label} failed; no retry'
                gate = json.loads((out/'gate.json').read_text())
                receipt = dict(status='complete' if label == 'complete' else 'interim',
                    variant=root.name, tasks=160 if label == 'complete' else None,
                    report=str(out), peak_own_gpu_mib=peak, gate=gate,
                    finished_unix=time.time(),
                    report_hashes={p.name: remote.sha(p) for p in out.iterdir() if p.is_file()})
                if label == 'complete':
                    assert gate['complete_worlds'] == 20
                remote.save(root/('RESULT.json' if label == 'complete' else f'INTERIM_{label}.json'), receipt)
                if label == 'complete':
                    return
                active = None
            if now >= remote.DEADLINE:
                raise TimeoutError('owner11:30 deadline; completed tasks preserved')
            if not active and os.getloadavg()[0] < 16:
                verify_snapshot(hashes)
                if complete:
                    p, out = score_start(root, 'complete', False, env)
                    active = p, out, 'complete'
                else:
                    for label, due in INTERIMS:
                        if now >= due and label not in attempted:
                            attempted.add(label)
                            if any(all((root/f'shard{i}/w{s}_k{k}.json').exists() for k in range(8))
                                   for i in range(3) for s in range(969420, 969440)):
                                p, out = score_start(root, label, True, env)
                                active = p, out, label
                            else:
                                remote.save(root/f'INTERIM_{label}_INSUFFICIENT.json', dict(unix=now, reason='no complete K8 world'))
                            break
            time.sleep(2)
    except BaseException as error:
        stop_workers(root, workers)
        if active and active[0].poll() is None:
            active[0].terminate()  # Our own newly created CPU-only scorer.
        if not (root/'FAILED.json').exists():
            remote.save(root/'FAILED.json', dict(error=repr(error), unix=time.time(), workers=workers))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--expected-supervisor', required=True)
    args = parser.parse_args()
    root = remote.safe(args.root)
    assert root.parent == remote.BASE and root.name == 'WM-PHYS-A2'
    monitor(root, json.loads(args.expected_supervisor))
