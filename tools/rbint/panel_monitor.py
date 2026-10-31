"""Bounded monitor/scorer for INT-owned FULL panel; never launch model inference.

Run outside the frozen candidate runtime. Every claimed output is create-once;
only exact INT-owned worker incarnations may receive a cooperative stop/signal.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

DEADLINE = datetime.fromisoformat('2026-11-01T11:30:00+08:00').timestamp()
CHECKPOINTS = {label: datetime.fromisoformat(f'2026-11-01T{clock}:00+08:00').timestamp()
               for label, clock in (('04', '04:00'), ('08', '08:00'))}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)


def alive(receipt):
    try:
        fields = Path(f"/proc/{receipt['pid']}/stat").read_text().split()
        return (fields[2] != 'Z' and fields[21] == receipt['start_ticks'] and
                Path('/proc/sys/kernel/random/boot_id').read_text().strip() == receipt['boot_id'])
    except (FileNotFoundError, ProcessLookupError):
        return False


def available_memory(proc=Path('/proc'), cgroup=Path('/sys/fs/cgroup')):
    physical = next(int(line.split()[1])*1024 for line in (proc/'meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:'))
    limit, current = cgroup/'memory.max', cgroup/'memory.current'
    if limit.exists() and current.exists() and limit.read_text().strip() != 'max':
        return min(physical, int(limit.read_text())-int(current.read_text()))
    return physical


def verify_completion(root):
    """A marker alone cannot certify a complete160-task ensemble."""
    total = 0
    for i in range(3):
        shard = root/f'shard{i}'
        marker = shard/'COMPLETE.json'
        if not marker.is_file():
            return False
        data = json.loads(marker.read_text())
        expected = {f'w{s}_k{k}.json' for j, s in enumerate(range(969420, 969440))
                    if j % 3 == i for k in range(8)}
        if data['tasks'] != len(expected) or set(data['hashes']) != expected:
            raise ValueError('noncanonical completed task manifest')
        if data['identity_sha256'] != digest(shard/'IDENTITY.json'):
            raise ValueError('completed identity changed')
        for name, sha in data['hashes'].items():
            if digest(shard/name) != sha:
                raise ValueError('completed task changed: '+name)
        total += len(expected)
    if total != 160:
        raise ValueError('full-panel task count differs')
    return True


def stop_owned(root, workers):
    for i in range(3):
        path = root/f'shard{i}'/'STOP'
        if path.parent.is_dir() and not path.exists():
            path.touch(exist_ok=False)
    end = time.monotonic()+20
    while any(alive(r) for r in workers) and time.monotonic() < end:
        time.sleep(1)
    for receipt in workers:
        if alive(receipt):
            os.kill(receipt['pid'], signal.SIGTERM)


def score_once(root, runtime, label, partial):
    marker = root/f'ANALYSIS_{label}.json'
    if marker.exists():
        return
    if os.getloadavg()[0] >= 16:
        return
    # Refuse to manufacture K8 scores from an incomplete ensemble.
    names = {p.name for i in range(3) for p in (root/f'shard{i}').glob('w*_k*.json')}
    complete = [s for s in range(969420, 969440)
                if all(f'w{s}_k{k}.json' in names for k in range(8))]
    if not complete:
        save(marker, dict(status='BLOCKED_NO_COMPLETE_K8_WORLD', label=label, tasks=len(names)))
        return
    reports = root/'reports'
    reports.mkdir(exist_ok=True)
    out = reports/label
    if out.exists():
        raise ValueError('preserve interrupted scoring output; no automatic retry')
    argv = [sys.executable, '-B', str(runtime/'tools/rbeval/score.py'), '--inputs',
            *[str(root/f'shard{i}') for i in range(3)], '--out', str(out)]
    if partial:
        argv.append('--allow-partial')
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (root/f'analysis_{label}.log').open('x') as stream:
        subprocess.run(argv, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    gate = json.loads((out/'gate.json').read_text())
    save(marker, dict(status='COMPLETE', label=label, report=str(out), gate=gate,
        complete_worlds=complete, tasks=len(names), partial=partial,
        score_source_sha256=digest(runtime/'tools/rbeval/score.py'),
        hashes={p.name: digest(p) for p in out.iterdir() if p.is_file()}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if root.parent != Path('/root/rbeval20261031/int'):
        raise ValueError('INT panel roots only')
    spec = json.loads((root/'SPEC.json').read_text())
    if spec['owner'] != 'S-RB-INT' or spec['variant'] != 'FULL':
        raise ValueError('never supervise peer inference')
    runtime = Path(spec['runtime'])
    workers = json.loads((root/'PANEL_PROCESS.json').read_text())['workers']
    if len(workers) != 3 or any(str(runtime/'tools/rbint/evaluate_FULL.py') not in r['argv'] for r in workers):
        raise ValueError('worker ownership binding differs')
    for r in workers:
        if '--out' not in r['argv'] or Path(r['argv'][r['argv'].index('--out')+1]).parent != root:
            raise ValueError('worker root differs')
    receipt = dict(pid=os.getpid(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21], deadline=DEADLINE,
        source_sha256=digest(Path(__file__)), worker_receipts=workers, started_unix=time.time())
    save(root/'MONITOR_ONCE.json', receipt)
    peak = 0
    try:
        while True:
            running = [r for r in workers if alive(r)]
            raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                '--format=csv,noheader,nounits'], text=True)
            ids = {r['pid'] for r in running}
            used = sum(int(row.split(',')[1]) for row in raw.splitlines()
                       if row.strip() and int(row.split(',')[0]) in ids)
            peak = max(peak, used)
            memory = available_memory() if spec.get('device') == 'cpu' else None
            save(root/f'MONITOR_STATUS_{time.time_ns()}.json', dict(unix=time.time(),
                alive=[r['pid'] for r in running], own_gpu_mib=used, peak_own_gpu_mib=peak,
                device=spec.get('device', 'cuda'), effective_free_memory_bytes=memory,
                load=os.getloadavg(), tasks=[len(list((root/f'shard{i}').glob('w*_k*.json'))) for i in range(3)]))
            if used >= 18500:
                raise RuntimeError('INT aggregate GPU guard exceeded')
            if running and memory is not None and memory < 2**30:
                raise RuntimeError('CPU memory headroom exhausted; stop only exact owned workers')
            complete = verify_completion(root)
            if not running and not complete:
                raise RuntimeError('INT worker terminal before complete panel; no automatic retry')
            for label, clock in CHECKPOINTS.items():
                if time.time() >= clock:
                    score_once(root, runtime, 'interim'+label, partial=not complete)
            if complete:
                score_once(root, runtime, 'full', partial=False)
                if (root/'ANALYSIS_full.json').exists():
                    save(root/'MONITOR_COMPLETE.json', dict(unix=time.time(), peak_own_gpu_mib=peak,
                        analysis=json.loads((root/'ANALYSIS_full.json').read_text())))
                    break
            if time.time() >= DEADLINE:
                raise TimeoutError('owner11:30 evaluation deadline')
            time.sleep(120)
    except BaseException as error:
        stop_owned(root, workers)
        save(root/'MONITOR_FAILED.json', dict(error=repr(error), unix=time.time(),
                                             peak_own_gpu_mib=peak))
        raise


if __name__ == '__main__':
    main()
