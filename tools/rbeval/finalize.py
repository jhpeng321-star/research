"""Own-root CPU-only completion watcher: verify all tasks, then score once.

No inference launch, process signalling, training or simulator execution. A
failed/stopped worker produces a blocked receipt rather than a replacement job.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__:
    from .core import HERE, SEEDS, digest, load, safe, save_new
else:
    from core import HERE, SEEDS, digest, load, safe, save_new


def owned_alive(receipt):
    p = Path('/proc')/str(receipt['pid'])
    try:
        return (Path('/proc/sys/kernel/random/boot_id').read_text().strip() == receipt['boot_id'] and
                (p/'stat').read_text().split()[21] == receipt['start_ticks'])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def panel_state(root):
    """Trust completion only after identity and every immutable task hash match."""
    shards, total = [], 0
    for i in range(4):
        path = root/f'wm2_shard{i}'
        marker = path/'COMPLETE.json'
        count = len(list(path.glob('w*_k*.json')))
        done = marker.is_file()
        if done:
            complete = load(marker)
            identity = digest(path/'IDENTITY.json')
            expected = {f'w{s}_k{k}.json' for j, s in enumerate(SEEDS) if j % 4 == i for k in range(8)}
            assert complete['identity_sha256'] == identity
            assert complete['tasks'] == 40 and set(complete['hashes']) == expected
            for name, sha in complete['hashes'].items():
                assert digest(path/name) == sha, ('changed task', path/name)
            assert count == 40
        active = any(owned_alive(load(p)) for p in path.glob('PROCESS_*.json'))
        shards.append(dict(shard=i, tasks=count, complete=done, live_owned_worker=active))
        total += count
    return dict(tasks=total, complete=all(s['complete'] for s in shards), shards=shards,
                blocked=any(not s['complete'] and not s['live_owned_worker'] for s in shards))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', required=True)
    p.add_argument('--poll-seconds', type=int, default=120)
    p.add_argument('--max-hours', type=float, default=8)
    args = p.parse_args()
    import fcntl
    root = safe(args.run_root)
    assert root.is_dir() and args.poll_seconds >= 30
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19
    lock = (root/'FINALIZE.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root/'FINALIZED.json').exists():
        print('Already finalized; no duplicate scoring', flush=True)
        return
    pid = os.getpid()
    sources = {str(HERE/name): digest(HERE/name) for name in
               ('finalize.py', 'score.py', 'core.py', 'references.json', 'real_observables.json')}
    receipt = dict(pid=pid, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                   start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21],
                   started_unix=time.time(), argv=sys.argv, sources=sources)
    save_new(root/f'FINALIZER_{pid}.json', receipt)
    deadline = time.monotonic()+args.max_hours*3600
    while True:
        state = panel_state(root)
        print(json.dumps(state), flush=True)
        if state['complete']:
            break
        if state['blocked'] or time.monotonic() > deadline:
            save_new(root/f'FINALIZER_BLOCKED_{time.time_ns()}.json',
                     dict(state=state, deadline_reached=time.monotonic() > deadline))
            raise RuntimeError('Panel incomplete; preserved workers/results; no replacement launched')
        time.sleep(args.poll_seconds)
    while os.getloadavg()[0] > 24:
        assert time.monotonic() <= deadline, 'CPU remained busy beyond finalizer deadline'
        time.sleep(args.poll_seconds)
    assert all(digest(path) == sha for path, sha in sources.items()), 'scoring source changed'
    reports = root/'reports'
    reports.mkdir(exist_ok=True)
    out = reports/f'wm2_complete_{time.time_ns()}'
    argv = [sys.executable, '-B', '-u', str(HERE/'score.py'), '--inputs',
            *[str(root/f'wm2_shard{i}') for i in range(4)], '--out', str(out)]
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    started = time.monotonic()
    with (reports/(out.name+'.log')).open('x') as log:
        subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    gate = load(out/'gate.json')
    assert gate['complete_worlds'] == 20
    save_new(root/'FINALIZED.json', dict(schema='rbeval-finalized/1', tasks=160, worlds=20,
             episodes=800, report=str(out), gate=gate, scoring_seconds=time.monotonic()-started,
             sources=sources, report_hashes={p.name: digest(p) for p in out.iterdir() if p.is_file()},
             finished_unix=time.time()))
    print('FINALIZED', out, flush=True)


if __name__ == '__main__':
    main()
