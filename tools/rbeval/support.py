"""Bounded standby for the existing WM2 CPU finalizer, never inference.

Preserve a live finalizer. Permit one deadline-only handover using exactly its
frozen scoring sources. Worker loss, scoring failure, or identity change alerts
instead of retrying. No process signalling and no candidate dispatch.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=1)


def alive(receipt):
    try:
        proc = Path('/proc')/str(receipt['pid'])
        return (Path('/proc/sys/kernel/random/boot_id').read_text().strip() == receipt['boot_id']
                and (proc/'stat').read_text().split()[21] == receipt['start_ticks'])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def decision(*, finalized, live_finalizer, blocked_workers, deadline_expired,
             handover_used, before_deadline, sources_match):
    if finalized:
        return 'REPORT_READY'
    if not before_deadline:
        return 'SUPPORT_DEADLINE'
    if live_finalizer:
        return 'PRESERVE_LIVE_FINALIZER'
    if not sources_match:
        return 'ALERT_SCORING_SOURCE_CHANGED'
    if blocked_workers:
        return 'ALERT_INFERENCE_BLOCKED'
    if deadline_expired and not handover_used:
        return 'EXTEND_CPU_FINALIZER_ONCE'
    return 'ALERT_FINALIZER_EXITED_NO_RETRY'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--until', required=True, help='timezone-qualified ISO8601')
    parser.add_argument('--poll-seconds', type=int, default=300)
    args = parser.parse_args()
    until = datetime.fromisoformat(args.until)
    assert until.tzinfo is not None and args.poll_seconds >= 30
    deadline = until.timestamp()
    root = Path(args.run_root).resolve()
    assert root.is_dir() and os.getpriority(os.PRIO_PROCESS, 0) == 19
    import fcntl
    lock = (root/'SUPPORT.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    originals = sorted(root.glob('FINALIZER_[0-9]*.json'))
    assert originals, 'existing finalizer receipt required'
    frozen = load(originals[0])['sources']
    scoring = root/'analysis_scoring'
    assert set(frozen) == {str(scoring/name) for name in
        ('finalize.py', 'score.py', 'core.py', 'references.json', 'real_observables.json')}
    # Import only the original local scoring observer, whose hashes are frozen.
    assert all(digest(Path(path)) == sha for path, sha in frozen.items())
    sys.path.insert(0, str(scoring))
    from finalize import panel_state
    pid = os.getpid()
    save_new(root/f'SUPPORT_{pid}.json', dict(pid=pid,
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21],
        started_unix=time.time(), until=args.until, deadline_unix=deadline,
        argv=sys.argv, support_sha256=digest(Path(__file__)), frozen_scoring=frozen))
    handover = root/'SUPPORT_HANDOVER.json'
    report_verified = False
    previous = None
    while True:
        now = time.time()
        state = panel_state(root)
        finalized = (root/'FINALIZED.json').is_file()
        live = any(alive(load(path)) for path in root.glob('FINALIZER_[0-9]*.json'))
        # Only the original finalizer's explicit deadline receipt admits extension.
        expired = any(load(path).get('deadline_reached') is True
                      for path in root.glob('FINALIZER_BLOCKED_*.json'))
        matches = all(digest(Path(path)) == sha for path, sha in frozen.items())
        action = decision(finalized=finalized, live_finalizer=live,
            blocked_workers=state['blocked'], deadline_expired=expired,
            handover_used=handover.exists(), before_deadline=now < deadline,
            sources_match=matches)
        status = dict(action=action, panel=state, live_finalizer=live,
                      frozen_sources_match=matches, report_verified=report_verified)
        if finalized and not report_verified:
            result = load(root/'FINALIZED.json')
            report = Path(result['report']).resolve()
            assert report.parent == root/'reports'
            assert result['tasks'] == 160 and result['worlds'] == 20 and state['complete']
            assert result['sources'] == frozen
            assert all(digest(report/name) == sha for name, sha in result['report_hashes'].items())
            report_verified = True
            status.update(report_verified=True, report=str(report))
        if action == 'EXTEND_CPU_FINALIZER_ONCE':
            dispatch = (root/'FINALIZE_DISPATCH.lock').open('a')
            fcntl.flock(dispatch, fcntl.LOCK_EX)
            try:
                # Recheck under the same dispatch lock as ops.finalize.
                live = any(alive(load(path)) for path in root.glob('FINALIZER_[0-9]*.json'))
                if not live and not (root/'FINALIZED.json').exists() and not handover.exists():
                    assert all(digest(Path(path)) == sha for path, sha in frozen.items())
                    assert not panel_state(root)['blocked']
                    # Reserve once BEFORE spawn; an interrupted handover never retries.
                    argv = ['nice', '-n', '19', sys.executable, '-B', '-u',
                        str(scoring/'finalize.py'), '--run-root', str(root),
                        '--max-hours', str(max(0., deadline-time.time())/3600)]
                    save_new(handover, dict(argv=argv, unix=time.time(), sources=frozen))
                    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
                    with (root/'support_finalizer.log').open('a') as log:
                        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                            env=env, cwd=root, start_new_session=True)
                    save_new(root/f'SUPPORT_CHILD_{child.pid}.json', dict(pid=child.pid, argv=argv, unix=time.time()))
            finally:
                fcntl.flock(dispatch, fcntl.LOCK_UN)
                dispatch.close()
        if status != previous:
            save_new(root/f'SUPPORT_STATE_{time.time_ns()}.json', dict(unix=now, **status))
            previous = status
        print(json.dumps(dict(unix=now, **status)), flush=True)
        if now >= deadline:
            break
        time.sleep(min(args.poll_seconds, deadline-time.time()) if deadline > time.time() else 0)


if __name__ == '__main__':
    main()
