"""Consume already-running FIT publications until the owner's09:00 deadline.

No fitting, WM/SUMO inference, remote dispatch, signalling or peer edits. Own
create-once claim and immutable dispatch/score receipts prevent duplicate work.
Only new verified-count summaries and the marked HANDOFF block are committed.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from .operate import RECEIPTS, ROOT

FIT_ROOT = Path('E:/agent-work/c4-rb-fit/artifacts/rbfit')
FIT_STATE = FIT_ROOT / 'receipts/LC_WATCH_0128_STATE.json'
DISPATCH = RECEIPTS / 'FIT_PRECISION_FOLLOWUP_0520_DISPATCH.json'
COMPLETE = RECEIPTS / 'FIT_PRECISION_FOLLOWUP_0520_COMPLETE.json'
RUNTIME = Path('E:/Temp/opencode/rblane-fit-precision-20261101-0520')
BEGIN, END = '<!-- LANE-FIT-PRECISION:BEGIN -->', '<!-- LANE-FIT-PRECISION:END -->'


def now():
    return datetime.now().astimezone().isoformat()


def dump(path, value, mode='w'):
    with path.open(mode, encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True, encoding='utf-8').rstrip('\r\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_block(old, text):
    assert old.count(BEGIN) == old.count(END) == 1
    prefix, rest = old.split(BEGIN)
    _, suffix = rest.split(END)
    return prefix + BEGIN + '\n' + text + '\n' + END + suffix


def latest():
    state = json.loads(FIT_STATE.read_text(encoding='utf-8'))
    publication = state.get('latest_publication', {})
    directory = Path(publication.get('directory', ''))
    if directory.parent.resolve() != FIT_ROOT.resolve() or not re.fullmatch(r'lane_raw_[a-f0-9]{12}', directory.name):
        return None
    path = directory / 'PUBLICATION.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('status') != 'PASS':
        return None
    assert digest(directory / 'VERIFIED.json') == record['verified_sha256']
    return directory, record['verified_sha256']


def publish(files, text, message):
    """Fail closed on any unrelated changes; never stage another worker's work."""
    allowed = {'HANDOFF.md', *(p.relative_to(ROOT).as_posix() for p in files)}
    changed = git('status', '--porcelain', '--untracked-files=all').splitlines()
    assert all(line[3:] in allowed for line in changed), 'unrelated work appeared; publication paused'
    handoff = ROOT / 'HANDOFF.md'
    old = handoff.read_text(encoding='utf-8')
    handoff.write_text(replace_block(old, text), encoding='utf-8')
    print(git('status', '--short'), flush=True)
    print(git('diff', '--', 'HANDOFF.md'), flush=True)
    print(git('log', '--oneline', '-10'), flush=True)
    subprocess.run(['git', 'diff', '--check'], cwd=ROOT, check=True)
    subprocess.run(['git', 'add', '--', *sorted(allowed)], cwd=ROOT, check=True)
    subprocess.run(['git', 'commit', '-m', message], cwd=ROOT, check=True)
    assert not git('status', '--porcelain', '--untracked-files=all'), 'post-commit worktree changed'


def launch(until):
    assert datetime.now().astimezone() < until, 'owner continuation window already closed'
    assert not git('status', '--porcelain', '--untracked-files=all'), 'commit current step before launching'
    assert RUNTIME.parent.is_dir() and not RUNTIME.exists() and not DISPATCH.exists()
    RUNTIME.mkdir()
    prior = latest()
    record = dict(status='RESERVED', started_at=now(), until=until.isoformat(), runtime=str(RUNTIME),
        root=str(ROOT), parent_commit=git('rev-parse', 'HEAD'),
        consume_source_sha256=digest(ROOT / 'tools/rblane/consume_fit_precision.py'),
        initially_seen_verified_sha256=[prior[1]] if prior else [],
        scope='only new completed FIT publication arithmetic; no model/fitter/SUMO/GPU/remote dispatch')
    dump(DISPATCH, record, 'x')
    command = [sys.executable, '-B', '-m', 'tools.rblane.watch_fit_precision', 'run', '--until', until.isoformat()]
    with (RUNTIME / 'run.log').open('x', encoding='utf-8') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)
    record.update(status='DISPATCHED', pid=child.pid, command=command)
    dump(DISPATCH, record)
    print(json.dumps(record, indent=2))


def run(until):
    # Exclusive create-once claim is never removed, including failures.
    dump(RUNTIME / 'CLAIM.json', dict(claimed_at=now(), until=until.isoformat()), 'x')
    for _ in range(20):
        record = json.loads(DISPATCH.read_text(encoding='utf-8'))
        if record['status'] == 'DISPATCHED':
            break
        time.sleep(.5)
    assert record['status'] == 'DISPATCHED'
    seen = set(record['initially_seen_verified_sha256'])
    published = []
    status = dict(status='RUNNING', pid=record['pid'], started_at=now(), published=published,
                  until=until.isoformat(), model_runs=0, fitter_runs=0, remote_dispatches=0)
    try:
        while datetime.now().astimezone() < until:
            status['heartbeat'] = now()
            status['repository_clean'] = not git('status', '--porcelain', '--untracked-files=all')
            dump(RUNTIME / 'STATUS.json', status)
            if status['repository_clean']:
                candidate = latest()
                if candidate and candidate[1] not in seen:
                    directory, identity = candidate
                    assert digest(ROOT / 'tools/rblane/consume_fit_precision.py') == record['consume_source_sha256']
                    name = f'FIT_{directory.name.upper()}_{identity[:8]}_PRECISION.json'
                    output = RECEIPTS / name
                    assert not output.exists(), 'existing receipt: reconcile rather than overwrite'
                    subprocess.run([sys.executable, '-B', '-m', 'tools.rblane.consume_fit_precision',
                        '--fit', str(directory), '--receipt', name], cwd=ROOT, check=True)
                    result = json.loads(output.read_text(encoding='utf-8'))
                    score = result['scores']['raw_physics_shrunk_posterior']
                    text = (f'Updated {now()}; follows completed FIT publications until09:00.\n'
                        f"Latest {directory.name}: precision{100*score['precision_ratio_of_expected_counts']:.3f}%, "
                        f"recall{100*score['recall_ratio_of_expected_counts']:.3f}% (ratios of expected counts).\n"
                        f'Frozen upstream source only; NOT ADOPTED. Receipt `{name}`; no new inference.')
                    publish([output], text, f'S-RB-LANE: publish completed {directory.name} precision tradeoff')
                    seen.add(identity)
                    published.append(dict(directory=directory.name, verified_sha256=identity,
                                          receipt=name, commit=git('rev-parse', 'HEAD')))
            time.sleep(min(45., max(0., (until - datetime.now().astimezone()).total_seconds())))
        if git('status', '--porcelain', '--untracked-files=all'):
            status.update(status='CLOSED_WITH_USER_WORK', ended_at=now())
        else:
            summary = dict(status='COMPLETE', ended_at=now(), published=published,
                           model_runs=0, fitter_runs=0, remote_dispatches=0, signals_sent=0,
                           worker_pid=record['pid'], worker_will_exit=True)
            dump(COMPLETE, summary, 'x')
            publish([COMPLETE], f'Closed {now()} at the09:00 deadline; {len(published)}new verified FIT summaries published.\n'
                    'No inference/fitting/remote dispatch. Keep-right qualification remains unresolved.',
                    'S-RB-LANE: close owner-window precision publication followup')
            status.update(status='COMPLETE', ended_at=now())
    except Exception as exc:
        status.update(status='BLOCKED_PRESERVED', ended_at=now(), error=repr(exc))
        raise
    finally:
        dump(RUNTIME / 'STATUS.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'run'))
    parser.add_argument('--until', default='2026-11-01T09:00:00+08:00')
    args = parser.parse_args()
    until = datetime.fromisoformat(args.until)
    assert until.tzinfo is not None and until.isoformat() == '2026-11-01T09:00:00+08:00'
    (launch if args.action == 'launch' else run)(until)


if __name__ == '__main__':
    main()
