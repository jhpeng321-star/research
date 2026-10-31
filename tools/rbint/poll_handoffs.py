"""Twenty-minute read-only peer handoff receipts; no merges or inference dispatch.

Survives a chat freeze. Inspect the newest receipt before resuming integration.
This observer never changes a peer file/process or retries a scientific job.
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

ROOT = Path(__file__).resolve().parents[2]
PEERS = [Path('E:/agent-work')/('c4-'+name) for name in (
    'rb-lane', 'rb-merge', 'rb-cf', 'rb-eval', 'rb-unc', 'rb-fit', 'rb-decide',
    'rb-runa', 'rb-runb', 'sumocore')]
DEADLINE = datetime.fromisoformat('2026-11-01T11:45:00+08:00').timestamp()


def save(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2)


def snapshot():
    rows = []
    for peer in PEERS:
        handoff = peer/'HANDOFF.md'
        if not handoff.is_file():
            rows.append(dict(peer=str(peer), handoff_missing=True))
            continue
        raw = handoff.read_bytes()
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=peer,
                              capture_output=True, text=True, timeout=15)
        status = subprocess.run(['git', 'status', '--short'], cwd=peer,
                                capture_output=True, text=True, timeout=15)
        rows.append(dict(peer=str(peer), head=head.stdout.strip(),
            status=status.stdout, handoff_sha256=hashlib.sha256(raw).hexdigest(),
            handoff=raw.decode('utf-8')))
    return dict(unix=time.time(), local_time=datetime.now().astimezone().isoformat(),
                read_only=True, peers=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('once', 'watch', 'start'))
    parser.add_argument('--out', default=str(ROOT/'runs/rbint_poll'))
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if not out.parent.is_dir():
        raise ValueError('poll receipt parent must already exist')
    out.mkdir(exist_ok=True)
    if args.action == 'start':
        save(out/'DISPATCH_ONCE.json', dict(unix=time.time(), deadline=DEADLINE,
            automatic_retry=False, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        argv = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), 'watch', '--out', str(out)]
        flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == 'nt' else 0
        with (out/'watch.log').open('x', encoding='utf-8') as log:
            child = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                creationflags=flags, start_new_session=os.name != 'nt')
        receipt = dict(pid=child.pid, argv=argv, deadline=DEADLINE, interval_s=1200)
        save(out/'PROCESS.json', receipt)
        print(json.dumps(receipt))
        return
    if args.action == 'watch':
        save(out/'WATCH_ONCE.json', dict(pid=os.getpid(), unix=time.time(), deadline=DEADLINE))
    while True:
        start = time.monotonic()
        try:
            data = snapshot()
            save(out/f'POLL_{time.time_ns()}.json', data)
            print(data['local_time'], 'read-only peer receipts', len(data['peers']), flush=True)
        except Exception as error:
            save(out/f'ERROR_{time.time_ns()}.json', dict(error=repr(error), unix=time.time()))
            print('POLL ERROR', repr(error), flush=True)
        if args.action == 'once' or time.time() >= DEADLINE:
            break
        time.sleep(min(max(1, 1200-(time.monotonic()-start)), max(0, DEADLINE-time.time())))


if __name__ == '__main__':
    main()
