"""Archive only raw branches of terminal, superseded diagnostic gates.

Uses the previously Linux-qualified SHA/rsync/removal engine unchanged. The
active gate_coop6 launch authority and every bank/checkpoint are out of scope.
"""
import json
import os
from pathlib import Path
import re
import time

GATES = ('coopmwm20261104_gate_coop2', 'coopmwm20261104_gate_coop4',
         'coopmwm20261104_gate_coop5')
BAD = re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vehicle_params|driver_params|vtype|assignment', re.I)


def under(path, parent):
    return Path(path).is_relative_to(Path(parent))


def kind(home, path):
    home, path = Path(home), Path(path)
    if BAD.search(str(path)) or path.is_symlink():
        return None
    try:
        parts = path.relative_to(home).parts
    except ValueError:
        return None
    if len(parts) != 4 or parts[0] not in GATES or parts[1] != 'branches':
        return None
    if not parts[2].startswith('psi_u000_B_sv_'):
        return None
    if parts[3] not in ('SNAPSHOT.pt', 'RESULT.json', 'RESULT.json.gz',
                        'observable.jsonl.gz') and not parts[3].endswith('.result.json'):
        return None
    return 0, 'B'


def terminal(root, now, minimum_age=1800):
    root = Path(root)
    marker = root / ('VOID_ENVIRONMENT_FAILURE.json' if root.name.endswith('coop4')
                     else 'COLLECT_STAGE_RESULT.json')
    if not marker.is_file() or now - marker.stat().st_mtime < minimum_age:
        return False
    # A final report or an explicit environment void is mandatory. Claims and
    # execution receipts alone do not prove the diagnostic has finished.
    return root.name.endswith('coop4') or (root / 'COOPFIX_GATE.json').is_file()


def select(home, opened, files, trees, refs, now):
    home = Path(home)
    out = {}
    for name in GATES:
        root = home / name
        if not terminal(root, now):
            continue
        for parent, dirs, names in os.walk(root / 'branches', followlinks=False):
            dirs[:] = [n for n in dirs if not BAD.search(str(Path(parent)/n))
                       and not (Path(parent)/n).is_symlink()]
            for name in names:
                p = Path(parent)/name
                key = str(p)
                if kind(home, p) is None or not p.is_file() or p.resolve() != p:
                    continue
                if now-p.stat().st_mtime < 1800 or p.stat().st_dev != home.stat().st_dev:
                    continue
                if key in opened or key in files:
                    continue
                if any(under(p, q) for q in trees):
                    continue
                # Protect every concrete reference, including a referenced
                # whole branch. A report/root storage declaration is not a raw
                # branch input. Gate root metadata is never moved.
                if any(key == q or (under(q, root/'branches') and under(p, q)) for q in refs):
                    continue
                out[key] = 'completed_superseded_gate_raw'
    return out


def run(control):
    import fcntl
    import hashlib
    import shutil
    import signal
    import subprocess
    import sys
    c = Path(control)
    b = json.loads((c/'AUTHORITY.json').read_text())
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if digest(__file__) != b['worker_sha256']:
        raise ValueError('archive scope source changed')
    sys.path.insert(0, b['qualified_archive_source'])
    import rolling_archive as m
    for n, h in b['source_hashes'].items():
        if digest(Path(b['qualified_archive_source'])/n) != h:
            raise ValueError('qualified archive engine changed')
    claim = (c/'RUN.claim').open('a')
    fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (c/'OWNER.json').exists():
        raise ValueError('adopt existing owner')
    m.CONTROL = c
    m.configure_target(b)
    home = Path('/home/ubuntu')
    old = Path(b['regular_mover_control'])
    for name in GATES:
        root = home/name
        for p in root.glob('*_OWNER.json'):
            owner = json.loads(p.read_text())
            proc = Path('/proc')/str(owner['pid'])
            if proc.exists():
                st = (proc/'stat').read_text().split()
                if st[21] == owner['start_ticks'] and st[2] != 'Z':
                    raise ValueError('diagnostic owner still running: '+str(p))
        for p, h in b['completion_hashes'].items():
            if digest(p) != h:
                raise ValueError('diagnostic completion changed')
    def transport():
        value = json.loads((old/'TRANSPORT.json').read_text())
        if not Path(value['socket']).exists():
            raise RuntimeError('forwarded agent unavailable; sources retained')
        return list(b['archive_target']['ssh_command']), dict(os.environ, SSH_AUTH_SOCK=value['socket'])
    m.ssh = transport
    m.diagnostic_kind = lambda p: kind(home, p)
    def candidates():
        files, trees, _ = m.base.protected(m.base.final_labels())
        return select(home, m.base.open_files(), files, trees, m.metadata_refs(), time.time()), {}
    m.candidates = candidates
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    os.nice(max(0, 19-os.getpriority(os.PRIO_PROCESS, 0)))
    m.base.write(c/'OWNER.json', m.base.owner())
    total, index, failures = 0, 0, 0
    while True:
        try:
            found, _ = candidates()
            m.atomic(c/'STATUS.json', dict(unix=time.time(), eligible_files=len(found),
                eligible_bytes=sum(Path(p).stat().st_size for p in found), moved_bytes=total,
                cpu_free=shutil.disk_usage(home).free, new_sim_s=0))
            if not found:
                m.base.write(c/'COMPLETE.json', dict(unix=time.time(), moved_bytes=total,
                    batches=index, cpu_free=shutil.disk_usage(home).free, new_sim_s=0))
                return
            items, size = [], 0
            for p, category in sorted(found.items()):
                items.append((p, category))
                size += Path(p).stat().st_size
                if size >= 128*1024**2:
                    break
            moved = m.batch(items, index)
            if not moved:
                raise ValueError('zero progress; preserve sources and inspect')
            total += moved
            index += 1
            failures = 0
            m.atomic(c/'PROGRESS.json', dict(unix=time.time(), moved_bytes=total,
                batches=index, cpu_free=shutil.disk_usage(home).free, new_sim_s=0))
        except BaseException as error:
            import traceback
            m.atomic(c/'ALERT.json', dict(unix=time.time(), error=repr(error),
                traceback=traceback.format_exc(), sources_retained=True, new_sim_s=0))
            index += 1
            connection = isinstance(error, subprocess.TimeoutExpired) or any(s in str(error) for s in (
                'Connection closed', 'Connection reset', 'timed out',
                'Temporary failure in name resolution', 'code 255', 'forwarded agent unavailable'))
            failures += 1
            if not connection or failures > 5:
                raise
            time.sleep((10, 20, 40, 80, 120)[failures-1])


if __name__ == '__main__':
    import sys
    run(sys.argv[1])
