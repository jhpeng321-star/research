"""Disjoint SHA-gated archive of completed V2V-u8 observable tapes only.

MAIN Nov3 16:47 authorises completed interim job observables. No outcome payload
is decoded; RESULTs, public receipts, checkpoints and all training tapes stay put.
The existing qualified rolling mover supplies transfer and deletion guards.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

BAD = re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vehicle_params|driver_params|vtype', re.I)
MIN_AGE = 1800


def tape_info(root, path):
    root, path = Path(root), Path(path)
    if BAD.search(str(path)) or '..' in path.parts or path.is_symlink():
        return None
    if path.name != 'observable.jsonl.gz' or path.parent.parent != root/'jobs':
        return None
    match = re.fullmatch(r'v2vtest008_([ABR])_([0-9]{6})', path.parent.name)
    if not match:
        return None
    seed = int(match[2])
    if not (989600 <= seed <= 989635 or 990600 <= seed <= 990659):
        return None
    return match[1], seed


def under(path, parent):
    try:
        Path(path).relative_to(Path(parent))
        return True
    except ValueError:
        return False


def select(root, opened, files, trees, refs, now, digest):
    """Completion metadata, fresh references and original tape fingerprint required."""
    root = Path(root)
    complete = root/'test8_COMPLETE.json'
    if not complete.is_file() or (root/'test8_FAILED.json').exists():
        return {}
    if now-complete.stat().st_mtime < MIN_AGE:
        return {}
    found = {}
    for p in (root/'jobs').glob('v2vtest008_*/observable.jsonl.gz'):
        info = tape_info(root, p)
        if info is None or not p.is_file() or p.resolve() != p:
            continue
        close = p.parent/'CLOSE.json'
        public = root/'final_receipts'/(p.parent.name+'.json')
        if not close.is_file() or not public.is_file() or not (p.parent/'RESULT.json').is_file():
            continue
        if public.is_symlink() or public.resolve() != public:
            continue
        if any(now-q.stat().st_mtime < MIN_AGE for q in (p, close, public)):
            continue
        name = str(p)
        if name in opened or name in files or any(under(p, t) for t in trees):
            continue
        # A root/jobs location is not a leaf dependency. Bank job references are.
        if any(name == q or (under(q, root/'jobs') and Path(q) != root/'jobs'
                            and under(p, q)) for q in refs):
            continue
        v = json.loads(public.read_text())  # public fingerprints only
        if (v['id'] != p.parent.name or v['arm'] != info[0]
                or v['world']['seed'] != info[1] or v['checkpoint_update'] != 8
                or v['kind'] != 'episode' or not v['outcomes_closed']):
            raise ValueError('interim public completion identity differs')
        if digest(p) != v['files']['observable.jsonl.gz']['sha256']:
            raise ValueError('interim tape differs from original completion')
        found[name] = 'completed_v2v_u8_observable_only'
    return found


def run(control):
    import fcntl
    import signal
    import subprocess
    import sys
    c = Path(control)
    authority = json.loads((c/'AUTHORITY.json').read_text())
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if digest(__file__) != authority['worker_sha256']:
        raise ValueError('interim archive source changed')
    if digest(c/'v2v_archive_scope.py') != authority['dependency_sha256']:
        raise ValueError('reference-reader source changed')
    root = Path(authority['v2v_root'])
    if digest(root/'test8_COMPLETE.json') != authority['completion_sha256']:
        raise ValueError('interim completion marker changed')
    old = Path(authority['regular_mover_control'])
    sys.path.insert(0, authority['qualified_archive_source'])
    import rolling_archive as m
    for name in ('rolling_archive.py', 'archive_move.py'):
        if digest(Path(authority['qualified_archive_source'])/name) != authority['source_hashes'][name]:
            raise ValueError('qualified archive helper changed')
    lock = (c/'RUN.claim').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (c/'OWNER.json').exists():
        raise ValueError('adopt existing interim archive; do not replay')
    m.CONTROL = c
    m.configure_target(authority)

    def ssh():
        transport = json.loads((old/'TRANSPORT.json').read_text())
        if not Path(transport['socket']).exists():
            raise RuntimeError('forwarded agent unavailable; sources retained')
        return list(authority['archive_target']['ssh_command']), dict(os.environ, SSH_AUTH_SOCK=transport['socket'])

    m.ssh = ssh
    m.job_kind = lambda p: tape_info(root, p)

    def candidates():
        from tools.coopmwm.v2v_archive_scope import pending_refs
        files, trees, _ = m.base.protected(m.base.final_labels())
        refs = m.metadata_refs() | pending_refs(root)
        paths = select(root, m.base.open_files(), files, trees, refs, time.time(), m.base.digest_file)
        if not set(paths).issubset(authority['planned_paths']):
            raise ValueError('interim archive cohort changed after registration')
        return paths, {}

    m.candidates = candidates
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    os.nice(max(0, 19-os.getpriority(os.PRIO_PROCESS, 0)))
    m.base.write(c/'OWNER.json', m.base.owner())
    total = 0
    index = 0
    failures = 0
    try:
        while True:
            paths, _ = candidates()
            m.atomic(c/'STATUS.json', dict(unix=time.time(), eligible_files=len(paths),
                eligible_bytes=sum(Path(p).stat().st_size for p in paths), moved_bytes=total,
                cpu_free=shutil.disk_usage(root).free, new_sim_s=0))
            if not paths:
                m.base.write(c/'COMPLETE.json', dict(unix=time.time(), moved_bytes=total,
                    cpu_free=shutil.disk_usage(root).free, results_retained=True,
                    outcomes_closed=True, new_sim_s=0))
                return
            entries = []
            size = 0
            for item in sorted(paths.items()):
                entries.append(item)
                size += Path(item[0]).stat().st_size
                if size >= 64*1024**2:
                    break
            try:
                amount = m.batch(entries, index)
                if not amount:
                    raise RuntimeError('no certified progress; retain remaining sources')
                total += amount
                failures = 0
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                m.atomic(c/'TRANSPORT_ALERT.json', dict(unix=time.time(), error=repr(error),
                    sources_retained=True, new_sim_s=0))
                connection = isinstance(error, subprocess.TimeoutExpired) or any(s in str(error) for s in
                    ('Connection closed', 'Connection reset', 'timed out', 'name resolution',
                     'code 255', 'forwarded agent unavailable', 'Connection timed out'))
                failures += 1
                if not connection or failures > 5:
                    raise
                time.sleep((10, 20, 40, 80, 120)[failures-1])
            finally:
                index += 1  # a transfer namespace is never replayed
    except BaseException as error:
        import traceback
        m.atomic(c/'FAILED.json', dict(unix=time.time(), error=repr(error),
            traceback=traceback.format_exc(), moved_bytes=total, new_sim_s=0))
        raise


if __name__ == '__main__':
    import sys
    run(sys.argv[1])
