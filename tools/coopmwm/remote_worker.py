"""Option A remote side (operator 2026-11-07): the replica host's half of RemotePool (remote_wave.py).

Runs on the replica host (cpuV) from the frozen source tree deployed at the IDENTICAL path, against the replica
run root at the IDENTICAL path. It never reserves, charges or writes a ledger: the replica's CHARGES.jsonl is a
DIRECTORY, so any accidental ledger write fails loudly, and every op refuses a replica whose CHARGES.jsonl is not
a directory. Every op prints ONE result line (MARKER + JSON) and exits 0, or prints MARKER + {"error": ...} and
exits 3 (refusal); the caller turns anything else into RemoteWaveLost.

Staging (outside jobs/, never landed): <root>/remote_r/<job id>/{JOB.pkl, JOB.json, PID.json, DONE.json,
ERROR.json, OPENED.json, worker.log}. The worker calls the SAME episode function as the local pool
(accel_plant.episode) on the unpickled job dict; everything it writes lives in jobs/<job id>/ except a registered
health pause (PAUSE_*.json at the root), which the poller pulls back.

Ops: check | admit | start | run | status | manifest | cleanup | abort.
CLI: python -B -m tools.coopmwm.remote_worker <op> --root <run root> [--job <id>]   (payload on stdin)
"""
import argparse
import base64
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

MARKER = 'OPTIONA_RESULT '
STAGE = 'remote_r'
FORBIDDEN = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth')
# Root-level control files mirrored cpuZ -> replica on every poll (health guards read them each tick).
SIGNAL_PATTERNS = ('STOP.json', 'FAULT.json', 'PAUSE_*.json', 'WARM_*.json', '*RELEASE*.json',
                   '*_REPORT_ONLY_*.json', '*_RULE_*.json', '*DIAGNOSIS*.json', 'AUTHORITY.md')
PULLBACK_PATTERNS = ('PAUSE_*.json',)


class Refused(Exception):
    pass


def guard(path):
    text = str(path).lower()
    if any(word in text for word in FORBIDDEN):
        raise Refused('forbidden path component: '+str(path))
    return Path(path)


def file_sha(path):
    h = hashlib.sha256()
    with guard(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def tree_manifest(directory):
    """{posix relative path: sha256} of every regular file below ``directory`` (sorted; {} when absent;
    __pycache__ skipped)."""
    directory = guard(directory)
    if not directory.is_dir():
        return {}
    out = {}
    for path in sorted(directory.rglob('*')):
        if '__pycache__' in path.relative_to(directory).parts:
            continue      # interpreter byte caches differ per host/run (workers run python -B)
        if path.is_file():
            out[path.relative_to(directory).as_posix()] = file_sha(path)
    return out


def path_digest(path):
    """File: its sha256. Directory: sha256 of its canonical tree manifest. Missing: None."""
    path = guard(path)
    if path.is_file():
        return file_sha(path)
    if path.is_dir():
        return hashlib.sha256(json.dumps(tree_manifest(path), sort_keys=True).encode()).hexdigest()
    return None


def signal_names(root):
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_file()
                  and any(fnmatch.fnmatchcase(p.name, pattern) for pattern in SIGNAL_PATTERNS))


def _charges_dir(root):
    if not (Path(root)/'CHARGES.jsonl').is_dir():
        raise Refused('replica CHARGES.jsonl must be a directory (replica ledger writes are refused)')


def _publish_bytes(path, data):
    """Complete temp file, then an exclusive hard link (refuses an existing target)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True)+'\n').encode()


def _versions():
    from importlib import metadata
    out = dict(python=sys.version.split()[0], executable=sys.executable)
    for name in ('numpy', 'torch'):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _meminfo_gb():
    try:
        rows = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        return int(rows['MemAvailable'].split()[0])/1024**2
    except (OSError, KeyError, ValueError):
        return None


def admit_numbers(root):
    root = Path(root)
    try:
        load1 = os.getloadavg()[0]
    except (AttributeError, OSError):
        load1 = None
    return dict(disk_free_gb=shutil.disk_usage(root).free/1024**3, mem_available_gb=_meminfo_gb(), load1=load1)


# ------------------------------------------------------------------ ops (pure functions; the CLI wraps them)

def op_check(root, payload, *, replica=True):
    """Replica identity facts. The SAME function runs on cpuZ (replica=False) so the caller compares like with like."""
    root = Path(root)
    if replica:
        _charges_dir(root)
    source = Path(payload['source_root'])
    manifest_path = source/'SOURCE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    bad = sorted(name for name, sha in manifest['files'].items() if path_digest(source/name) != sha)
    envsh = payload.get('envsh')
    return dict(source_manifest_sha256=file_sha(manifest_path), source_files_bad=bad,
                config_sha256=file_sha(root/'CONFIG.json'),
                inputs={p: path_digest(p) for p in sorted(payload.get('inputs', ()))},
                envsh_sha256=path_digest(envsh) if envsh else None,
                versions=_versions(), ld_preload=os.environ.get('LD_PRELOAD'),
                signals={name: file_sha(root/name) for name in signal_names(root)},
                charges_is_dir=(root/'CHARGES.jsonl').is_dir())


def op_admit(root, payload=None):
    _charges_dir(root)
    return admit_numbers(root)


def op_start(root, header, data, *, spawn=None):
    """Refuse an existing remote job dir; verify the job bytes; stage them; start ONE detached worker."""
    root = Path(root)
    _charges_dir(root)
    ident = header['id']
    if not ident or '/' in ident or '\\' in ident or ident in ('.', '..'):
        raise Refused('invalid job id')
    if len(data) != header['size'] or hashlib.sha256(data).hexdigest() != header['sha256']:
        raise Refused('job bytes do not match their sha256 (transport)')
    if (root/'jobs'/ident).exists():
        raise Refused('remote job directory already exists: '+ident)
    stage = root/STAGE/ident
    stage.parent.mkdir(parents=True, exist_ok=True)
    try:
        stage.mkdir()
    except FileExistsError:
        raise Refused('remote job staging already exists: '+ident) from None
    _publish_bytes(stage/'JOB.pkl', data)
    _publish_bytes(stage/'JOB.json', _json_bytes(header))
    pid, ticks = (spawn or default_spawn)(root, ident, header)
    _publish_bytes(stage/'PID.json', _json_bytes(dict(pid=pid, start_ticks=ticks, unix=time.time())))
    return dict(started=ident, pid=pid)


def _start_ticks(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def default_spawn(root, ident, header):
    import subprocess
    env = dict(os.environ)
    env.update(header.get('env') or {})
    stage = Path(root)/STAGE/ident
    with (stage/'worker.log').open('xb') as log:
        child = subprocess.Popen(['nice', '-n', '19', header['python'], '-B', '-m', 'tools.coopmwm.remote_worker',
                                  'run', '--root', str(root), '--job', ident],
                                 cwd=header['source_root'], env=env, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    return child.pid, _start_ticks(child.pid)


def op_run(root, ident, *, episode=None, job_root=None):
    """The detached worker body. Setup errors are transport-class (RemoteWaveLost upstream); episode errors are
    reported like a local pool worker's exception."""
    root, phase = Path(root), 'setup'
    stage = root/STAGE/ident
    opened = set()
    try:
        _charges_dir(root)
        header = json.loads((stage/'JOB.json').read_text(encoding='utf-8'))
        data = (stage/'JOB.pkl').read_bytes()
        if hashlib.sha256(data).hexdigest() != header['sha256']:
            raise Refused('staged job bytes changed')
        import pickle
        job = pickle.loads(data)
        home = root if job_root is None else Path(job_root)   # job_root: CPU tests only (paths are identical)
        if (job['id'] != ident or Path(job['path']) != home/'jobs'/ident or Path(job['queue_root']) != home
                or job['arm'] != 'R'):
            raise Refused('job dict does not belong to this replica path/arm')
        if header.get('audit_open'):
            jobdir = str(root/'jobs'/ident)

            def hook(event, args):
                if event == 'open' and isinstance(args[0], (str, os.PathLike)):
                    name = os.fspath(args[0])
                    if not name.startswith(jobdir) and not name.startswith(str(stage)):
                        opened.add(name)
            sys.addaudithook(hook)
        if episode is None:
            from .accel_plant import episode
        phase = 'episode'
        episode(job)
        phase = 'done'
        _publish_bytes(stage/'DONE.json', _json_bytes(dict(unix=time.time(), id=ident)))
        return 0
    except BaseException as error:
        args = list(error.args) if all(isinstance(a, (str, int, float)) for a in error.args) else [str(error)]
        _publish_bytes(stage/'ERROR.json', _json_bytes(dict(phase=phase, type=type(error).__name__,
            module=type(error).__module__, args=args, message=str(error), repr=repr(error),
            traceback=traceback.format_exc(), unix=time.time())))
        return 1
    finally:
        if opened:
            try:
                _publish_bytes(stage/'OPENED.json', _json_bytes(sorted(opened)))
            except OSError:
                pass


def _alive(record):
    pid, ticks = record.get('pid'), record.get('start_ticks')
    try:
        os.kill(pid, 0)
    except (OSError, TypeError):
        return False
    return ticks is None or _start_ticks(pid) == ticks


def op_status(root, payload, *, alive=_alive):
    """Mirror the caller's control files, report each job, and return replica-only health pauses to pull back."""
    root = Path(root)
    _charges_dir(root)
    sent = payload.get('signals') or {}
    for name, item in sent.items():
        if '/' in name or '\\' in name or not any(fnmatch.fnmatchcase(name, p) for p in SIGNAL_PATTERNS):
            raise Refused('not a mirrored control file: '+name)
        data = base64.b64decode(item['b64'])
        target = root/name
        if not target.exists() or target.read_bytes() != data:
            temporary = target.with_name('.'+name+f'.{os.getpid()}.mirror')
            temporary.write_bytes(data)
            os.replace(temporary, target)
        os.utime(target, ns=(item['mtime_ns'], item['mtime_ns']))
    jobs = {}
    for ident in payload.get('ids', ()):
        stage = root/STAGE/ident
        if (stage/'DONE.json').is_file():
            jobs[ident] = dict(state='done')
        elif (stage/'ERROR.json').is_file():
            jobs[ident] = dict(state='error', error=json.loads((stage/'ERROR.json').read_text(encoding='utf-8')))
        elif (stage/'PID.json').is_file():
            jobs[ident] = dict(state='running' if alive(json.loads((stage/'PID.json').read_text())) else 'dead')
        else:
            jobs[ident] = dict(state='missing')
    pull = {}
    for path in sorted(root.glob('PAUSE_*.json')):
        if path.name not in sent and path.is_file():
            pull[path.name] = base64.b64encode(path.read_bytes()).decode()
    return dict(jobs=jobs, pullback=pull, admit=admit_numbers(root))


def op_manifest(root, ident):
    root = Path(root)
    _charges_dir(root)
    jobdir = root/'jobs'/ident
    manifest = tree_manifest(jobdir)
    pointer = jobdir/'RESULT.json'
    opened = root/STAGE/ident/'OPENED.json'
    return dict(manifest=manifest, pointer_b64=base64.b64encode(pointer.read_bytes()).decode()
                if pointer.is_file() else None,
                opened=json.loads(opened.read_text()) if opened.is_file() else None)


def op_cleanup(root, payload):
    """Delete replica copies only when they still equal the manifest the caller re-verified on cpuZ."""
    root = Path(root)
    _charges_dir(root)
    out = {}
    for ident, expected in sorted(payload['expected'].items()):
        jobdir, stage = root/'jobs'/ident, root/STAGE/ident
        if not jobdir.exists() and not stage.exists():
            out[ident] = 'absent'
            continue
        if tree_manifest(jobdir) != expected:
            out[ident] = 'mismatch'
            continue
        if jobdir.exists():
            shutil.rmtree(jobdir)
        if stage.exists():
            shutil.rmtree(stage)
        out[ident] = 'deleted'
    return dict(cleanup=out, admit=admit_numbers(root))


def op_abort(root, payload, *, alive=_alive):
    """Best effort after a lost wave: SIGTERM our own still-running workers, by recorded PID + start ticks."""
    import signal
    root, out = Path(root), {}
    for ident in payload.get('ids', ()):
        stage = root/STAGE/ident
        if (stage/'DONE.json').exists() or (stage/'ERROR.json').exists() or not (stage/'PID.json').exists():
            out[ident] = 'not_running'
            continue
        record = json.loads((stage/'PID.json').read_text())
        if alive(record):
            os.kill(record['pid'], signal.SIGTERM)
            out[ident] = 'sigterm'
        else:
            out[ident] = 'dead'
    return dict(abort=out)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('op', choices=['check', 'admit', 'start', 'run', 'status', 'manifest', 'cleanup', 'abort'])
    parser.add_argument('--root', required=True)
    parser.add_argument('--job')
    args = parser.parse_args(argv)
    if args.op == 'run':
        return op_run(args.root, args.job)
    raw = sys.stdin.buffer.read()
    try:
        if args.op == 'start':
            line, _, data = raw.partition(b'\n')
            value = op_start(args.root, json.loads(line), data)
        else:
            payload = json.loads(raw) if raw.strip() else {}
            if args.op == 'check':
                value = op_check(args.root, payload)
            elif args.op == 'admit':
                value = op_admit(args.root, payload)
            elif args.op == 'status':
                value = op_status(args.root, payload)
            elif args.op == 'manifest':
                value = op_manifest(args.root, args.job)
            elif args.op == 'cleanup':
                value = op_cleanup(args.root, payload)
            else:
                value = op_abort(args.root, payload)
    except Refused as error:
        sys.stdout.write(MARKER+json.dumps(dict(error=str(error)))+'\n')
        return 3
    sys.stdout.write(MARKER+json.dumps(value, sort_keys=True)+'\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
