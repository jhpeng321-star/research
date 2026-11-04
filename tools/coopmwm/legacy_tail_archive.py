"""Scope adapter for the completed legacy run's final five TRAIN updates.

Transfer, SHA verification, fresh FD/reference checks and metric retention reuse
the completed old_ppo_archive_recovery2 worker unchanged. No V2V files belong here.
"""
import ast
import hashlib
from pathlib import PurePosixPath
import re


def in_scope(name):
    p = PurePosixPath(name)
    if '..' in p.parts:
        return False
    try:
        rel = p.relative_to('/home/ubuntu/coopmwm20261102_warm1/jobs')
    except ValueError:
        return False
    if len(rel.parts) != 2 or rel.name not in ('RESULT.json', 'RESULT.json.gz'):
        return False
    match = re.fullmatch(r'replay0735_warm_u([0-9]{3})_([BR])_(986[0-9]{3})', rel.parts[0])
    return bool(match and 19 <= int(match[1]) <= (21 if match[2] == 'B' else 20)
                and 986000 <= int(match[3]) <= 986071)


def build_worker(source, expected_sha256):
    """Replace only the scope; pin the already-completed transfer implementation."""
    if hashlib.sha256(source.encode()).hexdigest() != expected_sha256:
        raise ValueError('completed predecessor source SHA differs')
    tree = ast.parse(source)
    first = tree.body[0]
    if not isinstance(first, ast.FunctionDef) or first.name != 'in_scope':
        raise ValueError('predecessor scope ABI differs')
    import inspect
    prefix = 'from pathlib import PurePosixPath\nimport re\n' + inspect.getsource(in_scope)
    lines = source.splitlines(keepends=True)
    worker = prefix + ''.join(lines[first.end_lineno:])
    anchor = "sys.path.insert(0,'/home/ubuntu/coopmwm20261104_archiveflatcpu1');import rolling_archive as m"
    if worker.count(anchor) != 1:
        raise ValueError('predecessor transfer ABI differs')
    guards = '''import fcntl
lock=(r/'RUN.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
stop=run/'STOPPED_PERMANENT_MAIN0910.json'
assert hashlib.sha256(stop.read_bytes()).hexdigest()==authority['permanent_stop_sha256'],'permanent-stop authority changed'
assert not any(Path('/proc',str(pid)).exists() for pid in (3305070,3305071,3305066,3305068,3305069)),'legacy owner unexpectedly present'
assert not (r/'OWNER.json').exists(),'adopt existing tail archive owner'
'''
    worker = worker.replace(anchor, guards + anchor)
    compile(worker, '<legacy-tail-sha-archive>', 'exec')
    return worker
