"""New namespace, unchanged append-only accounting, all historical charges protected."""
import types
import gzip
import hashlib
import json
from pathlib import Path
from tools.coopnative import storage as old
from .accel_protocol import CAP,PHASES,SUPERSEDED_CHARGE
from .storage import ledger
_reserve=types.FunctionType(old.reserve.__code__,dict(old.reserve.__globals__,CAP=CAP-SUPERSEDED_CHARGE,CAPS=PHASES))


def read(path):
    value=old.read(path)
    if isinstance(value,dict) and value.get('encoding')=='coopmwm-lossless-gzip-v1':
        if value['file']!='RESULT.json.gz':raise ValueError('invalid result payload path')
        payload=gzip.decompress(old.checked(Path(path).parent/value['file']).read_bytes())
        if hashlib.sha256(payload).hexdigest()!=value['sha256']:raise ValueError('compressed result digest mismatch')
        return json.loads(payload)
    return value


def write(path,value):
    path=old.checked(path)
    if path.name!='RESULT.json':return old.write(path,value)
    # One-second tapes are highly redundant. Preserve exact JSON bytes/digest,
    # with an immutable small completion marker only after payload fsync.
    payload=(old.canonical(value)+'\n').encode('utf-8')
    target=path.parent/'RESULT.json.gz';target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:
        f.write(gzip.compress(payload,compresslevel=3,mtime=0));f.flush();old.os.fsync(f.fileno())
    old.write(path,dict(encoding='coopmwm-lossless-gzip-v1',file=target.name,
        sha256=hashlib.sha256(payload).hexdigest(),uncompressed_bytes=len(payload)))


def reserve(root,jobs,phase):
    import fcntl
    with (root/'ACCOUNTING.claim').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _reserve(root,jobs,phase)


def charged(root):return SUPERSEDED_CHARGE+sum(x['sim_s'] for x in ledger(root))
