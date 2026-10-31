"""Pure mechanics: exact RNG-only patch, budget admission and tape comparison."""
import hashlib
import json
import os
import re
from pathlib import Path

DENIED = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
          'hetdata_truth', 'vtype')
RNG = re.compile(rb'<rngState\b[^>]*?/>|<rngState\b[^>]*>.*?</rngState>', re.S)


def allowed(path):
    if any(x in str(path).lower() for x in DENIED):
        raise PermissionError('excluded path')
    return Path(path)


def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        allowed(os.fsdecode(args[0]))


def digest(path):
    return hashlib.sha256(allowed(path).read_bytes()).hexdigest()


def write(path, value):
    with allowed(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=1, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


def seed_guard(seed):
    if type(seed) is not int or not 966000 <= seed <= 969999:
        raise ValueError('only registered TRAIN seed initializers')


def rng_only(data):
    """Retain every non-RNG byte, including driver attributes and rngIndex."""
    patched, count = RNG.subn(b'', data)
    if not count or b'<rngState' in patched:
        raise ValueError('unrecognized or missing saved SUMO RNG state')
    return patched, count


def admit(old, job):
    if job['charge'] <= 0 or job['charge'] > 1500:
        raise ValueError('invalid per-attempt reservation')
    if len({r['id'] for r in old}) != len(old):
        raise ValueError('corrupt duplicate ledger')
    if any(r['id'] == job['id'] for r in old):
        raise ValueError('attempt already reserved: no retry')
    if sum(r['sim_s'] for r in old)+job['charge'] > 30000:
        raise ValueError('hard simulation budget exceeded')


def reserve(path, job, binding):
    old = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    admit(old, job)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(id=job['id'], sim_s=job['charge'], binding=binding))+'\n')
        f.flush()
        os.fsync(f.fileno())


def compare_tapes(first, second, end=600):
    import numpy as np
    with np.load(allowed(first), allow_pickle=False) as a, np.load(allowed(second), allow_pickle=False) as b:
        ia = np.flatnonzero((a['t'] >= 300) & (a['t'] <= end))
        ib = np.flatnonzero((b['t'] >= 300) & (b['t'] <= end))
        ia = ia[np.lexsort((a['vid'][ia], a['t'][ia]))]
        ib = ib[np.lexsort((b['vid'][ib], b['t'][ib]))]
        result = {}
        for k in ('t','vid','s','v','a','lane'):
            x,y = a[k][ia],b[k][ib]
            result[k] = dict(identical=x.dtype == y.dtype and x.tobytes() == y.tobytes(),
                             rows_first=len(x), rows_second=len(y))
        return result
