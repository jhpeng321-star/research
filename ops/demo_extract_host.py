"""DEMOX host-side tape reducer (stdlib + numpy only; read-only on the tapes; zero SUMO).

Reads a request JSON {"jobs": [{"key": str, "tape": <observable.jsonl.gz path>, "until": int}, ...]} and writes, for
each job, <out>/<key>.jsonl.gz = the tape's lines with time_s <= until, byte-for-byte (decompressed line text), plus
<out>/HOST_MANIFEST.json with the source tape sha256, size, line count and the written file sha256.
Runs at nice 19. Refuses any path containing a banned token (sealed/holdout/heldout/cal_split/chunk5/j4b/inc1,
hetdata_truth, driver_params, vType truth). Writes only under --out (must not exist yet, or be empty).

  nice -n 19 python3 demo_extract_host.py --request REQ.json --out /tmp/demox_out
"""
import argparse
import gzip
import hashlib
import json
import os
import sys

try:
    import numpy  # noqa: F401  (allowed dependency; not required)
except ImportError:  # pragma: no cover
    numpy = None

BANNED = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth', 'driver_params',
          'vtype')


def guard(path):
    low = str(path).lower().replace('\\', '/')
    for tok in BANNED:
        if tok in low:
            raise ValueError(f'refusing banned path token {tok!r}: {path}')
    return path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def reduce_tape(tape, until, dest):
    """Copy lines with int(round(time_s)) <= until (and stop after the first later line). Returns receipt."""
    guard(tape)
    n = 0
    last = None
    with gzip.open(tape, 'rt', encoding='utf-8') as src, gzip.open(dest, 'wt', encoding='utf-8', compresslevel=6) as out:
        for line in src:
            t = int(round(float(json.loads(line)['time_s'])))
            if t > int(until):
                break
            out.write(line if line.endswith('\n') else line+'\n')
            n += 1
            last = t
    return dict(tape=tape, tape_sha256=sha256_file(tape), tape_bytes=os.path.getsize(tape), until=int(until),
                lines=n, last_t=last, out=os.path.basename(dest), out_sha256=sha256_file(dest))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--request', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        os.nice(19 - os.nice(0))
    except (AttributeError, OSError):
        pass
    guard(a.request)
    guard(a.out)
    req = json.load(open(a.request, encoding='utf-8'))
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit(f'--out {a.out} exists and is not empty')
    os.makedirs(a.out, exist_ok=True)
    receipts, errors = [], []
    for job in req['jobs']:
        key = job['key']
        if '/' in key or '\\' in key or key.startswith('.'):
            raise ValueError(f'bad key {key!r}')
        try:
            receipts.append(dict(key=key, **reduce_tape(job['tape'], job['until'], os.path.join(a.out, key+'.jsonl.gz'))))
        except (OSError, ValueError, KeyError) as exc:
            errors.append(dict(key=key, tape=job.get('tape'), error=f'{type(exc).__name__}: {exc}'))
    manifest = dict(schema='demox.host.v1', request_sha256=sha256_file(a.request), receipts=receipts, errors=errors)
    with open(os.path.join(a.out, 'HOST_MANIFEST.json'), 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write('\n')
    print(json.dumps(dict(jobs=len(receipts), errors=len(errors))))
    return 0 if not errors else 2


if __name__ == '__main__':
    sys.exit(main())
