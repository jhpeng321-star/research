"""Read-only: sha256 of the DECOMPRESSED vehicle logs of two job series (A69 registered supplementary check).
usage: vlcmp.py <final_root> <tagA> <tagB> <n>   (jobs <tag>_0000 ..; every dir of a job, incl. replay dirs, is listed)"""
import gzip
import hashlib
import json
from pathlib import Path
import sys

root, ta, tb, n = Path(sys.argv[1])/'evaluation', sys.argv[2], sys.argv[3], int(sys.argv[4])
NL = bytes([10])


def one(job):
    out = []
    for d in sorted(d for d in root.glob(job+'*') if (d/'vehicles.jsonl.gz').is_file()):
        with gzip.open(d/'vehicles.jsonl.gz', 'rb') as f:
            raw = f.read()
        out.append(dict(dir=d.name, sha256=hashlib.sha256(raw).hexdigest(), lines=raw.count(NL), bytes=len(raw)))
    return out


rows = []
for i in range(n):
    a, b = one(f'{ta}_{i:04}'), one(f'{tb}_{i:04}')
    rows.append(dict(index=i, a=a, b=b, equal=len(a) == 1 and len(b) == 1 and a[0]['sha256'] == b[0]['sha256']))
print(json.dumps(dict(all_equal=all(r['equal'] for r in rows), rows=rows)))
