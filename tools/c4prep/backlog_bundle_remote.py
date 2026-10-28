"""Export ONLY already-converted clocks; inspect C3 tape schemas without assignments."""
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile

import numpy as np

BAD = re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype', re.I)


def safe(p):
    p = Path(p)
    assert not BAD.search(str(p)) and not BAD.search(str(p.resolve()))
    return p


home = Path.home()
audit = []
root = home / 'c4/main14/restart20261026/work'
for seed in range(968183, 968196):
    for p in sorted(root.glob(f'w{seed}/*/c03*/collect/branches/policy/*/*/episode.npz')):
        p = safe(p)
        with np.load(p, allow_pickle=False) as z:
            fields = [k for k in z.files if not BAD.search(k)]
        files = [q.name for q in p.parent.iterdir() if q.is_file() and not BAD.search(q.name)]
        audit.append({'tape': str(p), 'fields': fields, 'files': files,
                      'complete_clocks': all(k in fields for k in
                          ('entry_backlog', 'request_counter', 'entry_cav_counter'))})
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode='w:xz', preset=6) as tar:
    def add(name, blob):
        info = tarfile.TarInfo(name)
        info.size = len(blob)
        tar.addfile(info, io.BytesIO(blob))
    add('C3_TAPE_AUDIT.json', json.dumps(audit, indent=1).encode())
    base = home / 'backlog/reexport'
    for cycle in ((0, 1, 2) if sys.argv[-1:] != ['audit-only'] else ()):
        name = f'pairs_c{cycle:02}.a1.backlog'
        p = safe(base / name)
        if not p.is_dir():
            continue
        sums = json.loads((p / 'SHA256SUMS.json').read_text())
        for rel in ('C4_EPISODES_MANIFEST.json', 'SHA256SUMS.json'):
            add(name + '/' + rel, safe(p / rel).read_bytes())
        for rel, digest in sums.items():
            if Path(rel).name not in ('entry_backlog.npy', 'request_counter.npy',
                                      'entry_cav_counter.npy', 'episode.json'):
                continue
            blob = safe(p / rel).read_bytes()
            assert hashlib.sha256(blob).hexdigest() == digest
            add(name + '/' + rel, blob)
sys.stdout.buffer.write(buf.getvalue())
