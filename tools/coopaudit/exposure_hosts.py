"""Filename metadata only, pruning before descent and never opening data files."""
import hashlib
import json
import os
import re
from pathlib import Path

bad = re.compile('sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype|driver_params|vehicle_params|assignment', re.I)
seeds = set(range(969650, 969678)) | {969699}
root = Path('/root') if os.getuid() == 0 else Path('/home/ubuntu')
hits, count, index = [], 0, hashlib.sha256()
for folder, dirs, names in os.walk(root, followlinks=False):
    dirs[:] = [d for d in dirs if not bad.search(str(Path(folder)/d))
               and d not in ('.git', '.cache', 'miniconda3', '.venv', '__pycache__', 'node_modules')
               and not (Path(folder)/d).is_symlink()]
    for name in names:
        p = Path(folder)/name
        if bad.search(str(p)) or p.is_symlink():
            continue
        count += 1
        index.update(str(p).encode())
        found = seeds & {int(s) for s in re.findall(r'(?<!\d)969\d{3}(?!\d)', name)}
        if found:
            hits.append(dict(path=str(p), seeds=sorted(found)))
print(json.dumps(dict(root=str(root), files=count, hits=hits, path_index_sha256=index.hexdigest()), indent=1))
