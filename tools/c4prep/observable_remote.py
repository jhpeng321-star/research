"""Search only known collection directories for observable C3 tape fields.

Never descend into scenario directories or open route/type assignment files.
"""
from pathlib import Path
import json
import re
import numpy as np

BAD = re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype', re.I)
home = Path.home()
root = home / 'c4/main14/restart20261026/work'
for seed in (968183, 968187):
    world = root / f'w{seed}'
    hits = []
    if world.is_dir():
        hits = [p for p in world.glob('*/c03*/collect/branches/policy/*/t0360_policy/episode.npz')
                if not BAD.search(str(p)) and not BAD.search(str(p.resolve()))]
    for p in hits:
        with np.load(p, allow_pickle=False) as z:
            names = [k for k in z.files if not BAD.search(k)]
            print('tape_fields', str(p), names)
            for name in ('entry_backlog', 'request_counter', 'entry_cav_counter', 'source_wait', 'source_wait_vehicle_seconds'):
                if name in names:
                    print('clock_shape', name, list(z[name].shape))
for base in (home / 'backlog/reexport', home / 'backlog/stage'):
    print('existing_reexports', str(base),
          [p.name for p in base.iterdir() if p.is_dir() and p.name.startswith('pairs_c0') and not BAD.search(p.name)]
          if base.is_dir() else [])
for rel in ('entrant/FACTORIAL.json', 'entrant/VERDICT.json', 'entrant/RESULTS.json'):
    p = home / rel
    if p.is_file():
        print('entrant_receipt', rel, p.read_text()[:4000])
