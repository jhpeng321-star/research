"""Read-only explicit observable input inventory, executed on gpuA."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np

R = Path('/root/autodl-tmp/c4/restart20261026')
BAD = re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype', re.I)


def safe(p):
    p = Path(p)
    assert not BAD.search(str(p)) and not BAD.search(str(p.resolve()))
    return p


for cycle in (1, 2, 3):
    base = R / f'arm/loop/data/pairs_c{cycle:02}.a1'
    manifest = json.loads(safe(base / 'C4_EPISODES_MANIFEST.json').read_text())
    print('manifest', cycle, json.dumps(manifest['episodes'][0]))
    pair = R / f'arm/loop/returns/pairing_c{cycle:02}.a1.json'
    doc = json.loads(safe(pair).read_text())
    print('pairing', cycle, json.dumps({k: v for k, v in doc.items() if k != 'pairs'}))
    print('first_pair', cycle, json.dumps(doc['pairs'][0]))
    with np.load(safe(R / f'arm/loop/returns/returns_c{cycle:02}.a1.npz'), allow_pickle=False) as z:
        print('returns_fields', cycle, {k: list(z[k].shape) for k in z.files})
    worlds = json.loads(safe(R / f'arm/loop/worlds_c{cycle:02}.json').read_text())
    world = next(iter(worlds['worlds'].values()))
    artifact = safe(world['artifact'])
    print('artifact', cycle, str(artifact), sorted(p.name for p in artifact.iterdir() if not BAD.search(p.name)))
    sub = safe(artifact / 'obs')
    if sub.is_dir():
        print('obs', cycle, [p.name for p in sub.iterdir() if not BAD.search(p.name)])
for name in ('wmlossonly20261028',):
    base = Path('/root/autodl-tmp/c4') / name
    print(name, sorted(p.name for p in safe(base).iterdir() if not BAD.search(p.name)))
