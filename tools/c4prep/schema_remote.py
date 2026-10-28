"""Inspect only observable manifests/arrays for registered historical pairs."""
from pathlib import Path
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
    root = R / f'arm/loop/data/pairs_c{cycle:02}.a1'
    manifest = json.loads(safe(root / 'C4_EPISODES_MANIFEST.json').read_text())
    ep = manifest['episodes'][0]
    path = safe(root / ep['dir'])
    print('episode_files', cycle, [p.name for p in path.iterdir() if not BAD.search(p.name)])
    for name in ('meta.json', 'metadata.json'):
        p = safe(path / name)
        if p.is_file():
            print('metadata', cycle, p.read_text()[:4000])
    with np.load(safe(R / f'arm/loop/returns/returns_c{cycle:02}.a1.npz'), allow_pickle=False) as z:
        print('returns_meta', cycle, {k: z[k].tolist() for k in ('gamma', 'units', 'accounting', 'terminal_semantics', 'policy_checkpoint')})
        print('start_queues', cycle, z['state__entry_backlog'][:4].tolist())
    worlds = json.loads(safe(R / f'arm/loop/worlds_c{cycle:02}.json').read_text())
    artifact = safe(next(iter(worlds['worlds'].values()))['artifact'])
    summary = safe(artifact / 'summaries/0')
    if summary.is_dir():
        print('summary_files', cycle, [p.name for p in summary.iterdir() if not BAD.search(p.name)])
    pairs = safe(artifact / 'pairs')
    doc = json.loads(safe(pairs / 'C4_EPISODES_MANIFEST.json').read_text())
    print('artifact_pair_first', cycle, doc['episodes'][0])
    print('artifact_pair_files', cycle, [p.name for p in safe(pairs / doc['episodes'][0]['dir']).iterdir() if not BAD.search(p.name)])
for rel in ('arm/loop/policy/c01.a3', 'arm/loop/policy/c03.a6', 'arm/loop/longint/c3resume/r2'):
    p = safe(R / rel)
    print('receipts', rel, [q.name for q in p.iterdir() if q.suffix == '.json' and not BAD.search(q.name)])
