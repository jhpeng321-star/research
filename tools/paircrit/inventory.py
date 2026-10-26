"""Read only specifically authorized TRAIN numeric-artifact metadata."""
from pathlib import Path
import json
import numpy as np

root = Path('/root/main14_loop')
for name in ('returns/returns_c00.a1.receipt.json', 'returns/returns_c00.a1.npz',
             'data/pairs_c00.a1/C4_EPISODES_MANIFEST.json'):
    p = root / name
    print(str(p), p.exists())
    if not p.exists():
        continue
    if p.suffix == '.npz':
        with np.load(p, allow_pickle=False) as z:
            print({k: z[k].shape for k in z.files})
    else:
        d = json.loads(p.read_text())
        print('keys', list(d))
        rows = d.get('entries', d.get('episodes', []))
        print('first', rows[:1])
p = root / 'data/pairs_c00.a1/episodes/w968129/w968129m1900p05t300__loop14_pairs_b0360_native'
print('numeric names', [f.name for f in p.iterdir() if f.suffix == '.npy'])
p = root / 'worlds/c00/w968129.a1/artifact/summaries/0/summary.json'
print('native summary', json.loads(p.read_text()))
