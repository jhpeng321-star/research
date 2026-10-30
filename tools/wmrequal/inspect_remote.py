"""Read only explicitly named candidate inputs; no recursive traversal."""
from pathlib import Path
import hashlib
import json

roots = ['/root/autodl-tmp/c4/e3data',
         '/root/autodl-tmp/c4/panel120/package',
         '/root/autodl-tmp/c4/longloss20261027',
         '/root/autodl-tmp/c4/restart20261026',
         '/root/autodl-tmp/c4/wmcombined20261027',
         '/root/autodl-tmp/c4/cct2inc20261029',
         '/root/autodl-tmp/c4/v5/pairs300']
for root in roots:
    p = Path(root)
    print('ROOT', root, p.is_dir())
    if p.is_dir():
        print('NAMES', [q.name for q in p.iterdir() if not any(
            x in q.name.lower() for x in ('sealed','holdout','heldout','cal_split',
                                         'chunk5','j4b','inc1','hetdata_truth','vtype'))])
    for name in ('C4_EPISODES_MANIFEST.json', 'SPLIT.json', 'DESIGN.json'):
        q = p/name
        if not q.is_file():
            continue
        raw = q.read_bytes()
        d = json.loads(raw)
        print(name, 'SHA256', hashlib.sha256(raw).hexdigest())
        if name == 'C4_EPISODES_MANIFEST.json':
            print('MANIFEST', {k:v for k,v in d.items() if k != 'episodes'})
            for role in ('train','val','validation','calibration'):
                e = next((e for e in d['episodes'] if e.get('split_role') == role), None)
                if e:
                    print('EXAMPLE', e)
        elif name == 'DESIGN.json':
            print('DESIGN', json.dumps(d)[:14000])
        else:
            print('SPLIT', json.dumps(d)[:1000])
