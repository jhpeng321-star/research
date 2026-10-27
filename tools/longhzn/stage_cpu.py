"""Copy safe existing CPU source and TRAIN panel into an exclusive task directory."""
import hashlib
import json
import os
import shutil
from pathlib import Path

BAD = ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vtype')
home = Path.home()
assert home.is_dir()
dest = home/'longhzn'
dest.mkdir(exist_ok=True)
assert not (dest/'STAGED.json').exists()
source = home/'c4/main14/restart20261026/runtime_c1'
manifest = {}
for directory in ('src','tools','scripts','configs','traffic_shepherd'):
    for parent, dirs, files in os.walk(source/directory):
        dirs[:] = [d for d in dirs if d != '__pycache__' and not any(t in d.lower() for t in BAD)]
        for name in files:
            p = Path(parent)/name
            relative = p.relative_to(source)
            if any(t in str(relative).lower() for t in BAD) or p.suffix not in ('.py','.toml','.json'):
                continue
            data = p.read_bytes()
            target = dest/'runtime'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            manifest[str(relative)] = hashlib.sha256(data).hexdigest()
panel = home/'cqr/pairs300'
p = panel/'C4_EPISODES_MANIFEST.json'
doc = json.loads(p.read_text())
assert hashlib.sha256(p.read_bytes()).hexdigest() == 'd64fc0f2088531dc09da8dfed09ea75f4dab412024ba8c4942bc01949ebea19a'
assert all(e['split_role'] in ('train','val') and not 964000 <= int(e['seed']) <= 964199 for e in doc['episodes'])
assert not any(t in json.dumps(doc).lower() for t in BAD)
files = []
for parent, dirs, names in os.walk(panel):
    assert not any(t in str(parent).lower() for t in BAD)
    dirs[:] = [d for d in dirs if not any(t in d.lower() for t in BAD)]
    for name in names:
        if any(t in name.lower() for t in BAD):
            continue
        p = Path(parent)/name
        target = dest/'pairs300'/p.relative_to(panel)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        files.append((str(p.relative_to(panel)), p.stat().st_size))
proof = {'source_files': manifest, 'panel_manifest_sha256': hashlib.sha256((panel/'C4_EPISODES_MANIFEST.json').read_bytes()).hexdigest(),
         'panel_source': str(panel), 'panel_files': len(files), 'panel_bytes': sum(s for _,s in files),
         'free_bytes': shutil.disk_usage(dest).free}
(dest/'STAGED.json').write_text(json.dumps(proof, indent=1))
print(json.dumps({k:v for k,v in proof.items() if k != 'source_files'}))
