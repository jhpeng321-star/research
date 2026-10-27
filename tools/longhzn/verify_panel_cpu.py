"""Verify selected episode bytes and make CPUQ's selected panel independent copies."""
import hashlib
import json
import os
import shutil
from pathlib import Path

os.nice(19)
root = Path.home()/'longhzn'
original = json.loads((root/'original_gate.json').read_text())
keys = {r['pair_key'] for r in original['rows']}
doc = json.loads((root/'pairs300/C4_EPISODES_MANIFEST.json').read_text())
entries = [e for e in doc['episodes'] if e.get('pair_key') in keys]
assert len(entries) == 288
linked = 'panel_storage' in json.loads((root/'STAGED.json').read_text())
proof = {}
total = 0
for entry in entries:
    relative = Path(entry['dir'])
    assert not relative.is_absolute() and '..' not in relative.parts
    for parent, dirs, names in os.walk(root/'pairs300'/relative):
        for name in names:
            path = Path(parent)/name
            assert not any(t in str(path).lower() for t in ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vtype'))
            rel = path.relative_to(root/'pairs300')
            source = Path.home()/'cqr/pairs300'/rel
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
            if linked:
                tmp = path.with_name(path.name+'.longhzn-copy')
                assert not tmp.exists()
                tmp.write_bytes(data)
                tmp.replace(path)
                assert path.stat().st_ino != source.stat().st_ino
            proof[str(rel)] = digest
            total += len(data)
result = {'selected_episodes':len(entries),'files':len(proof),'bytes':total,
          'independent_copies':True,'sha256_by_file':proof,'free_bytes':shutil.disk_usage(root).free}
(root/'PANEL_VERIFIED.json').write_text(json.dumps(result,sort_keys=True))
print(json.dumps({k:v for k,v in result.items() if k != 'sha256_by_file'}))
