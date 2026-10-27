"""Retire only this task's incomplete copy and use immutable same-filesystem links."""
import hashlib
import json
import os
import shutil
from pathlib import Path

dest = Path.home()/'longhzn'
assert dest.is_dir() and not (dest/'STAGED.json').exists()
panel = dest/'pairs300'
assert panel.parent == dest and not panel.is_symlink()
shutil.rmtree(panel)
source = Path.home()/'cqr/pairs300'
bad = ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vtype')
for parent, dirs, names in os.walk(source):
    assert not any(t in str(parent).lower() for t in bad)
    dirs[:] = [d for d in dirs if not any(t in d.lower() for t in bad)]
    for name in names:
        if any(t in name.lower() for t in bad):
            continue
        p = Path(parent)/name
        target = panel/p.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Read-only inference; never modify these input bytes.
        os.link(p, target)
sha = hashlib.sha256((panel/'C4_EPISODES_MANIFEST.json').read_bytes()).hexdigest()
assert sha == 'd64fc0f2088531dc09da8dfed09ea75f4dab412024ba8c4942bc01949ebea19a'
proof = {'panel_manifest_sha256': sha, 'panel_source': str(source),
         'panel_storage': 'immutable hard-linked task-local mirror; incomplete byte-copy removed',
         'free_bytes': shutil.disk_usage(dest).free}
(dest/'STAGED.json').write_text(json.dumps(proof, indent=1))
print(json.dumps(proof))
