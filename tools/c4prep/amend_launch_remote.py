"""Append MAIN launch authority to loop state before any GPU work."""
import hashlib
import json
from pathlib import Path
import sys

r = Path('/root/autodl-tmp/c4/restart20261026')
out = r / 'arm/loop/c4prep'
# JSON is transported as reviewed source, not recovered from peer paths.
authority = json.loads(sys.argv[1])
filename = 'MAIN_1700.json' if authority['authority'] == 'MAIN2026-10-28T17:00' else 'MAIN_1500.json'
path = out / filename
if path.exists():
    assert json.loads(path.read_text()) == authority
else:
    with path.open('x') as stream:
        json.dump(authority, stream, indent=1)
state_path = r / 'arm/loop/LOOP14_STATE.json'
state = json.loads(state_path.read_text())
amendments = state.setdefault('main_launch_amendments', [])
if authority not in amendments:
    amendments.append(authority)
    temp = state_path.with_suffix('.c4launch.tmp')
    with temp.open('x') as stream:
        json.dump(state, stream, indent=1)
    temp.replace(state_path)
print(json.dumps({'authority': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'loop_amendment_present': authority in amendments, 'gpu_jobs_started': 0}))
