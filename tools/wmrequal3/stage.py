"""Freeze allowed original sources plus the explicitly named scorer repair."""
import base64
import json
import subprocess
from common import ROOT, OWN, OLD, GUARD, remote

NAMES = ('src/wms/control/entrant.py', 'tools/wmrequal/score.py',
         'tools/wmrequal/native_lanes.py', 'tools/wmrequal3/run.py',
         'tools/wmrequal3/cpu.py')

if __name__ == '__main__':
    payload = {n: base64.b64encode((ROOT/n).read_bytes().replace(b'\r\n',b'\n')).decode() for n in NAMES}
    sha = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote(GUARD + f'''
import base64,hashlib,json
from pathlib import Path
p=Path({OWN!r});old=Path({OLD!r})
assert p.parent.is_dir()
p.mkdir(exist_ok=True)
assert not (p/'LAUNCH.json').exists(),'running code is immutable'
source=json.loads((old/'SOURCE.json').read_text())
hashes={{}}
def put(name,raw,replace=False):
 q=p/name;q.parent.mkdir(parents=True,exist_ok=True)
 if q.exists() and not replace:assert q.read_bytes()==raw,name
 else:q.write_bytes(raw)
 hashes[name]=hashlib.sha256(raw).hexdigest()
overrides={payload!r}
for name,expected in source['sha256'].items():
 if name in overrides:continue
 raw=(old/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected,name
 put(name,raw)
for name in ('tools/wmrequal/grouped_feedback.py','REAL.json'):
 put(name,(old/name).read_bytes())
for name,raw in overrides.items():put(name,base64.b64decode(raw),replace=True)
receipt=dict(base={sha!r},original_source_sha256=hashlib.sha256((old/'SOURCE.json').read_bytes()).hexdigest(),sha256=hashes)
q=p/'SOURCE.json'
q.write_text(json.dumps(receipt,indent=1))
print('STAGED',len(hashes),str(p))
''')
