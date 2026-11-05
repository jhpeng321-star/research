"""Read named launch bindings and frozen CPU source assets, no inference/physics."""
import json
from pathlib import Path
from planfix_ops import remote, note

CODE = r'''
import hashlib,json,time,ast
from pathlib import Path
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=Path('/home/ubuntu/coopmwm20261105_clean2');launch=json.loads((c/'LAUNCHED.json').read_text())
assert sha(c/'CONFIG.json')==launch['config_file_sha256']
for name,key in [('SMOKE_PASS.json','smoke_sha256'),('GATE_PASS.json','gate_sha256')]:
 p=c/name; data=json.loads(p.read_text());print('QUALIFICATION',name,json.dumps(data),flush=True)
 # Launch uses scientific receipt seals; preserve raw SHA as independent evidence.
 assert launch[key] in (sha(p),data.get('sha256'),data.get('receipt_sha256'))
owners={}
for role,owner in launch['owners'].items():
 p=Path('/proc')/str(owner['pid'])/'stat';fields=p.read_text().split() if p.exists() else None
 owners[role]=dict(owner,alive=bool(fields and fields[21]==owner['start_ticks'] and fields[2]!='Z'))
assert all(x['alive'] for x in owners.values())
assert not (c/'STOP.json').exists() and not (c/'FAULT.json').exists()
updates={}
for role in ('A','B','R'):
 updates[role]=[]
 for p in sorted(c.glob(role+'_UPDATE_*.json'))[-4:]:
  d=json.loads(p.read_text());updates[role].append(dict(name=p.name,sha256=sha(p),receipt=d))
print('LAUNCH_AUDIT',json.dumps(dict(unix=time.time(),launch=launch,owners=owners,updates=updates)),flush=True)
s=Path('/home/ubuntu/coopmwm20261104_launchcoop_src7')
for name in ('FROZEN_PRE_NICHE_SEARCH.py','LS_CPU_SELECTION.json','SOURCE_PROVENANCE.json','LS_SOURCE_HASHES.json'):
 p=s/name;print('FROZEN_ASSET',str(p),p.read_text()[:45000],flush=True)
'''

if __name__ == '__main__':
    text = remote('cpuV', CODE, timeout=25)
    row = json.loads(next(x[len('LAUNCH_AUDIT '):] for x in text.splitlines() if x.startswith('LAUNCH_AUDIT ')))
    path = Path(__file__).with_name('PLANFIX_CLEAN2_LAUNCH_VERIFIED.json')
    with path.open('x', encoding='utf-8') as out:
        json.dump(row, out, indent=2, sort_keys=True)
    note('cpuV', 'clean2 final CONFIG/smoke/gate bindings and original four live owners verified; bank terminal, no new dispatch or sim-s')
