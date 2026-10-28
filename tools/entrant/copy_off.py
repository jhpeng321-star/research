"""Carry unchanged OFF trajectories into the corrected observation-history epoch."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
from remote import ROOT, ssh

host = sys.argv[1]
ssh(host, '''
import hashlib,json,shutil
from pathlib import Path
p=Path.home()/'entrant';out=p/'results_v4';out.mkdir(exist_ok=True)
copied={}
for f in (p/'results_v3').glob('*_s600.json'):
    d=json.loads(f.read_text())
    if d['variant'] not in ('baseline','ramp_adapt_off'):continue
    assert d['entrant_control'] is False
    target=out/f.name
    if target.exists():assert target.read_bytes()==f.read_bytes()
    else:shutil.copy2(f,target)
    copied[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
(p/'V4_OFF_CARRY.json').write_text(json.dumps({'unchanged_off_files':len(copied),'sha256':copied},sort_keys=True))
print(json.dumps({'unchanged_off_files':len(copied)}))
''')
old = ROOT/'evidence_entrant/raw'/host/'results_v3'
new = old.with_name('results_v4')
new.mkdir(exist_ok=True)
copied = 0
for path in old.glob('*_s600.json'):
    doc = json.loads(path.read_text())
    if doc['variant'] not in ('baseline', 'ramp_adapt_off'):
        continue
    target = new/path.name
    if target.exists():
        assert target.read_bytes() == path.read_bytes()
    else:
        shutil.copy2(path, target)
        copied += 1
print(json.dumps({'local_off_files_copied': copied}))
