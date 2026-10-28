"""Compare completed full-horizon OFF receipts; do not repeat the rollout."""
import sys
from remote import ssh

ssh(sys.argv[1], '''
import hashlib,json
from pathlib import Path
p=Path.home()/'entrant';name='e3_g01_baseline_k0_s600.json'
a=p/'results'/name;b=p/'results_v2'/name
x=json.loads(a.read_text());y=json.loads(b.read_text())
assert x['arms']==y['arms'],'OFF scientific outputs changed'
assert x.get('repeat_identical') and y.get('repeat_identical')
r={'old_sha256':hashlib.sha256(a.read_bytes()).hexdigest(),
   'new_sha256':hashlib.sha256(b.read_bytes()).hexdigest(),
   'steps':600,'arms_identical':True,'both_repeat_identical':True,
   'contract':'full delay horizons, timelines, breakdown and all speed/class bins'}
(p/'OFF_BIT_IDENTITY.json').write_text(json.dumps(r,indent=1));print(json.dumps(r))
''')
