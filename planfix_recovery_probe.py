import json
from pathlib import Path
r=Path('/home/ubuntu/coopmwm20261105_clean1');w=r/'plan_fix'
for name in ('STOPPED_BY_OWNER_CLEAN2_RESTART.json','RESUMED_1434.json','STOP.json'):
 p=r/name
 if p.exists():
  value=json.loads(p.read_text());print(name,json.dumps(value),flush=True)
for name in ('bank_verification_v3','bank_verification'):
 p=w/name
 print(name,dict(started=len(list(p.glob('*/STARTED.json'))),results=len(list(p.rglob('RESULT.json*'))),
     plan_results=len(list(p.glob('*/PLAN_RESULT.json')))),flush=True)
print('SSH_ENTRY_NAMES',[p.name for p in Path('/home/ubuntu/.ssh').iterdir()],flush=True)
print('CLEAN2_ROOT_EXISTS',Path('/home/ubuntu/coopmwm20261105_clean2').exists(),flush=True)
