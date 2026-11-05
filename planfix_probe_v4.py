import json,time
from pathlib import Path
r=Path('/home/ubuntu/coopmwm20261105_clean1');w=r/'plan_fix'
print('TIME',time.time(),flush=True)
for name in ('BANK_DISPATCH_v4.json','BANK_RESULTS.json'):
 p=w/name
 if p.exists():
  value=json.loads(p.read_text())
  if name=='BANK_DISPATCH_v4.json':
   s=Path('/proc')/str(value['pid'])/'stat'
   fields=s.read_text().split() if s.exists() else None
   value['alive_same_identity']=bool(fields and fields[21]==value['start_ticks'] and fields[2]!='Z')
  print(name,json.dumps(value),flush=True)
log=w/'BANK_JOB_v4.log'
if log.exists():print('V4_LOG',log.read_text()[-14000:],flush=True)
charges=[json.loads(line) for line in (r/'CHARGES.jsonl').read_text().splitlines()]
print('CHARGED',dict(plan_fix=sum(x['sim_s'] for x in charges if x['phase']=='plan_fix'),
 total_including_parity=sum(x['sim_s'] for x in charges)+960),flush=True)
for p in (w/'bank_verification_v4').glob('*/PLAN_RESULT.json'):
 value=json.loads(p.read_text());value['rows']=[{k:v for k,v in row.items() if k!='example'} for row in value['rows']]
 print('PLAN_RESULT',p.parent.name,json.dumps(value),flush=True)
