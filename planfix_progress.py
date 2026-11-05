import hashlib,json,sys
from pathlib import Path
from planfix_ops import remote,note
code='''import json,time
from pathlib import Path
r=Path('/home/ubuntu/coopmwm20261105_clean1');w=r/'plan_fix'
p=w/'BANK_DISPATCH_v5.json'
if p.exists():
 receipt=json.loads(p.read_text());s=Path('/proc')/str(receipt['pid'])/'stat'
 fields=s.read_text().split() if s.exists() else None
 receipt['alive_same_identity']=bool(fields and fields[21]==receipt['start_ticks'] and fields[2]!='Z')
 print('BANK_V5',json.dumps(receipt),flush=True)
log=w/'BANK_JOB_v5.log'
if log.exists():print('BANK_V5_LOG',log.read_text()[-10000:],flush=True)
for p in (w/'bank_verification_v5').glob('*/PLAN_RESULT.json'):
 result=json.loads(p.read_text());result['rows']=[{k:v for k,v in row.items() if k!='example'} for row in result['rows']]
 print('PLAN_RESULT',p.parent.name,json.dumps(result),flush=True)
charges=[json.loads(line) for line in (r/'CHARGES.jsonl').read_text().splitlines()]
print('LEDGER',json.dumps(dict(plan_fix=sum(row['sim_s'] for row in charges if row['phase']=='plan_fix'),
 cumulative_including_parity=sum(row['sim_s'] for row in charges)+960)),flush=True)
if (w/'BANK_RESULTS.json').exists():
 data=json.loads((w/'BANK_RESULTS.json').read_text());data['results']=[dict(result,rows=[{k:v for k,v in row.items() if k!='example'} for row in result['rows']]) for result in data['results']]
 print('BANK_RESULTS',json.dumps(data),flush=True)
'''
if len(sys.argv)>1 and sys.argv[1]=='6':
 code=code.replace('BANK_DISPATCH_v5.json','BANK_DISPATCH_v6.json').replace('BANK_JOB_v5.log','BANK_JOB_v6.log')
output=remote('cpuV',code,timeout=30)
lines=output.splitlines()
ledger=json.loads(next(line.split(' ',1)[1] for line in lines if line.startswith('LEDGER ')))
plans=[dict(id=line.split(' ',2)[1],**json.loads(line.split(' ',2)[2])) for line in lines if line.startswith('PLAN_RESULT ')]
report=dict(ledger=ledger,plans=plans)
sha=hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest()
run=Path('E:/research-dense/runs/active/C4-V7FT-20261017')
target=run/('PLAN_FIX_BANK_PROGRESS_'+sha[:12]+'.json')
if target.exists():assert json.loads(target.read_text())==report
else:
 with target.open('x',encoding='utf-8') as out:json.dump(report,out,indent=2,sort_keys=True)
kept=[row for plan in plans for row in plan['rows'] if row.get('keep')]
note('cpuV','BANK PROGRESS completed_source_inputs='+str(len(plans))+' kept_seed_examples='+str(len(kept))+' plan_fix_charged='+str(ledger['plan_fix'])+' cumulative_with_parity='+str(ledger['cumulative_including_parity'])+' receipt='+target.name)
final=next((json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('BANK_RESULTS ')),None)
if final is not None:
 target=run/'PLAN_FIX_BANK_RESULTS.json'
 if target.exists():assert json.loads(target.read_text())==final
 else:
  with target.open('x',encoding='utf-8') as out:json.dump(final,out,indent=2,sort_keys=True)
 note('cpuV','BANK FINAL cached locally; verified gains/families retained, feed-in remains prepared-only pending WM variety and MAIN registration')
