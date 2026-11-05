"""Read only named PLANFIX and clean2 receipts; no job discovery or dispatch."""
import json, time
from pathlib import Path
from planfix_ops import remote

CODE = r'''
import json,time
from pathlib import Path
r=Path('/home/ubuntu/coopmwm20261105_clean1'); w=r/'plan_fix'
for name in ('BANK_DISPATCH_v6.json','BANK_RESULTS.json'):
 p=w/name
 if p.exists():
  data=json.loads(p.read_text())
  if name.startswith('BANK_DISPATCH'):
   proc=Path('/proc')/str(data['pid'])/'stat'; fields=proc.read_text().split() if proc.exists() else None
   data['alive_same_identity']=bool(fields and fields[21]==data['start_ticks'] and fields[2]!='Z')
  else:data={k:v for k,v in data.items() if k!='results'}
  print(name,json.dumps(data),flush=True)
rows=[json.loads(x) for x in (r/'CHARGES.jsonl').read_text().splitlines()]
print('LEDGER',json.dumps(dict(plan_fix=sum(x['sim_s'] for x in rows if x['phase']=='plan_fix'),clean1_including_parity=sum(x['sim_s'] for x in rows)+960)))
c=Path('/home/ubuntu/coopmwm20261105_clean2')
print('CLEAN2_EXISTS',c.exists())
if c.exists():
 print('CLEAN2_CONTROL_NAMES',json.dumps([p.name for p in c.iterdir() if p.is_file() and (p.suffix=='.json' or p.name=='AUTHORITY.md')]))
 for name in ('LAUNCHED.json','STOP.json','FAULT.json','CONFIG.json'):
  p=c/name
  if p.exists():print('CLEAN2_'+name,p.read_text()[:24000],flush=True)
print('READ_UNIX',time.time(),flush=True)
previous=w/'cpu_prepare_95441ec4a7c4'
print('CPU_FIRST_STAGE',json.dumps(dict(root=str(previous),exists=previous.exists(),prepared=(previous/'PREPARED.json').exists())),flush=True)
for p in w.glob('cpu_prepare_*'):
 print('CPU_PREPARE_STAGE',str(p),[(n,(p/n).exists()) for n in ('SOURCE_MANIFEST.json','PREPARED.json')],flush=True)
p=w/'planfix_cpu_source_ea559c952ff0.z'
print('CPU_SOURCE_ARCHIVE',json.dumps(dict(exists=p.exists(),bytes=p.stat().st_size if p.exists() else None)),flush=True)
'''

if __name__ == '__main__':
    remote('cpuV', CODE, timeout=25)
