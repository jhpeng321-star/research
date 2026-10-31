"""Small read-only receipt/status query; no dispatch, deletion or signalling."""
from remote import ssh

SOURCE=r'''
from pathlib import Path
import json,os
r=Path('/home/ubuntu/sumocore20261031')
done=[];failed=[];active=[]
for p in (r/'jobs').iterdir():
 if (p/'result.json').exists():done.append(json.loads((p/'result.json').read_text()))
 elif (p/'failure.json').exists():failed.append(json.loads((p/'failure.json').read_text()))
 else:
  s=json.loads((p/'START.json').read_text());active.append(dict(id=p.name,pid=s['pid'],live=Path('/proc',str(s['pid'])).exists()))
rows=[json.loads(s) for s in (r/'ledger.jsonl').read_text().splitlines()]
print(json.dumps(dict(load=os.getloadavg(),complete=len(done),failed=len(failed),
 actual_sim_s_completed=sum(d['actual_sim_s'] for d in done)+sum(d['actual_sim_s'] for d in failed),
 precharged=sum(s['charge'] for s in rows),active=active,failures=[d['job']['id'] for d in failed])))
if (r/'DISPATCH_full.json').exists():
 d=json.loads((r/'DISPATCH_full.json').read_text());d['live']=Path('/proc',str(d['pid'])).exists();print(json.dumps(d))
'''
if __name__=='__main__':print(ssh(SOURCE).decode())
