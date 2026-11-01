"""Read-only first-update TRAIN progress; incomplete gzip tails are censored."""
import json
import time
from . import ops
from .storage import write
from .warm_deploy import ROOT
from .exact_ops import CPU


def collect():
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,time,zlib,os
r=pathlib.Path({ROOT!r});sys.path.insert(0,{CPU!r}+'/pytools')
from tools.coopmwm.warm_protocol import batch_worlds
from tools.coopmwm.accel_storage import read
now=time.time();episodes=[]
repair=r/'MAIN0508_REPAIR_BINDING.json'
plan=read(repair)['repair_plan'] if repair.exists() else None
ws=plan['worlds'] if plan else batch_worlds(1)
requests=[]
for p in (r/'requests').glob('*.json'):
 q=read(p)
 if q.get('mode')=='diagnostic':continue
 requests.append((p,q))
for arm in ('B','R'):
 for w in ws:
  ident=f'warm_u001_{{arm}}_{{w["seed"]}}';p=r/'jobs'/ident
  claim=read(p/'CLAIM.json') if (p/'CLAIM.json').exists() else None
  result=(p/'RESULT.json').exists();last=None;trace=p/'observable.jsonl.gz'
  if trace.exists():
   raw=zlib.decompressobj(16+zlib.MAX_WBITS).decompress(trace.read_bytes())
   lines=raw.split(b'\\n')[:-1]
   if lines:last=json.loads(lines[-1])['time_s']
  qs=[]
  for qp,q in requests:
   if arm!='B' or q['world_id']!=w['seed']:continue
   completed=r/'broker_completed'/qp.name;assigned=r/'broker_assignments'/qp.name
   response=r/'responses'/qp.name
   qs.append(dict(time_s=q['time_s'],key=q['state_id'],assigned=json.loads(assigned.read_text()) if assigned.exists() else None,
    response=response.exists(),age_s=now-qp.stat().st_mtime,
    completion=read(completed) if completed.exists() else None))
  pending=[q for q in qs if not q['response']]
  if pending:last=max([last or 300]+[q['time_s'] for q in pending])
  state=None
  if claim:
   stat=pathlib.Path('/proc')/str(claim['pid'])/'stat'
   if stat.exists():state=stat.read_text().split()[2]
  episodes.append(dict(id=ident,arm=arm,claimed=bool(claim),closed=result,pid=claim['pid'] if claim else None,
   process_state=state,observed_sim_time_s=1200 if result else last,
   progress_censored=not result,queries=len(qs),pending_queries=pending,
   completed_query_inference_s=sum(q['completion']['inference_wall_s'] for q in qs if q['completion']),
   current_query_wait_s=sum(q['age_s'] for q in pending)))
quota=r/'B_RANDOM_QUOTA_001.json'
print(json.dumps(dict(unix=now,episodes=episodes,R_quota_ready=quota.exists(),load=os.getloadavg(),
 superseded_jobs=plan['failed_jobs'] if plan else [],
 note='trace has buffered/incomplete tail; pending-query time is exact stopped decision; completed inference time excludes admission waits')))
''',timeout=60))
    path=ops.EVIDENCE/f'EPISODE_PROGRESS_{int(value["unix"])}.json'
    write(path,value)
    print(json.dumps(dict(receipt=str(path),unix=value['unix'],R_quota_ready=value['R_quota_ready'],
        B=[{k:e[k] for k in ('id','claimed','closed','observed_sim_time_s','current_query_wait_s')} for e in value['episodes'] if e['arm']=='B'],
        R_claimed=sum(e['claimed'] for e in value['episodes'] if e['arm']=='R'))))
    return value


if __name__=='__main__':collect()
