"""Read-only TRAIN progress; --current follows each arm's collection wave."""
import json
import time
from . import ops
from .storage import write
from .warm_deploy import ROOT
from .exact_ops import CPU


def collect(current=False):
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,time,zlib,os
r=pathlib.Path({ROOT!r});sys.path.insert(0,{CPU!r}+'/pytools')
from tools.coopmwm.warm_protocol import batch_worlds
from tools.coopmwm.accel_storage import read
now=time.time();episodes=[];current={current!r}
repair=r/'MAIN0508_REPAIR_BINDING_V2.json'
if not repair.exists():repair=r/'MAIN0508_REPAIR_BINDING.json'
plan=read(repair)['repair_plan'] if repair.exists() else None
recovery_path=next((r/n for n in ('FAST_RESUME_1844.json','RESOURCE_EXTENSION_1651.json','VERIFICATION_CONTINUATION_BINDING.json','MATCH_RECOVERY_BINDING.json','INCIDENT_RECOVERY_BINDING.json') if (r/n).exists()),None)
recovery=read(recovery_path) if recovery_path else None
retired=read(r/'INCIDENT_RETIRED_JOBS.json') if (r/'INCIDENT_RETIRED_JOBS.json').exists() else {{}}
if (r/'DNS_RETIRED_JOBS_1651.json').exists():
 dns=read(r/'DNS_RETIRED_JOBS_1651.json')
 retired={{k:retired.get(k,[])+dns.get(k,[]) for k in ('jobs','query_keys')}}
if (r/'FAST_RETIRED_JOBS_1844.json').exists():
 fast=read(r/'FAST_RETIRED_JOBS_1844.json')
 retired={{k:retired.get(k,[])+fast.get(k,[]) for k in ('jobs','query_keys')}}
requests=[]
for arm,folder in (('B','requests'),('R','centre_requests')):
 for p in (r/folder).glob('*.json'):
  q=read(p)
  if q.get('mode')=='diagnostic':continue
  if p.stem in retired.get('query_keys',[]):continue
  requests.append((arm,p,q))
collections={{}}
for arm in ('B','R'):
 completed=max([0]+[int(p.stem.rsplit('_',1)[1]) for p in r.glob(arm+'_UPDATE_[0-9][0-9][0-9].json')])
 endpoint=recovery['budget']['updates'][arm] if recovery else (48 if arm=='B' else 24)
 u=min(completed+1,endpoint) if current else 1
 collections[arm]=u
 ws=plan['worlds'] if u==1 and plan else batch_worlds(u)
 if recovery and u==recovery['resume'][arm]+1:ws=recovery['plans'][arm]['worlds']
 for w in ws:
    ident=f'warm_u{{u:03}}_{{arm}}_{{w["seed"]}}'
    ident=(recovery or {{}}).get('job_id_overrides',{{}}).get(ident,ident);p=r/'jobs'/ident
    claim=read(p/'CLAIM.json') if (p/'CLAIM.json').exists() else None
    result=(p/'RESULT.json').exists();last=None;trace=p/'observable.jsonl.gz'
    if trace.exists() and not result:
        raw=zlib.decompressobj(16+zlib.MAX_WBITS).decompress(trace.read_bytes())
        lines=raw.split(b'\\n')[:-1]
        if lines:last=json.loads(lines[-1])['time_s']
    qs=[]
    for qa,qp,q in requests:
        if qa!=arm or q['world_id']!=w['seed']:continue
        if claim and qp.stat().st_mtime<(p/'CLAIM.json').stat().st_mtime:continue
        completed=r/'broker_completed'/qp.name;assigned=r/'broker_assignments'/qp.name
        response=r/('responses' if arm=='B' else 'centre_responses')/qp.name
        qs.append(dict(time_s=q['time_s'],key=q['state_id'],assigned=json.loads(assigned.read_text()) if assigned.exists() else None,
            response=response.exists(),age_s=now-qp.stat().st_mtime,
            completion=read(completed) if completed.exists() else None))
    pending=[q for q in qs if not q['response']]
    if pending:last=max([last or 300]+[q['time_s'] for q in pending])
    state=None
    if claim:
        stat=pathlib.Path('/proc')/str(claim['pid'])/'stat'
        if stat.exists():state=stat.read_text().split()[2]
    episodes.append(dict(id=ident,arm=arm,update=u,claimed=bool(claim),closed=result,pid=claim['pid'] if claim else None,
        process_state=state,observed_sim_time_s=1200 if result else last,
        progress_censored=not result,queries=len(qs),pending_queries=pending,
        completed_query_inference_s=sum(q['completion']['inference_wall_s'] for q in qs if q['completion']),
        current_query_wait_s=max([0.]+[q['age_s'] for q in pending]),
        pending_query_count=len(pending),pending_query_age_sum_s=sum(q['age_s'] for q in pending)))
quota=r/f'B_RANDOM_QUOTA_{{collections["R"]:03}}.json'
reference=r/f'R_MATCH_REFERENCE_{{collections["R"]:03}}.json'
if reference.exists():quota=pathlib.Path(read(reference)['path'])
print(json.dumps(dict(unix=now,collections=collections,episodes=episodes,R_quota_ready=quota.exists(),load=os.getloadavg(),
  superseded_jobs=(plan.get('failed_jobs',[]) if plan else [])+retired.get('jobs',[]),
   schedule_source=recovery_path.name if recovery_path else 'registered original wave',
 note='trace has buffered/incomplete tail; pending-query time is exact stopped decision; current_query_wait_s is oldest unresolved request age (max, not sum), includes queue/admission time and is censored until response; pending_query_age_sum_s is aggregate concurrent request age, not episode wall time; completed inference time excludes admission waits')))
''',timeout=60))
    path=ops.EVIDENCE/f'EPISODE_PROGRESS_{int(value["unix"])}.json'
    write(path,value)
    print(json.dumps(dict(receipt=str(path),unix=value['unix'],R_quota_ready=value['R_quota_ready'],
        B=[{k:e[k] for k in ('id','claimed','closed','observed_sim_time_s','current_query_wait_s','pending_query_count','pending_query_age_sum_s')} for e in value['episodes'] if e['arm']=='B'],
        R=[{k:e[k] for k in ('id','claimed','closed','observed_sim_time_s','current_query_wait_s','pending_query_count','pending_query_age_sum_s')} for e in value['episodes'] if e['arm']=='R'])))
    return value


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--current',action='store_true')
    collect(parser.parse_args().current)
