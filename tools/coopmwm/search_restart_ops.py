"""Read-only current search admission/claim diagnostics before any restart."""
import json
import time
from . import ops
from .ls_validity_ops import remote
from .storage import write,read
from .protocol import checked,sha,REGISTRY

CONTROL='/home/ubuntu/coopmwm20261103_search1829_v2'
SOURCE='/home/ubuntu/coopmwm20261103_compositereleasecpu1'


def audit():
    value=json.loads(remote('cpuQ','''from pathlib import Path
import json,time
from collections import defaultdict
r=Path('/home/ubuntu/coopmwm20261102_warm1');c=Path('/home/ubuntu/coopmwm20261102_lscollect1')
out=dict(unix=time.time(),arms={})
failures=[json.loads(p.read_text()) for p in (c/'broker_failures').glob('*.json')]
for arm in ('B','R'):
 bank=r/'replay_bank'/arm;pending=[];answered=[]
 for p in (bank/'snapshots').glob('*.json'):
  meta=json.loads(p.read_text());key=p.stem
  request=r/('requests' if arm=='B' else 'centre_requests')/p.name
  if not request.exists():continue
  response=r/('responses' if arm=='B' else 'centre_responses')/p.name
  a=c/'broker_assignments'/p.name;complete=c/'broker_completed'/p.name
  row=dict(key=key,update=meta.get('policy_update'),world=meta['world']['seed'],t0=meta['t0'],members=len(meta['group']['members']),age_s=time.time()-request.stat().st_mtime,
   assigned=json.loads(a.read_text()) if a.exists() else None,
   completed=json.loads(complete.read_text()) if complete.exists() else None,
   failures=[x for x in failures if x.get('key')==key])
  if response.exists():row['response_unix']=response.stat().st_mtime;answered.append(row)
  else:pending.append(row)
 out['arms'][arm]=dict(pending=pending,answered_count=len(answered),latest_answers=sorted(answered,key=lambda x:x['response_unix'])[-4:])
out['endpoints']=json.loads((c/'LS_ENDPOINTS.json').read_text())
out['broker_failures_latest']=sorted(failures,key=lambda x:x.get('unix',0))[-6:]
budget_path=r/'B_REPLAY_BINDING.json'
if not budget_path.exists():budget_path=r/'V2V_CAP_1406.json'
fund=json.loads(budget_path.read_text());spent=defaultdict(int)
assert fund['amendment_sha256'] in (r/'AUTHORITY.md').read_text(),'budget authority absent'
out['budget_binding_path']=str(budget_path)
out['budget_binding_sha256']=__import__('hashlib').sha256(budget_path.read_bytes()).hexdigest()
for line in (r/'CHARGES.jsonl').read_text().splitlines():
 x=json.loads(line);spent[x['phase']]+=x['sim_s']
out['phase_room']={k:dict(allocated=v,charged=spent[k],remaining=v-spent[k]) for k,v in fund['budget']['phases'].items()}
# Superseded pre-acceleration charges live outside this ledger (accel_storage.charged).
import sys
sys.path.insert(0,'/home/ubuntu/coopmwm20261104_launchcoop_src3/pytools')
from tools.coopmwm.accel_protocol import SUPERSEDED_CHARGE
out['retained_superseded_sim_s']=SUPERSEDED_CHARGE
if 'superseded' in out['phase_room']:
 row=out['phase_room']['superseded'];row['charged']+=SUPERSEDED_CHARGE;row['remaining']-=SUPERSEDED_CHARGE
out['charged_including_superseded']=sum(spent.values())+SUPERSEDED_CHARGE
out['registered_cap_sim_s']=fund['budget']['total']
print(json.dumps(out))''',timeout=40))
    p=ops.EVIDENCE/f'SEARCH_CURRENT_AUDIT_{time.time_ns()}.json';write(p,value)
    print(json.dumps(dict(receipt=str(p),unix=value['unix'],arms=value['arms'],phase_room=value['phase_room'],
        budget_binding_path=value['budget_binding_path'],budget_binding_sha256=value['budget_binding_sha256'],
        charged_including_superseded=value['charged_including_superseded'],registered_cap_sim_s=value['registered_cap_sim_s'],
        workers=[dict(slot=w['endpoint']['slot'],owner={k:v for k,v in w['owner'].items() if k in ('pid','start_ticks','boot_id','root')}) for w in value['endpoints']['workers']],failures=value['broker_failures_latest'])))


def qualify():
    text=checked(ops.LOCAL/'tools/coopmwm/search_saved_current.py').read_text(encoding='utf-8')
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,subprocess,os,hashlib
r=Path({CONTROL!r});assert r.parent.is_dir();r.mkdir(exist_ok=True);p=r/'search_saved_current.py'
if p.exists():assert p.read_text()=={text!r}
else:
 with p.open('x') as f:f.write({text!r})
v=dict(source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),max_queries=16)
b=r/'CPU_BINDING.json'
if not b.exists():
 with b.open('x') as f:json.dump(v,f)
result=r/'CPU_RESULT.json'
if not result.exists():
 env=dict(os.environ,PYTHONPATH={SOURCE!r}+'/pytools:'+{SOURCE!r}+'/runtime_src:/home/ubuntu/e3final/runtime',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 x=subprocess.run(['/home/ubuntu/miniconda3/bin/python','-B',str(p),'/home/ubuntu/coopmwm20261102_warm1',str(r),'--preflight'],env=env,capture_output=True,text=True,timeout=60)
 assert x.returncode==0,x.stderr
 v.update(preflight=json.loads(x.stdout),pass_=True,new_sim_s=0)
 with result.open('x') as f:json.dump(v,f)
print(result.read_text())''',timeout=80))
    p=ops.EVIDENCE/'SEARCH_RESTART_CPU_1829.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def register():
    cpu=read(ops.EVIDENCE/'SEARCH_RESTART_CPU_1829.json');assert cpu['pass_'] and cpu['preflight']['pass_']
    record=ops.EVIDENCE/'SEARCH_RESTART_REGISTRATION_1829.json'
    line=('\nS-COOPMWM MAIN18:29 search input restart: recover up to16 distinct-world '
        'snapshots from already-charged completed B own-policy episodes u17+ '
        '(honest policy version=u-1, >=16). Zero new episode SUMO; exact bitwise '
        'own-action/previous-noise/RNG regeneration BEFORE publishing, 60 genuine '
        'observable seconds/t>=360 and original full-horizon fitted-phase bound. '
        'Focus worlds first; >=3CAV groups first, inherited downstream order, no '
        'behavioural threshold/outcome-based selection; no duplicate job/time/group '
        'snapshots and no old inference replay. Existing4slot broker/workers '
        'remain adopted; it2_K2/niches1:4 alternation, feasibility/actionable gates, '
        'warm memory/H300/finalK8Hfinal/fullpool logging unchanged. B-only helper '
        'bounded8pending, no SUMO waits; original per-update capture/graders stay. '
        'Quote upperbound16x9x1200=172800 sim-s labels in sumo_verification '
        '(remaining1198230 at admission); confirmations max307200 in existing '
        'bank_verification_ext(remaining719010). No new allocation or cap raise: '
        'shared accounting guards each complete branch set inside17471840. '
        'Source/rule/zero-SUMO real saved-state preflight recorded.\n')
    if not record.exists():write(record,dict(line=line,cpu=cpu,max_queries=16,source_sha256=cpu['source_sha256'],
        new_episode_sim_s=0,label_sim_s_upper=172800,confirmation_sim_s_upper=307200))
    h=sha(record);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+h+'\n')
    print(remote('cpuQ',f'''from pathlib import Path
import json
r=Path({CONTROL!r});p=r/'AUTHORITY.json';v=json.loads({json.dumps(dict(read(record),amendment_sha256=h))!r})
if p.exists():assert json.loads(p.read_text())==v
else:
 with p.open('x') as f:json.dump(v,f)
print(json.dumps(dict(registered=True,sha256={h!r},new_sim_s=0)))''',timeout=35))


def launch():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import os,subprocess,json,time
r=Path({CONTROL!r});assert (r/'AUTHORITY.json').exists();p=r/'DISPATCH.json'
if p.exists():print(p.read_text());raise SystemExit
assert not (r/'OWNER.json').exists(),'adopt existing search recovery owner'
env=dict(os.environ,PYTHONPATH={SOURCE!r}+'/pytools:'+{SOURCE!r}+'/runtime_src:/home/ubuntu/e3final/runtime',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
command=['/home/ubuntu/miniconda3/bin/python','-B','-u',str(r/'search_saved_current.py'),'/home/ubuntu/coopmwm20261102_warm1',str(r)]
with (r/'RUN.log').open('x') as log:
 child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
v=dict(pid=child.pid,start_ticks=Path('/proc',str(child.pid),'stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),argv=command,new_episode_sim_s=0)
with p.open('x') as f:json.dump(v,f)
print(json.dumps(v))''',timeout=40))
    p=ops.EVIDENCE/'SEARCH_RESTART_DISPATCH_1829.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def status():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time,shutil
r=Path({CONTROL!r});root=Path('/home/ubuntu/coopmwm20261102_warm1');out=dict(unix=time.time(),cpu_free=shutil.disk_usage(r).free)
for n in ('DISPATCH','OWNER','COMPLETE'):
 p=r/(n+'.json');out[n]=json.loads(p.read_text()) if p.exists() else None
progress=r/'PROGRESS.jsonl';rows=[json.loads(s) for s in progress.read_text().splitlines()] if progress.exists() else []
for row in rows:
 key=row['key'];answer=root/'responses'/(key+'.json');label=root/'replay_bank/B/decisions'/(key+'.json')
 row['answered_unix']=answer.stat().st_mtime if answer.exists() else None
 row['labelled_unix']=label.stat().st_mtime if label.exists() else None
out['progress']=rows;out['log']=(r/'RUN.log').read_text()[-4000:] if (r/'RUN.log').exists() else None
d=out['DISPATCH'];p=Path('/proc',str(d['pid'])) if d else None
out['alive']=bool(p and p.exists() and (p/'stat').read_text().split()[21]==d['start_ticks'] and (p/'stat').read_text().split()[2]!='Z')
print(json.dumps(out))''',timeout=35))
    write(ops.EVIDENCE/f'SEARCH_RESTART_STATUS_{time.time_ns()}.json',value)
    print(json.dumps(value))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=('audit','qualify','register','launch','status'))
    globals()[p.parse_args().action]()
