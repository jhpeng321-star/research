"""Isolated old-reference memory recovery; all local outputs are memory_* files."""
import argparse
import base64
import json
import time
import zlib
from pathlib import Path

from .ls_validity_ops import remote

LOCAL = Path(__file__).resolve().parents[2]
ROOT = '/home/ubuntu/coopmwm20261102_warm1'
BASE = '/home/ubuntu/coopmwm20261103_compositereleasecpu1'
SOURCE = '/home/ubuntu/coopmwm20261103_memory_recovery2'
CONTROL = ROOT + '/memory_recovery1'
NAME = 'memory_RECOVERY_BINDING_V1.json'
EVIDENCE = LOCAL / 'docs/paper_results/coopmwm'


def call(code, label, timeout=40):
    value = json.loads(remote('cpuQ', code, timeout=timeout))
    p = EVIDENCE / f'memory_RECOVERY_{label}_{time.time_ns()}.json'
    # Receipts exclusively belong to this recovery, never an existing receipt.
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, sort_keys=True, indent=2)
    if label in ('INSPECT', 'SOURCE'):
        print(json.dumps(dict(receipt=str(p), **{k: value[k] for k in
            ('latest', 'charged', 'train_charged', 'mem', 'disk_free', 'load') if k in value})))
    elif label == 'STATUS':
        print(json.dumps(dict(receipt=str(p), latest=value['latest'],
            watcher=value['watcher']['owner'], failures=value['failures'],
            episodes={a: dict(update=x['update'], claimed=x['claimed'], completed=x['completed'],
                observed_times=[j['observed_sim_time_s'] for j in x['jobs']])
                for a, x in value['episodes'].items()}, mem=value['mem'],
            children=[dict(pid=d['owner']['pid'], arm=d['owner'].get('arm'),
                component=d['owner'].get('component'), state=d['state'], rss=d['rss'],
                children=len(d['children'])) for d in value['children']],
            train_charged=value['train_charged'], disk_free=value['disk_free'])))
    else:
        print(json.dumps(dict(receipt=str(p), **value)))
    return value


def inspect():
    return call(f'''from pathlib import Path
import json,os,time,sys,hashlib,shutil
r=Path({ROOT!r});s=Path({BASE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.accel_storage import ledger,charged
from tools.coopmwm.warm_health import assert_running
v=json.loads((r/'B_REPLAY_EXPOSURE_V2.json').read_text())
start=json.loads((r/'B_REPLAY_COMPOSITE_STARTED_0957.json').read_text())
old=json.loads((r/'replay_composite_0957/DISPATCH.json').read_text())
def proc(d):
 p=Path('/proc',str(d['pid']));st=(p/'stat').read_text().split()
 assert st[21]==d['start_ticks'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
 children=[]
 for k in (p/'task'/str(d['pid'])/'children').read_text().split():
  q=Path('/proc',k)
  children.append(dict(pid=int(k),argv=(q/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()))
 return dict(owner=d,state=st[2],argv=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(),children=children,wchan=(p/'wchan').read_text(),status=[x for x in (p/'status').read_text().splitlines() if x.startswith(('VmRSS:','VmSize:'))])
entries=ledger(r);latest={{}};jobs={{}};health={{}}
for a in ('B','R'):
 u=max(int(p.stem.rsplit('_',1)[1]) for p in r.glob(a+'_UPDATE_[0-9][0-9][0-9].json'));latest[a]=u
 ids=[x for x in entries if x['id'].startswith('replay0735_warm_u'+str(u+1).zfill(3)+'_'+a+'_')]
 jobs[a]=dict(charges=ids,claims=[str(p.relative_to(r)) for p in (r/'jobs').glob('replay0735_warm_u'+str(u+1).zfill(3)+'_'+a+'_*/CLAIM.json')],checkpoint_sha256=hashlib.sha256((r/(a+'_checkpoint_'+str(u).zfill(3)+'.pt')).read_bytes()).hexdigest(),next_checkpoint=(r/(a+'_checkpoint_'+str(u+1).zfill(3)+'.pt')).exists(),last_start=min(p.stat().st_mtime for p in (r/'jobs').glob('replay0735_warm_u'+str(u).zfill(3)+'_'+a+'_*/CLAIM.json')))
 try:assert_running(r,a);health[a]='OK'
 except Exception as e:health[a]=repr(e)
cpu=json.loads((s/'LS_CPU_RESULT.json').read_text())
hashes=cpu['source_hashes']
for n,h in hashes.items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
print(json.dumps(dict(unix=time.time(),root=str(r),source_root=str(s),cpu={{k:cpu[k] for k in ('pass_','tests','skips','sim_s')}},source_hashes=hashes,old_watcher=proc(old),children=[proc(d) for d in start['children']],latest=latest,jobs=jobs,health=health,previous_binding=v,charged=charged(r),train_charged=sum(x['sim_s'] for x in entries if x['phase']=='train'),mem=[x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemAvailable:','MemTotal:'))],disk_free=shutil.disk_usage(r).free,load=os.getloadavg())))''', 'INSPECT')


def source():
    return call(f'''from pathlib import Path
import json
s=Path({BASE!r})
names=['warm_runner','replay_start','accel_runner','accel_learner','storage','accel_storage','warm_protocol']
print(json.dumps(dict(files={{n:(s/'pytools/tools/coopmwm'/(n+'.py')).read_text() for n in names}},admission=(s/'pytools/tools/coopnative/runner.py').read_text(),tests=json.loads((s/'LS_RELEASE_TESTS.json').read_text()))))''', 'SOURCE')


def qualify():
    files = {n: (LOCAL / 'tools/coopmwm' / (n + '.py')).read_text(encoding='utf-8')
             for n in ('memory_recovery', 'memory_test_cpu')}
    packed = base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
    return call(f'''from pathlib import Path
import os,json,shutil,hashlib,subprocess,time,base64,zlib
s=Path({SOURCE!r});base=Path({BASE!r});assert s.parent.is_dir()
files=json.loads(zlib.decompress(base64.b64decode({packed!r})))
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cpu=json.loads((base/'LS_CPU_RESULT.json').read_text())
assert cpu['pass_'] and cpu['tests']==727 and cpu['skips']==cpu['sim_s']==0
for n,digest in cpu['source_hashes'].items():assert h(base/n)==digest,n
if not s.exists():
 s.mkdir()
 for n in ('pytools','WM_PACKAGE'):
  shutil.copytree(base/n,s/n,ignore=shutil.ignore_patterns('__pycache__','sealed','*holdout*','*heldout*','*cal_split*','*chunk5*','*j4b*','*inc1*'))
 for n in ('FROZEN_COSTS.json','FROZEN_MOTION_SOURCE.py','FROZEN_GEOMETRY_SOURCE.py','FROZEN_NETWORK.net.xml.gz','FROZEN_PRE_NICHE_SEARCH.py'):shutil.copy2(base/n,s/n)
 p=s/'pytools/tools/coopmwm/replay_start.py';text=p.read_text()
 edits=[("NAME = 'B_REPLAY_EXPOSURE_V2.json'", "NAME = {NAME!r}"),
  ("resume_update=v.get('resume_update',3)","resume_update=v.get('resume_updates',{{}}).get(arm,v.get('resume_update',3))"),
  ("precharged_unattempted=[]", "precharged_unattempted=v.get('precharged_unattempted',{{}}).get(arm,[])")]
 for old,new in edits:assert text.count(old)==1,old;text=text.replace(old,new)
 p.write_text(text)
 p=s/'pytools/tools/coopmwm/warm_runner.py';text=p.read_text()
 old='        for u in range(first,last+1):\\n            assert_running(root,arm)'
 new='        for u in range(first,last+1):\\n            if b.get("b_replay") and arm in ("B","R"):\\n                # Drop every rollout alias before pacing/wave/admission.\\n                eps=[];e=s=None;candidates=[];pairs=[];weights=None;state=None\\n                from .memory_recovery import release_heap\\n                release_heap()\\n            assert_running(root,arm)'
 assert text.count(old)==1;text=text.replace(old,new);p.write_text(text)
 for n,text in files.items():(s/'pytools/tools/coopmwm'/(n+'.py')).write_text(text)
 hashes={{n:h(s/n) for n in cpu['source_hashes']}}
 for n in files:hashes['pytools/tools/coopmwm/'+n+'.py']=h(s/'pytools/tools/coopmwm'/(n+'.py'))
 changed={{n:dict(before=digest,after=hashes[n]) for n,digest in cpu['source_hashes'].items() if hashes[n]!=digest}}
 assert set(changed)=={{'pytools/tools/coopmwm/replay_start.py','pytools/tools/coopmwm/warm_runner.py'}},changed
 pins={{}}
 for folder in ('pytools','WM_PACKAGE'):
  for p in (s/folder).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:
    n=str(p.relative_to(s))
    if n not in hashes or n not in changed:
     if (base/n).exists():assert h(p)==h(base/n),n
    pins[n]=h(p)
 with (s/'memory_SOURCE_DELTA.json').open('x') as f:json.dump(dict(base=str(base),base_cpu_tests=727,changed=changed,pinned_files=pins),f)
 with (s/'LS_SOURCE_HASHES.json').open('x') as f:json.dump(hashes,f)
 tests=json.loads((base/'LS_RELEASE_TESTS.json').read_text())+['memory_test_cpu']
 with (s/'LS_RELEASE_TESTS.json').open('x') as f:json.dump(tests,f)
 shutil.copy2(base/'LS_CPU_SELECTION.json',s/'LS_CPU_SELECTION.json')
for n,text in files.items():assert (s/'pytools/tools/coopmwm'/(n+'.py')).read_text()==text,n
if (s/'LS_CPU_RESULT.json').exists():print((s/'LS_CPU_RESULT.json').read_text())
elif (s/'memory_CPU_DISPATCH.json').exists():print((s/'memory_CPU_DISPATCH.json').read_text())
else:
 env=dict(os.environ,PYTHONPATH=str(s/'pytools')+':'+str(s/'WM_PACKAGE')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',COOPMWM_NICHE_BASELINE=str(s/'FROZEN_PRE_NICHE_SEARCH.py'),COOPMWM_NULLFIX_GEOMETRY=str(s/'FROZEN_GEOMETRY_SOURCE.py'),COOPMWM_NULLFIX_NETWORK=str(s/'FROZEN_NETWORK.net.xml.gz'))
 code='from tools.coopmwm.nullfix_geometry import install;install('+repr(str(s/'FROZEN_GEOMETRY_SOURCE.py'))+');from tools.coopmwm.ls_cpu_test import run;run('+repr(str(s))+')'
 with (s/'memory_CPU.log').open('x') as log:
  p=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-u','-c',code],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 d=dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),source=str(s))
 with (s/'memory_CPU_DISPATCH.json').open('x') as f:json.dump(d,f)
 print(json.dumps(d))''', 'QUALIFY', 60)


def cpu_result():
    return call(f'''from pathlib import Path
import json
s=Path({SOURCE!r});p=s/'LS_CPU_RESULT.json'
v=json.loads(p.read_text()) if p.exists() else dict(pending=True)
v={{k:v[k] for k in ('pass_','tests','skips','sim_s','wall_s','root','pending') if k in v}}
if not v.get('pass_'):v['log']=(s/'memory_CPU.log').read_text()[-6500:]
print(json.dumps(v))''', 'CPU')


def register():
    """Register a fully qualified additive binding while old owners still sleep."""
    return call(f'''from pathlib import Path
import json,sys,time,hashlib,os
r=Path({ROOT!r});s=Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm import replay_start as rs
from tools.coopmwm.accel_storage import ledger,charged
from tools.coopmwm.warm_protocol import batch_worlds
from tools.coopmwm.warm_health import assert_running
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):
 if p.exists():assert json.loads(p.read_text())==v
 else:
  with p.open('x') as f:json.dump(v,f,sort_keys=True);f.flush();os.fsync(f.fileno())
assert r.is_dir()
cpu=json.loads((s/'LS_CPU_RESULT.json').read_text());assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
for n,digest in cpu['source_hashes'].items():assert h(s/n)==digest,n
pins=json.loads((s/'memory_SOURCE_DELTA.json').read_text())
for n,digest in pins['pinned_files'].items():assert h(s/n)==digest,n
target=r/{NAME!r}
if target.exists():
 print(json.dumps(dict(already_registered=True,binding_sha256=h(target),registration=json.loads((r/'memory_REGISTERED.json').read_text()))));raise SystemExit
old=json.loads((r/'B_REPLAY_EXPOSURE_V2.json').read_text())
prior=json.loads((r/'B_REPLAY_COMPOSITE_STARTED_0957.json').read_text())
watcher=json.loads((r/'replay_composite_0957/DISPATCH.json').read_text())
from tools.coopmwm.memory_recovery import verify_identity
watcher=dict(watcher,run_root=str(r));verify_identity(watcher,'watch')
arms={{}};graders={{}};resumes={{}};checkpoints={{}};precharged={{}};not_before={{}};identities=[];restore={{}}
entries=ledger(r);ids={{x['id']:x for x in entries}}
assert len(ids)==len(entries)
for a in ('B','R'):
 assert_running(r,a)
 arm=dict(next(d for d in prior['children'] if d['component']=='arm' and d['arm']==a),run_root=str(r))
 grader=dict(next(d for d in prior['children'] if d['component']=='grade' and d['arm']==a),run_root=str(r),completion_root=old['control_root'])
 p=verify_identity(arm,'arm',a);verify_identity(grader,'grade',a)
 assert (p/'wchan').read_text().strip()=='do_select'
 children=[]
 for k in (p/'task'/str(arm['pid'])/'children').read_text().split():
  q=Path('/proc',k);argv=(q/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
  assert 'multiprocessing.resource_tracker' in argv,k
  children.append(dict(pid=int(k),start_ticks=(q/'stat').read_text().split()[21],argv=argv))
 u=max(int(p.stem.rsplit('_',1)[1]) for p in r.glob(a+'_UPDATE_[0-9][0-9][0-9].json'))
 assert u<24 and not (r/(a+'_checkpoint_'+str(u+1).zfill(3)+'.pt')).exists()
 expected={{'replay0735_warm_u'+str(u+1).zfill(3)+'_'+a+'_'+str(w['seed']) for w in batch_worlds(u+1)}}
 charged_ids={{x['id'] for x in entries if x['id'].startswith('replay0735_warm_u'+str(u+1).zfill(3)+'_'+a+'_')}}
 assert charged_ids==expected,(a,charged_ids,expected)
 for key in expected:
  assert ids[key]==dict(id=key,phase='train',sim_s=1200),key
  assert not (r/'jobs'/key/'CLAIM.json').exists() and not (r/'jobs'/key/'RESULT.json').exists(),key
 resumes[a]=u;checkpoints[a]=h(r/(a+'_checkpoint_'+str(u).zfill(3)+'.pt'));precharged[a]=sorted(expected)
 import torch
 from tools.coopmwm.accel_learner import load
 torch.set_num_threads(1)
 policy,adam,generator,payload=load(r/(a+'_checkpoint_'+str(u).zfill(3)+'.pt'))
 assert payload['update']==u
 assert all(torch.equal(value,policy.state_dict()[key]) for key,value in payload['policy'].items())
 restored=adam.state_dict();assert restored['param_groups']==payload['optimizer']['param_groups']
 for key,values in payload['optimizer']['state'].items():
  for field,value in values.items():assert torch.equal(value,restored['state'][key][field])
 assert torch.equal(generator.get_state(),payload['generator'])
 restore[a]=dict(update=u,policy_bitwise=True,adam_bitwise=True,torch_generator_bitwise=True,parameters=len(payload['policy']),adam_states=len(restored['state']),generator_sha256=hashlib.sha256(bytes(generator.get_state().tolist())).hexdigest())
 del policy,adam,generator,payload,restored
 not_before[a]=min(p.stat().st_mtime for p in (r/'jobs').glob('replay0735_warm_u'+str(u).zfill(3)+'_'+a+'_*/CLAIM.json'))+old['pace_s']
 arms[a]=arm;graders[a]=grader;identities.append(dict(arm=a,resource_trackers=children))
fresh=sum((24-u)*24*1200-len(precharged[a])*1200 for a,u in resumes.items())
train_used=sum(x['sim_s'] for x in entries if x['phase']=='train')
assert fresh>=0 and old['budget']['phases']['train']-train_used>=fresh
line='MEMORY_RECOVERY MAIN19:45/20:05 owner urgent engineering correction. Frozen C1 legacy reference only; release previous rollout aliases plus gc/malloc_trim before next admission. No numerical source change, legacy bank_imitation.run, no bank_statistics install. B/R exact checkpoint Adam/torch Generator continuation to u24, pacing1800, R lag1, fairness, safety and admission8GiB unchanged. Existing shared warm1 ledger; adopt only 24 no-CLAIM precharges per arm; no new allocation; global registered17471840<=17500000. Retire identity-verified idle watcher/arms only; preserve/adopt graders and original broker/GPU/collection services. New V2V remains HELD. Registration before fresh SUMO.'
v=dict(old,source_root=str(s),control_root={CONTROL!r},resume_updates=resumes,checkpoints=checkpoints,precharged_unattempted=precharged,next_not_before_s=not_before,runtime_receipt_suffix='_MEMORY_RECOVERY1',start_receipt_name='memory_RECOVERY_STARTED.json',adopted_graders=graders,old_arms=arms,old_watcher=watcher,cpu=cpu,memory_recovery_line=line,memory_source_delta_sha256=h(s/'memory_SOURCE_DELTA.json'),memory_quote=dict(fresh_train_sim_s=fresh,adopted_precharged_sim_s=sum(len(x) for x in precharged.values())*1200,new_allocation_sim_s=0,train_charged=train_used,remaining_train=old['budget']['phases']['train']-train_used,global_registered=17471840),memory_old_identities=identities,memory_checkpoint_restore=restore)
v.pop('resume_update',None)
put(target,v)
reg=dict(unix=time.time(),binding_sha256=h(target),binding_name={NAME!r},source_root=str(s),source_delta_sha256=h(s/'memory_SOURCE_DELTA.json'),line=line,old_authority_sha256=h(r/'AUTHORITY.md'),old_binding_sha256=h(r/'B_REPLAY_EXPOSURE_V2.json'),quote=v['memory_quote'])
put(r/'memory_REGISTERED.json',reg)
with (r/'memory_AUTHORITY.md').open('x') as f:f.write(line+'\\nBindingSHA256 '+h(target)+'\\nOriginal authority and bindings preserved.\\n');f.flush();os.fsync(f.fileno())
print(json.dumps(dict(registered=True,binding_sha256=h(target),registration=reg,resumes=resumes,checkpoints=checkpoints,checkpoint_restore=restore,old_watcher=watcher,old_arms=arms,adopted_graders=graders,cpu={{k:cpu[k] for k in ('pass_','tests','skips','sim_s')}},source_delta=pins['changed'])))''', 'REGISTER', 60)


def launch():
    """Freeze metadata owners, recheck every boundary, then retire and detach once."""
    return call(f'''from pathlib import Path
import json,sys,time,hashlib,os,signal,subprocess,fcntl
r=Path({ROOT!r});s=Path({SOURCE!r});c=Path({CONTROL!r});assert c.parent.is_dir();c.mkdir(exist_ok=True)
lock=(c/'memory_DISPATCH.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm import replay_start as rs
from tools.coopmwm.memory_recovery import verify_identity
from tools.coopmwm.accel_storage import ledger
from tools.coopmwm.warm_health import assert_running
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v=rs.config(r);reg=json.loads((r/'memory_REGISTERED.json').read_text())
assert reg['binding_sha256']==h(r/{NAME!r}) and reg['binding_sha256'] in (r/'memory_AUTHORITY.md').read_text()
assert reg['old_authority_sha256']==h(r/'AUTHORITY.md') and reg['old_binding_sha256']==h(r/'B_REPLAY_EXPOSURE_V2.json')
for n,digest in json.loads((s/'memory_SOURCE_DELTA.json').read_text())['pinned_files'].items():assert h(s/n)==digest,n
if (c/'memory_DISPATCH.json').exists():
 d=json.loads((c/'memory_DISPATCH.json').read_text());assert rs.alive(d);print(json.dumps(dict(adopted=True,owner=d)));raise SystemExit
old=[v['old_watcher']]+[v['old_arms'][a] for a in ('B','R')]
expected_children={{d['pid'] for d in old[1:]}}|{{d['pid'] for d in v['adopted_graders'].values()}}
frozen=[];retired=[]
try:
 for d in old:
  component='arm' if d.get('component')=='arm' else 'watch'
  p=verify_identity(d,component,d.get('arm'))
  assert (p/'stat').read_text().split()[2]=='S' and (p/'wchan').read_text().strip()=='do_select','not idle'
  for k in (p/'task'/str(d['pid'])/'children').read_text().split():
   q=Path('/proc',k);st=(q/'stat').read_text().split();argv=(q/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
   if d==old[0]:assert int(k) in expected_children or st[2]=='Z',(k,argv)
   else:assert 'multiprocessing.resource_tracker' in argv or st[2]=='Z',(k,argv)
  os.kill(d['pid'],signal.SIGSTOP);frozen.append(d)
  for _ in range(100):
   if (p/'stat').read_text().split()[2]=='T':break
   time.sleep(.01)
  else:raise ValueError('metadata freeze not acknowledged')
 # Watcher is frozen first, so it cannot mistake intentional retirement for a fault.
 for d in old:
  p=verify_identity(d,'arm' if d.get('component')=='arm' else 'watch',d.get('arm'))
  for k in (p/'task'/str(d['pid'])/'children').read_text().split():
   q=Path('/proc',k);st=(q/'stat').read_text().split();argv=(q/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
   if d==old[0]:assert int(k) in expected_children or st[2]=='Z',(k,argv)
   else:assert 'multiprocessing.resource_tracker' in argv or st[2]=='Z',(k,argv)
 entries={{x['id']:x for x in ledger(r)}}
 for a in ('B','R'):
  assert_running(r,a);verify_identity(v['adopted_graders'][a],'grade',a)
  u=v['resume_updates'][a]
  assert max(int(p.stem.rsplit('_',1)[1]) for p in r.glob(a+'_UPDATE_[0-9][0-9][0-9].json'))==u,'latest advanced; register fresh binding'
  assert h(r/(a+'_checkpoint_'+str(u).zfill(3)+'.pt'))==v['checkpoints'][a]
  assert not (r/(a+'_checkpoint_'+str(u+1).zfill(3)+'.pt')).exists()
  for key in v['precharged_unattempted'][a]:
   assert entries[key]==dict(id=key,phase='train',sim_s=1200)
   assert not (r/'jobs'/key/'CLAIM.json').exists() and not (r/'jobs'/key/'RESULT.json').exists(),key
 # All validations precede the first TERM. Graders are never signalled.
 for d in old:
  os.kill(d['pid'],signal.SIGTERM);os.kill(d['pid'],signal.SIGCONT)
  for _ in range(100):
   if not rs.alive(d):break
   time.sleep(.05)
  else:raise RuntimeError('verified idle owner did not exit')
  retired.append(d)
finally:
 for d in frozen:
  if rs.alive(d):os.kill(d['pid'],signal.SIGCONT)
with (c/'memory_RETIRED.json').open('x') as f:json.dump(dict(unix=time.time(),owners=retired,graders_untouched=v['adopted_graders'],no_physics_boundary=True),f);f.flush();os.fsync(f.fileno())
env=dict(os.environ,PYTHONPATH=str(s/'pytools')+':'+str(s/'WM_PACKAGE')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo')
env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH'];assert Path(env['SUMO_HOME'],'bin/sumo').is_file()
with (c/'memory_WATCH.log').open('x') as log:
 p=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.memory_recovery',str(r)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(rs.owner(p.pid),source_root=str(s),control_root=str(c),binding_sha256=reg['binding_sha256'])
with (c/'memory_DISPATCH.json').open('x') as f:json.dump(d,f);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(owner=d,retired=retired,adopted_graders=v['adopted_graders'],resumes=v['resume_updates'],quote=v['memory_quote'])))''', 'LAUNCH', 60)


def status():
    return call(f'''from pathlib import Path
import json,sys,time,os,shutil,zlib,hashlib
r=Path({ROOT!r});s=Path({SOURCE!r});c=Path({CONTROL!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm import replay_start as rs
from tools.coopmwm.accel_storage import ledger,charged
def read(p):return json.loads(p.read_text())
def proc(d):
 p=Path('/proc',str(d['pid']));st=(p/'stat').read_text().split() if p.exists() else None
 assert st is None or st[21]==d['start_ticks']
 children=[]
 if st and st[2]!='Z':
  for k in (p/'task'/str(d['pid'])/'children').read_text().split():
   q=Path('/proc',k)
   children.append(dict(pid=int(k),state=(q/'stat').read_text().split()[2],argv=(q/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(),children=(q/'task'/k/'children').read_text().split()))
 return dict(owner=d,alive=rs.alive(d),state=st[2] if st else 'ABSENT',children=children,rss=[x for x in (p/'status').read_text().splitlines() if x.startswith('VmRSS:')] if st else [])
v=read(r/{NAME!r});d=read(c/'memory_DISPATCH.json')
started=r/'memory_RECOVERY_STARTED.json';children=read(started)['children'] if started.exists() else []
episodes={{}};latest={{}}
for a in ('B','R'):
 latest[a]=max(int(p.stem.rsplit('_',1)[1]) for p in r.glob(a+'_UPDATE_[0-9][0-9][0-9].json'))
 jobs=[]
 u=v['resume_updates'][a]+1
 for p in sorted((r/'jobs').glob('replay0735_warm_u'+str(u).zfill(3)+'_'+a+'_*')):
  result=(p/'RESULT.json').exists();claim=(p/'CLAIM.json').exists();trace=p/'observable.jsonl.gz';last=None
  if trace.exists() and not result:
   decompressor=zlib.decompressobj(16+zlib.MAX_WBITS);tail=b''
   with trace.open('rb') as f:
    while True:
     chunk=f.read(65536)
     if not chunk:break
     lines=(tail+decompressor.decompress(chunk)).split(b'\\n');tail=lines.pop()
     if lines:
      try:last=json.loads(lines[-1])['time_s']
      except (KeyError,ValueError):pass
  jobs.append(dict(id=p.name,claim=claim,result=result,observed_sim_time_s=1200 if result else last))
 episodes[a]=dict(update=u,jobs=jobs,claimed=sum(x['claim'] for x in jobs),completed=sum(x['result'] for x in jobs))
failures={{str(p.relative_to(r)):read(p) for p in list(c.glob('*FAILED.json'))+list(r.glob('WARM_*_FAILED_MEMORY_RECOVERY1.json'))}}
numeric=['accel_learner','acceleration','accel_plant','exact_grade','bank_imitation','replay_bank','mppi_search']
hashes={{n:hashlib.sha256((s/'pytools/tools/coopmwm'/(n+'.py')).read_bytes()).hexdigest() for n in numeric}}
for n,digest in hashes.items():assert hashlib.sha256((Path({BASE!r})/'pytools/tools/coopmwm'/(n+'.py')).read_bytes()).hexdigest()==digest,n
entries=ledger(r)
print(json.dumps(dict(unix=time.time(),root=str(r),source_root=str(s),binding_sha256=hashlib.sha256((r/{NAME!r}).read_bytes()).hexdigest(),watcher=proc(d),started=started.exists(),status=read(c/'memory_STATUS.json') if (c/'memory_STATUS.json').exists() else None,children=[proc(x) for x in children],retired=[proc(x) for x in v['old_arms'].values()]+[proc(v['old_watcher'])],latest=latest,episodes=episodes,failures=failures,watch_log=(c/'memory_WATCH.log').read_text()[-3500:],arm_logs={{a:(c/('arm_'+a+'.log')).read_text()[-3500:] if (c/('arm_'+a+'.log')).exists() else None for a in ('B','R')}},numerical_hashes=hashes,charged_sim_s=charged(r),train_charged=sum(x['sim_s'] for x in entries if x['phase']=='train'),mem=[x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemAvailable:','MemTotal:'))],disk_free=shutil.disk_usage(r).free,load=os.getloadavg())))''', 'STATUS', 60)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('inspect', 'source', 'qualify', 'cpu_result', 'register', 'launch', 'status'))
    globals()[p.parse_args().action]()
