"""MAIN18:44 minimal durable u4 recovery; adopt receipts before every dispatch."""
import argparse
import json
import time
from . import ops
from .storage import read,write

ROOT='/home/ubuntu/coopmwm20261102_warm1'
SOURCE='/home/ubuntu/coopmwm20261102_fast1844cpu1'
OVERLAYS=('quota_admission','parallel_groups','capacity_broker','warm_runner',
          'incident_recovery','resource_config','test_fast_resume_cpu','fast_resume_ops')


def admit(host):
    """Use already-qualified secondary spools; failed monitors did no admission."""
    from .protocol import checked,REGISTRY,sha
    original=read(ops.EVIDENCE/'SECOND_CPU_QUALIFICATION.json')
    assert original['pass_'] and original['tests']==26
    expected_hash=original['staged']['hashes']['pytools/tools/coopmwm/hotadd.py']
    target=ops.EVIDENCE/f'FAST1844_ADMISSION_REGISTRATION_{host}.json'
    if not target.exists():
        line=f'\nS-COOPMWM MAIN1844 {host} SECOND-SLOT ADMISSION REPAIR: existing V2 monitor exited BEFORE admission with missing hotadd import. Adopt existing owned READY spool/eight exact gates; invoke original 26-CPU-qualified gate helper by verified file SHA, same restricted endpoint/expected boot-start-PID and search SHA. No worker restart/inference replay/new physics.\n'
        write(target,dict(line=line,helper_sha256=expected_hash,new_sim_s=0))
        with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+sha(target)+'\n')
    reg=read(target)
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,time,sys,hashlib,importlib.util
r=pathlib.Path({ROOT!r});sys.path.insert(0,'/home/ubuntu/coopmwm20261102_verifybeta13/pytools')
from tools.coopmwm.warm_broker import Remote
from tools.coopmwm.warm_spool import immutable
p=pathlib.Path('/home/ubuntu/coopmwm20261102_hotadd1/pytools/tools/coopmwm/hotadd.py')
assert hashlib.sha256(p.read_bytes()).hexdigest()=={expected_hash!r}
spec=importlib.util.spec_from_file_location('qualified_hotadd',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
d=json.loads((r/('RESOURCE_MONITOR_DISPATCH_'+{host!r}+'_1651_V2.json')).read_text());c=d['config'];tag=c['tag']
proc=pathlib.Path('/proc',str(d['pid']))
assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert not proc.exists() or (proc/'stat').read_text().split()[21]==d['start_ticks'] and (proc/'stat').read_text().split()[2]=='Z'
assert 'ModuleNotFoundError' in (r/('RESOURCE_MONITOR_'+{host!r}+'_1651_V2.log')).read_text()
done=r/('RESOURCE_ADMITTED_'+tag+'.json')
if done.exists():
 print(json.dumps(dict(adopted=True,receipt=json.loads(done.read_text()))))
else:
 s=Remote(r,c['endpoint']).call(dict(action='status'));owner=s['receipts']['SPOOL_OWNER.json']
 for k in ('pid','boot_id','start_ticks'):assert owner[k]==c['expected'][k]
 assert m.qualified(s);service=s['receipts']['SERVICE_OWNER.json'];assert service['pid']==owner['pid']
 assert json.loads((r/'WARM_REMOTE_BINDING.json').read_text())['search_sha256']==c['search_sha256']
 assert c['amendment_sha256'] in (r/'AUTHORITY.md').read_text()
 with (r/'AUTHORITY.md').open('a') as f:f.write({(reg['line']+'amendmentSHA'+sha(target)+chr(10))!r})
 immutable(r/('WARM_REMOTE_ENDPOINT_ADMISSION_'+tag+'_SECOND.json'),dict(amendment_sha256=c['amendment_sha256'],search_sha256=c['search_sha256'],endpoints=[c['endpoint']],owner=owner,gates={{n:s['receipts'][n] for n in m.GATES}},unix=time.time()))
 immutable(r/('GPU_RESOURCE_READY_'+c['endpoint']['slot']+'_1651.json'),dict(slot=c['endpoint']['slot'],owner=owner,service=service,amendment_sha256=c['amendment_sha256']))
 v=dict(unix=time.time(),owner=owner,slot=c['endpoint']['slot'],new_sim_s=0)
 immutable(done,v);print(json.dumps(v))''',timeout=40)
    v=json.loads(out);dest=ops.EVIDENCE/f'FAST1844_ADMITTED_{host}.json'
    if not dest.exists():write(dest,v)
    print(out)


def gpu_status(host):
    roots=dict(gpuA=['/root/coopmwm20261102_incident1','/root/coopmwm20261102_deadlineA2'],
               gpuR=['/root/coopmwm20261102_concurrent3','/root/coopmwm20261102_deadlineR2'])
    out=ops.ssh.remote(host,f'''import pathlib,json,time,subprocess
out=[]
for name in {roots[host]!r}:
 r=pathlib.Path(name);v=dict(root=name)
 for n in ('SPOOL_OWNER.json','SERVICE_OWNER.json','SPOOL_FAILED.json'):
  if (r/n).exists():v[n]=json.loads((r/n).read_text())
 d=v.get('SPOOL_OWNER.json');p=pathlib.Path('/proc',str(d['pid'])) if d else None
 if p and p.exists():
  st=(p/'stat').read_text().split();assert st[21]==d['start_ticks'];v['state']=st[2]
 else:v['state']='exited'
 v['gates']={{p.name:json.loads(p.read_text()).get('pass_') for p in r.glob('STARTUP_*.json')}}
 v['inbox']=len(list((r/'inbox').glob('*.json')));v['forecasts']=len(list((r/'forecasts').glob('*.json')))
 out.append(v)
print(json.dumps(dict(host={host!r},unix=time.time(),workers=out,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader'],text=True))))''',timeout=40)
    v=json.loads(out);write(ops.EVIDENCE/f'FAST1844_GPU_{host}_{int(v["unix"])}.json',v);print(out)


def cpu():
    prior=read(ops.EVIDENCE/'CPU_coopmwm20261102_verifybeta13.json')
    files=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib
r=pathlib.Path({prior['root']!r});out={{}}
for n,h in json.loads({json.dumps(prior['source_hashes'])!r}).items():
 p=r/n;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 out[n.removeprefix('pytools/')]=p.read_text()
print(json.dumps(out))''',timeout=40))
    for name in OVERLAYS:
        files['tools/coopmwm/'+name+'.py']=ops.checked(ops.LOCAL/'tools/coopmwm'/(name+'.py')).read_text(encoding='utf-8')
    staged=json.loads(ops.stage('cpuQ',SOURCE,files))
    out=ops.ssh.remote('cpuQ',f'''import os,sys,json,time,unittest,pathlib
from unittest.mock import patch
sys.path[:0]=[{SOURCE!r}+'/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',COOPMWM_MAPPO='/home/ubuntu/mappospeed/production2/policy_00169.pt',COOPMWM_COST_SOURCE={SOURCE!r}+'/FROZEN_COSTS.json',COOPMWM_MOTION_SOURCE={SOURCE!r}+'/FROZEN_MOTION_SOURCE.py')
import torch,pytest;torch.set_num_threads(1)
r=pathlib.Path({SOURCE!r});attempt=1+len(list(r.glob('CPU_ATTEMPT_*_OWNER.json')))
with (r/f'CPU_ATTEMPT_{{attempt}}_OWNER.json').open('x') as f:json.dump(dict(pid=os.getpid(),start_ticks=pathlib.Path('/proc/self/stat').read_text().split()[21],unix=time.time()),f)
names=('test_fast_resume_cpu','test_acceleration_cpu','test_exact_grade_cpu','test_awr_cpu','test_incident_recovery_cpu','test_parallel_groups_cpu','test_warm_health_cpu','test_resource_config_cpu','test_warm_transport_cpu','test_warm_broker_cpu','test_capacity_broker_cpu')
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('tools.coopmwm.'+n) for n in names)
class Counts:
 def __init__(self):self.tests=0;self.skips=0
 def pytest_runtest_logreport(self,report):
  if report.when=='call':self.tests+=1
  if report.skipped:self.skips+=1
c=Counts();start=time.monotonic()
with patch('subprocess.Popen',side_effect=AssertionError('CPU qualification forbids physics/inference')):
 result=unittest.TextTestRunner(stream=sys.stdout,verbosity=1).run(suite)
 code=pytest.main([str(r/'pytools/tools/coopmwm/test_sumo_verify_cpu.py'),str(r/'pytools/tools/coopmwm/test_verify_continuation_cpu.py'),'-q','-p','no:cacheprovider'],plugins=[c])
v=dict(pass_=code==0 and result.wasSuccessful(),tests=c.tests+result.testsRun,skips=c.skips+len(result.skipped),sim_s=0,root=str(r),wall_s=time.monotonic()-start)
with (r/f'CPU_ATTEMPT_{{attempt}}_RESULT.json').open('x') as f:json.dump(v,f)
print(json.dumps(v))''',timeout=90000//1000)
    print(out);v=json.loads(out.strip().splitlines()[-1]);v['source_hashes']=staged['hashes']
    write(ops.EVIDENCE/('CPU_'+SOURCE.rsplit('/',1)[1]+'.json'),v)
    assert v['pass_'] and v['skips']==v['sim_s']==0


def register():
    from .protocol import checked,REGISTRY,sha
    cpu=read(ops.EVIDENCE/('CPU_'+SOURCE.rsplit('/',1)[1]+'.json'))
    assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    target=ops.EVIDENCE/'FAST1844_PLAN.json'
    if target.exists():raise ValueError('adopt existing plan/remote binding, never repeat registration')
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib,time
r=pathlib.Path({ROOT!r});pause=json.loads((r/'LS_COLLECT_PAUSED.json').read_text())
b=json.loads((r/'RESOURCE_EXTENSION_1651.json').read_text());assert b['resume']==dict(B=3,R=3)
assert not (r/'FAST_RESUME_1844.json').exists()
for a in ('B','R'):
 assert not (r/(a+'_checkpoint_004.pt')).exists()
 assert hashlib.sha256((r/(a+'_checkpoint_003.pt')).read_bytes()).hexdigest()==pause['checkpoints'][a]==b['checkpoints'][a]
 d=json.loads((r/('DNS_'+a+'_DISPATCH_1651.json')).read_text());p=pathlib.Path('/proc',str(d['pid']))
 assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 assert not p.exists() or (p/'stat').read_text().split()[21]==d['start_ticks'] and (p/'stat').read_text().split()[2]=='Z'
failed=pause['closed_jobs'];assert len(failed)==10
complete=[]
for w in b['plans']['B']['worlds']:
 key='warm_u004_B_'+str(w['seed'])
 if (r/'jobs'/key/'RESULT.json').exists():complete.append(key)
assert len(complete)==14
for w in b['plans']['R']['worlds']:
 assert not (r/'jobs'/('warm_u004_R_'+str(w['seed']))/'CLAIM.json').exists()
print(json.dumps(dict(binding=b,failed=failed,complete=complete,unix=time.time())))''',timeout=40)
    s=json.loads(out);b=s['binding'];cost=1200*len(s['failed']);assert cost==12000
    b['budget']['phases']['train']+=cost;b['budget']['phases']['discovery']-=cost
    assert b['budget']['phases']['discovery']>=120000
    assert b['budget']['total']==11000000 and sum(b['budget']['phases'].values())==11000000
    b.update(source_root=SOURCE,cpu=cpu,receipt_suffix='_FAST1844',bounded_fanout=True,concurrent_R=True)
    b['job_id_overrides']={f'warm_u004_B_{name.rsplit("_",1)[1]}':
        f'warmfast1844_u004_B_{name.rsplit("_",1)[1]}' for name in s['failed']}
    assert b['sumo_verify']['k']==4 and b['sumo_verify']['guide_beta']==1.
    plan=dict(binding=b,failed=s['failed'],complete=s['complete'],additional_train_sim_s=cost,
        no_design_change=True,authority='MAIN18:44 supersedes LS-COLLECT; MAIN09:58 reserve rule')
    write(target,plan);h=sha(target)
    line=f'\nS-COOPMWM MAIN1844 FAST RESUME REGISTERED planSHA{h}: cancel LS-COLLECT before any collection physics. Restore durable B3/R3/Adam/RNG/library; adopt14 original B4 results. Ten permanently closed DNS attempts get NEW warmfast1844 IDs, identical registered TRAIN worlds/noise; original charges retained. Transfer12000 discovery138200->126200/train2852400->2864400 within11M; extra1.5M k8 phase remains restricted INACTIVE. B quota10 verified-executed, atomic fan-out<=2*live GPU slots, quota-unmet capacity waits without extra RNG draws; first accepted inherited-order response enters verification without waiting later groups; unassigned stale requests decline, assigned searches finish once as data. R runs concurrently using its pinned latest completed B quota, best-effort strata/shortfalls reported. Frozen numerical search unchanged K4H300/finalK8Hfinal/population/iterations/groups/warm memory, k4 SUMO verification/beta1. MirrorFAILED/OFF, learned/k8 training inactive. CPU{cpu["tests"]}PASS/0skips/zeroSUMO. Remote durable supervisors, original unrelated owners retained. No final outcomes opened.\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/'FAST1844_REGISTRATION.json',dict(line=line,amendment_sha256=h,budget=b['budget']))
    b['amendment_sha256']=h
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,hashlib,time
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.incident_recovery import retired_key,install
b=json.loads({json.dumps(b)!r});failed={s['failed']!r}
for n,h in b['cpu']['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
keys=[]
for folder in ('requests','centre_requests'):
 for p in (r/folder).glob('*.json'):
  if retired_key(json.loads(p.read_text()),failed):keys.append(p.stem)
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
immutable(r/'FAST_RETIRED_JOBS_1844.json',dict(jobs=failed,query_keys=keys,amendment_sha256=b['amendment_sha256']))
immutable(r/'FAST_RESUME_1844.json',b)
for a in ('B','R'):
 wr,_=install(r,a);v=wr.verify(r,a)
 assert v['resume_update']==3 and v['sumo_verify']['k']==4 and v['sumo_verify']['guide_beta']==1.
 assert v['bounded_fanout'] and v['concurrent_R'] and v['gpu_deadline']==1793894340.
 immutable(r/('FAST_'+a+'_PREFLIGHT_1844.json'),dict(unix=time.time(),restored_state_sha256=__import__('tools.coopmwm.protocol',fromlist=['digest']).digest(v['resume_progress']),resume_update=3,new_sim_s=0))
print(json.dumps(dict(bound=True,unix=time.time(),retired_queries=len(keys),additional_sim_s={cost},discovery=b['budget']['phases']['discovery'],source=str(s))))''',timeout=60)
    write(ops.EVIDENCE/'FAST1844_BOUND.json',json.loads(out));print(out)


def dispatch():
    """Single broker handover plus two fresh training owners; no GPU signals."""
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,os,time,signal,subprocess
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.incident_recovery import binding,install
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.warm_health import assert_running
b=binding(r);assert_running(r,'B');assert_running(r,'R')
boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for a in ('B','R'):
 assert not (r/('FAST_'+a+'_DISPATCH_1844.json')).exists(),'adopt existing owner'
 pre=json.loads((r/('FAST_'+a+'_PREFLIGHT_1844.json')).read_text())
 assert pre['resume_update']==3 and pre['new_sim_s']==0
 assert __import__('hashlib').sha256((r/(a+'_checkpoint_003.pt')).read_bytes()).hexdigest()==b['checkpoints'][a]
 assert not (r/(a+'_checkpoint_004.pt')).exists()
old=json.loads((r/'GPU_RESOURCE_BROKER_DISPATCH_1651_V2.json').read_text());p=pathlib.Path('/proc',str(old['pid']))
assert old['boot_id']==boot
if p.exists() and (p/'stat').read_text().split()[2]!='Z':
 assert (p/'stat').read_text().split()[21]==old['start_ticks']
 argv=[v.decode() for v in (p/'cmdline').read_bytes().split(bytes([0])) if v]
 assert argv==['/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.capacity_broker',str(r),str(r/'GPU_RESOURCE_BROKER_CONFIG_1651_V2.json')]
 os.kill(old['pid'],signal.SIGTERM)
 for _ in range(150):
  if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
  assert (p/'stat').read_text().split()[21]==old['start_ticks'];time.sleep(.1)
 else:raise ValueError('old broker did not exit')
cfg=json.loads((r/'GPU_RESOURCE_BROKER_CONFIG_1651_V2.json').read_text());cfg.update(source_root=str(s),receipt_tag='_FAST1844',amendment_sha256=b['amendment_sha256'])
immutable(r/'FAST_BROKER_CONFIG_1844.json',cfg)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo')
env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH'];owners={{}}
for c in ('broker','B','R'):
 cmd=['/home/ubuntu/miniconda3/bin/python','-B','-u','-m']
 if c=='broker':cmd+=['tools.coopmwm.capacity_broker',str(r),str(r/'FAST_BROKER_CONFIG_1844.json')]
 else:cmd+=['tools.coopmwm.incident_recovery',str(r),'--component',c]
 with (r/('FAST_'+c.upper()+'_1844.log')).open('x') as log:
  proc=subprocess.Popen(['nice','-n','19']+cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 st=pathlib.Path('/proc',str(proc.pid),'stat').read_text().split()
 d=dict(pid=proc.pid,start_ticks=st[21],boot_id=boot,source=str(s),root=str(r),component=c,unix=time.time())
 immutable(r/('FAST_'+c.upper()+'_DISPATCH_1844.json'),d);owners[c]=d
immutable(r/'FAST_TRAINING_STARTED_1844.json',dict(unix=time.time(),owners=owners,resume=dict(B=3,R=3),verification_k=4,beta=1.,new_job_count=10))
print(json.dumps(owners))''',timeout=60)
    write(ops.EVIDENCE/'FAST1844_DISPATCH.json',json.loads(out));print(out)


def status():
    """TRAIN-only progress/ownership; no final outcomes or evaluation paths."""
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,time,os
r=pathlib.Path({ROOT!r});now=time.time();b=json.loads((r/'FAST_RESUME_1844.json').read_text())
started=json.loads((r/'FAST_TRAINING_STARTED_1844.json').read_text());owners={{}}
boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def tree(pid):
 p=pathlib.Path('/proc',str(pid));out=[]
 try:
  st=(p/'stat').read_text().split();argv=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
  if 'sumo ' in argv:out.append(dict(pid=pid,cpu_ticks=int(st[13])+int(st[14]),state=st[2]))
  for child in (p/'task'/str(pid)/'children').read_text().split():out+=tree(int(child))
 except FileNotFoundError:pass
 return out
for c in ('broker','B','R'):
 d=json.loads((r/('FAST_'+c.upper()+'_DISPATCH_1844.json')).read_text());assert d['boot_id']==boot
 p=pathlib.Path('/proc',str(d['pid']));st=(p/'stat').read_text().split() if p.exists() else None
 assert not st or st[21]==d['start_ticks']
 owners[c]=dict(owner=d,state=st[2] if st else 'exited',sumo=tree(d['pid']),log=(r/('FAST_'+c.upper()+'_1844.log')).read_text()[-2500:])
requests=[p for p in (r/'requests').glob('*.json') if p.stat().st_mtime>=started['unix']]
responses=[p for p in requests if (r/'responses'/p.name).exists()]
verifyroot=pathlib.Path(b['sumo_verify']['root']);decisions=[]
for p in (verifyroot/'verifications').glob('sv_warm*.json'):
 if p.stat().st_mtime<started['unix']:continue
 v=json.loads(p.read_text());decisions.append(dict(id=p.stem,status=v['status'],k=v['k'],t0=v['t0']))
admissions=sorted(r.glob('WORKER_ADMISSION_*.json'));slots=json.loads(admissions[-1].read_text())['workers']
value=dict(unix=now,training_start_unix=started['unix'],owners=owners,slots=[v['slot'] for v in slots],
 searches_requested=len(requests),searches_answered=len(responses),searches_pending=len(requests)-len(responses),
 admitted=sum((r/'capacity_admissions'/p.name).exists() for p in requests),
 accepted=sum(json.loads((r/'responses'/p.name).read_text()).get('accepted',False) for p in responses),
 stale=sum((r/'query_cancellations'/p.name).exists() for p in requests),verification_decisions=decisions,
 verification_launches=len([p for p in (verifyroot/'snapshots').glob('*') if p.stat().st_mtime>=started['unix']]),
 R_reference=json.loads((r/'R_MATCH_REFERENCE_004.json').read_text()) if (r/'R_MATCH_REFERENCE_004.json').exists() else None,
 discovery_allocation=b['budget']['phases']['discovery'],cap=b['budget']['total'],load=os.getloadavg(),
 preflight={{a:json.loads((r/('FAST_'+a+'_PREFLIGHT_1844.json')).read_text()) for a in ('B','R')}},
 eta_u4=None,eta_reason='pending first quota-compliant verified update; unresolved waits censored')
print(json.dumps(value))''',timeout=40)
    v=json.loads(out);write(ops.EVIDENCE/f'FAST1844_CURRENT_{int(v["unix"])}.json',v);print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('gpu_status','cpu','admit','register','dispatch','status'))
    p.add_argument('--host',choices=('gpuA','gpuR'));a=p.parse_args()
    if a.action in ('gpu_status','admit'):globals()[a.action](a.host)
    else:globals()[a.action]()
