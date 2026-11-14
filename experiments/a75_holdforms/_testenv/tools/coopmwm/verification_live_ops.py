"""Register the 9.5M verification allocation, qualify and launch ONE real smoke."""
import argparse
import copy
import json
from . import ops
from .storage import read,write
from .protocol import REGISTRY,checked,sha
from .warm_deploy import ROOT

SOURCE='/home/ubuntu/coopmwm20261102_verifycpu12'


def register():
    target=ops.EVIDENCE/'SUMO_VERIFICATION_PLAN.json'
    assert not target.exists(),'adopt existing registration'
    b=copy.deepcopy(read(ops.EVIDENCE/'MAIN1011_RECOVERY_PLAN.json')['budget'])
    b['total']=9500000
    b['phases']['sumo_verification']=2800000
    b['phases']['exact_plan_grading']-=328840
    b['allocated']=sum(b['phases'].values());b['unallocated']=b['total']-b['allocated']
    assert b['allocated']==9500000 and b['unallocated']==0
    v=dict(authority='OWNER10:20/10:22 and MAIN10:30/14:24/14:47',budget=b,
        selection='B top4 WM candidates and own-policy branch; R4 random candidates and own-policy branch; same common update4; beta BEFORE branches; exact lowest-delay winner only when strictly better than own policy; grade reused, no duplicate grader run',
        quote='remaining B21+R21 updates x minimum10 x max6000 =2520000 sim-s; 2800000 verification phase covers that lower-bound plan plus smoke/rejected decisions/extra plans up to phase cap. Variable acceptance is not guaranteed; phase/cap blockers STOP and report, never reduce horizon or bypass accounting',
        funding='owner cap+2000000, prior spare471160, replace328840 of unspent separate exact grading with reused verifier grades; remaining exact grade431160 (already charged16530 retained); no discovery transfer',
        smoke_rule='first B-u3 completed replacement episode in registered ascending seed order, first accepted forecast with exactly4 topk; no exact grades/outcomes consulted; new-ID replay through t0+10, five exact branches plus policy continuation, all charged in sumo_verification; never retry physics',
        invariants='K4H300/K8Hfinal, warm memory, search, trust, PPO/AWR, health pauses unchanged; primaryA24/B24/R24, secondaryA48; fixed sealed test plan retained; no outcome opening')
    write(target,v);h=sha(target)
    line='\nS-COOPMWM SUMO VERIFICATION ALLOCATION/COMMON-U4 REGISTERED planSHA'+h+': '+json.dumps(v,sort_keys=True)+'\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/'SUMO_VERIFICATION_REGISTRATION.json',dict(amendment_sha256=h,line=line))
    print(json.dumps(dict(registered=True,amendment_sha256=h,budget=b)))


def cpu():
    from . import verify_ops as v
    v.SOURCE=SOURCE
    v.OVERLAYS=tuple(dict.fromkeys(v.OVERLAYS+('warm_protocol','warm_runner','warm_health',
        'r_matching','test_r_matching_cpu','test_warm_health_cpu','incident_recovery',
        'capacity_broker','verification_smoke')))
    v.cpu()


def register_gate():
    target=ops.EVIDENCE/'VERIFICATION_SMOKE_GATE_AMENDMENT.json';assert not target.exists()
    v=dict(diagnosis='all five real branches completed prefix/CRN checks; candidates rejected and policy continued; helper wrongly required acceptance',
        correction='validate live continuation against chosen candidate when accepted or policy branch when rejected; SHA/prefix/CRN/no-new-plan checks retained; no outcome-conditioned repeat',
        recovery='CPU qualify then recheck original completed TRAIN live/branch receipts ONLY; zero new SUMO/inference/charges; retain initial gate failure and all source/dispatch receipts')
    write(target,v);h=sha(target)
    line='\nS-COOPMWM OUTCOME-INDEPENDENT VERIFICATION SMOKE GATE REGISTERED amendmentSHA'+h+': '+json.dumps(v,sort_keys=True)+'\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/'VERIFICATION_SMOKE_GATE_REGISTRATION.json',dict(amendment_sha256=h,line=line))
    print(json.dumps(dict(registered=True,amendment_sha256=h,new_sumo_sim_s=0)))


def recheck():
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_verifycpu11.json')
    reg=read(ops.EVIDENCE/'VERIFICATION_SMOKE_GATE_REGISTRATION.json')
    assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,hashlib,time
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.accel_storage import read
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.verification_grades import record_smoke
cpu=json.loads({json.dumps(cpu)!r})
for n,h in cpu['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h
v=read(r/'VERIFICATION_RUNTIME_BINDING.json');spec=v['smoke']
live=read(r/'jobs'/spec['id']/'RESULT.json')
paths=list((r/'sumo_verification_smoke'/'verifications').glob('*.json'));assert len(paths)==1
receipt=read(paths[0])
with (r/'AUTHORITY.md').open('a') as f:f.write({reg['line']!r})
gate=record_smoke(r/'sumo_verification_smoke',receipt,live)
record_smoke(r/'sumo_verification',receipt,live)
passed=dict(gate,branch_count=5,charged_original_sim_s=spec['quote_sim_s'],recheck_new_sim_s=0,source_root=str(s),amendment_sha256={reg['amendment_sha256']!r})
immutable(r/'VERIFICATION_SMOKE_PASSED.json',passed)
print(json.dumps(passed))''',timeout=45)
    write(ops.EVIDENCE/'VERIFICATION_SMOKE_PASSED.json',json.loads(out));print(out)


def bind():
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_verifycpu10.json')
    reg=read(ops.EVIDENCE/'SUMO_VERIFICATION_REGISTRATION.json')
    plan=read(ops.EVIDENCE/'SUMO_VERIFICATION_PLAN.json')
    assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    payload=dict(cpu=cpu,source_root=SOURCE,amendment_sha256=reg['amendment_sha256'],budget=plan['budget'])
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,hashlib,time
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.accel_storage import read
from tools.coopmwm.exact_grade import train_noise_seed
v=json.loads({json.dumps(payload)!r});assert not (r/'VERIFICATION_RUNTIME_BINDING.json').exists(),'adopt existing binding'
for n,h in v['cpu']['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h
old=read(r/'INCIDENT_RECOVERY_BINDING.json');found=None
for w in old['plans']['B']['worlds']:
 ident='warm_u003_B_'+str(w['seed']);ep=read(r/'jobs'/ident/'RESULT.json')
 for step in ep['steps']:
  p=step.get('proposal') or {{}}
  if p.get('accepted') and len(p.get('topk',[]))==4:
   t=int(step['time_s']);stop=(t//10+1)*10;end=min(t+max(c['duration_s'] for c in p['topk'])+120,1200)
   found=dict(id='verification_smoke_u003_B_'+str(w['seed'])+'_t'+str(t),original_episode=ident,time_s=t,stop=stop,noise_seed=train_noise_seed(w['seed'],3,20261101),original_result_sha256=hashlib.sha256((r/'jobs'/ident/'RESULT.json').read_bytes()).hexdigest(),quote_sim_s=stop+5*end)
   break
 if found:break
assert found is not None
v['smoke']=found;v['common_update']=4
with (r/'AUTHORITY.md').open('a') as f:f.write({reg['line']!r})
immutable(r/'VERIFICATION_RUNTIME_BINDING.json',v)
print(json.dumps(dict(bound=True,unix=time.time(),smoke=found,budget=v['budget'],new_physics=0)))''',timeout=45)
    write(ops.EVIDENCE/'SUMO_VERIFICATION_BOUND.json',json.loads(out));print(out)


def smoke():
    dest='VERIFICATION_SMOKE_DISPATCH.json'
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,os,sys,subprocess,time,shutil
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.warm_health import assert_running
assert_running(r,'B');assert not (r/{dest!r}).exists() and not (r/'VERIFICATION_SMOKE_STARTED.json').exists()
assert shutil.disk_usage(r).free>=2*1024**3
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo',PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH']
with (r/'VERIFICATION_SMOKE.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',{ops.ssh.PYTHON['cpuQ']!r},'-B','-u','-m','tools.coopmwm.verification_smoke',str(r)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=pathlib.Path('/proc',str(p.pid),'stat').read_text().split();v=dict(pid=p.pid,start_ticks=st[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source=str(s),root=str(r),unix=time.time())
immutable(r/{dest!r},v);print(json.dumps(v))''',timeout=40)
    write(ops.EVIDENCE/dest,json.loads(out));print(out)


def status():
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,time,shutil
r=pathlib.Path({ROOT!r});names=('VERIFICATION_SMOKE_DISPATCH.json','VERIFICATION_SMOKE_STARTED.json','VERIFICATION_SMOKE_PASSED.json','WARM_GUIDED_STOP_V4.json')
v=dict(unix=time.time(),receipts={{n:json.loads((r/n).read_text()) for n in names if (r/n).exists()}},log=(r/'VERIFICATION_SMOKE.log').read_text()[-4000:] if (r/'VERIFICATION_SMOKE.log').exists() else None,disk_free=shutil.disk_usage(r).free)
d=v['receipts'].get('VERIFICATION_SMOKE_DISPATCH.json');p=pathlib.Path('/proc',str(d['pid']),'stat') if d else None
v['state']=p.read_text().split()[2] if p and p.exists() else 'exited'
print(json.dumps(v))''',timeout=35))
    write(ops.EVIDENCE/f'VERIFICATION_SMOKE_STATUS_{int(value["unix"])}.json',value)
    print(json.dumps(value))


def activate_binding():
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_verifycpu12.json')
    reg=read(ops.EVIDENCE/'SUMO_VERIFICATION_REGISTRATION.json')
    gate=read(ops.EVIDENCE/'VERIFICATION_SMOKE_PASSED.json')
    assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0 and gate['pass_']
    repair=read(ops.EVIDENCE/'VERIFICATION_U4_REPAIR_BINDING.json')
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,hashlib,time
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.warm_protocol import batch_worlds
from tools.coopmwm.tail_runtime import restore_progress
from tools.coopmwm.warm_health import assert_running
cpu=json.loads({json.dumps(cpu)!r});assert_running(r,'B')
repair=json.loads({json.dumps(repair)!r})
assert not (r/'VERIFICATION_CONTINUATION_BINDING.json').exists(),'adopt existing activation'
for n,h in cpu['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h
assert json.loads((r/'VERIFICATION_SMOKE_PASSED.json').read_text())['pass_']
for a,n in [('B','INCIDENT_B_DISPATCH.json'),('R','MATCH_R_DISPATCH.json')]:
 d=json.loads((r/n).read_text());assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 p=pathlib.Path('/proc',str(d['pid']),'stat');st=p.read_text().split() if p.exists() else None
 assert not st or st[21]==d['start_ticks'] and st[2]=='Z','old arm still live'
 assert (r/f'{{a}}_UPDATE_003.json').exists() and not (r/f'{{a}}_UPDATE_004.json').exists()
b=json.loads((r/'MATCH_RECOVERY_BINDING.json').read_text());previous=json.loads((r/'MAIN0508_REPAIR_BINDING_V2.json').read_text())
history={{'B':{{'3':b['plans']['B']['worlds']}},'R':{{'2':b['plans']['R']['worlds']}}}}
plans={{}}
for a in ('B','R'):
 ws=repair['plan']['worlds'] if a=='B' else batch_worlds(4);pre=[]
 for w in ws:
  ident=f'warm_u004_{{a}}_{{w["seed"]}}';p=r/'jobs'/ident
  assert not (p/'CLAIM.json').exists(),'never replay attempted u4'
  if p.exists():pre.append(ident)
 plans[a]=dict(worlds=ws,precharged_unattempted=pre)
 def schedule(a,u):
  if u==1:return previous['repair_plan']['worlds']
  return history.get(a,{{}}).get(str(u),batch_worlds(u))
 progress={{a:restore_progress(r,a,3,lambda u:schedule(a,u),20261101) for a in ('B','R')}}
v=json.loads((r/'VERIFICATION_RUNTIME_BINDING.json').read_text())
with (r/'AUTHORITY.md').open('a') as f:f.write(repair['line'])
b.update(cpu=cpu,source_root=str(s),amendment_sha256=repair['amendment_sha256'],budget=repair['budget'],resume={{'B':3,'R':3}},
 checkpoints={{a:hashlib.sha256((r/f'{{a}}_checkpoint_003.pt').read_bytes()).hexdigest() for a in ('B','R')}},
 receipt_suffix='_SV1',stop_after_update={{}},history_plans=history,plans=plans,
 sumo_verify=dict(root=str(r/'sumo_verification'),run_root=str(r),phase='sumo_verification',k=4,workers=5,timeout_s=3600))
immutable(r/'VERIFICATION_CONTINUATION_BINDING.json',b)
immutable(r/'VERIFICATION_ACTIVATION_PREFLIGHT.json',dict(unix=time.time(),common_update=4,checkpoints=b['checkpoints'],precharged={{a:plans[a]['precharged_unattempted'] for a in plans}},restored={{a:dict(beta=progress[a]['beta'],exact_count=progress[a]['exact_count'],library_entries=len((progress[a]['learned'] or {{}}).get('library',[]))) for a in progress}},source_root=str(s),cpu_tests=cpu['tests'],new_physics=0))
print((r/'VERIFICATION_ACTIVATION_PREFLIGHT.json').read_text())''',timeout=60)
    write(ops.EVIDENCE/'VERIFICATION_ACTIVATION_PREFLIGHT.json',json.loads(out));print(out)


def register_u4_repair():
    from .incident_recovery import episode_plan
    from .warm_protocol import batch_worlds,worlds
    target=ops.EVIDENCE/'VERIFICATION_U4_REPAIR_PLAN.json';assert not target.exists()
    state=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,time
r=pathlib.Path({ROOT!r});jobs={{}}
for p in (r/'jobs').glob('warm_u004_B_*'):
 c=json.loads((p/'CLOSE.json').read_text()) if (p/'CLOSE.json').exists() else None
 jobs[p.name]=dict(closed=(p/'RESULT.json').exists(),claimed=(p/'CLAIM.json').exists(),close=c)
assert len(jobs)==24 and all(v['claimed'] and not v['closed'] and v['close']['actual_sim_s']==300 for v in jobs.values())
assert not (r/'VERIFICATION_CONTINUATION_BINDING.json').exists()
print(json.dumps(dict(unix=time.time(),jobs=jobs)))''',timeout=40))
    p=episode_plan(batch_worlds(4),worlds('train'),state['jobs'],'B',4)
    assert p['additional_sim_s']==28800
    b=copy.deepcopy(read(ops.EVIDENCE/'SUMO_VERIFICATION_PLAN.json')['budget'])
    b['phases']['train']+=28800;b['phases']['discovery']-=28800
    assert b['phases']['discovery']>=120000 and sum(b['phases'].values())==b['allocated']==9500000
    v=dict(authority='MAIN09:58 under50000 engineering replacement transfer from discovery reserve floor120000',
        diagnosis='old B-u4 passed disk admission after archive and all24 workers warmed to300 before noticing pre-existing R matching STOP; no control outcomes/results; original28800 charges retained',
        plan=p,budget=b,state=state,new_physics_dispatched=0,
        recovery='same-cell ascending unused registered TRAIN replacements; never reuse failed job IDs; original B3 optimizer/RNG/library; commonu4 verifier activation unchanged')
    write(target,v);h=sha(target)
    line='\nS-COOPMWM COMMON-U4 ENGINEERING RECOVERY REGISTERED amendmentSHA'+h+': '+json.dumps({k:x for k,x in v.items() if k!='state'},sort_keys=True)+'\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    v.update(amendment_sha256=h,line=line)
    # Keep the hashed registration payload immutable; enriched binding separate.
    write(ops.EVIDENCE/'VERIFICATION_U4_REPAIR_BINDING.json',v)
    print(json.dumps(dict(registered=True,amendment_sha256=h,replacement_sim_s=28800,discovery_allocation=b['phases']['discovery'],new_physics_dispatched=0)))


def replace_grader():
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,os,signal,time
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.exact_runtime import queue
assert (r/'VERIFICATION_CONTINUATION_BINDING.json').exists()
d=json.loads((r/'MATCH_GRADER_DISPATCH.json').read_text());assert d['pid']==544875 and d['start_ticks']=='61697761'
assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
p=pathlib.Path('/proc',str(d['pid']));st=(p/'stat').read_text().split();assert st[21]==d['start_ticks'] and st[2] in ('S','R')
args=[a.decode() for a in (p/'cmdline').read_bytes().split(bytes([0])) if a]
assert args[-4:]==['tools.coopmwm.incident_recovery',str(r),'--component','grader']
assert not (p/'task'/str(d['pid'])/'children').read_text().strip()
q=queue(r);assert not q.pending()
v=dict(owner=d,argv=args,unix=time.time(),reason='commonu4 verified grade reuse source replacement; queue empty/no children',signal='SIGTERM')
immutable(r/'VERIFIED_GRADER_HANDOVER.json',v);os.kill(d['pid'],signal.SIGTERM)
print(json.dumps(v))''',timeout=40)
    write(ops.EVIDENCE/'VERIFIED_GRADER_HANDOVER.json',json.loads(out));print(out)


def dispatch(component):
    assert component in ('grader','B','R')
    dest='VERIFIED_'+component.upper()+'_DISPATCH.json'
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,os,subprocess,time
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.incident_recovery import binding
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.warm_health import assert_running
b=binding(r);assert_running(r,'B');assert not (r/{dest!r}).exists(),'adopt existing owner'
if {component!r}=='grader':
 d=json.loads((r/'MATCH_GRADER_DISPATCH.json').read_text());p=pathlib.Path('/proc',str(d['pid']),'stat')
 assert not p.exists() or p.read_text().split()[21]==d['start_ticks'] and p.read_text().split()[2]=='Z'
else:
 from tools.coopmwm.exact_runtime import assert_live
 assert_live(r)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo',PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH']
with (r/{('VERIFIED_'+component.upper()+'.log')!r}).open('x') as log:
 p=subprocess.Popen(['nice','-n','19',{ops.ssh.PYTHON['cpuQ']!r},'-B','-u','-m','tools.coopmwm.incident_recovery',str(r),'--component',{component!r}],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=pathlib.Path('/proc',str(p.pid),'stat').read_text().split();v=dict(pid=p.pid,start_ticks=st[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source=str(s),root=str(r),unix=time.time(),component={component!r},first_update=4,sumo_verification=True)
immutable(r/{dest!r},v);print(json.dumps(v))''',timeout=45)
    write(ops.EVIDENCE/dest,json.loads(out));print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('register','cpu','bind','smoke','status','register_gate','recheck','register_u4_repair','activate_binding','replace_grader','dispatch'))
    p.add_argument('--component',choices=('grader','B','R'));a=p.parse_args()
    dispatch(a.component) if a.mode=='dispatch' else globals()[a.mode]()
