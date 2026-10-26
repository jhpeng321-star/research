"""Owner-authorized S-MAINR16 continuation using the cpuQ provenance copy."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess

import collectfix as cf

ROOT = cf.ROOT
OUT = ROOT / 'tools/mainstart/mainr16_resume'
A = '/root/autodl-tmp/c4/v5/mainstart/mainr16/restart'
R = A + '/runtime'
CPU = '/home/ubuntu/c4/main14/mainr16'
BASE = '/root/main14_loop/mainr16_base'
SHA = '5b2569418bf651b8ef2c3a2abdc110ac6860c32dd92dca8cdf3b2f48b1c8cbb2'


def receipt(name, text):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(text, encoding='utf-8')
    print(text)


def authorize():
    direction = (ROOT/'tools/mainstart/MAINR16_INVOCATION_DIRECTION.txt').read_text().strip()
    shared = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md').read_bytes()
    assert direction in shared.decode('utf-8')
    code = f'''
import pathlib,json,hashlib,shutil,datetime,sys
a=pathlib.Path({A!r});r=a/'runtime';root=pathlib.Path('/root/main14_loop')
sys.path[:0]=[str(r/'scripts/c4'),str(r),str(r/'src')]
import loop14 as loop,main14_autolaunch as auto
arm=loop.read_json(a/'ARMED.json');auto.verify_source(arm)
panel_sha=loop.sha256(a/'gate_panel.json');assert panel_sha==arm['gate_panel_sha256']
assert panel_sha=='3eaa69a170c17302660d12dd4d7fddb4657e43fa334eece17844dd0fa6c0411e'
state=loop.read_json(root/'LOOP14_STATE.json')
assert state['sim_s_charged']==58266 and state['cycles'][0]['phases']['collect']['status']=='failed'
wait=a/'WAIT_OWNER.json';archived=a/'WAIT_OWNER.pre_invocation_direction.json'
assert wait.is_file() and not archived.exists() and not (a/'MIGRATED.json').exists()
reg=loop.read_json(a/'REGISTRATION.json')
for name in ('REGISTRATION.json','ARMED.json','PROPOSAL.json'):
 shutil.copyfile(a/name,a/(name+'.pre_invocation_direction'))
reg['mainr16_amendment']['invocation_correction_direction']={direction!r}
reg['mainr16_amendment']['corrected_prearm']='policy14_cli --device cpu --max-updates 2 --compact --smoke; no --gate-panel or --horizon-gate; frozen panel hash-checked separately'
reg['mainr16_amendment']['frozen_panel_sha256']=panel_sha
reg['gates']['failure']='Scientific gate failure or second non-scientific invocation/deployment failure: WAIT_OWNER; one documented fix/retry permitted for further non-scientific errors'
loop.write_atomic(a/'REGISTRATION.json',reg)
arm['registration_sha256']=loop.sha256(a/'REGISTRATION.json');loop.write_atomic(a/'ARMED.json',arm)
proposal=loop.read_json(a/'PROPOSAL.json');proposal['registration_sha256']=arm['registration_sha256']
loop.write_atomic(a/'PROPOSAL.json',proposal)
wait.rename(archived)
for source,dest in (('prearm_smoke.log','prearm_smoke.invocation_error.log'),('smoke','smoke.invocation_error')):
 p=a/source
 if p.exists():
  assert not (a/dest).exists();p.rename(a/dest)
doc={{'t':loop.now(),'owner_direction':{direction!r},'archived_wait_owner':str(archived),
 'frozen_panel_sha256':panel_sha,'registration_sha256':arm['registration_sha256'],
 'source_manifest_sha256':arm['native_source_manifest_sha256'],'shared_register_sha256':{hashlib.sha256(shared).hexdigest()!r},
 'corrected_prearm':reg['mainr16_amendment']['corrected_prearm'],'retry_allowance':1,'new_failures':0}}
loop.write_atomic(a/'INVOCATION_AUTHORIZED.json',doc)
print(json.dumps({{'authorization':doc,'proposal':proposal}}))
'''
    result = cf.remote('gpuR', code)
    doc = json.loads(result)
    receipt('INVOCATION_AUTHORIZED.json', json.dumps(doc['authorization'],indent=1))
    receipt('PROPOSAL.json', json.dumps(doc['proposal'],indent=1))


def probe():
    code = '''
import pathlib,json,hashlib
r=pathlib.Path('/home/ubuntu/autopilot_r10/runs/r16/ck')
names=['wm_s_best.pt','physics.json','config.resolved.json','md5.txt',
 'provenance/SOURCE_SHA256.json','provenance/RUN_SUMMARY.json',
 'provenance/TRAIN_RECEIPT.json','provenance/INIT_FROM.json']
doc={}
for n in names:
 p=r/n
 if not p.is_file():
  doc[n]={'exists':False};continue
 doc[n]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 if p.suffix in ('.json','.txt'):doc[n]['text']=p.read_text()
print(json.dumps(doc))
'''
    receipt('SOURCE_PROBE.json', cf.remote('cpuQ', code))
    code = '''
import pathlib,json,shutil,subprocess
r=pathlib.Path('/root/main14_loop')
print(json.dumps({'state':json.loads((r/'LOOP14_STATE.json').read_text()),
 'free':shutil.disk_usage(r).free,'gpu':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True),
 'stops':[(str(p),p.exists()) for p in [r/'STOP_DRIVER',r/'policy/c00/STOP']]}))
'''
    receipt('STATE_PROBE.json', cf.remote('gpuR', code))


def stage_base():
    code = f'''
import pathlib,json,subprocess,hashlib,shutil
a=pathlib.Path({A!r});assert a.parent.is_dir();a.mkdir(exist_ok=True)
b=pathlib.Path({BASE!r});assert b.parent.is_dir();assert not b.exists();b.mkdir()
assert shutil.disk_usage(b).free>6*1024**3
src='cpuQ:/home/ubuntu/autopilot_r10/runs/r16/'
for name in ('wm_s_best.pt','physics.json','config.resolved.json','md5.txt'):
 subprocess.run(['rsync','-a','-e','ssh -o BatchMode=yes -o ConnectTimeout=15',src+'ck/'+name,str(b/name)],check=True,capture_output=True)
(b/'provenance').mkdir()
for name in ('SOURCE_SHA256.json','RUN_SUMMARY.json','TRAIN_RECEIPT.json','INIT_FROM.json'):
 subprocess.run(['rsync','-a','-e','ssh -o BatchMode=yes -o ConnectTimeout=15',src+'provenance/'+name,str(b/'provenance'/name)],check=True,capture_output=True)
source=json.loads((b/'provenance/SOURCE_SHA256.json').read_text())
checks={{}}
for name in ('wm_s_best.pt','physics.json','config.resolved.json'):
 key='wm_s.pt' if name=='wm_s_best.pt' else name
 digest=hashlib.sha256((b/name).read_bytes()).hexdigest();assert digest==source[key]
 checks[name]=digest
assert checks['wm_s_best.pt']=={SHA!r}
for name in ('RUN_SUMMARY.json','TRAIN_RECEIPT.json','INIT_FROM.json'):
 assert hashlib.sha256((b/'provenance'/name).read_bytes()).hexdigest()==source[name]
for line in (b/'md5.txt').read_text().splitlines():
 digest,name=line.split();assert name in ('wm_s_best.pt','physics.json','config.resolved.json')
 assert hashlib.md5((b/name).read_bytes()).hexdigest()==digest
physics=json.loads((b/'physics.json').read_text())
config=json.loads((b/'config.resolved.json').read_text())['config']
assert config['model']['ramp_speed_adapt_learnable'] and 'ramp_speed_adapt' in config['optim']['freeze_prefixes']
assert physics['emergency_onset']=='driver_b' and physics['lane_change_secure_gap']
doc={{'source':'cpuQ:/home/ubuntu/autopilot_r10/runs/r16/ck/wm_s_best.pt',
 'original_source':'gpuB:/root/autodl-tmp/c4/v5/runs/c4_wm_v10_round16_b/wm_s.pt',
 'destination':str(b),'sha256':checks,'provenance':source,'md5_all_verified':True,
 'physics':physics,'free_bytes':shutil.disk_usage(b).free}}
(a/'STAGED_BASE.json').write_text(json.dumps(doc,indent=1));print(json.dumps(doc))
'''
    receipt('STAGED_BASE.json', cf.remote('gpuR', code, timeout=300))


def deploy():
    cf.A, cf.R, cf.CPU, cf.OUT = A, R, CPU, OUT
    cf.stage()


def prepare():
    owner = (ROOT / 'tools/mainstart/MAINR16_OWNER_DECISIONS.txt').read_text().strip()
    substitution = ('Owner continuation: gpuB unreachable; substitute cpuQ ~/autopilot_r10/runs/r16/ck/wm_s_best.pt '
                    'and sibling provenance/ for the same round-16 FINAL; SHA256 ' + SHA +
                    '; verify SOURCE_SHA256.json and md5.txt, then continue the original approved restart plan.')
    code = f'''
import pathlib,json,hashlib,copy,shutil,sys
a=pathlib.Path({A!r});r=a/'runtime';b=pathlib.Path({BASE!r});root=pathlib.Path('/root/main14_loop')
assert not (a/'CONFIG.json').exists()
state=json.loads((root/'LOOP14_STATE.json').read_text())
cfg=json.loads((a.parent.parent/'collectfix/CONFIG.json').read_text())
reg=json.loads((a.parent/'REGISTRATION.json').read_text())
source=json.loads((r/'COLLECTFIX_SOURCE.json').read_text())
cfg['registration']=str(a/'REGISTRATION.json');cfg['hosts']['gpu']['code_root']=str(r)
for name in ('cpuP','cpuQ'):
 cfg['hosts'][name]['code_root']={CPU!r}+'/runtime'
 cfg['mainstart']['collect_hosts'][name]=copy.deepcopy(cfg['hosts'][name])
cfg['hosts']['cpu']=copy.deepcopy(cfg['hosts']['cpuP'])
cfg['collect']['work_root']={CPU!r}+'/work'
cfg['collect']['env']['PATH']='/home/ubuntu/miniconda3/bin:/usr/local/bin:/usr/bin:/bin:/home/ubuntu/.local/bin'
cfg['wm'].update(initial_wm=str(b/'wm_s_best.pt'),initial_wm_config=str(b/'config.resolved.json'),
 config='configs/c4_wm_loop14_r16_ft.toml')
cfg['policy']['allowed_roots'] += [str(a),str(b)]
for flag,name in (('--gate-panel','gate_panel.json'),('--horizon-gate','horizon_gate_current.json')):
 args=cfg['policy']['extra_args'];old=args[args.index(flag)+1]
 shutil.copyfile(old,a/name);args[args.index(flag)+1]=str(a/name)
cfgtext=json.dumps(cfg,indent=1)+'\\n';(a/'CONFIG.json').write_text(cfgtext)
primary=29133
reg['budget'].update(sim_s_total=210000,sim_s_per_cycle=state['sim_s_charged']+primary)
reg['sources'].update(initial_wm_rule='Round16 FINAL by explicit S-MAINR16 restart approval',
 initial_wm_sha256={SHA!r},code_sha='bc806211 + owner cpuQ continuation',
 config_sha256=hashlib.sha256(cfgtext.encode()).hexdigest(),source_manifest_sha256=source['sha256'])
reg['mainr16_amendment'].update(owner_decisions_verbatim={owner!r},status='registered for restart',
 base_checkpoint_sha256={SHA!r},checkpoint_sha_status='SHA256 and MD5 verified against copy-time provenance',
 checkpoint_stage=str(b),source_substitution={substitution!r},
 cycle0_combined_cap=state['sim_s_charged']+primary,
 accounting='Retain all58266 prior charges; add one fresh29133 restart cohort, cycle0 combined cap87399; no automatic failed-phase retries; total210000',
 source_manifest_sha256=source['sha256'])
(a/'REGISTRATION.json').write_text(json.dumps(reg,indent=1)+'\\n')
arm=json.loads((a.parent.parent/'collectfix/ARMED.json').read_text())
arm.update(config=str(a/'CONFIG.json'),registration=str(a/'REGISTRATION.json'),
 source_files=source['files'],native_source_manifest_sha256=source['sha256'],
 config_sha256=reg['sources']['config_sha256'],registration_sha256=hashlib.sha256((a/'REGISTRATION.json').read_bytes()).hexdigest(),
 wrapper_sha256=hashlib.sha256((r/'scripts/c4/mainstart.py').read_bytes()).hexdigest(),
 amendment='S-MAINR16 cpuQ source substitution',pairft_incarnations={{}})
(a/'ARMED.json').write_text(json.dumps(arm,indent=1)+'\\n')
doc={{'registration_sha256':arm['registration_sha256'],'config_sha256':arm['config_sha256'],
 'source_manifest_sha256':source['sha256'],'base_sha256':{SHA!r},'substitution':{substitution!r},
 'cycle0_cap':reg['budget']['sim_s_per_cycle'],'total_cap':210000}}
(a/'PROPOSAL.json').write_text(json.dumps(doc,indent=1));print(json.dumps(doc))
'''
    receipt('PROPOSAL.json', cf.remote('gpuR', code))


def tests():
    code = f'''
import os,sys,pathlib,subprocess,json,time
r=pathlib.Path({R!r});a=r.parent
sys.path[:0]=[str(r),str(r/'src'),str(r/'scripts/c4')]
import main14_autolaunch as auto
names=[str(r/'tests/unit'/n) for n in auto.TESTS]
names += [str(r/'tests/unit'/n) for n in ('test_fastloop.py','test_mainstart.py','test_collectfix.py','test_mainr16.py')]
names += [str(r/'tests'/n) for n in ('test_speed2_flags.py','test_loop14_emergency_onset.py','test_r16_gate.py','test_r16_ramp_speed_adapt.py')]
env={{**os.environ,'PYTHONPATH':str(r/'src')+':'+str(r),'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1',
 'MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','C4_ALLOWED_ROOTS':''}}
start=time.time()
with (a/'integrated_tests.log').open('wb') as log:
 p=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','--basetemp',str(a/'pytest_tmp'),*names],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT)
doc={{'exit':p.returncode,'elapsed_s':time.time()-start,'summary':(a/'integrated_tests.log').read_text()[-12000:],
 'source_manifest_sha256':json.loads((r/'COLLECTFIX_SOURCE.json').read_text())['sha256']}}
(a/'TESTS.json').write_text(json.dumps(doc,indent=1));print(json.dumps(doc));assert p.returncode==0
'''
    receipt('TESTS.json', cf.remote('gpuR', code, timeout=1800))


def prearm():
    code = f'''
import pathlib,json,sys,os,subprocess
r=pathlib.Path({R!r});a=r.parent
assert not (a/'WAIT_OWNER.json').exists(), 'owner continuation required after failed prearm'
sys.path[:0]=[str(r/'scripts/c4'),str(r),str(r/'src')]
import loop14 as loop,main14_autolaunch as auto,mainstart
cfg=loop.load_config(a/'CONFIG.json');arm=loop.read_json(a/'ARMED.json')
auto.verify_source(arm);loop.check_registration(cfg);loop.gpu_guard(cfg,loop.host(cfg,'gpu'))
ready=mainstart.collection_ready(cfg,arm,loop);assert ready['ready'],ready
base={{'checkpoint':cfg['wm']['initial_wm'],'config':cfg['wm']['initial_wm_config'],'sha256':{SHA!r}}}
state={{'accepted_wm':base,'policy':None,'real_returns':cfg['policy']['initial_real_returns'],'cycles':[{{'cycle':0}}]}}
argv,env=loop.policy_cmd(cfg,state,a/'smoke')
# Synthetic SmokeSampler has no real-data metadata. The frozen panel is hash-checked
# separately; it must only be loaded during the real policy/update-10 run.
assert loop.sha256(a/'gate_panel.json')==arm['gate_panel_sha256']
for flag in ('--gate-panel','--horizon-gate'):
 i=argv.index(flag);del argv[i:i+2]
argv[argv.index('--device')+1]='cpu';argv[argv.index('--max-updates')+1]='2';argv+=['--smoke']
auto.run(argv,cfg,a,'prearm_smoke',timeout=900,env={{**os.environ,**{{k:str(v) for k,v in env.items()}},'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'}})
summary=loop.read_json(a/'smoke/RUN_SUMMARY.json')
assert summary['status']=='completed' and len(summary['updates'])==2
auto.verify_source(arm)
doc={{'status':'PASS','updates':2,'real_wm_sha256':{SHA!r},'native_lane':True,'collector':ready,
 'source_manifest_sha256':arm['native_source_manifest_sha256'],'new_sumo_sim_s':0}}
loop.write_atomic(a/'PREARM.json',doc);print(json.dumps(doc))
'''
    result = cf.remote('gpuR', code, timeout=1200)
    receipt('PREARM.json', result.splitlines()[-1])


def migrate():
    shared = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md').read_bytes()
    proposal = json.loads((OUT/'PROPOSAL.json').read_text())
    assert proposal['registration_sha256'] in shared.decode('utf-8') and SHA in shared.decode('utf-8')
    code = f'''
import pathlib,json,sys,fcntl,hashlib,contextlib
r=pathlib.Path({R!r});a=r.parent;root=pathlib.Path('/root/main14_loop')
sys.path[:0]=[str(r/'scripts/c4'),str(r),str(r/'src')]
import loop14 as loop,main14_autolaunch as auto
from mainr16_restart import restarted_state
assert not (a/'WAIT_OWNER.json').exists(), 'owner continuation required after failed prearm'
assert loop.read_json(a/'PREARM.json')['status']=='PASS' and not (a/'MIGRATED.json').exists()
cfg=loop.load_config(a/'CONFIG.json');arm=loop.read_json(a/'ARMED.json');auto.verify_source(arm)
with contextlib.ExitStack() as stack:
 for d in (a,a.parent.parent,a.parent.parent/'collectfix'):
  f=stack.enter_context((d/'driver.lock').open('a'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=loop.read_json(root/'LOOP14_STATE.json')
 assert state['sim_s_charged']==58266 and state['seeds_used']==list(range(968103,968129))
 assert loop.sha256(state['policy']['checkpoint'])==state['policy']['sha256']
 assert loop.sha256(state['registration'])==state['registration_sha256']
 loop.write_atomic(a/'STATE_BEFORE.json',state)
 journal=root/'worlds_c00.json'; archived=root/'worlds_c00.superseded_13e.json'
 assert journal.is_file() and not archived.exists()
 oldjournal=loop.read_json(journal);assert not oldjournal['worlds']
 assert oldjournal['policy_sha256']==state['policy']['sha256']
 base={{'checkpoint':cfg['wm']['initial_wm'],'config':cfg['wm']['initial_wm_config'],
 'sha256':{SHA!r},'source':'S-MAINR16 owner-selected round16 FINAL, cpuQ provenance copy'}}
 new=restarted_state(state,registration=str(a/'REGISTRATION.json'),
 registration_sha256=loop.sha256(a/'REGISTRATION.json'),base=base,
 initial_returns=cfg['policy']['initial_real_returns'],phases=loop.PHASES)
 journal.rename(archived)
 loop.append(root/'cycle_ledger.jsonl',{{'event':'owner_cycle0_restart','id':'S-MAINR16',
 'previous_registration_sha256':state['registration_sha256'],'registration_sha256':new['registration_sha256'],
 'shared_register_sha256':{hashlib.sha256(shared).hexdigest()!r},'superseded_policy':state['policy'],
 'prior_charged_sim_s':58266,'prior_actual_sim_s':0,'prior_world_journal':str(archived),'base':base}})
 loop.save_state(cfg,new);loop.load_state(cfg)
 argv,_=loop.policy_cmd(cfg,new,root/'policy/c00.a2');assert '--resume' not in argv
 doc={{'status':'migrated','registration_sha256':new['registration_sha256'],'charged_sim_s':new['sim_s_charged'],
 'seeds_used':new['seeds_used'],'policy':new['policy'],'base':base,'superseded_journal':str(archived),
 'next_policy_output':str(root/'policy/c00.a2'),'shared_register_sha256':{hashlib.sha256(shared).hexdigest()!r}}}
 loop.write_atomic(a/'MIGRATED.json',doc);print(json.dumps(doc))
'''
    receipt('MIGRATED.json', cf.remote('gpuR', code))


def launch():
    code = f'''
import pathlib,json,os,sys,subprocess,time
r=pathlib.Path({R!r});a=r.parent
assert (a/'MIGRATED.json').is_file() and not (a/'LAUNCH.json').exists()
assert not (a/'WAIT_OWNER.json').exists(), 'owner continuation required after failed prearm'
env={{**os.environ,'C4_MAINSTART_CODE':str(r),'PYTHONPATH':str(r/'src')+':'+str(r),
 'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}}
argv=[sys.executable,'-u',str(r/'scripts/c4/mainstart.py'),'--arm',str(a/'ARMED.json')]
p=subprocess.run([*argv,'--check-only'],cwd=r,env=env,capture_output=True,text=True,timeout=300)
assert p.returncode==0,p.stdout+p.stderr
with (a/'driver.log').open('ab') as log:
 child=subprocess.Popen(['setsid','nohup',*argv,'--resume'],cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
time.sleep(5);assert child.poll() is None,(a/'driver.log').read_text()[-4000:]
f=pathlib.Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()
doc={{'pid':child.pid,'ppid':int(f[1]),'sid':int(f[3]),'start_ticks':f[19],'log':str(a/'driver.log'),
 'check_only':json.loads(p.stdout),'argv':argv+['--resume']}}
(a/'LAUNCH.json').write_text(json.dumps(doc,indent=1));print(json.dumps(doc))
'''
    receipt('LAUNCH.json', cf.remote('gpuR', code, timeout=420))


def audit_failure():
    code = f'''
import pathlib,json,hashlib,datetime,shutil
a=pathlib.Path({A!r});root=pathlib.Path('/root/main14_loop')
state=(root/'LOOP14_STATE.json').read_bytes()
before=(a.parent/'STATE_BEFORE.json').read_bytes();assert state==before
s=json.loads(state);assert s['sim_s_charged']==58266 and s['seeds_used']==list(range(968103,968129))
assert not (a/'MIGRATED.json').exists() and not (a/'LAUNCH.json').exists()
assert hashlib.sha256(pathlib.Path(s['policy']['checkpoint']).read_bytes()).hexdigest()==s['policy']['sha256']
doc={{'id':'S-MAINR16 cpuQ continuation','status':'WAIT_OWNER',
 't':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'error':"Prearm invocation failed before updates: AttributeError: 'SmokeSampler' object has no attribute 'metadata'",
 'cause':'Synthetic --smoke was incorrectly combined with --gate-panel; frozen-panel loader requires a real-data sampler',
 'log':str(a/'prearm_smoke.log'),'log_text':(a/'prearm_smoke.log').read_text(),
 'checkpoint_sha256':{SHA!r},'source_provenance_sha256_and_md5_verified':True,
 'staged_base':{BASE!r},'integrated_tests':json.loads((a/'TESTS.json').read_text()),
 'live_state_unchanged':True,'migrated':False,'launched':False,'owned_background_pids':[],
 'policy_updates':0,'update10_verified':False,'both_box_collection_started':False,
 'charged_sim_s':58266,'actual_sim_s':0,'new_seeds':[],'new_sumo_sim_s':0,
 'prior_policy_sha256_verified':s['policy']['sha256'],
 'prepared_correction':'Remove --gate-panel/--horizon-gate from synthetic smoke only; hash-check the unchanged frozen panel separately. Real policy retains both args and the update10 gate. No corrected smoke retry executed.',
 'output_free_bytes':shutil.disk_usage(root).free,'scratch_free_bytes':shutil.disk_usage(a).free}}
(a/'WAIT_OWNER.json').write_text(json.dumps(doc,indent=1)+'\\n');print(json.dumps(doc))
'''
    receipt('WAIT_OWNER.json', cf.remote('gpuR', code))


def progress():
    code = f'''
import pathlib,json,datetime,os
a=pathlib.Path({A!r});root=pathlib.Path('/root/main14_loop');policy=root/'policy/c00.a2'
def tail(p,n=3500):
 if not p.is_file():return None
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-n));return f.read().decode(errors='replace')
def read(p):return json.loads(p.read_text()) if p.is_file() else None
def identity(row):
 stat=pathlib.Path('/proc')/str(row['pid'])/'stat'
 f=stat.read_text().rsplit(')',1)[1].split() if stat.exists() else None
 return {{**row,'live':bool(f and f[0]!='Z' and f[19]==row['start_ticks']),
 'ppid':int(f[1]) if f else None,'sid':int(f[3]) if f else None}}
pid=read(a/'PID.json');state=read(root/'LOOP14_STATE.json')
procs=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace')
  if 'base64.b64decode' in cmd or not ({R!r} in cmd or str(policy) in cmd):continue
  f=(p/'stat').read_text().rsplit(')',1)[1].split()
  procs.append({{'pid':int(p.name),'ppid':int(f[1]),'sid':int(f[3]),'start_ticks':f[19],
   'cpu_ticks':int(f[11])+int(f[12]),'cmd':cmd[:650]}})
 except (OSError,PermissionError):pass
h=read(a/'horizon_{SHA}.json');hshort={{k:v for k,v in h.items() if k!='rows'}} if h else None
j=read(root/'worlds_c00.json')
doc={{'t':datetime.datetime.now(datetime.timezone.utc).isoformat(),'driver':identity(pid) if pid else None,
 'phase':state['cycles'][-1],'charged_sim_s':state['sim_s_charged'],'seeds_used':state['seeds_used'],
 'processes':procs,'driver_tail':tail(a/'driver.log'),'horizon':hshort,
 'horizon_log_tail':tail(a/'horizon_{SHA[:12]}.log',1200),'policy_files':sorted(p.name for p in policy.iterdir()) if policy.is_dir() else [],
 'policy_log_tail':tail(root/'policy/c00.a2.log'),'gate_result':read(policy/'gate_panel_result.json'),
 'summary':None,'wait_owner':read(a/'WAIT_OWNER.json'),
 'world_journal':{{'cases':j['cases'],'attempts':j['attempts'],'charged':j['charged'],'worlds_complete':len(j['worlds'])}} if j else None}}
s=read(policy/'RUN_SUMMARY.json')
if s:doc['summary']={{'status':s['status'],'updates':len(s.get('updates',[])),'error':s.get('error'),'horizon_gate':s.get('horizon_gate')}}
doc['rescores_tail']=tail(policy/'rescores.jsonl',2500)
(a/'PROGRESS.json').write_text(json.dumps(doc,indent=1));print(json.dumps(doc))
'''
    receipt('PROGRESS.json', cf.remote('gpuR', code))


def repair_horizon():
    names = subprocess.check_output(['git','ls-files','tools'],cwd=ROOT).decode().splitlines()
    files = {}
    for name in names:
        if Path(name).suffix != '.py' or any(token in name.lower() for token in cf.BLOCK):
            continue
        p = ROOT/name
        data = p.read_bytes() if p.is_file() else subprocess.check_output(['git','show',':'+name],cwd=ROOT)
        files[name] = base64.b64encode(data).decode()
    test = (ROOT/'tests/unit/test_mainr16.py').read_text()
    payload = json.dumps(files).encode()
    for box in ('gpuR','cpuP','cpuQ'):
        runtime = R if box == 'gpuR' else CPU+'/runtime'
        code = f'''
import pathlib,json,base64,hashlib,sys
r=pathlib.Path({runtime!r});assert r.is_dir()
original=json.loads((r/'COLLECTFIX_SOURCE.json').read_text())
for n,h in original['files'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
added={{}}
for name,text in json.loads(sys.stdin.buffer.read()).items():
 assert name.startswith('tools/') and name.endswith('.py') and not any(t in name.lower() for t in {cf.BLOCK!r})
 data=base64.b64decode(text);p=r/name
 if p.exists():continue
 p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 added[name]=hashlib.sha256(data).hexdigest()
assert 'tools/r10_attribution.py' in added and 'tools/wm_eval_common.py' in added
doc={{'host':{box!r},'original_manifest_sha256':original['sha256'],'added_files':added,
 'supplement_manifest_sha256':hashlib.sha256(json.dumps(added,sort_keys=True).encode()).hexdigest(),
 'reason':'first non-scientific horizon deployment error; owner-authorized dependency/invocation repair',
 'original_registered_files_unchanged':True}}
(r/'CHECK_DEPENDENCIES.json').write_text(json.dumps(doc,indent=1));print(json.dumps(doc))
'''
        result = cf.remote(box,code,payload,timeout=300)
        doc = json.loads(result)
        (OUT/('CHECK_DEPENDENCIES_'+box+'.json')).write_text(json.dumps(doc,indent=1))
        print(json.dumps({k:v for k,v in doc.items() if k!='added_files'}))
    code = f'''
import pathlib,json,sys,os,subprocess,importlib.util
r=pathlib.Path({R!r});a=r.parent
sys.path[:0]=[str(r/'scripts/c4'),str(r),str(r/'src')]
from tools.loop14_checks import accept
cfg,settings,model=accept.load_arm({BASE!r}+'/config.resolved.json',{BASE!r}+'/wm_s_best.pt')
assert model.ramp_speed_adapt_learnable and model.physics.emergency_onset=='driver_b'
from tools.wm_eval_common import delay_proxy
for name in ('tools/r10b_checks/run_onestep.py','tools/r13e_checks/offline.py','tools/r12_checks/base_calibration.py','tools/r12_checks/midrun_check.py'):
 assert (r/name).is_file(),name
env={{**os.environ,'PYTHONPATH':str(r/'src')+':'+str(r),'CUDA_VISIBLE_DEVICES':'',
 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}}
test=a/'test_horizon_resolved.py';test.write_text({test!r})
p=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(test),'-k','horizon_loader_accepts_resolved'],cwd=r,env=env,capture_output=True,text=True)
assert p.returncode==0,p.stdout+p.stderr
doc={{'status':'PASS','resolved_json_strict_checkpoint_load':True,'test':p.stdout,'missing_tensor_names':model.loop14_new_tensors}}
(a/'CHECK_DEPENDENCY_PREFLIGHT.json').write_text(json.dumps(doc,indent=1));print(json.dumps(doc))
'''
    receipt('CHECK_DEPENDENCY_PREFLIGHT.json', cf.remote('gpuR',code,timeout=300))
    wrapper = (ROOT/'tools/mainstart/mainr16_horizon_retry.py').read_text()
    code = f'''
import pathlib,json,sys,os,subprocess,time
a=pathlib.Path({A!r});assert (a/'CHECK_DEPENDENCY_PREFLIGHT.json').is_file()
p=a/'mainr16_horizon_retry.py';assert not p.exists();p.write_text({wrapper!r})
env={{**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'}}
with (a/'horizon_retry_driver.log').open('xb') as log:
 child=subprocess.Popen(['setsid','nohup',sys.executable,'-u',str(p)],cwd=a,env=env,
 stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
time.sleep(2)
f=pathlib.Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()
doc={{'pid':child.pid,'start_ticks':f[19],'sid':int(f[3]),'log':str(a/'horizon_retry1.log'),'retry':1}}
(a/'HORIZON_RETRY_PID.json').write_text(json.dumps(doc,indent=1));print(json.dumps(doc))
'''
    receipt('HORIZON_RETRY_PID.json',cf.remote('gpuR',code))


def watch():
    code = f'''
import pathlib,json,time,datetime
a=pathlib.Path({A!r});root=pathlib.Path('/root/main14_loop');policy=root/'policy/c00.a2'
rows=[]
for i in range(46):
 s=json.loads((root/'LOOP14_STATE.json').read_text());cyc=s['cycles'][-1]
 phase=next((k for k,v in cyc['phases'].items() if v['status']!='done'),None)
 updates=0;p=policy/'updates.jsonl'
 if p.exists():
  with p.open() as f:
   for line in f:
    try:updates=max(updates,int(json.loads(line).get('update',0)))
    except (ValueError,TypeError):pass
 row={{'t':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cycle':cyc['cycle'],'phase':phase,
 'updates':updates,'horizon_repaired':(a/'HORIZON_REPAIRED.json').exists(),
 'wait_owner':(a/'WAIT_OWNER.json').exists(),'charged_sim_s':s['sim_s_charged']}}
 p=a/'horizon_retry1.log'
 if p.exists():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-500));row['horizon_tail']=f.read().decode(errors='replace')[-200:]
 rows.append(row)
 if i%3==0 or row['wait_owner'] or phase!='policy':print(json.dumps(row),flush=True)
 (a/'WATCH.json').write_text(json.dumps(rows,indent=1))
 if row['wait_owner'] or phase!='policy':break
 if i<45:time.sleep(20)
print(json.dumps({{'last':rows[-1],'samples':len(rows)}}))
'''
    receipt('WATCH.jsonl',cf.remote('gpuR',code,timeout=1020))


def collectors():
    code = f'''
import pathlib,json,sys,subprocess,shlex,datetime,hashlib
a=pathlib.Path({A!r});root=pathlib.Path('/root/main14_loop')
state=json.loads((root/'LOOP14_STATE.json').read_text());journal=root/'worlds_c00.json'
doc={{'t':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':state['cycles'][-1],
 'charged_sim_s':state['sim_s_charged'],'seeds_used':state['seeds_used'],'boxes':{{}}}}
probe=r"""
import pathlib,json
root=pathlib.Path('/home/ubuntu/c4/main14/mainr16')
processes=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace')
  if 'base64.b64decode' in cmd or '-c import' in cmd or str(root) not in cmd:continue
  f=(p/'stat').read_text().rsplit(')',1)[1].split()
  processes.append(dict(pid=int(p.name),ppid=int(f[1]),start_ticks=f[19],cpu_ticks=int(f[11])+int(f[12]),cmd=cmd[:850]))
 except (OSError,PermissionError):pass
episodes=[];summaries=[]
for p in (root/'work').glob('w*/C4-LOOP14-COOPON-20261026/c00/collect/episodes/policy/*/episode.npz'):
 episodes.append(dict(path=str(p),bytes=p.stat().st_size))
for p in (root/'work').glob('w*/C4-LOOP14-COOPON-20261026/c00/collect/COLLECT_SUMMARY.json'):
 s=json.loads(p.read_text());summaries.append(dict(path=str(p),episodes_complete=s.get('episodes_complete'),
 episodes_failed=s.get('episodes_failed'),simulated_seconds=s.get('simulated_seconds'),pair_sim_seconds=s.get('pair_sim_seconds')))
print(json.dumps(dict(processes=processes,policy_episodes=episodes,summaries=summaries)))
"""
for box in ('cpuP','cpuQ'):
 cmd=shlex.join(['/home/ubuntu/miniconda3/bin/python','-c',probe])
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',box,cmd],capture_output=True,text=True,timeout=90)
 assert p.returncode==0,p.stderr;doc['boxes'][box]=json.loads(p.stdout)
if journal.is_file():
 j=json.loads(journal.read_text());doc['journal']={{'cases':j['cases'],'attempts':j['attempts'],
 'charged':j['charged'],'worlds':j['worlds'],'policy_sha256':j['policy_sha256'],'wm_sha256':j['wm_sha256']}}
h=a/'HORIZON_REPAIRED.json';doc['horizon_repaired']=json.loads(h.read_text()) if h.is_file() else None
doc['wait_owner']=json.loads((a/'WAIT_OWNER.json').read_text()) if (a/'WAIT_OWNER.json').is_file() else None
(a/'COLLECTOR_PROGRESS.json').write_text(json.dumps(doc,indent=1));print(json.dumps(doc))
'''
    receipt('COLLECTOR_PROGRESS.json',cf.remote('gpuR',code,timeout=240))


def wait_horizon():
    code = f'''
import pathlib,json,time
a=pathlib.Path({A!r})
for i in range(46):
 if (a/'WAIT_OWNER.json').exists():
  print((a/'WAIT_OWNER.json').read_text());break
 if (a/'HORIZON_REPAIRED.json').exists():
  print((a/'HORIZON_REPAIRED.json').read_text());break
 if i%6==0:
  p=a/'horizon_retry1.log'
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-160));print(json.dumps({{'waiting_s':i*20,'tail':f.read().decode(errors='replace')}}),flush=True)
 if i<45:time.sleep(20)
else: print(json.dumps({{'status':'still_running'}}))
'''
    receipt('HORIZON_WATCH.jsonl',cf.remote('gpuR',code,timeout=1020))


def final_audit():
    code = f'''
import pathlib,json,sys,fcntl,shutil,datetime
a=pathlib.Path({A!r});r=a/'runtime';root=pathlib.Path('/root/main14_loop')
sys.path[:0]=[str(r/'scripts/c4'),str(r),str(r/'src')]
import loop14 as loop,main14_autolaunch as auto
arm=loop.read_json(a/'ARMED.json');auto.verify_source(arm)
cfg=loop.load_config(a/'CONFIG.json');state=loop.load_state(cfg)
pid=loop.read_json(a/'PID.json');f=(pathlib.Path('/proc')/str(pid['pid'])/'stat').read_text().rsplit(')',1)[1].split()
assert f[0]!='Z' and f[19]==pid['start_ticks'] and int(f[1])==1 and int(f[3])==pid['pid']
lock_held=False
with (a/'driver.lock').open('a') as lock:
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:lock_held=True
assert lock_held
assert not (a/'WAIT_OWNER.json').exists()
assert loop.sha256(cfg['wm']['initial_wm'])=={SHA!r}
assert loop.sha256(a/'gate_panel.json')==arm['gate_panel_sha256']
h=loop.read_json(a/'HORIZON_REPAIRED.json');assert h['status']=='PASS' and h['wm_sha256']=={SHA!r}
gate=loop.read_json(root/'policy/c00.a2/gate_panel_result.json');assert gate['passed'] and gate['update']==10
summary=loop.read_json(root/'policy/c00.a2/RUN_SUMMARY.json');assert summary['status']=='completed' and len(summary['updates'])==150
assert summary.get('resumed_from') is None
prior=state['superseded_policies'][0];assert loop.sha256(prior['checkpoint'])==prior['sha256']
col=loop.read_json(a/'COLLECTOR_PROGRESS.json')
boxes={{}}
for box,b in col['boxes'].items():
 procs=[p for p in b['processes'] if 'traffic_shepherd.c4.collect' in p['cmd']]
 completed=[s for s in b['summaries'] if s['episodes_complete']==1 and s['episodes_failed']==0]
 assert (procs or completed) and b['policy_episodes'], 'actual collector/recording not observed on '+box
 assert len(procs)<=7
 boxes[box]={{'collector_processes_observed':len(procs),'pids':[{{'pid':p['pid'],'start_ticks':p['start_ticks']}} for p in procs],
 'recorded_policy_episodes':len(b['policy_episodes']),'completed_worlds':len(completed),
 'actual_sim_s':sum(s['pair_sim_seconds'] for s in completed)}}
supp=loop.read_json(r/'CHECK_DEPENDENCIES.json')
for name,sha in supp['added_files'].items():assert loop.sha256(r/name)==sha,name
j=loop.read_json(root/'worlds_c00.json')
assert j['wm_sha256']=={SHA!r} and j['policy_sha256']==loop.sha256(root/'policy/c00.a2/policy_latest.pt')
assert [c['seed'] for c in j['cases']]==list(range(968129,968142))
assert state['sim_s_charged']<=210000 and len(state['seeds_used'])==len(set(state['seeds_used']))
doc={{'status':'RUNNING_UNATTENDED','t':loop.now(),'driver':{{**pid,'ppid':int(f[1]),'sid':int(f[3]),'lock_held':lock_held}},
 'policy_updates':150,'policy_wm_sha256':{SHA!r},'policy_sha256':j['policy_sha256'],'superseded_policy_sha256':prior['sha256'],
 'policy_started':state['cycles'][0]['phases']['policy']['started'],'policy_finished':state['cycles'][0]['phases']['policy']['finished'],
 'update10':{{'passed':gate['passed'],'n_improved':gate['n_improved'],'n':gate['n'],'mean_delta':gate['mean_delta']}},
 'horizon':h,'collectors_observed':boxes,'collector_observation_time':col['t'],
 'charged_sim_s':state['sim_s_charged'],'total_cap':210000,'prior_failed_charge_sim_s':58266,'prior_failed_actual_sim_s':0,
 'actual_sim_s':sum(b['actual_sim_s'] for b in boxes.values()),
 'actual_sim_s_basis':'completed per-world pair_sim_seconds already includes base episode plus all branch/credit/gate SUMO; do not add simulated_seconds again',
 'new_seeds':[c['seed'] for c in j['cases']],'registered_source_manifest_sha256':arm['native_source_manifest_sha256'],
 'supplement_manifest_sha256':supp['supplement_manifest_sha256'],'supplement_files':len(supp['added_files']),
 'registration_sha256':state['registration_sha256'],'config_sha256':loop.sha256(a/'CONFIG.json'),
 'current_cycle':state['cycles'][-1],'output_free_bytes':shutil.disk_usage(root).free,'scratch_free_bytes':shutil.disk_usage(a).free,
 'wait_owner_present':False}}
wm_config=root/'wm/c00/config.resolved.json'
if wm_config.is_file():
 c=loop.read_json(wm_config)['config']
 assert c['model']['ramp_speed_adapt_learnable'] and 'ramp_speed_adapt' in c['optim']['freeze_prefixes']
 doc['wm_ft_round16']={{'ramp_speed_adapt_learnable':True,'ramp_scalars_frozen':True,
 'init_from':c['model']['init_from'],'emergency_onset':c['physics']['emergency_onset']}}
loop.write_atomic(a/'FINAL_RUNNING.json',doc);print(json.dumps(doc))
'''
    receipt('FINAL_RUNNING.json',cf.remote('gpuR',code,timeout=180))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('authorize','probe','stage-base','deploy','prepare','tests','prearm','migrate','launch','audit-failure','progress','repair-horizon','watch','collectors','wait-horizon','final-audit'))
    args = p.parse_args()
    {'authorize':authorize,'probe': probe,'stage-base':stage_base,'deploy':deploy,'prepare':prepare,'tests':tests,
     'prearm':prearm,'migrate':migrate,'launch':launch,'audit-failure':audit_failure,'progress':progress,
     'repair-horizon':repair_horizon,'watch':watch,'collectors':collectors,'wait-horizon':wait_horizon,
     'final-audit':final_audit}[args.action]()
