"""Sequential S-C1COLLECT staging, registration, launch and compact receipts."""
import argparse
import base64

from tools.loopresume.operate import ROOT, remote

RELEASE = '/root/autodl-tmp/c4/restart20261026'
FILES = ['scripts/c4/loop14_fast.py', 'tools/loopresume/c1collect_box.py', 'tests/unit/test_c1collect.py']
PRELUDE = f'''
import pathlib,json,sys,os,subprocess,hashlib,shutil,fcntl,time
r=pathlib.Path({RELEASE!r}); root=r/'runtime_c1'; a=r/'arm'
sys.path[:0]=[str(root),str(root/'src'),str(root/'scripts/c4')]
import loop14 as loop
from tools.loopresume import continue_c1 as previous
'''


def stage():
    payload = {n: base64.b64encode((ROOT/n).read_bytes()).decode() for n in FILES}
    remote('gpuA', PRELUDE+f'''
import base64
old_pid=loop.read_json(a/'PID.json'); stat=pathlib.Path('/proc/'+str(old_pid['pid'])+'/stat')
assert not stat.exists() or stat.read_text().rsplit(')',1)[1].split()[19]!=old_pid['start_ticks']
assert not (r/'C1COLLECT_BINDINGS.json').exists()
backup=r/'c1collect_before'; backup.mkdir(exist_ok=True)
if (r/'C1COLLECT_TESTS.json').exists():
 assert loop.read_json(r/'C1COLLECT_TESTS.json')['exit']!=0
 revision=1
 while (backup/f'tests_failed_{{revision}}.json').exists(): revision+=1
 shutil.copy2(r/'C1COLLECT_TESTS.json',backup/f'tests_failed_{{revision}}.json')
 shutil.copy2(r/'C1COLLECT_TESTS.log',backup/f'tests_failed_{{revision}}.log')
manifest={{}}
for name,encoded in {payload!r}.items():
 p=root/name; assert p.parent.is_dir()
 if p.exists():
  dest=backup/name; dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists(): shutil.copy2(p,dest)
 p.write_bytes(base64.b64decode(encoded)); manifest[name]=loop.sha256(p)
loop.write_atomic(r/'C1COLLECT_SOURCE.json',manifest)
env={{**os.environ,'PYTHONPATH':str(root/'src')+':'+str(root),'CUDA_VISIBLE_DEVICES':'',
 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1',
 'NATIVEINIT_BASELINE':str(r/'nativeinit_baseline')}}
env.pop('C4_ALLOWED_ROOTS',None)
args=['/root/miniconda3/bin/python','-B','-m','pytest','-q','-ra','-p','no:cacheprovider','--import-mode=importlib',
 '--basetemp',str(r/'pytest_c1collect'),'tests/unit/test_c1collect.py','tests/unit/test_loopresume.py',
 'tests/unit/test_fastloop.py','tests/unit/test_nativeinit.py','tests/unit/test_loop14.py']
p=subprocess.run(args,cwd=root,env=env,capture_output=True,text=True)
(r/'C1COLLECT_TESTS.log').write_text(p.stdout+p.stderr); print(p.stdout+p.stderr)
doc={{'t':loop.now(),'exit':p.returncode,'source_sha256':loop.sha256(r/'C1COLLECT_SOURCE.json'),
 'log_sha256':loop.sha256(r/'C1COLLECT_TESTS.log')}}
loop.write_atomic(r/'C1COLLECT_TESTS.json',doc); print(json.dumps(doc)); raise SystemExit(p.returncode)
''', timeout=180)


def install():
    remote('gpuA', PRELUDE+'''
with (a/'cycle1.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not previous.live(loop.read_json(a/'PID.json'))
 tests=loop.read_json(r/'C1COLLECT_TESTS.json'); assert tests['exit']==0
 source=loop.read_json(r/'C1COLLECT_SOURCE.json')
 assert tests['source_sha256']==loop.sha256(r/'C1COLLECT_SOURCE.json')
 for name,sha in source.items(): assert loop.sha256(root/name)==sha
 cfg=loop.load_config(a/'CONFIG.json'); state=loop.load_state(cfg); reg=loop.check_registration(cfg)
 journal=loop.read_json(a/'loop/worlds_c01.json')
 assert state['sim_s_charged']==136699 and reg['budget']['sim_s_total']==210000
 assert state['cycles'][-1]['cycle']==1 and state['cycles'][-1]['phases']['collect']['status']=='pending'
 assert state['cycles'][-1]['phases']['policy']['status']=='done' and not state['stopped']
 assert not journal['worlds'] and journal['charged']==34092
 assert [c['seed'] for c in journal['cases']]==list(range(968157,968170))
 assert state['policy']['sha256']=='64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c'
 from wms.control.nativeinit import verify_collection_gate
 gate=verify_collection_gate(state['policy']); assert gate['passed']
 assert gate['wm_sha256']==[state['policy_wm']['sha256']]
 assert gate['wm_config_sha256']==[loop.sha256(state['policy_wm']['config'])]
 amendment={'id':'S-C1COLLECT','authority':'main decision 2026-10-27 05:20 UTC+8; owner act-on-recommendation',
  'registered_at':loop.now(),'cycle':1,'attempt_allowance_sim_s':71025,'long_credit_allowance_sim_s':15000,
  'total_cap_sim_s':210000,'charged_before':136699,'pre_SUMO_loss_charged_sim_s':34092,
  'included_long_credit_loss_sim_s':7200,'new_collection_attempts':1,'no_refunds':True,
  'seeds':list(range(968157,968170)),'policy':state['policy'],'source':source,'tests':tests,
  'canary':'one world at a time; verify actual SUMO plus numerical outputs on cpuP and cpuQ before remainder',
  'failure':'OPERATIONAL_PAUSED; main decides; no automatic collection retry',
  'guard':'existing four-world bootstrap; stop only HARMFUL, lower CI > 0; inconclusive continues',
  'independent_wm0_pids_untouched':[62645,62910],
  'previous_registration_sha256':loop.sha256(cfg['registration'])}
 for name in ('CONFIG.json','REGISTRATION.json','loop/LOOP14_STATE.json','PID.json','OPERATIONAL_PAUSED.json'):
  p=a/name
  if p.exists():
   dest=r/'c1collect_before'/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dest)
 # Append prospective effective fields without rewriting any historical amendment.
 reg.setdefault('amendments',[]).append(amendment)
 reg['budget'].setdefault('cycle_attempt_allowances',{})['1']=71025
 reg['collection'].setdefault('cycle_long_attempt_allowances',{})['1']=15000
 cfg['collect']['startup_canary']={'cycle':1,'authority':'S-C1COLLECT',
  'next_attempts':{str(c['seed']):journal['attempts'].get(str(c['seed']),0)+1 for c in journal['cases']}}
 assert cfg['collect']['native_smoke_harmful_only_from_cycle']==1
 loop.write_atomic(a/'CONFIG.json',cfg)
 reg['sources']['config_sha256']=loop.sha256(a/'CONFIG.json')
 loop.write_atomic(pathlib.Path(cfg['registration']),reg)
 state['registration_sha256']=loop.sha256(cfg['registration']); loop.save_state(cfg,state)
 binding={**amendment,'registration_sha256':state['registration_sha256'],'config_sha256':loop.sha256(a/'CONFIG.json')}
 loop.write_atomic(r/'C1COLLECT_BINDINGS.json',binding)
 previous.event('S-C1COLLECT_amendment_installed',**binding)
 loop.append(a/'loop/cycle_ledger.jsonl',{'event':'S-C1COLLECT_amendment_installed',**binding})
 print(json.dumps(binding))
''', timeout=120)


def launch():
    remote('gpuA', PRELUDE+'''
with (a/'cycle1.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 binding=loop.read_json(r/'C1COLLECT_BINDINGS.json')
 assert loop.sha256(a/'CONFIG.json')==binding['config_sha256']
 assert loop.sha256(a/'REGISTRATION.json')==binding['registration_sha256']
 assert not previous.live(loop.read_json(a/'PID.json'))
 assert not (r/'C1COLLECT_ATTEMPT.json').exists()
 for name in ('PID.json','OPERATIONAL_PAUSED.json'):
  p=a/name
  if p.exists(): p.rename(p.with_name(p.name+'.before_c1collect'))
 with (a/'driver_c1collect.log').open('xb') as log:
  env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
  proc=subprocess.Popen([sys.executable,'-B','-u',str(root/'tools/loopresume/c1collect_box.py')],
   cwd=root,env=env,start_new_session=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 pid={'pid':proc.pid,'start_ticks':loop.fast_module().process_token(proc.pid),'t':loop.now(),'task':'S-C1COLLECT'}
 loop.write_atomic(a/'PID.json',pid); loop.write_atomic(r/'C1COLLECT_LAUNCH.json',pid)
 print(json.dumps(pid))
''')


def status():
    remote('gpuA', PRELUDE+'''
cfg=loop.load_config(a/'CONFIG.json'); state=loop.load_state(cfg); journal=loop.read_json(a/'loop/worlds_c01.json')
ph=state['cycles'][1]['phases']; pid=loop.read_json(a/'PID.json')
canaries=journal.get('startup_canaries',{})
probes={}
from tools.loopresume.c1collect_box import probe
for box in ('cpuP','cpuQ'):
 candidates=[(c,journal['attempts'].get(str(c['seed']),0)) for c in journal['cases']]
 candidates=[(c,n) for c,n in candidates if n>=4 and (c['box'] if n%2 else ('cpuQ' if c['box']=='cpuP' else 'cpuP'))==box]
 for c,n in candidates:
  probes[str(c['seed'])]=dict(box=box,seed=c['seed'],attempt=n,**probe(cfg,c,box,n))
doc={'t':loop.now(),'pid':pid,'live':previous.live(pid),'charged':state['sim_s_charged'],
 'cycle_charged':journal['charged'],'phase_status':{k:v['status'] for k,v in ph.items()},
 'main_active':any(c['phases'][n]['status']=='running' for c in state['cycles'] for n in ('collect','check')),
 'completed_worlds':list(journal['worlds']),'canaries':canaries,'probes':probes,
 'guard':journal.get('native_smoke'),'stopped':state.get('stopped'),
 'wm0_status':loop.read_json(r/'LOOPRESUME_WM_ONLY.json')['status']}
for name in ('OPERATIONAL_PAUSED.json','WAIT_OWNER.json'):
 if (a/name).exists(): doc[name]=loop.read_json(a/name)
loop.write_atomic(r/'C1COLLECT_STATUS.json',doc)
print(json.dumps(doc)); print('LOG', (a/'driver_c1collect.log').read_text()[-4000:])
''', timeout=180)


def final_checks():
    names = ['tools/loopresume/c1collect_box.py', 'tests/unit/test_c1collect.py']
    payload = {n: base64.b64encode((ROOT/n).read_bytes()).decode() for n in names}
    remote('gpuA', PRELUDE+f'''
import base64
assert not previous.live(loop.read_json(a/'PID.json'))
assert (a/'OPERATIONAL_PAUSED.json').exists()
manifest={{}}
for name,encoded in {payload!r}.items():
 p=root/name; backup=r/'c1collect_before'/('before_final_'+p.name); assert not backup.exists()
 shutil.copy2(p,backup); p.write_bytes(base64.b64decode(encoded)); manifest[name]=loop.sha256(p)
env={{**os.environ,'PYTHONPATH':str(root/'src')+':'+str(root),'CUDA_VISIBLE_DEVICES':'',
 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1',
 'NATIVEINIT_BASELINE':str(r/'nativeinit_baseline')}}
env.pop('C4_ALLOWED_ROOTS',None)
args=['/root/miniconda3/bin/python','-B','-m','pytest','-q','-p','no:cacheprovider','--import-mode=importlib',
 '--basetemp',str(r/'pytest_c1collect_final'),'tests/unit/test_c1collect.py','tests/unit/test_loopresume.py',
 'tests/unit/test_fastloop.py','tests/unit/test_nativeinit.py','tests/unit/test_loop14.py']
p=subprocess.run(args,cwd=root,env=env,capture_output=True,text=True)
(r/'C1COLLECT_FINAL_TESTS.log').write_text(p.stdout+p.stderr); print(p.stdout+p.stderr)
doc={{'t':loop.now(),'exit':p.returncode,'source':manifest,'log_sha256':loop.sha256(r/'C1COLLECT_FINAL_TESTS.log'),
 'repair':'startup probe consumes configured ssh argv list; reporting-only repair, NO redispatch'}}
loop.write_atomic(r/'C1COLLECT_FINAL_TESTS.json',doc)
previous.event('S-C1COLLECT_final_offline_checks',**doc)
print(json.dumps(doc)); raise SystemExit(p.returncode)
''', timeout=180)


def finalize():
    remote('gpuA', PRELUDE+'''
from tools.loopresume.c1collect_box import probe
cfg=loop.load_config(a/'CONFIG.json'); state=loop.load_state(cfg); journal=loop.read_json(a/'loop/worlds_c01.json')
assert state['sim_s_charged']==139540 and journal['charged']==36933
assert not previous.live(loop.read_json(a/'PID.json')) and (a/'OPERATIONAL_PAUSED.json').exists()
assert state['cycles'][1]['phases']['collect']['status']=='failed'
proofs={}
for box in ('cpuP','cpuQ'):
 h=loop.host(cfg,box)
 code="""import pathlib,json,subprocess
base=pathlib.Path('/home/ubuntu/c4/main14/restart20261026/work')
active=[]
for row in subprocess.check_output(['ps','--no-headers','-eo','pid,ppid,args'],text=True).splitlines():
 f=row.strip().split(None,2)
 if len(f)!=3: continue
 args=f[2].split()
 if 'traffic_shepherd.c4.collect' in args and '--out' in args:
  out=args[args.index('--out')+1]
  if out.startswith(str(base)+'/w') and '/C4-LOOP14-COOPON-20261026/c01' in out: active.append({'pid':int(f[0]),'out':out})
p=base/'w968158/C4-LOOP14-COOPON-20261026/c01.a4/collect'
d={'active_collectors':active,'canary_path_exists':p.exists()}
if p.exists():
 d['records']=[json.loads(line) for line in (p/'collect_log.jsonl').read_text().splitlines()]
 d['summary']=json.loads((p/'COLLECT_SUMMARY.json').read_text())
print(json.dumps(d))
"""
 result=subprocess.run([*h['ssh'],h['python']+' -B -'],input=code,text=True,capture_output=True,check=True,timeout=60)
 proofs[box]=json.loads(result.stdout); assert not proofs[box]['active_collectors']
assert proofs['cpuP']['summary']['simulated_seconds']==0
assert not proofs['cpuQ']['canary_path_exists']
canary=probe(cfg,{'seed':968158},'cpuP',4); assert not canary['verified']
ledger=[json.loads(line) for line in (a/'loop/seed_ledger.jsonl').read_text().splitlines()]
new=[row for row in ledger if row.get('cycle')==1 and row.get('event')=='world_attempt' and row.get('attempt')==4]
assert len(new)==1 and new[0]['seed']==968158 and new[0]['sim_s_charged']==2841
long_charged=sum(row.get('long_credit_sim_s_charged',0) for row in ledger if row.get('cycle')==1)
assert long_charged==7800
doc={'id':'S-C1COLLECT','t':loop.now(),'status':'OPERATIONAL_PAUSED','driver':loop.read_json(a/'PID.json'),
 'driver_live':False,'charged':139540,'total_cap':210000,'cycle_charged':36933,'cycle_allowance':71025,
 'long_credit_charged':long_charged,'long_credit_allowance':15000,'historical_pre_SUMO_loss':34092,
 'canary_charge':2841,'new_world_attempts':new,'completed_worlds':0,'guard':None,
 'per_world_policy_minus_native_veh_s':None,'mean_veh_s':None,'ci95_veh_s':None,
 'guard_unavailable_reason':'No completed paired starts; first canary failed before its first simulation frame',
 'primary_blocker':'runtime_c1/baselines/circles/model.onnx missing; env.reset reached SUMO startup but failed before recorder initialization/stepping',
 'secondary_blocker':'startup probe passed configured ssh argv list as a scalar; fixed offline after pause, no redispatch',
 'remaining_dispatched':False,'main_active':False,'main_flag_was_active_during_attempt':True,
 'main_flag_basis':'collect status running from 21:30:10Z to21:30:23Z; existing priority watcher derives active from collect/check running',
 'cpu_evidence':proofs,'canary_probe':canary,'pause':loop.read_json(a/'OPERATIONAL_PAUSED.json'),
 'wm0_status':loop.read_json(r/'LOOPRESUME_WM_ONLY.json')['status'],
 'final_checks':loop.read_json(r/'C1COLLECT_FINAL_TESTS.json'),'renewed_attempt_consumed':True,
 'next_action':'main decides; no further collection authority; no retry/refund'}
loop.write_atomic(r/'C1COLLECT_RESULT.json',doc)
previous.event('S-C1COLLECT_final_paused_result',**doc)
loop.append(a/'loop/cycle_ledger.jsonl',{'event':'S-C1COLLECT_final_paused_result',**doc})
reg=loop.read_json(cfg['registration'])
reg.setdefault('operational_events',[]).append({'id':'S-C1COLLECT_result','t':doc['t'],
 'result':str(r/'C1COLLECT_RESULT.json'),'result_sha256':loop.sha256(r/'C1COLLECT_RESULT.json'),
 'status':'OPERATIONAL_PAUSED','charged':139540,'new_attempts':1,'no_redispatch':True})
loop.write_atomic(pathlib.Path(cfg['registration']),reg)
state['registration_sha256']=loop.sha256(cfg['registration']); loop.save_state(cfg,state)
previous.status()
print('FINAL_RESULT',json.dumps(doc))
''', timeout=180)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['stage', 'install', 'launch', 'status', 'final-checks', 'finalize'])
    globals()[parser.parse_args().action.replace('-', '_')]()
