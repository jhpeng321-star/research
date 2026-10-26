"""Sequential S-C1COLLECT3 operations, bound to the one renewed main-authorized attempt."""
import argparse
import base64
from pathlib import Path
import subprocess

from tools.loopresume.c1collect import PRELUDE
from tools.loopresume.operate import ROOT, remote


def manifest():
    import hashlib
    data = subprocess.check_output(['git', 'show', 'HEAD:assets/research/migration_manifest.json'], cwd=ROOT)
    blob = subprocess.check_output(['git', 'rev-parse', 'HEAD:assets/research/migration_manifest.json'],
                                   cwd=ROOT, text=True).strip()
    sha = hashlib.sha256(data).hexdigest()
    for box in ('cpuP', 'cpuQ'):
        remote(box, f'''
import pathlib,base64,hashlib,json
b=pathlib.Path('/home/ubuntu/c4/main14/restart20261026')
p=b/'runtime_c1/assets/research/migration_manifest.json'
assert p.parent.is_dir()
data=base64.b64decode({base64.b64encode(data).decode()!r})
assert hashlib.sha256(data).hexdigest()=={sha!r}
if p.exists(): assert p.read_bytes()==data
else: p.write_bytes(data)
assert hashlib.sha256(p.read_bytes()).hexdigest()=={sha!r}
doc={{'path':str(p),'sha256':{sha!r},'bytes':len(data),'verified':True,
 'original':'git HEAD:assets/research/migration_manifest.json','git_blob':{blob!r},'content_changed':False}}
(b/'C1COLLECT3_MANIFEST.json').write_text(json.dumps(doc,indent=2))
print(json.dumps(doc))
''')


def preflight():
    source = Path(__file__).with_name('c1collect3_cpu.py').read_text()
    remote('gpuA', PRELUDE+rf'''
import base64
cfg=loop.load_config(a/'CONFIG.json'); state=loop.load_state(cfg)
assert state['sim_s_charged']==139540 and not previous.live(loop.read_json(a/'PID.json'))
j=loop.read_json(a/'loop/worlds_c01.json'); acc=state['policy_wm']; pol=state['policy']
results={{}}
for box in ('cpuP','cpuQ'):
 one=json.loads(json.dumps(cfg)); one['hosts']['cpu']=one['hosts'][box]
 stage='/home/ubuntu/c4/main14/restart20261026/preflight_c1collect3'
 files=[('policy.pt',pol['checkpoint']),('native_gate_result.json',str(pathlib.Path(pol['checkpoint']).with_name('native_gate_result.json'))),
  ('wm/wm_s.pt',acc['checkpoint']),('wm/config.json',acc['config']),('REGISTRATION.json',cfg['registration'])]
 for name in ('physics.json','TRAIN_RECEIPT.json','lane_calib.json'):
  p=pathlib.Path(acc['checkpoint']).with_name(name)
  if p.is_file(): files.append(('wm/'+name,str(p)))
 for name,src in files: loop.sync(one,'gpu',src,'cpu',stage+'/'+name,r/'C1COLLECT3_STAGE.log')
 binding={{'policy_sha256':pol['sha256'],'wm_sha256':acc['sha256'],'wm_config_sha256':loop.sha256(acc['config'])}}
 h=loop.host(one,'cpu')
 setup="import pathlib,json; pathlib.Path("+repr(stage+'/BINDINGS.json')+").write_text("+repr(json.dumps(binding))+")\n"
 code=setup+"__name__='c1collect3_cpu'\n"+base64.b64decode({base64.b64encode(source.encode()).decode()!r}).decode()+"\npreflight("+repr(j['cases'])+")\n"
 env="import os; os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')\n"
 result=subprocess.run([*h['ssh'],h['python']+' -B -'],input=env+code,text=True,capture_output=True,timeout=180)
 if result.returncode:
  doc={{'status':'FAIL','error':result.stderr,'stdout':result.stdout,'zero_sumo':True,'charged_sim_s':0}}
 else: doc=json.loads(result.stdout)
 results[box]=doc; loop.write_atomic(r/('C1COLLECT3_PREFLIGHT_'+box+'.json'),doc)
 print(box,json.dumps(doc),result.stderr)
assert loop.load_state(cfg)['sim_s_charged']==139540
loop.write_atomic(r/'C1COLLECT3_PREFLIGHT.json',results)
print('BOTH_PASS',all(v['status']=='PASS' for v in results.values()))
''', timeout=550)


def stage():
    names = ['tools/loopresume/c1collect3.py', 'tools/loopresume/c1collect3_cpu.py',
             'tools/loopresume/c1collect3_patch.py', 'tools/loopresume/c1collect3_box.py',
             'tests/unit/test_c1collect3.py']
    payload = {name: base64.b64encode((ROOT/name).read_bytes()).decode() for name in names}
    remote('gpuA', PRELUDE+rf'''
import base64
assert not previous.live(loop.read_json(a/'PID.json'))
assert all(loop.read_json(r/('C1COLLECT3_PREFLIGHT_'+box+'.json'))['status']=='PASS' for box in ('cpuP','cpuQ'))
backup=r/'c1collect3_before'; backup.mkdir(exist_ok=False)
source={{}}
for name,encoded in {payload!r}.items():
 dest=root/name; assert dest.parent.is_dir() and not dest.exists()
 dest.write_bytes(base64.b64decode(encoded)); source[name]=loop.sha256(dest)
from tools.loopresume.c1collect3_patch import single_canary_source
p=root/'scripts/c4/loop14_fast.py'
shutil.copy2(p,backup/'loop14_fast.py')
before=loop.sha256(p); p.write_text(single_canary_source(p.read_text()))
source['scripts/c4/loop14_fast.py']=loop.sha256(p)
loop.write_atomic(r/'C1COLLECT3_SOURCE.json',source)
loop.write_atomic(r/'C1COLLECT3_PATCH.json',{{'before_sha256':before,'after_sha256':source['scripts/c4/loop14_fast.py'],
 'change':"for box_name in canary.get('boxes', ('cpuP', 'cpuQ')):; config boxes=['cpuP']; all other source retained"}})
env={{**os.environ,'PYTHONPATH':str(root/'src')+':'+str(root),'CUDA_VISIBLE_DEVICES':'',
 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1',
 'NATIVEINIT_BASELINE':str(r/'nativeinit_baseline')}}
env.pop('C4_ALLOWED_ROOTS',None)
args=[sys.executable,'-B','-m','pytest','-q','-ra','-p','no:cacheprovider','--import-mode=importlib',
 '--basetemp',str(r/'pytest_c1collect3'),'tests/unit/test_c1collect3.py','tests/unit/test_c1collect.py',
 'tests/unit/test_loopresume.py','tests/unit/test_fastloop.py','tests/unit/test_nativeinit.py','tests/unit/test_loop14.py']
result=subprocess.run(args,cwd=root,env=env,capture_output=True,text=True)
(r/'C1COLLECT3_TESTS.log').write_text(result.stdout+result.stderr)
doc={{'t':loop.now(),'exit':result.returncode,'source_sha256':loop.sha256(r/'C1COLLECT3_SOURCE.json'),
 'log_sha256':loop.sha256(r/'C1COLLECT3_TESTS.log')}}
loop.write_atomic(r/'C1COLLECT3_TESTS.json',doc)
print(result.stdout+result.stderr); print(json.dumps(doc)); raise SystemExit(result.returncode)
''', timeout=180)


def install():
    remote('gpuA', PRELUDE+'''
with (a/'cycle1.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not previous.live(loop.read_json(a/'PID.json'))
 cfg=loop.load_config(a/'CONFIG.json'); state=loop.load_state(cfg); reg=loop.check_registration(cfg)
 journal=loop.read_json(a/'loop/worlds_c01.json')
 assert state['sim_s_charged']==139540 and journal['charged']==36933 and not journal['worlds']
 assert reg['budget']['cycle_attempt_allowances']['1']==73866
 assert reg['collection']['cycle_long_attempt_allowances']['1']==15000
 assert reg['budget']['sim_s_total']==210000 and not state.get('stopped')
 tests=loop.read_json(r/'C1COLLECT3_TESTS.json'); assert tests['exit']==0
 assert tests['source_sha256']==loop.sha256(r/'C1COLLECT3_SOURCE.json')
 source=loop.read_json(r/'C1COLLECT3_SOURCE.json')
 for name,sha in source.items(): assert loop.sha256(root/name)==sha
 preflights={box:loop.read_json(r/('C1COLLECT3_PREFLIGHT_'+box+'.json')) for box in ('cpuP','cpuQ')}
 assert all(doc['status']=='PASS' and doc['onnx']['loaded'] and doc['learned_policy_loaded'] for doc in preflights.values())
 assert [c['seed'] for c in journal['cases']]==list(range(968157,968170))
 assert state['policy']['sha256']=='64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c'
 from wms.control.nativeinit import verify_collection_gate
 gate=verify_collection_gate(state['policy']); assert gate['passed']
 assert gate['wm_sha256']==[state['policy_wm']['sha256']]
 assert gate['wm_config_sha256']==[loop.sha256(state['policy_wm']['config'])]
 for name in ('CONFIG.json','REGISTRATION.json','loop/LOOP14_STATE.json','PID.json','OPERATIONAL_PAUSED.json'):
  p=a/name; dest=r/'c1collect3_before'/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dest)
 amendment={'id':'S-C1COLLECT3','authority':'main decision 2026-10-27 06:00 UTC+8',
  'registered_at':loop.now(),'cycle':1,'attempt_allowance_sim_s':73866,'long_credit_allowance_sim_s':15000,
  'total_cap_sim_s':210000,'charged_before':139540,'no_refunds':True,'new_collection_attempts':1,
  'seeds':list(range(968157,968170)),'policy_sha256':state['policy']['sha256'],
  'canary':'ONE cpuP world; remaining reserved worlds only after verified numerical output',
  'preflight':{box:loop.sha256(r/('C1COLLECT3_PREFLIGHT_'+box+'.json')) for box in preflights},
  'source':source,'tests':tests,'failure':'OPERATIONAL_PAUSED; no automatic collection retry',
  'checker_change':preflights['cpuP']['checker_change'],
  'previous_registration_sha256':loop.sha256(cfg['registration'])}
 cfg['collect']['startup_canary']={'cycle':1,'authority':'S-C1COLLECT3','boxes':['cpuP'],
  'next_attempts':{str(c['seed']):journal['attempts'].get(str(c['seed']),0)+1 for c in journal['cases']}}
 assert cfg['collect']['native_smoke_harmful_only_from_cycle']==1
 loop.write_atomic(a/'CONFIG.json',cfg)
 reg.setdefault('amendments',[]).append(amendment); reg['sources']['config_sha256']=loop.sha256(a/'CONFIG.json')
 loop.write_atomic(pathlib.Path(cfg['registration']),reg)
 state['registration_sha256']=loop.sha256(cfg['registration']); loop.save_state(cfg,state)
 binding={**amendment,'registration_sha256':state['registration_sha256'],'config_sha256':loop.sha256(a/'CONFIG.json')}
 loop.write_atomic(r/'C1COLLECT3_BINDINGS.json',binding)
 previous.event('S-C1COLLECT3_authority_installed',**binding)
 loop.append(a/'loop/cycle_ledger.jsonl',{'event':'S-C1COLLECT3_authority_installed',**binding})
 print(json.dumps(binding))
''', timeout=120)


def launch():
    remote('gpuA', PRELUDE+'''
with (a/'cycle1.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 recovery=(r/'C1COLLECT3_RECOVERY_BINDINGS.json').exists()
 binding=loop.read_json(r/('C1COLLECT3_RECOVERY_BINDINGS.json' if recovery else 'C1COLLECT3_BINDINGS.json'))
 assert loop.sha256(a/'CONFIG.json')==binding['config_sha256']
 assert loop.sha256(a/'REGISTRATION.json')==binding['registration_sha256']
 for name,sha in binding['source'].items(): assert loop.sha256(root/name)==sha
 assert not previous.live(loop.read_json(a/'PID.json'))
 assert loop.load_state(loop.load_config(a/'CONFIG.json'))['sim_s_charged']==139540
 if recovery:
  assert loop.read_json(r/'C1COLLECT3_LAUNCH.json')['pid']==77640
  for p in (r/'C1COLLECT3_ATTEMPT.json',a/'driver_c1collect3.log',r/'C1COLLECT3_LAUNCH.json'):
   dest=r/'c1collect3_before'/('failed_transition_'+p.name); assert not dest.exists(); p.rename(dest)
 assert not (r/'C1COLLECT3_ATTEMPT.json').exists()
 for name in ('PID.json','OPERATIONAL_PAUSED.json'):
  p=a/name; dest=r/'c1collect3_before'/(('recovery_launch_' if recovery else 'launch_')+name); assert not dest.exists(); p.rename(dest)
 with (a/'driver_c1collect3.log').open('xb') as log:
  env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
  proc=subprocess.Popen([sys.executable,'-B','-u',str(root/'tools/loopresume/c1collect3_box.py')],
   cwd=root,env=env,start_new_session=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 pid={'pid':proc.pid,'start_ticks':loop.fast_module().process_token(proc.pid),'t':loop.now(),'task':'S-C1COLLECT3'}
 loop.write_atomic(a/'PID.json',pid); loop.write_atomic(r/'C1COLLECT3_LAUNCH.json',pid)
 print(json.dumps(pid))
''')


def status():
    remote('gpuA', PRELUDE+'''
from tools.loopresume.c1collect_box import probe
cfg=loop.load_config(a/'CONFIG.json'); state=loop.load_state(cfg); journal=loop.read_json(a/'loop/worlds_c01.json')
ph=state['cycles'][1]['phases']; pid=loop.read_json(a/'PID.json')
binding=loop.read_json(r/('C1COLLECT3_RECOVERY_BINDINGS.json' if (r/'C1COLLECT3_RECOVERY_BINDINGS.json').exists() else 'C1COLLECT3_BINDINGS.json')); probes={}
for item in journal['cases']:
 n=journal['attempts'].get(str(item['seed']),0)
 if n!=cfg['collect']['startup_canary']['next_attempts'][str(item['seed'])]: continue
 box=item['box'] if n%2 else ('cpuQ' if item['box']=='cpuP' else 'cpuP')
 probes[str(item['seed'])]=dict(box=box,attempt=n,**probe(cfg,item,box,n))
doc={'id':'S-C1COLLECT3','t':loop.now(),'pid':pid,'live':previous.live(pid),'charged':state['sim_s_charged'],
 'cycle_charged':journal['charged'],'phase_status':{k:v['status'] for k,v in ph.items()},
 'main_priority_active':any(c['phases'][n]['status']=='running' for c in state['cycles'] for n in ('collect','check')),
 'completed_worlds':list(journal['worlds']),'canaries':journal.get('startup_canaries',{}),'probes':probes,
 'guard':journal.get('native_smoke'),'stopped':state.get('stopped'),
 'preflight':{box:loop.read_json(r/('C1COLLECT3_PREFLIGHT_'+box+'.json'))['status'] for box in ('cpuP','cpuQ')},
 'cycle_attempt_allowance':73866,'long_credit_attempt_allowance':15000,'total_cap':210000,
 'registration_sha256':state['registration_sha256'],'source':binding['source']}
ledger=[json.loads(line) for line in (a/'loop/seed_ledger.jsonl').read_text().splitlines()]
doc['long_credit_charged_included']=sum(row.get('long_credit_sim_s_charged',0) for row in ledger if row.get('cycle')==1)
for name in ('OPERATIONAL_PAUSED.json','WAIT_OWNER.json'):
 if (a/name).exists(): doc[name]=loop.read_json(a/name)
doc['status']='OPERATIONAL_PAUSED' if 'OPERATIONAL_PAUSED.json' in doc else ('WAIT_OWNER' if state.get('stopped') else ph['collect']['status'])
loop.write_atomic(r/'C1COLLECT3_STATUS.json',doc)
print(json.dumps(doc)); print('LOG',(a/'driver_c1collect3.log').read_text()[-6000:])
''', timeout=240)


def repair():
    names = ['tools/loopresume/c1collect3.py', 'tools/loopresume/c1collect3_box.py',
             'tests/unit/test_c1collect3.py']
    payload = {name: base64.b64encode((ROOT/name).read_bytes()).decode() for name in names}
    remote('gpuA', PRELUDE+rf'''
import base64
with (a/'cycle1.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not previous.live(loop.read_json(a/'PID.json'))
 assert loop.read_json(a/'PID.json')['pid']==77640
 assert 'collect failed; inspect' in loop.read_json(a/'OPERATIONAL_PAUSED.json')['error']
 cfg=loop.load_config(a/'CONFIG.json'); state=loop.load_state(cfg); journal=loop.read_json(a/'loop/worlds_c01.json')
 assert state['sim_s_charged']==139540 and journal['charged']==36933 and not journal['worlds']
 assert not journal.get('startup_canaries')
 assert state['cycles'][1]['phases']['collect']['attempt']==4
 assert not (r/'C1COLLECT3_RECOVERY_BINDINGS.json').exists()
 source=loop.read_json(r/'C1COLLECT3_SOURCE.json')
 for name,encoded in {payload!r}.items():
  dest=root/name; assert loop.sha256(dest)==source[name]
  backup=r/'c1collect3_before'/('failed_transition_'+dest.name); assert not backup.exists()
  shutil.copy2(dest,backup); dest.write_bytes(base64.b64decode(encoded)); source[name]=loop.sha256(dest)
 loop.write_atomic(r/'C1COLLECT3_RECOVERY_SOURCE.json',source)
 env={{**os.environ,'PYTHONPATH':str(root/'src')+':'+str(root),'CUDA_VISIBLE_DEVICES':'',
  'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1',
  'NATIVEINIT_BASELINE':str(r/'nativeinit_baseline')}}
 env.pop('C4_ALLOWED_ROOTS',None)
 args=[sys.executable,'-B','-m','pytest','-q','-ra','-p','no:cacheprovider','--import-mode=importlib',
  '--basetemp',str(r/'pytest_c1collect3_recovery'),'tests/unit/test_c1collect3.py','tests/unit/test_c1collect.py',
  'tests/unit/test_loopresume.py','tests/unit/test_fastloop.py','tests/unit/test_nativeinit.py','tests/unit/test_loop14.py']
 result=subprocess.run(args,cwd=root,env=env,capture_output=True,text=True)
 (r/'C1COLLECT3_RECOVERY_TESTS.log').write_text(result.stdout+result.stderr)
 tests={{'t':loop.now(),'exit':result.returncode,'source_sha256':loop.sha256(r/'C1COLLECT3_RECOVERY_SOURCE.json'),
  'log_sha256':loop.sha256(r/'C1COLLECT3_RECOVERY_TESTS.log')}}
 loop.write_atomic(r/'C1COLLECT3_RECOVERY_TESTS.json',tests)
 print(result.stdout+result.stderr); assert result.returncode==0
 reg=loop.check_registration(cfg)
 event={{'id':'S-C1COLLECT3_zero_dispatch_driver_repair','t':loop.now(),'additional_charged':0,
  'failed_driver':loop.read_json(a/'PID.json'),'source':source,'tests':tests,
  'repair':'failed phase must enter via retry(); initial driver refused before any charge/dispatch; same ONE authorized attempt'}}
 reg.setdefault('operational_events',[]).append(event)
 loop.write_atomic(pathlib.Path(cfg['registration']),reg)
 state['registration_sha256']=loop.sha256(cfg['registration']); loop.save_state(cfg,state)
 binding={{**loop.read_json(r/'C1COLLECT3_BINDINGS.json'),'source':source,'tests':tests,
  'registration_sha256':state['registration_sha256'],'zero_dispatch_repair':event}}
 loop.write_atomic(r/'C1COLLECT3_RECOVERY_BINDINGS.json',binding)
 previous.event('S-C1COLLECT3_zero_dispatch_driver_repair',**event)
 loop.append(a/'loop/cycle_ledger.jsonl',{{'event':'S-C1COLLECT3_zero_dispatch_driver_repair',**event}})
 print(json.dumps(binding))
''', timeout=180)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['manifest', 'preflight', 'stage', 'install', 'launch', 'status', 'repair'])
    globals()[parser.parse_args().action]()
