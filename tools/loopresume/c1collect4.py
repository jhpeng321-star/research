"""Sequential, source-bound S-C1COLLECT4 operations; no implicit dispatch."""
import argparse
import base64

from tools.loopresume.c1collect import PRELUDE
from tools.loopresume.operate import ROOT, remote

NAMES = ['tools/loopresume/c1collect4.py', 'tools/loopresume/c1collect4_patch.py',
         'tools/loopresume/c1collect4_box.py', 'tests/unit/test_c1collect4.py']


def stage():
    payload = {n: base64.b64encode((ROOT/n).read_bytes()).decode() for n in NAMES}
    remote('gpuA', PRELUDE+f'''
import base64
assert not previous.live(loop.read_json(a/'PID.json'))
cfg=loop.load_config(a/'CONFIG.json'); state=loop.load_state(cfg)
assert state['sim_s_charged']==150904 and not state.get('stopped')
backup=r/'c1collect4_before'
revision=backup.exists()
if revision:
 assert not (r/'C1COLLECT4_BINDINGS.json').exists()
 assert loop.read_json(r/'C1COLLECT4_TESTS.json')['exit']!=0
 for name in ('SOURCE.json','TESTS.json','TESTS.log'):
  p=r/('C1COLLECT4_'+name); shutil.copy2(p,backup/('failed_fixture_'+name))
else: backup.mkdir()
baseline=r/('C1COLLECT4_SOURCE.json' if revision else 'C1COLLECT3_RECOVERY_SOURCE.json')
for name,sha in loop.read_json(baseline).items():
 assert loop.sha256(root/name)==sha,name
source={{}}
for name,encoded in {payload!r}.items():
 p=root/name; assert p.parent.is_dir()
 if p.exists():
  assert revision
  shutil.copy2(p,backup/('failed_fixture_'+p.name))
 else: assert not revision
 p.write_bytes(base64.b64decode(encoded)); source[name]=loop.sha256(p)
from tools.loopresume.c1collect4_patch import resume_source
p=root/'scripts/c4/loop14_fast.py'
if not revision:
 shutil.copy2(p,backup/p.name)
 p.write_text(resume_source(p.read_text()))
for name in ('scripts/c4/loop14_fast.py','scripts/c4/loop14.py','scripts/c4/loop14_resume.py',
             'tools/loopresume/continue_c1.py','tools/loopresume/c1collect_box.py'):
 source[name]=loop.sha256(root/name)
loop.write_atomic(r/'C1COLLECT4_SOURCE.json',source)
env={{**os.environ,'PYTHONPATH':str(root/'src')+':'+str(root),'CUDA_VISIBLE_DEVICES':'',
 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1',
 'NATIVEINIT_BASELINE':str(r/'nativeinit_baseline')}}
env.pop('C4_ALLOWED_ROOTS',None)
args=[sys.executable,'-B','-m','pytest','-q','-ra','-p','no:cacheprovider','--import-mode=importlib',
 '--basetemp',str(r/'pytest_c1collect4'),'tests/unit/test_c1collect4.py','tests/unit/test_c1collect3.py',
 'tests/unit/test_c1collect.py','tests/unit/test_loopresume.py','tests/unit/test_fastloop.py',
 'tests/unit/test_nativeinit.py','tests/unit/test_loop14.py']
result=subprocess.run(args,cwd=root,env=env,capture_output=True,text=True)
(r/'C1COLLECT4_TESTS.log').write_text(result.stdout+result.stderr)
doc={{'t':loop.now(),'exit':result.returncode,'source_sha256':loop.sha256(r/'C1COLLECT4_SOURCE.json'),
 'log_sha256':loop.sha256(r/'C1COLLECT4_TESTS.log')}}
loop.write_atomic(r/'C1COLLECT4_TESTS.json',doc)
print(result.stdout+result.stderr); print(json.dumps(doc)); raise SystemExit(result.returncode)
''', timeout=180)


def install():
    remote('gpuA', PRELUDE+'''
with (a/'cycle1.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not previous.live(loop.read_json(a/'PID.json'))
 cfg=loop.load_config(a/'CONFIG.json'); state=loop.load_state(cfg); reg=loop.check_registration(cfg)
 j=loop.read_json(a/'loop/worlds_c01.json')
 assert state['sim_s_charged']==150904 and j['charged']==48297 and not state.get('stopped')
 assert sorted(map(int,j['worlds']))==list(range(968157,968161))
 assert [c['seed'] for c in j['cases']]==list(range(968157,968170))
 assert j['native_smoke']['passed'] and j['native_smoke']['verdict']=='pass'
 assert j['startup_canaries']['cpuP']['verified']
 assert state['cycles'][1]['phases']['collect']['status']=='pending'
 assert all(state['cycles'][1]['phases'][n]['status']=='pending' for n in ('convert','wm_ft','check'))
 assert reg['budget']['sim_s_total']==210000 and reg['budget']['cycle_attempt_allowances']['1']==73866
 assert reg['collection']['cycle_long_attempt_allowances']['1']==15000
 assert state['policy']['sha256']=='64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c'
 from wms.control.nativeinit import verify_collection_gate
 gate=verify_collection_gate(state['policy']); assert gate['passed']
 assert gate['wm_sha256']==[state['policy_wm']['sha256']]
 assert gate['wm_config_sha256']==[loop.sha256(state['policy_wm']['config'])]
 assert loop.read_json(r/'LOOPRESUME_WM_ONLY.json')['status']=='completed'
 tests=loop.read_json(r/'C1COLLECT4_TESTS.json'); assert tests['exit']==0
 assert tests['source_sha256']==loop.sha256(r/'C1COLLECT4_SOURCE.json')
 source=loop.read_json(r/'C1COLLECT4_SOURCE.json')
 for name,sha in source.items(): assert loop.sha256(root/name)==sha,name
 from loop14_fast import artifact_valid
 for world in j['worlds'].values(): assert artifact_valid(world['artifact'])
 preflights={box:loop.read_json(r/('C1COLLECT3_PREFLIGHT_'+box+'.json')) for box in ('cpuP','cpuQ')}
 assert all(p['status']=='PASS' and p['onnx']['loaded'] and p['learned_policy_loaded'] for p in preflights.values())
 ledger=[json.loads(x) for x in (a/'loop/seed_ledger.jsonl').read_text().splitlines()]
 long_charged=sum(x.get('long_credit_sim_s_charged',0) for x in ledger if x.get('cycle')==1)
 assert long_charged==10200 and 10200+9*600==15600
 assert 150904+9*2841==176473 and 48297+9*2841==73866
 for name in ('CONFIG.json','REGISTRATION.json','loop/LOOP14_STATE.json','PID.json','OPERATIONAL_PAUSED.json'):
  p=a/name; dest=r/'c1collect4_before'/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dest)
 amendment={'id':'S-C1COLLECT4','authority':'main 2026-10-27 06:20 UTC+8',
  'registered_at':loop.now(),'cycle':1,'long_credit_allowance_before':15000,
  'long_credit_allowance_sim_s':15600,'exact_shortfall_sim_s':600,'included_in_total':True,
  'attempt_allowance_sim_s':73866,'total_cap_sim_s':210000,'no_refunds':True,
  'charged_before':150904,'projected_collection_charge':176473,'remaining_worlds':list(range(968161,968170)),
  'preserved_completed_worlds':list(range(968157,968161)),'guard':j['native_smoke'],
  'preflights':{box:loop.sha256(r/('C1COLLECT3_PREFLIGHT_'+box+'.json')) for box in preflights},
  'policy_sha256':state['policy']['sha256'],'source':source,'tests':tests,
  'operational_retry_limit':1,'continuation':'collect -> convert -> reconcile WM0 -> next boundary -> wm_ft -> unchanged check',
  'resume_repair':'empty completed-bootstrap batch skips startup canary selection; consumed canary retained',
  'previous_registration_sha256':loop.sha256(cfg['registration'])}
 reg.setdefault('amendments',[]).append(amendment)
 reg['collection']['cycle_long_attempt_allowances']['1']=15600
 loop.write_atomic(pathlib.Path(cfg['registration']),reg)
 state['registration_sha256']=loop.sha256(cfg['registration']); loop.save_state(cfg,state)
 binding={**amendment,'registration_sha256':state['registration_sha256'],'config_sha256':loop.sha256(a/'CONFIG.json')}
 loop.write_atomic(r/'C1COLLECT4_BINDINGS.json',binding)
 previous.event('S-C1COLLECT4_amendment_installed',**binding)
 loop.append(a/'loop/cycle_ledger.jsonl',{'event':'S-C1COLLECT4_amendment_installed',**binding})
 print(json.dumps(binding))
''')


def launch():
    remote('gpuA', PRELUDE+'''
with (a/'cycle1.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 binding=loop.read_json(r/'C1COLLECT4_BINDINGS.json')
 assert loop.sha256(a/'CONFIG.json')==binding['config_sha256']
 assert loop.sha256(a/'REGISTRATION.json')==binding['registration_sha256']
 for name,sha in binding['source'].items(): assert loop.sha256(root/name)==sha
 assert not previous.live(loop.read_json(a/'PID.json'))
 assert not (r/'C1COLLECT4_ATTEMPT.json').exists()
 cfg=loop.load_config(a/'CONFIG.json'); assert loop.load_state(cfg)['sim_s_charged']==150904
 for name in ('PID.json','OPERATIONAL_PAUSED.json'):
  p=a/name; dest=r/'c1collect4_before'/('launch_'+name); assert not dest.exists(); p.rename(dest)
 with (a/'driver_c1collect4.log').open('xb') as log:
  env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
  proc=subprocess.Popen([sys.executable,'-B','-u',str(root/'tools/loopresume/c1collect4_box.py')],
   cwd=root,env=env,start_new_session=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 pid={'pid':proc.pid,'start_ticks':loop.fast_module().process_token(proc.pid),'t':loop.now(),'task':'S-C1COLLECT4'}
 loop.write_atomic(a/'PID.json',pid); loop.write_atomic(r/'C1COLLECT4_LAUNCH.json',pid)
 print(json.dumps(pid))
''')


def status():
    remote('gpuA', PRELUDE+'''
from tools.loopresume.c1collect_box import probe
cfg=loop.load_config(a/'CONFIG.json'); state=loop.load_state(cfg); j=loop.read_json(a/'loop/worlds_c01.json')
ph=state['cycles'][1]['phases']; pid=loop.read_json(a/'PID.json'); probes={}
for item in j['cases'][4:]:
 n=j['attempts'].get(str(item['seed']),0)
 if not n: continue
 box=item['box'] if n%2 else ('cpuQ' if item['box']=='cpuP' else 'cpuP')
 probes[str(item['seed'])]=dict(box=box,attempt=n,**probe(cfg,item,box,n))
ledger=[json.loads(x) for x in (a/'loop/seed_ledger.jsonl').read_text().splitlines()]
new=[x for x in ledger if x.get('cycle')==1 and x.get('event')=='world_attempt' and x.get('seed',0)>=968161]
reg=loop.check_registration(cfg)
doc={'id':'S-C1COLLECT4','t':loop.now(),'driver':pid,'driver_live':previous.live(pid),
 'charged':state['sim_s_charged'],'remaining':210000-state['sim_s_charged'],'cycle_charged':j['charged'],
 'long_credit_charged_included':sum(x.get('long_credit_sim_s_charged',0) for x in ledger if x.get('cycle')==1),
 'long_credit_allowance':reg['collection']['cycle_long_attempt_allowances']['1'],'total_cap':210000,
 'completed_worlds':sorted(map(int,j['worlds'])),'new_attempts':new,'probes':probes,
 'phases':{k:v['status'] for k,v in ph.items()},'guard':j['native_smoke'],
 'main_priority_active':any(c['phases'][n]['status']=='running' for c in state['cycles'] for n in ('collect','check')),
 'stopped':state.get('stopped'),'registration_sha256':state['registration_sha256'],
 'bindings_sha256':loop.sha256(r/'C1COLLECT4_BINDINGS.json')}
for name in ('OPERATIONAL_PAUSED.json','WAIT_OWNER.json'):
 if (a/name).exists(): doc[name]=loop.read_json(a/name)
loop.write_atomic(r/'C1COLLECT4_STATUS.json',doc)
if doc['driver_live'] and all(any(p['box']==box and p.get('verified') for p in probes.values()) for box in ('cpuP','cpuQ')):
 if not (r/'C1COLLECT4_RUNNING.json').exists(): loop.write_atomic(r/'C1COLLECT4_RUNNING.json',doc)
print(json.dumps(doc)); print('LOG',(a/'driver_c1collect4.log').read_text()[-5000:])
''', timeout=240)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['stage', 'install', 'launch', 'status'])
    globals()[parser.parse_args().action]()
