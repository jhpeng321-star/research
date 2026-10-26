"""Sequential, scoped S-C1FIX transport and durable operational receipts."""
import argparse
import base64

from tools.loopresume.operate import ROOT, remote

RELEASE = '/root/autodl-tmp/c4/restart20261026'
FILES = ['src/wms/control/policy14_cli.py', 'src/wms/control/policy14_recovery.py',
         'tests/unit/test_policy14_recovery.py', 'tools/loopresume/continue_c1.py']


def stage():
    payload = {n: base64.b64encode((ROOT/n).read_bytes()).decode() for n in FILES}
    config = base64.b64encode((ROOT/'configs/c4_policy14_c1_fallback.toml').read_bytes()).decode()
    remote('gpuA', f'''
import base64,hashlib,json,pathlib,shutil,subprocess,os
r=pathlib.Path({RELEASE!r}); root=r/'runtime_c1'; assert root.is_dir()
backup=r/'c1fix_before'; backup.mkdir(exist_ok=False)
manifest={{}}
for name,encoded in {payload!r}.items():
 p=root/name
 if p.exists():
  b=backup/name; b.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,b)
 data=base64.b64decode(encoded); p.write_bytes(data); manifest[name]=hashlib.sha256(data).hexdigest()
p=root/'configs/c4_policy14_c1_fallback.toml'
shutil.copy2(p,backup/p.name)
(r/'C1FIX_CONFIG.pending.toml').write_bytes(base64.b64decode({config!r}))
(r/'C1FIX_SOURCE.json').write_text(json.dumps(manifest,sort_keys=True))
env={{**os.environ,'PYTHONPATH':str(root/'src')+':'+str(root),'CUDA_VISIBLE_DEVICES':'',
 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1',
 'NATIVEINIT_BASELINE':str(r/'nativeinit_baseline')}}
env.pop('C4_ALLOWED_ROOTS',None)
argv=['/root/miniconda3/bin/python','-B','-m','pytest','-q','-ra','-p','no:cacheprovider',
 '--import-mode=importlib','--basetemp',str(r/'pytest_c1fix'),
 'tests/unit/test_policy14_recovery.py','tests/unit/test_nativeinit.py','tests/unit/test_loopresume.py',
 'tests/unit/test_resumeprep.py','tests/unit/test_loop14.py']
p=subprocess.run(argv,cwd=root,env=env,capture_output=True,text=True)
log=p.stdout+p.stderr; (r/'C1FIX_TESTS.log').write_text(log); print(log)
proof={{'exit':p.returncode,'source':manifest,'log_sha256':hashlib.sha256(log.encode()).hexdigest()}}
(r/'C1FIX_TESTS.json').write_text(json.dumps(proof,indent=1)); print(json.dumps(proof))
raise SystemExit(p.returncode)
''', timeout=180)


def arm():
    remote('gpuA', r'''
import hashlib,json,pathlib,shutil,sys,os,subprocess,time
r=pathlib.Path('/root/autodl-tmp/c4/restart20261026'); root=r/'runtime_c1'; a=r/'arm'
sys.path[:0]=[str(root),str(root/'src'),str(root/'scripts/c4')]
import loop14 as loop
import loop14_resume as resume
from wms.control.train_cli import _read_config
from wms.control.policy14_recovery import load_recovery
from argparse import Namespace
assert loop.read_json(r/'C1FIX_TESTS.json')['exit']==0
assert loop.fast_module().process_token(64792)==loop.read_json(a/'PID.json')['start_ticks']
cfg=loop.load_config(a/'CONFIG.json'); state=loop.load_state(cfg)
assert state['cycles'][-1]['cycle']==1 and state['cycles'][-1]['phases']['policy']['attempt']==2
assert state['cycles'][-1]['phases']['policy']['status']=='running'
assert state['sim_s_charged']==102607
original=a/'loop/policy/c01'; replay=a/'loop/policy/c01.a2'
failed=loop.read_json(original/'RUN_SUMMARY.json')
assert failed['status']=='failed' and len(failed['updates'])==144
assert failed['updates'][-1]['update']==174
assert 'OutOfMemoryError: CUDA out of memory' in failed['error']
children=pathlib.Path('/proc/64792/task/64792/children').read_text().split()
assert len(children)==1
pid=int(children[0]); cmd=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
assert b'wms.control.policy14_cli' in cmd and str(replay).encode() in cmd
assert not (replay/'STOP').exists()
memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.free','--format=csv,noheader,nounits'],text=True)
total,free=map(float,memory.strip().split(','))
assert total>=47000 and total*(1-.65-.20)>=6144
config=root/'configs/c4_policy14_c1_fallback.toml'
receipt={'id':'S-C1FIX','t':loop.now(),'seed':965141,'phase_updates':150,'saved_update':170,
 'retry_outputs':[str(a/f'loop/policy/c01.a{i}') for i in (3,4)],
 'summary':str(original/'RUN_SUMMARY.json'),'checkpoint':str(original/'policy_native_best.pt'),
 'config':'configs/c4_policy14_c1_fallback.toml','gate_panel':str(a/'gate_panel.json'),
 'failed_update':174,'failed_updates':144,'cooperative_replay_pid':pid,
 'replay_start_ticks':loop.fast_module().process_token(pid),'policy_memory_fraction':.65,
 'total_mib':total,'free_mib_before':free,'minimum_margin_mib_even_at_wm_cap':total*.15,
 'repair':'retain original u170 full checkpoint and curve; replay u171-174, finish u180; no new warmup/share search'}
for key in ('summary','checkpoint','gate_panel'):
 receipt[key+'_sha256']=loop.sha256(receipt[key])
receipt['config_sha256']=loop.sha256(r/'C1FIX_CONFIG.pending.toml')
loop.write_atomic(r/'C1FIX_RECOVERY.json',receipt)
loop.append(r/'C1_EVENTS.jsonl',{'event':'S-C1FIX_amendment_before_retry',**receipt,'original_error':failed['error'],
 'source_sha256':loop.sha256(r/'C1FIX_SOURCE.json'),'tests_sha256':loop.sha256(r/'C1FIX_TESTS.json')})
loop.append(a/'loop/cycle_ledger.jsonl',{'event':'S-C1FIX_amendment_before_retry',**receipt})
shutil.copyfile(r/'C1FIX_CONFIG.pending.toml',config)
os.chdir(root)
args=Namespace(native_init=True,seed=965141,max_updates=150,config=receipt['config'],
 gate_panel=receipt['gate_panel'],resume=failed['resumed_from']['checkpoint'],output_dir=receipt['retry_outputs'][0],
 real_returns=state['real_returns'],wm_ckpt=[state['accepted_wm']['checkpoint']],wm_config=[state['accepted_wm']['config']])
os.environ['C4_ALLOWED_ROOTS']=':'.join(cfg['policy']['allowed_roots']+[cfg['loop_root']])
validated=load_recovery(_read_config(args.config),args)
assert args.max_updates==10 and validated['phase_start']==30
# Cooperative trainer boundary only; the healthy driver and independent WM are untouched.
(replay/'STOP').write_text('S-C1FIX: preserve automatic replay; next owned retry continues original best u170 with raised cap.\n')
print(json.dumps(receipt))
''', timeout=120)


def status():
    remote('gpuA', r'''
import pathlib,sys,json,subprocess
r=pathlib.Path('/root/autodl-tmp/c4/restart20261026'); root=r/'runtime_c1'
sys.path[:0]=[str(root),str(root/'src'),str(root/'scripts/c4')]
from tools.loopresume import continue_c1 as c
d=c.status()
cfg=c.loop.load_config(r/'arm/CONFIG.json')
print('COLLECT_PLAN',json.dumps({'work_root':cfg['collect']['work_root'],'loop_id':cfg['loop_id']}))
out=pathlib.Path(d['policy_output'])
for name in ('RUN_SUMMARY.json','updates.jsonl'):
 p=out/name
 if p.exists():
  if name.endswith('jsonl'):
   rows=p.read_text().splitlines(); row=json.loads(rows[-1]) if rows else {}
   print('ACTIVE_UPDATES',len(rows),{k:row.get(k) for k in ('update','wall_time_s','actor_loss','critic_loss')})
  else:
   s=json.loads(p.read_text()); print('SUMMARY',json.dumps({k:s.get(k) for k in ('status','error','operational_recovery','gpu_memory_fraction','recommended_updates_per_cycle')}))
children=pathlib.Path('/proc/64792/task/64792/children')
print('CHILDREN',children.read_text() if children.exists() else 'driver exited')
print('LOG', (r/'arm/driver_c1.log').read_text()[-5000:])
log=out.with_suffix(out.suffix+'.log')
if log.exists(): print('POLICY_LOG',log.read_text()[-3500:])
wm=r/'arm/loop/wm/c00/train_log.jsonl'
if wm.exists():
 rows=wm.read_text().splitlines()
 if rows:
  row=json.loads(rows[-1]); print('WM_LAST',json.dumps({k:row.get(k) for k in ('step','finite','loss','seconds','cuda_peak_mb')}))
print(subprocess.check_output(['ps','-p','64792,62910,62645','-o','pid,ppid,stat,etime,args'],text=True))
''')


def verify():
    remote('gpuA', r'''
import pathlib,sys,json
r=pathlib.Path('/root/autodl-tmp/c4/restart20261026'); root=r/'runtime_c1'; a=r/'arm'
sys.path[:0]=[str(root),str(root/'src'),str(root/'scripts/c4')]
import loop14 as loop
from wms.control.nativeinit import verify_collection_gate
cfg=loop.load_config(a/'CONFIG.json'); state=loop.load_state(cfg)
ph=state['cycles'][1]['phases']; assert ph['policy']['status']=='done'
out=pathlib.Path(ph['policy']['receipt']['out']); summary=loop.read_json(out/'RUN_SUMMARY.json')
gate=verify_collection_gate(state['policy'])
assert summary['status']=='completed' and len(summary['updates'])==150
assert summary['updates'][0]['update']==31 and summary['updates'][-1]['update']==180
assert summary['gpu_memory_fraction']==.65 and summary['critic_warmup']['updates']==0
assert gate['measured_through']==180 and gate['phase_start_update']==30
assert gate['wm_sha256']==[state['policy_wm']['sha256']]
assert gate['wm_config_sha256']==[loop.sha256(state['policy_wm']['config'])]
assert loop.sha256(state['real_returns'])==summary['real_share']['source_sha256']
assert state['sim_s_charged']<=210000 and ph['collect']['status'] in ('running','done')
journal=loop.read_json(a/'loop/worlds_c01.json')
doc={'id':'S-C1FIX','t':loop.now(),'driver':loop.read_json(a/'PID.json'),
 'gate':gate,'policy':state['policy'],'phase_updates':len(summary['updates']),
 'phases':{k:v['status'] for k,v in ph.items()},'charged':state['sim_s_charged'],
 'journal_charged':journal['charged'],'cases':journal['cases'],
 'collect_work_root':cfg['collect']['work_root'],'loop_id':cfg['loop_id'],
 'source_sha256':loop.sha256(r/'C1FIX_SOURCE.json'),'tests':loop.read_json(r/'C1FIX_TESTS.json'),
 'recovery_sha256':loop.sha256(r/'C1FIX_RECOVERY.json'),
 'wm0':loop.read_json(r/'LOOPRESUME_WM_ONLY.json')['status']}
loop.write_atomic(r/'C1FIX_VERIFIED.json',doc)
loop.append(r/'C1_EVENTS.jsonl',{'event':'S-C1FIX_gate_verified_collection_dispatched',**doc})
loop.append(a/'loop/cycle_ledger.jsonl',{'event':'S-C1FIX_gate_verified_collection_dispatched',**doc})
print(json.dumps(doc))
''')


def wait():
    remote('gpuA', r'''
import pathlib,json,time
r=pathlib.Path('/root/autodl-tmp/c4/restart20261026'); a=r/'arm'; out=a/'loop/policy/c01.a3'
deadline=time.monotonic()+900
while time.monotonic()<deadline:
 state=json.loads((a/'loop/LOOP14_STATE.json').read_text()); ph=state['cycles'][1]['phases']
 path=out/'updates.jsonl'; rows=path.read_text().splitlines() if path.exists() else []
 print(json.dumps({'policy':ph['policy']['status'],'collect':ph['collect']['status'],
  'retry_updates':len(rows),'last_update':json.loads(rows[-1])['update'] if rows else None}),flush=True)
 if (out/'RUN_SUMMARY.json').exists():
  summary=json.loads((out/'RUN_SUMMARY.json').read_text())
  if summary['status']!='completed': raise RuntimeError(summary.get('error'))
 if state['stopped'] or (a/'OPERATIONAL_PAUSED.json').exists() or (a/'WAIT_OWNER.json').exists():
  raise RuntimeError('owned driver stopped; inspect recorded receipt')
 if ph['policy']['status']=='done' and ph['collect']['status'] in ('running','done'):
  print('GATE_COMPLETE_COLLECTION_PHASE_ENTERED',flush=True); break
 time.sleep(45)
else: raise TimeoutError('waiting for corrected native curve/collection')
''', timeout=960)


def collectors():
    remote('gpuA', r'''
import pathlib,json
r=pathlib.Path('/root/autodl-tmp/c4/restart20261026'); a=r/'arm'; loop=a/'loop'
state=json.loads((loop/'LOOP14_STATE.json').read_text())
print('PHASES',json.dumps(state['cycles'][1]['phases']))
print('JOURNAL',(loop/'worlds_c01.json').read_text())
rows=[json.loads(x) for x in (loop/'seed_ledger.jsonl').read_text().splitlines()]
print('C1_CHARGES',json.dumps([x for x in rows if x.get('cycle')==1]))
print('DRIVER_LOG',(a/'driver_c1.log').read_text()[-5000:])
for seed in range(968157,968161):
 p=loop/f'worlds/c01/w{seed}.a1/logs/c01_collect.log'
 if p.exists(): print('WORLD_LOG',seed,p.read_text()[-3500:])
''')
    for host, seeds in [('cpuP', [968157, 968159]), ('cpuQ', [968158, 968160])]:
        remote(host, f'''
import json,pathlib,subprocess
seeds={seeds!r}; base=pathlib.Path('/home/ubuntu/c4/main14/restart20261026/work')
rows=subprocess.check_output(['ps','--no-headers','-eo','pid,ppid,etimes,args'],text=True).splitlines()
owned=[]
for row in rows:
 fields=row.strip().split(None,3)
 if len(fields)!=4: continue
 args=fields[3].split()
 if 'traffic_shepherd.c4.collect' in args and '--out' in args:
  output=args[args.index('--out')+1]
  if any(output.startswith(str(base/f'w{{s}}/C4-LOOP14-COOPON-20261026/c01')) for s in seeds):
   pid=int(fields[0]); children=pathlib.Path(f'/proc/{{pid}}/task/{{pid}}/children').read_text().split()
   owned.append({{'pid':pid,'ppid':int(fields[1]),'elapsed_s':int(fields[2]),'out':output,'children':children}})
print('COLLECTORS',json.dumps({{'host':{host!r},'owned':owned}}))
for item in owned:
 for name in ('INDEX.json','COLLECT_SUMMARY.json'):
  p=pathlib.Path(item['out'])/name
  if p.exists(): print(name,p.read_text()[:5000])
for seed in seeds:
 p=base/f'w{{seed}}'/'C4-LOOP14-COOPON-20261026/c01/collect/collect_log.jsonl'
 if p.exists(): print('FAILURE',seed,p.read_text())
''')


def failures():
    for host, seed in [('cpuP', 968157), ('cpuQ', 968158)]:
        remote(host, f'''
import pathlib,json
p=pathlib.Path('/home/ubuntu/c4/main14/restart20261026/work/w{seed}/C4-LOOP14-COOPON-20261026/c01/collect/collect_log.jsonl')
print({host!r},p.read_text())
''')


def repair_collectors():
    for host in ('cpuP', 'cpuQ'):
        remote(host, r'''
import pathlib,hashlib,json,shutil,sys
r=pathlib.Path('/home/ubuntu/c4/main14/restart20261026'); old=r/'runtime'; new=r/'runtime_c1'
assert old.is_dir() and new.is_dir()
name='baselines/circles/adapter.json'; source=old/name; dest=new/name
assert source.is_file() and not dest.exists()
dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,dest)
sys.path[:0]=[str(new),str(new/'src')]
from traffic_shepherd.config import load_config
cfg=load_config()
proof={'id':'S-C1FIX','repair':'copy identical original adapter; no collection redispatch',
 'adapter':str(dest),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
 'original_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'network':str(cfg.network_path),'network_exists':cfg.network_path.exists(),
 'scenario_template':str(cfg.scenario_template_path),'scenario_template_exists':cfg.scenario_template_path.exists()}
failures=[]
for seed in range(968157,968161):
 for attempt in (1,2,3):
  suffix='' if attempt==1 else f'.a{attempt}'
  p=r/f'work/w{seed}/C4-LOOP14-COOPON-20261026/c01{suffix}/collect/collect_log.jsonl'
  if p.exists():
   for line in p.read_text().splitlines():
    row=json.loads(line); failures.append({k:row.get(k) for k in ('pid','seed','status','error','started_utc','wall_seconds')})
proof['failed_dispatches']=failures
(r/'C1FIX_COLLECTOR_REPAIR.json').write_text(json.dumps(proof,indent=1)); print(json.dumps(proof))
''')


def link_assets():
    for host in ('cpuP', 'cpuQ'):
        remote(host, r'''
import pathlib,json
r=pathlib.Path('/home/ubuntu/c4/main14/restart20261026'); old=r/'runtime'; new=r/'runtime_c1'
proof=json.loads((r/'C1FIX_COLLECTOR_REPAIR.json').read_text()); links=[]
# Metadata/symlinks only: never open the scenario's vehicle-type assignments.
for name in ('assets/research/highway_merge_shepherd_long/osm.net.xml.gz','scenarios/idm/driver_routes.rou.xml'):
 source=old/name; dest=new/name
 assert source.is_file() and not dest.exists()
 dest.parent.mkdir(parents=True,exist_ok=True); dest.symlink_to(source)
 assert dest.is_file()
 links.append({'path':str(dest),'target':str(source),'metadata_only':True})
proof.update(asset_links=links,network_exists=True,scenario_template_exists=True)
(r/'C1FIX_COLLECTOR_REPAIR.json').write_text(json.dumps(proof,indent=1)); print(json.dumps(proof))
''')


def finalize():
    encoded = base64.b64encode((ROOT/'tools/loopresume/continue_c1.py').read_bytes()).decode()
    remote('gpuA', f'''
import base64,pathlib,json,sys,subprocess,hashlib
r=pathlib.Path({RELEASE!r}); root=r/'runtime_c1'; a=r/'arm'
p=root/'tools/loopresume/continue_c1.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()==json.loads((r/'C1FIX_SOURCE.json').read_text())['tools/loopresume/continue_c1.py']
p.write_bytes(base64.b64decode({encoded!r}))
sys.path[:0]=[str(root),str(root/'src'),str(root/'scripts/c4')]
from tools.loopresume import continue_c1 as c
status=c.status(); assert not status['live'] and status['charged']==136699
assert status['updates']==150 and status['updates_this_attempt']==10
assert status['native_gate_result.json']['passed']
assert status['phases']['collect']=='pending' and status['collection']['worlds']==[]
repairs={{}}
for host in ('cpuP','cpuQ'):
 source="""import pathlib,json,subprocess
r=pathlib.Path('/home/ubuntu/c4/main14/restart20261026')
d=json.loads((r/'C1FIX_COLLECTOR_REPAIR.json').read_text())
active=[]
for row in subprocess.check_output(['ps','--no-headers','-eo','pid,ppid,etimes,args'],text=True).splitlines():
 fields=row.strip().split(None,3)
 if len(fields)!=4: continue
 args=fields[3].split()
 if 'traffic_shepherd.c4.collect' in args and '--out' in args:
  out=args[args.index('--out')+1]
  if '/restart20261026/work/w' in out and '/C4-LOOP14-COOPON-20261026/c01' in out:
   active.append(fields)
d['active_collectors']=active
print(json.dumps(d))
"""
 result=subprocess.run(['ssh','-o','BatchMode=yes',host,'/home/ubuntu/miniconda3/bin/python -B -'],
  input=source,capture_output=True,text=True,check=True,timeout=60)
 repairs[host]=json.loads(result.stdout); assert not repairs[host]['active_collectors']
cfg=c.loop.load_config(a/'CONFIG.json'); state=c.loop.load_state(cfg)
journal=c.loop.read_json(a/'loop/worlds_c01.json')
assert journal['charged']==34092
assert c.loop.collect_sim_s(cfg)==36933 and c.loop.long_group_sim_s(cfg)==7800
doc={{'id':'S-C1FIX','t':c.loop.now(),'status':'OPERATIONAL_PAUSED',
 'gate_passed':True,'selected_update':180,'phase_updates':150,
 'gate_mean_delta':status['native_gate_result.json']['mean_delta'],
 'policy':status['policy'],'charged':136699,'total_cap':210000,
 'cycle_charged':34092,'cycle_cap':36933,'long_credit_charged':7200,'long_credit_cap':8000,
 'failed_collection_attempts':3,'completed_worlds':0,'active_collectors':[],
 'pause':status['OPERATIONAL_PAUSED.json'],'collector_repairs':repairs,
 'next_boundary_requires_owner_amendment':{{'no_refunds':True,'same_13_worlds':True,
  'minimum_cycle_attempt_allowance':71025,'minimum_long_credit_attempt_allowance':15000,
  'projected_total_after_13_worlds':173632,'total_cap_unchanged':210000,
  'renewed_operational_retry_authority':True}},
 'driver_live':False,'wm0_status':status['wm0'],
 'status_source_sha256':c.loop.sha256(p),'eta':'blocked pending owner amendment; no active collector ETA'}}
c.loop.write_atomic(r/'C1FIX_COLLECTION_BLOCKED.json',doc)
c.event('S-C1FIX_final_gate_pass_collection_blocked',**doc)
c.loop.append(a/'loop/cycle_ledger.jsonl',{{'event':'S-C1FIX_final_gate_pass_collection_blocked',**doc}})
print('FINAL',json.dumps(doc))
''', timeout=180)


def final_checks():
    names = ['src/wms/control/policy14_recovery.py', 'tests/unit/test_policy14_recovery.py']
    payload = {n: base64.b64encode((ROOT/n).read_bytes()).decode() for n in names}
    remote('gpuA', f'''
import base64,json,pathlib,subprocess,os,sys
r=pathlib.Path({RELEASE!r}); root=r/'runtime_c1'
sys.path[:0]=[str(root),str(root/'src'),str(root/'scripts/c4')]
import loop14 as loop
original=loop.read_json(r/'C1FIX_SOURCE.json'); final={{}}
for name,encoded in {payload!r}.items():
 p=root/name; assert loop.sha256(p)==original[name]
 (r/'c1fix_before'/('before_final_'+p.name)).write_bytes(p.read_bytes())
 p.write_bytes(base64.b64decode(encoded)); final[name]=loop.sha256(p)
env={{**os.environ,'PYTHONPATH':str(root/'src')+':'+str(root),'CUDA_VISIBLE_DEVICES':'',
 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1',
 'NATIVEINIT_BASELINE':str(r/'nativeinit_baseline')}}
env.pop('C4_ALLOWED_ROOTS',None)
args=['/root/miniconda3/bin/python','-B','-m','pytest','-q','-p','no:cacheprovider','--import-mode=importlib',
 '--basetemp',str(r/'pytest_c1fix_final'),'tests/unit/test_policy14_recovery.py','tests/unit/test_nativeinit.py',
 'tests/unit/test_loopresume.py','tests/unit/test_resumeprep.py','tests/unit/test_loop14.py']
p=subprocess.run(args,cwd=root,env=env,capture_output=True,text=True)
(r/'C1FIX_FINAL_TESTS.log').write_text(p.stdout+p.stderr); print(p.stdout+p.stderr)
proof={{'t':loop.now(),'exit':p.returncode,'source':final,'log_sha256':loop.sha256(r/'C1FIX_FINAL_TESTS.log'),
 'note':'explicit pre-training corpus hash veto added; successful retry corpus independently matched during gate verification'}}
loop.write_atomic(r/'C1FIX_FINAL_TESTS.json',proof)
loop.append(r/'C1_EVENTS.jsonl',{{'event':'S-C1FIX_final_provenance_checks',**proof}})
print(json.dumps(proof)); raise SystemExit(p.returncode)
''', timeout=180)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['stage', 'arm', 'status', 'verify', 'wait', 'collectors', 'failures', 'repair-collectors', 'link-assets', 'finalize', 'final-checks'])
    globals()[parser.parse_args().action.replace('-', '_')]()
