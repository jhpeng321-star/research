"""Bounded SSH coordination; all heavy files/tests/staging stay on boxes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RELEASE = '/root/autodl-tmp/c4/restart20261026'
RUNTIME = RELEASE+'/runtime'
INPUTS = RELEASE+'_inputs'
CPU = '/home/ubuntu/c4/main14/restart20261026/runtime'


def remote(host, source, timeout=900):
    python = '/home/ubuntu/miniconda3/bin/python' if host in ('cpuP', 'cpuQ') else '/root/miniconda3/bin/python'
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, python+' -B -'],
                       input=source, text=True, encoding='utf-8', capture_output=True, timeout=timeout)
    print(p.stdout, end='')
    if p.stderr:
        print(p.stderr, file=sys.stderr, end='')
    if p.returncode:
        raise SystemExit(p.returncode)
    return p.stdout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['authorize', 'qviews', 'returns', 'collectors', 'collector-readiness', 'registration', 'refresh-prearm', 'mini-policy-detached', 'seed-recheck', 'smoke-detached', 'diagnose-mask', 'mini-status', 'wm-progress', 'wm-config', 'profile-wm', 'wait-mini', 'queue-auto', 'auto-status', 'wait-first-gates', 'final-stop', 'final-evidence', 'check-mirror', 'progress', 'record-repairs', 'status'])
    a = p.parse_args()
    if a.action == 'authorize':
        remote('gpuA', f'''import pathlib,json,datetime
root=pathlib.Path({RELEASE!r}); assert root.is_dir()
p=root/'WAIT_OWNER.json'; archived=root/'WAIT_OWNER.continuation2.json'
if p.exists():
 assert not archived.exists(); p.rename(archived)
receipt={{'t':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authority':'owner explicit continuation3',
 'operational_retries_per_error':3,'scientific_retry':False,'laptop_pause_floor_bytes':314572800,
 'baseline_fixture':'3 original modules SHA-verified from gpuR; 401 tests PASS19.42s',
 'prior_stop_archived':str(archived),'no_new_sumo_or_seeds':True}}
(root/'CONTINUATION3.json').write_text(json.dumps(receipt,indent=1)); print(json.dumps(receipt))
''')
    elif a.action == 'qviews':
        # This is a tiny source helper, not a laptop data/archive stage.
        subprocess.run(['scp', str(ROOT/'tools/restart/qstage.py'), 'gpuR:/root/resumeprep_probe/restart_qstage.py'], check=True)
        remote('gpuR', f'''import runpy,sys
sys.argv=['qstage.py',{INPUTS+'/data'!r}]
runpy.run_path('/root/resumeprep_probe/restart_qstage.py',run_name='__main__')
''', timeout=900)
    elif a.action == 'returns':
        remote('gpuA', f'''import sys,json,pathlib,hashlib
sys.path[:0]=[{RUNTIME!r}]
from tools.resumeprep.boxstage import stage
print(json.dumps(stage('local','/root/autodl-tmp/c4/longcritic','local',{INPUTS+'/longcritic'!r},
 ['current_real_returns.npz','paired_native_returns.npz'])))
p=pathlib.Path({INPUTS+'/longcritic/current_real_returns.npz'!r})
assert hashlib.sha256(p.read_bytes()).hexdigest()=='6756b3cb8a5295e8385cec3fbca9f4b5c2e64924c8074673e2b54f7e86f82cfa'
''')
    elif a.action in ('collectors', 'collector-readiness'):
        if a.action == 'collectors':
            remote('gpuA', f'''import sys,json
sys.path[:0]=[{RUNTIME!r}]
from tools.resumeprep.boxstage import stage
for host in ('cpuP','cpuQ'):
 print(json.dumps({{'host':host,**stage('local',{RUNTIME!r},host,{CPU!r},['src','traffic_shepherd','configs','scripts','tools'])}}),flush=True)
''')
        else:
            remote('gpuA', f'''import pathlib,json,datetime
p=pathlib.Path({RELEASE!r})/'OPERATIONS_C3.jsonl'
with p.open('a') as f: f.write(json.dumps({{'t':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'error':'collector readiness ModuleNotFoundError: output','host':'cpuP','attempt_failed':1,
 'repair':'link existing approved output fixture modules, alongside baselines/assets/scenarios; no SUMO',
 'retry':1,'retry_limit':3}})+'\\n')
''')
        for host in ('cpuP', 'cpuQ'):
            remote(host, f'''import pathlib,os,sys,json
root=pathlib.Path({CPU!r}); old=pathlib.Path('/home/ubuntu/c4/main14/mainr16/runtime')
assert root.is_dir() and old.is_dir()
links={{}}
for name in ('baselines','assets','scenarios','output'):
 source=old/name; target=root/name
 assert source.is_dir(),str(source)
 if target.is_symlink(): assert target.resolve()==source.resolve()
 else: target.symlink_to(source,target_is_directory=True)
 links[name]=str(source)
os.environ.update(PATH='/home/ubuntu/miniconda3/bin:'+os.environ.get('PATH',''),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path[:0]=[str(root),str(root/'src')]
from traffic_shepherd.c4.collector_readiness import check_runtime_dependencies,check_controller_assets
from traffic_shepherd.config import load_config
receipt={{'fixture_links':links,'dependencies':check_runtime_dependencies(),'controller':check_controller_assets(load_config())}}
(root.parent/'READINESS.json').write_text(json.dumps(receipt,indent=1)); print(json.dumps(receipt))
''')
    elif a.action == 'record-repairs':
        remote('gpuA', f'''import pathlib,json,datetime
root=pathlib.Path({RELEASE!r}); t=datetime.datetime.now(datetime.timezone.utc).isoformat()
rows=[{{'error':'mini-policy SubsetProvider missing _returns in final reporting','attempt_failed':1,
 'repair':'report subset indices.size and actual trainer config; catch AttributeError; training/selection unchanged',
 'source_commit':'87e7fce3','retry_limit':3}},
 {{'error':'source overlay baseline mismatch on generated __pycache__ bytecode','attempt_failed':1,
 'repair':'exclude derived .pyc only, retain all source SHA checks; retry verified93overlays',
 'source_commit':'43e91800','retry':1,'retry_limit':3}},
 {{'error':'foreground SSH wait reached600s while exact owned remote coordinator25838 remained live',
 'repair':'inspect exact PID/start tick and existing phase; no duplicate phase launched; use durable status checks',
 'start_ticks':'840636061','retry_limit':3}}]
with (root/'OPERATIONS_C3.jsonl').open('a') as f:
 for row in rows: f.write(json.dumps({{'t':t,**row}})+'\\n')
print(json.dumps({{'recorded':len(rows)}}))
''')
    elif a.action == 'progress':
        remote('gpuA', f'''import pathlib,json
arm=pathlib.Path({RELEASE+'/arm'!r}); root=arm/'mini/loop'
s=json.loads((root/'LOOP14_STATE.json').read_text()); ph=s['cycles'][-1]['phases']['policy']; out=root/'policy'/('c00' if ph['attempt']==1 else 'c00.a'+str(ph['attempt']))
doc={{'phase':ph,'charged':s['sim_s_charged'],'out':str(out)}}
p=out/'updates.jsonl'
if p.exists():
 rows=p.read_text().splitlines(); doc['recorded_updates']=len(rows)
 if rows:
  row=json.loads(rows[-1]); doc['last_update']={{k:row.get(k) for k in ('update','wall_time_s','actor_loss','actor_grad_norm','real_return_mix','imagination_horizon_s','longcritic_active','longcritic_horizon_s')}}
for name in ('native_gate_result.json','REAL_SHARE.json','RUN_SUMMARY.json'):
 p=out/name
 if p.exists():
  d=json.loads(p.read_text()); doc[name]={{k:d[k] for k in ('status','error','passed','mean_delta','selected_share','offpolicy_bootstrap') if k in d}}
p=pathlib.Path('/proc')/str(ph.get('pid',0))/'stat'
if p.exists():
 fields=p.read_text().rsplit(')',1)[1].split(); doc['coordinator_token']=fields[19]; doc['coordinator_state']=fields[0]
 print(json.dumps(doc))
else: print(json.dumps(doc))
''')
    elif a.action == 'refresh-prearm':
        remote('gpuA', f'''import pathlib,json,datetime
arm=pathlib.Path({RELEASE+'/arm'!r}); s=json.loads((arm/'mini/loop/LOOP14_STATE.json').read_text()); ph=s['cycles'][0]['phases']['policy']
assert ph['status']=='failed' and not pathlib.Path('/proc/'+str(ph['pid'])+'/stat').exists(), 'owned mini coordinator still active'
for name in ('prearm','prearm.log','PREARM.json'):
 p=arm/name; dst=arm/(name+'.before_reporting_repair')
 if p.exists(): assert not dst.exists(); p.rename(dst)
with (arm.parent/'OPERATIONS_C3.jsonl').open('a') as f:
 f.write(json.dumps({{'t':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'error':'mini-policy attempt2 also completed10updates and gates but failed final SubsetProvider reporting',
  'repair':'reporting-only source43e91800 installed; full402testsPASS18.45s; refresh synthetic prearm and official attempt3',
  'attempt_failed':2,'retry_next':2,'retry_limit':3,'scientific_failure':False,'sim_s_charged':0}})+'\\n')
print(json.dumps({{'ready_for_prearm_refresh':True,'mini_attempt':ph['attempt'],'new_sumo':0}}))
''')
    elif a.action == 'mini-policy-detached':
        remote('gpuA', f'''import pathlib,json,subprocess,sys
arm=pathlib.Path({RELEASE+'/arm'!r}); receipt=arm/'MINI_POLICY_PID.json'
assert not receipt.exists()
with (arm/'mini_policy.detached.log').open('xb') as log:
 p=subprocess.Popen([sys.executable,'-B',{RUNTIME+'/tools/resumeprep/control.py'!r},'mini-policy','--arm',str(arm)],
  start_new_session=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
token=pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
d={{'pid':p.pid,'start_ticks':token,'action':'mini-policy only; no SUMO dispatch'}}
receipt.write_text(json.dumps(d)); print(json.dumps(d))
''')
    elif a.action == 'seed-recheck':
        raw = remote('gpuR', '''import pathlib,json,hashlib
root=pathlib.Path('/root/main14_loop'); p=root/'LOOP14_STATE.json'; s=json.loads(p.read_text()); ledger=root/'seed_ledger.jsonl'; rows=[json.loads(x) for x in ledger.read_text().splitlines()]
used=set(s['seeds_used'])
for row in rows:
 used.update(row.get('seeds',[]))
 if 'seed' in row: used.add(row['seed'])
assert not used.intersection({968142,968143}), 'reserved smoke seeds consumed; never substitute'
print(json.dumps({'smoke_seeds':[968142,968143],'unused_in_live_predecessor':True,'state_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'ledger_sha256':hashlib.sha256(ledger.read_bytes()).hexdigest(),'charged':s['sim_s_charged']}))
''')
        receipt = json.loads(raw)
        remote('gpuA', f'''import pathlib,json,datetime
p=pathlib.Path({RELEASE+'/arm/SMOKE_LIVE_SEED_RECHECK.json'!r})
if p.exists():
 old=p.with_name(p.name+'.before_retry1'); assert not old.exists(); p.rename(old)
d={receipt!r}; d['t']=datetime.datetime.now(datetime.timezone.utc).isoformat(); p.write_text(json.dumps(d)); print(json.dumps(d))
''')
    elif a.action == 'smoke-detached':
        remote('gpuA', f'''import pathlib,json,subprocess,sys,hashlib
arm=pathlib.Path({RELEASE+'/arm'!r}); receipt=arm/'MINI_SMOKE_PID.json'; assert not receipt.exists()
assert json.loads((arm/'SMOKE_LIVE_SEED_RECHECK.json').read_text())['unused_in_live_predecessor']
ack=json.loads((arm/'SHARED_SMOKE_HASHES_APPENDED.json').read_text()); assert ack['bindings_sha256']==hashlib.sha256((arm/'SMOKE_BINDINGS.json').read_bytes()).hexdigest()
logpath=arm/'mini_smoke.detached.log'
if logpath.exists():
 assert (arm/'MINI_SMOKE_PID.json.before_junction_fix').exists()
 old=arm/'mini_smoke.detached.log.before_retry1'; assert not old.exists(); logpath.rename(old)
with logpath.open('xb') as log:
 p=subprocess.Popen([sys.executable,'-B',{RUNTIME+'/tools/resumeprep/control.py'!r},'mini','--arm',str(arm)],
  start_new_session=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
token=pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
d={{'pid':p.pid,'start_ticks':token,'action':'registered1922sim-s smoke/conversion/WM2 only'}}
receipt.write_text(json.dumps(d)); print(json.dumps(d))
''')
    elif a.action == 'diagnose-mask':
        rows = []
        for host, seed, share in (('cpuP', 968142, '0050'), ('cpuQ', 968143, '0400')):
            work = f'/home/ubuntu/c4/main14/restart20261026/work/w{seed}/C4-LOOP14-COOPON-20261026/c00'
            raw = remote(host, f'''import os,sys,pathlib,json,hashlib
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',C4_ALLOWED_ROOTS={work!r})
sys.path[:0]=[{CPU!r},{CPU+'/src'!r}]
import torch,numpy as np
torch.set_num_threads(1)
from wms.control.policy import load_policy_components,PolicyExecutor
from wms.control.obs import build_observations
root=pathlib.Path({work!r}); ep=root/'collect/episodes/policy/s{seed}_m1900_p{share}'
with np.load(ep/'episode.npz',allow_pickle=False) as z:
 initial=int(np.searchsorted(z['time_s'],360.)); assert z['time_s'][initial]==360.
 candidates=np.flatnonzero((z['present']&z['is_cav']&(z['lane']<0)).any(axis=1))
 assert len(candidates), 'no recorded native junction-CAV state for executor diagnosis'
 later=candidates[z['time_s'][candidates]>=300.]
 i=int(later[0] if len(later) else candidates[0]); observed_time=float(z['time_s'][i])
 st={{k:torch.as_tensor(z[n][i:i+1].copy()) for k,n in [('s_m','s_m'),('v_mps','v_mps'),('a_mps2','a_mps2'),('length_m','length_true_m'),('lane','lane'),('present','present'),('is_cav','is_cav')]}}
policy,encoder,rules,_=load_policy_components(root/'inputs/policy.pt'); exe=PolicyExecutor(policy,encoder,lane_rules=rules)
cmd,lanes=exe(st,with_lanes=True); obs=build_observations(st); live=(st['present']&st['is_cav']).numpy(); acting=obs.cav_mask.numpy(); nonfinite=~np.isfinite(cmd.residual)
d={{'seed':{seed},'checkpoint_sha256':hashlib.sha256((root/'inputs/policy.pt').read_bytes()).hexdigest(),
 'observed_native_replay_time_s':observed_time,'actual_failing_policy_tick_not_preserved':True,
 'live_cavs':int(live.sum()),'acting_cavs':int(acting.sum()),'nonfinite_on_acting_cavs':int((nonfinite&acting).sum()),
 'nonfinite_on_unobserved_cavs':int((nonfinite&live&~acting).sum()),
 'unobserved_cav_lanes':st['lane'].numpy()[nonfinite&live].tolist(),
 'all_policy_parameters_finite':all(bool(torch.isfinite(p).all()) for p in policy.parameters()),
 'native_base_summary':json.loads((ep/'summary.json').read_text()),'no_new_simulation':True}}
assert d['nonfinite_on_acting_cavs']==0 and d['all_policy_parameters_finite'], 'SCIENTIFIC nonfinite acting policy commands'
assert d['nonfinite_on_unobserved_cavs']>0 and all(x==-1 for x in d['unobserved_cav_lanes']), 'unclassified executor failure'
print(json.dumps(d))
''')
            rows.append(json.loads(raw))
        remote('gpuA', f'''import pathlib,json
d={{'classification':'non-scientific executor contract mismatch: NaN absence sentinel on junction CAVs; acting head outputs finite',
 'worlds':{rows!r},'new_sumo':0}}
p=pathlib.Path({RELEASE+'/arm/SMOKE_MASK_DIAGNOSIS.json'!r}); p.write_text(json.dumps(d,indent=1)); print(json.dumps({{'classification':d['classification'],'worlds':len(d['worlds'])}}))
''')
    elif a.action == 'mini-status':
        remote('gpuA', f'''import pathlib,json
arm=pathlib.Path({RELEASE+'/arm'!r}); cfg=json.loads((arm/'CONFIG.json').read_text()); mini=pathlib.Path(cfg.get('resumeprep',{{}}).get('mini_root',str(arm/'mini'))); s=json.loads((mini/'loop/LOOP14_STATE.json').read_text()); parent=json.loads((arm/'loop/LOOP14_STATE.json').read_text())
d={{'phases':{{k:{{n:v.get(n) for n in ('status','attempt','error','receipt') if n in v}} for k,v in s['cycles'][0]['phases'].items()}},
 'mini_charged':s['sim_s_charged'],'total_charged':parent['sim_s_charged'],'mini_pass':(arm/'MINI_PASS.json').exists()}}
receipt=arm/'MINI_SMOKE_PID.json'
if receipt.exists():
 p=json.loads(receipt.read_text()); st=pathlib.Path('/proc/'+str(p['pid'])+'/stat'); d['owned_pid']=p
 if st.exists():
  f=st.read_text().rsplit(')',1)[1].split(); d['same_incarnation']=f[19]==p['start_ticks']; d['pid_state']=f[0]
log=arm/'mini_smoke.detached.log'
if log.exists(): d['log_tail']=log.read_text()[-3000:]
print(json.dumps(d))
''')
    elif a.action == 'wm-progress':
        remote('gpuA', f'''import pathlib,json,datetime
arm=pathlib.Path({RELEASE+'/arm'!r}); cfg=json.loads((arm/'CONFIG.json').read_text()); mini=pathlib.Path(cfg.get('resumeprep',{{}}).get('mini_root',str(arm/'mini')))
s=json.loads((mini/'loop/LOOP14_STATE.json').read_text()); ph=s['cycles'][0]['phases']['wm_ft']; out=mini/'loop/wm/c00'
d={{'t':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':ph,'checkpoint_exists':(out/'wm_s.pt').exists(),'mini_pass':(arm/'MINI_PASS.json').exists()}}
p=out/'train_log.jsonl'
if p.exists():
 rows=[json.loads(x) for x in p.read_text().splitlines()]; d['rows']=len(rows)
 if rows:
  d['last']={{k:rows[-1].get(k) for k in ('step','loss','grad_norm','finite','seconds','cuda_peak_mb')}}
  d['step_seconds']=[r.get('seconds') for r in rows]
v=out/'val_log.jsonl'
if v.exists():
 vals=[json.loads(x) for x in v.read_text().splitlines()]
 d['validation_rows']=len(vals)
 d['validation_horizons']=[r.get('horizon_s') for r in vals]
pid=s['cycles'][0]['phases']['wm_ft'].get('pid'); proc=pathlib.Path('/proc')/str(pid)
if (proc/'stat').exists():
 d['owned_coordinator']=pid; d['owned_children']=(proc/'task'/str(pid)/'children').read_text()
 children=[]
 for child in d['owned_children'].split():
  cp=pathlib.Path('/proc')/child
  if (cp/'stat').exists():
   fields=(cp/'stat').read_text().rsplit(')',1)[1].split()
   children.append({{'pid':int(child),'start_ticks':fields[19],'state':fields[0],'cpu_ticks':[fields[11],fields[12]],
    'status':[x for x in (cp/'status').read_text().splitlines() if x.startswith(('VmRSS:','VmSwap:','Threads:'))],
    'wchan':(cp/'wchan').read_text(),'io':(cp/'io').read_text()}})
 d['children_detail']=children
print(json.dumps(d))
''')
    elif a.action == 'wait-mini':
        remote('gpuA', f'''import pathlib,json,time
arm=pathlib.Path({RELEASE+'/arm'!r}); pid=json.loads((arm/'MINI_SMOKE_PID.json').read_text()); cfg=json.loads((arm/'CONFIG.json').read_text()); mini=pathlib.Path(cfg['resumeprep']['mini_root'])
for _ in range(60):
 if (arm/'MINI_PASS.json').exists(): break
 p=pathlib.Path('/proc/'+str(pid['pid'])+'/stat')
 if not p.exists(): break
 f=p.read_text().rsplit(')',1)[1].split()
 if f[19]!=pid['start_ticks'] or f[0]=='Z': break
 time.sleep(10)
s=json.loads((mini/'loop/LOOP14_STATE.json').read_text())
print(json.dumps({{'mini_pass':(arm/'MINI_PASS.json').exists(),'phases':{{k:v['status'] for k,v in s['cycles'][0]['phases'].items()}},
 'wm':s['cycles'][0]['phases']['wm_ft'],'log_tail':(arm/'mini_smoke.detached.log').read_text()[-2000:]}}))
''', timeout=650)
    elif a.action == 'queue-auto':
        subprocess.run(['scp', str(ROOT/'tools/restart/auto_restart.py'), 'gpuA:'+RELEASE+'/auto_restart.py'], check=True)
        remote('gpuA', f'''import pathlib,json,subprocess,sys,hashlib
root=pathlib.Path({RELEASE!r}); arm=root/'arm'; assert not (arm/'AUTO_PID.json').exists()
with (arm/'auto_restart.log').open('xb') as log:
 p=subprocess.Popen([sys.executable,'-B',str(root/'auto_restart.py')],start_new_session=True,
  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
token=pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
d={{'pid':p.pid,'start_ticks':token,'script_sha256':hashlib.sha256((root/'auto_restart.py').read_bytes()).hexdigest(),
 'scope':'MINI_PASS + mechanics validation -> official detached launch -> first gates + portable own-arm mirror'}}
(arm/'AUTO_PID.json').write_text(json.dumps(d)); print(json.dumps(d))
''')
    elif a.action in ('auto-status', 'wait-first-gates'):
        remote('gpuA', f'''import pathlib,json,time
arm=pathlib.Path({RELEASE+'/arm'!r})
for _ in range({360 if a.action == 'wait-first-gates' else 1}):
 first=arm/'FIRST_GATES.json'
 if first.exists() and json.loads(first.read_text()).get('real_guard') is not None and (arm/'MIRROR_RECEIPT.json').exists(): break
 if (arm/'AUTO_BLOCKED.json').exists() or (arm/'WAIT_OWNER.json').exists(): break
 if {a.action == 'auto-status'!r}: break
 time.sleep(10)
d={{}}
for name in ('MINI_PASS.json','SMOKE_RESULT.json','PID.json','AUTO_PID.json','FIRST_GATES.json','MIRROR_RECEIPT.json','WAIT_OWNER.json','AUTO_BLOCKED.json'):
 p=arm/name
 if p.exists():
  row=json.loads(p.read_text())
  if name in ('PID.json','AUTO_PID.json'):
   st=pathlib.Path('/proc/'+str(row['pid'])+'/stat')
   if st.exists():
    f=st.read_text().rsplit(')',1)[1].split(); row['same_incarnation']=f[19]==row['start_ticks']; row['process_state']=f[0]
   else: row['same_incarnation']=False
  if name=='SMOKE_RESULT.json': row={{k:v for k,v in row.items() if k!='rows'}}
  d[name]=row
p=arm/'auto_restart.log'
if p.exists(): d['auto_log_tail']=p.read_text()[-1600:]
p=arm/'loop/LOOP14_STATE.json'
if p.exists():
 s=json.loads(p.read_text()); d['main_phase']={{k:v['status'] for k,v in s['cycles'][-1]['phases'].items()}}
 ph=s['cycles'][-1]['phases']['policy']; out=pathlib.Path(ph.get('receipt',{{}}).get('out',str(arm/'loop/policy/c00')))
 u=out/'updates.jsonl'
 if u.exists(): d['main_recorded_updates']=len(u.read_text().splitlines())
print(json.dumps(d))
''', timeout=3700)
    elif a.action == 'wm-config':
        remote('gpuA', f'''import pathlib,json
root=pathlib.Path({RELEASE+'/arm/mini_retry1/loop/wm/c00'!r})
for name in ('train_config.json','config.resolved.json'):
 p=root/name
 if p.exists():
  d=json.loads(p.read_text()); d=d.get('config',d)
  print(json.dumps({{'file':name,'batch_size':d.get('batch_size'),'val_max_windows':d.get('val_max_windows'),
   'selection':d.get('selection'),'optim':d.get('optim'),'selection_horizons_s':d.get('selection_horizons_s'),
   'val_all_horizons':d.get('val_all_horizons'),'select_by_long_horizon':d.get('select_by_long_horizon')}}))
''')
    elif a.action == 'profile-wm':
        remote('gpuA', f'''import pathlib,json,subprocess,sys,os
root=pathlib.Path({RELEASE!r}); assert root.is_dir(); target=root/'profiler'
p=pathlib.Path('/proc/37793/stat'); assert p.exists()
f=p.read_text().rsplit(')',1)[1].split(); assert f[19]=='840958791' and f[1]=='37473', 'not the exact owned WM child'
if not (target/'bin/py-spy').exists():
 subprocess.run([sys.executable,'-m','pip','install','--no-cache-dir','--no-deps','--target',str(target),'py-spy'],check=True,env={{**os.environ,'OMP_NUM_THREADS':'1'}})
result=subprocess.run([str(target/'bin/py-spy'),'dump','--pid','37793','--nonblocking','--full-filenames'],text=True,capture_output=True)
(root/'WM_STACK.txt').write_text(result.stdout+result.stderr)
print(result.stdout+result.stderr); print(json.dumps({{'read_only_profile_exit':result.returncode,'owned_pid':37793,'start_ticks':'840958791'}}))
''', timeout=180)
    elif a.action == 'final-stop':
        remote('gpuA', f'''import pathlib,json,sys,datetime
sys.path[:0]=[{RUNTIME+'/scripts/c4'!r},{RUNTIME!r},{RUNTIME+'/src'!r}]
import loop14 as loop
from wms.control.nativeinit import verify_collection_gate
arm=pathlib.Path({RELEASE+'/arm'!r}); cfg=loop.load_config(arm/'CONFIG.json'); state=loop.load_state(cfg)
j=loop.read_json(pathlib.Path(cfg['loop_root'])/'worlds_c00.json'); guard=j['native_smoke']; pid=loop.read_json(arm/'PID.json')
p=pathlib.Path('/proc/'+str(pid['pid'])+'/stat'); assert not p.exists() or p.read_text().rsplit(')',1)[1].split()[0]=='Z'
assert state['stopped']=='native smoke inconclusive' and not guard['passed']
current_attempts=[c['seed'] for c in j['cases'] if str(c['seed']) in j['attempts']]
assert len(j['worlds'])==4 and len(current_attempts)==4 and j['charged']==11364 and state['sim_s_charged']==102607
gate=verify_collection_gate(state['policy']); panel=loop.read_json(pathlib.Path(state['policy']['checkpoint']).with_name('gate_panel_result.json'))
receipt={{'t':loop.now(),'status':'SCIENTIFIC_WAIT_OWNER','driver':pid,'driver_same_incarnation':False,
 'policy':state['policy'],'wm':state['accepted_wm'],'updates':30,
 'update10':{{'passed':panel['passed'],'mean_delta':panel['mean_delta'],'n_improved':panel['n_improved']}},
 'native_gate':{{'passed':gate['passed'],'mean_delta':gate['mean_delta'],'selected_update':gate['update']}},
 'real_guard':guard,'real_guard_units':'mean per-world policy-minus-native J; each world sums two300s starts',
 'sim_s_charged':state['sim_s_charged'],'stopped':state['stopped'],
 'dispatched_worlds':[w['case']['seed'] for w in j['worlds'].values()],
 'undispatched_reserved_worlds':[c['seed'] for c in j['cases'] if str(c['seed']) not in j['worlds']],
 'main_collection_charged':j['charged']}}
loop.write_atomic(arm/'FIRST_GATES.json',receipt)
print(json.dumps(receipt))
''')
    elif a.action == 'final-evidence':
        remote('gpuA', f'''import pathlib,json,sys,importlib.util,datetime
sys.path[:0]=[{RUNTIME+'/scripts/c4'!r},{RUNTIME!r},{RUNTIME+'/src'!r}]
import loop14 as loop
arm=pathlib.Path({RELEASE+'/arm'!r}); cfg=loop.load_config(arm/'CONFIG.json'); state=loop.load_state(cfg)
source_sha=loop.sha256(loop.state_path(cfg)); mirror=loop.read_json(arm/'MIRROR_RECEIPT.json')
if mirror['sha256']['state']!=source_sha:
 spec=importlib.util.spec_from_file_location('own_final_mirror',arm.parent/'auto_restart.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
 assert not m.live(loop.read_json(arm/'AUTO_PID.json')) and not m.live(loop.read_json(arm/'PID.json'))
 m.mirror(cfg,state); mirror=loop.read_json(arm/'MIRROR_RECEIPT.json')
assert mirror['sha256']['state']==source_sha
wm=pathlib.Path(cfg['resumeprep']['mini_root'])/'loop/wm/c00'
log=[json.loads(x) for x in (wm/'train_log.jsonl').read_text().splitlines()]
wm_receipt=loop.read_json(pathlib.Path(cfg['resumeprep']['mini_root'])/'loop/LOOP14_STATE.json')['cycles'][0]['phases']['wm_ft']['receipt']
d={{'status':'SCIENTIFIC_WAIT_OWNER','guard':loop.read_json(arm/'FIRST_GATES.json'),
 'current_state_sha256':source_sha,'mirror':mirror,'mini_wm_receipt':wm_receipt,
 'mini_step_seconds':[r['seconds'] for r in log], 'mini_steps_finite':all(r['finite'] for r in log),
 'scientific_failure_retried':False,'source_commit':'daac879c59b9570bf7d10ec5b840aa028305cfa4',
 'own_driver_pid':loop.read_json(arm/'PID.json'),'own_auto_pid':loop.read_json(arm/'AUTO_PID.json')}}
loop.write_atomic(arm/'RESTART_FINAL.json',d)
with (arm.parent/'OPERATIONS_C3.jsonl').open('a') as f:
 for row in [{{'error':'initial offline junction diagnosis inspected t360 with no junction present','repair':'use existing native junction frames; no new SUMO','retry':1}},
  {{'error':'read-only py-spy ownPID37793 denied by container ptrace permissions','repair':'no privilege changes; retained finite logs and waited for full validation','retry':0}},
  {{'error':'final-report assertion counted historical cycle0 attempts as current','repair':'filter attempt keys to successor cases; no state/charge/world changes','retry':1}}]:
  f.write(json.dumps({{'t':loop.now(),'diagnostic_only':True,**row}})+'\\n')
print(json.dumps(d))
''', timeout=900)
    elif a.action == 'check-mirror':
        remote('cpuP', '''import pathlib,json,hashlib
root=pathlib.Path('/home/ubuntu/c4/main14/restart20261026/mirror'); arm=root/'arm'
d=json.loads((arm/'RESTART_FINAL.json').read_text()); expected=d['mirror']['sha256']
assert json.loads((arm/'FIRST_GATES.json').read_text())==d['guard']
for path,key in ((arm/'PORTABLE_STATE.json','state'),(arm/'loop/LOOP14_STATE.json','state'),
                 (arm/'loop/policy/c00/policy_latest.pt','policy'),(root/'base/wm_s_best.pt','base')):
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected[key],str(path)
assert json.loads((arm/'PORTABLE_STATE.json').read_text())['stopped']=='native smoke inconclusive'
print(json.dumps({'status':'PASS','mirror':'cpuP','current_and_portable_state_match':True,'policy_base_hashes_match':True,'scientific_stop_preserved':True}))
''')
    elif a.action == 'registration':
        remote('gpuA', f'''import pathlib,sys,json,copy
sys.path[:0]=[{RUNTIME+'/scripts/c4'!r},{RUNTIME!r},{RUNTIME+'/src'!r}]
import loop14 as loop
arm=pathlib.Path({RELEASE+'/arm'!r}); cfg=loop.load_config(arm/'CONFIG.json'); state=loop.load_state(cfg)
reg=loop.read_json(cfg['registration'])
reg['gates']['failure']='Scientific native identity/native-relative veto/smoke behavior/update10/native-gate failure: WAIT_OWNER; owner continuation3 permits three recorded fixes/retries per non-scientific setup/staging/SSH/fixture error; E below300MB pauses'
reg['sources']['superseded_restart_sources']=copy.deepcopy(reg['sources'])
reg['sources'].update(code_sha='43e91800ee06571a2ca42ae1d0e70307385a3e2e',config_sha256=loop.sha256(arm/'CONFIG.json'),source_overlay_sha256=loop.sha256(arm.parent/'SOURCE_OVERLAY.json'))
loop.write_atomic(pathlib.Path(cfg['registration']),reg)
state['registration_sha256']=loop.sha256(cfg['registration']); loop.save_state(cfg,state)
mini=arm/'mini'; mr=loop.read_json(mini/'REGISTRATION.json'); ms=loop.read_json(mini/'loop/LOOP14_STATE.json')
mr['sources']['reporting_only_repair_commit']='43e91800ee06571a2ca42ae1d0e70307385a3e2e'
mr['sources']['source_overlay_sha256']=reg['sources']['source_overlay_sha256']
loop.write_atomic(mini/'REGISTRATION.json',mr); ms['registration_sha256']=loop.sha256(mini/'REGISTRATION.json'); loop.write_atomic(mini/'loop/LOOP14_STATE.json',ms)
print(json.dumps({{'config_sha256':loop.sha256(arm/'CONFIG.json'),'registration_sha256':state['registration_sha256'],
 'source_overlay_sha256':loop.sha256(arm.parent/'SOURCE_OVERLAY.json'),'charged':state['sim_s_charged'],
 'loop_root':cfg['loop_root'],'retry_limit':cfg['resumeprep']['operational_retries_per_error']}}))
''')
    else:
        remote('gpuA', f'''import pathlib,json,subprocess
root=pathlib.Path({RELEASE!r}); arm=root/'arm'
for name in ('PREPARED.json','READINESS.json','PREARM.json','SMOKE_BINDINGS.json','MINI_PASS.json','PID.json','WAIT_OWNER.json'):
 p=arm/name
 if p.exists(): print(json.dumps({{'name':name,'doc':json.loads(p.read_text())}}))
for name in ('mini/loop/LOOP14_STATE.json','loop/LOOP14_STATE.json'):
 p=arm/name
 if p.exists():
  s=json.loads(p.read_text()); print(json.dumps({{'state':name,**{{k:s.get(k) for k in ('sim_s_charged','seeds_used','policy','stopped','cycles')}}}}))
print(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free','--format=csv,noheader'],text=True))
''')


if __name__ == '__main__':
    main()
