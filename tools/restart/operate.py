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
    p.add_argument('action', choices=['authorize', 'qviews', 'returns', 'collectors', 'collector-readiness', 'registration', 'refresh-prearm', 'mini-policy-detached', 'seed-recheck', 'smoke-detached', 'diagnose-mask', 'mini-status', 'progress', 'record-repairs', 'status'])
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
p=pathlib.Path({RELEASE+'/arm/SMOKE_LIVE_SEED_RECHECK.json'!r}); assert not p.exists()
d={receipt!r}; d['t']=datetime.datetime.now(datetime.timezone.utc).isoformat(); p.write_text(json.dumps(d)); print(json.dumps(d))
''')
    elif a.action == 'smoke-detached':
        remote('gpuA', f'''import pathlib,json,subprocess,sys,hashlib
arm=pathlib.Path({RELEASE+'/arm'!r}); receipt=arm/'MINI_SMOKE_PID.json'; assert not receipt.exists()
assert json.loads((arm/'SMOKE_LIVE_SEED_RECHECK.json').read_text())['unused_in_live_predecessor']
ack=json.loads((arm/'SHARED_SMOKE_HASHES_APPENDED.json').read_text()); assert ack['bindings_sha256']==hashlib.sha256((arm/'SMOKE_BINDINGS.json').read_bytes()).hexdigest()
with (arm/'mini_smoke.detached.log').open('xb') as log:
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
arm=pathlib.Path({RELEASE+'/arm'!r}); s=json.loads((arm/'mini/loop/LOOP14_STATE.json').read_text()); parent=json.loads((arm/'loop/LOOP14_STATE.json').read_text())
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
