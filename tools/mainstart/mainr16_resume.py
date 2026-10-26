"""Owner-authorized S-MAINR16 continuation using the cpuQ provenance copy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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
    receipt('PREARM.json', cf.remote('gpuR', code, timeout=1200))


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


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('probe','stage-base','deploy','prepare','tests','prearm','migrate','launch','audit-failure'))
    args = p.parse_args()
    {'probe': probe,'stage-base':stage_base,'deploy':deploy,'prepare':prepare,'tests':tests,
     'prearm':prearm,'migrate':migrate,'launch':launch,'audit-failure':audit_failure}[args.action]()
