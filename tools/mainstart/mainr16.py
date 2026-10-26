"""S-MAINR16 blocked-prearm receipts and isolated integrated CPU verification.

There is deliberately no launch action: the owner requires WAIT_OWNER on failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import collectfix as cf

ROOT = cf.ROOT
A = '/root/autodl-tmp/c4/v5/mainstart/mainr16'
R = A + '/runtime'
OUT = ROOT / 'tools/mainstart/mainr16'
REVISION = 1


def save(name, text):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(text, encoding='utf-8')
    print(text)


def block():
    owner = (ROOT / 'tools/mainstart/MAINR16_OWNER_DECISIONS.txt').read_text(encoding='utf-8').strip()
    register = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
    shared = register.read_bytes()
    assert owner in shared.decode('utf-8')
    code = f'''
import pathlib,json,hashlib,copy,datetime
a=pathlib.Path({A!r}); assert a.parent.is_dir(); a.mkdir(exist_ok=True)
root=pathlib.Path('/root/main14_loop')
state_bytes=(root/'LOOP14_STATE.json').read_bytes(); state=json.loads(state_bytes)
assert state['sim_s_charged']==58266 and state['seeds_used']==list(range(968103,968129))
prior_bytes=pathlib.Path(state['registration']).read_bytes()
assert hashlib.sha256(prior_bytes).hexdigest()==state['registration_sha256']
reg=copy.deepcopy(json.loads(prior_bytes))
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
amendment={{'id':'S-MAINR16','owner_approved':'2026-10-26 ~16:15 UTC+8',
 'owner_decisions_verbatim':{owner!r},'registered_at':now,
 'predecessor_registration_sha256':state['registration_sha256'],
 'status':'WAIT_OWNER / uninstalled amendment snapshot; no launch',
 'prior_charged_sim_s':58266,'prior_actual_sim_s':0,'prior_seeds':state['seeds_used'],
 'total_cap_supersedes_all_prior_caps':210000,
 'base_checkpoint_source':'gpuB:/root/autodl-tmp/c4/v5/runs/c4_wm_v10_round16_b/wm_s.pt',
 'base_checkpoint_sha256':None,'checkpoint_sha_status':'UNVERIFIED: source SSH failed before hash output',
 'wm_code':'c4/v10r16 3b3c201df9dc04b11bc2f647e52fc508bb7a6bce integrated into c4/mainstart',
 'physics_config':'configs/c4_wm_v10round16_stage_b.toml',
 'physics_flags':{{'emergency_braking':True,'secure_gap_veto':True,'lane_change_secure_gap':True,
  'emergency_onset':'driver_b','merge_end_wait':False,'ramp_lane_end_stop':True,'ramp_risk_retain':True}},
 'model_flag':{{'ramp_speed_adapt_learnable':True}},'native_lane_change':True,
 'wm_ft_config':'configs/c4_wm_loop14_r16_ft.toml',
 'policy_flags':{{'WMS_SPEED_PIVOT_DIAG_OFF':'1','WMS_SPEED_GRAD_AUDIT_EVERY':'10'}},
 'workers_per_box':7,'workers_cap_per_box':12,
 'policy_superseded':copy.deepcopy(state['policy']),
 'restart':'not installed; prior state, seeds, policy, world journal and all artifacts preserved',
 'shared_registration_sha256':{hashlib.sha256(shared).hexdigest()!r}}}
reg['budget']['sim_s_total']=210000
reg['mainr16_amendment']=amendment
assert reg['seed_block']==[968100,968227] and reg['exclude']==[[968100,968102]]
assert reg['budget']['max_cycles']==4 and reg['budget']['wm_steps_per_cycle']==40
assert not (a/'REGISTRATION.json').exists(), 'immutable amendment snapshot already exists'
(a/'REGISTRATION.json').write_text(json.dumps(reg,indent=1)+'\\n')
(a/'STATE_BEFORE.json').write_bytes(state_bytes)
receipt={{'id':'S-MAINR16','status':'WAIT_OWNER','t':now,
 'error':'gpuB source SHA probe failed: Connection closed by 172.19.0.169 port 16573',
 'failed_command':'ssh -o BatchMode=yes -o ConnectTimeout=15 gpuB sha256sum <round16 FINAL> && ls -lh <FINAL> <config.resolved.json>',
 'checkpoint_sha256':None,'checkpoint_copy_started':False,'launched':False,'owned_pids':[],
 'charged_sim_s':state['sim_s_charged'],'actual_sim_s':0,'new_sim_s':0,
 'seeds_used':state['seeds_used'],'new_seeds':[],
 'superseded_policy':state['policy'],'live_state_registration':state['registration'],
 'amendment_snapshot':str(a/'REGISTRATION.json'),
 'amendment_sha256':hashlib.sha256((a/'REGISTRATION.json').read_bytes()).hexdigest(),
 'live_state_migrated':False,'shared_registration_sha256':{hashlib.sha256(shared).hexdigest()!r},
 'next_owner_action':'Restore/confirm gpuB read-only access, then explicitly authorize continuation from WAIT_OWNER; require SHA-verified staging, state restart migration, prearm smoke, fresh policy gate and both-box collection verification before unattended operation'}}
(a/'WAIT_OWNER.json').write_text(json.dumps(receipt,indent=1)+'\\n')
assert (root/'LOOP14_STATE.json').read_bytes()==state_bytes
print(json.dumps({{'receipt':receipt,'registration':reg}},indent=1))
'''
    save('BLOCKED.json', cf.remote('gpuR', code))


def stage():
    # Isolated verification only: no CPU deployment and no live arm/state repoint.
    payload, _digest = cf.bundle()
    code = f'''
import pathlib,tarfile,io,sys,json,hashlib
a=pathlib.Path({A!r}); assert (a/'WAIT_OWNER.json').is_file()
r=pathlib.Path({R!r}); assert not r.exists(); r.mkdir()
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for m in tar:
  assert m.isfile() and (r/m.name).resolve().is_relative_to(r.resolve())
  assert not any(t in m.name.lower() for t in {cf.BLOCK!r})
  tar.extract(m,r)
doc=json.loads((r/'COLLECTFIX_SOURCE.json').read_text())
for n,h in doc['files'].items(): assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
print(json.dumps({{'files':len(doc['files']),'source_manifest_sha256':doc['sha256'],
 'verification_only':True,'runtime':str(r)}}))
'''
    save(f'STAGE_v{REVISION}.json', cf.remote('gpuR', code, payload, timeout=300))


def tests():
    code = f'''
import pathlib,sys,os,subprocess,json,time
r=pathlib.Path({R!r}); a=r.parent; assert (a/'WAIT_OWNER.json').is_file()
sys.path[:0]=[str(r),str(r/'src'),str(r/'scripts/c4')]
import main14_autolaunch as auto
names=[str(r/'tests/unit'/n) for n in auto.TESTS]
names += [str(r/'tests/unit'/n) for n in ('test_fastloop.py','test_mainstart.py','test_collectfix.py','test_mainr16.py')]
names += [str(r/'tests'/n) for n in ('test_speed2_flags.py','test_loop14_emergency_onset.py',
 'test_r16_gate.py','test_r16_ramp_speed_adapt.py')]
env={{**os.environ,'PYTHONPATH':str(r/'src')+':'+str(r),'CUDA_VISIBLE_DEVICES':'',
 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
 'PYTHONDONTWRITEBYTECODE':'1','C4_ALLOWED_ROOTS':''}}
start=time.time()
with (a/'integrated_tests_v{REVISION}.log').open('wb') as log:
 p=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
 '--basetemp',str(a/'pytest_tmp'),*names],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT)
text=(a/'integrated_tests_v{REVISION}.log').read_text()
doc={{'exit':p.returncode,'elapsed_s':time.time()-start,'summary':text[-16000:],'tests':names,
 'source_manifest_sha256':json.loads((r/'COLLECTFIX_SOURCE.json').read_text())['sha256'],
 'prearm_smoke':'NOT RUN: source SHA/copy prerequisite failed','new_sumo_sim_s':0}}
(a/'TESTS_v{REVISION}.json').write_text(json.dumps(doc,indent=1));print(json.dumps(doc))
'''
    save(f'TESTS_v{REVISION}.json', cf.remote('gpuR', code, timeout=1800))


def finalize():
    code = f'''
import pathlib,json,hashlib,datetime,shutil
a=pathlib.Path({A!r}); root=pathlib.Path('/root/main14_loop')
before=(a/'STATE_BEFORE.json').read_bytes()
assert (root/'LOOP14_STATE.json').read_bytes()==before, 'live state changed during verification'
state=json.loads(before)
assert hashlib.sha256(pathlib.Path(state['policy']['checkpoint']).read_bytes()).hexdigest()==state['policy']['sha256']
tests=json.loads((a/'TESTS_v2.json').read_text());assert tests['exit']==0
drivers=[]
for directory in (a.parent,a.parent/'collectfix'):
 p=directory/'PID.json'
 if not p.exists(): continue
 row=json.loads(p.read_text()); stat=pathlib.Path('/proc')/str(row['pid'])/'stat'
 f=stat.read_text().rsplit(')',1)[1].split() if stat.exists() else None
 live=bool(f and f[0]!='Z' and f[19]==row['start_ticks'])
 drivers.append({{'receipt':str(p),'pid':row['pid'],'start_ticks':row['start_ticks'],'live':live}})
assert not any(row['live'] for row in drivers), 'predecessor driver still live'
receipt={{'status':'WAIT_OWNER','t':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'launch_blocker':'gpuB read-only SSH failed before checkpoint SHA output',
 'new_owned_background_pids':[],'predecessor_drivers':drivers,'live_state_unchanged':True,
 'prior_policy_sha256_verified':state['policy']['sha256'],'charged_sim_s':state['sim_s_charged'],
 'new_seeds':[],'new_sumo_sim_s':0,'integrated_tests':tests,
 'checkpoint_staged':False,'prearm_smoke_run':False,'update10_verified':False,
 'cpuP_collection_started_by_task':False,'cpuQ_collection_started_by_task':False,
 'registration_snapshot_sha256':hashlib.sha256((a/'REGISTRATION.json').read_bytes()).hexdigest(),
 'output_free_bytes':shutil.disk_usage(root).free,'scratch_free_bytes':shutil.disk_usage(a).free}}
(a/'FINAL.json').write_text(json.dumps(receipt,indent=1));print(json.dumps(receipt))
'''
    save('FINAL.json', cf.remote('gpuR', code))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('block', 'stage', 'tests', 'finalize'))
    parser.add_argument('--revision', type=int, default=1, choices=(1, 2))
    args = parser.parse_args()
    REVISION = args.revision
    if REVISION != 1:
        R = A + f'/runtime_v{REVISION}'
    {'block': block, 'stage': stage, 'tests': tests, 'finalize': finalize}[args.action]()
