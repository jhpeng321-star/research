"""Sequential S-C2H60 staging, detached launch and progress verification."""
import argparse
import base64
import hashlib
from pathlib import Path

from tools.loopresume.c1collect import PRELUDE
from tools.loopresume.operate import ROOT, remote

NAMES = ['configs/c4_policy14_c2h60.toml', 'tools/loopresume/c2h60.py',
         'tools/loopresume/c2h60_box.py', 'tests/unit/test_c2h60.py']
REGISTER = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
WARM_SHA = 'ed0d4242493ba3012d440db484cb837ee785836fed31cfbd41e1e613fcc5e283'


def stage():
    payload = {name: base64.b64encode((ROOT / name).read_bytes()).decode() for name in NAMES}
    remote('gpuA', PRELUDE + f'''
import base64
assert not (r/'C2H60_LAUNCH.json').exists()
source={{}}
for name, encoded in {payload!r}.items():
 p=root/name; assert p.parent.is_dir()
 assert not p.exists(),name
 p.write_bytes(base64.b64decode(encoded)); source[name]=loop.sha256(p)
loop.write_atomic(r/'C2H60_SOURCE.json',source)
env={{**os.environ,'PYTHONPATH':str(root/'src')+':'+str(root),'CUDA_VISIBLE_DEVICES':'',
 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1',
 'NATIVEINIT_BASELINE':str(r/'nativeinit_baseline')}}
env.pop('C4_ALLOWED_ROOTS',None)
result=subprocess.run([sys.executable,'-B','-m','pytest','-q','-ra','-p','no:cacheprovider',
 '--import-mode=importlib','--basetemp',str(r/'pytest_c2h60'), 'tests/unit/test_c2h60.py',
 'tests/unit/test_nativeinit.py','tests/unit/test_c2prep.py','tests/unit/test_loop14.py'],
 cwd=root,env=env,capture_output=True,text=True)
(r/'C2H60_TESTS.log').write_text(result.stdout+result.stderr)
doc={{'t':loop.now(),'exit':result.returncode,'source_sha256':loop.sha256(r/'C2H60_SOURCE.json'),
 'log_sha256':loop.sha256(r/'C2H60_TESTS.log')}}
loop.write_atomic(r/'C2H60_TESTS.json',doc)
print(result.stdout+result.stderr); print(json.dumps(doc)); raise SystemExit(result.returncode)
''', timeout=240)


def launch():
    text = REGISTER.read_text(encoding='utf-8')
    assert '## S-C2H60 — prospective' in text and WARM_SHA in text
    register_sha = hashlib.sha256(REGISTER.read_bytes()).hexdigest()
    remote('gpuA', PRELUDE + f'''
assert not (r/'C2H60_LAUNCH.json').exists()
tests=loop.read_json(r/'C2H60_TESTS.json'); assert tests['exit']==0
source=loop.read_json(r/'C2H60_SOURCE.json')
assert tests['source_sha256']==loop.sha256(r/'C2H60_SOURCE.json')
for name,sha in source.items(): assert loop.sha256(root/name)==sha,name
stop=loop.read_json(r/'C2H60_STOP_REQUEST.json')
assert not previous.live(stop['coordinator']) and not previous.live(stop['child'])
import torch
warm=a/'loop/policy/c02/policy_latest.pt'
assert loop.sha256(warm)=={WARM_SHA!r}
ck=torch.load(warm,map_location='cpu',weights_only=True)
assert ck['format']=='c4_dreamer/3' and ck['update_count']==233 and ck['coopw']['anchor']==0
required=('policy','group_encoder','critic','target_critic','actor_optimizer','critic_optimizer','rng','sampler','paircrit')
assert all(key in ck for key in required)
def finite(value):
 if isinstance(value,torch.Tensor): return bool(torch.isfinite(value).all())
 if isinstance(value,dict): return all(finite(v) for v in value.values())
 if isinstance(value,(list,tuple)): return all(finite(v) for v in value)
 return True
assert all(finite(ck[key]) for key in required)
cfg=loop.load_config(a/'CONFIG.json'); state=loop.load_state(cfg)
assert state['sim_s_charged']==176473
gate=loop.read_json(a/'horizon_gate_current.json')
assert gate['wm_sha256']==state['accepted_wm']['sha256'] and gate['horizons']['60']['verdict']=='PASS'
summary=loop.read_json(warm.parent/'RUN_SUMMARY.json')
assert summary['error']=='RuntimeError: own-run STOP requested; checkpoint saved'
warm_start={{'checkpoint':str(warm),'sha256':loop.sha256(warm),'update':233,'resumable':True,
 'choice':'latest30s C2 full checkpoint; C1fallback unnecessary'}}
superseded={{'coordinator':stop['coordinator'],'child':stop['child'],'update':233,
 'checkpoint':str(warm),'checkpoint_sha256':loop.sha256(warm),'out':str(warm.parent),
 'summary_sha256':loop.sha256(warm.parent/'RUN_SUMMARY.json'),'retained':True}}
amendment={{'id':'S-C2H60','t':loop.now(),'authority':'main2026-10-27 11:20 act-on-recommendation',
 'markdown_registration_sha256':{register_sha!r},'warm_start':warm_start,'superseded':superseded,
 'horizon_s':60,'logical_updates':150,'curriculum_updates':[0,150,233],
 'low_share_mask':'unchanged','paircrit':True,'gpu_cap':0.65,'time_limit_waived':True,
 'native_gate':'unchanged frozen16/10s earliest eligible minimum; phase-relative10progress',
 'after_gate':'WAIT_OWNER_BUDGET unless SHA-bound owner-approved registered budget raise',
 'charged':176473,'cap':210000,'collection_need':36933}}
binding={{'t':loop.now(),'source':source,'tests':tests,'amendment':amendment,
 'warm_start':warm_start,'superseded':superseded,'wm_sha256':state['accepted_wm']['sha256'],
 'gate_sha256':loop.sha256(a/'horizon_gate_current.json'),
 'pairing_source_policy_sha256':ck['paircrit']['policy_checkpoint_sha256']}}
loop.write_atomic(r/'C2H60_BINDINGS.json',binding)
with (a/'driver_c2h60.log').open('xb') as log:
 proc=subprocess.Popen([sys.executable,'-B','-u',str(root/'tools/loopresume/c2h60_box.py')],
 cwd=root,env={{**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}},
 start_new_session=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
launch={{'pid':proc.pid,'start_ticks':loop.fast_module().process_token(proc.pid),'t':loop.now(),
 'task':'S-C2H60','bindings_sha256':loop.sha256(r/'C2H60_BINDINGS.json')}}
loop.write_atomic(r/'C2H60_LAUNCH.json',launch); print(json.dumps(launch))
''')


def progress():
    remote('gpuA', PRELUDE + '''
import math,datetime
from tools.loopresume.c2h60_box import verify_updates
binding=loop.read_json(r/'C2H60_BINDINGS.json'); launch=loop.read_json(r/'C2H60_LAUNCH.json')
doc=loop.read_json(r/'C2H60_PROGRESS.json')
print('PROGRESS',json.dumps(doc)); print('COORDINATOR_LIVE',previous.live(launch))
out=a/'loop/policy/c02.h60'; rows=[]
if (out/'updates.jsonl').exists():
 for line in (out/'updates.jsonl').read_text().splitlines():
  try: rows.append(json.loads(line))
  except json.JSONDecodeError: pass
if rows:
 verify_updates(rows,binding['warm_start']['update'],len(rows))
 times=[row['wall_time_s'] for row in rows]
 samples=[json.loads(line) for line in (r/'C2H60_MEMORY.jsonl').read_text().splitlines()]
 # Warm allocator capacity is assessed over updates5+ separately from initial growth.
 stable=[s for s in samples if s.get('update',0)>=binding['warm_start']['update']+5 and 'used_memory_mib' in s]
 mem=[s['used_memory_mib'] for s in stable]
 elapsed=time.time()-doc['started_unix']; seconds_per_update=elapsed/len(rows)
 remaining=(150-len(rows))*seconds_per_update
 result={'t':loop.now(),'updates':len(rows),'last_update':rows[-1]['update'],
  'horizon_counts':{str(h):sum(row['imagination_horizon_s']==h for row in rows) for h in (10,60)},
  'finite_losses_and_gradients':True,'training_mean_seconds':sum(times)/len(times),
  'observed_wall_seconds_per_update':seconds_per_update,'updates_per_hour':3600/seconds_per_update,
  'eta_utc':datetime.datetime.fromtimestamp(time.time()+remaining,datetime.timezone.utc).isoformat(),
  'remaining_hours':remaining/3600,'memory_samples_updates5plus':len(mem),
  'memory_mib_min':min(mem) if mem else None,'memory_mib_max':max(mem) if mem else None,
  'memory_samples':stable,'last_metrics':{k:rows[-1][k] for k in ('actor_loss','critic_loss','actor_grad_norm','critic_grad_norm')},
  'gate':loop.read_json(out/'gate_panel_result.json') if (out/'gate_panel_result.json').exists() else None,
  'coordinator':launch,'child':doc['child'],'coordinator_live':previous.live(launch),
  'child_live':previous.live(doc['child']),'charged':loop.load_state(loop.load_config(a/'CONFIG.json'))['sim_s_charged']}
 loop.write_atomic(r/'C2H60_OBSERVATION.json',result)
 print('OBSERVATION',json.dumps({k:v for k,v in result.items() if k!='memory_samples'}))
print('LOG',(a/'driver_c2h60.log').read_text()[-4000:])
''')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['stage', 'launch', 'progress'])
    globals()[parser.parse_args().action]()
