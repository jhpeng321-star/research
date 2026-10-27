"""S-C2PREP sequential transport and read-only progress inspection."""
import argparse
import base64
import hashlib

from tools.loopresume.c1collect import PRELUDE
from tools.loopresume.operate import ROOT, remote

NAMES = ['configs/c4_policy14_c2prep.toml', 'tools/loopresume/c2prep.py',
         'tools/loopresume/c2prep_common.py', 'tools/loopresume/c2prep_smoke.py',
         'tools/loopresume/c2prep_box.py', 'tests/unit/test_c2prep.py', 'tests/unit/test_longcritic.py']


def status():
    remote('gpuA', PRELUDE+'''
cfg=loop.load_config(a/'CONFIG.json'); s=loop.load_state(cfg)
print(json.dumps({'time':loop.now(),'policy_config':cfg['policy'],'loopresume':cfg.get('loopresume'),
 'resumeprep':cfg.get('resumeprep'),'charged':s['sim_s_charged'],'stopped':s.get('stopped'),
 'wm':s['accepted_wm'],'policy':s['policy'],'cycles':[{ 'cycle':c['cycle'],
 'phases':{n:{k:v for k,v in p.items() if k!='receipt'} for n,p in c['phases'].items()}}
 for c in s['cycles']], 'driver':loop.read_json(a/'PID.json')}))
for n in ('C1COLLECT4_COMPLETE.json','C1COLLECT4_RETRY.json'):
 if (r/n).exists(): print(n,json.dumps(loop.read_json(r/n)))
for n in ('arm/loop/wm/c01/train_log.jsonl','arm/loop/wm/c01.log'):
 if (r/n).exists(): print(n,(r/n).read_text()[-3000:])
print((a/'driver_c1collect4.log').read_text()[-7000:])
''')


def inspect():
    remote('gpuA', PRELUDE+'''
cfg=loop.load_config(a/'CONFIG.json'); s=loop.load_state(cfg)
os.chdir(root); os.environ['C4_ALLOWED_ROOTS']=':'.join(cfg['policy']['allowed_roots'])
from wms.control.policy14_cli import _read_config
raw=_read_config(cfg['policy']['config']); print('RESOLVED',json.dumps(raw))
c=s['cycles'][1]['phases']['convert']['receipt']
print('CONVERT',json.dumps({k:v for k,v in c.items() if k not in ('sumocredit','artifacts')}))
print('REG_BUDGET',json.dumps(loop.check_registration(cfg)['budget']))
print('SUMMARY',json.dumps({k:v for k,v in loop.read_json(pathlib.Path(s['policy']['checkpoint']).with_name('RUN_SUMMARY.json')).items() if k in ('recommended_updates_per_cycle','horizon','horizon_gate')}))
import torch
ck=torch.load(s['policy']['checkpoint'],map_location='cpu',weights_only=True)
print('CHECKPOINT',json.dumps({'update_count':ck['update_count'],'anchor':ck.get('coopw',{}).get('anchor')}))
''')


def stage():
    payload = {name: base64.b64encode((ROOT/name).read_bytes()).decode() for name in NAMES}
    remote('gpuA', PRELUDE+f'''
import base64
assert not (r/'C2PREP_LAUNCH.json').exists()
source={{}}
if (r/'C2PREP_TESTS.json').exists():
 history=r/'c2prep_stage_history'; history.mkdir(exist_ok=True)
 for n in ('C2PREP_TESTS.json','C2PREP_TESTS.log','C2PREP_SOURCE.json'):
  p=r/n; (history/(n+'.'+loop.sha256(p))).write_bytes(p.read_bytes())
for name, encoded in {payload!r}.items():
 p=root/name; assert p.parent.is_dir()
 data=base64.b64decode(encoded)
 if p.exists() and p.read_bytes()!=data:
  backup=r/'c2prep_stage_history'; backup.mkdir(exist_ok=True)
  (backup/(p.name+'.'+loop.sha256(p))).write_bytes(p.read_bytes())
 p.write_bytes(data); source[name]=loop.sha256(p)
loop.write_atomic(r/'C2PREP_SOURCE.json',source)
env={{**os.environ,'PYTHONPATH':str(root/'src')+':'+str(root),'CUDA_VISIBLE_DEVICES':'',
 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1',
 'NATIVEINIT_BASELINE':str(r/'nativeinit_baseline')}}
env.pop('C4_ALLOWED_ROOTS',None)
result=subprocess.run([sys.executable,'-B','-m','pytest','-q','-ra','-p','no:cacheprovider',
 '--import-mode=importlib','--basetemp',str(r/'pytest_c2prep'), 'tests/unit/test_c2prep.py',
 'tests/unit/test_loopresume.py','tests/unit/test_nativeinit.py','tests/unit/test_loop14.py',
 'tests/unit/test_longcritic.py'],cwd=root,env=env,capture_output=True,text=True)
(r/'C2PREP_TESTS.log').write_text(result.stdout+result.stderr)
doc={{'t':loop.now(),'exit':result.returncode,'source_sha256':loop.sha256(r/'C2PREP_SOURCE.json'),
 'log_sha256':loop.sha256(r/'C2PREP_TESTS.log')}}
loop.write_atomic(r/'C2PREP_TESTS.json',doc)
print(result.stdout+result.stderr); print(json.dumps(doc)); raise SystemExit(result.returncode)
''', timeout=240)


def launch():
    registration = ROOT.parent.parent/'research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md'
    # The authoritative Markdown register is appended locally before this call.
    assert '## S-C2PREP — prospective main cycle-2' in registration.read_text(encoding='utf-8')
    register_sha = hashlib.sha256(registration.read_bytes()).hexdigest()
    remote('gpuA', PRELUDE+f'''
assert not (r/'C2PREP_LAUNCH.json').exists()
tests=loop.read_json(r/'C2PREP_TESTS.json'); assert tests['exit']==0
assert tests['source_sha256']==loop.sha256(r/'C2PREP_SOURCE.json')
source=loop.read_json(r/'C2PREP_SOURCE.json')
for name,sha in source.items(): assert loop.sha256(root/name)==sha,name
cfg=loop.load_config(a/'CONFIG.json'); s=loop.load_state(cfg)
assert s['sim_s_charged']==176473
pid=loop.read_json(a/'PID.json'); assert pid['pid']==80364 and pid['start_ticks']=='843373550'
amendment={{'id':'S-C2PREP','authority':'main 2026-10-27 08:05 act-on-recommendation',
 'applies_to':'main C2; twin mirrors at own C2 boundary','registered_at':loop.now(),
 'markdown_registration_sha256':{register_sha!r}, 'policy_only':True,'collection_authorized':False,
 'policy_sha256':'64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c',
 'horizons_s':[10,30,60],'curriculum_updates':[0,150,190],'horizon_ceiling_s':60,
 'policy_gpu_cap':0.65,'paircrit':True,'warmup_updates':100,'breakdown':False,'conformal':False,
 'wm1_requires':'unchanged ACCEPT and matching candidate SHA',
 'resource_rule':'three isolated updates; measured max seconds/update * measured update count + elapsed <=3h; allocator<=0.65',
 'fallback':'largest current-WM gate-qualified horizon in 10/30/60 fitting memory/time',
 'native_relative_gate':'unchanged10s frozen16panel and checkpoint/WM/config vetoes',
 'stop_after_gate':'WAIT_OWNER_BUDGET','charged':176473,'collection_need':36933,'additional_cap_needed':3406}}
binding={{'t':loop.now(),'source':source,'tests':tests,'wm_driver':pid,'amendment':amendment}}
loop.write_atomic(r/'C2PREP_BINDINGS.json',binding)
loop.write_atomic(a/'loop/C2PREP_AMENDMENT.json',amendment)
with (a/'driver_c2prep.log').open('xb') as log:
 proc=subprocess.Popen([sys.executable,'-B','-u',str(root/'tools/loopresume/c2prep_box.py')],
 cwd=root,env={{**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}},
 start_new_session=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
launch={{'pid':proc.pid,'start_ticks':loop.fast_module().process_token(proc.pid),'t':loop.now(),
 'task':'S-C2PREP','bindings_sha256':loop.sha256(r/'C2PREP_BINDINGS.json')}}
loop.write_atomic(r/'C2PREP_LAUNCH.json',launch); print(json.dumps(launch))
''')


def progress():
    remote('gpuA', PRELUDE+'''
cfg=loop.load_config(a/'CONFIG.json'); s=loop.load_state(cfg)
print(json.dumps({'time':loop.now(),'charged':s['sim_s_charged'],'stopped':s.get('stopped'),
 'cycles':[{ 'cycle':c['cycle'],'phases':{n:p['status'] for n,p in c['phases'].items()}} for c in s['cycles']]}))
for name in ('C2PREP_LAUNCH.json','C2PREP_PROGRESS.json','C2PREP_SELECTION.json','C2PREP_RESULT.json'):
 p=r/name
 if p.exists():
  doc=loop.read_json(p)
  for key in ('smokes','selection','critic_warmup'): doc.pop(key,None)
  print(name,json.dumps(doc))
for name in ('arm/loop/wm/c01/train_log.jsonl','arm/loop/policy/c02/updates.jsonl'):
 p=r/name
 if p.exists() and p.stat().st_size:
  row=json.loads(p.read_text().splitlines()[-1])
  print(name,json.dumps({k:v for k,v in row.items() if k in ('update','step','seconds','loss','imagination_horizon_s','actor_loss','critic_loss')}))
p=r/'C2PREP_LAUNCH.json'
if p.exists():
 pid=loop.read_json(p); print('C2DRIVER_LIVE',previous.live(pid))
 if previous.live(pid):
  children=pathlib.Path('/proc/'+str(pid['pid'])+'/task/'+str(pid['pid'])+'/children').read_text().split()
  print('OWN_CHILDREN',json.dumps([{ 'pid':int(n),'start_ticks':loop.fast_module().process_token(int(n))} for n in children]))
p=a/'driver_c2prep.log'
if p.exists(): print('C2LOG',p.read_text()[-2500:])
''')


def revise_waiter():
    remote('gpuA', PRELUDE+'''
import signal
pid=loop.read_json(r/'C2PREP_LAUNCH.json')
assert pid['task']=='S-C2PREP' and previous.live(pid)
assert loop.read_json(r/'C2PREP_PROGRESS.json')['status']=='WAIT_WM1_ACCEPT'
assert not pathlib.Path('/proc/'+str(pid['pid'])+'/task/'+str(pid['pid'])+'/children').read_text().strip()
assert loop.load_state(loop.load_config(a/'CONFIG.json'))['cycles'][1]['phases']['wm_ft']['status']=='running'
os.kill(pid['pid'],signal.SIGTERM)
for _ in range(50):
 if not previous.live(pid): break
 time.sleep(0.1)
assert not previous.live(pid)
backup=r/'c2prep_waiter_before_resource_classification'; backup.mkdir()
for n in ('C2PREP_LAUNCH.json','C2PREP_BINDINGS.json','C2PREP_PROGRESS.json','C2PREP_SOURCE.json','C2PREP_TESTS.json','C2PREP_TESTS.log'):
 p=r/n; shutil.copy2(p,backup/n)
for name in loop.read_json(r/'C2PREP_SOURCE.json'):
 p=backup/name; p.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(root/name,p)
(r/'C2PREP_LAUNCH.json').rename(backup/'retired_launch.json')
(a/'driver_c2prep.log').rename(backup/'driver.log')
print(json.dumps({'retired_owned_waiter':pid,'children':0,'reason':'fail closed on non-resource smoke failure; no WM/check/policy touched'}))
''')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['status', 'inspect', 'stage', 'launch', 'progress', 'revise_waiter'])
    globals()[parser.parse_args().action]()
