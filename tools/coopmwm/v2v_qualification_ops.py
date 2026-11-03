"""Isolated, receipt-adopting Linux CPU qualification of the merged V2V release.

No live training source or binding is modified; subprocess physics is forbidden
inside the inherited complete release suite. Launching SUMO is not an action here.
"""
import argparse
import base64
import hashlib
import json
import time
import zlib
from . import ops
from .ls_validity_ops import remote
from .protocol import checked
from .storage import write

BASE = '/home/ubuntu/coopmwm20261103_v2vcpu2'
GENERATION = 'V4'
SOURCE = '/home/ubuntu/coopmwm20261103_v2vcpu4'
FILES = (
    'tools/coopmwm/bank_v2v.py', 'tools/coopmwm/v2v_obs.py',
    'tools/coopmwm/v2v_model.py', 'tools/coopmwm/v2v_integrate.py',
    'tools/coopmwm/replay_bank.py', 'tools/coopmwm/policy_synergy.py',
    'tools/coopmwm/sumo_verify.py', 'tools/coopmwm/warm_runner.py',
    'tools/coopmwm/v2v_runner.py', 'tools/coopmwm/test_v2v_runner_cpu.py',
    'tools/coopmwm/test_v2v_model_cpu.py', 'tools/coopmwm/test_v2v_int_e2e_cpu.py',
    'tools/coopmwm/test_replay_bank_v2v_cpu.py',
    'tools/coopmwm/v2v_runtime.py', 'tools/coopmwm/test_v2v_runtime_cpu.py',
    'tools/coopmwm/accel_plant.py',
    'traffic_shepherd/c4/env.py', 'traffic_shepherd/ownership.py',
    'tests/unit/test_c4_v2v_obs.py', 'tests/fixtures/v2v_golden.npz',
    'V2V_INT_READY.json',
)


def qualify():
    files = {n: base64.b64encode(checked(ops.LOCAL/n).read_bytes()).decode() for n in FILES}
    packed = base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
    code = f'''from pathlib import Path
import json,base64,zlib,shutil,hashlib,subprocess,os,time,fcntl
r=Path({SOURCE!r});base=Path({BASE!r});assert r.parent.is_dir() and base.is_dir()
r.mkdir(exist_ok=True)
lock=(r/'QUALIFY.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
dispatch=r/'LS_CPU_DISPATCH.json';done=r/'LS_CPU_RESULT.json'
if done.exists():print(done.read_text());raise SystemExit
if dispatch.exists():print(dispatch.read_text());raise SystemExit
files=json.loads(zlib.decompress(base64.b64decode({packed!r})))
if not (r/'STAGED.json').exists():
 for n in ('pytools','WM_PACKAGE'):
  if not (r/n).exists():shutil.copytree(base/n,r/n,ignore=shutil.ignore_patterns('__pycache__'))
 for n in ('FROZEN_COSTS.json','FROZEN_MOTION_SOURCE.py','FROZEN_GEOMETRY_SOURCE.py','FROZEN_NETWORK.net.xml.gz','FROZEN_PRE_NICHE_SEARCH.py'):
  if not (r/n).exists():shutil.copy2(base/n,r/n)
 candidates=[Path('/home/ubuntu/e3final/runtime/src/traffic_shepherd'),Path('/home/ubuntu/e3final/runtime/traffic_shepherd')]
 present=[p for p in candidates if p.is_dir()];assert len(present)==1,present
 runtime=present[0]
 for p in runtime.rglob('*.py'):
  assert not any(s in str(p).lower() for s in ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vehicle_params','driver_params'))
  assert not p.is_symlink()
  dest=r/'runtime_src/traffic_shepherd'/p.relative_to(runtime);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 for name,data in files.items():
  target=(r/'pytools'/name if name.startswith('tools/') else r/'runtime_src'/name if name.startswith('traffic_shepherd/') else r/name)
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(base64.b64decode(data))
 tests=json.loads((base/'LS_RELEASE_TESTS.json').read_text())
 tests+=['test_v2v_runner_cpu','test_v2v_model_cpu','test_v2v_int_e2e_cpu','test_replay_bank_v2v_cpu','test_v2v_runtime_cpu']
 with (r/'LS_RELEASE_TESTS.json').open('x') as f:json.dump(sorted(set(tests)),f)
 selection=json.loads((base/'LS_CPU_SELECTION.json').read_text())
 selection['pytest_args']=[x.replace(str(base),str(r)) for x in selection.get('pytest_args',[])]
 if str(r/'tests/unit/test_c4_v2v_obs.py') not in selection['pytest_args']:selection['pytest_args'].append(str(r/'tests/unit/test_c4_v2v_obs.py'))
 with (r/'LS_CPU_SELECTION.json').open('x') as f:json.dump(selection,f)
 hashes={{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for parent in (r/'pytools',r/'runtime_src',r/'tests') for p in parent.rglob('*') if p.is_file() and not p.is_symlink()}}
 for n in ('FROZEN_COSTS.json','FROZEN_MOTION_SOURCE.py','FROZEN_GEOMETRY_SOURCE.py','FROZEN_NETWORK.net.xml.gz','FROZEN_PRE_NICHE_SEARCH.py','V2V_INT_READY.json'):hashes[n]=hashlib.sha256((r/n).read_bytes()).hexdigest()
 with (r/'LS_SOURCE_HASHES.json').open('x') as f:json.dump(hashes,f,sort_keys=True)
 with (r/'STAGED.json').open('x') as f:json.dump(dict(base=str(base),integration='c4/v2v-int 816d045a + current runner; merged in c4/coopnative',sim_s=0),f)
for name,data in files.items():
 p=(r/'pytools'/name if name.startswith('tools/') else r/'runtime_src'/name if name.startswith('traffic_shepherd/') else r/name)
 assert p.read_bytes()==base64.b64decode(data),name
env=dict(os.environ,PYTHONPATH=str(r/'pytools')+':'+str(r/'WM_PACKAGE')+':'+str(r/'runtime_src')+':/home/ubuntu/e3final/runtime',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',COOPMWM_NICHE_BASELINE=str(r/'FROZEN_PRE_NICHE_SEARCH.py'),COOPMWM_NULLFIX_GEOMETRY=str(r/'FROZEN_GEOMETRY_SOURCE.py'),COOPMWM_NULLFIX_NETWORK=str(r/'FROZEN_NETWORK.net.xml.gz'))
code='from tools.coopmwm.nullfix_geometry import install;install('+repr(str(r/'FROZEN_GEOMETRY_SOURCE.py'))+');from tools.coopmwm.ls_cpu_test import run;run('+repr(str(r))+')'
with (r/'LS_CPU.log').open('x') as log:
 p=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-u','-c',code],env=env,cwd=str(r),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
v=dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),source_root=str(r),sim_s=0)
with dispatch.open('x') as f:json.dump(v,f)
print(json.dumps(v))'''
    value = json.loads(remote('cpuQ', code, timeout=60))
    target = ops.EVIDENCE/f'V2V_LINUX_CPU_DISPATCH_{GENERATION}.json'
    if not target.exists(): write(target, value)
    print(json.dumps(value))


def status():
    value = json.loads(remote('cpuQ', f'''from pathlib import Path
import json,time
r=Path({SOURCE!r});d=json.loads((r/'LS_CPU_DISPATCH.json').read_text());p=Path('/proc',str(d['pid']))
alive=False
if p.exists():
 s=(p/'stat').read_text().split();alive=s[21]==d['start_ticks'] and s[2]!='Z'
done=r/'LS_CPU_RESULT.json'
print(json.dumps(dict(unix=time.time(),owner=d,alive=alive,result=json.loads(done.read_text()) if done.exists() else None,log=(r/'LS_CPU.log').read_text()[-16000:])))''', timeout=35))
    write(ops.EVIDENCE/f'V2V_LINUX_CPU_STATUS_{time.time_ns()}.json', value)
    if value['result']:
        p = ops.EVIDENCE/f'V2V_LINUX_CPU_RESULT_{GENERATION}.json'
        if not p.exists(): write(p, value['result'])
    result = value.pop('result')
    if result:
        value['result'] = {k: result[k] for k in ('pass_', 'tests', 'skips', 'sim_s', 'wall_s', 'root')}
        if result['pass_']: value['log'] = 'Complete; full log retained in immutable remote source root.'
    print(json.dumps(value))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('qualify', 'status'))
    globals()[p.parse_args().action]()
