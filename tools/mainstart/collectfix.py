"""S-COLLECTFIX scoped deployment, verification and explicit owner retry receipts."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tools/mainstart/collectfix'
A = '/root/autodl-tmp/c4/v5/mainstart/collectfix'
R = A + '/runtime'
CPU = '/home/ubuntu/c4/main14/collectfix'
BLOCK = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth', 'vtype')


def remote(host, code, data=None, timeout=180):
    py = '/root/miniconda3/bin/python' if host == 'gpuR' else '/home/ubuntu/miniconda3/bin/python'
    command = shlex.join([py, '-c', 'import base64;exec(base64.b64decode(' + repr(base64.b64encode(code.encode()).decode()) + '))'])
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, command],
                       input=data, capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(p.stdout.decode(errors='replace') + p.stderr.decode(errors='replace'))
    return p.stdout.decode()


def receipt(name, text):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(text, encoding='utf-8')
    print(text)


def bundle():
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT).decode().splitlines()
    prefixes = ('src/', 'traffic_shepherd/', 'configs/', 'scripts/c4/', 'tools/loop14_checks/',
                'tools/main14/', 'tools/fastloop/', 'tools/coopw/', 'tools/nativelane/adopt.py', 'tests/unit/', 'baselines/circles/')
    files = {}
    for name in dict.fromkeys(names):
        if any(t in name.lower() for t in BLOCK):
            continue
        if not (name.startswith(prefixes) or name in ('pyproject.toml', 'tests/conftest.py', 'tests/test_speed2_flags.py',
                 'tests/test_merge_pivot_training.py', 'tests/test_merge_pivot_pre_window.py', 'scripts/round13c_waiter.py')):
            continue
        if Path(name).suffix not in ('.py', '.toml', '.json', '.onnx'):
            continue
        p = ROOT / name
        files[name] = p.read_bytes() if p.is_file() else subprocess.check_output(['git', 'show', ':' + name], cwd=ROOT)
    manifest = {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    files['COLLECTFIX_SOURCE.json'] = json.dumps({'files': manifest, 'sha256': digest}, indent=1).encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for n, b in files.items():
            info = tarfile.TarInfo(n)
            info.size = len(b)
            tar.addfile(info, io.BytesIO(b))
    return buf.getvalue(), digest


def stage():
    payload, digest = bundle()
    for host in ('gpuR', 'cpuP', 'cpuQ'):
        target = A if host == 'gpuR' else CPU
        parent = '/root/autodl-tmp/c4/v5/mainstart' if host == 'gpuR' else '/home/ubuntu/c4/main14'
        code = f'''
import pathlib,tarfile,io,sys,json,hashlib
parent=pathlib.Path({parent!r}); assert parent.is_dir()
a=pathlib.Path({target!r}); a.mkdir(exist_ok=True)
assert not (a/'PID.json').exists(), 'never mutate an installed driver'
r=a/'runtime'; r.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for m in tar:
  assert m.isfile() and (r/m.name).resolve().is_relative_to(r.resolve())
  assert not any(t in m.name.lower() for t in {BLOCK!r})
  tar.extract(m,r)
doc=json.loads((r/'COLLECTFIX_SOURCE.json').read_text())
for n,h in doc['files'].items(): assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
print(json.dumps({{'host':{host!r},'files':len(doc['files']),'source_manifest_sha256':doc['sha256']}}))
'''
        if host != 'gpuR':
            code += f'''
base=parent/'runtime'
for name in ('scenarios','assets'):
 p=r/name
 if not p.exists(): p.symlink_to(base/name,target_is_directory=True)
 assert p.resolve()==(base/name).resolve()
sys.path[:0]=[str(r/'src'),str(r)]
from traffic_shepherd.config import load_config
from traffic_shepherd.c4.collector_readiness import check_controller_assets
c=load_config(); assert c.network_path.is_file() and c.scenario_template_path.is_file()
controller=check_controller_assets(c)
import torch,numpy,libsumo,traci,sumolib
ready={{'source_manifest_sha256':doc['sha256'],'source_files':doc['files'],
 'fixture_verified_without_new_sumo':True,'controller':controller,'new_sumo_sim_s':0}}
(a/'COLLECTION_READY.json').write_text(json.dumps(ready,indent=1))
print(json.dumps({{'host':{host!r},'controller':controller,'ready':True}}))
'''
        receipt('STAGE_' + host + '.jsonl', remote(host, code, payload, timeout=300))
    receipt('SOURCE.json', json.dumps({'source_manifest_sha256': digest, 'bytes': len(payload)}, indent=1))


def tests():
    code = f'''
import os,sys,pathlib,subprocess,json,time
r=pathlib.Path({R!r}); a=r.parent; assert r.is_dir()
sys.path[:0]=[str(r),str(r/'src'),str(r/'scripts/c4')]
import main14_autolaunch as auto
names=[str(r/'tests/unit'/n) for n in auto.TESTS]
names += [str(r/'tests/unit'/n) for n in ('test_fastloop.py','test_mainstart.py','test_collectfix.py')]
names += [str(r/'tests/test_speed2_flags.py')]
env={{**os.environ,'PYTHONPATH':str(r/'src')+':'+str(r),'CUDA_VISIBLE_DEVICES':'',
 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1',
 'C4_ALLOWED_ROOTS':''}}
start=time.time()
with (a/'integrated_tests.log').open('wb') as log:
 p=subprocess.run([sys.executable,'-m','pytest','-q','-o','cache_dir='+str(a/'pytest_cache'),
 '--basetemp',str(a/'pytest_tmp'),*names],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT)
text=(a/'integrated_tests.log').read_text()
doc={{'exit':p.returncode,'elapsed_s':time.time()-start,'summary':text[-12000:],'tests':names,
 'source_manifest_sha256':json.loads((r/'COLLECTFIX_SOURCE.json').read_text())['sha256']}}
(a/'TESTS.json').write_text(json.dumps(doc,indent=1)); print(json.dumps(doc))
'''
    receipt('TESTS.json', remote('gpuR', code, timeout=1800))


def dry_run():
    code = f'''
import os,pathlib,subprocess,sys,json,time
r=pathlib.Path({R!r}); a=r.parent; assert r.is_dir()
env={{**os.environ,'PYTHONPATH':str(r/'src')+':'+str(r),'CUDA_VISIBLE_DEVICES':'',
 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','C4_ALLOWED_ROOTS':str(a)}}
out=a/('dry_run_'+str(time.time_ns()))
p=subprocess.run([sys.executable,str(r/'tools/fastloop/dry_run.py'),'--out',str(out)],cwd=r,env=env,capture_output=True,text=True)
assert p.returncode==0,p.stdout+p.stderr
doc=json.loads((out/'DRY_RUN.json').read_text())
doc['source_manifest_sha256']=json.loads((r/'COLLECTFIX_SOURCE.json').read_text())['sha256']
(a/'DRY_RUN.json').write_text(json.dumps(doc,indent=1))
print(json.dumps(doc))
'''
    receipt('DRY_RUN.json', remote('gpuR', code, timeout=600))


def prepare():
    """Create a proposal; the live state remains pinned until the shared register is appended."""
    code = rf'''
import pathlib,json,hashlib,copy,sys,shutil,fcntl
a=pathlib.Path({A!r}); r=a/'runtime'; original=a.parent; root=pathlib.Path('/root/main14_loop')
assert not (a/'ARMED.json').exists()
source=json.loads((r/'COLLECTFIX_SOURCE.json').read_text())
tests=json.loads((a/'TESTS.json').read_text()); dry=json.loads((a/'DRY_RUN.json').read_text())
assert tests['exit']==0 and dry['status']=='PASS' and dry['worlds']==1
assert tests['source_manifest_sha256']==dry['source_manifest_sha256']==source['sha256']
state=json.loads((root/'LOOP14_STATE.json').read_text())
assert state['cycles'][0]['phases']['collect']['status']=='failed' and not state['stopped']
assert state['sim_s_charged']==29133 and state['seeds_used']==list(range(968103,968116))
with (original/'driver.lock').open('a') as lock: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
cfg=json.loads((original/'CONFIG.json').read_text())
prior_reg=json.loads(pathlib.Path(cfg['registration']).read_text())
assert hashlib.sha256(pathlib.Path(cfg['registration']).read_bytes()).hexdigest()==state['registration_sha256']
reg=copy.deepcopy(prior_reg)
cfg['registration']=str(a/'REGISTRATION.json'); cfg['hosts']['gpu']['code_root']=str(r)
for box,h in cfg['mainstart']['collect_hosts'].items():
 h['code_root']={CPU!r}+'/runtime'; cfg['hosts'][box]=copy.deepcopy(h)
cfg['hosts']['cpu']=copy.deepcopy(cfg['hosts']['cpuP'])
cfg['collect'].update(work_root={CPU!r}+'/work',workers=7)
cfg['collect']['env']['CUDA_VISIBLE_DEVICES']=''
cfg['fastloop']={{'speed2_policy':True,'two_box':True,'spot_resume':True,'overlap_policy':False,
 'cpu_checks':False,'repoint':False,'base_three':False,'boxes':['cpuP','cpuQ'],'workers_per_box':7}}
cfg['collectfix']=True
cfg['policy']['allowed_roots'].append(str(a))
args=cfg['policy']['extra_args']
for flag,name in (('--gate-panel','gate_panel.json'),('--horizon-gate','horizon_gate_current.json')):
 old=args[args.index(flag)+1]; shutil.copyfile(old,a/name); args[args.index(flag)+1]=str(a/name)
text=json.dumps(cfg,indent=1)+'\n'; (a/'CONFIG.json').write_text(text)
reg['budget'].update(sim_s_per_cycle=58266,sim_s_total=233064)
reg['collection'].update(host='cpuP+cpuQ',workers_max=12,workers_per_box=7,
 primary_sim_s_per_cycle=29133,retry_reserve_sim_s_per_cycle=29133)
reg['sources'].update(config_sha256=hashlib.sha256(text.encode()).hexdigest(),
 source_manifest_sha256=source['sha256'],fastloop_options=cfg['fastloop'],fastloop_commit='72300236')
reg['collectfix_amendment']={{'id':'S-COLLECTFIX','owner_task':'2026-10-26 13:30 UTC+8',
 'predecessor_registration_sha256':state['registration_sha256'],
 'prior_charge_sim_s':29133,'prior_actual_sim_s':0,'prior_seeds':state['seeds_used'],
 'accounting':'No refund; official retry allocates fresh worlds after legacy batch; later journal retries reuse only their own worlds',
 'scheduling':'Deterministic alternating cpuP/cpuQ, 7 one-world workers each, numerical threads1; caps<=12 alongside scoring',
 'policy_flags':'SAFE SPEED2 only: PIVOT_DIAG_OFF=1, GRAD_AUDIT_EVERY=10, numerical accelerations OFF; WM flags OFF/audit1',
 'gates':'419 integrated CPU tests and one-world mechanics dry run PASS; no SUMO consumed by dry run',
 'preserved':'seeds/exclusions, 4 cycles, policy150/WM40, native-lane ON, existing base-boundary comparator and all scientific stops'}}
for key in ('seed_block','exclude','stop'): assert reg[key]==prior_reg[key]
for key,value in prior_reg['budget'].items():
 if key not in ('sim_s_per_cycle','sim_s_total'): assert reg['budget'][key]==value
(a/'REGISTRATION.json').write_text(json.dumps(reg,indent=1)+'\n')
arm=json.loads((original/'ARMED_V2.json').read_text())
arm.update(config=str(a/'CONFIG.json'),registration=str(a/'REGISTRATION.json'),source_files=source['files'],
 native_source_manifest_sha256=source['sha256'],config_sha256=reg['sources']['config_sha256'],
 registration_sha256=hashlib.sha256((a/'REGISTRATION.json').read_bytes()).hexdigest(),
 wrapper_sha256=hashlib.sha256((r/'scripts/c4/mainstart.py').read_bytes()).hexdigest(),amendment='S-COLLECTFIX')
# This exact S-PAIRFT incarnation is independently recorded in the shared register; no process is signalled.
stat=pathlib.Path('/proc/34370/stat')
if stat.exists() and stat.read_text().rsplit(')',1)[1].split()[19]=='4161932163':
 arm['pairft_incarnations']['34370']='4161932163'
(a/'ARMED.json').write_text(json.dumps(arm,indent=1)+'\n')
shutil.copyfile(root/'LOOP14_STATE.json',a/'STATE_BEFORE.json')
shutil.copyfile(original/'WAIT_OWNER.json',a/'WAIT_OWNER.json')
summary=json.loads((root/'collect/c00_a1_COLLECT_SUMMARY.json').read_text())
assert summary['simulated_seconds']==summary['pair_sim_seconds']==0 and summary['episodes_failed']==13
account={{'attempt':1,'host':'cpuP','seeds':state['seeds_used'],'charged_sim_s':29133,
 'actual_sim_s':0,'basis':'collector summary: simulated_seconds=0, pair_sim_seconds=0; all13 reset failures before first time step',
 'root_cause':'missing frozen CIRCLES model.onnx; PROJ warnings were not the fatal traceback',
 'refund':False,'summary_sha256':hashlib.sha256((root/'collect/c00_a1_COLLECT_SUMMARY.json').read_bytes()).hexdigest()}}
(a/'ACCOUNTING.json').write_text(json.dumps(account,indent=1))
proposal={{'config_sha256':arm['config_sha256'],'registration_sha256':arm['registration_sha256'],
 'source_manifest_sha256':source['sha256'],'wrapper_sha256':arm['wrapper_sha256'],
 'previous_registration_sha256':state['registration_sha256'],'accounting':account,
 'pairft_incarnations':arm['pairft_incarnations'],'fastloop':cfg['fastloop']}}
(a/'PROPOSAL.json').write_text(json.dumps(proposal,indent=1)); print(json.dumps(proposal))
'''
    receipt('PROPOSAL.json', remote('gpuR', code))


def install():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    register = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
    proposal = json.loads((OUT / 'PROPOSAL.json').read_text())
    text = register.read_text(encoding='utf-8')
    assert 'S-COLLECTFIX — registered BEFORE retry' in text and proposal['registration_sha256'] in text
    shared_sha = hashlib.sha256(register.read_bytes()).hexdigest()
    code = f'''
import pathlib,json,sys,os,fcntl,hashlib
a=pathlib.Path({A!r}); root=pathlib.Path('/root/main14_loop'); r=a/'runtime'
sys.path[:0]=[str(r/'scripts/c4'),str(r),str(r/'src')]
import loop14 as loop
state=json.loads((root/'LOOP14_STATE.json').read_text()); before=json.loads((a/'STATE_BEFORE.json').read_text())
assert state==before and not state['stopped']; assert not (a/'INSTALLED.json').exists()
with (a.parent/'driver.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 old=state['registration_sha256']; state['registration']=str(a/'REGISTRATION.json')
 state['registration_sha256']=loop.sha256(a/'REGISTRATION.json')
 loop.append(root/'cycle_ledger.jsonl',{{'event':'registration_amendment','id':'S-COLLECTFIX',
  'previous_registration_sha256':old,'registration_sha256':state['registration_sha256'],
  'shared_register_sha256':{shared_sha!r},'code_commit':{commit!r},'charged_sim_s':state['sim_s_charged'],
  'prior_actual_sim_s':0,'owner_authorized_retry':True}})
 loop.save_state({{'loop_root':str(root)}},state)
 cfg=loop.load_config(a/'CONFIG.json'); loop.load_state(cfg)
 doc={{'commit':{commit!r},'shared_register_sha256':{shared_sha!r},'registration_sha256':state['registration_sha256']}}
 loop.write_atomic(a/'INSTALLED.json',doc)
 print(json.dumps(doc))
'''
    receipt('INSTALLED.json', remote('gpuR', code))


def launch():
    code = f'''
import os,sys,pathlib,subprocess,json,time
a=pathlib.Path({A!r}); r=a/'runtime'; assert (a/'INSTALLED.json').is_file()
assert not (a/'LAUNCH.json').exists()
env={{**os.environ,'C4_MAINSTART_CODE':str(r),'PYTHONPATH':str(r/'src')+':'+str(r),
 'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}}
argv=[sys.executable,'-u',str(r/'scripts/c4/mainstart.py'),'--arm',str(a/'ARMED.json')]
p=subprocess.run([*argv,'--check-only'],cwd=r,env=env,capture_output=True,text=True,timeout=300)
assert p.returncode==0,p.stdout+p.stderr
with (a/'driver.log').open('ab') as log:
 child=subprocess.Popen(['setsid','nohup',*argv,'--retry-collect'],cwd=r,env=env,stdin=subprocess.DEVNULL,
 stdout=log,stderr=subprocess.STDOUT)
time.sleep(5); assert child.poll() is None,(a/'driver.log').read_text()[-4000:]
f=pathlib.Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()
doc={{'pid':child.pid,'ppid':int(f[1]),'sid':int(f[3]),'start_ticks':f[19],
 'log':str(a/'driver.log'),'official_retry':['loop14.py','retry','--config',str(a/'CONFIG.json'),'collect'],
 'check_only':json.loads(p.stdout)}}
(a/'LAUNCH.json').write_text(json.dumps(doc,indent=1)); print(json.dumps(doc))
'''
    receipt('LAUNCH.json', remote('gpuR', code, timeout=420))


def progress():
    code = f'''
import pathlib,json,sys,subprocess,shlex
a=pathlib.Path({A!r}); root=pathlib.Path('/root/main14_loop')
state=json.loads((root/'LOOP14_STATE.json').read_text()); journal=json.loads((root/'worlds_c00.json').read_text())
pid=json.loads((a/'PID.json').read_text()); fields=pathlib.Path('/proc/'+str(pid['pid'])+'/stat').read_text().rsplit(')',1)[1].split()
assert fields[19]==pid['start_ticks']
doc={{'driver':{{**pid,'ppid':int(fields[1]),'sid':int(fields[3])}},'phase':state['cycles'][-1]['phases']['collect'],
 'charged_sim_s':state['sim_s_charged'],'seeds_used':state['seeds_used'],'world_attempts':journal['attempts'],
 'worlds_complete':len(journal['worlds']),'journal_charged':journal['charged'],'boxes':{{}}}}
probe="""import pathlib,json,os
root=pathlib.Path('/home/ubuntu/c4/main14/collectfix')
rows=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit(): continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace')
  if 'base64.b64decode' in cmd or '-c import pathlib,json,os' in cmd: continue
  if str(root) not in cmd: continue
  f=(p/'stat').read_text().rsplit(')',1)[1].split()
  rows.append({'pid':int(p.name),'ppid':int(f[1]),'start_ticks':f[19],'cpu_ticks':int(f[11])+int(f[12]),'cmd':cmd[:500]})
 except (OSError,PermissionError): pass
files=[]
for p in (root/'work').glob('w*/C4-LOOP14-COOPON-20261026/c00/collect/episodes/policy/*/episode.npz'):
 files.append({'path':str(p),'bytes':p.stat().st_size})
print(json.dumps({'processes':rows,'recorded_episodes':files}))
"""
for box in ('cpuP','cpuQ'):
 cmd=shlex.join(['/home/ubuntu/miniconda3/bin/python','-c',probe])
 p=subprocess.run(['ssh','-o','BatchMode=yes',box,cmd],capture_output=True,text=True,timeout=60)
 assert p.returncode==0,p.stderr
 doc['boxes'][box]=json.loads(p.stdout)
doc['driver_log_tail']=(a/'driver.log').read_text()[-4000:]
(a/'PROGRESS.json').write_text(json.dumps(doc,indent=1)); print(json.dumps(doc))
'''
    receipt('PROGRESS.json', remote('gpuR', code, timeout=180))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('stage', 'tests', 'dry-run', 'prepare', 'install', 'launch', 'progress'))
    args = p.parse_args()
    {'stage': stage, 'tests': tests, 'dry-run': dry_run, 'prepare': prepare,
     'install': install, 'launch': launch, 'progress': progress}[args.action]()


if __name__ == '__main__':
    main()
