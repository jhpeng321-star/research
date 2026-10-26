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
    p.add_argument('action', choices=['authorize', 'qviews', 'returns', 'collectors', 'collector-readiness', 'registration', 'status'])
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
    elif a.action == 'registration':
        remote('gpuA', f'''import pathlib,sys,json,copy
sys.path[:0]=[{RUNTIME+'/scripts/c4'!r},{RUNTIME!r},{RUNTIME+'/src'!r}]
import loop14 as loop
arm=pathlib.Path({RELEASE+'/arm'!r}); cfg=loop.load_config(arm/'CONFIG.json'); state=loop.load_state(cfg)
reg=loop.read_json(cfg['registration'])
reg['gates']['failure']='Scientific native identity/native-relative veto/smoke behavior/update10/native-gate failure: WAIT_OWNER; owner continuation3 permits three recorded fixes/retries per non-scientific setup/staging/SSH/fixture error; E below300MB pauses'
reg['sources']['superseded_restart_sources']=copy.deepcopy(reg['sources'])
reg['sources'].update(code_sha='6e458c8b8b4448c0ea519f129365d1bfe6c7e27c',config_sha256=loop.sha256(arm/'CONFIG.json'))
loop.write_atomic(pathlib.Path(cfg['registration']),reg)
state['registration_sha256']=loop.sha256(cfg['registration']); loop.save_state(cfg,state)
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
