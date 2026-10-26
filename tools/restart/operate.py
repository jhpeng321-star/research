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
    p.add_argument('action', choices=['authorize', 'qviews', 'returns', 'collectors', 'status'])
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
    elif a.action == 'collectors':
        remote('gpuA', f'''import sys,json
sys.path[:0]=[{RUNTIME!r}]
from tools.resumeprep.boxstage import stage
for host in ('cpuP','cpuQ'):
 print(json.dumps({{'host':host,**stage('local',{RUNTIME!r},host,{CPU!r},['src','traffic_shepherd','configs','scripts','tools'])}}),flush=True)
''')
        for host in ('cpuP', 'cpuQ'):
            remote(host, f'''import pathlib,os,sys,json
root=pathlib.Path({CPU!r}); old=pathlib.Path('/home/ubuntu/c4/main14/mainr16/runtime')
assert root.is_dir() and old.is_dir()
links={{}}
for name in ('baselines','assets','scenarios'):
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
