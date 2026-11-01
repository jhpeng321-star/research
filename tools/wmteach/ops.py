"""Short explicit SSH operations, never process-name signals or implicit dispatch."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PHYS = '/root/rbeval20261031/phys_code_22cc32f65a80'
REMOTE = '/root/autodl-tmp/wmteach20261101_v1'


def remote(source, host='gpuA', timeout=60):
    command = '/root/miniconda3/bin/python -c "import sys;exec(compile(sys.stdin.read(),\'<wmteach-remote>\',\'exec\'))"'
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, command],
                       input=source,capture_output=True, text=True, timeout=timeout, encoding='utf-8')
    if r.returncode:
        raise RuntimeError(r.stdout+r.stderr)
    return r.stdout


def inspect():
    return remote(f'''import pathlib,json,os,subprocess,shutil
roots={[PHYS, REMOTE, '/root/autodl-tmp/coopbandit20261031_v3', '/root/autodl-tmp/coopplan20261101_v1']!r}
print(json.dumps(dict(load=os.getloadavg(),disk_free=shutil.disk_usage('/root/autodl-tmp').free,
 roots={{r:([p.name for p in pathlib.Path(r).iterdir() if p.name in
 ('exam','pytools','jobs','BINDING.json','configs','artifacts','tools','COMPLETE.json','PROCESS.json','IDENTITY.json')]
 if pathlib.Path(r).is_dir() else None) for r in roots}},
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True))))
''')


def inspect_data():
    return remote(f'''import pathlib,json
root=pathlib.Path('/root/autodl-tmp/coopbandit20261031_v3')
screens=sorted((root/'jobs').glob('s*_screen/RESULT.json'))
print('SCREENS',len(screens))
for p in screens[-2:]:
 d=json.loads(p.read_text()); s=d['checkpoints']['300']
 print('SAMPLE',str(p),json.dumps(dict(world=d['world'],keys=list(s),row=s['rows'][:1],prefix_keys=list(s['prefix']),counters=s['prefix']['counters'])))
print('BRANCHES',len(list((root/'jobs').glob('s*_t300_O1/RESULT.json'))))
print('PHYS_CONFIG',pathlib.Path({PHYS!r}+'/configs/wm_phys.toml').read_text())
''')


def stage():
    from tools.distill.data import load_exam,EXAM,read
    from tools.coopbandit.protocol import checked,sha
    records,provenance=load_exam()
    states=[json.loads(s) for s in checked(EXAM/'states.jsonl').read_text().splitlines()]
    raw={r['state_id']:r for r in states}
    cells={r['world']['seed']:r['world']['cell'] for r in read(EXAM/'worlds.json')['worlds']}
    chosen=[]
    for cell in range(12):
        candidates=[r for r in records if cells[r['world_id']]==cell]
        candidates.sort(key=lambda r:(-sum(r['available']),r['time_s'],r['world_id']))
        chosen.append(candidates[0])  # causal availability only, never exact D.
    inputs=[dict(state_id=r['state_id'],world_id=r['world_id'],role='QUALITY_ONLY',
                 time_s=r['time_s'],setting=r['setting'],rows=raw[r['state_id']]['observed_vehicles']) for r in chosen]
    exact=[dict(state_id=r['state_id'],world_id=r['world_id'],horizon_s=h,
                effects={o:r['effects'][str(h)][i] for i,o in enumerate(('O0','O1','O2','O3','O4')) if r['available'][i]})
           for r in chosen for h in (120,300)]
    files={}
    paths=[*sorted((ROOT/'tools/wmteach').glob('*.py')),*sorted((ROOT/'tools/distill').glob('*.py')),
           ROOT/'tools/coopbandit/protocol.py',ROOT/'tools/coopbandit/__init__.py',
           ROOT/'tools/coopplan/planner.py',ROOT/'tools/coopplan/predictor.py',ROOT/'tools/coopplan/__init__.py']
    for p in paths: files[str(p.relative_to(ROOT)).replace('\\','/')]=p.read_text(encoding='utf-8')
    intervention=Path('E:/agent-work/c4-leverprobe/tools/leverprobe/intervention.py')
    files['tools/leverprobe/intervention.py']=intervention.read_text(encoding='utf-8')
    files['tools/leverprobe/__init__.py']=''
    files['QUALITY_INPUTS.json']=json.dumps(inputs,sort_keys=True)
    files['EXACT_QUALITY_ONLY.json']=json.dumps(exact,sort_keys=True)
    files['DATA_PROVENANCE.json']=json.dumps(provenance,sort_keys=True)
    return remote(f'''import pathlib,json,hashlib,subprocess
root=pathlib.Path({REMOTE!r});assert root.parent.is_dir()
root.mkdir(exist_ok=True)
assert not (root/'IDENTITY.json').exists(), 'existing staged run; do not duplicate'
files=json.loads({json.dumps(files)!r})
for name,text in files.items():
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:f.write(text)
(root/'tasks').mkdir()
sources={{str(root/name):hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}}
phys=pathlib.Path({PHYS!r})
for directory in ('src/wms','tools/rbint','tools/rbeval','tools/rbfit'):
 for p in (phys/directory).rglob('*.py'):sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
for name in ('configs/wm_phys.toml','artifacts/rbfit/shrinkage_v1/idm_population.json',
 'artifacts/rbfit/shrinkage_v1/idm_posteriors.json.gz','artifacts/rbfit/shrinkage_v1/idm_shrinkage.json',
 'docs/paper_results/rbunc/observed_arrivals.json'):
 p=phys/name;sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
identity=dict(schema='wmteach/1',backend='FULL-v4',source='22cc32f65a80af482a1a277c5f22377fa727d0a3',
 sources=sources,K=8,horizons=[120,300],fit_worlds=list(range(969400,969432)),settings=[0,1,2,3],
 quality_states={[r['state_id'] for r in inputs]!r},new_sumo_sim_s=0,live_approved=False,
 shared_student_commit='4b139e5f',training_target='WM_expected_team_delay_only')
with (root/'IDENTITY.json').open('x') as f:json.dump(identity,f,sort_keys=True)
print(json.dumps(dict(root=str(root),files=len(files),sources=len(sources),quality_states=len(identity['quality_states']))))
''',timeout=90)


def cpu():
    return remote(f'''import sys,os,unittest,json,pathlib
sys.path[:0]=[{REMOTE!r},{REMOTE!r}+'/deps'];os.environ['CUDA_VISIBLE_DEVICES']=''
suite=unittest.defaultTestLoader.loadTestsFromNames(['tools.wmteach.test_cpu','tools.distill.test_cpu'])
def flatten(s):
 for t in s:
  if isinstance(t,unittest.TestSuite):yield from flatten(t)
  else:yield t
excluded=('test_all_planner_dev_extraction','test_exam_complete_builder_and_hash_parity')
tests=[t for t in flatten(suite) if t._testMethodName not in excluded]
r=unittest.TextTestRunner().run(unittest.TestSuite(tests))
receipt=dict(portable_tests=r.testsRun,errors=len(r.errors),failures=len(r.failures),
 source_data_tests='both passed in full local 19-test gate; Windows-only source paths')
print(json.dumps(receipt));assert r.wasSuccessful()
p=pathlib.Path({REMOTE!r})/'CPU.json'
if not p.exists():
 with p.open('x') as f:json.dump(receipt,f)
''',timeout=90)


def dependencies():
    return remote(f'''import pathlib,subprocess,json
root=pathlib.Path({REMOTE!r});assert root.is_dir()
assert not (root/'deps').exists()
r=subprocess.run(['/root/miniconda3/bin/python','-m','pip','install','--target',str(root/'deps'),'--no-deps',
 'scikit-learn==1.7.2','scipy==1.16.3','joblib==1.5.2','threadpoolctl==3.6.0'],capture_output=True,text=True)
print(r.stdout+r.stderr)
assert r.returncode==0
with (root/'DEPENDENCIES.json').open('x') as f:json.dump(dict(command=r.args,returncode=r.returncode),f)
''',timeout=180)


def launch_canary():
    return remote(f'''import pathlib,json,subprocess,os,time
root=pathlib.Path({REMOTE!r});assert (root/'CPU.json').is_file()
assert not (root/'CANARY_DISPATCH.json').exists()
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not gpu, 'GPU occupied; do not displace any job'
assert os.getloadavg()[0]<24
with (root/'CANARY.log').open('x') as log:
 p=subprocess.Popen(['/root/miniconda3/bin/python','-B','-u','-m','tools.wmteach.run',str(root),'--canary'],
  cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
  env={{**os.environ,'PYTHONPATH':str(root/'deps'),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}})
receipt=dict(pid=p.pid,start_ticks=pathlib.Path(f'/proc/{{p.pid}}/stat').read_text().split()[21],
 boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),started_unix=time.time(),canary=True)
with (root/'CANARY_DISPATCH.json').open('x') as f:json.dump(receipt,f)
print(json.dumps(receipt))
''')


def status():
    return remote(f'''import pathlib,json,subprocess
root=pathlib.Path({REMOTE!r})
for name in ('CANARY_DISPATCH.json','DISPATCH.json','QUALITY.json','COMPLETE.json'):
 p=root/name
 if p.is_file():print(name,p.read_text())
for name in ('CANARY.log','RUN.log'):
 p=root/name
 if p.is_file():print(name,p.read_text()[-12000:])
print('GPU',subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True))
print('TASKS',len(list((root/'tasks').glob('*/k*.json'))))
''')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['inspect', 'inspect-data','stage','cpu','deps','canary','status'])
    args = parser.parse_args()
    print({'inspect':inspect,'inspect-data':inspect_data,'stage':stage,'cpu':cpu,'deps':dependencies,
           'canary':launch_canary,'status':status}[args.action]())
