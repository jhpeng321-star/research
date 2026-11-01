"""Short explicit SSH operations, never process-name signals or implicit dispatch."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PHYS = '/root/rbeval20261031/phys_code_22cc32f65a80'
REMOTE = '/root/autodl-tmp/wmteach20261101_v3'
HOST = 'gpuA'


def remote(source, host=None, timeout=60):
    host=host or HOST
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
(root/'deps').symlink_to('/root/autodl-tmp/wmteach20261101_v1/deps',target_is_directory=True)
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
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()
assert int(gpu)>12*1024, 'insufficient free capacity for own strict8GiB cap'
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


def launch():
    return remote(f'''import pathlib,json,subprocess,os,time,hashlib
root=pathlib.Path({REMOTE!r});assert (root/'CPU.json').is_file()
canary=json.loads((root/'CANARY_DISPATCH.json').read_text())
p=pathlib.Path(f'/proc/{{canary["pid"]}}/stat')
if p.exists():
 stat=p.read_text().split()
 assert stat[21]!=canary['start_ticks'] or stat[2]=='Z', 'canary still running; no duplicate'
assert not (root/'DISPATCH.json').exists()
tasks=list((root/'tasks').glob('*/k0.json'));assert len(tasks)==1
task=json.loads(tasks[0].read_text());assert task['population_audit']['finite'] and task['population_audit']['conserved']
assert set(task['results'])=={{'120','300'}}
for h,actions in task['results'].items():
 assert 'O0' in actions
 for a,f in actions.items():
  road,source=task['road_source'][h][a]
  assert abs(f['team_delay']-road-source)<1e-6
gate=dict(task_sha256=hashlib.sha256(tasks[0].read_bytes()).hexdigest(),canary_reused=True,
 population_audit=task['population_audit'],wall_s=task['wall_s'],new_sumo_sim_s=0)
with (root/'CANARY_PASS.json').open('x') as f:json.dump(gate,f)
with (root/'RUN.log').open('x') as log:
 p=subprocess.Popen(['/root/miniconda3/bin/python','-B','-u','-m','tools.wmteach.run',str(root)],
  cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
  env={{**os.environ,'PYTHONPATH':str(root/'deps'),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}})
receipt=dict(pid=p.pid,start_ticks=pathlib.Path(f'/proc/{{p.pid}}/stat').read_text().split()[21],
 boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),started_unix=time.time(),
 canary_reused=True,canary_task_sha256=gate['task_sha256'],strict_memory_gib=8,new_sumo_sim_s=0)
with (root/'DISPATCH.json').open('x') as f:json.dump(receipt,f)
print(json.dumps(receipt))
''')


def collect():
    """One bounded read-only snapshot; never launches a watcher or GPU task."""
    source=f'''import pathlib,json,time,hashlib,os
root=pathlib.Path({REMOTE!r})
names=('IDENTITY.json','CPU.json','CANARY_PASS.json','CANARY_DISPATCH.json','DISPATCH.json',
 'QUALITY.json','QUALITY_RECORDS.json','COMPLETE.json','DATA_PROVENANCE.json')
files={{n:json.loads((root/n).read_text()) for n in names if (root/n).is_file()}}
dispatch=files.get('DISPATCH.json',files['CANARY_DISPATCH.json'])
p=pathlib.Path(f'/proc/{{dispatch["pid"]}}/stat')
stat=p.read_text().split() if p.is_file() else None
process=dict(pid=dispatch['pid'],exists=stat is not None,
 own_identity_matches=(stat is not None and stat[21]==dispatch['start_ticks'] and
 pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==dispatch['boot_id']),
 state=stat[2] if stat else None)
labels=[json.loads(p.read_text()) for p in sorted((root/'tasks').glob('*/LABEL_H*.json'))]
for p in sorted(root.glob('student_setting*/SUMMARY.json')):
 files[str(p.relative_to(root))]=json.loads(p.read_text())
live_root=pathlib.Path('/root/autodl-tmp/wmteach_live20261101_v1');live=None
if live_root.is_dir():
 live=dict(root=str(live_root),complete=(live_root/'COMPLETE.json').is_file(),stopped=(live_root/'STOP').exists(),pairs=[])
 for name in ('FREEZE.json','CPU.json','PROCESS.json','MECHANICS.json'):
  if (live_root/name).is_file():live[name]=json.loads((live_root/name).read_text())
 if 'PROCESS.json' in live:
  owner=live['PROCESS.json'];p=pathlib.Path(f'/proc/{{owner["pid"]}}/stat')
  st=p.read_text().split() if p.exists() else None
  live['owned_process_live']=(st is not None and st[21]==owner['start_ticks'] and st[2]!='Z' and
   pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==owner['boot_id'])
  live['terminal_incomplete']=not live['owned_process_live'] and not live['complete']
 ledger=live_root/'CHARGES.jsonl'
 live['charged_sim_s']=sum(json.loads(s)['sim_s'] for s in ledger.read_text().splitlines()) if ledger.exists() else 0
 live['closed_actual_sim_s']=sum(json.loads(p.read_text())['actual_sim_s'] for p in (live_root/'jobs').glob('*/CLOSE.json'))
 for seed in range(984000,984024):
  base=live_root/'jobs'/f's{{seed}}_mappo'/'RESULT.json';student=live_root/'jobs'/f's{{seed}}_student'/'RESULT.json'
  if base.is_file() and student.is_file():
   b=json.loads(base.read_text());s=json.loads(student.read_text())
   live['pairs'].append(dict(seed=seed,clocks={{h:{{k:b['clocks'][h][k]-s['clocks'][h][k] for k in
    ('team_delay_vs','road_delay_vs','source_wait_vs')}} for h in ('600','900')}},
    baseline_safety={{k:b['clocks']['900'][k] for k in ('collisions','teleports','ttc_below_3_vs','hard_brake')}},
    student_safety={{k:s['clocks']['900'][k] for k in ('collisions','teleports','ttc_below_3_vs','hard_brake')}},
    moves=sum(e['executed_action']['id']!='O0' for e in s['history']),warmup_pass=s['warmup_pass'],
    overrides=s['overrides'],requests=s['requests']))
 if live['complete']:
  import numpy as np
  assert len(live['pairs'])==24 and live['charged_sim_s']<=58800
  rng=np.random.default_rng(20261101);indices=rng.integers(0,24,(10000,24))
  live['summary']={{}}
  for h in ('600','900'):
   values=np.array([r['clocks'][h]['team_delay_vs'] for r in live['pairs']])
   live['summary'][h]=dict(mean=float(values.mean()),ci95=np.quantile(values[indices].mean(1),[.025,.975]).tolist())
  live['summary']['moves']=sum(r['moves'] for r in live['pairs'])
  live['summary']['meaning']='paired full-team MAPPO minus WM student; positive saves delay; fresh shared DEV'
print(json.dumps(dict(root=str(root),unix=time.time(),process=process,files=files,labels=labels,
 live=live,
 completed_future_tasks=len(list((root/'tasks').glob('*/k*.json'))),
 source_hashes={{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names if (root/n).is_file()}},
 teacher_new_sumo_sim_s=0,new_sumo_sim_s=(live['closed_actual_sim_s'] if live else 0),
 sumo_accounting='closed-actual lower bound until all charged jobs close',load=os.getloadavg())))
'''
    snapshot=json.loads(remote(source))
    registry=Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
    snapshot['registration_lines']=[line for line in registry.read_text(encoding='utf-8').splitlines()
        if line.startswith('- S-WMTEACH ') or 'S-WMTEACH BUDGET APPROVED:' in line]
    dest=ROOT/'docs/paper_results/wmteach'/('SNAPSHOT_'+str(int(snapshot['unix']))+'.json')
    assert dest.parent.is_dir()
    with dest.open('x',encoding='utf-8') as f:json.dump(snapshot,f,sort_keys=True,allow_nan=False);f.write('\n')
    return json.dumps(dict(path=str(dest),process=snapshot['process'],
                          completed_future_tasks=snapshot['completed_future_tasks'],labels=len(snapshot['labels']),
                          teacher_quality_complete='QUALITY.json' in snapshot['files']))


def watch():
    import sys,time
    receipt_dir=ROOT/'docs/paper_results/wmteach'
    assert receipt_dir.is_dir()
    # WATCH.lock inside the child enforces sole ownership even across launch races.
    path=receipt_dir/f'WATCH_{int(time.time())}.log'
    with path.open('x',encoding='utf-8') as log:
        child=subprocess.Popen([sys.executable,'-B','-u','-m','tools.wmteach.watch'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess,'CREATE_NEW_PROCESS_GROUP') else 0))
    return json.dumps(dict(pid=child.pid,log=str(path),role='collector_and_approved_first_student_live_gate',new_gpu_dispatch=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',choices=['gpuA','gpuR'],default='gpuA')
    parser.add_argument('action', choices=['inspect', 'inspect-data','stage','cpu','deps','canary','status','launch','collect','watch'])
    args = parser.parse_args()
    HOST=args.host
    print({'inspect':inspect,'inspect-data':inspect_data,'stage':stage,'cpu':cpu,'deps':dependencies,
           'canary':launch_canary,'status':status,'launch':launch,'collect':collect,'watch':watch}[args.action]())
