"""Own immutable candidate canary/continuation; never dispatch a peer's claim.

A successful one-task canary is retained in shard0 and reused by the panel.
An unsuccessful canary/worker is terminal: no automatic scientific retry.
"""
import argparse
import json
from pathlib import Path
import re

from tools.rbint.remote import ssh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('canary', 'continue', 'monitor', 'status', 'stop-canary'))
    parser.add_argument('--host', choices=('gpuA', 'gpuR'), default='gpuA')
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--recipe', choices=('wm_phys.toml', 'wm_phys_cpu.toml'), default='wm_phys.toml')
    parser.add_argument('--reason', default='')
    parser.add_argument('--baseline-endpoint', action='store_true')
    args = parser.parse_args()
    if args.baseline_endpoint and args.name != 'WM2-HDV60':
        raise ValueError('baseline endpoint has its own exact INT namespace')
    if not args.baseline_endpoint and not re.fullmatch(r'WM-PHYS-FULL-[A-Za-z0-9_.-]+', args.name):
        raise ValueError('INT owns FULL candidates only; A/B/C belong to runners')
    if not re.fullmatch(r'/root/rbeval20261031/phys_code_[a-f0-9]{12}', args.runtime):
        raise ValueError('use an immutable INT stage')
    root = '/root/rbeval20261031/int/'+args.name
    common = f'''
from pathlib import Path
import hashlib,json,os,signal,subprocess,time
root=Path({root!r});runtime=Path({args.runtime!r})
device={args.device!r};recipe={args.recipe!r};baseline_endpoint={args.baseline_endpoint!r}
if baseline_endpoint: recipe='wm2_original'
config=(Path('/root/autodl-tmp/c4/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml') if baseline_endpoint else runtime/'configs'/recipe)
model_runtime=Path('/root/autodl-tmp/c4/setupaudit20261030/fix') if baseline_endpoint else runtime
entry=runtime/'tools/rbint'/('evaluate_WM2_60.py' if baseline_endpoint else 'evaluate_FULL.py')
def save(path,data):
 with path.open('x') as f: json.dump(data,f,indent=2)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def alive(r):
 try:
  fields=Path('/proc/'+str(r['pid'])+'/stat').read_text().split()
  return fields[2]!='Z' and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==r['boot_id'] and fields[21]==r['start_ticks']
 except FileNotFoundError: return False
def incarnation(pid): return dict(pid=pid,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=Path('/proc/'+str(pid)+'/stat').read_text().split()[21])
def admit(canary):
 if baseline_endpoint:
  assert os.getloadavg()[0]<os.cpu_count()/2, 'leave half-host CPU headroom'
  cg=Path('/sys/fs/cgroup');stat=cg/'cpu.stat';maximum=cg/'cpu.max'
  if stat.exists() and maximum.exists() and maximum.read_text().split()[0]!='max':
   quota,period=map(int,maximum.read_text().split());capacity=quota/period
   def usage(): return int(dict(line.split() for line in stat.read_text().splitlines())['usage_usec'])
   begin=time.monotonic();before=usage();time.sleep(1);used=(usage()-before)/1e6/(time.monotonic()-begin)
   assert used+(1 if canary else 3)<.8*capacity, 'wait for spare container CPU quota'
 else: assert os.getloadavg()[0]<16, 'wait for a free CPU'
 resources=dict(load=os.getloadavg(),device=device)
 if device=='cuda':
  free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
  assert free>(6500 if canary else 19500), 'wait for free GPU allocation'
  resources['gpu_free_mib']=free
 else:
  mem={{line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}}
  available=mem['MemAvailable']
  limit=Path('/sys/fs/cgroup/memory.max');current=Path('/sys/fs/cgroup/memory.current')
  if limit.exists() and current.exists() and limit.read_text().strip()!='max': available=min(available,int(limit.read_text())-int(current.read_text()))
  assert available>(4 if canary else 12)*2**30, 'wait for free CPU memory'
  resources['effective_free_memory_bytes']=available
 return resources
def launch(i,canary):
 out=root/f'shard{{i}}';out.mkdir(exist_ok=True)
 argv=['nice','-n','19','/root/miniconda3/bin/python','-B','-u',str(entry),'--out',str(out),'--runtime-root',str(model_runtime),'--wm','/root/autodl-tmp/c4/wmcombined_wm2/wm_s.pt','--wm-config',str(config),'--model-name','wm2_hdv60' if baseline_endpoint else {args.name!r},'--shard',str(i),'--shards','3','--memory-gib','5','--device',device]
 if canary: argv+=['--max-tasks','1']
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if device=='cuda' else '',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH='/root/autodl-tmp/c4/speedhead20261030/pythondeps')
 with (root/('canary.log' if canary else f'shard{{i}}.log')).open('x') as log:
  p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=runtime,start_new_session=True)
 return dict(**incarnation(p.pid),argv=argv,unix=time.time())
'''
    if args.action == 'canary':
        code = '''
assert runtime.is_dir() and (runtime/'INT_STAGE.json').is_file()
resources=admit(True)
if baseline_endpoint:
 prior_path=runtime.parent/'wm2_shard0/IDENTITY.json'
 prior=json.loads(prior_path.read_text())
 assert prior['model']=='wm2' and prior['wm_sha256']=='b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436'
 assert prior['config_sha256']==sha(config) and prior['horizon']==600 and prior['K']==8
 assert all(sha(Path(p))==expected for p,expected in prior['sources'].items()), 'preserve original WM2 source'
 gate_sha=sha(prior_path)
else:
 suffix='_cpu' if recipe=='wm_phys_cpu.toml' else ''
 smoke=runtime.parent/('SMOKE_'+runtime.name.removeprefix('phys_code_')+suffix+'.json')
 gate=json.loads(smoke.read_text());assert gate['passed'] and gate['recipe_sha256']==sha(config)
 gate_sha=sha(smoke)
assert root.parent.parent.is_dir();root.parent.mkdir(exist_ok=True);root.mkdir()
save(root/'SPEC.json',dict(owner='S-RB-INT',host='''+repr(args.host)+''',runtime=str(runtime),model_runtime=str(model_runtime),stage=json.loads((runtime/'INT_STAGE.json').read_text()),smoke_sha256=gate_sha,variant='WM2_HDV60' if baseline_endpoint else 'FULL',tasks=160,world_order='reserved-first',training=False,device=device,recipe=recipe,resources=resources))
save(root/'CANARY_CLAIM.json',dict(time=time.time(),automatic_retry=False))
r=launch(0,True);save(root/'CANARY_PROCESS.json',r);print(json.dumps(r))
'''
    elif args.action == 'continue':
        code = '''
spec=json.loads((root/'SPEC.json').read_text());assert spec['owner']=='S-RB-INT' and spec['runtime']==str(runtime)
assert spec.get('device','cuda')==device and spec.get('recipe','wm_phys.toml')==recipe, 'preserve original device/recipe'
assert (spec['variant']=='WM2_HDV60')==baseline_endpoint, 'preserve original endpoint/full contract'
assert not alive(json.loads((root/'CANARY_PROCESS.json').read_text())), 'canary still live'
assert not (root/'shard0/STOP').exists(), 'terminal canary failure'
files=list((root/'shard0').glob('w*_k*.json'));assert len(files)==1, 'need successful one-task canary'
task=json.loads(files[0].read_text());assert task['future']==0 and len(task['episodes'])==5
if baseline_endpoint:
 assert task['schema']=='wmphys-hdv60/1' and all('endpoint60' in e for e in task['episodes'])
 parent_shard=(task['seed']-969420)%4
 parent_path=runtime.parent/f'wm2_shard{parent_shard}'/files[0].name
 parent=json.loads(parent_path.read_text())
 assert parent['identity_sha256']==sha(parent_path.parent/'IDENTITY.json')
 import math
 for episode in task['episodes']:
  old=next(e for e in parent['episodes'] if e['arm']==episode['arm'])
  assert old['initial_sha256']==episode['initial_sha256']
  for reg,prefix in episode['prefix60_bins'].items():
   assert prefix['post_counts']==old['bins'][reg]['post_counts'][:6], 'endpoint prefix support changed'
   for a,b in zip(prefix['speed'],old['bins'][reg]['speed'][6:12]):
    assert (a is None and b is None) or (a is not None and b is not None and math.isclose(a,b,rel_tol=0.,abs_tol=1e-8)), 'endpoint prefix speed changed'
 verified=dict(parent=str(parent_path),parent_sha256=sha(parent_path),endpoint_sha256=sha(files[0]),all5arms=True,regions=3,bins=6,speed_abs_tolerance=1e-8)
 if (root/'PREFIX_VERIFIED.json').exists(): assert json.loads((root/'PREFIX_VERIFIED.json').read_text())==verified
 else: save(root/'PREFIX_VERIFIED.json',verified)
else: assert task['future_sampling']['population_audit']['conserved']
resources=admit(False)
save(root/'PANEL_CLAIM.json',dict(time=time.time(),reuse=files[0].name,automatic_retry=False,resources=resources))
workers=[launch(i,False) for i in range(3)]
save(root/'PANEL_PROCESS.json',dict(workers=workers));print(json.dumps(workers))
'''
    elif args.action == 'stop-canary':
        if not args.reason:
            raise ValueError('an explicit operational stop reason is required')
        code = '''
spec=json.loads((root/'SPEC.json').read_text());assert spec['owner']=='S-RB-INT' and spec['runtime']==str(runtime)
assert not (root/'PANEL_PROCESS.json').exists(), 'never stop a panel through its canary'
r=json.loads((root/'CANARY_PROCESS.json').read_text())
assert str(entry) in r['argv']
assert r['argv'][r['argv'].index('--out')+1]==str(root/'shard0')
save(root/'CANARY_STOP_ONCE.json',dict(unix=time.time(),reason='''+repr(args.reason)+''',process=r))
marker=root/'shard0/STOP'
if not marker.exists(): marker.touch(exist_ok=False)
end=time.monotonic()+20
while alive(r) and time.monotonic()<end: time.sleep(1)
signalled=False
if alive(r): os.kill(r['pid'],signal.SIGTERM);signalled=True
end=time.monotonic()+5
while alive(r) and time.monotonic()<end: time.sleep(.2)
result=dict(unix=time.time(),process=r,signalled_owned_pid=signalled,alive=alive(r),tasks=len(list((root/'shard0').glob('w*_k*.json'))))
save(root/'CANARY_STOPPED.json',result);print(json.dumps(result))
'''
    elif args.action == 'monitor':
        source = Path(__file__).with_name('panel_monitor.py').read_text(encoding='utf-8')
        code = '''
assert (root/'PANEL_PROCESS.json').is_file()
assert not (root/'MONITOR_ONCE.json').exists(), 'preserve existing monitor'
save(root/'MONITOR_DISPATCH_CLAIM.json',dict(unix=time.time(),automatic_retry=False))
source='''+repr(source)+'''
path=root/'PANEL_MONITOR.py'
with path.open('x') as f: f.write(source)
argv=['nice','-n','19','/root/miniconda3/bin/python','-B','-u',str(path),'--root',str(root)]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (root/'monitor.log').open('x') as log:
 p=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=root,start_new_session=True)
r=dict(**incarnation(p.pid),argv=argv,source_sha256=sha(path));save(root/'MONITOR_PROCESS.json',r);print(json.dumps(r))
'''
    else:
        code = '''
print('LOAD',os.getloadavg())
print(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.used,utilization.gpu','--format=csv'],text=True))
print('ROOT',str(root),'exists',root.is_dir())
if root.is_dir():
 for name in ('CANARY_PROCESS.json','PANEL_PROCESS.json','MONITOR_PROCESS.json'):
  p=root/name
  if p.exists():
   r=json.loads(p.read_text());print(name,[(v['pid'],alive(v)) for v in r.get('workers',[r])])
 for i in range(3):
  out=root/f'shard{i}';print(out.name,len(list(out.glob('w*_k*.json'))),'complete',(out/'COMPLETE.json').exists(),'STOP',(out/'STOP').exists())
 for name in ('MONITOR_FAILED.json','MONITOR_COMPLETE.json','ANALYSIS_full.json','ANALYSIS_interim04.json','ANALYSIS_interim08.json','canary.log','shard0.log','shard1.log','shard2.log','monitor.log'):
  p=root/name
  if p.exists(): print(name,p.read_text()[-5000:])
'''
    print(ssh(args.host, common+code, timeout=90))


if __name__ == '__main__':
    main()
