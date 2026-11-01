"""Explicit own-root development dispatch. No peer signalling; no reserved task reads."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

from tools.rbint.remote import ssh

PARENT = '/root/rbeval20261031'
JOB = PARENT+'/rb2break'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('stage', 'launch', 'queue', 'queue-corrections', 'queue-native', 'queue-k8', 'pause-k8', 'validate-envelope', 'fetch-envelope', 'fit-birth', 'bind-birth', 'fetch-birth', 'fetch-source-audit', 'resources', 'inspect-start', 'inspect-source', 'wait', 'retire-cpu3', 'retire-slow-cpu', 'retire-old-queues', 'status', 'fetch', 'fetch-baseline', 'baseline'))
    parser.add_argument('--host', default='gpuA', choices=('gpuA', 'gpuR'))
    parser.add_argument('--runtime')
    parser.add_argument('--factor', default='baseline')
    parser.add_argument('--worlds', nargs='+', type=int, default=list(range(969420, 969432)))
    parser.add_argument('--futures', nargs='+', type=int, default=[0])
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--tag', default='screen')
    parser.add_argument('--snapshot-name')
    parser.add_argument('--lane', type=int, choices=(0, 1), default=0)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--jobs', nargs='+')
    parser.add_argument('--wait-seconds', type=int, default=480)
    parser.add_argument('--cpu-fast', action='store_true')
    parser.add_argument('--native-probe', action='store_true')
    parser.add_argument('--birth-version', choices=('v1', 'v2', 'v3'), default='v1')
    args = parser.parse_args()
    if not set(args.worlds) <= set(range(969420, 969432)) or not args.worlds:
        raise ValueError('development only')
    if not args.tag.replace('_', '').isalnum():
        raise ValueError('simple unique tag required')
    root = Path(__file__).resolve().parents[2]
    if args.action == 'stage':
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        paths = ['src/wms', 'tools/rbint', 'tools/rbeval', 'tools/rb2break', 'tools/rbfit',
            'configs/execparity_cav_idm_v1.json', 'configs/native_cav_head_v5.json', 'configs/wm_phys.toml', 'configs/wm_phys_cpu.toml',
            'artifacts/rbfit/shrinkage_v1', 'docs/paper_results/rbunc/observed_arrivals.json',
            'tests/test_rb2break.py']
        packaged = root/'artifacts/rb2break/startup_v3'
        if packaged.is_dir():
            paths.append('artifacts/rb2break/startup_v3')
        archive = subprocess.check_output(['git', 'archive', '--format=tar', commit, *paths], cwd=root)
        sha = hashlib.sha256(archive).hexdigest()
        dest = PARENT+'/rb2break_code_'+commit[:12]
        code = f'''import sys,io,tarfile,hashlib,json
from pathlib import Path
data=sys.stdin.buffer.read();assert hashlib.sha256(data).hexdigest()=={sha!r}
parent=Path({PARENT!r});assert parent.is_dir()
dest=Path({dest!r});assert not dest.exists(), 'new immutable stage only'
dest.mkdir()
with tarfile.open(fileobj=io.BytesIO(data)) as tar:
 for member in tar.getmembers():
  assert not member.issym() and not member.islnk() and '..' not in Path(member.name).parts and not Path(member.name).is_absolute()
 tar.extractall(dest,filter='data')
receipt=dict(owner='S-RB2-BREAK',commit={commit!r},archive_sha256={sha!r},runtime=str(dest))
with (dest/'STAGE.json').open('x') as f: json.dump(receipt,f,indent=1)
print(json.dumps(receipt))'''
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', args.host,
            '/root/miniconda3/bin/python', '-B', '-c', shlex.quote(code)],
            input=archive, capture_output=True, timeout=180)
        if result.returncode:
            raise RuntimeError(result.stdout.decode()+result.stderr.decode())
        print(result.stdout.decode())
        return
    common = f'''from pathlib import Path
import json,os,subprocess,time,hashlib
parent=Path({PARENT!r});root=Path({JOB!r})
assert parent.is_dir()
def save(path,data):
 with path.open('x') as f: json.dump(data,f,indent=1)
'''
    out_name = f'{args.tag}_{args.factor}_s{args.shard}'
    if args.action == 'launch':
        if args.factor not in ('baseline', 'no_mainline_lc', 'no_merge_advice', 'depart_last',
                               'depart_first', 'population', 'raw_posterior', 'point_v0', 'point_tau', 'fixed_rate', 'lane_limits', 'cav_settings', 'native_uncommanded', 'startup_birth_population'):
            raise ValueError('unregistered factor')
        if not args.runtime or not args.runtime.startswith(PARENT+'/rb2break_code_'):
            raise ValueError('own immutable runtime required')
        code = f'''
runtime=Path({args.runtime!r});assert (runtime/'STAGE.json').is_file()
resources=dict(load=os.getloadavg(),device={args.device!r})
assert resources['load'][0]<16, 'host busy'
if {args.device!r}=='cuda':
 resources['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 assert int(resources['gpu'].split(',')[0])>12000, 'leave shared GPU headroom'
root.mkdir(exist_ok=True)
out=root/{out_name!r};assert not out.exists(), 'never duplicate a dispatch'
out.mkdir()
argv=['nice','-n','19','/root/miniconda3/bin/python','-B','-u','-m','tools.rb2break.run',
 '--out',str(out),'--runtime-root',str(runtime),'--wm-config',str(runtime/'configs'/{('wm_phys_cpu.toml' if args.cpu_fast else 'wm_phys.toml')!r}),
 '--model-name',{('RB2BREAK_'+args.tag+'_'+args.factor)!r},'--factor',{args.factor!r},
 '--dev-worlds',*map(str,{args.worlds!r}),'--futures',*map(str,{args.futures!r}),
 '--shard',{str(args.shard)!r},'--shards',{str(args.shards)!r},'--memory-gib','5','--device',{args.device!r}]
if {args.native_probe!r}: argv+=['--native-probe']
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES={'0' if args.device == 'cuda' else ''!r},
 PYTHONPATH='/root/autodl-tmp/c4/speedhead20261030/pythondeps')
with (out/'worker.log').open('x') as log:
 p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=runtime,start_new_session=True)
receipt=dict(owner='S-RB2-BREAK',pid=p.pid,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 start_ticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],argv=argv,resources=resources,unix=time.time())
save(out/'DISPATCH.json',receipt);print(json.dumps(receipt))
'''
    elif args.action in ('queue', 'queue-corrections', 'queue-native'):
        assert args.runtime and args.runtime.startswith(PARENT+'/rb2break_code_')
        if args.action == 'queue':
            wait_for = 'screen_no_mainline_lc_s0' if args.lane == 0 else 'screen_no_merge_advice_s0'
            factors = ['depart_last', 'population', 'point_v0'] if args.lane == 0 else ['depart_first', 'point_tau', 'fixed_rate']
        elif args.action == 'queue-corrections':
            wait_for = 'screen_point_v0_s0' if args.lane == 0 else 'screen_fixed_rate_s0'
            factors = ['raw_posterior', 'lane_limits'] if args.lane == 0 else ['cav_settings']
        else:
            assert args.lane == 1, 'native followup uses the released CAV-settings worker slot'
            wait_for, factors = 'screen_cav_settings_s0', ['native_uncommanded']
        code = f'''
runtime=Path({args.runtime!r});assert (runtime/'STAGE.json').is_file()
claim=root/('QUEUE2_'+runtime.name+'_'+{str(args.lane)!r}+'.json');assert not claim.exists()
argv=['nice','-n','19','/root/miniconda3/bin/python','-B','-u','-m','tools.rb2break.factor_queue',
 '--wait-for',str(root/{wait_for!r}),'--factors',*{factors!r},'--lane',{str(args.lane)!r}]
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0',
 PYTHONPATH='/root/autodl-tmp/c4/speedhead20261030/pythondeps')
with claim.with_suffix('.log').open('x') as log:
 child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=runtime,start_new_session=True)
save(claim,dict(owner='S-RB2-BREAK',pid=child.pid,argv=argv,unix=time.time(),
 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 start_ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21]))
print(claim.read_text())
'''
    elif args.action == 'queue-k8':
        assert args.runtime and args.runtime.startswith(PARENT+'/rb2break_code_')
        code = f'''
runtime=Path({args.runtime!r});assert (runtime/'STAGE.json').is_file()
claim=root/{('K8_QUEUE_'+str(args.lane)+'.json')!r};assert not claim.exists()
argv=['nice','-n','19','/root/miniconda3/bin/python','-B','-u','-m','tools.rb2break.k8_queue','--lane',{str(args.lane)!r}]
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0',
 PYTHONPATH='/root/autodl-tmp/c4/speedhead20261030/pythondeps')
with claim.with_suffix('.log').open('x') as log:
 child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=runtime,start_new_session=True)
save(claim,dict(owner='S-RB2-BREAK',pid=child.pid,argv=argv,unix=time.time(),
 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 start_ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21]))
print(claim.read_text())
'''
    elif args.action in ('retire-slow-cpu', 'retire-cpu3'):
        names = (('controlcpu2_baseline_s0', 'fixcpu2_lane_limits_s0') if args.action == 'retire-slow-cpu'
                 else ('controlcpu3_baseline_s0', 'fixcpu3_lane_limits_s0', 'cavcpu3_cav_settings_s0', 'rawcpu3_raw_posterior_s0'))
        code = '''
import ctypes,signal
libc=ctypes.CDLL(None,use_errno=True)
for name in OWN_CPU_NAMES:
 out=root/name;assert not list(out.glob('w*_k*.json')), 'task exists; no pre-inference retirement'
 r=json.loads((out/'DISPATCH.json').read_text())
 assert r['owner']=='S-RB2-BREAK' and r['argv'][r['argv'].index('--device')+1]=='cpu'
 assert r['argv'][r['argv'].index('--out')+1]==str(out)
 pidfd=libc.syscall(434,r['pid'],0);assert pidfd>=0
 try:
  assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==r['boot_id']
  assert (Path('/proc')/str(r['pid'])/'stat').read_text().split()[21]==r['start_ticks']
  actual=(Path('/proc')/str(r['pid'])/'cmdline').read_bytes().split(b'\\0')
  assert str(out).encode() in actual and b'tools.rb2break.run' in actual
  save(out/'OPERATIONAL_STOP.json',dict(process=r,reason='No completed task after engineering runtime budget; preserve output and move to GPU queue without viewing efficacy',unix=time.time()))
  assert libc.syscall(424,pidfd,int(signal.SIGTERM),0,0)==0
 finally: os.close(pidfd)
print('Stopped exact own slow CPU pair; all outputs retained')
'''.replace('OWN_CPU_NAMES', repr(names))
    elif args.action == 'retire-old-queues':
        code = '''
import ctypes,signal
assert not any((root/f'screen_{factor}_s0').exists() for factor in ('depart_last','depart_first','population','point_v0','point_tau','fixed_rate')), 'old queues already launched work'
result=[]
for lane in (0,1):
 path=root/f'QUEUE_{lane}.json';r=json.loads(path.read_text())
 assert r['owner']=='S-RB2-BREAK' and 'tools.rb2break.queue' in r['argv']
 assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==r['boot_id']
 libc=ctypes.CDLL(None,use_errno=True)
 pidfd=libc.syscall(434,r['pid'],0)  # x86_64 pidfd_open; bind incarnation before inspecting
 assert pidfd>=0, ctypes.get_errno()
 try:
  stat=Path('/proc')/str(r['pid'])/'stat'
  assert stat.read_text().split()[21]==r['start_ticks']
  actual=(Path('/proc')/str(r['pid'])/'cmdline').read_bytes().split(b'\\0')
  assert b'tools.rb2break.queue' in actual
  assert libc.syscall(424,pidfd,int(signal.SIGTERM),0,0)==0, ctypes.get_errno()
  result.append(dict(pid=r['pid'],lane=lane,reason='engineering namespace shadowing; no factor inference launched'))
 finally: os.close(pidfd)
save(root/'RETIRED_OLD_QUEUES.json',result);print(json.dumps(result))
'''
    elif args.action == 'pause-k8':
        code = '''
reason="MAIN 2026-11-01 17:35 GPU priority: checkpoint current BREAK runs; no new GPU jobs until MAIN appends S-RB2-BREAK RESUME"
rows=[]
for name,pid in (('devk8_native2_s0',501141),('devk8_native2_s1',500921)):
 out=root/name;r=json.loads((out/'DISPATCH.json').read_text())
 assert r['owner']=='S-RB2-BREAK' and r['pid']==pid
 assert r['argv'][r['argv'].index('--out')+1]==str(out)
 assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==r['boot_id']
 proc=Path('/proc')/str(pid)
 alive=proc.exists() and (proc/'stat').read_text().split()[2]!='Z'
 if alive:
  assert (proc/'stat').read_text().split()[21]==r['start_ticks']
  actual=(proc/'cmdline').read_bytes().split(b'\\0')
  assert str(out).encode() in actual and b'tools.rb2break.run' in actual
 runtime=Path(r['argv'][r['argv'].index('--runtime-root')+1])
 assert "assert not (out/'STOP').exists()" in (runtime/'tools/rbeval/run_wm.py').read_text()
 receipt=dict(owner='S-RB2-BREAK',reason=reason,unix=time.time(),process=r,
  alive_at_request=alive,checkpoint_policy='finish in-flight task; STOP before next task; retain all files',
  completed_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('w*_k*.json'))})
 pause=out/'GPU_PAUSE_1735.json'
 if not pause.exists(): save(pause,receipt)
 if not (out/'STOP').exists():
  with (out/'STOP').open('x') as f: f.write(reason+'\\n')
 rows.append(dict(job=name,pid=pid,alive=alive,tasks=len(receipt['completed_hashes']),stop_exists=(out/'STOP').exists()))
print(json.dumps(rows))
'''
    elif args.action == 'wait':
        if not args.jobs or any(not name.replace('_', '').isalnum() for name in args.jobs):
            raise ValueError('explicit own job names required')
        if not 0 <= args.wait_seconds <= 900:
            raise ValueError('bounded wait required')
        code = f'''
jobs={args.jobs!r};end=time.monotonic()+{args.wait_seconds!r}
while True:
 rows=[]
 for name in jobs:
  out=root/name;r=dict(job=name,tasks=len(list(out.glob('w*_k*.json'))),complete=(out/'COMPLETE.json').exists(),failed=False)
  dispatch=out/'DISPATCH.json'
  if dispatch.exists():
   receipt=json.loads(dispatch.read_text());assert receipt['owner']=='S-RB2-BREAK'
   proc=Path('/proc')/str(receipt['pid'])/'stat'
   r['alive']=proc.exists() and proc.read_text().split()[2]!='Z' and proc.read_text().split()[21]==receipt['start_ticks']
   r['failed']=not r['alive'] and not r['complete']
  rows.append(r)
 if all(r['complete'] for r in rows) or any(r['failed'] for r in rows) or time.monotonic()>=end: break
 time.sleep(15)
print(json.dumps(dict(jobs=rows,load=os.getloadavg())))
'''
    elif args.action == 'resources':
        code = '''
result=dict(load=os.getloadavg(),cpus=os.cpu_count())
for name in ('cpu.max','cpu.stat','memory.max','memory.current'):
 path=Path('/sys/fs/cgroup')/name
 if path.exists(): result[name]=path.read_text()
print(json.dumps(result))
'''
    elif args.action == 'validate-envelope':
        assert args.runtime and args.runtime.startswith(PARENT+'/rb2break_code_')
        code = f'''
runtime=Path({args.runtime!r});assert (runtime/'STAGE.json').is_file()
assert os.getloadavg()[0]<16, 'wait for spare CPU'
out=root/'envelope_validation';assert not out.exists();out.mkdir()
argv=['nice','-n','19','/root/miniconda3/bin/python','-B','-u','-m','tools.rb2break.validate_envelope','--out',str(out/'VALIDATION.json')]
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
with (out/'worker.log').open('x') as log:
 child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=runtime,start_new_session=True)
save(out/'DISPATCH.json',dict(owner='S-RB2-BREAK',pid=child.pid,argv=argv,unix=time.time(),
 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 start_ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21]))
print((out/'DISPATCH.json').read_text())
'''
    elif args.action == 'fit-birth':
        assert args.runtime and args.runtime.startswith(PARENT+'/rb2break_code_')
        code = f'''
runtime=Path({args.runtime!r});assert (runtime/'STAGE.json').is_file()
assert os.getloadavg()[0]<os.cpu_count()/2, 'host busy'
cg=Path('/sys/fs/cgroup');quota,period=(cg/'cpu.max').read_text().split()
capacity=os.cpu_count() if quota=='max' else int(quota)/int(period)
def usage(): return int(dict(line.split() for line in (cg/'cpu.stat').read_text().splitlines())['usage_usec'])
before=usage();begin=time.monotonic();time.sleep(2)
used=(usage()-before)/1e6/(time.monotonic()-begin)
assert used+1<.8*capacity, 'wait for spare container CPU quota'
limit=(cg/'memory.max').read_text().strip()
assert limit=='max' or int(limit)-int((cg/'memory.current').read_text())>2*2**30, 'leave memory headroom'
out=root/{('birth_fit_'+args.birth_version)!r};assert not out.exists()
claim=root/{('BIRTH_FIT_'+args.birth_version+'_PROCESS.json')!r};assert not claim.exists()
argv=['nice','-n','19','/root/miniconda3/bin/python','-B','-u','-m','tools.rb2break.birth_population','--out',str(out)]
if {args.birth_version!r}=='v2': argv+=['--acceleration-envelope']
if {args.birth_version!r}=='v3': argv+=['--motion-seconds','10']
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
with (root/{('birth_fit_'+args.birth_version+'.log')!r}).open('x') as log:
 child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=runtime,start_new_session=True)
save(claim,dict(owner='S-RB2-BREAK',pid=child.pid,argv=argv,unix=time.time(),
 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 start_ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21]))
print(claim.read_text())
'''
    elif args.action == 'bind-birth':
        assert args.runtime and args.runtime.startswith(PARENT+'/rb2break_code_')
        code = f'''
import sys
runtime=Path({args.runtime!r});sys.path[:0]=[str(runtime),str(runtime/'src')]
from tools.rbint.observations import load_observations
fit=root/{('birth_fit_'+args.birth_version)!r};artifact=json.loads((fit/'entrant_population.json').read_text())
assert artifact['source_sha256']==hashlib.sha256((runtime/'tools/rb2break/birth_population.py').read_bytes()).hexdigest()
observations={{}}
for seed in range(969420,969432):
 _,_,hashes=load_observations(seed);observations[str(seed)]=hashes
save(fit/'OBSERVATION_BINDING.json',dict(observations=observations,stage=json.loads((runtime/'STAGE.json').read_text()),
 source_sha256=artifact['source_sha256'],fitting_end_s=300,traffic_outcomes_used=False))
print(json.dumps(dict(worlds=len(observations),motion_validation_improved=artifact['motion_validation_improved'])))
'''
    elif args.action == 'inspect-source':
        code = '''
import sys,numpy as np
runtime=parent/'rb2break_code_e164b30946af';sys.path[:0]=[str(runtime),str(runtime/'src')]
from tools.rbint.observations import load_observations
from wms.model.entries import ENTRY_S_M
data=[];bindings={}
for seed in range(969420,969432):
 start,_,hashes=load_observations(seed);bindings[str(seed)]=hashes
 for e in start['past_entries']:
  if e['time']>=180: data.append(dict(seed=seed,**e))
result={}
for entrance,lanes in (('mainline',(0,1)),('ramp',(2,))):
 selected=[e for e in data if e['lane'] in lanes]
 lengths=np.array([e['s_m']-ENTRY_S_M[e['lane']]-.1 for e in selected])
 tolerances=np.array([2*abs(float(np.spacing(np.float32(e['s_m'])))) for e in selected])
 speeds=np.array([e['v_mps'] for e in selected])
 result[entrance]=dict(n=len(selected),base_position_fraction=float((abs(lengths-5)<=tolerances).mean()),
  implied_length_quantiles=np.quantile(lengths,[0,.25,.5,.75,1]).tolist(),
  birth_speed_quantiles_mps=np.quantile(speeds,[0,.25,.5,.75,.95,1]).tolist())
payload=dict(conditioning_interval_s=[180,300],worlds=list(range(969420,969432)),
 result=result,observable_hashes=bindings,traffic_outcomes_used=False)
save(root/'SOURCE_OBSERVABLE_AUDIT.json',payload);print(json.dumps(payload))
'''
    elif args.action == 'inspect-start':
        code = '''
import sys
runtime=parent/'rb2break_code_116b214765e9'
sys.path[:0]=[str(runtime),str(runtime/'src')]
from tools.rbint.observations import load_motion_history
start,history=load_motion_history(969420)
print(json.dumps(dict(seed=start['seed'],entry_keys=sorted(start['past_entries'][0]),
 history_shapes={k:list(v.shape) for k,v in history.items()},
 names_type=type(start['vehicle_ids']).__name__,entry_observable={k:v for k,v in start['past_entries'][0].items() if k in ('time','id','vehicle_id','vid','lane','is_cav')})))
'''
    elif args.action == 'status':
        code = f'''
print('LOAD',os.getloadavg())
print(subprocess.check_output(['nvidia-smi'],text=True))
if root.is_dir():
 for out in sorted(root.iterdir()):
  if not out.is_dir(): continue
  if {args.jobs!r} and out.name not in {args.jobs!r}: continue
  print('JOB',out.name,'tasks',len(list(out.glob('w*_k*.json'))),'complete',(out/'COMPLETE.json').exists())
  dispatch=out/'DISPATCH.json'
  if dispatch.exists():
   receipt=json.loads(dispatch.read_text());proc=Path('/proc')/str(receipt['pid'])/'stat'
   alive=proc.exists() and proc.read_text().split()[2]!='Z' and proc.read_text().split()[21]==receipt['start_ticks']
   print('OWN_PID',receipt['pid'],'alive',alive)
  log=out/'worker.log'
  if log.exists(): print(log.read_text()[-3500:])
'''
    elif args.action == 'baseline':
        code = '''
source=parent/'int/WM-PHYS-FULL-v4-gpu'
assert source.is_dir()
root.mkdir(exist_ok=True)
out=root/'frozen_baseline';assert not out.exists();out.mkdir()
manifest={}
for seed in range(969420,969432):
 for future in range(8):
  name=f'w{seed}_k{future}.json'
  files=list(source.glob('shard*/'+name));assert len(files)==1
  path=files[0];raw=path.read_bytes();task=json.loads(raw)
  assert task['seed']==seed and task['future']==future
  identity=json.loads((path.parent/'IDENTITY.json').read_text())
  assert identity['model']=='WM-PHYS-FULL-v4-gpu' and identity['K']==8
  assert hashlib.sha256((path.parent/'IDENTITY.json').read_bytes()).hexdigest()==task['identity_sha256']
  with (out/name).open('xb') as f: f.write(raw)
  manifest[name]=dict(source=str(path),sha256=hashlib.sha256(raw).hexdigest())
save(out/'BASELINE.json',dict(owner='S-RB2-BREAK',development_only=True,source='FULL-v4/22cc32f6',files=manifest))
print(json.dumps(dict(tasks=len(manifest),development_only=True)))
'''
    else:
        snapshot_name = args.snapshot_name or args.tag
        if not snapshot_name.replace('_', '').isalnum():
            raise ValueError('simple snapshot name required')
        target = root/'docs/rb2break'/snapshot_name
        if not target.parent.is_dir():
            raise ValueError('local evidence parent missing')
        if target.exists():
            raise ValueError('new immutable local snapshot required')
        remote_name = ('frozen_baseline' if args.action == 'fetch-baseline' else
                       'birth_fit_'+args.birth_version if args.action == 'fetch-birth' else
                       'envelope_validation' if args.action == 'fetch-envelope' else
                       '' if args.action == 'fetch-source-audit' else out_name)
        code = f'''import hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path({JOB!r})/{remote_name!r};assert root.is_dir()
files={{}}
for seed in range(969420,969432):
 for future in range(8):
  path=root/f'w{{seed}}_k{{future}}.json'
  if path.exists():
   data=path.read_bytes()
   try: task=json.loads(data)
   except json.JSONDecodeError: continue
   assert task['seed']==seed and task['future']==future
   files[path.name]=data
for name in ('IDENTITY.json','DISPATCH.json','COMPLETE.json','BASELINE.json','FIT.json','VALIDATION.json','entrant_population.json','OBSERVATION_BINDING.json','OPERATIONAL_STOP.json','SOURCE_OBSERVABLE_AUDIT.json','GPU_PAUSE_1735.json','STOP','EXIT.json','worker.log'):
 path=root/name
 if path.exists(): files[name]=path.read_bytes()
receipt=dict(source=str(root),hashes={{name:hashlib.sha256(data).hexdigest() for name,data in files.items()}})
files['SNAPSHOT.json']=json.dumps(receipt,indent=1).encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for name,data in files.items():
  info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
'''
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', args.host,
            '/root/miniconda3/bin/python -B -c '+shlex.quote(code)], capture_output=True, timeout=180)
        if result.returncode:
            raise RuntimeError(result.stderr.decode())
        with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:gz') as tar:
            for member in tar.getmembers():
                assert member.isfile() and not member.name.startswith('/') and '..' not in Path(member.name).parts
            receipt = json.loads(tar.extractfile('SNAPSHOT.json').read())
            for name, expected in receipt['hashes'].items():
                assert hashlib.sha256(tar.extractfile(name).read()).hexdigest() == expected
            target.mkdir()
            tar.extractall(target, filter='data')
        print(json.dumps(dict(snapshot=str(target), files=len(receipt['hashes']))))
        return
    if args.action == 'wait':
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', args.host,
            '/root/miniconda3/bin/python -B -'], input=common+code, text=True,
            capture_output=True, timeout=args.wait_seconds+45)
        if result.returncode:
            raise RuntimeError(result.stdout+result.stderr)
        print(result.stdout)
    else:
        print(ssh(args.host, common+code, timeout=120))


if __name__ == '__main__':
    main()
