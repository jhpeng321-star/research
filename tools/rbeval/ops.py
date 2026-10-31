"""Explicit own-root staging/launch/status, no process signalling."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = '/root/rbeval20261031'


def ssh(code):
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'gpuR',
                        '/root/miniconda3/bin/python -B -'], input=code,
                       text=True, capture_output=True, timeout=180)
    if p.returncode:
        raise RuntimeError(p.stdout+'\n'+p.stderr)
    return p.stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('audit', 'stage', 'canary', 'launch', 'status', 'fetch',
                                    'stage-scoring', 'finalize', 'fetch-reports'))
    args = p.parse_args()
    if args.action in ('audit', 'status'):
        print(ssh(f'''
from pathlib import Path
import json, os, subprocess
p=Path({ROOT!r})
print('load',os.getloadavg(),'cpus',os.cpu_count())
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv'],text=True))
if p.exists():
 for f in sorted(p.glob('FINAL*.json')): print(f.name,f.read_text())
 log=p/'finalize.log'
 if log.exists(): print('FINALIZER',log.read_text()[-2000:])
 for d in sorted(p.glob('wm2_shard*')):
  if not d.is_dir(): continue
  print(d.name,'tasks',len(list(d.glob('w*_k*.json'))),'complete',(d/'COMPLETE.json').exists())
  log=p/(d.name+'.log')
  if log.exists(): print(log.read_text()[-6000:])
for d in Path('/proc').iterdir():
 if d.name.isdecimal():
  try:
   cmd=(d/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace')
   if 'run_wm.py' in cmd or 'rbeval' in cmd: print('PROCESS',d.name,cmd)
  except (FileNotFoundError,PermissionError,ProcessLookupError): pass
'''))
    elif args.action == 'stage':
        files = [HERE/n for n in ('run_wm.py', 'core.py', 'references.json', 'real_observables.json', 'PREPARED.json')]
        manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        print(ssh(f'''
from pathlib import Path
import hashlib
p=Path({ROOT!r}); assert p.parent.is_dir(); p.mkdir(exist_ok=True)
c=p/'code'; c.mkdir(exist_ok=True)
expected={manifest!r}
for name,sha in expected.items():
 f=c/name
 if f.exists(): assert hashlib.sha256(f.read_bytes()).hexdigest()==sha, 'preserve existing source'
print('Staging admitted: own root, existing sources identical')
'''))
        subprocess.run(['scp', '-o', 'BatchMode=yes', *map(str, files), f'gpuR:{ROOT}/code/'], check=True, timeout=180)
    elif args.action in ('canary', 'launch'):
        canary = args.action == 'canary'
        print(ssh(f'''
from pathlib import Path
import fcntl, json, os, subprocess, time
root=Path({ROOT!r}); assert (root/'code/run_wm.py').is_file()
lock=(root/'DISPATCH.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert os.getloadavg()[0]<24, 'host busy'
mem=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
assert int(mem.strip())>{6000 if canary else 21500}, 'GPU busy'
launched=[]
for shard in { [0] if canary else [0,1,2,3] }:
 out=root/f'wm2_shard{{shard}}'; out.mkdir(exist_ok=True)
 for receipt in out.glob('PROCESS_*.json'):
  old=json.loads(receipt.read_text()); proc=Path('/proc')/str(old['pid'])
  if proc.exists():
   try: alive=(proc/'stat').read_text().split()[21]==old['start_ticks']
   except FileNotFoundError: alive=False
   assert not alive, 'existing owned worker live: never duplicate'
 log=(root/f'wm2_shard{{shard}}.log').open('a')
 argv=['nice','-n','19','/root/miniconda3/bin/python','-B','-u',str(root/'code/run_wm.py'),
       '--out',str(out),'--shard',str(shard),'--shards','4','--memory-gib','5']
 if {canary!r}: argv+=['--max-tasks','1']
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
   PYTHONPATH='/root/autodl-tmp/c4/speedhead20261030/pythondeps')
 child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,cwd=root)
 launched.append(dict(pid=child.pid,argv=argv,unix=time.time()))
receipt=root/('DISPATCH_'+str(time.time_ns())+'.json')
with receipt.open('x') as f: json.dump(launched,f,indent=1)
print(json.dumps(dict(receipt=str(receipt),launched=launched)))
'''))
    elif args.action == 'fetch':
        target = HERE/'evidence'
        target.mkdir(exist_ok=True)
        subprocess.run(['scp','-r','-o','BatchMode=yes',f'gpuR:{ROOT}/wm2_shard*',str(target)], check=True, timeout=600)
        subprocess.run(['scp','-o','BatchMode=yes',f'gpuR:{ROOT}/DISPATCH_*.json',str(target)], check=True, timeout=60)
    elif args.action == 'stage-scoring':
        files = [HERE/n for n in ('finalize.py', 'score.py', 'core.py', 'references.json', 'real_observables.json')]
        manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        print(ssh(f'''
from pathlib import Path
import hashlib
p=Path({ROOT!r}); assert p.is_dir()
c=p/'analysis_scoring'; c.mkdir(exist_ok=True)
for name,sha in {manifest!r}.items():
 f=c/name
 if f.exists(): assert hashlib.sha256(f.read_bytes()).hexdigest()==sha, 'preserve scoring source'
print('Own-root CPU scoring staging admitted')
'''))
        subprocess.run(['scp', '-o', 'BatchMode=yes', *map(str, files), f'gpuR:{ROOT}/analysis_scoring/'], check=True, timeout=180)
    elif args.action == 'finalize':
        print(ssh(f'''
from pathlib import Path
import fcntl,json,os,subprocess,time
root=Path({ROOT!r}); assert (root/'analysis_scoring/finalize.py').is_file()
lock=(root/'FINALIZE_DISPATCH.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (root/'FINALIZED.json').exists(), 'already scored'
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for p in root.glob('FINALIZER_*.json'):
 old=json.loads(p.read_text()); proc=Path('/proc')/str(old['pid']) if 'pid' in old else None
 if proc is not None and proc.exists():
  assert old['boot_id']!=boot or (proc/'stat').read_text().split()[21]!=old['start_ticks'], 'existing finalizer alive'
assert os.getloadavg()[0]<24, 'host CPU busy'
argv=['nice','-n','19','/root/miniconda3/bin/python','-B','-u',str(root/'analysis_scoring/finalize.py'),'--run-root',str(root)]
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (root/'finalize.log').open('a') as log:
 child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,cwd=root)
receipt=dict(pid=child.pid,argv=argv,unix=time.time())
with (root/('ANALYSIS_DISPATCH_'+str(time.time_ns())+'.json')).open('x') as f: json.dump(receipt,f,indent=1)
print(json.dumps(receipt))
'''))
    elif args.action == 'fetch-reports':
        target = HERE/'evidence'
        target.mkdir(exist_ok=True)
        subprocess.run(['scp','-o','BatchMode=yes',f'gpuR:{ROOT}/FINAL*.json',f'gpuR:{ROOT}/ANALYSIS_DISPATCH_*.json',
                        f'gpuR:{ROOT}/finalize.log',str(target)], check=True, timeout=60)
        available = ssh(f"from pathlib import Path; print(int((Path({ROOT!r})/'reports').is_dir()))").strip()
        if available == '1':
            subprocess.run(['scp','-r','-o','BatchMode=yes',f'gpuR:{ROOT}/reports',str(target)], check=True, timeout=600)


if __name__ == '__main__':
    main()
