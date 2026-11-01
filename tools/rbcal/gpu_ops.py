"""Owned dev12-only inference, immutable stage, finite auto-scoring continuation.

Two five-GiB workers leave most GPU memory for S-WMTEACH. Never signal a PID.
Successful canary is reused, existing tasks/runs are immutable, no retry loop.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from tools.rbint.remote import ssh

ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/root/rbeval20261031/rb2cal'


def stage():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    paths = ['src/wms', 'tools/rbcal', 'tools/rbint', 'tools/rbeval', 'tools/rbfit',
        'tools/wmdecide', 'configs/wm_phys_rb2_cal_v1.toml',
        'configs/execparity_cav_idm_v1.json', 'configs/native_cav_head_v5.json',
        'artifacts/rbfit/shrinkage_v1', 'docs/paper_results/rbunc/observed_arrivals.json',
        'docs/paper_results/rb2cal/v1/fit/calibration.json']
    archive = subprocess.check_output(['git', 'archive', '--format=tar', commit, *paths], cwd=ROOT)
    digest = hashlib.sha256(archive).hexdigest()
    target = REMOTE+'/code_'+commit[:12]
    code = f'''
from pathlib import Path
import json,sys,hashlib,io,tarfile
parent=Path({REMOTE!r}); assert parent.parent.is_dir(); parent.mkdir(exist_ok=True)
target=Path({target!r}); assert not target.exists(); target.mkdir()
data=sys.stdin.buffer.read(); assert hashlib.sha256(data).hexdigest()=={digest!r}
with tarfile.open(fileobj=io.BytesIO(data)) as tar:
 for m in tar.getmembers():
  assert not m.issym() and not m.islnk() and '..' not in Path(m.name).parts and not Path(m.name).is_absolute()
 tar.extractall(target,filter='data')
receipt=dict(commit={commit!r},archive_sha256={digest!r},runtime=str(target))
with (target/'STAGE.json').open('x') as f:json.dump(receipt,f,indent=1)
print(json.dumps(receipt))
'''
    run = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'gpuA', '/root/miniconda3/bin/python',
        '-B', '-c', shlex.quote(code)], input=archive, capture_output=True, timeout=180)
    if run.returncode:
        raise RuntimeError(run.stdout.decode()+run.stderr.decode())
    print(run.stdout.decode())


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('command', choices=('stage', 'resources', 'canary', 'panel', 'status', 'fetch'))
    p.add_argument('--runtime')
    p.add_argument('--version', default='dev_v1')
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = p.parse_args()
    if not args.version.replace('_', '').isalnum():
        raise ValueError('simple owned version required')
    if args.command == 'stage':
        stage()
        return
    if args.command == 'resources':
        print(ssh('gpuA', '''
import os,json,subprocess,time
from pathlib import Path
root=Path('/sys/fs/cgroup')
maximum=(root/'cpu.max').read_text().split()
quota=os.cpu_count() if maximum[0]=='max' else int(maximum[0])/int(maximum[1])
def usage():return int(dict(line.split() for line in (root/'cpu.stat').read_text().splitlines())['usage_usec'])
before=usage();started=time.monotonic();time.sleep(1);cores=(usage()-before)/1e6/(time.monotonic()-started)
maximum=(root/'memory.max').read_text().strip();current=int((root/'memory.current').read_text())
available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
if maximum!='max':available=min(available,int(maximum)-current)
print(json.dumps(dict(cpu_quota=quota,cpu_used_cores=cores,free_memory_gib=available/2**30,load=os.getloadavg())))
print(subprocess.check_output(['nvidia-smi'],text=True))
'''))
        return
    root = REMOTE+'/'+args.version
    if args.command in ('canary', 'panel'):
        if not args.runtime or not args.runtime.startswith(REMOTE+'/code_'):
            raise ValueError('immutable own runtime required')
        print(ssh('gpuA', f'''
from pathlib import Path
import json,os,subprocess,time
root=Path({root!r});runtime=Path({args.runtime!r});device={args.device!r}; assert (runtime/'STAGE.json').is_file()
assert os.getloadavg()[0]<os.cpu_count()*.75
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
if device=='cuda':assert free>16000, 'leave headroom for teacher'
else:
 cg=Path('/sys/fs/cgroup');maximum=(cg/'cpu.max').read_text().split()
 quota=os.cpu_count() if maximum[0]=='max' else int(maximum[0])/int(maximum[1])
 def usage():return int(dict(line.split() for line in (cg/'cpu.stat').read_text().splitlines())['usage_usec'])
 before=usage();started=time.monotonic();time.sleep(1);cores=(usage()-before)/1e6/(time.monotonic()-started)
 assert cores+(1 if {args.command!r}=='canary' else 2)<quota*.8, 'preserve CPU quota headroom'
 maximum=(cg/'memory.max').read_text().strip();current=int((cg/'memory.current').read_text())
 if maximum!='max':assert int(maximum)-current>8*2**30, 'preserve CPU memory headroom'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if device=='cuda' else '',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 PYTHONPATH='/root/autodl-tmp/c4/speedhead20261030/pythondeps')
def launch(shard,canary):
 out=root/('shard'+str(shard));out.mkdir(exist_ok=True)
 command=['nice','-n','19','/root/miniconda3/bin/python','-B','-u',str(runtime/'tools/rbcal/evaluate_dev.py'),
  '--out',str(out),'--runtime-root',str(runtime),'--wm','/root/autodl-tmp/c4/wmcombined_wm2/wm_s.pt',
  '--wm-config',str(runtime/'configs/wm_phys_rb2_cal_v1.toml'),'--model-name','WM-PHYS-RB2-CAL-v1-dev'+('-cpu' if device=='cpu' else ''),
  '--shard',str(shard),'--shards','2','--memory-gib','5','--device',device]
 if canary:command+=['--max-tasks','1']
 with (root/('canary.log' if canary else 'worker'+str(shard)+'.log')).open('x') as log:
  child=subprocess.Popen(command,cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 return dict(pid=child.pid,start_ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21],command=command)
if {args.command!r}=='canary':
 assert root.parent.is_dir() and not root.exists();root.mkdir()
 receipt=dict(owner='S-RB2-CAL',runtime=str(runtime),worlds=list(range(969420,969432)),
  gpu_free_mib=free,device=device,source=json.loads((runtime/'STAGE.json').read_text()),process=launch(0,True),start_unix=time.time())
 with (root/'CANARY.json').open('x') as f:json.dump(receipt,f,indent=1)
 print(json.dumps(receipt))
else:
 canary=json.loads((root/'CANARY.json').read_text()); assert canary['runtime']==str(runtime) and canary.get('device','cuda')==device
 process=canary['process']; stat=Path('/proc/'+str(process['pid'])+'/stat')
 assert not stat.exists() or stat.read_text().split()[2]=='Z' or stat.read_text().split()[21]!=process['start_ticks'], 'canary still live'
 tasks=list((root/'shard0').glob('w*_k*.json'));assert len(tasks)==1 and tasks[0].name=='w969420_k0.json'
 assert not (root/'shard0/STOP').exists()
 task=json.loads(tasks[0].read_text());assert task['future_sampling']['population_audit']['conserved']
 assert not (root/'PANEL.json').exists()
 processes=[launch(i,False) for i in range(2)]
 receipt=dict(owner='S-RB2-CAL',runtime=str(runtime),workers=processes,start_unix=time.time(),canary_reused=tasks[0].name)
 with (root/'PANEL.json').open('x') as f:json.dump(receipt,f,indent=1)
 print(json.dumps(receipt))
''', timeout=90))
    elif args.command == 'status':
        print(ssh('gpuA', f'''
from pathlib import Path
import json,os
root=Path({root!r})
counts={{s:sum((root/('shard'+str(i))/('w'+str(s)+'_k'+str(k)+'.json')).is_file() for i in range(2) for k in range(8)) for s in range(969420,969432)}}
print(json.dumps(dict(counts=counts,load=os.getloadavg(),stops=[str(p) for p in root.glob('shard*/STOP')])))
for name in ['CANARY.json','PANEL.json','canary.log','worker0.log','worker1.log']:
 path=root/name
 if path.exists():print(name,path.read_text())
''', timeout=60))
    else:
        # Download only explicitly named dev12 task files and identity/own receipts.
        text = ssh('gpuA', f'''
from pathlib import Path
import json,base64,hashlib
root=Path({root!r});paths=[]
for i in range(2):
 shard=root/('shard'+str(i))
 if not shard.is_dir():continue
 paths+=[shard/'IDENTITY.json']
 for s in range(969420,969432):
  for k in range(8):
   path=shard/('w'+str(s)+'_k'+str(k)+'.json')
   if path.exists():paths.append(path)
paths += [root/n for n in ['CANARY.json','PANEL.json','canary.log','worker0.log','worker1.log'] if (root/n).exists()]
print(json.dumps({{p.relative_to(root).as_posix():dict(data=base64.b64encode(p.read_bytes()).decode(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}}))
''', timeout=180)
        import base64
        destination = ROOT/'runs'/('rb2cal_'+args.version+'_fetched')
        assert destination.parent.is_dir() and not destination.exists()
        destination.mkdir()
        for name, value in json.loads(text).items():
            path = destination/name
            path.parent.mkdir(exist_ok=True)
            raw = base64.b64decode(value['data'])
            assert hashlib.sha256(raw).hexdigest() == value['sha256']
            with path.open('xb') as f:
                f.write(raw)
        print(destination)


if __name__ == '__main__':
    main()
