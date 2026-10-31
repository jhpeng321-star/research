"""Explicit safe staging, CPU dispatch and retrieval for S-C1CLOSED."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from probe import ssh, allowed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REMOTE = '/root/autodl-tmp/c4/c1closed20261031'


def run(argv, timeout=180):
    result = subprocess.run(argv, text=True, encoding='utf-8', capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    return result.stdout


def scp(source, dest):
    return run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', str(source), str(dest)], 600)


def stage():
    print(ssh('gpuA', f'''from pathlib import Path
import json,os,hashlib
p=Path({REMOTE!r})
assert p.parent.is_dir()
p.mkdir(exist_ok=True)
for name in ('ckpt','cpu_smoke','cpu_verified','evidence','panel'):
 (p/name).mkdir(exist_ok=True)
for name in ('historical','bbtraj','bbid'):
 (p/'ckpt'/name).mkdir(exist_ok=True)
for name in ('worker.py','dispatch.py'):
 q=p/name
 if q.exists():
  b=q.read_bytes(); backup=p/'evidence'/(name+'.'+hashlib.sha256(b).hexdigest())
  if not backup.exists(): backup.write_bytes(b)
print('verified destination',p,'load',os.getloadavg())
'''))
    for name in ('worker.py', 'dispatch.py', 'verify_cpu.py', 'aggregates.py'):
        if (HERE/name).exists():
            print(scp(HERE/name, f'gpuA:{REMOTE}/{name}'))
    old = Path('E:/research-dense/runs/active/WM-S-C1FAST-20261006/box')
    for tag, directory in (('historical','full_struct'), ('bbtraj','full_bb_traj'), ('bbid','full_bb_id')):
        path = allowed(old/directory/'wm_s.pt')
        print(scp(path, f'gpuA:{REMOTE}/ckpt/{tag}/wm_s.pt'))
        print(scp(old/directory/'TRAIN_RECEIPT.json', f'gpuA:{REMOTE}/ckpt/{tag}/TRAIN_RECEIPT.json'))


def smoke(model):
    command = f'nice -n 19 /root/miniconda3/bin/python -B -u {REMOTE}/worker.py --model {model} --world 0 --seconds 10 --out cpu_verified'
    code = f'''import os,subprocess,json,time
from pathlib import Path
assert os.getloadavg()[0] < 80
p=subprocess.run({command!r},shell=True,text=True,capture_output=True,
 env={{**os.environ,'PYTHONPATH':'/root/autodl-tmp/c4/speedhead20261030/pythondeps','CUDA_VISIBLE_DEVICES':''}})
print(p.stdout);print(p.stderr)
log=Path({REMOTE!r})/'evidence'/('smoke_{model}_'+str(time.time_ns())+'.json')
log.write_text(json.dumps(dict(command={command!r},returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)))
raise SystemExit(p.returncode)
'''
    print(ssh('gpuA', code))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('stage','smoke','fetch','launch','status','verify','canary','receipts','labels'))
    parser.add_argument('--model', default='wm2', choices=('wm2','bbtraj','bbid','historical'))
    args = parser.parse_args()
    if args.action == 'stage':
        stage()
    elif args.action == 'smoke':
        smoke(args.model)
    elif args.action == 'fetch':
        dest = HERE/'out'
        dest.mkdir(exist_ok=True)
        print(scp(f'gpuA:{REMOTE}/cpu_smoke/{args.model}_common_w0_k0_h2.json', dest))
    elif args.action == 'receipts':
        dest = HERE/'evidence'
        dest.mkdir(exist_ok=True)
        for name in ('CPU_VERIFIED*.json','CANARY_DONE.json','CANARY_OWNED.json'):
            print(scp(f'gpuA:{REMOTE}/evidence/{name}',dest))
    elif args.action == 'labels':
        dest = HERE/'out'
        dest.mkdir(exist_ok=True)
        print(ssh('cpuQ', '''from pathlib import Path
import os
p=Path('/home/ubuntu/wmdata20261031')
assert p.is_dir()
q=p/'c1closed';q.mkdir(exist_ok=True)
print('verified',q,'load',os.getloadavg())
''',python='python3'))
        for name in ('labels.py','aggregates.py'):
            print(scp(HERE/name,f'cpuQ:/home/ubuntu/wmdata20261031/c1closed/{name}'))
        print(ssh('cpuQ', '''import os,subprocess,json,fcntl
from pathlib import Path
assert os.getloadavg()[0] < 48
root=Path('/home/ubuntu/wmdata20261031/c1closed')
lock=(root/'LABELS.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
target=root/'labels.json'
env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
if not target.exists():
 with target.open('x') as out, (root/'LABELS.stderr').open('x') as err:
  p=subprocess.run(['nice','-n','19','/home/ubuntu/miniconda3/bin/python','-B',str(root/'labels.py')],env=env,stdout=out,stderr=err)
 assert p.returncode == 0, 'preserved label failure, inspect stderr before retry'
x=json.loads(target.read_text());assert len(x['rows']) == 200
print('label rows',len(x['rows']))
''',python='python3'))
        print(scp('cpuQ:/home/ubuntu/wmdata20261031/c1closed/labels.json',dest/'labels.json'))
    elif args.action == 'canary':
        print(ssh('gpuA', f'''import subprocess,os,json,time
from pathlib import Path
p=Path({REMOTE!r})
assert (p/'evidence/CPU_VERIFIED.json').is_file()
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free >= 5120, free
print('GPU admission MiB',free,'load',os.getloadavg())
out=p/'gpu_canary';out.mkdir(exist_ok=True)
command=['nice','-n','19','/root/miniconda3/bin/python','-B','-u',str(p/'worker.py'),
 '--model','wm2','--world','0','--seconds','10','--out','gpu_canary','--device','cuda']
with (p/'evidence/CANARY.log').open('x') as log:
 child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
  env={{**os.environ,'PYTHONPATH':'/root/autodl-tmp/c4/speedhead20261030/pythondeps'}})
 receipt=dict(pid=child.pid,start_ticks=Path(f'/proc/{{child.pid}}/stat').read_text().split(') ',1)[1].split()[19],command=command,peak_owned_mib=0)
 with (p/'evidence/CANARY_OWNED.json').open('x') as f: json.dump(receipt,f)
 while child.poll() is None:
  raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
  for line in raw.splitlines():
   pid,mem=[x.strip() for x in line.split(',')]
   if pid == str(child.pid) and mem.isdigit(): receipt['peak_owned_mib']=max(receipt['peak_owned_mib'],int(mem))
  time.sleep(2)
 receipt['returncode']=child.wait()
 with (p/'evidence/CANARY_DONE.json').open('x') as f: json.dump(receipt,f)
 print(json.dumps(receipt)); print((p/'evidence/CANARY.log').read_text())
 raise SystemExit(child.returncode)
'''))
    elif args.action == 'status':
        print(ssh('gpuA', f'''import json,os
from pathlib import Path
p=Path({REMOTE!r})
for name in ('DISPATCH.json','COMPLETE.json','STATUS.json'):
 q=p/name
 print(name,q.read_text() if q.exists() else 'PENDING')
print('completed',len(list((p/'panel').glob('*_h300.json'))),'load',os.getloadavg())
'''))
    elif args.action == 'verify':
        print(ssh('gpuA', f'''import subprocess,os
p=subprocess.run(['nice','-n','19','/root/miniconda3/bin/python','-B','{REMOTE}/verify_cpu.py'],
 env={{**os.environ,'PYTHONPATH':'/root/autodl-tmp/c4/speedhead20261030/pythondeps','CUDA_VISIBLE_DEVICES':''}},
 text=True,capture_output=True)
print(p.stdout);print(p.stderr)
raise SystemExit(p.returncode)
'''))
    else:
        print(ssh('gpuA', f'''import subprocess,os,json,time
from pathlib import Path
p=Path({REMOTE!r})
assert (p/'evidence/CPU_VERIFIED.json').is_file()
assert not (p/'DISPATCH.json').exists(), 'existing dispatch: inspect and resume, never duplicate'
with (p/'evidence/DISPATCH.log').open('x') as log:
 child=subprocess.Popen(['nice','-n','19','/root/miniconda3/bin/python','-B','-u',str(p/'dispatch.py')],
  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
  env={{**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}})
 ticks=Path(f'/proc/{{child.pid}}/stat').read_text().split(') ',1)[1].split()[19]
 receipt=dict(pid=child.pid,start_ticks=ticks,started=time.time())
 with (p/'evidence/LAUNCH.json').open('x') as f: json.dump(receipt,f)
 print(json.dumps(receipt))
'''))


if __name__ == '__main__':
    main()
