"""Stage task-owned code to one new immutable CPU directory; no process signalling."""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PYTHON = '/home/ubuntu/miniconda3/bin/python'
REMOTE = '/home/ubuntu/rbunc20261031'


def ssh(source, data=None, timeout=300):
    encoded = base64.b64encode(source.encode()).decode()
    command = PYTHON + " -B -c 'import base64;exec(base64.b64decode(\"" + encoded + "\"))'"
    run = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'cpuQ', command],
                         input=data, text=True, capture_output=True, timeout=timeout)
    if run.returncode:
        raise RuntimeError(run.stdout + run.stderr)
    return run.stdout


def stage():
    paths = list((ROOT/'src/wms').rglob('*.py')) + list((ROOT/'src/wms/model').glob('*.json'))
    paths += list((ROOT/'tools/rbunc').glob('*.py'))
    payload = {p.relative_to(ROOT).as_posix(): base64.b64encode(p.read_bytes()).decode()
               for p in paths}
    version = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    code = f'''
from pathlib import Path
import json,base64,sys
root=Path({REMOTE!r}); assert root.parent.is_dir()
root.mkdir(exist_ok=True)
target=root/'code'/{version!r}; target.mkdir(parents=True,exist_ok=True)
for name,encoded in json.load(sys.stdin).items():
 p=target/name; p.parent.mkdir(parents=True,exist_ok=True)
 data=base64.b64decode(encoded)
 if p.exists(): assert p.read_bytes()==data
 else:
  with p.open('xb') as f:f.write(data)
print(str(target))
'''
    return ssh(code, json.dumps(payload)).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('stage', 'inspect', 'run', 'status', 'panel', 'fetch'))
    p.add_argument('--code')
    p.add_argument('--world', type=int, default=969400)
    p.add_argument('--horizon', type=int, default=120)
    args = p.parse_args()
    if args.command == 'stage':
        print(stage())
    elif args.command == 'inspect':
        print(ssh('''
from pathlib import Path
import hashlib,os,json,torch
torch.set_num_threads(1)
p=Path('/home/ubuntu/e3final/inputs/wm_s_best.pt')
d=torch.load(p,map_location='cpu',weights_only=True)
print(json.dumps(dict(load=os.getloadavg(),cpus=os.cpu_count(),checkpoint=str(p),
 sha256=hashlib.sha256(p.read_bytes()).hexdigest(),config=d['config']),default=str))
'''))
    elif args.command == 'status':
        print(ssh(f'''
from pathlib import Path
import json,os
root=Path({REMOTE!r})/'fit_probe_v1'
print('LOAD',os.getloadavg())
print('COMPLETE',[s for s in range(969400,969432) if (root/f'w{{s}}_h{args.horizon}.json').exists()])
p=root/'w{args.world}_h{args.horizon}.log'
if p.exists():print(p.read_text())
'''))
    elif args.command == 'fetch':
        text = ssh(f'''
from pathlib import Path
import json
root=Path({REMOTE!r})/'fit_probe_v1'
records=[]
for seed in range(969400,969432):
 p=root/f'w{{seed}}_h{args.horizon}.json'
 if p.exists():records.append(json.loads(p.read_text()))
print(json.dumps(records))
''')
        out = ROOT/'docs/paper_results/rbunc'
        assert out.parent.is_dir()
        out.mkdir(exist_ok=True)
        with (out/f'fit_h{args.horizon}.json').open('x', encoding='utf-8') as f:
            f.write(text)
        print('FETCHED', len(json.loads(text)))
    elif args.command == 'panel':
        if not args.code or not args.code.startswith(REMOTE+'/code/'):
            raise ValueError('explicit immutable staged code required')
        print(ssh(f'''
import subprocess,os,json
from pathlib import Path
root=Path({REMOTE!r});out=root/'fit_probe_v1';assert out.is_dir()
assert os.getloadavg()[0]<16,'insufficient free CPUs'
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 PYTHONPATH={args.code!r}+'/src:'+{args.code!r})
log=out/'PANEL.log'
with log.open('x') as f:
 job=subprocess.Popen([{PYTHON!r},'-B',{args.code!r}+'/tools/rbunc/panel.py',
  '--out',str(out),'--horizon','{args.horizon}'],cwd={args.code!r},env=env,
  stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(job.pid)
'''))
    else:
        if not args.code or not args.code.startswith(REMOTE+'/code/'):
            raise ValueError('explicit immutable staged code required')
        if args.world not in range(969400, 969432):
            raise ValueError('FIT world only')
        code = f'''
import subprocess,os
from pathlib import Path
root=Path({REMOTE!r}); assert root.is_dir()
out=root/'fit_probe_v1'; out.mkdir(exist_ok=True)
log=out/'w{args.world}_h{args.horizon}.log'
with log.open('x') as f:
 env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
  PYTHONPATH={args.code!r}+'/src:'+{args.code!r})
 job=subprocess.Popen([{PYTHON!r},'-B',{args.code!r}+'/tools/rbunc/fit_probe.py',
  '--out',str(out),'--world','{args.world}','--horizon','{args.horizon}'],
  cwd={args.code!r},env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 print(job.pid)
'''
        print(ssh(code))


if __name__ == '__main__':
    main()
