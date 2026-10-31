"""Exclusive UNC observer claim; no peer dispatches, signals, or scoring calls."""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/root/rbeval20261031/rbunc_support_v1'
PYTHON = '/root/miniconda3/bin/python'
PANEL = '/root/rbeval20261031/int/WM-PHYS-FULL-v1'


def ssh(source, data=None):
    encoded = base64.b64encode(source.encode()).decode()
    command = PYTHON + " -B -c 'import base64;exec(base64.b64decode(\"" + encoded + "\"))'"
    process = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'gpuA', command],
                             input=data, text=True, capture_output=True, timeout=30)
    if process.returncode:
        raise RuntimeError(process.stdout+process.stderr)
    return process.stdout


def target_snapshot():
    path = ROOT/'docs/paper_results/rbunc/preferred_source_probe.json'
    raw = path.read_bytes()
    bundle = json.loads(raw)
    assert [r['world'] for r in bundle['records']] == list(range(969400, 969432))
    assert all(r['rc'] == 0 for r in bundle['receipts']['probe_TERMINAL.json']['results'])
    return dict(schema='rbunc_native_interim_targets/1', source_sha256=hashlib.sha256(raw).hexdigest(),
        worlds={str(r['world']):dict(initial_sha256=r['receipt']['initial_sha256'],
            real=r['real'], real_sha256=r['real_sha256'])
            for r in bundle['records'] if 969420 <= r['world'] <= 969431})


def dispatch():
    print(ssh(f'''
from pathlib import Path
import json,os,subprocess
assert Path({PANEL!r}).is_dir()
assert not Path({REMOTE!r}).exists(), 'UNC observer already claimed; use status'
assert os.getloadavg()[0]+1<os.cpu_count(), 'wait for a free CPU'
gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
print(json.dumps(dict(load=os.getloadavg(),cpus=os.cpu_count(),gpu_status=gpu.returncode,gpu=gpu.stdout,cpu_only=True)))
'''), flush=True)
    files = {'watcher.py':(ROOT/'tools/rbunc/interim_watch.py').read_bytes(),
             'targets.json':json.dumps(target_snapshot(), sort_keys=True).encode(),
             'DESIGN.md':(ROOT/'docs/paper_results/RBUNC_INTERIM_DESIGN.md').read_bytes()}
    payload = {name:base64.b64encode(raw).decode() for name,raw in files.items()}
    print(ssh(f'''
from pathlib import Path
from datetime import datetime
import base64,hashlib,json,os,subprocess,sys,time
assert time.time()<datetime.fromisoformat('2026-11-01T09:00:00+08:00').timestamp()
root=Path({REMOTE!r});assert root.parent.is_dir()
assert os.getloadavg()[0]+1<os.cpu_count(), 'CPU admission changed'
root.mkdir(exist_ok=False)
hashes={{}}
for name,encoded in json.load(sys.stdin).items():
 raw=base64.b64decode(encoded)
 with (root/name).open('xb') as f:f.write(raw)
 hashes[name]=hashlib.sha256(raw).hexdigest()
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
argv=['nice','-n','19',{PYTHON!r},'-B','-u',str(root/'watcher.py'),
 '--panel',{PANEL!r},'--out',str(root),'--targets',str(root/'targets.json')]
with (root/'watcher.log').open('x') as log:
 child=subprocess.Popen(argv,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=child.pid,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 start_ticks=Path(f'/proc/{{child.pid}}/stat').read_text().split()[21],source_hashes=hashes,
 argv=argv,unix=time.time(),owner='S-RB-UNC',cpu_only=True,peer_process_actions=0)
with (root/'DISPATCH.json').open('x') as f:json.dump(receipt,f)
print(json.dumps(receipt))
''', json.dumps(payload)), flush=True)


def snapshot():
    return ssh(f'''
from pathlib import Path
import json
root=Path({REMOTE!r});receipt=json.loads((root/'DISPATCH.json').read_text())
try:
 stat=Path('/proc/'+str(receipt['pid'])+'/stat').read_text().split()
 live=(stat[2]!='Z' and stat[21]==receipt['start_ticks'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==receipt['boot_id'])
except FileNotFoundError:live=False
names=['DISPATCH.json','STARTED.json','TERMINAL.json','FAILED.json']
names += sorted(p.name for p in root.glob('DIAGNOSTIC_*.json'))
status=sorted(root.glob('STATUS_*.json'))
if status:names.append(status[-1].name)
print(json.dumps(dict(remote_root=str(root),live=live,files={{name:json.loads((root/name).read_text()) for name in names if (root/name).exists()}},log=(root/'watcher.log').read_text()),allow_nan=False))
''')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=('dispatch', 'status', 'fetch', 'inspect'))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.command == 'inspect':
        print(ssh(f'''
from pathlib import Path
import json,os
print(json.dumps(dict(root_exists=Path({REMOTE!r}).exists(),panel_exists=Path({PANEL!r}).is_dir(),
 cpus=os.cpu_count(),load=os.getloadavg(),one_cpu_available=os.getloadavg()[0]+1<os.cpu_count())))
'''))
    elif args.command == 'dispatch':
        dispatch()
    else:
        text = snapshot()
        if args.command == 'fetch':
            if args.out is None or not args.out.parent.is_dir():
                raise ValueError('explicit NEW snapshot path in an existing directory required')
            with args.out.open('x', encoding='utf-8') as stream:
                stream.write(text)
        print(text)
