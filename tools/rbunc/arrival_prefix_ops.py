"""One bounded CPU-only arrival diagnostic using cpuQ's existing SciPy install."""
from pathlib import Path
import argparse
import base64
import hashlib
import json

from remote import PYTHON, REMOTE, ROOT, ssh, stage

OUT = REMOTE + '/arrival_prefix_v1'
RESULT = 'docs/paper_results/rbunc/arrival_prefix_v1.json'
FILES = ('tests/unit/test_arrival_prefix.py',
         'docs/paper_results/RBUNC_ARRIVAL_PREFIX_DESIGN.md',
         'docs/paper_results/rbunc/observed_arrivals.json')


def run():
    # Inspect before staging; never relaunch a completed or running diagnostic.
    print(ssh(f'''
import os,json,subprocess
from pathlib import Path
assert not Path({OUT!r}).exists(), 'diagnostic already claimed; use fetch'
assert os.cpu_count()>=32 and os.getloadavg()[0]<24, 'CPU admission unavailable'
gpu=subprocess.run(['sh','-c','command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'],capture_output=True,text=True)
print(json.dumps(dict(load=os.getloadavg(),cpus=os.cpu_count(),gpu_status=gpu.returncode,
 gpu=gpu.stdout,cpu_only=True)))
'''), flush=True)
    code = stage()
    payload = {name:base64.b64encode((ROOT/name).read_bytes()).decode() for name in FILES}
    snapshot = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    print(ssh(f'''
from pathlib import Path
import base64,json,os,subprocess,sys,time
code=Path({code!r});assert code.is_dir()
for name,data in json.load(sys.stdin).items():
 p=code/name;p.parent.mkdir(parents=True,exist_ok=True);raw=base64.b64decode(data)
 if p.exists():assert p.read_bytes()==raw
 else:
  with p.open('xb') as f:f.write(raw)
root=Path({OUT!r});assert root.parent.is_dir();root.mkdir(exist_ok=False)
assert os.cpu_count()>=32 and os.getloadavg()[0]<24, 'CPU admission changed'
os.nice(19)
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(code/'src')+':'+str(code))
commands=[['-m','pytest','-q','-p','no:cacheprovider','tests/unit/test_arrival_prefix.py'],
 ['tools/rbunc/arrival_prefix.py','--out',str(root/'result.json')]]
receipt=dict(code=str(code),input_snapshot_sha256={snapshot!r},load=os.getloadavg(),
 cpu_only=True,start_unix=time.time(),commands=commands,pid=os.getpid())
with (root/'DISPATCH.json').open('x') as f:json.dump(receipt,f)
results=[]
for i,args in enumerate(commands):
 with (root/f'step{{i}}.log').open('x') as log:
  child=subprocess.run([{PYTHON!r},'-B',*args],cwd=code,env=env,
   stdout=log,stderr=subprocess.STDOUT,timeout=120)
 results.append(dict(step=i,rc=child.returncode))
 if child.returncode:break
with (root/'TERMINAL.json').open('x') as f:json.dump(dict(results=results,end_unix=time.time()),f)
print(json.dumps(dict(code=str(code),results=results,output=str(root))))
for i in range(len(results)):print((root/f'step{{i}}.log').read_text())
assert len(results)==2 and all(r['rc']==0 for r in results)
''', json.dumps(payload), timeout=280), flush=True)


def fetch():
    text = ssh(f'''
from pathlib import Path
import json
root=Path({OUT!r})
terminal=json.loads((root/'TERMINAL.json').read_text())
assert len(terminal['results'])==2 and all(r['rc']==0 for r in terminal['results'])
result=json.loads((root/'result.json').read_text())
result['execution_receipts']=dict(dispatch=json.loads((root/'DISPATCH.json').read_text()),terminal=terminal,
 tests=(root/'step0.log').read_text())
print(json.dumps(result,allow_nan=False))
''')
    path = ROOT/RESULT
    assert path.parent.is_dir()
    with path.open('x', encoding='utf-8') as stream:
        stream.write(text)
    print(json.dumps(dict(output=str(path), summaries=json.loads(text)['summaries']), indent=1))


def verify():
    """Verify a changed optional-dependency guard; do not repeat the analysis."""
    code = stage()
    raw = (ROOT/'tests/unit/test_arrival_prefix.py').read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    print(ssh(f'''
from pathlib import Path
import base64,json,os,subprocess
code=Path({code!r});root=Path({OUT!r});assert root.is_dir()
assert os.getloadavg()[0]<24 and os.cpu_count()>=32
target=root/('test_'+{digest!r}[:12]+'.py')
data=base64.b64decode({base64.b64encode(raw).decode()!r})
if target.exists():assert target.read_bytes()==data
else:
 with target.open('xb') as f:f.write(data)
receipt=root/('VERIFY_'+{digest!r}[:12]+'.json')
assert not receipt.exists(), 'verification already recorded'
os.nice(19)
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(code/'src')+':'+str(code))
child=subprocess.run([{PYTHON!r},'-B','-m','pytest','-q','-p','no:cacheprovider',str(target)],
 cwd=code,env=env,capture_output=True,text=True,timeout=60)
result=dict(code=str(code),test_sha256={digest!r},rc=child.returncode,
 stdout=child.stdout,stderr=child.stderr,cpu_only=True)
with receipt.open('x') as f:json.dump(result,f)
print(json.dumps(result));assert child.returncode==0
'''))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=('run', 'fetch', 'verify'))
    args = parser.parse_args()
    {'run':run, 'fetch':fetch, 'verify':verify}[args.command]()
