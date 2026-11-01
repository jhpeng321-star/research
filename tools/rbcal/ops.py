"""Exclusive single-thread CPU calibration staging; no peer process control."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

from tools.rbint.remote import ssh

ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/home/ubuntu/rb2cal20261101'


def stage():
    paths = list((ROOT/'src/wms').rglob('*.py'))
    paths += list((ROOT/'src/wms/model').glob('*.json'))
    for folder in ('tools/rbcal', 'tools/wmdecide', 'tools/rbint', 'tools/rbfit'):
        paths += list((ROOT/folder).glob('*.py'))
    paths += [ROOT/'docs/paper_results/rbunc/observed_arrivals.json',
              ROOT/'docs/paper_results/RB2_CAL_DESIGN.md']
    for name in ('idm_population.json', 'idm_shrinkage.json'):
        paths.append(ROOT/'artifacts/rbfit/shrinkage_v1'/name)
    payload = {p.relative_to(ROOT).as_posix(): base64.b64encode(p.read_bytes()).decode() for p in paths}
    version = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    text = ssh('cpuQ', f'''
from pathlib import Path
import json,base64,os,subprocess
payload=json.loads({json.dumps(payload)!r})
root=Path({REMOTE!r}); assert root.parent.is_dir()
root.mkdir(exist_ok=True)
target=root/'code'/{version!r}; target.mkdir(parents=True,exist_ok=True)
for name,encoded in payload.items():
 p=target/name; p.parent.mkdir(parents=True,exist_ok=True)
 data=base64.b64decode(encoded)
 if p.exists(): assert p.read_bytes()==data
 else:
  with p.open('xb') as f:f.write(data)
print(json.dumps(dict(code=str(target),load=os.getloadavg(),cpus=os.cpu_count())))
''', timeout=180)
    print(text)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('command', choices=('stage', 'fit', 'status', 'fetch'))
    p.add_argument('--code')
    p.add_argument('--version', default='v1')
    args = p.parse_args()
    if not args.version.replace('_', '').isalnum():
        raise ValueError('simple output version required')
    out = REMOTE+'/'+args.version
    if args.command == 'stage':
        stage()
    elif args.command == 'fit':
        if not args.code or not args.code.startswith(REMOTE+'/code/'):
            raise ValueError('explicit immutable stage required')
        print(ssh('cpuQ', f'''
from pathlib import Path
import os,json,subprocess,time
root=Path({out!r}); assert root.parent.is_dir() and not root.exists()
assert os.getloadavg()[0] < os.cpu_count()*.8, 'CPU busy'
root.mkdir()
code={args.code!r}
command=['nice','-n','19','/home/ubuntu/miniconda3/bin/python','-B','-m','tools.rbcal.fit',
 '--root','/home/ubuntu/wmdata20261031/views',
 '--population',code+'/artifacts/rbfit/shrinkage_v1/idm_population.json',
 '--shrinkage',code+'/artifacts/rbfit/shrinkage_v1/idm_shrinkage.json',
 '--arrivals',code+'/docs/paper_results/rbunc/observed_arrivals.json', '--out',str(root/'fit')]
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 CUDA_VISIBLE_DEVICES='',PYTHONPATH=code+'/src:'+code)
# The shell owns only this child; its terminal receipt is never a peer PID signal.
supervisor='import subprocess,json,time; from pathlib import Path; r=subprocess.run('+repr(command)+'); Path('+repr(str(root/'TERMINAL.json'))+').write_text(json.dumps(dict(rc=r.returncode,end_unix=time.time())))'
with (root/'fit.log').open('x') as log:
 child=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-c',supervisor],
  cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 receipt=dict(pid=child.pid,code=code,command=command,start_unix=time.time(),
  start_ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21],load=os.getloadavg())
 with (root/'DISPATCH.json').open('x') as f:json.dump(receipt,f)
 print(json.dumps(receipt))
'''))
    elif args.command == 'status':
        print(ssh('cpuQ', f'''
from pathlib import Path
import json,os
root=Path({out!r})
print(json.dumps(dict(load=os.getloadavg(),terminal=(root/'TERMINAL.json').read_text() if (root/'TERMINAL.json').exists() else None)))
print((root/'fit.log').read_text())
'''))
    else:
        text = ssh('cpuQ', f'''
from pathlib import Path
import json,base64,hashlib
root=Path({out!r})
assert json.loads((root/'TERMINAL.json').read_text())['rc']==0
paths=[root/'DISPATCH.json',root/'TERMINAL.json',root/'fit.log']+sorted((root/'fit').glob('*.json'))
print(json.dumps({{p.relative_to(root).as_posix():dict(data=base64.b64encode(p.read_bytes()).decode(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}}))
''', timeout=180)
        dest = ROOT/'docs/paper_results/rb2cal'/args.version
        assert dest.parent.is_dir() and not dest.exists()
        dest.mkdir()
        for name, record in json.loads(text).items():
            path = dest/name
            path.parent.mkdir(exist_ok=True)
            raw = base64.b64decode(record['data'])
            assert hashlib.sha256(raw).hexdigest() == record['sha256']
            with path.open('xb') as f:
                f.write(raw)
        print(dest)


if __name__ == '__main__':
    main()
