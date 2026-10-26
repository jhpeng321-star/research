"""Compare exact tensors and training state against pinned c4/restart parent."""
import base64
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent))
from remote import ROOT, REMOTE, ssh

parent = subprocess.check_output(['git', 'merge-base', 'HEAD', 'c4/restart'], cwd=ROOT, text=True).strip()
overrides = {}
for name in ('dreamer', 'bridge', 'policy', 'policy14_cli', 'nativeinit'):
    path = f'src/wms/control/{name}.py'
    overrides[path] = base64.b64encode(subprocess.check_output(['git', 'show', f'{parent}:{path}'], cwd=ROOT)).decode()
code = f'''import os,sys,subprocess,pathlib,shutil,json,base64
root=pathlib.Path('{REMOTE}'); parent=root/'parent'
parent.mkdir(exist_ok=True)
shutil.copytree(root/'src',parent/'src',dirs_exist_ok=True)
for path,raw in json.loads({json.dumps(overrides)!r}).items():
 (parent/path).write_bytes(base64.b64decode(raw))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',C4_ALLOWED_ROOTS='/root')
for version,source in [('parent',parent/'src'),('new',root/'src')]:
 env.update(PAIRCRIT_VERSION=version,PYTHONPATH=str(source)+':'+str(root))
 subprocess.run([sys.executable,'-B',str(root/'tools/paircrit/bitprobe.py')],env=env,check=True,cwd=root)
import torch,numpy as np
a,b=[torch.load(root/('bitid_'+v+'.pt'),weights_only=False) for v in ('parent','new')]
count=[0]
def check(a,b,path=''):
 if torch.is_tensor(a):
  assert torch.equal(a,b),path
  count[0]+=1
 elif isinstance(a,np.ndarray): assert np.array_equal(a,b),path
 elif isinstance(a,dict):
  assert a.keys()==b.keys(),path
  for k in a: check(a[k],b[k],path+'/'+str(k))
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):check(x,y,path+'/'+str(i))
 else: assert a==b,(path,a,b)
check(a,b)
result={{'parent':'{parent}','exact_tensors':count[0],'updates_per_case':2,'cases':list(a),'rngs':'torch,numpy,python','metrics':'exact, excluding wall_time_s'}}
(root/'BIT_IDENTITY.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
'''
print(ssh(code))
