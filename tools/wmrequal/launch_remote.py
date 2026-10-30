"""Launch one finite, owned, sequential WM-scoring chain on gpuA."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

p=Path('/root/autodl-tmp/c4/wmrequal20261030')
rt=Path('/root/autodl-tmp/c4/restart20261026/runtime_longint')
assert p.is_dir() and (p/'smoke/DONE.json').is_file()
assert not (p/'LAUNCH.json').exists()
env=dict(os.environ,PYTHONPATH=f'{p}:{p}/tools/wmrequal:{p}/src:{rt}/src:{rt}',
         OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
         PYTHONDONTWRITEBYTECODE='1',WMS_SPEED_PIVOT_DIAG_OFF='1',
         WMS_SPEED_BATCH_REPLAY='1',WMS_SPEED_FAST_CAV_DRIVE='1',
         WMS_SPEED_FAST_LONG_HISTORY='1',WMS_LOOPLONG_SPEED='1',
         C4_SUMO_LEADER_RULE='1',C4_EXECUTION_PARITY='2',
         PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
chain='''import json,os,pathlib,subprocess,sys,time
p=pathlib.Path('/root/autodl-tmp/c4/wmrequal20261030')
for panel in ('validation','calibration','panel120'):
 argv=[sys.executable,'-B',str(p/'tools/wmrequal/score.py'),'--panel',panel]
 with (p/(panel+'.log')).open('x') as log:
  child=subprocess.Popen(argv,cwd=p,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  receipt=dict(pid=child.pid,start_ticks=pathlib.Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19],argv=argv,time=time.time())
  with (p/'OWNED_CHILDREN.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\\n')
  code=child.wait()
 if code:
  (p/'ERROR.json').write_text(json.dumps(dict(panel=panel,code=code,pid=child.pid)))
  raise SystemExit(code)
(p/'DONE.json').write_text(json.dumps(dict(pid=os.getpid(),time=time.time(),panels=['validation','calibration','panel120'])))
'''
with (p/'chain.log').open('x') as log:
    child=subprocess.Popen([sys.executable,'-B','-c',chain],cwd=p,env=env,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19],
             time=time.time(),gpu_fraction=.85,panels=['validation','calibration','panel120'])
(p/'LAUNCH.json').write_text(json.dumps(receipt,indent=1))
print(json.dumps(receipt))
