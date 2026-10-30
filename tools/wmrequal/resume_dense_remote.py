"""Resume the unchanged dense scorer, reusing every completed owned group."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

p=Path('/root/autodl-tmp/c4/wmrequal20261030')
assert (p/'fast/validation/batch_000.json').is_file()
assert json.loads((p/'COMPACT_PARITY.json').read_text())['pass_numeric'] is False
assert not (p/'DENSE_RESUME_LAUNCH.json').exists()
for name in ('FAST_CHILDREN.jsonl','COMPACT_PROBE_LAUNCH.json'):
    raw=(p/name).read_text()
    r=json.loads(raw.splitlines()[-1] if name.endswith('.jsonl') else raw)
    q=Path(f'/proc/{r["pid"]}/stat')
    assert not q.is_file() or q.read_text().rsplit(')',1)[1].split()[0]=='Z', 'old owned child still live'
env=dict(os.environ,PYTHONPATH=f'{p}:{p}/tools/wmrequal:{p}/src:/root/autodl-tmp/c4/restart20261026/runtime_longint',
 OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
 WMS_SPEED_PIVOT_DIAG_OFF='1',WMS_SPEED_BATCH_REPLAY='1',WMS_SPEED_FAST_CAV_DRIVE='1',
 WMS_SPEED_FAST_LONG_HISTORY='1',WMS_LOOPLONG_SPEED='1',C4_SUMO_LEADER_RULE='1',C4_EXECUTION_PARITY='2',
 PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
chain='''import json,os,pathlib,subprocess,sys,time
p=pathlib.Path('/root/autodl-tmp/c4/wmrequal20261030')
for panel in ('validation','calibration','panel120'):
 argv=[sys.executable,'-B',str(p/'tools/wmrequal/score_fast.py'),'--panel',panel]
 with (p/('resume_'+panel+'.log')).open('x') as log:
  child=subprocess.Popen(argv,cwd=p,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  r=dict(pid=child.pid,start_ticks=pathlib.Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19],argv=argv,time=time.time())
  with (p/'DENSE_RESUME_CHILDREN.jsonl').open('a') as f:f.write(json.dumps(r)+chr(10))
  try:code=child.wait(timeout=18000)
  except subprocess.TimeoutExpired:
   child.terminate();child.wait(timeout=60);code=124
 if code:
  (p/'DENSE_RESUME_ERROR.json').write_text(json.dumps(dict(panel=panel,code=code,pid=child.pid)))
  raise SystemExit(code)
(p/'DENSE_RESUME_DONE.json').write_text(json.dumps(dict(pid=os.getpid(),time=time.time())))
'''
compile(chain,'<owned-dense-resume>','exec')
with (p/'dense_resume_chain.log').open('x') as log:
    child=subprocess.Popen([sys.executable,'-B','-c',chain],env=env,cwd=p,stdin=subprocess.DEVNULL,
                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19],
       time=time.time(),gpu_fraction=.85,per_panel_timeout_s=18000,compact_slots=False)
(p/'DENSE_RESUME_LAUNCH.json').write_text(json.dumps(r,indent=1))
print(json.dumps(r))
