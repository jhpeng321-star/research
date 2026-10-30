"""Freeze the additive grouped adapter and launch the bounded scoring chain."""
import base64
from pathlib import Path
from remote import remote

raw=base64.b64encode((Path(__file__).parent/'score_fast.py').read_bytes()).decode()
remote(f'''import base64,json,os,pathlib,subprocess,sys,time,hashlib
p=pathlib.Path('/root/autodl-tmp/c4/wmrequal20261030')
if (p/'FAST_LAUNCH.json').exists():
 old=json.loads((p/'FAST_LAUNCH.json').read_text());stat=pathlib.Path(f'/proc/{{old["pid"]}}/stat')
 assert not stat.is_file() or stat.read_text().rsplit(')',1)[1].split()[0]=='Z'
 assert not (p/'FAST_CHILDREN.jsonl').exists() and 'SyntaxError' in (p/'fast_chain.log').read_text()
 (p/'FAST_LAUNCH.json').rename(p/'FAST_ATTEMPT1_LAUNCH.json')
 (p/'fast_chain.log').rename(p/'fast_attempt1_chain.log')
assert json.loads((p/'FEEDBACK_PARITY.json').read_text())['max_abs_difference']==0
(p/'tools/wmrequal/score_fast.py').write_bytes(base64.b64decode({raw!r}))
env=dict(os.environ,PYTHONPATH=f'{{p}}:{{p}}/tools/wmrequal:{{p}}/src:/root/autodl-tmp/c4/restart20261026/runtime_longint',
 OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
 WMS_SPEED_PIVOT_DIAG_OFF='1',WMS_SPEED_BATCH_REPLAY='1',WMS_SPEED_FAST_CAV_DRIVE='1',
 WMS_SPEED_FAST_LONG_HISTORY='1',WMS_LOOPLONG_SPEED='1',C4_SUMO_LEADER_RULE='1',C4_EXECUTION_PARITY='2',
 PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
chain="""import json,os,pathlib,subprocess,sys,time
p=pathlib.Path('/root/autodl-tmp/c4/wmrequal20261030')
for panel in ('validation','calibration','panel120'):
 argv=[sys.executable,'-B',str(p/'tools/wmrequal/score_fast.py'),'--panel',panel]
 with (p/('fast_'+panel+'.log')).open('x') as log:
  child=subprocess.Popen(argv,cwd=p,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  r=dict(pid=child.pid,start_ticks=pathlib.Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19],argv=argv,time=time.time())
  with (p/'FAST_CHILDREN.jsonl').open('a') as f:f.write(json.dumps(r)+chr(10))
  code=child.wait()
 if code:
  (p/'FAST_ERROR.json').write_text(json.dumps(dict(panel=panel,code=code,pid=child.pid)))
  raise SystemExit(code)
(p/'FAST_DONE.json').write_text(json.dumps(dict(pid=os.getpid(),time=time.time())))
"""
compile(chain,'<wmrequal-fast-chain>','exec')
with (p/'fast_chain.log').open('x') as log:
 child=subprocess.Popen([sys.executable,'-B','-c',chain],cwd=p,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=child.pid,start_ticks=pathlib.Path(f'/proc/{{child.pid}}/stat').read_text().rsplit(')',1)[1].split()[19],time=time.time(),gpu_fraction=.85,
 source_sha256={{name:hashlib.sha256((p/'tools/wmrequal'/name).read_bytes()).hexdigest() for name in ('score_fast.py','grouped_feedback.py')}})
(p/'FAST_LAUNCH.json').write_text(json.dumps(r,indent=1));print(json.dumps(r))
''')
