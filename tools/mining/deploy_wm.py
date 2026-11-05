"""Stage immutable GPU worker sources and dispatch exactly one model owner."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import zlib
sys.path.insert(0,str(Path(__file__).parent))
from remote import run
commit='f631ce90cfd2f23213f7c57a828c848e2db10ecd'
repo=Path('E:/agent-work/c4-variety-collect')
names=subprocess.check_output(['git','ls-tree','-r','--name-only',commit,'tools/coopmwm'],cwd=repo,text=True).splitlines()
files={}
for name in names:
 if not name.endswith('.py') or '/test_' in name or name.endswith('/conftest.py'):continue
 if any(x in name.lower() for x in ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vehicle_params','vtype')):continue
 files['pytools/'+name]=base64.b64encode(subprocess.check_output(['git','show',commit+':'+name],cwd=repo)).decode()
for name in ('__init__.py','wm_adapter.py','wm_contract.py','wm_prescreen.py','wm_coordinator.py'):
 files['pytools/tools/mining/'+name]=base64.b64encode((Path(__file__).parent/name).read_bytes()).decode()
payload=base64.b64encode(zlib.compress(json.dumps(files).encode(),9)).decode()
code=r'''
import base64,hashlib,json,os,subprocess,sys,time,zlib
from pathlib import Path
parent=Path('/root/mining20261106');root=parent/'wm_worker01'
old=Path('/root/coopmwm20261102_deadlineA2')
assert json.loads((parent/'PARITY_PASS.json').read_bytes())['pass_']
assert not root.exists(),'do not duplicate or restage a model worker'
root.mkdir()
files=json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))
for name,data in files.items():
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(data))
for name in ('INPUTS.json','FROZEN_COSTS.json','FROZEN_MOTION_SOURCE.py','pytools/tools/wmteach/runtime.py','pytools/tools/coopnative/policy.py'):
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((old/name).read_bytes())
for pkg in ('tools','tools/coopnative','tools/wmteach'):
 p=root/'pytools'/pkg/'__init__.py';p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text('__path__.append('+repr(str(old/'pytools'/pkg))+')\n'.replace('\\n','\n'))
sys.path.insert(0,str(root/'pytools'))
from tools.coopmwm import varcollect
from tools.coopmwm.fresh_worker import pending
deadline=time.time()+21600
broker=dict(amendment_sha256=hashlib.sha256(b'S-MINING-20261106 GPU ranking only').hexdigest(),gpu_deadline=deadline,
 workers=[dict(endpoint=dict(slot='gpuA-mining1'),owner=dict(pid=1,start_ticks='PENDING_FRESH_STARTUP'))])
binding=varcollect.fresh_binding(broker,'gpuA')
for rel in ('INPUTS.json','FROZEN_COSTS.json','FROZEN_MOTION_SOURCE.py','pytools/tools/wmteach/runtime.py','pytools/tools/coopnative/policy.py'):
 binding['hashes'][rel]=hashlib.sha256((root/rel).read_bytes()).hexdigest()
binding['qualification']=pending(binding['hashes'])
(root/'GPU_BINDING.json').write_text(json.dumps(binding,sort_keys=True))
manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.py')}
for name in ('GPU_BINDING.json','FROZEN_COSTS.json','INPUTS.json'):
 p=root/name;manifest[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
(root/'MINING_SOURCE_MANIFEST.json').write_text(json.dumps(manifest,sort_keys=True))
(root/'queries').mkdir();(root/'results').mkdir()
env=dict(os.environ,**binding['worker_env']['variables'],PYTHONPATH=str(root/'pytools'),PYTHONDONTWRITEBYTECODE='1')
cmd=['/root/miniconda3/bin/python','-B','-u','-m','tools.mining.wm_coordinator',str(root)]
with (root/'MINING_WM.log').open('x') as log:
 p=subprocess.Popen(cmd,env=env,cwd=str(root),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,start_ticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],
 argv=cmd,root=str(root),deadline_unix=deadline,source_commit=COMMIT,unix=time.time(),new_sumo_sim_s=0)
(root/'MINING_WM_DISPATCH.json').write_text(json.dumps(record,sort_keys=True))
print(json.dumps(record),flush=True)
'''
result=run('gpuA','PAYLOAD='+repr(payload)+'\nCOMMIT='+repr(commit)+'\n'+code,
           '/root/miniconda3/bin/python -B -',timeout=90)
receipt=Path('E:/Temp/opencode/MINING_WM_DISPATCH_0200.json')
assert receipt.parent.is_dir() and not receipt.exists()
receipt.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
raise SystemExit(result['returncode'])
