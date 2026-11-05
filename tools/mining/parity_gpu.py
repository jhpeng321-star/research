"""Registered P3 then P4, one attempt each; exact SUMO bytes, fail closed."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

r=Path('/root/mining20261106')
h=Path('/home/ubuntu')
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_bytes())
def write(p,v):
 with p.open('x') as f:
  f.write(json.dumps(v,indent=2,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def payload(p):
 marker=read(p/'RESULT.json');b=gzip.decompress((p/marker['file']).read_bytes())
 assert sha(b)==marker['sha256'];return b

assert read(r/'CLONE_QUALIFICATION.json')['returncode']==0
budget=read(r/'BUDGET_ADMISSION.json')
assert budget['bound_sim_s']==5500000 and budget['mining_cap_sim_s']==500000
assert time.time()-budget['unix']<3600
assert (r/'CHARGES.jsonl').read_bytes()==b''
write(r/'PARITY_EXECUTION.claim',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],unix=time.time()))
os.nice(19);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
source=str(h/'coopmwm20261104_launchcoop_src7')
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 PYTHONDONTWRITEBYTECODE='1',LD_PRELOAD=str(h/'cpus_env/libmkl_intel_shim.so'),
 SUMO_HOME=str(h/'miniconda3/lib/python3.10/site-packages/sumo'))
env['PATH']=env['SUMO_HOME']+'/bin:'+str(h/'miniconda3/bin')+':'+env['PATH']
env['PYTHONPATH']=':'.join(source+'/'+p for p in ('pytools','runtime_src','WM_PACKAGE'))
pre=read(h/'cpuv_parity_20261105/PREFLIGHT.json')
results=[]
for case in pre['inputs']:
 name=case['name'];inp=Path(case['source']);cost=case['sim_s']
 assert sha((inp/'SNAPSHOT.pt').read_bytes())==case['snapshot_sha256']
 assert sha((inp/'cand1.spec.json').read_bytes())==case['spec_sha256']
 spec=read(inp/'cand1.spec.json')
 assert spec['episode']['role']=='train' and 986000<=spec['episode']['world']['seed']<=986071
 assert sha(payload(inp/'cand1'))==case['reference_payload_sha256']
 if name=='P4':assert results[0]['pass_']
 work=r/('parity_'+name);work.mkdir(exist_ok=False)
 with (r/'CHARGES.jsonl').open('a') as ledger:
  ledger.write(json.dumps(dict(id='S-MINING-PARITY-'+name,phase='mining',sim_s=cost,
                              purpose='registered_bit_parity',unix=time.time()))+'\n')
  ledger.flush();os.fsync(ledger.fileno())
 cmd=[str(h/'miniconda3/bin/python'),'-B','-u','-m','tools.coopmwm.sumo_verify','run-branch',
      '--snapshot',str(inp/'SNAPSHOT.pt'),'--spec',str(inp/'cand1.spec.json'),
      '--work',str(work),'--out',str(r/(name+'.result.json'))]
 started=time.time()
 with (r/(name+'.log')).open('x') as log:
  p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  write(r/(name+'_DISPATCH.json'),dict(pid=p.pid,argv=cmd,unix=started))
  rc=p.wait(timeout=600)
 if rc!=0 or not (r/(name+'.result.json')).is_file():
  write(r/'PARITY_FAULT.json',dict(case=name,returncode=rc,physics_attempted=True,decision='cpuV fallback; GPU ranking remains available'))
  raise RuntimeError('parity branch execution failed; no replay')
 receipt=read(r/(name+'.result.json'))
 actual=payload(work/inp.name/'cand1');reference=payload(inp/'cand1')
 a=json.loads(actual);b=json.loads(reference)
 checks=dict(payload_bytes=actual==reference,payload_sha256=sha(actual)==case['reference_payload_sha256'],
             result_sha256=receipt.get('result_sha256')==case['reference_result_sha256'],
             branch_receipt=receipt==read(inp/'cand1.result.json'),
             steps=a['steps']==b['steps'],prefixes=a['prefixes']==b['prefixes'],
             totals=a['totals']==b['totals'],final=a['final']==b['final'],full_result=a==b)
 record=dict(name=name,pass_=all(checks.values()),checks=checks,sim_s=cost,
             actual_sim_s=receipt.get('actual_sim_s'),wall_s=time.time()-started,
             payload_sha256=sha(actual),result_sha256=receipt.get('result_sha256'),
             totals=a['totals'],unix=time.time())
 write(r/(name+'_COMPARE.json'),record);results.append(record)
 print(json.dumps(record),flush=True)
 if not record['pass_']:
  write(r/'PARITY_FAIL.json',dict(cases=results,decision='Stop gpuA SUMO; mining physics must use resource-capped cpuV.',unix=time.time()))
  raise RuntimeError('bit parity FAILED; no next case or retry')
write(r/'PARITY_PASS.json',dict(cases=results,pass_=True,charged_sim_s=sum(x['sim_s'] for x in results),
 actual_sim_s=sum(x['actual_sim_s'] for x in results),unix=time.time()))
