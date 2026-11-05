"""Whitelisted saved observable context schema only, no replay or inference."""
from planfix_ops import remote
CODE=r'''
import os,subprocess
code="""
import json,hashlib,torch
from pathlib import Path
w=Path('/home/ubuntu/coopmwm20261105_clean1/plan_fix')
rows=json.loads((w/'BANK_INVENTORY.json').read_text())['rows']
row=next(x for x in rows if x['world_seed']==986026)
p=Path(row['snapshot_path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==row['snapshot_sha256']
snap=torch.load(p,map_location='cpu',weights_only=False)
print('ROW',{k:row[k] for k in ('id','snapshot_path','snapshot_sha256','world_seed','t0')},flush=True)
print('SNAP_KEYS',list(snap),flush=True)
sample=snap['samples'][str(snap['t0'])]
print('SAMPLE_KEYS',list(sample),flush=True)
print('CONTEXT_KEYS',list(sample['context']),flush=True)
frozen=w/'bank_verification_v5'/row['id']/'FROZEN_SNAPSHOT.pt'
print('FROZEN_EXISTS',str(frozen),frozen.exists(),flush=True)
from tools.coopmwm.bank_v2v import trace_frames
frames=trace_frames(row['trace']);frame=frames[int(snap['t0'])]
print('FRAME_KEYS',list(frame),flush=True)
print('OBS_ROW_KEYS',list(frame['vehicles'][0]),flush=True)
for key in ('record','meta','obs','observation','state','rows','commands','policy_state','controller'):
 if key in snap:
  d=snap[key];print('CONTEXT_SCHEMA',key,list(d) if isinstance(d,dict) else type(d).__name__,flush=True)
"""
env=dict(os.environ,PYTHONPATH='/home/ubuntu/coopmwm20261105_clean1_src_7cc6edaa/pytools:/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',
 LD_PRELOAD='/home/ubuntu/cpus_env/libmkl_intel_shim.so',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
subprocess.run(['nice','-n','19','/home/ubuntu/miniconda3/bin/python','-B','-c',code],env=env,check=True)
'''
if __name__=='__main__':remote('cpuV',CODE,timeout=25)
