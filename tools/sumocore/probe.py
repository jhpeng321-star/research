"""Targeted observable-only schema and installed-runtime probe (no simulation)."""
from remote import ssh

SOURCE = r'''
from pathlib import Path
import sys, importlib.metadata, json, numpy as np, torch
print('PYTHON',sys.executable)
for name in ('torch','eclipse-sumo','traci'):
 print(name,importlib.metadata.version(name))
base=Path('/home/ubuntu/wmdata20261031/jobs/s969420_native')
for name in ('start_history.npz','traj_base.npz','deploy/episode/episode.npz'):
 with np.load(base/name,allow_pickle=False) as z:
  print(name, [(k,z[k].shape,str(z[k].dtype)) for k in z.files])
s=torch.load(base/'start.pt',map_location='cpu',weights_only=False)
print('START',[(k, list(v) if isinstance(v,dict) else str(type(v))) for k,v in s.items()])
print('STATE', [(k,tuple(v.shape)) for k,v in s['state'].items()])
print('PAST',s['past_entries'][-3:]);print('QUEUE',s['observed_queue_history'][-3:])
print('CAV IDS', [v for v in s['vehicle_ids'][:8]])
ck=torch.load('/home/ubuntu/e3final/inputs/wm_s_best.pt',map_location='cpu',weights_only=False)
print('CKPT',list(ck))
for k,v in ck.items():
 if isinstance(v,dict): print('CKPTDICT',k,list(v)[:20])
print('SAFE_NETWORKS', [str(p) for p in Path('/home/ubuntu/e3final/runtime/scenarios').glob('**/*.net.xml')])
'''

if __name__ == '__main__':
    print(ssh(SOURCE, python='/home/ubuntu/miniconda3/bin/python').decode())
