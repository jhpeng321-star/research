"""Read-only, narrowly scoped inspection of the registered observable tapes."""
import json
import subprocess

CODE = r'''
from pathlib import Path
import json
root=Path(ROOT)
print('ROOT',root,'exists',root.exists())
print('ENTRIES', [p.name for p in root.iterdir() if not any(x in str(p).lower() for x in ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','truth','vehicle_params','driver_params','vtype'))][:18])
w=json.loads((root/('world_969400.json' if 'wmdata20261031' in str(root) else 'world_969550.json')).read_text())
print('WORLD',[(k,v['episode']) for k,v in w.items()])
job=Path(w['native']['job'])
print('JOB_FILES',[p.name for p in job.iterdir() if not any(x in p.name.lower() for x in ('truth','params','vtype'))])
for name in ('traj_base.npz',):
 p=job/name
 if p.exists():
  if name.endswith('.npz'):
   import numpy as np
   with np.load(p,allow_pickle=False) as f:
    print('RAW',name,[(k,f[k].shape,str(f[k].dtype)) for k in ('time_s','time','s_m','v_mps','lane','mask','veh_kin','veh_mask') if k in f.files])
    print('RAW_NAMES',[k for k in f.files if not any(x in k.lower() for x in ('truth','params','vtype'))])
    for k in ('t','s','v','lane'):
     a=f[k]; print(k,a.min().item(),a.max().item(),a[:8].tolist())
  else: print('RAW_META',json.loads(p.read_text()))
eps=root/'episodes'
print('EPISODES',[p.name for p in eps.iterdir()][:8])
episode=eps / ('s969400_native' if 'wmdata20261031' in str(root) else 's969550_native')
if not episode.exists(): episode=next(eps.iterdir())
print('EPISODE',episode,episode.exists())
if episode.exists():
 print('FILES',[p.name for p in episode.iterdir() if not any(x in p.name.lower() for x in ('truth','params','vtype'))])
 for name in ('episode.json',):
  p=episode/name
  if p.exists():
   d=json.loads(p.read_text()); print(name,d)
 import numpy as np
 for name in ('veh_kin.npy','veh_mask.npy','time_s.npy'):
  a=np.load(episode/name,mmap_mode='r',allow_pickle=False)
  print(name,a.shape,str(a.dtype),'sample',a.reshape(-1)[:8].tolist())
 for name in ('trajectory.npz','trajectory_raw.npz','start_history.npz','recorded.npz','observations.npz'):
  p=episode/name
  if p.exists():
   import numpy as np
   with np.load(p,allow_pickle=False) as f:
    print(name,[(k,f[k].shape,str(f[k].dtype)) for k in f.files if not any(x in k.lower() for x in ('truth','params','vtype'))])
'''

if __name__ == '__main__':
    for host, root, python in (
        ('cpuQ','/home/ubuntu/wmdata20261031/views','/home/ubuntu/miniconda3/bin/python'),
        ('gpuR','/root/wmdata2/views','/root/miniconda3/bin/python')):
        p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,python+' -B -'],
            input='ROOT='+repr(root)+'\n'+CODE,text=True,capture_output=True,timeout=90)
        print(host,p.returncode,p.stdout,p.stderr)
