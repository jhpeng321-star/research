"""Read selected public runtime contracts without running a simulator."""
from remote import ssh

SOURCE = r'''
from pathlib import Path
import torch, json
ck=torch.load('/home/ubuntu/e3final/inputs/wm_s_best.pt',map_location='cpu',weights_only=False)
print('ENCODER_CONFIG', {k:v for k,v in ck['config'].items() if any(x in k for x in ('history','driver_param','rollout_mode'))})
print('ENCODER_KEYS',[k for k in ck['model'] if k.startswith('driver_encoder.')])
print('TRAJ_META')
import numpy as np
with np.load('/home/ubuntu/wmdata20261031/jobs/s969420_native/traj_base.npz',allow_pickle=False) as z:
 m=json.loads(str(z['meta']));print({k:v for k,v in m.items() if k != 'vehicle_ids'})
print('NETWORK_CONFIG')
print(Path('/home/ubuntu/e3final/runtime/configs/default.toml').exists())
'''
if __name__ == '__main__':
    print(ssh(SOURCE, python='/home/ubuntu/miniconda3/bin/python').decode())
