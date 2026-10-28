"""Read execution metadata from the two hash-bound frozen controller checkpoints."""
import sys
from remote import ssh

ssh(sys.argv[1], '''
import json,sys
from pathlib import Path
p=Path.home()/'entrant/runtime';sys.path[:0]=[str(p),str(p/'src')]
import torch
from tools.entrant.run import specs
for arm,s in specs().items():
    d=torch.load(s['checkpoint'],weights_only=True,map_location='cpu')
    keys=('format','residual_on_native','lane_control','wm_controller_kwargs','counterfactual','lane_rules','accel_bounds')
    print(arm,{k:d[k] for k in keys if k in d})
    print(arm,'has_lane_head',any(k.startswith('lane_head.') for k in d['policy']))
    print(arm,'lane_head_nonzero',{k:int(v.count_nonzero()) for k,v in d['policy'].items() if k.startswith('lane_head.2.')})
''')
