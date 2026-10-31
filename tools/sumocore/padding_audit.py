"""Prove frozen MAPPO commands invariant to absent representation padding; no SUMO."""
from pathlib import Path
import json
import sys
import numpy as np
import torch
R=Path('/home/ubuntu/e3final/runtime');sys.path[:0]=[str(R/'src'),str(R)]
from wms.control.policy import load_policy_fn
from admission import ROOT


def main():
    torch.set_num_threads(1)
    worlds=sorted({j['world'] for j in json.loads((ROOT/'plan.json').read_text())['jobs']})
    rows=[]
    for world in worlds:
        p=Path('/home/ubuntu/wmdata20261031/jobs')/f's{world}_native/start.pt'
        s=torch.load(p,map_location='cpu',weights_only=False)['state']
        padded={k:torch.cat([v,torch.full((1,512-v.shape[1]),-1 if k=='lane' else 5 if k=='length_m' else 0,dtype=v.dtype)],dim=1) for k,v in s.items()}
        a=load_policy_fn('/home/ubuntu/mappospeed/production2/policy_latest.pt',device='cpu')(s)
        b=load_policy_fn('/home/ubuntu/mappospeed/production2/policy_latest.pt',device='cpu')(padded)[:,:192]
        assert np.array_equal(np.isnan(a),np.isnan(b))
        diff=float(np.nanmax(abs(a-b)))
        assert diff<=1e-6,(world,diff)
        rows.append(dict(world=world,max_command_difference=diff))
    receipt=dict(status='PASS',sim_s=0,old_slots=192,new_slots=512,worlds=rows)
    with (ROOT/'padding_audit.json').open('x') as f:json.dump(receipt,f,indent=1)
    print(json.dumps(receipt))


if __name__=='__main__':main()
