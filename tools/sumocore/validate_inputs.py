"""Read-only validation and world-guard/exposure receipt; zero SUMO."""
from pathlib import Path
import json
import sys
import numpy as np
import torch

R = Path('/home/ubuntu/e3final/runtime')
sys.path[:0] = [str(R/'src'),str(R)]
from inference import infer

WORLDS = tuple(range(969420,969440))


def main():
    from traffic_shepherd.seeds import require_development_case
    from traffic_shepherd.c4.seeds import check_case
    torch.set_num_threads(1)
    root = Path('/home/ubuntu/wmdata20261031')
    design = json.loads((root/'design.json').read_text())
    worlds = {w['seed']:w for w in design['worlds']}
    rows=[]
    for seed in WORLDS:
        w=worlds[seed]
        require_development_case(seed,w['demand'],'exposed');check_case(seed,w['demand'])
        p=root/'jobs'/f's{seed}_native'
        start=torch.load(p/'start.pt',map_location='cpu',weights_only=False)
        with np.load(p/'start_history.npz',allow_pickle=False) as z: raw=dict(z)
        active=start['state']['present'][0]
        info=dict(seed=seed,guard='PASS',exposure='existing WMDATA TRAIN FIT; no new worlds',
                  present=int(active.sum()),junction=int((start['state']['junction'][0]&active).sum()),
                  history_ticks=len(raw['kin']),queue=start['queue'])
        if seed==WORLDS[0]:
            inferred,receipt=infer(start,raw,'/home/ubuntu/e3final/inputs/wm_s_best.pt')
            info['inference']={k:v for k,v in receipt.items() if k!='per_vehicle_observed_samples'}
            info['sample_parameters']=list(inferred.items())[:2]
        print(json.dumps(info),flush=True)
        rows.append(info)
    out=Path('/home/ubuntu/sumocore20261031')/'input_audit.json'
    with out.open('x') as f:json.dump(rows,f,indent=1)


if __name__=='__main__':main()
