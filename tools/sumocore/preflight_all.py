"""All-state, both-variant dry audit after float32 endpoint repair; no SUMO."""
from pathlib import Path
import json
import sys
import torch
R=Path('/home/ubuntu/e3final/runtime');sys.path[:0]=[str(R/'src'),str(R)]
from supervisor import jobs,ROOT
from prepare_bundle import prepare,sha


def main():
    torch.set_num_threads(1)
    out=ROOT/'preflight_all';out.mkdir(exist_ok=False)
    rows=[]
    for world in sorted({j['world'] for j in jobs()}):
        receipts=[]
        for variant in ('default','inferred'):
            job=next(j for j in jobs() if j['world']==world and j['variant']==variant)
            d=out/f'{world}_{variant}';d.mkdir()
            b,c=prepare(job,d);receipts.append(b['receipt'])
        assert receipts[0]['future_request_sha256']==receipts[1]['future_request_sha256']
        rows.append(dict(world=world,present=receipts[0]['initial_count'],common_demand=True))
    receipt=dict(status='PASS',sim_s=0,worlds=rows,prepare_sha256=sha(Path(__file__).parent/'prepare_bundle.py'))
    with (out/'receipt.json').open('x') as f:json.dump(receipt,f,indent=1)
    print(json.dumps(receipt))


if __name__=='__main__':main()
