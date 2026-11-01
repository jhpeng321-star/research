"""Audit native-pilot initial lane validity using causal starts only."""
import json
from pathlib import Path

from tools.rbint.remote import ssh


def main():
    root=Path(__file__).resolve().parents[2]
    out=root/'docs/rb2lane/INITIAL_LANE_AUDIT.json'
    if out.exists():raise ValueError('audit exists')
    result=json.loads(ssh('gpuA','''import os,json,pathlib,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(1)
root=pathlib.Path('/root/wmcalib');expected=json.loads((root/'expected_hashes.json').read_text());rows=[]
for world in range(969420,969432):
 p=root/'data/jobs'/f's{world}_ctrl0/start.pt'
 digest=hashlib.sha256(p.read_bytes()).hexdigest()
 assert digest==expected[f'jobs/s{world}_ctrl0/start.pt']
 start=torch.load(p,map_location='cpu',weights_only=False);assert start['seed']==world and start['t0']==300
 state=start['state'];present=state['present'].bool();lane=state['lane']
 invalid=present&((lane<0)|(lane>2))
 rows.append(dict(world=world,present=int(present.sum()),invalid_lanes=int(invalid.sum()),
  junction=int((present&state.get('junction',torch.zeros_like(present))).sum()),start_sha256=digest))
print(json.dumps(dict(status='PASS' if all(r['invalid_lanes']==0 for r in rows) else 'INVALID_INITIAL_LANES',
 source='causal t0 only; no forecast/outcome reads',rows=rows)))
'''))
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
