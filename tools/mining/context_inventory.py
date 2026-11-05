"""Named parity TRAIN snapshot schema only, excluding population truth."""
import json
from pathlib import Path
import torch

pre=json.loads(Path('/home/ubuntu/cpuv_parity_20261105/PREFLIGHT.json').read_bytes())
out=[]
for case in pre['inputs']:
 p=Path(case['source'])
 snap=torch.load(p/'SNAPSHOT.pt',map_location='cpu',weights_only=False)
 spec=json.loads((p/'cand1.spec.json').read_bytes())
 assert spec['episode']['role']=='train' and 986000<=spec['episode']['world']['seed']<=986071
 sample=snap['samples'][str(snap['t0'])]
 context=sample.get('context') or {}
 out.append(dict(case=case,keys=list(snap),episode=snap['episode'],t0=snap['t0'],
  sample_keys=list(sample),context_keys=list(context),context=context,
  rng_keys=list(snap['rng']),sumo=snap['sumo'],
  files=[x.name for x in p.iterdir() if x.is_file()]))
print(json.dumps(out,default=str))
