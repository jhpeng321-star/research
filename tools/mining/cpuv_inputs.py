"""Read exact parity dependency paths and public budget receipts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

h = Path('/home/ubuntu')
out = {}
for rel in ('cpuv_parity_20261105/PREFLIGHT.json', 'cpuv_parity_20261105/PARITY_PASS.json',
            'coopmwm20261105_clean2/LAUNCHED.json','coopmwm20261105_clean2/CHAIN_STATUS.json',
            'coopmwm20261105_clean2/STOP.json','coopmwm20261105_clean2/FAULT.json',
            'coopmwm20261106_s_mining/MINING_REGISTRATION.json'):
    p = h/rel
    out[rel] = json.loads(p.read_bytes()) if p.is_file() else None
out['ledgers'] = {}
for rel in ('coopmwm20261105_clean1/CHARGES.jsonl','coopmwm20261105_clean2/CHARGES.jsonl',
            'coopmwm20261106_clean3/CHARGES.jsonl','coopmwm20261106_s_mining/CHARGES.jsonl'):
    p = h/rel
    if p.is_file():
        rows = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
        out['ledgers'][rel] = dict(rows=len(rows), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                   first=rows[:1], last=rows[-1:])
out['sizes'] = subprocess.check_output(['du','-sh',str(h/'miniconda3'),
    str(h/'coopmwm20261104_launchcoop_src7'),str(h/'e3final/runtime')], text=True)
out['conda_packages'] = [p.name for p in (h/'miniconda3/conda-meta').glob('python-*.json')]
import torch
out['inputs'] = []
pre = out['cpuv_parity_20261105/PREFLIGHT.json']
for case in pre['inputs']:
    p = Path(case['source'])
    snap = torch.load(p/'SNAPSHOT.pt', map_location='cpu', weights_only=False)
    spec = json.loads((p/'cand1.spec.json').read_bytes())
    out['inputs'].append(dict(pin=case, sumo=snap['sumo'], policy=snap['policy'],
                              episode=spec['episode'], spec=spec,
                              files=[(f.name,f.stat().st_size) for f in p.iterdir() if f.is_file()]))
print(json.dumps(out,sort_keys=True,default=str))
