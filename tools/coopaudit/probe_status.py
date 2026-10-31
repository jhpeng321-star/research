"""Read registered TRAIN-probe progress only; no simulator state or peer signals."""
import json
from pathlib import Path
root=Path('/home/ubuntu/sumocf20261031')
out={}
for name in ('STATUS.json','receipts/CANARY_V2.json','receipts/PROBE_PROGRESS.json','receipts/SNR_PROBE.json','STOP'):
    p=root/name
    if p.is_file():
        out[name]=json.loads(p.read_text())
print(json.dumps(out,indent=1))
