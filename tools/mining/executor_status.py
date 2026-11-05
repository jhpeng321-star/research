"""Named own receipts only; no process discovery or control."""
import json
from pathlib import Path
import subprocess

root = Path('/root/mining20261106')
out = {}
for sub, names in (('executor_b903c6c2', ('COLLECTION_DISPATCH.json', 'COLLECTION_OWNER.json',
                                       'COLLECTION_DONE.json', 'QUALIFICATION.json')),
                   ('executor_b903c6c2_v2', ('COLLECTION_DISPATCH.json', 'COLLECTION_OWNER.json',
                                       'COLLECTION_DONE.json', 'QUALIFICATION.json')),
                   ('wm_worker01', ('MINING_WM_FAULT.json', 'MINING_WM_EXIT.json'))):
    p = root / sub
    out[sub] = {name: json.loads((p / name).read_bytes()) for name in names if (p / name).is_file()}
    if sub.startswith('executor'):
        out[sub]['log'] = (p / 'COLLECTION.log').read_text()[-12000:]
        out[sub]['receipts'] = {str(x.relative_to(p)): json.loads(x.read_bytes())
                               for x in p.glob('*train_*/COLLECTION_RECEIPT.json')}
out['charges'] = (root / 'CHARGES.jsonl').read_text()
out['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                     '--format=csv,noheader'], text=True)
print(json.dumps(out, sort_keys=True))
