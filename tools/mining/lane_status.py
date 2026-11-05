import json
from pathlib import Path
import subprocess
root = Path('/root/mining20261106/wm_lane02')
out = {}
for name in ('DISPATCH.json', 'OWNER.json', 'MODEL_READY.json', 'LANE_NOOP_QUALIFICATION.json', 'FAULT.json', 'DONE.json'):
    p = root / name
    if p.is_file():
        out[name] = json.loads(p.read_bytes())
out['log'] = (root / 'WORKER.log').read_text()[-10000:]
out['results'] = [p.name for p in root.glob('*.result.json')]
source = root.parent / 'executor_b903c6c2_v2'
out['collection'] = json.loads((source / 'COLLECTION_DONE.json').read_bytes()) if (source / 'COLLECTION_DONE.json').exists() else None
out['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader'], text=True)
print(json.dumps(out, sort_keys=True))
