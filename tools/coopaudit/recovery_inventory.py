"""Read-only, zero-SUMO inventory of the preserved audit and delivered policy."""
import hashlib
import json
import os
from pathlib import Path

root = Path('/root/coopaudit20261031')
rows = []
for p in sorted((root/'jobs').glob('census_*/FROZEN.json')):
    d = json.loads(p.read_text())
    rows.append(dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), frozen=d))
checkpoint = Path('/root/sumocf20261031/checkpoints/policy_u15.pt')
print(json.dumps(dict(load=os.getloadavg(), free_bytes=__import__('shutil').disk_usage(root).free,
                     census=rows, checkpoint=str(checkpoint),
                     checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()), indent=1))
