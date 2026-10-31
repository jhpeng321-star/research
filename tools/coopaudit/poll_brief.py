"""Bounded foreground read-only observation; never launches/restarts work."""
import json
import os
from pathlib import Path
import time

root=Path('/root/coopaudit20261031')
storage=Path('/root/autodl-tmp/coopaudit20261101_v2')
for _ in range(6):
    if (root/'V2_MAIN_COMPLETE.json').exists() or (root/'STOP').exists():
        break
    time.sleep(10)
jobs=list(storage.glob('v2_s*'))
print(json.dumps(dict(unix=time.time(),load=os.getloadavg(),
                     complete_worlds=sum((p/'result.json').exists() for p in jobs),
                     complete_branches=sum(len(list(p.glob('branch_*.json'))) for p in jobs),
                     started_branches=sum(len(list(p.glob('START_*.json'))) for p in jobs),
                     complete=(root/'V2_MAIN_COMPLETE.json').exists(),stop=(root/'STOP').exists(),
                     charged_sim_s=sum(json.loads(s)['sim_s'] for s in (root/'accounting.jsonl').read_text().splitlines()))))
