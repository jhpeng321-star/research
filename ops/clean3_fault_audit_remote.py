"""Read the r10 failure evidence; never clear STOP or retry physics."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

root = Path('/home/ubuntu/coopmwm20261106_clean3_r4')
control = Path('/home/ubuntu/cpuv_setup_20261105/clean3_chain_ac9e4691')
out = dict(timestamp=datetime.now(timezone.utc).isoformat(), status='READ_ONLY_FAULT_AUDIT',
           root=str(root), control=str(control), files={})
for base, names in ((root, ('SMOKE_STARTED.json', 'SMOKE_FAILED.json', 'SMOKE_PASS.json',
        'SMOKE_FIX_RECEIPTS_PASS.json', 'SMOKE_LEVER_RECEIPTS_PASS.json', 'SMOKE_REPORTING_PASS.json',
        'STOP.json', 'FAULT.json', 'GATE_STARTED.json', 'GATE_PASS.json', 'LAUNCHED.json', 'CLEAN3_BUDGET.json')),
        (control, ('CHAIN_STATUS.json', 'CHAIN_STARTED.json', 'SMOKE_EXIT.json', 'smoke.log'))):
    for name in names:
        path = base/name
        if path.is_file():
            raw = path.read_bytes()
            out['files'][str(path)] = dict(sha256=hashlib.sha256(raw).hexdigest(),
                value=raw.decode(errors='replace').splitlines() if name.endswith('.log') else json.loads(raw))
        else:
            out['files'][str(path)] = None
rows = [json.loads(line) for line in (root/'CHARGES.jsonl').read_bytes().splitlines()]
out['charges'] = dict(rows=len(rows), sim_s=sum(row['sim_s'] for row in rows))
owner = out['files'][str(control/'CHAIN_STARTED.json')]['value']
path = Path('/proc')/str(owner['pid'])/'stat'
out['owner'] = dict(owner, proc_stat=path.read_text() if path.is_file() else None)
print(json.dumps(out, sort_keys=True))
