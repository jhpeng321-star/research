"""Read-only compact recovery of existing mining results and exact charges."""
import hashlib
import json
from pathlib import Path
import time

root = Path('/root/mining20261106')
ledger = root/'CHARGES.jsonl'
rows = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
out = dict(unix=time.time(), ledger_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest(),
           charged_sim_s=sum(x['sim_s'] for x in rows), ledger_rows=len(rows), files={})
verify = root/'op2_verify01'
for pattern in ('RUN_*.jsonl', 'MINED_MOVES*.jsonl', 'EXAMPLES*.jsonl', 'G5_SOLO_PATCH.json', 'summ.py', 'smoke2.py', 'order.py', 'ctx3_*.txt'):
    for p in verify.glob(pattern):
        out['files'][str(p)] = dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), size=p.stat().st_size,
                                    text=p.read_text() if p.stat().st_size < 180000 else 'large')
for bank in ('bank', 'bank_v2'):
    p = verify/bank
    out[bank] = [str(x.relative_to(p)) for x in p.glob('*/*') if x.is_dir()]
out['ranked'] = [p.name for p in (root/'wm_lane04').glob('*.rank.json')]
print(json.dumps(out, sort_keys=True))
