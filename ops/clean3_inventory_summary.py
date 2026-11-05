"""Summarize the completed read-only inventory without re-reading bank tapes."""
import json
from pathlib import Path
repo = Path(__file__).resolve().parents[1]
receipt = json.loads((repo/'CLEAN3_BANK_INVENTORY_RESULT.json').read_text())
assert receipt['result']['returncode'] == 0
value = json.loads(receipt['log'])
for row in value['rows']:
    print(json.dumps({k: row.get(k) for k in ('arm', 'example_id', 'bound', 'reason',
        'macro', 'actors', 'window', 'policy')}, sort_keys=True))
    if row.get('macro'):
        print(json.dumps([v for v in row.get('observable_t0', [])
            if v['vid'] in row['macro']['members']], sort_keys=True))
