"""Summarize preserved transport receipts without dumping their large payloads."""
import json
import sys
from pathlib import Path

for name in sys.argv[1:]:
    outer = json.loads(Path(name).read_text())
    data = json.loads(outer['stdout'])
    print(name)
    if 'charged_sim_s' in data:
        print(json.dumps({k: v for k, v in data.items() if k != 'files'}, indent=2))
        for key, value in data['files'].items():
            print(key, value['size'], value['sha256'])
            if 'EXAMPLES' in key:
                from collections import Counter
                records = [json.loads(x) for x in value['text'].splitlines() if x.strip()]
                print(Counter((r['family'], r['status']) for r in records))
                print('positive lower:', [r for r in records if (r.get('synergy_lower95') or 0) > 0])
            elif 'MINED_MOVES' in key:
                print(value['text'])
        continue
    print(json.dumps({k: data.get(k) for k in ('unix', 'gpu', 'free_bytes', 'owners')}, indent=2))
    for root, item in data.get('roots', {}).items():
        print(root, 'files:', ', '.join(item['files']))
        for key, value in item.get('receipts', {}).items():
            print(key, json.dumps(value)[:8000])
        for key, value in item.get('logs', {}).items():
            print(key, value)
