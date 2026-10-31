"""Explicit transport for the owner-authorized fallback; preserves prior root."""
from remote import ssh

if __name__ == '__main__':
    print(ssh('''from pathlib import Path
import json
r=Path('/root/c1floor20261031')
print('ledger', (r/'accounting.jsonl').read_text())
print('blocked', (r/'BLOCKED.json').read_text())
print('coop_canary',Path('/root/coopaudit20261031/CANARY_VERIFIED.json').read_text())
for name in ('/root/autodl-tmp/c4/speedhead20261030/e3final/runtime/traffic_shepherd/c4/branching.py','/root/wmdata2/tools/ctrlarm/zero.py'):
 p=Path(name); text=p.read_text(); print(name,text[text.index('def load_state'):text.index('def run_branch')] if 'def load_state' in text else text)
'''))
