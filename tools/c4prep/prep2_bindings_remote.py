"""Compact explicit bindings; no runtime mutation or compute dispatch."""
import hashlib
import json
from pathlib import Path
import numpy as np

r = Path('/root/autodl-tmp/c4/restart20261026')
for c in range(4):
    p = r / f'arm/loop/returns/pairing_c{c:02}.a1.json'
    d = json.loads(p.read_text())
    print('pairing', c, {k: v for k, v in d.items() if k != 'pairs'},
          'worlds', sorted({x['seed'] for x in d['pairs']}))
    p = r / f'arm/loop/returns/returns_c{c:02}.a1.receipt.json'
    d = json.loads(p.read_text())
    print('returns_receipt', c, {k: v for k, v in d.items() if k != 'entries'})
    with np.load(r / f'arm/loop/returns/returns_c{c:02}.a1.npz', allow_pickle=False) as z:
        print('returns_headers', c, {k: str(z[k]) for k in z.files
              if k in ('policy_checkpoint', 'policy_checkpoint_path', 'accounting', 'gamma')})
p = r / 'arm/loop/c4prep/backlog_import_v2/IMPORT.json'
d = json.loads(p.read_text())
print('import_verified', {'receipt_sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
      'datasets': d['datasets'], 'c3_tapes': sum(len(x['tapes']) for x in d['c3_audits'])})
gate = r / 'arm/loop/checks/c03.a2/HORIZON_GATE.json'
print('gate_exists', gate.is_file())
if gate.is_file():
    print('gate', gate.read_text())
else:
    log = r / 'arm/loop/checks/c03.a2/HORIZON_GATE.log'
    print('gate_log_tail', log.read_text()[-1000:])
