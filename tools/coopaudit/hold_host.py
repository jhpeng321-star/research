"""Pause only the receipt-identified audit worker at an idle world boundary."""
import json
import os
from pathlib import Path
import signal
import time

root = Path('/root/coopaudit20261031')
assert (root/'YIELD').exists() and (root/'ARTIFACT_HOLD.json').exists()
receipt = json.loads((root/'main_PID.json').read_text())
proc = Path('/proc')/str(receipt['pid'])
fields = (proc/'stat').read_text().split()
assert fields[21] == receipt['proc_start_ticks']
command = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode()
assert command.strip() == '/root/miniconda3/bin/python -B -m tools.coopaudit.runner main'
started = sorted(p.parent.name for p in (root/'jobs').glob('main_*/START.json'))
completed = sorted(p.parent.name for p in (root/'jobs').glob('main_*/result.json'))
assert started == completed, 'active world: YIELD blocks next dispatch; do not interrupt'
os.kill(receipt['pid'], signal.SIGSTOP)
time.sleep(.1)
assert (proc/'stat').read_text().split()[2] == 'T'
ledger = [json.loads(line) for line in (root/'accounting.jsonl').read_text().splitlines()]
actual = sum(json.loads(p.read_text())['actual_sim_s']
             for p in (root/'jobs').glob('*/result.json'))
actual += sum(json.loads(p.read_text())['actual_sim_s']
              for p in (root/'jobs').glob('*/failure.json'))
result = dict(active=True, reason='OWNER 17:20 artifact-check HOLD', unix=time.time(),
              worker=receipt, worker_state='T', completed_main=completed,
              actual_sim_s=actual, charged_sim_s=sum(r['sim_s'] for r in ledger),
              local_continuation_stopped=39620, artifact_release_required=True)
(root/'ARTIFACT_HOLD.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
