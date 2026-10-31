"""Owned receipts only, no peer process modifications."""
import json
import os
from pathlib import Path
root=Path('/root/coopaudit20261031')
out=dict(load=os.getloadavg())
for name in ('canary_PID.json','canary-repair_PID.json','census_PID.json','main_PID.json',
             'CPU_TESTS.json','WAIT.json','STOP','CANARY_VERIFIED.json',
             'SELECTED_DESIGN.json','CENSUS_COMPLETE.json','MAIN_COMPLETE.json'):
    p=root/name
    if p.is_file():
        out[name]=json.loads(p.read_text())
        if name.endswith('_PID.json'):
            try:
                proc=Path('/proc')/str(out[name]['pid'])/'stat'
                fields=proc.read_text().split()
                out[name]['live']=fields[21]==out[name]['proc_start_ticks'] and fields[2]!='Z'
            except FileNotFoundError:
                out[name]['live']=False
ledger=root/'accounting.jsonl'
if ledger.exists():
    out['ledger']=[json.loads(s) for s in ledger.read_text().splitlines()]
for p in sorted((root/'jobs').glob('*/result.json')):
    r=json.loads(p.read_text())
    out[str(p.relative_to(root))]=dict(ok=r['ok'],actual_sim_s=r['actual_sim_s'],
                                     group=r['frozen']['group'],hashes=r.get('trajectory_sha256'))
for p in sorted((root/'jobs').glob('*/failure.json')):
    out[str(p.relative_to(root))]=json.loads(p.read_text())
for p in sorted(root.glob('*.log')):
    out[p.name]=p.read_text()[-12000:]
print(json.dumps(out,indent=1))
