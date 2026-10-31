"""Read only own v2 receipts and branch completion; no polling/dispatch side effects."""
import json
import os
from pathlib import Path
import shutil

root=Path('/root/coopaudit20261031')
selection=json.loads((root/'CORRECTED_DESIGN.json').read_text())
storage=Path(selection['storage_root'])
out=dict(load=os.getloadavg(),free_bytes=shutil.disk_usage(storage).free if storage.exists() else None)
for name in ('CPU_TESTS.json','v2-canary_PID.json','v2-main_PID.json','V2_CANARY_COMPLETE.json',
             'V2_MAIN_COMPLETE.json','V2_CANARY_VERIFIED.json','STOP'):
    if (root/name).exists():
        out[name]=json.loads((root/name).read_text())
        if name.endswith('_PID.json'):
            p=Path('/proc')/str(out[name]['pid'])/'stat'
            out[name]['live']=p.exists() and p.read_text().split()[21]==out[name]['proc_start_ticks'] and p.read_text().split()[2]!='Z'
for p in sorted(storage.glob('*/START.json')) if storage.exists() else []:
    d=p.parent
    item=dict(branches_complete=len(list(d.glob('branch_*.json'))),started=len(list(d.glob('START_*.json'))),
              complete=(d/'result.json').exists())
    if (d/'failure.json').exists():
        item['failure']=json.loads((d/'failure.json').read_text())
    out[d.name]=item
out['charged_sim_s']=sum(json.loads(s)['sim_s'] for s in (root/'accounting.jsonl').read_text().splitlines())
for p in root.glob('v2-*.log'):
    out[p.name]=p.read_text()[-6000:]
print(json.dumps(out,indent=1))
