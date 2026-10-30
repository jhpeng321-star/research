"""Read named own-run timing receipts without querying any GPU."""
import json
from pathlib import Path
import subprocess
import time

from tools.candB.pilot_lean_deploy import ROOT

if __name__=='__main__':
    code=f'''
import json,time
from pathlib import Path
output=Path({ROOT!r})/'pilot_EXPLORATORY-LEAN'
p=output/'speedprof_execution.jsonl'
doc=dict(epoch=time.time(),profile=[])
if p.exists():
    doc['profile']=[json.loads(line) for line in p.read_text().splitlines()[-20:]]
doc['pools']=[dict(root=str(pool),pids=json.loads((pool/'PIDS.json').read_text()))
    for pool in (output/'future_workers').glob('*') if (pool/'PIDS.json').exists()]
print(json.dumps(doc))
'''
    result=subprocess.run(['ssh','gpuA','/root/miniconda3/bin/python','-'],input=code,
        text=True,capture_output=True,check=True,timeout=25)
    doc=json.loads(result.stdout)
    with Path(__file__).with_name(f'pilot_lean_profile_{int(time.time())}.json').open('x') as stream:
        json.dump(doc,stream,indent=1)
    print(json.dumps(doc))
