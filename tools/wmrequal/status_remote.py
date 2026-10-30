"""Read only this task's receipts and logs; no peer process inspection/signals."""
from pathlib import Path
import json
import subprocess
import time
p=Path('/root/autodl-tmp/c4/wmrequal20261030')
print('NOW',time.time())
for name in ('LAUNCH.json','OWNED_CHILDREN.jsonl','ERROR.json','DONE.json',
             'FAST_LAUNCH.json','FAST_CHILDREN.jsonl','FAST_ERROR.json','FAST_DONE.json'):
    q=p/name
    if q.is_file():print(name,q.read_text())
if (p/'fast_chain.log').is_file():
    print('FAST_CHAIN', (p/'fast_chain.log').read_text())
for panel in ('validation','calibration','panel120'):
    folder=p/'fast'/panel if (p/'FAST_LAUNCH.json').is_file() else p/panel
    print('PANEL',panel,'batches',len(list(folder.glob('batch_*.json'))),
          'done',(folder/'DONE.json').is_file())
    q=p/(('fast_' if (p/'FAST_LAUNCH.json').is_file() else '')+panel+'.log')
    if q.is_file():
        with q.open('rb') as stream:
            stream.seek(max(0,q.stat().st_size-4000))
            print(stream.read().decode(errors='replace'))
print(subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                      '--format=csv,noheader'],capture_output=True,text=True).stdout)
