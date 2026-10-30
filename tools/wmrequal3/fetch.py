"""Fetch only named task-owned JSON/log outputs; preserve existing local evidence."""
import argparse
import base64
import io
import json
from contextlib import redirect_stdout
from common import ROOT, OWN, GUARD, remote


def get(names):
    with redirect_stdout(io.StringIO()):
        text=remote(GUARD+f'''
import base64,json
from pathlib import Path
p=Path({OWN!r})
out={{}}
for name in {names!r}:
 q=p/name
 if q.is_file():out[name]=base64.b64encode(q.read_bytes()).decode()
print(json.dumps(out))
''',timeout=180)
    return {n:base64.b64decode(raw) for n,raw in json.loads(text).items()}


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--cpu',action='store_true')
    ap.add_argument('--panel',action='store_true')
    ap.add_argument('--status',action='store_true')
    a=ap.parse_args()
    if a.status:
        remote(GUARD+f'''
import json,subprocess
from pathlib import Path
p=Path({OWN!r})
for name in ('LAUNCH.json','MONITOR_DONE.json','MONITOR_ERROR.json','validation/DONE.json'):
 q=p/name
 if q.is_file():print(name,q.read_text())
q=p/'panel.log'
if q.is_file():print('OWN_LOG_TAIL','\n'.join(q.read_text().splitlines()[-8:]))
print(subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.used,memory.free,utilization.gpu','--format=csv'],text=True))
''')
    else:
        names=['CPU.json','SOURCE.json'] if a.cpu else []
        if a.panel:names+=['LAUNCH.json','MONITOR_DONE.json','MONITOR_ERROR.json','MEMORY.jsonl','panel.log',
            'smoke.log','smoke/batch_000.json','validation/DONE.json']+[f'validation/batch_{i:03d}.json' for i in range(36)]
        dest=ROOT/'tools/wmrequal3/evidence'
        assert dest.parent.is_dir()
        dest.mkdir(exist_ok=True)
        for name,raw in get(names).items():
            q=dest/name
            q.parent.mkdir(parents=True,exist_ok=True)
            if q.exists():assert q.read_bytes()==raw,name
            else:q.write_bytes(raw)
            print('FETCHED',name,len(raw))
