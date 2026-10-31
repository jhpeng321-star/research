"""Stage only task-owned code, immutable per SHA; never overwrite remote files."""
from pathlib import Path
import base64
import hashlib
import json
from remote import ssh

REMOTE = '/home/ubuntu/sumocore20261031'
HERE = Path(__file__).resolve().parent


def stage(names):
    payload = {}
    for name in names:
        extra={'eval_core.py':Path('E:/agent-work/c4-rb-eval/tools/rbeval/core.py'),
               'references.json':Path('E:/agent-work/c4-rb-eval/tools/rbeval/references.json'),
               'real_observables.json':Path('E:/agent-work/c4-rb-eval/tools/rbeval/real_observables.json'),
               'wm2_forecast_draws.csv':Path('E:/agent-work/c4-wmbreakdown/docs/paper_results/wmbreakdown/forecast_draws.csv')}
        assert Path(name).name == name
        data = extra.get(name,HERE/name).read_bytes()
        payload[name] = dict(data=base64.b64encode(data).decode(),
                             sha=hashlib.sha256(data).hexdigest())
    version = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    source = r'''
from pathlib import Path
import json, base64, hashlib, sys
PAYLOAD=json.load(sys.stdin)
root=Path(ROOT);assert root.parent.is_dir()
root.mkdir(exist_ok=True)
dest=root/'code'/VERSION;dest.mkdir(parents=True,exist_ok=True)
for name,record in PAYLOAD.items():
 data=base64.b64decode(record['data']);p=dest/name
 if p.exists(): assert p.read_bytes()==data
 else:
  with p.open('xb') as f:f.write(data)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha']
print(str(dest))
'''
    return ssh('ROOT='+repr(REMOTE)+'\nVERSION='+repr(version)+'\n'+source,
               data=json.dumps(payload).encode()).decode().strip()


if __name__ == '__main__':
    import sys
    print(stage(sys.argv[1:]))
