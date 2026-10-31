"""Durable cross-host relay for hash-bound file RPC; never dispatches SUMO."""
import argparse
import json
import os
from pathlib import Path
import time

from .ops import ssh
from .storage import write

CPU = os.environ.get('C1SWITCH_CPU_RPC','/home/ubuntu/c1switch20261031/main_v6/rpc')
GPU = os.environ.get('C1SWITCH_GPU_RPC','/root/c1switch20261031/main_v6/rpc')


def markers():
    rows = ssh('cpuQ',f'''import json
from pathlib import Path
root=Path({CPU!r});out={{}}
if (root.parent/'STOP').exists():
 for arm in ('wm','bb'):out[arm+'/STOP']=json.loads((root.parent/'STOP').read_text())
else:
 for arm in ('wm','bb'):
  p=root/arm/'COMPLETE'
  if p.exists():out[arm+'/COMPLETE']=json.loads(p.read_text())
print(json.dumps(out))
''')
    if rows:
        ssh('gpuR',f'''import json
from pathlib import Path
root=Path({GPU!r})
for name,row in {rows!r}.items():
 p=root/name
 if not p.exists():p.write_text(json.dumps(row))
print(json.dumps(dict(count={len(rows)})))
''')
    failures = ssh('gpuR',f'''import json
from pathlib import Path
root=Path({GPU!r});out={{}}
for arm in ('wm','bb'):
 p=root/arm/'STOP'
 if p.exists():out[arm]=json.loads(p.read_text())
print(json.dumps(out))
''')
    if failures:
        ssh('cpuQ',f'''import json
from pathlib import Path
p=Path({CPU!r}).parent/'STOP'
if not p.exists():p.write_text(json.dumps(dict(reason='GPU service stopped',models={failures!r})))
print(json.dumps(dict(stopped=True)))
''')


def transfer(source, source_root, dest, dest_root, suffix):
    rows = ssh(source,f'''import json
from pathlib import Path
root=Path({source_root!r});out={{}}
for arm in ('wm','bb'):
 for p in sorted((root/arm).glob('*{suffix}.json')):
  if not p.with_name(p.name+'.relayed').exists():out[arm+'/'+p.name]=json.loads(p.read_text())
print(json.dumps(out))
''',timeout=45)
    if not rows:
        return 0
    # Remote write validates every path against traversal and forbidden components;
    # hash equality prevents replaying a stale or conflicting response.
    ssh(dest,f'''import json,sys
from pathlib import Path
sys.path.insert(0,str(Path({dest_root!r}).parent/'pytools'))
from tools.c1switch.storage import canonical,write
root=Path({dest_root!r});rows=json.loads({json.dumps(rows)!r})
for name,row in rows.items():
 parts=Path(name).parts
 assert len(parts)==2 and parts[0] in ('wm','bb') and len(parts[1].split('.')[0])==64
 p=root/name
 if p.exists():assert canonical(json.loads(p.read_text()))==canonical(row)
 else:write(p,row)
print(json.dumps(dict(count=len(rows))))
''',timeout=45)
    ssh(source,f'''import json
from pathlib import Path
root=Path({source_root!r})
for name in {list(rows)!r}:
 p=(root/name).with_name(Path(name).name+'.relayed')
 if not p.exists():p.write_text('ack')
print(json.dumps(dict(count={len(rows)})))
''',timeout=45)
    return len(rows)


def main(root):
    from .storage import exclusive
    with exclusive(root):
        write(root/'START.json',dict(unix=time.time(),cpu=CPU,gpu=GPU))
        while not (root/'STOP').exists():
            try:
                count = transfer('cpuQ',CPU,'gpuR',GPU,'.request')
                count += transfer('gpuR',GPU,'cpuQ',CPU,'.result')
                markers()
                write(root/'STATUS.json',dict(unix=time.time(),last_transferred=count))
            except Exception as exc:
                # Transport failures never dispatch/retry scientific jobs; requests
                # are immutable and resumed by content hash after connectivity returns.
                write(root/'TRANSPORT_ERROR.json',dict(unix=time.time(),reason=str(exc)))
            time.sleep(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
