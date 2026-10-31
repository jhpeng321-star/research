"""Read-only in-session audit of the existing watcher and gpuR evaluations."""
import argparse
from datetime import datetime
import hashlib
import json
import time

import watch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sleep', type=int, default=0)
    a = p.parse_args()
    assert 0 <= a.sleep <= 600
    time.sleep(a.sleep)
    state = watch.WORK/'runs'/'rbrunb'
    handoffs = []
    for path in watch.HANDOFFS:
        text = path.read_text(encoding='utf-8-sig')
        handoffs.append(dict(path=str(path), sha256=hashlib.sha256(text.encode()).hexdigest()))
    # No dispatch, process action, task replay, scoring or raw data access.
    remote = json.loads(watch.ssh(r'''
from pathlib import Path
import json,os,subprocess
r=Path('/root/rbeval20261031/runb/WM-PHYS-B2')
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
live=[]
for f in r.glob('SUPERVISOR_*.json'):
 x=json.loads(f.read_text()); p=Path('/proc')/str(x['pid'])
 try:
  alive=x['boot_id']==boot and (p/'stat').read_text().split()[21]==x['start_ticks'] and (p/'stat').read_text().split()[2]!='Z'
 except (FileNotFoundError,ProcessLookupError): alive=False
 if alive: live.append(dict(pid=x['pid'],deadline_unix=x['deadline_unix']))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
shards=[d for d in r.glob('shard*') if d.is_dir()]
workers=[]
for f in r.glob('LAUNCHED_*.json'):
 x=json.loads(f.read_text()); p=Path('/proc')/str(x['pid'])
 workers.append(dict(pid=x['pid'],exists=p.exists()))
terminal={n:json.loads((r/n).read_text()) for n in ('FAILED.json','DEADLINE.json','FINALIZED.json') if (r/n).exists()}
log=r/'supervisor_noon.log'
lines=log.read_text().splitlines() if log.exists() else []
print(json.dumps(dict(gpu=gpu,load=os.getloadavg()[0],supervisors=live,workers=workers,
 tasks=sum(len(list(d.glob('w*_k*.json'))) for d in shards),
 shards={d.name:len(list(d.glob('w*_k*.json'))) for d in shards},
 dispatched=(r/'DISPATCH.json').exists(),terminal=terminal,last_log=lines[-1] if lines else None)))
'''))
    status = json.loads((state/'STATUS.json').read_text())
    event = dict(time=datetime.now().astimezone().isoformat(), owner=watch.OWNER,
                 handoffs=handoffs, watcher_status=status, remote=remote)
    target = state/f'OBSERVE_{time.time_ns()}.json'
    watch.save_new(target, event)
    # Interim coordination is operational status, never a partial efficacy claim.
    now = datetime.now().astimezone()
    for hour in (4, 8):
        note = watch.SHARED/f'RUNB_INTERIM_{hour:02d}00.json'
        if now.hour >= hour and not note.exists():
            watch.save_new(note, event)
    print(json.dumps(dict(receipt=str(target), **event)))


if __name__ == '__main__':
    main()
