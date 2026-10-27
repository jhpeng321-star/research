"""Archive the dense attempt and interrupt only its token-bound own coordinator."""
import json
import os
from pathlib import Path
import signal
import time

p=Path.home()/'lrdiag';assert p.is_dir()
owner=json.loads((p/'OWNER.json').read_text());pid=owner['pid']
def stat(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    except (FileNotFoundError,ProcessLookupError):return None
s=stat(pid);assert s and s[19]==owner['start_ticks']
progress=json.loads((p/'PROGRESS.json').read_text())
children=[]
for row in progress['live']:
    cs=stat(row['pid'])
    if cs and int(cs[1])==pid:children.append((row['pid'],cs[19]))
archive=p/'dense_reference';assert not archive.exists();archive.mkdir()
(archive/'INTERRUPT.json').write_text(json.dumps({'owner':owner,'progress':progress,'children':children,
    'reason':'existing supported compact-slots inference path, pre-verdict numerical parity check',
    'time':time.time()},indent=1))
os.kill(pid,signal.SIGINT)
deadline=time.monotonic()+90
while True:
    alive=[]
    for child,token in children:
        s=stat(child)
        if s and s[19]==token and s[0]!='Z':alive.append(child)
    s=stat(pid)
    own_alive=bool(s and s[19]==owner['start_ticks'] and s[0]!='Z')
    if not alive and not own_alive:break
    if time.monotonic()>deadline:raise RuntimeError(('owned shutdown did not finish',alive,own_alive))
    time.sleep(1)
for sub in ('results','traces'):
    q=archive/sub;q.mkdir()
    for f in (p/sub).iterdir():
        if sub=='traces' or f.name.endswith('_s600.json'):f.rename(q/f.name)
for name in ('OWNER.json','PROGRESS.json','DONE.json','coordinator.log','priority.log'):
    f=p/name
    if f.exists():f.rename(archive/name)
(archive/'STOPPED.json').write_text(json.dumps({'status':'own coordinator and children exited','time':time.time()}))
print('archived dense references',len(list((archive/'results').glob('*.json'))),'own workers stopped')
