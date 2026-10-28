"""Stop only the recorded entrant coordinator and its verified descendants by PID."""
import sys
from remote import ssh

ssh(sys.argv[1], '''
import json,os,signal,time
from pathlib import Path
p=Path.home()/'entrant';owner=json.loads((p/'OWNER.json').read_text())
assert owner['panel']=='e3'
def stat(pid):
    f=Path('/proc')/str(pid)/'stat'
    return f.read_text().split() if f.exists() else None
root=owner['pid'];s=stat(root)
assert s and s[21]==owner['start_ticks'],'recorded owner is not live'
owned={root:s[21]};todo=[root]
while todo:
    parent=todo.pop();f=Path('/proc')/str(parent)/'task'/str(parent)/'children'
    for text in f.read_text().split() if f.exists() else []:
        pid=int(text);child=stat(pid)
        if child and int(child[3])==parent:
            owned[pid]=child[21];todo.append(pid)
receipt={'reason':'supersede ON scoring after correcting recorded-neighbour lane-intent and one-second history',
         'owner':owner,'owned_pid_incarnations':owned,'stopped':time.time()}
(p/'SUPERSEDED_V3_STOP.json').write_text(json.dumps(receipt,indent=1))
for pid,ticks in owned.items():
    current=stat(pid)
    if current and current[21]==ticks and current[2]!='Z':os.kill(pid,signal.SIGTERM)
time.sleep(.5)
receipt['still_running']=[pid for pid,ticks in owned.items() if (stat(pid) or [None]*22)[21]==ticks
                          and (stat(pid) or [None]*22)[2]!='Z']
(p/'SUPERSEDED_V3_STOP.json').write_text(json.dumps(receipt,indent=1))
print(json.dumps({'verified_owned_pids':len(owned),'still_running':receipt['still_running']}))
''')
