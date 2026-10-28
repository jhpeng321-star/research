"""Read task-owned receipts and PID incarnations without scanning processes."""
import sys
from remote import ssh

ssh(sys.argv[1], '''
import json
from pathlib import Path
p=Path.home()/'entrant'
print('EXISTS',p.is_dir())
for f in sorted(p.glob('*.json')):
    if f.name=='SOURCE.json': continue
    value=json.loads(f.read_text())
    if isinstance(value,dict) and 'pid' in value:
        stat=Path('/proc')/str(value['pid'])/'stat'
        value['live_same_incarnation']=stat.exists() and stat.read_text().split()[21]==value.get('start_ticks')
    print(f.name,json.dumps(value))
for name in ('results','results_v2','errors'):
    files=list((p/name).glob('*.json'))
    print(name,len(files))
    if name=='errors':
        for f in files[-3:]: print(f.name,f.read_text())
for name in ('canary.log','queue.log'):
    f=p/name
    if f.exists(): print(name,f.read_text()[-12000:])
''')
