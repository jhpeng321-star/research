"""Inspect only this preparation's named local F6 reservation and owner."""
import json,time
from pathlib import Path

p=Path('/tmp/c4-cpuv-1000/active.json')
if not p.exists():
    print('NO_ACTIVE_RESERVATION')
else:
    value=json.loads(p.read_text())
    command=value['argv'][-1]
    own='cpu_prepare_' in command and ('95441ec4a7c4' in command or 'ea559c952ff0' in command)
    print(json.dumps(dict(own_cpu_preparation=own,parent=value['parent'],deadline=value['deadline'],
        now=time.monotonic(),result=value['result'],cancel=value['cancel'])))
    if own:
        parent=Path('/proc')/str(value['parent'])
        print('PARENT_EXISTS',parent.exists())
        if parent.exists():
            print('PARENT_CMD', (parent/'cmdline').read_bytes().replace(b'\0',b' ').decode())
            children=parent/'task'/str(value['parent'])/'children'
            print('PARENT_CHILDREN',children.read_text())
            for pid in children.read_text().split():
                proc=Path('/proc')/pid
                print('CHILD',pid,(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode())
