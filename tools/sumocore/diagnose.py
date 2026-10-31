"""Read-only collision and source-accounting diagnostics; never retunes runs."""
from remote import ssh

SOURCE=r'''
from pathlib import Path
import json,re,collections
r=Path('/home/ubuntu/sumocore20261031')
for p in (r/'full.log',r/'resume1.log',r/'resume2.log'):
 if not p.exists():continue
 lines=[s for s in p.read_text().splitlines() if 'collision' in s.lower() and 'Warning' in s]
 times=[float(m.group(1)) for s in lines if (m:=re.search(r'time=([0-9.]+)',s))]
 print(json.dumps(dict(log=p.name,collision_warning_lines=len(lines),times=times[:30],examples=lines[:8])))
'''
if __name__=='__main__':print(ssh(SOURCE).decode())
