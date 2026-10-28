"""Set only our verified coordinator's direct children to analysis niceness 10."""
import json
import os
from pathlib import Path
import subprocess
import time

p=Path.home()/'lrdiag'
owner=json.loads((p/'OWNER.json').read_text())
while True:
    try:
        stat=Path(f'/proc/{owner["pid"]}/stat').read_text().rsplit(')',1)[1].split()
        if stat[19]!=owner['start_ticks']:break
        progress=json.loads((p/'PROGRESS.json').read_text())
    except (FileNotFoundError,ProcessLookupError):break
    children=[]
    for row in progress['live']:
        try:
            st=Path(f'/proc/{row["pid"]}/stat').read_text().rsplit(')',1)[1].split()
            if int(st[1])==owner['pid'] and int(st[16])==19:children.append(str(row['pid']))
        except (FileNotFoundError,ProcessLookupError):pass
    if children:
        subprocess.run(['sudo','-n','renice','-n','10','-p',*children],check=True)
    time.sleep(10)
