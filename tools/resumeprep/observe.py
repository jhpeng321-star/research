"""Read own exact probe PIDs and compact timing receipts."""
import json
from pathlib import Path
import statistics
import sys

root=Path('/root/resumeprep_probe/probe3')
result={}
for name in ('POLICY_TIMING','WM_TIMING'):
    p=root/(name+'.json')
    if p.exists():
        row=json.loads(p.read_text())
        if name=='POLICY_TIMING':
            samples=[x['wall_time_s'] for x in row.pop('policy_updates',[])][1:]
            row.update(samples_after_warmup_s=samples,median_s=statistics.median(samples) if samples else None)
        result[name]=row
for arg in sys.argv[1:]:
    path=Path('/proc')/str(int(arg))
    if (path/'stat').exists():
        fields=(path/'stat').read_text().rsplit(')',1)[1].split()
        result['pid_'+arg]={'state':fields[0],'start_ticks':fields[19],
                           'command':(path/'cmdline').read_bytes().replace(b'\0',b' ').decode()}
    else:
        result['pid_'+arg]={'live':False}
print(json.dumps(result,indent=1))
