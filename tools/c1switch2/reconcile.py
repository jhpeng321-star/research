"""Read-only fresh-session receipts for exact owned roots and resource accounting."""
import json
import time

from tools.c1switch.storage import write
from .operations import EVIDENCE, ssh
from .watch import collect, local_process


CPUQ = '''import json,os,shutil,time
from pathlib import Path
from tools.c1switch.design import FORBIDDEN
out=dict(unix=time.time(),uptime_s=float(Path('/proc/uptime').read_text().split()[0]),
         disk=shutil.disk_usage('/home/ubuntu')._asdict(),processes={},own_outputs={})
for pid in (649272,645235,606968):
 p=Path('/proc',str(pid))
 if p.exists():
  fields=(p/'stat').read_text().split()
  out['processes'][str(pid)]=dict(state=fields[2],start_ticks=int(fields[21]),uid=p.stat().st_uid,
      command=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace'))
 else:out['processes'][str(pid)]=dict(state='EXITED')
for item in os.scandir('/home/ubuntu'):
 if not item.name.startswith('c1switch2_') or not item.is_dir(follow_symlinks=False):continue
 stack=[item.path];size=count=0
 while stack:
  current=stack.pop()
  for entry in os.scandir(current):
   if entry.is_symlink() or FORBIDDEN.search(entry.path):continue
   if entry.is_dir(follow_symlinks=False):stack.append(entry.path)
   elif entry.is_file(follow_symlinks=False):
    size+=entry.stat(follow_symlinks=False).st_size;count+=1
 out['own_outputs'][item.path]=dict(bytes=size,files=count)
print(json.dumps(out))
'''


def run():
    snapshot = collect()
    # Install only the already-frozen source path to obtain the path-pruning regex.
    snapshot['cpuQ_resources'] = ssh('cpuQ',
        "import sys\nsys.path.insert(0,'/home/ubuntu/c1switch2_run_292bfffe94b13ed8/pytools')\n" + CPUQ,
        timeout=45)
    snapshot['observer'] = local_process(25348)
    snapshot['bb_startup_stop'] = json.loads((EVIDENCE/'BB_RELOCATION_STOP.json').read_text())
    snapshot['read_only'] = True
    path = EVIDENCE/('RESUME_RECONCILIATION_'+str(int(time.time()))+'.json')
    write(path, snapshot)
    resources = snapshot['cpuQ_resources']
    print(json.dumps(dict(receipt=str(path), observer=snapshot['observer'],
        uptime_days=resources['uptime_s']/86400, disk=resources['disk'],
        own_output_bytes=sum(row['bytes'] for row in resources['own_outputs'].values()),
        processes=resources['processes'], bb_stop=snapshot['bb_startup_stop'],
        alerts=snapshot['alerts']), indent=2))


if __name__ == '__main__':
    run()
