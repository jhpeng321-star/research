"""Read-only host capacity/owned-task telemetry; never dispatch or signal jobs."""
import argparse
import json
import time
from . import ops
from .storage import write


def inspect(host):
    value=json.loads(ops.ssh.remote(host,'''import pathlib,json,os,subprocess,time
processes=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
  if 'tools.coopmwm.' not in cmd or 'cmd=(p/' in cmd:continue
  s=(p/'stat').read_text().split()
  processes.append(dict(pid=int(p.name),start_ticks=s[21],state=s[2],cpu_ticks=int(s[13])+int(s[14]),cmd=cmd))
 except (OSError,PermissionError,ProcessLookupError):pass
quota={}
for name in ('/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/cpu/cpu.cfs_quota_us','/sys/fs/cgroup/cpu/cpu.cfs_period_us','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'):
 p=pathlib.Path(name)
 if p.is_file():quota[name]=p.read_text().strip()
mem={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','MemTotal:'))}
v=dict(unix=time.time(),boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),load=os.getloadavg(),cpu_count=os.cpu_count(),affinity_count=len(os.sched_getaffinity(0)),cgroup=quota,memory=mem,task_processes=processes,new_sumo_sim_s=0)
if pathlib.Path('/usr/bin/nvidia-smi').exists():
 for key,args in [('gpu',['--query-gpu=uuid,memory.total,memory.used,utilization.gpu,power.draw,power.limit','--format=csv,noheader']),('compute',['--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])]:
  p=subprocess.run(['nvidia-smi']+args,capture_output=True,text=True,check=True);v[key]=p.stdout.strip()
print(json.dumps(v))''',timeout=35))
    value['host']=host
    write(ops.EVIDENCE/f'WARM_CAPACITY_{host}_{int(value["unix"])}.json',value)
    print(json.dumps(value));return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',choices=('cpuQ','gpuA','gpuR'),required=True)
    inspect(p.parse_args().host)
