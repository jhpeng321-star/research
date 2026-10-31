"""Read-only GPU/CPU and fixed registered-panel ownership audit, no dispatch."""
import argparse
from tools.rbint.remote import ssh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', choices=('gpuA', 'gpuR'), required=True)
    args = parser.parse_args()
    code = '''
import json, os, subprocess, time
from pathlib import Path
base=Path('/root/rbeval20261031')
cg=Path('/sys/fs/cgroup')
def cpu_usage():
 return int(dict(x.split() for x in (cg/'cpu.stat').read_text().splitlines())['usage_usec'])
before=cpu_usage(); begin=time.monotonic(); time.sleep(1)
used=(cpu_usage()-before)/1e6/(time.monotonic()-begin)
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def alive(r):
 try:
  fields=Path('/proc/'+str(r['pid'])+'/stat').read_text().split()
  return fields[2]!='Z' and r.get('boot_id')==boot and str(r.get('start_ticks'))==fields[21]
 except FileNotFoundError: return False
roots=['runa/WM-PHYS-A2','runa/WM-PHYS-C2','runb/WM-PHYS-B2',
 'int/WM-PHYS-FULL-v1','int/WM-PHYS-FULL-v3-gpu','int/WM-PHYS-FULL-v4-gpu',
 'int/WM2-HDV60','int/WM-PHYS-C2']
items=[]
for name in roots:
 root=base/name
 if not root.is_dir(): continue
 worlds={}; shards=[]; owned=[]
 for i in range(3):
  d=root/f'shard{i}'
  files=list(d.glob('w*_k*.json'))
  for f in files:
   w,k=f.stem[1:].split('_k');worlds.setdefault(int(w),set()).add(int(k))
  shards.append(dict(tasks=len(files),complete=(d/'COMPLETE.json').exists(),stop=(d/'STOP').exists()))
 for receipt in ('PANEL_PROCESS.json','MONITOR_PROCESS.json'):
  p=root/receipt
  if p.exists():
   data=json.loads(p.read_text())
   for r in data.get('workers',[data]): owned.append(dict(pid=r['pid'],alive=alive(r),receipt=receipt))
 items.append(dict(root=str(root),shards=shards,complete_worlds=sorted(w for w,k in worlds.items() if k==set(range(8))),
  processes=owned,terminal=[n for n in ('RESULT.json','FAILED.json','MONITOR_COMPLETE.json','MONITOR_FAILED.json') if (root/n).exists()]))
runtime=base/'phys_code_4015b6412cdf'
print(json.dumps(dict(unix=time.time(),load=os.getloadavg(),logical_cpus=os.cpu_count(),
 cpu_quota=(cg/'cpu.max').read_text().strip(),cpu_cores_used=used,
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.used,utilization.gpu','--format=csv'],text=True).strip(),
 compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv'],text=True).strip(),
 source4015b641_present=runtime.is_dir(),C_entry_present=(runtime/'tools/rbint/evaluate_C.py').is_file(),panels=items),indent=1))
'''
    print(ssh(args.host, code, timeout=60))


if __name__ == '__main__':
    main()
