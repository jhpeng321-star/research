"""Read-only resource, exact recorded-path availability and INT receipt audit."""
import argparse
from datetime import datetime
import json
from pathlib import Path

from tools.rbint.remote import ssh
from tools.rb2int.audit import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host', choices=('gpuA', 'gpuR'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--resources-only', action='store_true')
    args = parser.parse_args()
    code = 'SKIP_RECORDS='+repr(args.resources_only)+'\n'+r'''
import hashlib,json,os,shutil,subprocess
from pathlib import Path
def command(argv):
 r=subprocess.run(argv,capture_output=True,text=True)
 return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
def incarnation(pid):
 try:
  f=Path('/proc/'+str(pid)+'/stat').read_text().split()
  return dict(pid=pid,start_ticks=f[21],state=f[2],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
 except FileNotFoundError: return dict(pid=pid,exited=True)
result=dict(load=os.getloadavg(),cpus=os.cpu_count(),
 disk={p:dict(zip(('total','used','free'),shutil.disk_usage(p))) for p in ('/root','/root/autodl-tmp')},
 gpu=command(['nvidia-smi','--query-gpu=memory.free,memory.total,utilization.gpu','--format=csv']),
 compute=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv']))
cg=Path('/sys/fs/cgroup')
result['cgroup']={n:(cg/n).read_text() for n in ('cpu.max','cpu.stat','memory.max','memory.current','memory.stat') if (cg/n).is_file()}
result['host_mem_available']=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
result['records']=[]
for source,base,seeds in ([] if SKIP_RECORDS else [('WMDATA','/root/wmcalib/data/jobs',range(969432,969440)),('WMDATA2','/root/wmdata2/jobs',range(969550,969600))]):
 for s in seeds:
  row=dict(collection=source,seed=s,arms={})
  for arm in ('native','ctrl0','mappo','pi','circles'):
   root=Path(base)/('s'+str(s)+'_'+arm)
   names=('start.pt','start_history.npz','result.json','traj_base.npz','cumulative_120.json','cumulative_300.json','cumulative_600.json')
   row['arms'][arm]={n:dict(exists=(root/n).is_file(),size=(root/n).stat().st_size if (root/n).is_file() else None) for n in names}
  result['records'].append(row)
result['own_receipts']=[]
for base in (() if SKIP_RECORDS else ('/root/rbeval20261031/int','/root/rb2int20261101')):
 root=Path(base)
 if not root.is_dir(): continue
 dirs=[root]+[p for p in root.iterdir() if p.is_dir() and p.name.startswith(('WM-PHYS-','FULL-','WM2','shard'))]
 for directory in dirs:
  for name in ('SPEC.json','CANARY_PROCESS.json','PANEL_PROCESS.json','MONITOR_PROCESS.json','PROCESS.json','DISPATCH.json','COMPLETE.json'):
   p=directory/name
   if not p.is_file(): continue
   raw=p.read_bytes();data=json.loads(raw)
   processes=data.get('workers',[data]) if isinstance(data,dict) else []
   result['own_receipts'].append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),receipt=data,
       incarnations=[incarnation(r['pid']) for r in processes if isinstance(r,dict) and 'pid' in r]))
print(json.dumps(result))
'''
    data = json.loads(ssh(args.host, code, timeout=60))
    data.update(schema='rb2int-read-only-resources/1', host=args.host,
        recorded_at=datetime.now().astimezone().isoformat(), source_sha256=sha(__file__),
        opened_record_outcomes=False, launched_jobs=False)
    out = Path(args.out)
    if not out.parent.is_dir():
        raise ValueError('receipt parent required')
    with out.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=1)
    missing = [dict(seed=r['seed'], arm=a, name=n) for r in data['records']
               for a, files in r['arms'].items() for n, f in files.items() if not f['exists']]
    print(json.dumps(dict(receipt=str(out), sha256=sha(out), host=args.host,
        load=data['load'], gpu=data['gpu'], compute=data['compute'],
        cgroup=data['cgroup'], host_mem_available=data['host_mem_available'], disk=data['disk'], missing_record_files=len(missing),
        missing_by_collection={collection: sum(1 for m in missing
            if (m['seed'] < 969550) == (collection == 'WMDATA'))
            for collection in ('WMDATA', 'WMDATA2')},
        prior_receipts=len(data['own_receipts'])), indent=1))


if __name__ == '__main__':
    main()
