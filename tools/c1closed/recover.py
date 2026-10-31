"""Read-only recovery of owned jobs and receipts; never signals processes."""
import json
from probe import ssh

print(ssh('gpuA', '''from pathlib import Path
import json,os,subprocess,hashlib
p=Path('/root/autodl-tmp/c4/c1closed20261031')
owned=[]
for q in Path('/proc').iterdir():
 if not q.name.isdigit(): continue
 try:
  argv=(q/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace')
  if str(p) in argv:
   owned.append(dict(pid=int(q.name),command=argv,start_ticks=(q/'stat').read_text().split(') ',1)[1].split()[19]))
 except (FileNotFoundError,PermissionError,ProcessLookupError): pass
print('OWNED_PATH_PROCESSES',json.dumps(owned))
for name in ('DISPATCH.json','COMPLETE.json','STATUS.json','evidence/CPU_VERIFIED.json'):
 q=p/name
 print(name,q.read_text() if q.exists() else 'ABSENT')
print('SMOKE_SUMMARY')
for q in sorted((p/'cpu_smoke').glob('*_common_*json')):
 x=json.loads(q.read_text()); print(json.dumps({k:x.get(k) for k in ('model','seconds','steps','runtime_seconds','pid','device','plant')}))
print('load',os.getloadavg(),'cpus',os.cpu_count())
print(subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free','--format=csv,noheader'],text=True))
print(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True))
'''))
