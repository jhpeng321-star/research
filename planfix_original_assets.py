"""Read-only, bounded original-asset lookup; never contacts gpuA or runs a model."""
import argparse
import base64
import json
from pathlib import Path
import shlex
import subprocess
import time
import zlib
from planfix_ops import note, TEMP

CODE = r'''
import hashlib,json,time
from pathlib import Path
paths=__PATHS__
for name in paths:
 p=Path(name)
 try:
  exists=p.is_file()
  row=dict(path=name,exists=exists)
  if exists:
   row.update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
   if p.suffix=='.json' and p.stat().st_size<200000:row['json']=json.loads(p.read_text())
  print(json.dumps(row),flush=True)
 except PermissionError:print(json.dumps(dict(path=name,inaccessible=True)),flush=True)
'''

PATHS = {
 'cpuQ': ['/home/ubuntu/wmdata20261031/design.json',
          '/home/ubuntu/wmdata20261031/jobs/s969420_ctrl0/start.pt',
          '/home/ubuntu/wmdata20261031/jobs/s969420_ctrl0/start_history.npz'],
 'gpuR': ['/root/wmcalib/expected_hashes.json','/root/wmcalib/data/design.json',
          '/root/autodl-tmp/c4/setupaudit20261030/policy_000190.pt',
          '/root/autodl-tmp/c4/speedhead20261030/EXPORT_BOUND.json'],
}

if __name__ == '__main__':
 p=argparse.ArgumentParser();p.add_argument('host',choices=tuple(PATHS));a=p.parse_args()
 code=CODE.replace('__PATHS__',repr(PATHS[a.host]))
 packed=base64.b64encode(zlib.compress(code.encode(),9)).decode()
 target='python3 -B -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
 command=shlex.join(['timeout','25','ssh','-F','/mnt/c/Users/jpeng/.ssh/config',
  '-i','/home/jinghan/.ssh/sol_cpuv_rented_box',
  '-o','UserKnownHostsFile=/mnt/c/Users/jpeng/.ssh/known_hosts','-o','StrictHostKeyChecking=yes',
  '-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath=/tmp/planfix-assets-%C',
  '-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ConnectionAttempts=1',a.host,target])
 note('cpuV',f'original-asset availability lookup {a.host}, attempt 1/1, no process/model/SUMO action')
 r=subprocess.run(['wsl','-d','Ubuntu-22.04','--','bash','-c',command],capture_output=True,timeout=35)
 out=r.stdout.decode(errors='replace');err=r.stderr.decode(errors='replace')
 print(out,err,flush=True)
 receipt=dict(host=a.host,unix=time.time(),returncode=r.returncode,stdout=out,stderr=err,new_sim_s=0)
 assert TEMP.is_dir()
 with (TEMP/f'PLANFIX_ORIGINAL_ASSETS_{a.host}_{time.time_ns()}.json').open('x') as f:json.dump(receipt,f,indent=2)
 note('cpuV',f'original-asset lookup {a.host} outcome rc={r.returncode}')
