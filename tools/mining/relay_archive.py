"""Resumable immutable-archive relay; F6 cpuV reads, native laptop gpuA SFTP."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

PIN='212737b1a2c63ead99632db15c40e5fdba8be40281b44e105a3716a1b64cfa4e'
SIZE=767203965
SOURCE='/home/ubuntu/coopmwm20261106_s_mining/parity_stage_01/cpuv_parity.tar.gz'
LOCAL=Path('E:/Temp/opencode/mining_cpuv_parity.tar.gz')
DEST='/root/mining20261106/cpuv_parity.tar.gz'

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()

def download():
 assert LOCAL.parent.is_dir()
 offset=LOCAL.stat().st_size if LOCAL.exists() else 0
 assert offset<=SIZE
 while offset<SIZE:
  length=min(64*1024**2,SIZE-offset)
  reader=("from pathlib import Path; import sys; "
          f"f=Path({SOURCE!r}).open('rb');f.seek({offset});"
          f"sys.stdout.buffer.write(f.read({length}))")
  import shlex
  command='python3 -B -c '+shlex.quote(reader)
  helper=("import sys;sys.dont_write_bytecode=True;"
          "sys.path.insert(0,'/mnt/e/agent-work/c4-rerun-launch');"
          "from tools.coopmwm.cpuv_ssh import Transport;"
          f"r=Transport().run({command!r},timeout=100,capture_output=True);"
          "sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);"
          "sys.exit(r.returncode)")
  result=subprocess.run(['wsl','-d','Ubuntu-22.04','--','python3','-c',helper],
                         capture_output=True,timeout=115)
  assert result.returncode==0 and len(result.stdout)==length,(result.returncode,len(result.stdout),result.stderr)
  with LOCAL.open('ab') as f:f.write(result.stdout);f.flush()
  offset+=length
  print(json.dumps(dict(downloaded=offset,total=SIZE)),flush=True)
 assert digest(LOCAL)==PIN
 print(json.dumps(dict(local_verified=True,sha256=PIN)),flush=True)

def upload():
 assert LOCAL.stat().st_size==SIZE and digest(LOCAL)==PIN
 for attempt in range(1,4):
  script=f'reput {LOCAL.as_posix()} {DEST}\n'
  p=subprocess.run(['sftp','-o','BatchMode=yes','-o','ConnectTimeout=12',
                    '-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3','-b','-','gpuA'],
                   input=script,text=True,capture_output=True,timeout=240)
  print(json.dumps(dict(upload_attempt=attempt,returncode=p.returncode,
                         stdout=p.stdout,stderr=p.stderr)),flush=True)
  if p.returncode==0:return
  time.sleep(10)
 raise RuntimeError('bounded resumable archive upload failed')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['download','upload']);a=p.parse_args()
 download() if a.action=='download' else upload()
