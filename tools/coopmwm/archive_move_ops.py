"""Qualify/prepare/register/dispatch the one authorised SHA-gated archive move."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from . import ops
from .protocol import checked,sha,REGISTRY
from .storage import read,write
from .ls_validity_ops import remote

ROOT='/home/ubuntu/coopmwm20261103_archive1450'
CPU_RECORD='ARCHIVE_MOVE_CPU_1450.json'
PREPARE_RECORD='ARCHIVE_MOVE_PREPARE_DISPATCH_1450.json'
REG_RECORD='ARCHIVE_MOVE_REGISTRATION_1450.json'
CLIENT_RECORD='ARCHIVE_MOVE_CLIENT_DISPATCH_1450.json'
CLIENT_LOG='ARCHIVE_MOVE_CLIENT_1450.log'
RECOVERY_NOTE=''


def qualify():
    files={n:checked(ops.LOCAL/'tools/coopmwm'/n).read_text(encoding='utf-8') for n in ('archive_move.py','test_archive_move_cpu.py')}
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,subprocess,hashlib,time
r=Path({ROOT!r});assert r.parent.is_dir();r.mkdir(exist_ok=True)
files=json.loads({json.dumps(files)!r})
for n,text in files.items():
 p=r/n
 if p.exists():assert p.read_text()==text,n
 else:
  with p.open('x') as f:f.write(text)
p=r/'CPU_RESULT.json'
if not p.exists():
 start=time.time();x=subprocess.run(['sudo','-n','/home/ubuntu/miniconda3/bin/python','-B','-m','unittest','discover','-s',str(r),'-p','test_archive_move_cpu.py','-v'],capture_output=True,text=True,timeout=60)
 v=dict(pass_=x.returncode==0,returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_s=time.time()-start,new_sim_s=0,moved_bytes=0,source_hashes={{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in files}})
 with p.open('x') as f:json.dump(v,f)
print(p.read_text())''',timeout=75))
    p=ops.EVIDENCE/CPU_RECORD
    if not p.exists():write(p,value)
    print(json.dumps(dict(receipt=str(p),**value)))


def prepare():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,subprocess,time
r=Path({ROOT!r});cpu=json.loads((r/'CPU_RESULT.json').read_text());assert cpu['pass_']
p=r/'PREPARE_DISPATCH.json'
if not p.exists():
 with (r/'PREPARE.log').open('x') as log:
  child=subprocess.Popen(['sudo','-n','/home/ubuntu/miniconda3/bin/python','-B','-u',str(r/'archive_move.py'),'prepare'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 v=dict(pid=child.pid,start_ticks=Path('/proc',str(child.pid),'stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),mode='manifest only; no transfer/deletion')
 with p.open('x') as f:json.dump(v,f)
print(p.read_text())''',timeout=45))
    p=ops.EVIDENCE/PREPARE_RECORD
    if not p.exists():write(p,value)
    print(json.dumps(value))


def status():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time,shutil
r=Path({ROOT!r});out=dict(unix=time.time(),disk_free=shutil.disk_usage('/home/ubuntu').free)
for n in ('PREPARE_OWNER','PREPARE_COMPLETE','PREPARE_FAILED','MOVE_OWNER','MOVE_FAILED','COMPLETE'):
 p=r/(n+'.json');out[n]=json.loads(p.read_text()) if p.exists() else None
out['batches']=len(list(r.glob('ARCHIVE_MOVE_batch*.json')))
for n in ('PREPARE_OWNER','MOVE_OWNER'):
 d=out[n]
 if d:
  p=Path('/proc',str(d['pid']));s=(p/'stat').read_text().split() if p.exists() else None
  assert s is None or s[21]==d['start_ticks'];out[n+'_state']=s[2] if s else 'ABSENT'
print(json.dumps(out))''',timeout=40))
    p=ops.EVIDENCE/f'ARCHIVE_MOVE_STATUS_{time.time_ns()}.json';write(p,value)
    print(json.dumps(dict(receipt=str(p),**value)))


def register():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,hashlib
r=Path({ROOT!r});v=json.loads((r/'PREPARE_COMPLETE.json').read_text());cpu=json.loads((r/'CPU_RESULT.json').read_text());m=json.loads((r/'MANIFEST.json').read_text())
assert hashlib.sha256((r/'MANIFEST.json').read_bytes()).hexdigest()==v['manifest_sha256']
assert not (r/'PREPARE_FAILED.json').exists() and cpu['pass_']
categories={{}}
for e in m['entries']:
 item=categories.setdefault(e['category'],dict(files=0,bytes=0));item['files']+=1;item['bytes']+=e['bytes']
print(json.dumps(dict(prepared=v,source_sha256=cpu['source_hashes']['archive_move.py'],categories=categories)))''',timeout=45))
    p=ops.EVIDENCE/REG_RECORD
    if not p.exists():
        line=('\nS-COOPMWM MAIN2026-11-03 14:50 SHA-GATED MOVE: selected completed '
            'LS label_attempts/pending, completed original policy-synergy branch artifacts, '
            'and interim observables only; RESULTs for interim remain on cpuQ. '
            'All admitted-bank/lesson inputs and active FD/direct references protected; '
            'no replay_bank/checkpoints/bank receipts/scenario XML/forbidden file hashed '
            'or moved. ManifestSHA'+value['prepared']['manifest_sha256']+
            '; rsync to gpuR/root/cpuq_archive_20261103/coopmwm_1450_v1; verify EVERY '
            'destination SHA, recheck source SHA/inode/open-FD protection, then unlink '
            'ONLY exact verified sources. Batch receipts, target>=10GiB free. '
            'Existing unrestricted identity forwarded through SSH agent, no private '
            'key copied/no host config changed. Own durable CPU worker/PID receipts, '
            'no simulation/inference/reservation, no failed physics replay.\n')
        write(p,dict(authority='MAIN2026-11-03 14:50',line=line+RECOVERY_NOTE,**value,new_sim_s=0,moved_bytes=0))
    v=read(p);h=sha(p);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(v['line']+'amendmentSHA'+h+'\n')
    print(remote('cpuQ',f'''from pathlib import Path
import json
r=Path({ROOT!r});p=r/'AUTHORITY.json';v=json.loads({json.dumps(dict(v,amendment_sha256=h,manifest_sha256=v['prepared']['manifest_sha256']))!r})
if p.exists():assert json.loads(p.read_text())==v
else:
 with p.open('x') as f:json.dump(v,f)
print(json.dumps(dict(registered=True,sha256={h!r},manifest_sha256=v['manifest_sha256'],categories=v['categories'],new_sim_s=0,moved_bytes=0)))''',timeout=40))


def launch():
    import psutil
    p=ops.EVIDENCE/CLIENT_RECORD
    if p.exists():
        d=read(p)
        try:q=psutil.Process(d['pid']);alive=q.create_time()==d['create_time'] and 'ssh' in q.name().lower()
        except psutil.NoSuchProcess:alive=False
        print(json.dumps(dict(owner=d,alive=alive,adopt=True)));return
    v=read(ops.EVIDENCE/REG_RECORD);h=v['prepared']['manifest_sha256']
    # SSH-agent forwarding keeps key bytes on Windows. The root worker ignores
    # HUP; any transport/verification error retains unverified source files.
    command=['ssh','-A','-T','-o','BatchMode=yes','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=8','cpuQ',
        'sudo -n env SSH_AUTH_SOCK="$SSH_AUTH_SOCK" /home/ubuntu/miniconda3/bin/python -B -u '+ROOT+'/archive_move.py move --manifest-sha '+h]
    assert ops.EVIDENCE.is_dir()
    with (ops.EVIDENCE/CLIENT_LOG).open('x',encoding='utf-8') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=(subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0,
            start_new_session=os.name!='nt')
    d=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),unix=time.time(),manifest_sha256=h,
        source_sha256=v['source_sha256'],mode='registered SHA-gated move; not physics',agent_forwarding=True)
    write(p,d);print(json.dumps(d))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('qualify','prepare','status','register','launch'))
    globals()[p.parse_args().action]()
