"""Operational entry points for MAIN16:47 rolling storage admission."""
import argparse
import json
import time
from . import ops
from .ls_validity_ops import remote
from .protocol import checked, sha, REGISTRY
from .storage import read, write

ROOT='/home/ubuntu/coopmwm20261103_rolling1647'


def inventory():
    print(remote('cpuQ', '''from pathlib import Path
import json,shutil,time,subprocess,shlex
r=Path('/home/ubuntu/coopmwm20261102_warm1');groups={}
for p in (r/'jobs').glob('*/observable.jsonl.gz'):
 if any(x in str(p).lower() for x in ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1')):continue
 if not (p.parent/'RESULT.json').exists() or not (p.parent/'CLOSE.json').exists():continue
 if time.time()-max(p.stat().st_mtime,(p.parent/'CLOSE.json').stat().st_mtime)<1800:continue
 k=p.parent.name.rsplit('_',1)[0];row=groups.setdefault(k,[0,0]);row[0]+=1;row[1]+=p.stat().st_size
key=Path('/home/ubuntu/.ssh/id_ed25519_boxsync')
code="import shutil,json;print(json.dumps(dict(free=shutil.disk_usage('/root/cpuq_archive_20261103').free)))"
cmd=['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/home/ubuntu/coopmwm20261102_remote_routes5/known_hosts_gpuR','-o','ConnectTimeout=20','-i',str(key),'-p','53897','root@connect.westb.seetacloud.com','/root/miniconda3/bin/python -c '+shlex.quote(code)]
p=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
print(json.dumps(dict(cpu_free=shutil.disk_usage(r).free,groups=groups,key_exists=key.exists(),auth_probe_returncode=p.returncode,auth_probe=p.stdout,auth_error=p.stderr)))''',timeout=60))


def qualify():
    names=('rolling_archive.py','archive_move.py','test_rolling_archive_cpu.py')
    files={n:checked(ops.LOCAL/'tools/coopmwm'/n).read_text(encoding='utf-8') for n in names}
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,hashlib,subprocess,time
r=Path({ROOT!r});assert r.parent.is_dir();r.mkdir(exist_ok=True)
files=json.loads({json.dumps(files)!r})
for n,text in files.items():
 p=r/n
 if p.exists():assert p.read_text()==text,n
 else:
  with p.open('x') as f:f.write(text)
result=r/'CPU_RESULT.json'
if not result.exists():
 start=time.time();p=subprocess.run(['sudo','-n','/home/ubuntu/miniconda3/bin/python','-B','-m','unittest','discover','-s',str(r),'-p','test_rolling_archive_cpu.py','-v'],capture_output=True,text=True,timeout=60)
 v=dict(pass_=p.returncode==0,stdout=p.stdout,stderr=p.stderr,wall_s=time.time()-start,new_sim_s=0,
  source_hashes={{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in files}})
 with result.open('x') as f:json.dump(v,f)
print(result.read_text())''',timeout=75))
    p=ops.EVIDENCE/'ROLLING_ARCHIVE_CPU_1647.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def register():
    cpu=read(ops.EVIDENCE/'ROLLING_ARCHIVE_CPU_1647.json');assert cpu['pass_']
    record=ops.EVIDENCE/'ROLLING_ARCHIVE_REGISTRATION_1647.json'
    if not record.exists():
        line=('\nS-COOPMWM MAIN2026-11-03 16:27/16:47 DURABLE ROLLING SHA MOVE: '
            'completed artifacts older than1800s only; every900s, target cpuQ8GiB, '
            'V2V launch threshold6GiB; gpuR4GiB hard floor checked before each batch. '
            'Initial conservative scope: previously authorised final LS attempts/pending '
            'and completed original synergy artifacts (isolated report also complete), '
            'plus completed TRAIN job observables whose arm update is published; A '
            'observables only after BOTH B/R same-update receipts, all older than30min. '
            'No final detector inputs in initial scope. RESULT/checkpoints/bank/live '
            'receipts/scenario/forbidden files stay. All direct public binding references '
            'and openFDs excluded; admitted bank/lesson dependencies protected. Per-batch '
            'SHA manifest before rsync, gpuR sha256sum proof durable BEFORE source '
            'SHA/inode/mtime/age/FD/reference recheck and exact source unlink. '
            'Destination gpuR:/root/cpuq_archive_20261103/rolling1647. '
            'nohup-equivalent independent cpuQ owner/flock; forwarded existing Windows '
            'agent transport, no private key copied. Transport loss pauses transfers with '
            'ALERT/sources retained and automatic reconnect; never affects training. '
            'No simulation/inference/budget allocation change. Qualified source hashes '
            'and each batch manifest SHA recorded. Existing archive completion adopted.\n')
        write(record,dict(line=line,cpu=cpu,source_hashes=cpu['source_hashes'],new_sim_s=0))
    h=sha(record);v=read(record);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(v['line']+'amendmentSHA'+h+'\n')
    print(remote('cpuQ',f'''from pathlib import Path
import json
p=Path({ROOT!r})/'AUTHORITY.json';v=json.loads({json.dumps(dict(v,amendment_sha256=h))!r})
if p.exists():assert json.loads(p.read_text())==v
else:
 with p.open('x') as f:json.dump(v,f)
print(json.dumps(dict(registered=True,sha256={h!r},new_sim_s=0)))''',timeout=40))


def launch():
    import subprocess,sys,os,psutil
    p=ops.EVIDENCE/'ROLLING_ARCHIVE_BRIDGE_DISPATCH_1647.json'
    if p.exists():
        d=read(p)
        try:q=psutil.Process(d['pid']);alive=q.create_time()==d['create_time']
        except psutil.NoSuchProcess:alive=False
        print(json.dumps(dict(adopt=True,alive=alive,owner=d)));return
    assert (ops.EVIDENCE/'ROLLING_ARCHIVE_REGISTRATION_1647.json').exists()
    command=[sys.executable,'-X','utf8','-m','tools.coopmwm.rolling_archive_ops','bridge']
    with (ops.EVIDENCE/'ROLLING_ARCHIVE_BRIDGE_1647.log').open('x',encoding='utf-8') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=(subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0,
            start_new_session=os.name!='nt')
    d=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),unix=time.time(),argv=command)
    write(p,d);print(json.dumps(d))


def bridge():
    import subprocess,base64,shlex
    # Only transport is local. The remote storage owner is detached and durable.
    code=f'''from pathlib import Path
import os,json,subprocess,time
r=Path({ROOT!r});assert r.is_dir();assert (r/'AUTHORITY.json').is_file()
sock=os.environ['SSH_AUTH_SOCK'];assert Path(sock).exists()
p=r/'TRANSPORT.json';tmp=r/'TRANSPORT.tmp'
with tmp.open('w') as f:json.dump(dict(socket=sock,unix=time.time()),f)
os.chmod(tmp,0o644);tmp.replace(p)
owner=r/'OWNER.json'
if owner.exists():
 d=json.loads(owner.read_text());proc=Path('/proc',str(d['pid']))
 assert proc.exists() and (proc/'stat').read_text().split()[21]==d['start_ticks'] and (proc/'stat').read_text().split()[2]!='Z','rolling owner absent; inspect before recovery'
else:
 dispatch=r/'DISPATCH.json'
 if not dispatch.exists():
  with (r/'RUN.log').open('x') as log:
   child=subprocess.Popen(['sudo','-n','/home/ubuntu/miniconda3/bin/python','-B','-u',str(r/'rolling_archive.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  with dispatch.open('x') as f:json.dump(dict(pid=child.pid,unix=time.time()),f)
print('ROLLING_TRANSPORT_READY',flush=True)
while True:time.sleep(60)
'''
    encoded=base64.b64encode(code.encode()).decode()
    command=['ssh','-A','-T','-o','BatchMode=yes','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=8','cpuQ',
        '/home/ubuntu/miniconda3/bin/python -u -c '+shlex.quote("import base64;exec(compile(base64.b64decode('"+encoded+"'),'<rolling-transport>','exec'))")]
    while True:
        p=subprocess.run(command)
        print(json.dumps(dict(unix=time.time(),transport_exit=p.returncode,retry_in_s=30)),flush=True)
        time.sleep(30)


def status():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time,shutil
r=Path({ROOT!r});out=dict(unix=time.time(),cpu_free=shutil.disk_usage('/home/ubuntu').free)
for name in ('OWNER','STATUS','PROGRESS','ALERT','STOP_GPU_SPACE'):
 p=r/(name+'.json');out[name]=json.loads(p.read_text()) if p.exists() else None
d=out['OWNER']
if d:
 p=Path('/proc',str(d['pid']));s=(p/'stat').read_text().split() if p.exists() else None
 out['alive']=bool(s and s[21]==d['start_ticks'] and s[2]!='Z')
out['completed_batches']=len(list(r.glob('batch_*/ARCHIVE_MOVE.json')))
print(json.dumps(out))''',timeout=40))
    write(ops.EVIDENCE/f'ROLLING_ARCHIVE_STATUS_{time.time_ns()}.json',value)
    print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('inventory','qualify','register','launch','bridge','status'))
    globals()[p.parse_args().action]()
