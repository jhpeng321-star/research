"""MAIN23:18 second archive destination; one source mover, immutable v4 control."""
import argparse
import base64
import json
import time
from . import ops
from .ls_validity_ops import remote
from .protocol import checked, sha, REGISTRY
from .storage import read, write

ROOT='/home/ubuntu/coopmwm20261103_rolling2318_gpua_v4'
PRIOR='/home/ubuntu/coopmwm20261103_rolling1647_v3'
CPU='ROLLING_ARCHIVE_CPU_GPUA_V4.json'
REG='ROLLING_ARCHIVE_REGISTRATION_GPUA_V4.json'
DISPATCH='ROLLING_ARCHIVE_BRIDGE_GPUA_V4B.json'
ROUTE=['ssh','-T','-o','BatchMode=yes','-o','IdentityFile=none','-o','IdentitiesOnly=no',
    '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/home/ubuntu/coopmwm20261102_remote_routes5/known_hosts_gpuA',
    '-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=8',
    '-p','11000','root@connect.bjb1.seetacloud.com']
TARGET=dict(host='gpuA',destination='/root/cpuq_archive_20261103/rolling2318_gpua_v4',
    floor_bytes=8*1024**3,cpu_target_bytes=12*1024**3,ssh_command=ROUTE)


def qualify():
    names=('rolling_archive.py','archive_move.py','test_rolling_archive_cpu.py')
    files={n:checked(ops.LOCAL/'tools/coopmwm'/n).read_text(encoding='utf-8') for n in names}
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,hashlib,subprocess,time
r=Path({ROOT!r});assert r.parent.is_dir();r.mkdir(exist_ok=True)
for n,text in json.loads({json.dumps(files)!r}).items():
 p=r/n
 if p.exists():assert p.read_text()==text,n
 else:
  with p.open('x') as f:f.write(text)
p=r/'CPU_RESULT.json'
if not p.exists():
 start=time.time();x=subprocess.run(['sudo','-n','/home/ubuntu/miniconda3/bin/python','-B','-m','unittest','discover','-s',str(r),'-p','test_rolling_archive_cpu.py','-v'],capture_output=True,text=True,timeout=45)
 v=dict(pass_=x.returncode==0,stdout=x.stdout,stderr=x.stderr,wall_s=time.time()-start,new_sim_s=0,source_hashes={{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in {names!r}}})
 with p.open('x') as f:json.dump(v,f)
print(p.read_text())''',timeout=60))
    p=ops.EVIDENCE/CPU
    if p.exists():assert read(p)==value
    else:write(p,value)
    print(json.dumps(value))
    if not value['pass_']:raise ValueError('storage CPU qualification failed')


def register():
    cpu=read(ops.EVIDENCE/CPU);assert cpu['pass_']
    record=ops.EVIDENCE/REG
    if not record.exists():
        line=('\nS-COOPMWM MAIN2026-11-03 23:18 SHA-ROLLING ARCHIVE GPUA EXTENSION: '
            'gpuA second namespace /root/cpuq_archive_20261103/rolling2318_gpua_v4; '
            'gpuA floor8GiB before/after complete batches; cpuQ maintenance12GiB. '
            'Single source mover: identity-proven idle/nochildren prior v3 retired '
            'before v4 start; gpuR archives/prior receipts retained. Same completed '
            '>1800s scope/reference/FD/sourceSHA+inode+mtime gates, no scope expansion; '
            'per-file manifest, rsync, remote sha256sum, durable proof before unlink. '
            'No private keys copied; existing forwarded agent/strict known-host route. '
            'Transport faults retain sources. No SUMO/inference/budget change.\n')
        write(record,dict(line=line,archive_target=TARGET,source_hashes=cpu['source_hashes'],cpu=cpu,new_sim_s=0))
    h=sha(record);v=read(record);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(v['line']+'amendmentSHA'+h+'\n')
    print(remote('cpuQ',f'''from pathlib import Path
import json
r=Path({ROOT!r});assert r.is_dir();p=r/'AUTHORITY.json'
v=json.loads({json.dumps(dict(v,amendment_sha256=h))!r})
if p.exists():assert json.loads(p.read_text())==v
else:
 with p.open('x') as f:json.dump(v,f)
print(json.dumps(dict(registered=True,sha256={h!r},new_sim_s=0)))''',timeout=35))
    print(remote('gpuA', '''from pathlib import Path
import json,shutil
p=Path('/root/cpuq_archive_20261103');assert p.parent.is_dir();p.mkdir(exist_ok=True)
assert not p.is_symlink()
print(json.dumps(dict(archive_parent=str(p),free=shutil.disk_usage(p).free,new_sim_s=0)))''',timeout=30))


def route():
    """Verify the existing forwarded-agent route before retiring its old mover."""
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import os,json,subprocess
p=Path({PRIOR!r})/'TRANSPORT.json';socket=json.loads(p.read_text())['socket'];assert Path(socket).exists()
code="from pathlib import Path;import json,shutil;p=Path('/root/cpuq_archive_20261103');assert p.is_dir();print(json.dumps(dict(free=shutil.disk_usage(p).free)))"
import shlex
x=subprocess.run({ROUTE!r}+['/root/miniconda3/bin/python -c '+shlex.quote(code)],env=dict(os.environ,SSH_AUTH_SOCK=socket),capture_output=True,text=True,timeout=35)
if x.returncode:raise RuntimeError(x.stderr)
v=json.loads(x.stdout);assert v['free']>={TARGET['floor_bytes']!r}
print(json.dumps(dict(pass_=True,target='gpuA',free=v['free'],floor={TARGET['floor_bytes']!r},strict_known_host=True,new_sim_s=0)))''',timeout=50))
    p=ops.EVIDENCE/'ROLLING_ARCHIVE_ROUTE_GPUA_V4.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def drain():
    """Refuse during any old transfer; a later call may adopt its idle boundary."""
    assert read(ops.EVIDENCE/'ROLLING_ARCHIVE_ROUTE_GPUA_V4.json')['pass_']
    value=json.loads(remote('cpuQ',f'''import subprocess,json
code="""from pathlib import Path
import json,os,signal,time,hashlib
r=Path({PRIOR!r});new=Path({ROOT!r});assert (new/'AUTHORITY.json').is_file()
assert (new/'HANDOVER_BOUNDARY_V4B.json').is_file(),'safe-boundary amendment missing'
d=json.loads((r/'OWNER.json').read_text());p=Path('/proc',str(d['pid']))
assert d['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
v=dict(owner=d,unix=time.time(),new_sim_s=0,prior_root=str(r),source_receipts_preserved=True)
if p.exists():
 s=(p/'stat').read_text().split();assert s[21]==d['start_ticks']
 if s[2]!='Z':
  assert str(r/'rolling_archive.py') in (p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
  assert not (p/'task'/str(d['pid'])/'children').read_text().strip(),'prior archive transfer still active'
  os.kill(d['pid'],signal.SIGSTOP)
  try:
   for _ in range(50):
    if (p/'stat').read_text().split()[2]=='T':break
    time.sleep(.01)
   assert (p/'stat').read_text().split()[2]=='T','prior did not acknowledge stop'
   assert not (p/'task'/str(d['pid'])/'children').read_text().strip(),'prior acquired a child before stop'
   folders=sorted(r.glob('batch_*'))
   last=folders[-1] if folders else None
   completed=last is None or (last/'ARCHIVE_MOVE.json').exists() or (last/'CAPACITY_BLOCKED.json').exists()
   pre_proof=last is not None and not (last/'REMOTE_VERIFIED.json').exists()
   assert completed or pre_proof,'prior mover finishing SHA-certified removals'
   v['boundary']='completed_batch' if completed else 'source_hashing_or_pre_verification; no current-batch deletion'
  except BaseException:
   os.kill(d['pid'],signal.SIGCONT);raise
  os.kill(d['pid'],signal.SIGTERM);os.kill(d['pid'],signal.SIGCONT)
  for _ in range(50):
   if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
   time.sleep(.1)
  assert not p.exists() or (p/'stat').read_text().split()[2]=='Z'
v['idle_identity_verified']=True
out=new/'PRIOR_DRAINED.json'
if not out.exists():
 with out.open('x') as f:json.dump(v,f)
print(out.read_text())"""
x=subprocess.run(['sudo','-n','/home/ubuntu/miniconda3/bin/python','-B','-c',code],capture_output=True,text=True,timeout=15)
if x.returncode:raise RuntimeError(x.stderr)
print(x.stdout)''',timeout=35))
    # Retire only the old transport owned by this task; it cannot restart v3.
    import psutil
    p=ops.EVIDENCE/'ROLLING_ARCHIVE_BRIDGE_DISPATCH_1647_V3.json'
    if p.exists():
        d=read(p)
        try:
            proc=psutil.Process(d['pid']);assert proc.create_time()==d['create_time']
            assert 'tools.coopmwm.rolling_archive_ops' in proc.cmdline()
            children=proc.children(recursive=True)
            for child in children:assert child.name().lower() in ('ssh.exe','ssh','conhost.exe')
            proc.terminate()
            for child in children:
                if child.name().lower() in ('ssh.exe','ssh'):child.terminate()
        except psutil.NoSuchProcess:pass
    p=ops.EVIDENCE/'ROLLING_ARCHIVE_PRIOR_DRAINED_GPUA_V4.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def launch():
    import subprocess,sys,os,psutil
    p=ops.EVIDENCE/DISPATCH
    if p.exists():
        d=read(p)
        try:alive=psutil.Process(d['pid']).create_time()==d['create_time']
        except psutil.NoSuchProcess:alive=False
        print(json.dumps(dict(adopt=True,alive=alive,owner=d)));return
    assert read(ops.EVIDENCE/'ROLLING_ARCHIVE_ROUTE_GPUA_V4.json')['pass_']
    assert (ops.EVIDENCE/REG).is_file()
    assert (ops.EVIDENCE/'ROLLING_ARCHIVE_WAITER_RETIRED_V4B.json').is_file()
    command=[sys.executable,'-B','-X','utf8','-m','tools.coopmwm.rolling_gpua_ops','handover']
    with (ops.EVIDENCE/DISPATCH).with_suffix('.log').open('x',encoding='utf-8') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=(subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0,start_new_session=os.name!='nt')
    d=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),unix=time.time(),argv=command)
    write(p,d);print(json.dumps(d))


def handover():
    """Wait out an existing transfer without interrupting it or adding a mover."""
    proof=ops.EVIDENCE/'ROLLING_ARCHIVE_PRIOR_DRAINED_GPUA_V4.json'
    while not proof.exists():
        try:drain()
        except RuntimeError as error:
            if not any(x in str(error) for x in ('prior archive transfer still active','prior mover not idle','prior mover finishing SHA-certified removals')):raise
            print(json.dumps(dict(unix=time.time(),waiting_for_prior_idle=True,new_sim_s=0)),flush=True)
            time.sleep(30)
    bridge()


def boundary_register():
    """No source deletion or signal; qualify a safe pre-proof boundary as idle."""
    record=ops.EVIDENCE/'ROLLING_ARCHIVE_BOUNDARY_AMENDMENT_V4B.json'
    if not record.exists():write(record,dict(new_sim_s=0,original_amendment_sha256=sha(ops.EVIDENCE/REG),
        rule='One mover only. Old parent may be retired with no active children at a completed-batch OR source-hashing/pre-verification boundary (no REMOTE_VERIFIED proof, hence no current-batch deletion). SIGSTOP then recheck identity/state/children/latest proof; any incomplete proof-certified removal phase resumes untouched. Transfers are never interrupted. This prevents starvation while old cpuQ maintenance target is never reached.',
        source_sha256=sha(ops.LOCAL/'tools/coopmwm/rolling_gpua_ops.py')))
    h=sha(record);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write('\nS-COOPMWM archive safe-boundary engineering amendment '+read(record)['rule']+' amendmentSHA'+h+'\n')
    print(remote('cpuQ',f'''from pathlib import Path
import json
p=Path({ROOT!r})/'HANDOVER_BOUNDARY_V4B.json';v=json.loads({json.dumps(dict(read(record),amendment_sha256=h))!r})
if p.exists():assert json.loads(p.read_text())==v
else:
 with p.open('x') as f:json.dump(v,f)
print(json.dumps(dict(registered=True,sha256={h!r},new_sim_s=0)))''',timeout=35))


def retire_waiter():
    """Retire only the old local waiter while it has no SSH child/remote operation."""
    import psutil
    old=ops.EVIDENCE/'ROLLING_ARCHIVE_BRIDGE_GPUA_V4.json';d=read(old)
    proc=psutil.Process(d['pid']);assert proc.create_time()==d['create_time']
    assert 'tools.coopmwm.rolling_gpua_ops' in proc.cmdline()
    assert not proc.children(recursive=True),'old waiter has an active metadata/drain call'
    proc.suspend()
    try:
        assert not proc.children(recursive=True),'waiter acquired a child'
        print(remote('cpuQ',f'''from pathlib import Path
import json
r=Path({ROOT!r});assert not (r/'OWNER.json').exists() and not (r/'DISPATCH.json').exists(),'archive worker already started; adopt it'
print(json.dumps(dict(no_remote_worker=True,new_sim_s=0)))''',timeout=30))
        proc.terminate()
    except BaseException:
        proc.resume();raise
    proc.wait(timeout=5)
    value=dict(owner=d,unix=time.time(),idle_local_no_children=True,remote_worker_started=False,new_sim_s=0)
    p=ops.EVIDENCE/'ROLLING_ARCHIVE_WAITER_RETIRED_V4B.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def bridge():
    import subprocess,shlex
    code=f'''from pathlib import Path
import os,json,subprocess,time
r=Path({ROOT!r});assert r.is_dir();assert (r/'AUTHORITY.json').is_file();assert (r/'PRIOR_DRAINED.json').is_file()
sock=os.environ['SSH_AUTH_SOCK'];assert Path(sock).exists()
p=r/'TRANSPORT.json';tmp=r/'TRANSPORT.tmp'
with tmp.open('w') as f:json.dump(dict(socket=sock,unix=time.time()),f)
os.chmod(tmp,0o644);tmp.replace(p)
owner=r/'OWNER.json'
if owner.exists():
 d=json.loads(owner.read_text());proc=Path('/proc',str(d['pid']))
 assert proc.exists() and (proc/'stat').read_text().split()[21]==d['start_ticks'] and (proc/'stat').read_text().split()[2]!='Z','adopt absent archive owner: inspect before recovery'
elif not (r/'DISPATCH.json').exists():
 with (r/'RUN.log').open('x') as log:
  child=subprocess.Popen(['sudo','-n','env','COOPMWM_ARCHIVE_CONTROL='+str(r),'/home/ubuntu/miniconda3/bin/python','-B','-u',str(r/'rolling_archive.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 with (r/'DISPATCH.json').open('x') as f:json.dump(dict(pid=child.pid,unix=time.time()),f)
print('ROLLING_GPUA_TRANSPORT_READY',flush=True)
while True:time.sleep(60)
'''
    encoded=base64.b64encode(code.encode()).decode()
    command=['ssh','-A','-T','-o','BatchMode=yes','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=8','cpuQ',
        '/home/ubuntu/miniconda3/bin/python -u -c '+shlex.quote("import base64;exec(compile(base64.b64decode('"+encoded+"'),'<rolling-gpua-transport>','exec'))")]
    while True:
        p=subprocess.run(command)
        print(json.dumps(dict(unix=time.time(),transport_exit=p.returncode,retry_in_s=30)),flush=True)
        time.sleep(30)


def status():
    from . import rolling_archive_ops as parent
    parent.ROOT=ROOT
    parent.status()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('qualify','register','route','boundary_register','retire_waiter','drain','launch','handover','bridge','status'))
    globals()[p.parse_args().action]()
