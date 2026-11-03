"""Operational entry points for MAIN16:47 rolling storage admission."""
import argparse
import json
import time
from . import ops
from .ls_validity_ops import remote
from .protocol import checked, sha, REGISTRY
from .storage import read, write

ROOT='/home/ubuntu/coopmwm20261103_rolling1647_v3'


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
    p=ops.EVIDENCE/'ROLLING_ARCHIVE_CPU_1647_V3.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def register():
    cpu=read(ops.EVIDENCE/'ROLLING_ARCHIVE_CPU_1647_V3.json');assert cpu['pass_']
    record=ops.EVIDENCE/'ROLLING_ARCHIVE_REGISTRATION_1647_V3.json'
    if not record.exists():
        line=('\nS-COOPMWM MAIN2026-11-03 16:27/16:47 DURABLE ROLLING SHA MOVE: '
            'completed artifacts older than1800s only; every900s, target cpuQ8GiB, '
            'V2V launch threshold6GiB; gpuR4GiB hard floor checked before each batch. '
            'Initial conservative scope: previously authorised final LS attempts/pending '
            'and completed original synergy artifacts (isolated report also complete), '
            'plus completed TRAIN job observables whose arm update is published; A '
            'observables only after BOTH B/R same-update receipts throughu24; A25..48 '
            'after their own completed update (B/R end24), all older than30min. '
            'V3 additionally includes completed ISOLATED_V2 diagnostic branch artifacts '
            'after the exact report status=complete and30min; root scientific reports stay; '
            'No final detector inputs in initial scope. Episode-job RESULT/checkpoints/bank/live '
            'receipts/scenario/forbidden files stay. All direct public binding references '
            'and openFDs excluded; admitted bank/lesson dependencies protected. Per-batch '
            'SHA manifest before rsync, gpuR sha256sum proof durable BEFORE source '
            'SHA/inode/mtime/age/FD/reference recheck and exact source unlink. '
            'Destination gpuR:/root/cpuq_archive_20261103/rolling1647_v3. '
            'nohup-equivalent independent cpuQ owner/flock; forwarded existing Windows '
            'agent transport, no private key copied. Transport loss pauses transfers with '
            'ALERT/sources retained and automatic reconnect; never affects training. '
            'No simulation/inference/budget allocation change. Qualified source hashes '
            'and each batch manifest SHA recorded. Existing archive completion adopted. '
            'V1 mover must be identity-verified idle/nochildren and retired before V3; '
            'all V1 batches/receipts remain immutable, no repeated moved sources.\n')
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
    p=ops.EVIDENCE/'ROLLING_ARCHIVE_BRIDGE_DISPATCH_1647_V3.json'
    if p.exists():
        d=read(p)
        try:q=psutil.Process(d['pid']);alive=q.create_time()==d['create_time']
        except psutil.NoSuchProcess:alive=False
        print(json.dumps(dict(adopt=True,alive=alive,owner=d)));return
    assert (ops.EVIDENCE/'ROLLING_ARCHIVE_REGISTRATION_1647_V3.json').exists()
    command=[sys.executable,'-X','utf8','-m','tools.coopmwm.rolling_archive_ops','bridge']
    with (ops.EVIDENCE/'ROLLING_ARCHIVE_BRIDGE_1647_V3.log').open('x',encoding='utf-8') as log:
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


def preview():
    print(remote('cpuQ',f'''import subprocess
code="""import sys,json,shutil
from pathlib import Path
sys.path.insert(0,{ROOT!r})
import rolling_archive as m
found,skipped=m.candidates();groups={{}}
for name,category in found.items():
 row=groups.setdefault(category,dict(files=0,bytes=0));row['files']+=1;row['bytes']+=Path(name).stat().st_size
old=Path('/home/ubuntu/coopmwm20261103_rolling1647');out={{}}
for n in ('OWNER','PROGRESS','ALERT','STATUS'):
 p=old/(n+'.json');out[n]=json.loads(p.read_text()) if p.exists() else None
d=out['OWNER']
if d:
 p=Path('/proc',str(d['pid']));out['wchan']=(p/'wchan').read_text() if p.exists() else 'ABSENT'
print(json.dumps(dict(eligible=groups,skipped=skipped,prior=out,cpu_free=shutil.disk_usage('/home/ubuntu').free)))"""
p=subprocess.run(['sudo','-n','/home/ubuntu/miniconda3/bin/python','-B','-c',code],capture_output=True,text=True,timeout=60)
if p.returncode:raise RuntimeError(p.stderr)
print(p.stdout)''',timeout=75))


def audit_training():
    print(remote('cpuQ', '''from pathlib import Path
import json,time
r=Path('/home/ubuntu/coopmwm20261102_warm1');rows=[]
for u in (10,25,48):
 ps=list((r/'jobs').glob('warm_u'+str(u).zfill(3)+'_A_*/observable.jsonl.gz'))
 rows.append(dict(u=u,count=len(ps),examples=[p.parent.name for p in ps[:2]],
  update_exists=(r/('A_UPDATE_'+str(u).zfill(3)+'.json')).exists(),
  close_exists=[(p.parent/'CLOSE.json').exists() for p in ps[:2]],
  resolved=[str(p.resolve()) for p in ps[:2]],age_update=(time.time()-(r/('A_UPDATE_'+str(u).zfill(3)+'.json')).stat().st_mtime)))
print(json.dumps(rows))''',timeout=30))


def audit_diagnostics():
    print(remote('cpuQ',f'''import subprocess
code="""import os,sys,json
from pathlib import Path
sys.path.insert(0,{ROOT!r});import rolling_archive as m
rows={{}}
for case in (m.base.RUN/'policy_synergy_isolated_1226_v2').glob('u[0-9][0-9][0-9]_[BR]'):
 report=m.base.RUN/('POLICY_SYNERGY_'+case.name+'_ISOLATED_V2.json')
 if not report.exists() or json.loads(report.read_text()).get('status')!='complete':continue
 for directory,dirs,names in os.walk(case/'branches',followlinks=False):
  dirs[:]=[n for n in dirs if not m.base.BAD.search(str(Path(directory)/n)) and not (Path(directory)/n).is_symlink()]
  for name in names:
   p=Path(directory)/name
   if m.base.BAD.search(str(p)) or p.is_symlink():continue
   k='xml_or_cfg' if name.endswith(('.xml','.xml.gz','.sumocfg','.sumocfg.gz')) else name if name in ('SNAPSHOT.pt','observable.jsonl.gz') else p.suffix
   r=rows.setdefault(k,[0,0]);r[0]+=1;r[1]+=p.stat().st_size
print(json.dumps(rows))"""
p=subprocess.run(['sudo','-n','/home/ubuntu/miniconda3/bin/python','-B','-c',code],capture_output=True,text=True,timeout=60)
if p.returncode:raise RuntimeError(p.stderr)
print(p.stdout)''',timeout=75))


def drain_prior():
    value=json.loads(remote('cpuQ',f'''import subprocess
code="""from pathlib import Path
import json,os,signal,time
old=Path('/home/ubuntu/coopmwm20261103_rolling1647');new=Path({ROOT!r})
assert (new/'AUTHORITY.json').is_file()
d=json.loads((old/'OWNER.json').read_text());proc=Path('/proc',str(d['pid']))
assert d['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
if proc.exists():
 s=(proc/'stat').read_text().split();assert s[21]==d['start_ticks']
 if s[2]!='Z':
  assert (proc/'wchan').read_text().strip() in ('hrtimer_nanosleep','do_nanosleep','do_select'),'prior mover not idle'
  assert not (proc/'task'/str(d['pid'])/'children').read_text().strip(),'active transfer children'
  assert all((p/'ARCHIVE_MOVE.json').exists() or (p/'CAPACITY_BLOCKED.json').exists() for p in old.glob('batch_*')),'unfinished prior batch'
  assert str(old/'rolling_archive.py') in (proc/'cmdline').read_bytes().decode().replace(chr(0),' ')
  os.kill(d['pid'],signal.SIGTERM)
  for _ in range(50):
   if not proc.exists() or (proc/'stat').read_text().split()[2]=='Z':break
   time.sleep(.1)
  assert not proc.exists() or (proc/'stat').read_text().split()[2]=='Z'
p=new/'PRIOR_DRAINED.json'
v=dict(owner=d,unix=time.time(),idle_verified=True,children=[],new_sim_s=0)
if not p.exists():
 with p.open('x') as f:json.dump(v,f)
print(p.read_text())"""
p=subprocess.run(['sudo','-n','/home/ubuntu/miniconda3/bin/python','-B','-c',code],capture_output=True,text=True,timeout=20)
if p.returncode:raise RuntimeError(p.stderr)
print(p.stdout)''',timeout=40))
    import psutil
    d=read(ops.EVIDENCE/'ROLLING_ARCHIVE_BRIDGE_DISPATCH_1647.json')
    try:
        proc=psutil.Process(d['pid']);assert proc.create_time()==d['create_time']
        assert 'tools.coopmwm.rolling_archive_ops' in proc.cmdline()
        children=proc.children(recursive=True)
        for child in children:assert child.name().lower() in ('ssh.exe','ssh','conhost.exe')
        proc.terminate()
        for child in children:
            if child.name().lower() in ('ssh.exe','ssh'):child.terminate()
    except psutil.NoSuchProcess:pass
    p=ops.EVIDENCE/'ROLLING_ARCHIVE_PRIOR_DRAINED_1647_V3.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('inventory','qualify','register','launch','bridge','status','preview','drain_prior','audit_training','audit_diagnostics'))
    globals()[p.parse_args().action]()
