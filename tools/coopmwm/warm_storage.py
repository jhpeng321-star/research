"""Prospective nondestructive RAM outputs + durable allowed-artifact backup.

Existing job directories are never moved or deleted. Only new, unclaimed job
paths use the owned RAM root. Backups open an explicit observable/receipt list;
scenario contents are never traversed, read or archived.
"""
import argparse
import base64
import json
import subprocess
import time
from . import ops
from .storage import write
from .protocol import checked,REGISTRY,sha
from .warm_deploy import ROOT
from .warm_protocol import worlds,batch_worlds

RAM='/dev/shm/coopmwm20261102_warm1'


def prepare():
    target=ops.EVIDENCE/'WARM_RAM_REGISTRATION.json';assert not target.exists()
    jobs=[]
    for arm,end in (('A',48),('B',48),('R',24)):
        for u in range(5 if arm=='A' else 1,end+1):
            jobs.extend(f'warm_u{u:03}_{arm}_{w["seed"]}' for w in batch_worlds(u))
            if u%4==0:jobs.extend(f'warmcheck{u:03}_{arm}_{arm}_{w["seed"]}' for w in worlds('check'))
            if u%16==0:jobs.extend(f'warmdev{u:03}_{arm}_{arm}_{w["seed"]}' for w in worlds('dev'))
    amendment=dict(root=ROOT,ram=RAM,job_ids=jobs,existing='leave all existing paths/reservations in place',
        backup='explicit RESULT.json/RESULT.json.gz/observable.jsonl.gz/CLAIM.json/CLOSE.json only; SHA-verified GPU A copy; no scenario reads',
        deletion=False,numerical_change=False,new_sumo_sim_s=0)
    write(ops.EVIDENCE/'WARM_RAM_AMENDMENT.json',amendment);h=sha(ops.EVIDENCE/'WARM_RAM_AMENDMENT.json')
    line=('\nS-COOPMWM PROSPECTIVE NONDESTRUCTIVE STORAGE REGISTERED amendmentSHA'+h+
        ': future unclaimed warm1 job paths use owned /dev/shm/coopmwm20261102_warm1 after >=16GiB free admission; '
        'all existing paths/claims/charges remain in place, no moves/deletions. '
        'Closed observable/RESULT/CLAIM/CLOSE artifacts copied to GPU A, source/destination SHA verified; '
        'no scenario/truth/assignment contents traversed or opened. Source/policy/budget unchanged; newSUMO0.\n')
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(target,dict(amendment_sha256=h,line=line))
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,shutil,time
r=pathlib.Path({ROOT!r});ram=pathlib.Path({RAM!r});assert r.is_dir() and ram.parent.is_dir()
assert not ram.exists() and shutil.disk_usage(ram.parent).free>=16*1024**3
ram.mkdir();created=[];preserved=[]
for name in json.loads({json.dumps(jobs)!r}):
 p=r/'jobs'/name
 if p.exists() or p.is_symlink():preserved.append(name);continue
 q=ram/name;q.mkdir();p.symlink_to(q,target_is_directory=True);created.append(name)
v=dict(unix=time.time(),ram=str(ram),created=created,preserved=preserved,
 ram_free=shutil.disk_usage(ram).free,disk_free=shutil.disk_usage(r).free,new_sumo_sim_s=0)
with (r/'RAM_STORAGE.json').open('x') as f:json.dump(v,f)
print(json.dumps(v))''',timeout=45))
    write(ops.EVIDENCE/'WARM_RAM_ADMISSION.json',value)
    print(json.dumps(dict(created=len(value['created']),preserved=len(value['preserved']),ram_free=value['ram_free'],disk_free=value['disk_free'])))


def backup(arm,update,phase='train'):
    """Copies completed allowed artifacts only, without deleting RAM or disk data."""
    assert phase in ('train','check','dev')
    suffix='' if phase=='train' else phase.upper()+'_'
    dest=ops.EVIDENCE/f'WARM_BACKUP_{arm}_{suffix}{update:03}.json';assert not dest.exists()
    pattern=f'warm_u{update:03}_{arm}_*' if phase=='train' else f'warm{phase}{update:03}_{arm}_{arm}_*'
    archive=f'{arm}_u{update:03}.tar' if phase=='train' else f'{arm}_{phase}_{update:03}.tar'
    source=f'''import pathlib,subprocess,hashlib,json,sys
r=pathlib.Path({ROOT!r});ram=pathlib.Path({RAM!r});assert r.is_dir() and ram.is_dir()
names=[]
for p in (r/'jobs').glob({pattern!r}):
 if not p.is_symlink():continue
 assert p.resolve().parent==ram
 assert (p/'CLOSE.json').is_file() and (p/'RESULT.json').is_file()
 for name in ('RESULT.json','RESULT.json.gz','observable.jsonl.gz','CLAIM.json','CLOSE.json'):
  if (p/name).is_file():names.append(p.name+'/'+name)
assert names
p=subprocess.Popen(['tar','-C',str(ram),'-cf','-',*sorted(names)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
h=hashlib.sha256();n=0
for block in iter(lambda:p.stdout.read(1024*1024),b''):
 h.update(block);n+=len(block);sys.stdout.buffer.write(block)
sys.stdout.buffer.flush();err=p.stderr.read();assert p.wait()==0,err
print(json.dumps(dict(sha256=h.hexdigest(),bytes=n,files=len(names))),file=sys.stderr)
'''
    destination=f'''import pathlib,sys,hashlib,json,shutil
parent=pathlib.Path('/root/autodl-tmp');assert parent.is_dir() and shutil.disk_usage(parent).free>2*1024**3
d=parent/'coopmwm20261102_warm_receipts';d.mkdir(exist_ok=True)
p=d/{archive!r};assert not p.exists();h=hashlib.sha256();n=0
with p.open('xb') as f:
 for b in iter(lambda:sys.stdin.buffer.read(1024*1024),b''):f.write(b);h.update(b);n+=len(b)
h2=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1024*1024),b''):h2.update(b)
assert h.hexdigest()==h2.hexdigest()
print(json.dumps(dict(path=str(p),sha256=h2.hexdigest(),bytes=n)))'''
    def command(host,code):
        payload=base64.b64encode(code.encode()).decode()
        return ['ssh','-o','BatchMode=yes',host,ops.ssh.PYTHON[host]+f' -c "import base64;exec(base64.b64decode(\'{payload}\'))"']
    a=subprocess.Popen(command('cpuQ',source),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    b=subprocess.Popen(command('gpuA',destination),stdin=a.stdout,stdout=subprocess.PIPE,stderr=subprocess.PIPE);a.stdout.close()
    out,err=b.communicate();aerr=a.stderr.read();arc=a.wait()
    assert arc==0 and b.returncode==0,(arc,b.returncode,aerr.decode(),err.decode())
    src=json.loads(aerr.decode().strip().splitlines()[-1]);dst=json.loads(out.decode().strip().splitlines()[-1])
    assert src['sha256']==dst['sha256'] and src['bytes']==dst['bytes']
    write(dest,dict(unix=time.time(),source=src,destination=dst,verified=True,deleted=False,new_sumo_sim_s=0))
    print(json.dumps(dict(arm=arm,update=update,phase=phase,bytes=src['bytes'],sha256=src['sha256'],verified=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','backup'));p.add_argument('--arm',choices=('A','B','R'));p.add_argument('--update',type=int);p.add_argument('--phase',choices=('train','check','dev'),default='train')
    a=p.parse_args();prepare() if a.mode=='prepare' else backup(a.arm,a.update,a.phase)
