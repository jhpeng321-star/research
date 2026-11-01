"""Explicit OWNER20:35 recovery operations; no physics dispatch or peer signals."""
import argparse
import json
import subprocess
import sys
import time
from . import ops
from .storage import write


def inventory():
    from .storage import read
    if (ops.EVIDENCE/'ARCHIVE2035_INVENTORY.json').exists():
        value=read(ops.EVIDENCE/'ARCHIVE2035_INVENTORY.json')
        print(json.dumps(dict(value,blocked=len(value['blocked']),links=len(value['links']))));return
    value=json.loads(ops.ssh.remote('cpuQ',r'''import pathlib,json,os,re,shutil,time
r=pathlib.Path('/home/ubuntu/coopmwm20261101_v2');assert r.is_dir()
d=json.loads((r/'A_DISPATCH_1854.json').read_text())
p=pathlib.Path('/proc')/str(d['pid']);s=(p/'stat').read_text().split() if (p/'stat').exists() else None
assert not s or (s[21]==d['start_ticks'] and s[2] in ('T','Z') and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id'])
forbidden=re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vehicle_params|driver_params|vtype',re.I)
blocked=[];files=0;size=0;links=[]
for parent,dirs,names in os.walk(r,followlinks=False):
 for n in dirs+names:
  p=pathlib.Path(parent)/n
  if forbidden.search(str(p)):blocked.append(str(p))
  elif p.is_symlink():links.append(str(p))
  elif p.is_file():files+=1;size+=p.stat().st_size
 dirs[:]=[n for n in dirs if not forbidden.search(str(pathlib.Path(parent)/n)) and not (pathlib.Path(parent)/n).is_symlink()]
active=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
  if str(r) in cmd:active.append(dict(pid=int(p.name),state=(p/'stat').read_text().split()[2],cmd=cmd))
 except (OSError,PermissionError):pass
print(json.dumps(dict(unix=time.time(),source=str(r),files=files,bytes=size,blocked=blocked,links=links,processes=active,disk_free=shutil.disk_usage(r).free,owner=d,new_sumo_sim_s=0)))
''',timeout=40))
    write(ops.EVIDENCE/'ARCHIVE2035_INVENTORY.json',value)
    print(json.dumps(dict(value,blocked=len(value['blocked']),links=len(value['links']))))


def archive():
    """Opaque backup transport only: never inspect/parse archived scenario contents."""
    from .storage import read
    inv=read(ops.EVIDENCE/'ARCHIVE2035_INVENTORY.json')
    assert not inv['links']
    assert all(p['state'] in ('T','Z') for p in inv['processes']),inv['processes']
    assert not (ops.EVIDENCE/'ARCHIVE2035_VERIFIED.json').exists()
    # Stream through the existing SSH aliases/keys; no extra 3GB copy on cpuQ.
    source=r'''import subprocess,sys,hashlib,json,pathlib
r=pathlib.Path('/home/ubuntu/coopmwm20261101_v2');assert r.is_dir()
d=json.loads((r/'A_DISPATCH_1854.json').read_text());p=pathlib.Path('/proc')/str(d['pid'])
s=(p/'stat').read_text().split();assert s[21]==d['start_ticks'] and s[2]=='T'
assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
p=subprocess.Popen(['tar','-C','/home/ubuntu','-cf','-','coopmwm20261101_v2'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
h=hashlib.sha256();n=0
while True:
 b=p.stdout.read(1024*1024)
 if not b:break
 h.update(b);n+=len(b);sys.stdout.buffer.write(b)
sys.stdout.buffer.flush();err=p.stderr.read();assert p.wait()==0,err
print(json.dumps(dict(sha256=h.hexdigest(),bytes=n)),file=sys.stderr)
'''
    destination=r'''import pathlib,hashlib,json,sys,shutil
parent=pathlib.Path('/root/autodl-tmp');assert parent.is_dir()
assert shutil.disk_usage(parent).free>4000000000
d=parent/'archive_cpuQ_20261101';d.mkdir(exist_ok=True)
p=d/'coopmwm20261101_v2.tar';assert not p.exists()
h=hashlib.sha256();n=0
with p.open('xb') as f:
 while True:
  b=sys.stdin.buffer.read(1024*1024)
  if not b:break
  f.write(b);h.update(b);n+=len(b)
# Independently rehash the written opaque archive, without extracting any file.
h2=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1024*1024),b''):h2.update(b)
assert h.hexdigest()==h2.hexdigest()
print(json.dumps(dict(path=str(p),sha256=h2.hexdigest(),bytes=n)))
'''
    import base64
    def command(host,code):
        payload=base64.b64encode(code.encode()).decode()
        return ['ssh','-o','BatchMode=yes',host,ops.ssh.PYTHON[host]+f' -c "import base64;exec(base64.b64decode(\'{payload}\'))"']
    a=subprocess.Popen(command('cpuQ',source),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    b=subprocess.Popen(command('gpuA',destination),stdin=a.stdout,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    a.stdout.close()
    out,err=b.communicate();aerr=a.stderr.read();arc=a.wait()
    assert arc==0 and b.returncode==0,(arc,b.returncode,aerr.decode(),err.decode())
    src=json.loads(aerr.decode().strip().splitlines()[-1]);dst=json.loads(out.decode().strip().splitlines()[-1])
    assert src['sha256']==dst['sha256'] and src['bytes']==dst['bytes']
    value=dict(unix=time.time(),source=inv['source'],source_stream=src,destination=dst,
        verified=True,transport='opaque tar stream via existing cpuQ/gpuA SSH keys; no extraction/content inspection',new_sumo_sim_s=0)
    write(ops.EVIDENCE/'ARCHIVE2035_VERIFIED.json',value);print(json.dumps(value))


def start_archive():
    pth=ops.EVIDENCE/'ARCHIVE2035_DISPATCH.json';assert not pth.exists()
    assert ops.EVIDENCE.is_dir()
    with (ops.EVIDENCE/'ARCHIVE2035.log').open('x') as log:
        p=subprocess.Popen([sys.executable,'-X','utf8','-m','tools.coopmwm.recovery2035','archive'],
            cwd=ops.LOCAL,stdout=log,stderr=subprocess.STDOUT)
    v=dict(pid=p.pid,unix=time.time(),purpose='opaque archive only; removal is a separate hash-gated operation')
    write(pth,v);print(json.dumps(v))


def remove_verified_source():
    from .storage import read
    v=read(ops.EVIDENCE/'ARCHIVE2035_VERIFIED.json');assert v['verified']
    # Validate the remote durable copy again BEFORE the explicitly approved rm.
    result=ops.ssh.remote('gpuA',f'''import pathlib,hashlib,json
p=pathlib.Path('/root/autodl-tmp/archive_cpuQ_20261101/coopmwm20261101_v2.tar');h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
assert h.hexdigest()=={v['source_stream']['sha256']!r}
print(json.dumps(dict(sha256=h.hexdigest(),bytes=p.stat().st_size)))
''',timeout=60)
    assert json.loads(result)['bytes']==v['source_stream']['bytes']
    out=json.loads(ops.ssh.remote('cpuQ',r'''import pathlib,json,subprocess,shutil,time
r=pathlib.Path('/home/ubuntu/coopmwm20261101_v2');assert r.is_dir()
d=json.loads((r/'A_DISPATCH_1854.json').read_text());p=pathlib.Path('/proc')/str(d['pid'])
s=(p/'stat').read_text().split();assert s[21]==d['start_ticks'] and s[2]=='T'
assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
# OWNER20:35 explicitly authorizes this single literal root after verified backup.
subprocess.run(['rm','-rf','--','/home/ubuntu/coopmwm20261101_v2'],check=True)
assert not r.exists()
print(json.dumps(dict(unix=time.time(),source_removed=True,disk_free=shutil.disk_usage('/home/ubuntu').free,new_sumo_sim_s=0)))
''',timeout=50))
    out['archive']=v;write(ops.EVIDENCE/'ARCHIVE2035_COMPLETE.json',out);print(json.dumps(out))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('inventory','archive','start-archive','remove-verified-source'));a=p.parse_args()
    {'inventory':inventory,'archive':archive,'start-archive':start_archive,'remove-verified-source':remove_verified_source}[a.mode]()
