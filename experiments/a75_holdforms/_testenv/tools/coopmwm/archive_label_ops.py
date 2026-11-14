"""Stage and observe one agent-forwarded archive relay; never duplicate a relay."""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import time
from . import ops
from .ls_validity_ops import remote
from .protocol import checked
from .storage import write, read

ROOT = '/home/ubuntu/coopmwm_archive_20261103_label_runner_copies'
DISPATCH = 'ARCHIVE_LABEL_SNAPSHOTS_RELAY_DISPATCH.json'


def stage():
    files = {}
    for name in ('archive_completed', 'archive_label_snapshots'):
        files['pytools/tools/coopmwm/'+name+'.py'] = checked(ops.LOCAL/f'tools/coopmwm/{name}.py').read_text(encoding='utf-8')
    files.update({'pytools/tools/__init__.py':'', 'pytools/tools/coopmwm/__init__.py':''})
    files['ENTRY.py'] = f'''import sys,os,json,time
from pathlib import Path
r=Path({ROOT!r});sys.path.insert(0,str(r/'pytools'))
from tools.coopmwm import archive_label_snapshots as a
a.configure()
if not (r/'ARCHIVE_MOVE_ls_label_runner_copies.json').exists():
 v=dict(pid=os.getpid(),start_ticks=Path('/proc',str(os.getpid()),'stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time())
 with (r/('OWNER_'+str(os.getpid())+'.json')).open('x') as f:json.dump(v,f)
 print(json.dumps(v),flush=True)
a.core.run()
'''
    code = f'''from pathlib import Path
import json,sys,hashlib
r=Path({ROOT!r});assert r.is_absolute() and r.parent.is_dir();r.mkdir(exist_ok=True)
files=json.loads({json.dumps(files)!r});hashes={{}}
for n,t in files.items():
 p=r/n;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_text()==t
 else:p.write_text(t)
 hashes[n]=hashlib.sha256(p.read_bytes()).hexdigest()
p=r/'SOURCE_HASHES.json'
if p.exists():assert json.loads(p.read_text())==hashes
else:
 with p.open('x') as f:json.dump(hashes,f,sort_keys=True)
sys.path.insert(0,str(r/'pytools'))
from tools.coopmwm.archive_label_snapshots import final_snapshot_paths
paths=sorted(p for p in final_snapshot_paths() if p.is_file())
print(json.dumps(dict(root=str(r),files=len(paths),bytes=sum(p.stat().st_size for p in paths),hashes=hashes,sim_s=0)))'''
    value = json.loads(remote('cpuQ', code, timeout=45))
    target = ops.EVIDENCE/'ARCHIVE_LABEL_RUNNER_COPIES_STAGE.json'
    if not target.exists(): write(target, value)
    print(json.dumps(value))


def relay():
    import psutil
    target = ops.EVIDENCE/DISPATCH
    if target.exists(): print(json.dumps(read(target))); return
    assert (ops.EVIDENCE/'ARCHIVE_LABEL_RUNNER_COPIES_STAGE.json').exists()
    assert ops.EVIDENCE.is_dir()
    with (ops.EVIDENCE/'ARCHIVE_LABEL_SNAPSHOTS_RELAY.claim').open('x') as claim:
        claim.write(str(time.time()))
    log = ops.EVIDENCE/'ARCHIVE_LABEL_SNAPSHOTS_RELAY.log'
    with log.open('x', encoding='utf-8') as f:
        p = subprocess.Popen(['ssh','-A','-o','BatchMode=yes','cpuQ', 'sudo','-n',
            '--preserve-env=SSH_AUTH_SOCK','/home/ubuntu/miniconda3/bin/python','-B','-u',ROOT+'/ENTRY.py'],
            stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NEW_PROCESS_GROUP',0))
    v = dict(pid=p.pid,create_time=psutil.Process(p.pid).create_time(),unix=time.time(),
        remote_root=ROOT,agent_forwarding=True,log=str(log),sim_s=0)
    write(target,v);print(json.dumps(v))


def status():
    value = json.loads(remote('cpuQ', f'''from pathlib import Path
import json,shutil,time
r=Path({ROOT!r});receipt=r/'ARCHIVE_MOVE_ls_label_runner_copies.json'
owners=[]
for p in r.glob('OWNER_*.json'):
 d=json.loads(p.read_text());proc=Path('/proc',str(d['pid']));alive=False
 if proc.exists():
  st=(proc/'stat').read_text().split();alive=st[21]==d['start_ticks'] and st[2]!='Z'
 owners.append(dict(owner=d,alive=alive))
print(json.dumps(dict(unix=time.time(),owners=owners,receipt=json.loads(receipt.read_text()) if receipt.exists() else None,disk_free_bytes=shutil.disk_usage('/home/ubuntu').free)))''', timeout=35))
    write(ops.EVIDENCE/f'ARCHIVE_LABEL_SNAPSHOTS_STATUS_{time.time_ns()}.json', value)
    if value['receipt']:
        p=ops.EVIDENCE/'ARCHIVE_MOVE_ls_label_runner_copies.json'
        if not p.exists(): write(p,value['receipt'])
    log=ops.EVIDENCE/'ARCHIVE_LABEL_SNAPSHOTS_RELAY.log'
    if log.exists():value['relay_log']=log.read_text(encoding='utf-8',errors='replace')[-4000:]
    print(json.dumps(value))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('stage','relay','status'))
    globals()[p.parse_args().action]()
