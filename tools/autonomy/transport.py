"""Sequential explicit-file deployment; no laptop dependency after detached launch."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

from tools.loopresume.operate import remote

ROOT = Path(__file__).resolve().parents[2]
REGISTER = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
RELEASES = {'gpuA': '/root/autodl-tmp/c4/restart20261026',
            'gpuR': '/root/autodl-tmp/c4/coopoff220261027'}
FILES = ['remote.py', 'collector.py', 'memory_entry.py', 'cpu_preflight.py', 'spot_watch.py', 'gpu_mirror.py']


def stage(host):
    payload = {n: base64.b64encode((ROOT / 'tools/autonomy' / n).read_bytes()).decode() for n in FILES}
    remote(host, f'''
import pathlib,base64,hashlib,json
r=pathlib.Path({RELEASES[host]!r}); assert r.is_dir()
d=r/'autonomy'; d.mkdir(exist_ok=True)
manifest={{}}
for name,data in {payload!r}.items():
 p=d/name; decoded=base64.b64decode(data)
 if p.exists() and p.read_bytes()!=decoded:
  backup=d/(name+'.'+hashlib.sha256(p.read_bytes()).hexdigest()+'.before'); backup.write_bytes(p.read_bytes())
 p.write_bytes(decoded); manifest[name]=hashlib.sha256(decoded).hexdigest()
(r/'AUTONOMY_SOURCE.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest))
''')


def budget(host):
    sha = hashlib.sha256(REGISTER.read_bytes()).hexdigest()
    remote(host, f'''
import subprocess,sys
p=subprocess.run([sys.executable,'-B',{RELEASES[host] + '/autonomy/remote.py'!r},'budget',
 '--release',{RELEASES[host]!r},'--markdown-sha',{sha!r}],capture_output=True,text=True)
print(p.stdout); print(p.stderr); raise SystemExit(p.returncode)
''')


def launch(host):
    assert host == 'gpuA', 'Owner priority: coop-off waiter/relaunch prohibited'
    remote(host, f'''
import subprocess,sys,pathlib,json,os,time
r=pathlib.Path({RELEASES[host]!r}); assert not (r/'AUTONOMY_PID.json').exists()
with (r/'arm/driver_autonomy.log').open('xb') as stream:
 p=subprocess.Popen([sys.executable,'-B','-u',str(r/'autonomy/remote.py'),'drive','--release',str(r)],
  stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
  env={{**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}})
time.sleep(1)
fields=pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text().rsplit(')',1)[1].split()
assert fields[0]!='Z' and os.getsid(p.pid)==p.pid
doc={{'pid':p.pid,'start_ticks':fields[19],'sid':os.getsid(p.pid),'ppid':os.getppid(),
 'task':'S-AUTONOMY main successor','detached':True}}
(r/'AUTONOMY_PID.json').write_text(json.dumps(doc,indent=2)); print(json.dumps(doc))
''')


def rearm(host):
    assert host == 'gpuA'
    remote(host, '''
import pathlib,json,os,time
r=pathlib.Path('/root/autodl-tmp/c4/restart20261026')
d=json.loads((r/'AUTONOMY_PID.json').read_text())
s=json.loads((r/'AUTONOMY_STATUS.json').read_text())
assert s['status']=='WAIT_MAIN_C2H60B' and s['pid']==d['pid']
assert d['task']=='S-AUTONOMY main successor' and not (r/'AUTONOMY_CHILDREN.jsonl').exists()
f=pathlib.Path('/proc/'+str(d['pid'])+'/stat').read_text().rsplit(')',1)[1].split()
assert f[19]==d['start_ticks'] and os.getsid(d['pid'])==d['pid']
os.kill(d['pid'],15); time.sleep(1)
for name in ('AUTONOMY_PID.json','AUTONOMY_STATUS.json','arm/driver_autonomy.log'):
 p=r/name;p.rename(p.with_name(p.name+'.preflight_repair.'+str(d['pid'])))
print(json.dumps({'retired_own_waiter':d,'no_policy_or_collect_child_existed':True,
 'reason':'zero-charge preflight requires native gate sidecar; corrected before any phase launch'}))
''')
    launch(host)


def status(host):
    remote(host, f'''
import pathlib,json,os
r=pathlib.Path({RELEASES[host]!r})
for name in ('AUTONOMY_PID.json','AUTONOMY_STATUS.json','arm/OWNER_PRIORITY_PAUSE.json'):
 p=r/name
 if p.exists(): print(name,p.read_text())
p=r/'AUTONOMY_PID.json'
if p.exists():
 d=json.loads(p.read_text()); fields=pathlib.Path('/proc/'+str(d['pid'])+'/stat').read_text().rsplit(')',1)[1].split()
 print('DETACHED_VERIFIED',json.dumps({{'live':fields[0]!='Z' and fields[19]==d['start_ticks'],'sid':os.getsid(d['pid']),'ppid':fields[1]}}))
reg=json.loads((r/'arm/REGISTRATION.json').read_text()); s=json.loads((r/'arm/loop/LOOP14_STATE.json').read_text())
print('BUDGET',reg['budget']);print('CHARGED',s['sim_s_charged']);print('STOPPED',s.get('stopped'))
''')


def spot(box):
    assert box in ('cpuP', 'cpuQ')
    data = base64.b64encode((ROOT / 'tools/autonomy/spot_watch.py').read_bytes()).decode()
    remote(box, f'''
import pathlib,base64,subprocess,sys,os,json,time
b=pathlib.Path('/home/ubuntu/c4/main14/restart20261026'); assert b.is_dir()
d=b/'autonomy'; d.mkdir(exist_ok=True)
p=d/'spot_watch.py'; p.write_bytes(base64.b64decode({data!r}))
pidfile=d/'SPOT_PID.json'
if pidfile.exists():
 old=json.loads(pidfile.read_text()); f=pathlib.Path('/proc/'+str(old['pid'])+'/stat')
 if f.exists() and f.read_text().rsplit(')',1)[1].split()[19]==old['start_ticks']:
  assert os.getsid(old['pid'])==old['pid']
  (d/('SPOT_PID.'+str(old['pid'])+'.before.json')).write_text(json.dumps(old))
  os.kill(old['pid'],15)
  time.sleep(1)
with (d/'spot_watch.log').open('ab') as stream:
 proc=subprocess.Popen([sys.executable,'-B','-u',str(p),'--base',str(b),'--box',{box!r}],
  stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(1)
f=pathlib.Path('/proc/'+str(proc.pid)+'/stat').read_text().rsplit(')',1)[1].split()
doc={{'pid':proc.pid,'start_ticks':f[19],'sid':os.getsid(proc.pid),'detached':True}}
assert f[0]!='Z' and doc['sid']==proc.pid
pidfile.write_text(json.dumps(doc,indent=2)); print(json.dumps(doc))
''')


def mirror(host):
    assert host == 'gpuA'
    remote(host, '''
import pathlib,json,os,subprocess,sys,time
r=pathlib.Path('/root/autodl-tmp/c4/restart20261026');assert not (r/'AUTONOMY_MIRROR_PID.json').exists()
with (r/'autonomy/gpu_mirror.log').open('xb') as stream:
 p=subprocess.Popen([sys.executable,'-B','-u',str(r/'autonomy/gpu_mirror.py')],
  stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(1)
f=pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text().rsplit(')',1)[1].split()
doc={'pid':p.pid,'start_ticks':f[19],'sid':os.getsid(p.pid),'detached':True}
assert f[0]!='Z' and doc['sid']==p.pid
(r/'AUTONOMY_MIRROR_PID.json').write_text(json.dumps(doc,indent=2)); print(json.dumps(doc))
''')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['stage', 'budget', 'launch', 'rearm', 'status', 'spot', 'mirror'])
    parser.add_argument('--host', choices=[*RELEASES, 'cpuP', 'cpuQ'], default='gpuA')
    args = parser.parse_args()
    globals()[args.action](args.host)
