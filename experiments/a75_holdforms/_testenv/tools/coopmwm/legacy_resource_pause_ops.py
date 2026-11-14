"""MAIN21:58b reversible old-reference pause, at an identity-proven idle boundary."""
import json
import time

from . import ops
from .ls_validity_ops import remote
from .protocol import REGISTRY, checked, sha
from .storage import read, write

ROOT='/home/ubuntu/coopmwm20261102_warm1'
SOURCE='/home/ubuntu/coopmwm20261103_memory_recovery2'
RECORD='LEGACY_RESOURCE_PAUSE_REGISTRATION_2158B.json'


def register():
    path=ops.EVIDENCE/RECORD
    if not path.exists():
        write(path,dict(authority='MAIN2026-11-03 21:58b / 22:10',new_sim_s=0,
            line='\nS-COOPMWM MAIN21:58b/22:10 RESOURCE PAUSE: old non-V2V B/R '
            'pause at their CURRENT durable checkpoint boundary, all checkpoints/charges '
            'and receipts retained. Verify boot/start/argv of owned arms/graders; briefly '
            'freeze only those exact owners, prove no active child except resource_tracker '
            'and no attempted next-wave jobs, write registered warm_health RESOURCE pause, '
            'then CONT those owners to acknowledge through the existing HealthPause path. '
            'No checkpoint selection, no physics replay, no deletion or budget change.\n'))
    value=read(path);h=sha(path);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(value['line']+'amendmentSHA'+h+'\n')
    return value,h


def pause():
    registration,h=register()
    code=f'''from pathlib import Path
import os,sys,time,json,signal,hashlib,shutil
r=Path({ROOT!r});s=Path({SOURCE!r});control=r/'memory_recovery1'
sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm import replay_start as rs
from tools.coopmwm.memory_recovery import verify_identity
from tools.coopmwm.warm_health import pause,HealthPause
from tools.coopmwm.storage import write,read
out=r/'LEGACY_RESOURCE_PAUSE_2158B.json'
if out.exists():print(out.read_text());raise SystemExit
reg=json.loads({json.dumps(registration)!r});reg['amendment_sha256']={h!r}
registered=r/'LEGACY_RESOURCE_PAUSE_REGISTERED_2158B.json'
if registered.exists():assert read(registered)==reg
else:write(registered,reg)
authority=r/'AUTHORITY.md'
if {h!r} not in authority.read_text():
 with authority.open('a') as f:f.write(reg['line']+'amendmentSHA'+{h!r}+'\\n')
owners=[]
for a in ('B','R'):
 d=dict(read(control/('arm_'+a+'_DISPATCH.json')),run_root=str(r))
 verify_identity(d,'arm',a);owners.append((d,'arm',a))
 old=read(r/'B_REPLAY_COMPOSITE_STARTED_0957.json')
 d=dict(next(x for x in old['children'] if x['component']=='grade' and x['arm']==a),run_root=str(r))
 verify_identity(d,'grade',a);owners.append((d,'grade',a))
def idle(d,component,arm,frozen=False):
 p=verify_identity(d,component,arm);children=[]
 for k in (p/'task'/str(d['pid'])/'children').read_text().split():
  q=Path('/proc',k);argv=(q/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
  if not argv:continue
  assert 'multiprocessing.resource_tracker' in argv,'active owned child: '+argv
  children.append(dict(pid=int(k),argv=argv))
 if component=='arm' and not frozen:assert (p/'wchan').read_text().strip()=='do_select','not at pacing boundary'
 return children
for d,c,a in owners:idle(d,c,a)
frozen=[]
try:
 for d,c,a in owners:
  verify_identity(d,c,a);os.kill(d['pid'],signal.SIGSTOP);frozen.append(d)
 for d,c,a in owners:
  for _ in range(100):
   if Path('/proc',str(d['pid']),'stat').read_text().split()[2]=='T':break
   time.sleep(.01)
  else:raise RuntimeError('owned pause not acknowledged')
  idle(d,c,a,True)
 latest={{}};checkpoints={{}}
 for a in ('B','R'):
  u=max(int(p.stem.rsplit('_',1)[1]) for p in r.glob(a+'_UPDATE_[0-9][0-9][0-9].json'))
  p=r/(a+'_checkpoint_'+str(u).zfill(3)+'.pt');assert p.is_file()
  assert not (r/(a+'_checkpoint_'+str(u+1).zfill(3)+'.pt')).exists()
  assert not list((r/'jobs').glob('replay0735_warm_u'+str(u+1).zfill(3)+'_'+a+'_*/CLAIM.json')),'next wave already attempted'
  latest[a]=u;checkpoints[a]=dict(path=str(p),update=u,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 evidence=dict(unix=time.time(),authority='MAIN21:58b/22:10 reversible resource pause',
  amendment_sha256={h!r},updates=latest,checkpoints=checkpoints,owners=[d for d,c,a in owners],
  boundary='identity-verified idle pacing, resource_tracker only, no next-wave claim',
  disk_free_bytes=shutil.disk_usage(r).free,new_sim_s=0,physics_replay=False)
 try:pause(r,'RESOURCE_MAIN2158B',evidence)
 except HealthPause:pass
 write(out,evidence)
 print(json.dumps(evidence))
finally:
 for d in frozen:
  if rs.alive(d):os.kill(d['pid'],signal.SIGCONT)
'''
    value=json.loads(remote('cpuQ',code,timeout=40))
    target=ops.EVIDENCE/'LEGACY_RESOURCE_PAUSE_2158B.json'
    if not target.exists():write(target,value)
    print(json.dumps(value))


def scope():
    """MAIN22:03: keep old owners stopped, unblock tooling on the shared root."""
    path=ops.EVIDENCE/'LEGACY_ONLY_HOLD_REGISTRATION_2203.json'
    if not path.exists():
        write(path,dict(authority='MAIN2026-11-03 22:03',new_sim_s=0,
            line='\nS-COOPMWM MAIN22:03 scoped resource HOLD: after all four exact '
            'legacy arm/grader owners acknowledge and exit, retain their checkpoint '
            'hold but retire ONLY our SHA-matched RESOURCE_MAIN2158B pause marker '
            'into memory_recovery1/ metadata. This preserves the pause bytes and all '
            'health/safety guards while allowing required gate/solo/search tooling '
            'to call assert_running on the shared warm1 root. No old runner restart.\n'))
    reg=read(path);h=sha(path);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(reg['line']+'amendmentSHA'+h+'\n')
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,sys,time,hashlib,shutil
r=Path({ROOT!r});s=Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm import replay_start as rs
from tools.coopmwm.storage import read,write
proof=read(r/'LEGACY_RESOURCE_PAUSE_2158B.json')
for d in proof['owners']:assert not rs.alive(d),'legacy owner still alive'
p=r/'PAUSE_RESOURCE_MAIN2158B.json';dest=r/'memory_recovery1/RESOURCE_MAIN2158B_RETIRED_2203.json'
out=r/'LEGACY_ONLY_RESOURCE_HOLD_2203.json'
if out.exists():
 assert not p.exists() and dest.exists()
 print(out.read_text());raise SystemExit
if p.exists():
 marker=read(p);assert marker['evidence']==proof and marker['rule']=='RESOURCE_MAIN2158B'
else:
 assert dest.exists();marker=read(dest);assert marker['evidence']==proof
raw=(p if p.exists() else dest).read_bytes();digest=hashlib.sha256(raw).hexdigest()
reg=json.loads({json.dumps(reg)!r});authority=r/'AUTHORITY.md'
if {h!r} not in authority.read_text():
 with authority.open('a') as f:f.write(reg['line']+'amendmentSHA'+{h!r}+'\\n')
v=dict(unix=time.time(),authority='MAIN22:03',amendment_sha256={h!r},
 owners_exited=proof['owners'],checkpoints=proof['checkpoints'],updates=proof['updates'],
 legacy_training_held=True,shared_root_health_pause_retired=str(dest),
 original_pause_sha256=digest,no_old_restart=True,new_sim_s=0,disk_free_bytes=shutil.disk_usage(r).free)
intent=r/'memory_recovery1/RESOURCE_SCOPE_INTENT_2203.json'
if not intent.exists():write(intent,v)
if p.exists():assert not dest.exists();p.rename(dest)
assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest
write(out,v);print(json.dumps(v))''',timeout=35))
    target=ops.EVIDENCE/'LEGACY_ONLY_RESOURCE_HOLD_2203.json'
    if not target.exists():write(target,value)
    print(json.dumps(value))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',nargs='?',default='pause',choices=('pause','scope'))
    globals()[parser.parse_args().action]()
