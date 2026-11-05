"""Stage one CPU-resident resource controller for the existing WM PID only.

It never creates/restarts a model or simulator process, changes a pace limit,
or signals training. It resumes only after a NEW completed interval satisfies
the unchanged prospective limits; RAM/partial gate work remains in one PID.
"""
import base64
import hashlib
import json
from pathlib import Path
from planfix_ops import remote,note
from planfix_cpu_pace_control import CODE as CONTROL

SUPERVISOR=r'''
import fcntl,hashlib,json,os,time,traceback
from pathlib import Path
w=Path(__file__).parent
lock=(w/'PACE_SUPERVISOR.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
os.nice(19);os.sched_setaffinity(0,{47})
owner=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),
 target_pid=364064,target_start_ticks='8210497',affinity=[47],new_sim_s=0)
with (w/'PACE_SUPERVISOR_OWNER.json').open('x') as out:json.dump(owner,out,sort_keys=True);out.flush();os.fsync(out.fileno())
source=(w/'PACE_CONTROL_REMOTE.py').read_text();assert hashlib.sha256(source.encode()).hexdigest()==__CONTROL_SHA__
deadline=time.monotonic()+21600
terminal=None
try:
 while time.monotonic()<deadline:
  if (w/'SPEED_SEARCH_RESULT.json').exists():terminal='speed_result_complete';break
  if (w/'CPU_FAILED.json').exists():terminal='worker_failed_no_retry';break
  proc=Path('/proc/364064/stat');fields=proc.read_text().split() if proc.exists() else None
  if not fields or fields[21]!='8210497' or fields[2]=='Z':terminal='owned_worker_exited_no_retry';break
  state=json.loads((w/'PACE_CONTROL_STATE.json').read_text())
  action='resume' if state['action']=='paused' else 'pause'
  # Only the fixed, fully identity/pidfd-bound controller performs a signal.
  try:exec(compile(source.replace('__ACTION__',repr(action)),str(w/'PACE_CONTROL_REMOTE.py'),'exec'),{})
  except SystemExit as error:
   if error.code not in (0,None):raise
  time.sleep(60)
 else:
  # Leave unfinished work stopped in RAM at this finite controller deadline.
  state=json.loads((w/'PACE_CONTROL_STATE.json').read_text())
  if state['action']!='paused':
   raise RuntimeError('controller deadline reached while WM active; inspect exact own worker before any further work')
  terminal='controller_deadline_own_worker_paused'
except BaseException as error:
 terminal='control_error_no_retry:'+repr(error);traceback.print_exc()
finally:
 with (w/'PACE_SUPERVISOR_TERMINAL.json').open('x') as out:json.dump(dict(owner=owner,reason=terminal,unix=time.time(),new_sim_s=0),out,sort_keys=True);out.flush();os.fsync(out.fileno())
'''

DEPLOY=r'''
import base64,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
w=Path('/home/ubuntu/coopmwm20261105_clean1/plan_fix/cpu_wm_v6')
assert not (w/'PACE_SUPERVISOR_STARTED.json').exists() and not (w/'PACE_SUPERVISOR_OWNER.json').exists(),'inspect/adopt prior own controller, never duplicate'
state=json.loads((w/'PACE_CONTROL_STATE.json').read_text());assert state['action']=='paused'
owner=state['owner'];assert owner['pid']==364064 and owner['start_ticks']=='8210497'
fields=Path('/proc/364064/stat').read_text().split();assert fields[21]=='8210497' and fields[2]=='T'
assert owner['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
manifest={}
for name,encoded in __FILES__.items():
 raw=base64.b64decode(encoded);p=w/name
 if p.exists():assert p.read_bytes()==raw
 else:
  with p.open('xb') as out:out.write(raw);out.flush();os.fsync(out.fileno())
 manifest[name]=hashlib.sha256(raw).hexdigest()
assert json.loads((w.parent.parent/'plan_fix/REGISTRATION.json').read_text())['allocation_sim_s']==600000
registration=dict(schema='planfix.cpu.own-resource-controller.v1',target=owner,source_sha256=manifest,
 interval_s=60,maximum_controller_wall_s=21600,new_sim_s=0,
 rule='Never signal training or change limits. pidfd own364064/start8210497 only; pause on any fresh interval over original limits; resume only NEW completed good intervals, health/config/original owners unchanged. No model or SUMO dispatch/retry.')
marker='PLANFIX CPU pace supervisor '+hashlib.sha256(json.dumps(registration,sort_keys=True).encode()).hexdigest()
r=w.parent.parent
with (r/'ACCOUNTING.claim').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if marker not in (r/'AUTHORITY.md').read_text():
  with (r/'AUTHORITY.md').open('a') as out:out.write(chr(10)+marker+chr(10)+json.dumps(registration,sort_keys=True)+chr(10));out.flush();os.fsync(out.fileno())
with (w/'PACE_SUPERVISOR_STARTED.json').open('x') as out:json.dump(dict(registration=registration,unix=time.time()),out,sort_keys=True);out.flush();os.fsync(out.fileno())
with (w/'PACE_SUPERVISOR.log').open('xb') as log:
 p=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-u',str(w/'PACE_SUPERVISOR.py')],cwd=w,
  stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
f=Path('/proc',str(p.pid),'stat').read_text().split()
receipt=dict(pid=p.pid,start_ticks=f[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 target_pid=364064,target_start_ticks='8210497',root=str(w),registration=registration,unix=time.time(),new_sim_s=0)
with (w/'PACE_SUPERVISOR_DISPATCH.json').open('x') as out:json.dump(receipt,out,sort_keys=True);out.flush();os.fsync(out.fileno())
print('PACE_SUPERVISOR_DISPATCH',json.dumps(receipt),flush=True)
'''

if __name__=='__main__':
    supervisor=SUPERVISOR.replace('__CONTROL_SHA__',repr(hashlib.sha256(CONTROL.encode()).hexdigest()))
    compile(supervisor,'<pace-supervisor-preflight>','exec')
    compile(CONTROL.replace('__ACTION__',repr('resume')),'<pace-controller-preflight>','exec')
    files={'PACE_CONTROL_REMOTE.py':base64.b64encode(CONTROL.encode()).decode(),
           'PACE_SUPERVISOR.py':base64.b64encode(supervisor.encode()).decode()}
    text=remote('cpuV',DEPLOY.replace('__FILES__',repr(files)),timeout=30)
    value=json.loads(next(x[len('PACE_SUPERVISOR_DISPATCH '):] for x in text.splitlines() if x.startswith('PACE_SUPERVISOR_DISPATCH ')))
    with Path(__file__).with_name('PLANFIX_CPU_PACE_SUPERVISOR_DISPATCH.json').open('x') as out:json.dump(value,out,indent=2,sort_keys=True)
    note('cpuV',f"sole resource supervisor pid={value['pid']} owns no model; target existing364064/start8210497 preserved; unchanged pace predicate, no training signals/SUMO/new WM dispatch")
