"""Resident resource gate for ONE corrected CPU worker; never a simulator route.

The old WM stays stopped until unchanged pace permits replacement. The known
query-ABI bug is fixed before a fresh process, not hotpatched into a bound one.
"""
import base64
import hashlib
import json
from pathlib import Path
from planfix_ops import remote,note

PROGRAM=r'''
import fcntl,hashlib,json,os,signal,subprocess,time,traceback
from pathlib import Path
control=Path(__file__).parent;config=json.loads((control/'GATED_CONFIG.json').read_text());root=Path(config.get('worker_root',str(control)))
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(Path(__file__))==config['expected_program_sha256']
lock=(control/'GATED_COORDINATOR.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
os.nice(19);os.sched_setaffinity(0,{47})
owner=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),affinity=[47],new_sim_s=0)
def write(path,value):
 with path.open('x') as out:json.dump(value,out,sort_keys=True);out.flush();os.fsync(out.fileno())
write(control/'GATED_COORDINATOR_OWNER.json',owner)
assert sha(root/'GPU_BINDING.json')==config['binding_file_sha256']
from planfix_cpu_worker import probe_health
model=None;last=None
def event(kind,**values):
 global last
 row=dict(kind=kind,unix=time.time(),new_sim_s=0,**values)
 with (control/'GATED_EVENTS.jsonl').open('a') as out:out.write(json.dumps(row,sort_keys=True)+chr(10));out.flush();os.fsync(out.fileno())
 temporary=control/('GATED_PHASE.'+str(time.time_ns())+'.json')
 write(temporary,row);os.replace(temporary,control/'GATED_PHASE.json')
 print('GATED_EVENT '+json.dumps(row),flush=True);last=kind
def fields_for(target):
 p=Path('/proc',str(target['pid']),'stat');f=p.read_text().split() if p.exists() else None
 return f if f and f[21]==target['start_ticks'] and f[2]!='Z' else None
def signal_owned(target,script,which):
 assert target['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 f=fields_for(target);assert f,'exact own incarnation must still exist'
 status=Path('/proc',str(target['pid']),'status').read_text().splitlines()
 assert int(next(x.split()[1] for x in status if x.startswith('Uid:')))==os.getuid()
 args=Path('/proc',str(target['pid']),'cmdline').read_bytes().split(bytes([0]))
 assert str(script).encode() in args and str(script.parent).encode() in args
 fd=os.pidfd_open(target['pid']);assert fields_for(target)
 signal.pidfd_send_signal(fd,which);os.close(fd)
def retire_known_old():
 old=root.parent/'cpu_wm_v6';target=read(old/'CPU_OWNER.json')
 assert target['pid']==364064 and target['start_ticks']=='8210497'
 assert sha(old/'GPU_BINDING.json')=='452e7ae008aa9a92bdc68a2f4ecda4251a0ad0c4b6946fddbc549c9ab0eb8621'
 assert (old/'QUERY_ABI_CORRECTION.json').is_file() and (old/'STOP.json').is_file()
 assert not (old/'SPEED_SEARCH_STARTED.json').exists() and not list(old.glob('STARTUP_*.json'))
 f=fields_for(target);signalled=False
 if f:
  assert f[2]=='T','old known-ABI worker must remain stopped'
  event('retire_exact_old_query_abi_worker',target=target)
  signal_owned(target,old/'planfix_cpu_worker.py',signal.SIGTERM)
  # A stopped process receives its pending default SIGTERM after SIGCONT.
  if fields_for(target):signal_owned(target,old/'planfix_cpu_worker.py',signal.SIGCONT)
  signalled=True
  for _ in range(50):
   if not fields_for(target):break
   time.sleep(.1)
 assert not fields_for(target),'no replacement before exact old model exits'
 proof=dict(owner=target,reason='pre-search export ABI correction, no completed gate or search',
  pidfd_sigterm_then_sigcont=signalled,exited=True,unix=time.time(),new_sim_s=0,files_and_prior_measurements_preserved=True)
 p=old/'CPU_OPERATOR_RETIRED.json'
 if not p.exists():write(p,proof)
 return read(p)
def launch():
 assert not (root/'DISPATCH_STARTED.json').exists() and not (root/'CPU_OWNER.json').exists()
 reg=root/'CPU_EXECUTION_REGISTRATION.md'
 assert 'PLANFIX CPU execution registration SHA '+sha(reg) in (root.parent.parent/'AUTHORITY.md').read_text()
 old=retire_known_old()
 if probe_health(root)['blocked']:return None
 write(root/'DISPATCH_STARTED.json',dict(unix=time.time(),new_sim_s=0,retired_old=old))
 env=read(root/'STAGE_ENV.json')['env'];env['PYTHONPATH']=str(root/'compat')+':'+str(root)+':'+env['PYTHONPATH']
 with (root/'CPU_WORKER.log').open('xb') as log:
  p=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-u',str(root/'planfix_cpu_worker.py'),str(root)],
   cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 f=Path('/proc',str(p.pid),'stat').read_text().split()
 target=dict(pid=p.pid,start_ticks=f[21],boot_id=owner['boot_id'],unix=time.time(),root=str(root),
  max_cpu_cores=1,nice=19,new_sim_s=0,procedure='original8gates then ONE speed search, no SUMO or expansion',coordinator=owner)
 write(root/'DISPATCH.json',target);event('corrected_model_dispatched_once',target=target)
 return target
reason=None;deadline=time.monotonic()+config['maximum_controller_wall_s'];paused_by_us=False;paused_health=None
try:
 while time.monotonic()<deadline:
  if (root/'SPEED_SEARCH_RESULT.json').exists():reason='speed_complete_no_expansion';break
  if (root/'CPU_FAILED.json').exists():reason='worker_failed_no_retry';break
  health=probe_health(root)
  latest=control/('GATED_POLL.'+str(time.time_ns())+'.json')
  write(latest,dict(unix=time.time(),health={k:v for k,v in health.items() if k!='launch'},model=model,new_sim_s=0))
  os.replace(latest,control/'GATED_POLL_LATEST.json')
  if model is None:
   if health['blocked']:
    if last!='wait_unchanged_pace_before_model':event('wait_unchanged_pace_before_model',health={k:v for k,v in health.items() if k!='launch'})
   else:model=launch()
  else:
   f=fields_for(model)
   if not f:reason='model_exited_no_retry';break
   if health['blocked'] and f[2]!='T':
    signal_owned(model,root/'planfix_cpu_worker.py',signal.SIGSTOP)
    paused_by_us=True;paused_health=health
    event('own_model_paused_ram_preserved',target=model,health={k:v for k,v in health.items() if k!='launch'})
   elif not health['blocked'] and f[2]=='T':
    assert paused_by_us,'never release an unrelated stop'
    if health['reason']!='proved_completed_clean2_idle':
     assert all(health['pace'][k]['update']>v['update'] for k,v in paused_health['pace'].items() if v['over'])
    signal_owned(model,root/'planfix_cpu_worker.py',signal.SIGCONT)
    paused_by_us=False;event('same_own_model_resumed_no_baseline_change',target=model,health={k:v for k,v in health.items() if k!='launch'})
  time.sleep(60)
 else:reason='finite_controller_deadline_no_retry'
except BaseException as error:
 reason='controller_error_no_retry:'+repr(error);traceback.print_exc()
finally:
 if model is not None:
  f=fields_for(model)
  if f and f[2]!='T' and not (root/'SPEED_SEARCH_RESULT.json').exists():
   signal_owned(model,root/'planfix_cpu_worker.py',signal.SIGSTOP)
   event('unfinished_own_model_preserved_stopped_at_controller_exit',target=model)
 write(control/'GATED_COORDINATOR_TERMINAL.json',dict(owner=owner,model=model,reason=reason,unix=time.time(),new_sim_s=0))
'''

DEPLOY=r'''
import base64,fcntl,hashlib,json,os,subprocess,time,signal,sys
from pathlib import Path
root=Path('/home/ubuntu/coopmwm20261105_clean1/plan_fix/cpu_wm_v7')
assert (root/'CPU_PORT_QUALIFIED.json').is_file() and (root/'CPU_EXECUTION_REGISTRATION.md').is_file()
assert not (root/'DISPATCH_STARTED.json').exists() and not (root/'CPU_OWNER.json').exists()
prior=json.loads((root/'GATED_COORDINATOR_TERMINAL.json').read_text());assert "pidfd_open" in prior['reason'] and prior['model'] is None
old=json.loads((root/'GATED_COORDINATOR_OWNER.json').read_text());p=Path('/proc',str(old['pid']),'stat');f=p.read_text().split() if p.exists() else None
assert not (f and f[21]==old['start_ticks'] and f[2]!='Z'),'prior resource coordinator still live'
assert hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'),'resource interpreter must support tested pidfd API'
control=root/'gated_control_v2';assert not control.exists();control.mkdir()
reg=root/'CPU_EXECUTION_REGISTRATION.md';r=root.parent.parent
assert 'PLANFIX CPU execution registration SHA '+hashlib.sha256(reg.read_bytes()).hexdigest() in (r/'AUTHORITY.md').read_text()
program=base64.b64decode(__PROGRAM__);p=control/'GATED_COORDINATOR.py'
with p.open('xb') as out:out.write(program);out.flush();os.fsync(out.fileno())
config=dict(binding_file_sha256=hashlib.sha256((root/'GPU_BINDING.json').read_bytes()).hexdigest(),
 expected_program_sha256=hashlib.sha256(program).hexdigest(),worker_root=str(root),no_sumo=True,maximum_controller_wall_s=86400)
with (control/'GATED_CONFIG.json').open('x') as out:json.dump(config,out,sort_keys=True);out.flush();os.fsync(out.fileno())
registration=dict(root=str(root),control=str(control),resource_python=sys.executable,model_python='/home/ubuntu/miniconda3/bin/python',program_sha256=config['expected_program_sha256'],config=config,
 rule='Preserve stopped old364064 until unchanged pace/health or sealed-clean2-idle permits exact own retirement and ONE correctedv7; same-process8gates+ONE speed search, no SUMO/retry/expansion/bank switch',new_sim_s=0)
with (r/'ACCOUNTING.claim').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 with (r/'AUTHORITY.md').open('a') as out:out.write(chr(10)+'PLANFIX guarded CPUv7 '+json.dumps(registration,sort_keys=True)+chr(10));out.flush();os.fsync(out.fileno())
with (control/'GATED_COORDINATOR_STARTED.json').open('x') as out:json.dump(registration,out,sort_keys=True);out.flush();os.fsync(out.fileno())
env=json.loads((root/'STAGE_ENV.json').read_text())['env'];env['PYTHONPATH']=str(root/'compat')+':'+str(root)+':'+env['PYTHONPATH']
with (control/'GATED_COORDINATOR.log').open('xb') as log:
 p=subprocess.Popen([sys.executable,'-B','-u',str(control/'GATED_COORDINATOR.py')],cwd=root,env=env,
  stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
f=Path('/proc',str(p.pid),'stat').read_text().split()
receipt=dict(pid=p.pid,start_ticks=f[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),
 root=str(root),control=str(control),registration=registration,new_sim_s=0,may_dispatch_one_model_only_after_unchanged_resource_gate=True)
with (control/'GATED_COORDINATOR_DISPATCH.json').open('x') as out:json.dump(receipt,out,sort_keys=True);out.flush();os.fsync(out.fileno())
print('GATED_COORDINATOR_DISPATCH',json.dumps(receipt),flush=True)
'''

if __name__=='__main__':
    compile(PROGRAM,'<guarded-cpu-v7-preflight>','exec')
    text=remote('cpuV',DEPLOY.replace('__PROGRAM__',repr(base64.b64encode(PROGRAM.encode()).decode())),timeout=30)
    value=json.loads(next(x[len('GATED_COORDINATOR_DISPATCH '):] for x in text.splitlines() if x.startswith('GATED_COORDINATOR_DISPATCH ')))
    with Path(__file__).with_name('PLANFIX_CPU_V7_GATED_DISPATCH_v2.json').open('x') as out:json.dump(value,out,indent=2,sort_keys=True)
    note('cpuV',f"guarded coordinator {value['pid']}/{value['start_ticks']} waits unchanged resource gate before exact own retirement/ONE corrected model; no duplicate model, training signal, SUMO or expansion")
