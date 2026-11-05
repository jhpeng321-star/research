"""Pause/resume ONLY the exact owned CPU WM incarnation; preserve its RAM work.

No training signal/configuration change. Resume requires fresh completed update
intervals back within the unchanged prospective pace limits.
"""
import argparse
import json
from pathlib import Path
from planfix_ops import remote,note

CODE=r'''
import fcntl,hashlib,json,os,signal,time
from pathlib import Path
w=Path('/home/ubuntu/coopmwm20261105_clean1/plan_fix/cpu_wm_v6');c=Path('/home/ubuntu/coopmwm20261105_clean2')
read=lambda p:json.loads(Path(p).read_text())
owner=read(w/'CPU_OWNER.json');dispatch=read(w/'DISPATCH.json');baseline=read(w/'CLEAN2_BASELINE.json')
assert owner['pid']==dispatch['pid']==364064 and owner['start_ticks']==dispatch['start_ticks']=='8210497'
assert owner['boot_id']==dispatch['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert hashlib.sha256((w/'GPU_BINDING.json').read_bytes()).hexdigest()=='452e7ae008aa9a92bdc68a2f4ecda4251a0ad0c4b6946fddbc549c9ab0eb8621'
assert hashlib.sha256((c/'LAUNCHED.json').read_bytes()).hexdigest()==baseline['launch_file_sha256']
assert hashlib.sha256((c/'CONFIG.json').read_bytes()).hexdigest()==read(c/'LAUNCHED.json')['config_file_sha256']
pace={}
for role,b in baseline['pace'].items():
 values=[read(p) for p in sorted(c.glob(role+'_UPDATE_*.json'))[-2:]]
 dt=values[-1]['finished_unix']-values[-2]['finished_unix']
 pace[role]=dict(update=values[-1]['update'],interval_s=dt,limit_s=b['ceiling_s'],
  over=values[-1]['update']>b['last_update'] and dt>b['ceiling_s'])
training_alive=True
for d in read(c/'LAUNCHED.json')['owners'].values():
 p=Path('/proc',str(d['pid']),'stat');f=p.read_text().split() if p.exists() else None
 training_alive=training_alive and bool(f and f[21]==d['start_ticks'] and f[2]!='Z')
blocked=any(p['over'] for p in pace.values()) or not training_alive or (c/'STOP.json').exists() or (c/'FAULT.json').exists() or (w/'STOP.json').exists() or (w.parent/'STOP.json').exists()
action=__ACTION__;p=Path('/proc',str(owner['pid']),'stat')
f=p.read_text().split() if p.exists() else None
assert f and f[21]==owner['start_ticks'] and f[2]!='Z','owned WM has already exited; inspect receipt, never signal'
status=Path('/proc',str(owner['pid']),'status').read_text().splitlines()
uid=next(line.split()[1] for line in status if line.startswith('Uid:'));assert int(uid)==os.getuid()
command=Path('/proc',str(owner['pid']),'cmdline').read_bytes().split(b'\0')
assert str(w/'planfix_cpu_worker.py').encode() in command and str(w).encode() in command
with (w/'PACE_CONTROL.claim').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if action=='pause' and not blocked:
  print('PACE_CONTROL',json.dumps(dict(action='no_pause_needed',pace=pace,owner=owner,new_sim_s=0)),flush=True)
 elif action=='resume' and blocked:
  print('PACE_CONTROL',json.dumps(dict(action='remain_paused',pace=pace,owner=owner,new_sim_s=0)),flush=True)
 else:
  state=w/'PACE_CONTROL_STATE.json'
  if action=='resume':
   previous=read(state);assert previous['action']=='paused' and previous['owner']==owner
   assert all(pace[k]['update']>previous['pace'][k]['update'] for k in previous['pace'] if previous['pace'][k]['over'])
   assert not (w/'STOP.json').exists()
  elif f[2]=='T':
   previous=read(state);assert previous['action']=='paused' and previous['owner']==owner
   print('PACE_CONTROL',json.dumps(previous),flush=True);raise SystemExit(0)
  assert hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal')
  fd=os.pidfd_open(owner['pid']);f=Path('/proc',str(owner['pid']),'stat').read_text().split()
  assert f[21]==owner['start_ticks'] and f[2]!='Z'
  event=dict(action='pause_requested' if action=='pause' else 'resume_requested',owner=owner,pace=pace,unix=time.time(),new_sim_s=0)
  with (w/'PACE_CONTROL_EVENTS.jsonl').open('a') as out:out.write(json.dumps(event,sort_keys=True)+'\n');out.flush();os.fsync(out.fileno())
  signal.pidfd_send_signal(fd,signal.SIGSTOP if action=='pause' else signal.SIGCONT);os.close(fd)
  time.sleep(.1)
  f=Path('/proc',str(owner['pid']),'stat').read_text().split();assert f[21]==owner['start_ticks']
  assert (f[2]=='T') if action=='pause' else (f[2]!='T')
  event.update(action='paused' if action=='pause' else 'resumed',unix=time.time(),process_state=f[2])
  with (w/'PACE_CONTROL_EVENTS.jsonl').open('a') as out:out.write(json.dumps(event,sort_keys=True)+'\n');out.flush();os.fsync(out.fileno())
  temporary=w/('PACE_CONTROL_STATE.'+str(time.time_ns())+'.json')
  with temporary.open('x') as out:json.dump(event,out,sort_keys=True);out.flush();os.fsync(out.fileno())
  os.replace(temporary,state)
  print('PACE_CONTROL',json.dumps(event),flush=True)
'''

if __name__=='__main__':
    import time
    p=argparse.ArgumentParser();p.add_argument('action',choices=('pause','resume'));a=p.parse_args()
    text=remote('cpuV',CODE.replace('__ACTION__',repr(a.action)),timeout=25)
    value=json.loads(next(x[len('PACE_CONTROL '):] for x in text.splitlines() if x.startswith('PACE_CONTROL ')))
    with Path(__file__).with_name(f'PLANFIX_CPU_PACE_CONTROL_{time.time_ns()}.json').open('x') as out:json.dump(value,out,indent=2,sort_keys=True)
    note('cpuV',f"own WM pace control {value['action']} exact pid364064/start8210497 via pidfd; preserved RAM/source/gates; training never signalled; new_sim_s=0")
