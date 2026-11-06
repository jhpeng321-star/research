"""Read-only final status capture for the already-armed gpuR waiter."""
import json

from deploy import ROOT, remote

record = json.loads(remote("gpuR", """
import fcntl,hashlib,json,os,pathlib,shutil,subprocess,time
v5=pathlib.Path('/root/autodl-tmp/c4/v5'); arm=v5/'main14_arm'; code=v5/'repo_main14'
a=json.loads((arm/'ARMED.json').read_text()); pid=int((arm/'waiter.pid').read_text())
fields=pathlib.Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
assert fields[0]!='Z' and int(fields[1])==1 and int(fields[3])==pid
fds={str(i):os.readlink('/proc/'+str(pid)+'/fd/'+str(i)) for i in (0,1,2)}
assert fds['0']=='/dev/null' and fds['1']==str(v5/'main14_autolaunch.log') and fds['2']==fds['1']
with (arm/'waiter.lock').open('a') as lock:
 try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError: held=True
 else:
  fcntl.flock(lock,fcntl.LOCK_UN); held=False
assert held, 'waiter does not hold its exclusive lock'
check=subprocess.run(['/root/miniconda3/bin/python',str(code/'scripts/c4/main14_autolaunch.py'),
 '--arm',str(arm/'ARMED.json'),'--check-only'],capture_output=True,text=True,check=True)
record={'captured_epoch':time.time(),'pid':pid,'ppid':int(fields[1]),'session':int(fields[3]),
 'start_ticks':fields[19],'code_sha':a['code_sha'],'registration_sha256':a['registration_sha256'],
 'state':fields[0],'fds':fds,'flock_held':held,
 'config_sha256':a['config_sha256'],'source_verification':json.loads(check.stdout),
 'gate_panel_sha256':hashlib.sha256((arm/'gate_panel.json').read_bytes()).hexdigest(),
 'free_bytes':shutil.disk_usage(v5).free,
 'launch_record_exists':(arm/'LAUNCHED.json').exists(),
 'loop_state_exists':(v5/'main14_loop/LOOP14_STATE.json').exists(),
 'wait_owner':(arm/'WAIT_OWNER.json').exists(),'expired':(arm/'EXPIRED.json').exists(),
 'prearm_smoke_status':json.loads((arm/'prearm_smoke/RUN_SUMMARY.json').read_text())['status'],
 'installed_record':json.loads((arm/'WAITER_INSTALLED.json').read_text()),
 'log_tail':(v5/'main14_autolaunch.log').read_text()[-3000:]}
assert not record['wait_owner'] and not record['expired']
print(json.dumps(record))
"""))
(ROOT / "tools/main14/ARMING_RECEIPT.json").write_text(json.dumps(record, indent=1) + "\n")
installed = {**record["installed_record"], "installer_ppid": record["installed_record"]["ppid"],
             "ppid": record["ppid"], "state": record["state"], "fds": record["fds"],
             "flock_held": record["flock_held"], "verified_epoch": record["captured_epoch"],
             "check_only": record["source_verification"]}
(ROOT / "tools/main14/WAITER_INSTALLED.json").write_text(json.dumps(installed, indent=1) + "\n")
print(json.dumps(record, indent=1))
