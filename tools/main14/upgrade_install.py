"""Gracefully disarm only MAIN14's unlaunched waiter and archive its complete first install."""
from deploy import remote

print(remote("gpuR", """
import json,pathlib,shutil,time
v5=pathlib.Path('/root/autodl-tmp/c4/v5'); arm=v5/'main14_arm'
assert not (arm/'LAUNCHED.json').exists() and not (v5/'main14_loop/LOOP14_STATE.json').exists()
record=json.loads((arm/'WAITER_INSTALLED.json').read_text())
pid=record['pid']; expected=record['start_ticks']
def identity():
 try:
  fields=pathlib.Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
  return None if fields[0]=='Z' else fields[19]
 except FileNotFoundError: return None
assert identity()==expected, 'the owned waiter is not live with its recorded identity'
cmd=pathlib.Path('/proc/'+str(pid)+'/cmdline').read_bytes().decode().replace('\\0',' ')
assert 'repo_main14/scripts/c4/main14_autolaunch.py' in cmd
(arm/'STOP_ARM').touch()
until=time.monotonic()+150
while identity()==expected and time.monotonic()<until: time.sleep(2)
assert identity()!=expected, 'waiter has not yet stopped; leave files intact'
archive=v5/'main14_arm_install1'; assert not archive.exists()
shutil.copy2(v5/'main14_REGISTRATION.json',arm/'REGISTERED_AT_INSTALL.json')
shutil.copy2(v5/'main14_config.json',arm/'CONFIG_AT_INSTALL.json')
arm.rename(archive); arm.mkdir()
shutil.copy2(archive/'gate_panel.json',arm/'gate_panel.json')
(arm/'prearm_smoke').mkdir()
shutil.copy2(archive/'prearm_smoke/RUN_SUMMARY.json',arm/'prearm_smoke/RUN_SUMMARY.json')
print(json.dumps({'owned_waiter_gracefully_stopped':pid,'old_install_retained':str(archive),
 'preserved_gate_panel':True,'preserved_prearm_smoke_receipt':True}))
"""))
