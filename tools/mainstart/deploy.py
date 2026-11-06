"""Install and launch the separately owned MAINSTART wrapper, never overwrite native code."""
import base64
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from ops import remote

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tools/mainstart'
V5 = '/root/autodl-tmp/c4/v5'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare():
    info = json.loads(remote('gpuR', r'''
import json,pathlib,shutil
v=pathlib.Path('/root/autodl-tmp/c4/v5')
def identity(pid):
 p=pathlib.Path('/proc/'+str(pid)+'/stat')
 if not p.exists(): return None
 f=p.read_text().rsplit(')',1)[1].split()
 return None if f[0]=='Z' else {'start_ticks':f[19],'ppid':f[1]}
assert identity(657216) is None, 'waiter has not exited by itself'
assert (v/'main14_arm/STOP_ARM').is_file()
assert not (v/'main14_loop').exists() and not pathlib.Path('/root/main14_loop').exists()
print(json.dumps({'arm':json.loads((v/'main14_arm/ARMED.json').read_text()),
 'config':json.loads((v/'main14_config.json').read_text()),
 'registration':json.loads((v/'main14_REGISTRATION.json').read_text()),
 'pairft':{str(pid):identity(pid)['start_ticks'] for pid in (717015,718536) if identity(pid)},
 'root_free':shutil.disk_usage('/root').free,'scratch_free':shutil.disk_usage(v).free}))
'''))
    cfg, reg = copy.deepcopy(info['config']), copy.deepcopy(info['registration'])
    cfg['loop_root'] = '/root/main14_loop'
    cfg['collect'].update(workers=12, work_root='/home/ubuntu/c4/main14')
    hosts = {}
    for alias in ('cpuP', 'cpuQ'):
        hosts[alias] = {'python': '/home/ubuntu/miniconda3/bin/python',
                        'code_root': '/home/ubuntu/c4/main14/runtime',
                        'ssh': ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', alias],
                        'rsync_target': alias,
                        'rsync_ssh': ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']}
    cfg['hosts']['cpu'] = hosts['cpuP']
    cfg['mainstart'] = {'collect_hosts': hosts}
    cfg['policy']['env']['C4_GPU_MEM_FRACTION'] = '0.45'
    cfg['policy']['allowed_roots'].append(V5 + '/mainstart')
    cfg['policy']['extra_args'] = [x.replace('/main14_arm/', '/mainstart/') for x in cfg['policy']['extra_args']]
    config_text = json.dumps(cfg, indent=1) + '\n'
    amendment = {
        'id': 'S-MAINSTART', 'owner_request': '2026-10-26 ~12:00 UTC+8: start policy while scores pending',
        'reason': 'cpuX scoring host spot reclaimed; owner requested immediate start',
        'initial_base': '13E b2 FINAL', 'initial_base_sha256': reg['sources']['fallback_wm_sha256'],
        'collect_host': 'cpuP', 'fallback_host': 'cpuQ', 'sumo_workers_max': 12,
        'collect_wait': 'Readiness/code/fixture/RAM checks before phase entry or seed charge',
        'later_base_swap': 'cycle>=1 unstarted policy boundary only; best eligible of 14b/round16 by lower strict TTC ratio, tie14b; no new aggregate/per-set HDV reaction sign/late-sign/magnitude FAIL versus13E and strict TTC ratio<=13E; FINAL SHA must match scored SHA; record every decision',
        'native_lane_change': True, 'speed2': 'OFF; fastloop changes uncommitted at launch',
        'policy_cuda_memory_fraction': 0.45, 'pairft_incarnations': info['pairft'],
        'wm_gpu_guard': 'exclusive, unchanged; waits if any CUDA process remains',
        'loop_output': '/root/main14_loop', 'output_free_bytes_before_launch': info['root_free'],
        'scratch_free_bytes_after_archive': info['scratch_free'],
        'storage': 'Protected scratch occupancy prevents6GB safely; output filesystem has>=6GiB;1.5GiB phase floor retained',
        'wrapper_code_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'registered_at': datetime.now(timezone.utc).isoformat(),
    }
    reg['mainstart_amendment'] = amendment
    reg['sources'].update(initial_wm_rule='13E b2 FINAL now; S-MAINSTART later-boundary eligibility rule',
                          config_sha256=sha(config_text.encode()))
    reg['collection'].update(host='cpuP', fallback='cpuQ', workers_max=12)
    reg['resources'].update(cuda_memory_fraction=0.45, loop_output='/root/main14_loop')
    reg['gates'].update(before_launch='S-MAINSTART: source/hash checks; exact PAIRFT concurrency only; integrated tests and2-update real-WM CPU smoke; frozen panel;>=6GiB output space',
                        verdict_deadline='superseded for initial launch by S-MAINSTART')
    reg_text = json.dumps(reg, indent=1) + '\n'
    wrapper = (ROOT / 'scripts/c4/mainstart.py').read_bytes()
    arm = copy.deepcopy(info['arm'])
    arm.update(config=V5 + '/mainstart/CONFIG.json', config_sha256=sha(config_text.encode()),
               registration_sha256=sha(reg_text.encode()), wrapper_sha256=sha(wrapper),
               pairft_incarnations=info['pairft'],
               native_source_manifest_sha256=info['registration']['sources']['source_manifest_sha256'],
               amendment='S-MAINSTART', wrapper_code_sha=amendment['wrapper_code_sha'])
    for name, doc in [('PRIOR_STATE.json', info), ('CONFIG.json', cfg), ('REGISTERED.json', reg), ('ARMED.json', arm)]:
        (OUT / name).write_text(json.dumps(doc, indent=1) + '\n')
    request = {'config': config_text, 'registration': reg_text, 'arm': arm,
               'wrapper': base64.b64encode(wrapper).decode(), 'previous': info}
    out = remote('gpuR', r'''
import json,pathlib,sys,base64,shutil,hashlib
q=json.load(sys.stdin); v=pathlib.Path('/root/autodl-tmp/c4/v5'); assert v.is_dir()
a=v/'mainstart'; assert not a.exists(); a.mkdir()
for name,rel in [('PRIOR_ARMED.json','main14_arm/ARMED.json'),('PRIOR_CONFIG.json','main14_config.json'),('PRIOR_REGISTRATION.json','main14_REGISTRATION.json')]:
 shutil.copyfile(v/rel,a/name)
shutil.copyfile(v/'main14_arm/gate_panel.json',a/'gate_panel.json')
assert hashlib.sha256((a/'gate_panel.json').read_bytes()).hexdigest()==q['arm']['gate_panel_sha256']
(a/'mainstart.py').write_bytes(base64.b64decode(q['wrapper']))
(a/'CONFIG.json').write_text(q['config'])
(v/'main14_REGISTRATION.json').write_text(q['registration'])
(a/'REGISTERED.json').write_text(q['registration'])
(a/'ARMED.json').write_text(json.dumps(q['arm'],indent=1)+'\n')
print(json.dumps({'installed':True,'wrapper_sha256':q['arm']['wrapper_sha256'],
 'registration_sha256':hashlib.sha256((v/'main14_REGISTRATION.json').read_bytes()).hexdigest(),
 'config_sha256':q['arm']['config_sha256'],'root_free':shutil.disk_usage('/root').free}))
''', json.dumps(request).encode())
    (OUT / 'INSTALL_RECEIPT.json').write_text(out)
    print(out)


def launch():
    out = remote('gpuR', r'''
import json,os,pathlib,subprocess,time
v=pathlib.Path('/root/autodl-tmp/c4/v5'); a=v/'mainstart'; assert a.is_dir()
assert not (a/'PID.json').exists() and not (a/'WAIT_OWNER.json').exists()
argv=['/root/miniconda3/bin/python','-u',str(a/'mainstart.py'),'--arm',str(a/'ARMED.json')]
subprocess.run([*argv,'--check-only'],check=True)
with (a/'driver.log').open('ab') as log:
 p=subprocess.Popen(['setsid','nohup',*argv],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
 cwd=v/'repo_main14',env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'})
time.sleep(3); assert p.poll() is None, 'driver exited; inspect log'
f=pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text().rsplit(')',1)[1].split()
receipt={'pid':p.pid,'ppid':int(f[1]),'sid':int(f[3]),'start_ticks':f[19],'log':str(a/'driver.log')}
(a/'INSTALLED.json').write_text(json.dumps(receipt,indent=1))
print(json.dumps(receipt))
''', timeout=240)
    (OUT / 'LAUNCH_RECEIPT.json').write_text(out)
    print(out)


def handoff():
    content = (OUT / 'handoff.py').read_bytes()
    out = remote('gpuR', r'''
import json,base64,pathlib,subprocess,os,time,sys,hashlib
a=pathlib.Path('/root/autodl-tmp/c4/v5/mainstart'); assert a.is_dir()
assert not (a/'HANDOFF_PID.json').exists()
data=base64.b64decode(sys.stdin.buffer.read()); (a/'handoff.py').write_bytes(data)
with (a/'handoff.log').open('ab') as log:
 p=subprocess.Popen(['setsid','nohup','/root/miniconda3/bin/python','-u',str(a/'handoff.py')],
 stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=a,
 env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
time.sleep(3); assert p.poll() is None
f=pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text().rsplit(')',1)[1].split()
receipt={'pid':p.pid,'sid':int(f[3]),'start_ticks':f[19],'source_sha256':hashlib.sha256(data).hexdigest()}
(a/'HANDOFF_INSTALLED.json').write_text(json.dumps(receipt,indent=1))
print(json.dumps(receipt))
''', base64.b64encode(content))
    (OUT / 'HANDOFF_INSTALLED.json').write_text(out)
    print(out)


def continuation():
    request = {'wrapper': base64.b64encode((ROOT / 'scripts/c4/mainstart.py').read_bytes()).decode(),
               'supervisor': base64.b64encode((OUT / 'continue.py').read_bytes()).decode(),
               'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    out = remote('gpuR', r'''
import json,base64,pathlib,subprocess,os,time,sys,hashlib
q=json.load(sys.stdin); a=pathlib.Path('/root/autodl-tmp/c4/v5/mainstart'); assert a.is_dir()
assert not (a/'CONTINUATION_REQUEST.json').exists()
old=json.loads((a/'PID.json').read_text()); assert old['pid']==822919
f=pathlib.Path('/proc/'+str(old['pid'])+'/stat').read_text().rsplit(')',1)[1].split()
assert f[19]==old['start_ticks']
state=json.loads(pathlib.Path('/root/main14_loop/LOOP14_STATE.json').read_text())
assert state['cycles'][0]['phases']['policy']['status']=='running' and state['sim_s_charged']==0
assert not (a/'WAIT_OWNER.json').exists()
wrapper=base64.b64decode(q['wrapper']); (a/'mainstart_v2.py').write_bytes(wrapper)
(a/'continue.py').write_bytes(base64.b64decode(q['supervisor']))
arm=json.loads((a/'ARMED.json').read_text()); arm['wrapper_sha256']=hashlib.sha256(wrapper).hexdigest(); arm['wrapper_code_sha']=q['commit']
(a/'ARMED_V2.json').write_text(json.dumps(arm,indent=1)+'\n')
request={'previous_pid':old['pid'],'previous_start_ticks':old['start_ticks'],'wrapper_sha256':arm['wrapper_sha256'],'commit':q['commit']}
(a/'CONTINUATION_REQUEST.json').write_text(json.dumps(request,indent=1))
with (a/'continuation.log').open('ab') as log:
 p=subprocess.Popen(['setsid','nohup','/root/miniconda3/bin/python','-u',str(a/'continue.py')],
 stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=a,
 env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
time.sleep(3); assert p.poll() is None
# Own driver's documented phase-boundary marker; the active policy child is never interrupted.
(a/'STOP_ARM').touch()
f=pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text().rsplit(')',1)[1].split()
print(json.dumps({**request,'supervisor_pid':p.pid,'supervisor_sid':int(f[3]),'supervisor_start_ticks':f[19]}))
''', json.dumps(request).encode())
    (OUT / 'CONTINUATION_REQUEST.json').write_text(out)
    print(out)


if __name__ == '__main__':
    {'prepare': prepare, 'launch': launch, 'handoff': handoff, 'continuation': continuation}[sys.argv[1]]()
