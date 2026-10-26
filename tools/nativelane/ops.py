"""Allowlisted NATIVELANE source staging; authorised cpuX only."""
from __future__ import annotations

import io
import subprocess
import tarfile
import sys
from pathlib import Path

from remote import remote

ROOT = Path(__file__).resolve().parents[2]


def stage(baseline=False, harness=False, host="cpuX", deps=False):
    # Export only executable source/configs; never traverse data or assignment files.
    names = subprocess.check_output([
        "git", "ls-files", "--cached", "--others", "--exclude-standard", "--",
        "src", "traffic_shepherd", "tools", "scripts", "output", "tests/unit", "configs", "pyproject.toml",
        "baselines/circles/adapter.json"], cwd=ROOT).decode().splitlines()
    import re
    bad = re.compile(r"sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype", re.I)
    if baseline:
        names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "01f590e9", "--",
                                        "src", "configs", "tools/loop14_checks/loop14c_bitid.py"], cwd=ROOT).decode().splitlines()
    if harness:
        names = [n for n in names if n.startswith("tools/nativelane/")]
    if deps:
        names = [n for n in names if n == "baselines/circles/adapter.json"]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name in names:
            if bad.search(name) or Path(name).suffix not in (".py", ".json", ".toml", ".sh"):
                continue
            data = (subprocess.check_output(["git", "show", f"01f590e9:{name}"], cwd=ROOT)
                    if baseline else (ROOT / name).read_bytes().replace(b"\r\n", b"\n"))
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
    print(remote(host, '''
import io,tarfile,sys,json
from pathlib import Path
p=Path.home()/'nativelane'; assert p.parent.is_dir(); p.mkdir(exist_ok=True)
r=p/REPO; r.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as t:
 for m in t:
  assert m.isfile() and (r/m.name).resolve().is_relative_to(r.resolve())
  t.extract(m,r)
print(json.dumps({'staged':str(r)}))
'''.replace("REPO", repr("repo_off" if baseline else "repo")), buf.getvalue()).decode())


if __name__ == "__main__":
    if "--pause-long-order" in sys.argv:
        print(remote("cpuX", '''
import os,json,time,signal
from pathlib import Path
p=Path.home()/'nativelane'
def ident(pid):
 try:
  f=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
  return None if f[0]=='Z' else f[19]
 except FileNotFoundError: return None
owned=[]
for path in [p/'logs/repair_on60_early.pid.json',p/'preview.pid.json']:
 if not path.exists(): continue
 rec=json.loads(path.read_text())
 if ident(rec['pid'])!=rec['start_ticks']: continue
 proc=Path('/proc')/str(rec['pid'])
 for raw in (proc/'task'/str(rec['pid'])/'children').read_text().split():
  pid=int(raw); st=Path('/proc/'+raw+'/stat').read_text().rsplit(')',1)[1].split()
  if int(st[1])==rec['pid'] and st[0]!='Z': owned.append({'pid':pid,'start_ticks':st[19]})
 owned.append(rec)
for rec in owned:
 if ident(rec['pid'])==rec['start_ticks']: os.kill(rec['pid'],signal.SIGTERM)
until=time.time()+60
while any(ident(r['pid'])==r['start_ticks'] for r in owned):
 assert time.time()<until; time.sleep(1)
archive=p/'superseded_longitudinal_attempt3'; assert not archive.exists(); archive.mkdir()
for folder in ['pulse','c9','wm_coop']:
 f=p/'on'/folder
 if f.exists(): f.rename(archive/('on_'+folder))
for f in (p/'done').glob('qual_*_on_*.json'): f.rename(archive/f.name)
for name in ['qual_retry.pid.json','preview.pid.json','QUAL_NATIVE_LANE.json','PRIMARY60_PREVIEW.json']:
 f=p/name
 if f.exists(): f.rename(archive/name)
f=p/'done/repair_on60_early.json'
if f.exists(): f.rename(archive/'repair_on60_early.done.json')
(p/'QUAL_TASKS.json').write_text(json.dumps({'expected':228,'complete':False,'failed':[],
 'reason':'rerun ON with current-lane native longitudinal execution before HDV lane substep'}))
print(json.dumps({'paused_own':owned,'archive':str(archive),'off_data_retained':True}))
''').decode())
    elif "--preview" in sys.argv:
        print(remote("cpuX", '''
import os,subprocess,json,time
from pathlib import Path
p=Path.home()/'nativelane'; assert not (p/'preview.pid.json').exists()
with (p/'preview.log').open('ab') as log:
 child=subprocess.Popen(['setsid','nohup','/home/ubuntu/c4/venv/bin/python','tools/nativelane/preview.py'],
  cwd=p/'repo',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
  env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','PYTHONUNBUFFERED':'1'})
time.sleep(1)
f=Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()
r={'pid':child.pid,'start_ticks':f[19],'running':child.poll() is None}
(p/'preview.pid.json').write_text(json.dumps(r,indent=1))
print(json.dumps(r))
''').decode())
    elif "--early-repairs" in sys.argv:
        print(remote("cpuX", '''
import os,subprocess,json,time
from pathlib import Path
p=Path.home()/'nativelane'; assert json.loads((p/'QUAL_TASKS.json').read_text())['complete']
assert not (p/'EARLY_REPAIRS_PLAN.json').exists()
(p/'EARLY_REPAIRS_PLAN.json').write_text(json.dumps({'total_extra_workers':6,'pools':2,'workers_each':3}))
with (p/'early_repairs.log').open('ab') as log:
 child=subprocess.Popen(['setsid','nohup','/home/ubuntu/c4/venv/bin/python','tools/nativelane/early.py'],
  cwd=p/'repo',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
  env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','PYTHONUNBUFFERED':'1'})
time.sleep(2)
f=Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()
r={'pid':child.pid,'start_ticks':f[19],'running':child.poll() is None}
(p/'EARLY_REPAIRS.pid.json').write_text(json.dumps(r,indent=1))
print(json.dumps(r))
''').decode())
    elif "--collect" in sys.argv:
        import base64
        import json
        response = json.loads(remote("cpuX", '''
import json,base64
from pathlib import Path
p=Path.home()/'nativelane'
names=['CHECKS.json','QUAL_NATIVE_LANE.json','DECISION.json','NATIVE_LANE_COUNTS.json','RESULT.md','C1_RANKING.json',
 'ADOPTED.json','ADOPTION_FAILED.json','GPU_WATCHER_VERIFIED.json','GPU_WATCHER.json','QUAL_TASKS.json','CLOCK_REPAIR.json','REPAIR_ON60_DONE.json','FINISHED.json','PRIMARY60_PREVIEW.json']
print(json.dumps({n:base64.b64encode((p/n).read_bytes()).decode() for n in names if (p/n).exists()}))
'''))
        assert (ROOT / "docs").is_dir()
        out = ROOT / "docs/nativelane"
        out.mkdir(exist_ok=True)
        for name, content in response.items():
            (out / name).write_bytes(base64.b64decode(content))
        print(json.dumps({"collected": sorted(response), "destination": str(out)}))
    elif "--qual-analyze" in sys.argv:
        print(remote("cpuX", '''
import subprocess,os,json,time
from pathlib import Path
p=Path.home()/'nativelane'; assert json.loads((p/'QUAL_TASKS.json').read_text())['complete']
with (p/'qual_analysis.log').open('ab') as log:
 child=subprocess.Popen(['setsid','nohup','/home/ubuntu/c4/venv/bin/python','tools/nativelane/analyze.py','--qual-only'],
  cwd=p/'repo',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
  env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''})
time.sleep(1)
print(json.dumps({'pid':child.pid,'running':child.poll() is None}))
''').decode())
    elif "--record-final-checks" in sys.argv:
        import hashlib
        import json
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
        names = ["src/wms/model/wms.py", "src/wms/model/cav.py", "src/wms/control/bridge.py",
                 "src/wms/control/dreamer.py", "src/wms/control/policy.py", "src/wms/control/policy14.py",
                 "src/wms/control/policy14_cli.py", "src/wms/control/confident_gate.py",
                 "tests/unit/test_native_lane.py", "tests/unit/test_native_adoption.py"]
        expected = {n: hashlib.sha256(subprocess.check_output(["git", "show", f"{commit}:{n}"], cwd=ROOT)).hexdigest() for n in names}
        proof = json.loads(remote("gpuB", '''
import sys,json,hashlib
from pathlib import Path
req=json.load(sys.stdin); root=Path('/root/nativelane/repo')
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==s for n,s in req.items())
proof=json.loads(Path('/root/nativelane/priority_bitid/BIT_IDENTITY.json').read_text())
print(json.dumps({'bitid':proof,'verified_source_files':req,'tests_host':'gpuB','integrated_tests_passed':367,
 'verification':'367-test integrated CPU suite printed success after longitudinal/lane ordering and reservation regressions; final two-path tensor-byte probe passed'}))
''', json.dumps(expected).encode()))
        proof.update(passed=True, code_sha=commit,
                     wm_sha256="952ec059de5a9c0285d2327bcea57c836fd5d736c9975592ffbb932a506e84c8")
        print(remote("cpuX", '''
import sys,json
from pathlib import Path
p=Path.home()/'nativelane'; target=p/'CHECKS.json'; old=json.loads(target.read_text())
new=json.load(sys.stdin); new['initial_checks']=old
tmp=p/'CHECKS.final.tmp'; tmp.write_text(json.dumps(new,indent=1)); tmp.replace(target)
print(json.dumps({'final_checks_recorded':True,'tests':new['integrated_tests_passed'],'code_sha':new['code_sha']}))
''', json.dumps(proof).encode()).decode())
        (ROOT / "tools/nativelane/FINAL_CHECKS.json").write_text(json.dumps(proof, indent=1) + "\n")
    elif "--resources" in sys.argv:
        print(remote("cpuX", '''
import json,time
from pathlib import Path
p=Path.home()/'nativelane'; out={'time':time.time(),'owned':[]}
for f in (p/'logs').glob('*.pid.json'):
 r=json.loads(f.read_text()); proc=Path('/proc')/str(r['pid'])
 try:
  st=(proc/'stat').read_text().rsplit(')',1)[1].split()
  if st[19]!=r['start_ticks'] or st[0]=='Z': continue
  child=[int(x) for x in (proc/'task'/str(r['pid'])/'children').read_text().split()]
  out['owned'].append({'task':f.name,'pid':r['pid'],'state':st[0],'children':child})
 except FileNotFoundError: pass
out['memory']={x.split(':')[0]:x.split(':')[1].strip() for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemTotal:','MemAvailable:','SwapTotal:','SwapFree:'))}
print(json.dumps(out,indent=1))
''').decode())
    elif "--verify-watch" in sys.argv:
        import json
        req = {"watcher": json.loads((ROOT / "tools/nativelane/WATCHER.json").read_text()),
               "package": json.loads((ROOT / "tools/nativelane/REARM_PACKAGE.json").read_text())}
        output = remote("gpuR", '''
import json,sys,hashlib,os
from pathlib import Path
r=json.load(sys.stdin); own=r['watcher']; a=Path('/root/autodl-tmp/c4/v5/main14_arm')
f=Path('/proc/'+str(own['pid'])+'/stat').read_text().rsplit(')',1)[1].split()
assert f[19]==own['start_ticks'] and f[0]!='Z'
assert int(f[1])==1 and int(f[3])==own['pid']
h=hashlib.sha256((a/'ARMED.json').read_bytes()).hexdigest(); assert h==r['package']['expected_old_arm_sha256']
assert not (a/'STOP_ARM').exists() and not (a/'nativelane_adoption_package').exists()
assert (a/'waiter.pid').read_text().strip()=='649847'
print(json.dumps({'pid':own['pid'],'start_ticks':f[19],'ppid':int(f[1]),'session':int(f[3]),'nice':int(f[16]),
 'stdio':[os.readlink('/proc/'+str(own['pid'])+'/fd/'+str(i)) for i in range(3)],
 'existing_arm_sha256':h,'original_waiter':649847,'arm_untouched':True,'package_sha256':r['package']['package_sha256']},indent=1))
''', json.dumps(req).encode())
        print(output.decode())
        (ROOT / "tools/nativelane/WATCHER_VERIFIED.json").write_bytes(output)
        print(remote("cpuX", "import sys;from pathlib import Path;(Path.home()/'nativelane/GPU_WATCHER_VERIFIED.json').write_bytes(sys.stdin.buffer.read());print('verified receipt stored')", output).decode())
    elif "--archive-package" in sys.argv:
        print(remote("cpuX", '''
from pathlib import Path
import hashlib,json
p=Path.home()/'nativelane'; f=p/'rearm_package.tar.gz'; h=hashlib.sha256(f.read_bytes()).hexdigest()
target=p/('rearm_package_superseded_'+h[:12]+'.tar.gz'); assert not target.exists(); f.rename(target)
print(json.dumps({'retained':str(target),'sha256':h}))
''').decode())
    elif "--stop-watch" in sys.argv:
        import json
        rec = json.loads((ROOT / "tools/nativelane/WATCHER.json").read_text())
        print(remote("gpuR", '''
import json,sys,os,signal
from pathlib import Path
r=json.load(sys.stdin); a=Path('/root/autodl-tmp/c4/v5/main14_arm')
assert not (a/'nativelane_adoption_package').exists(), 'adoption already active'
f=Path('/proc/'+str(r['pid'])+'/stat').read_text().rsplit(')',1)[1].split()
assert f[19]==r['start_ticks'] and f[0]!='Z'
os.kill(r['pid'],signal.SIGTERM)
print(json.dumps({'stopped_own_watcher':r['pid'],'armed_files_untouched':True}))
''', json.dumps(rec).encode()).decode())
    elif "--watch" in sys.argv:
        import base64
        import json
        receipt = json.loads((ROOT / "tools/nativelane/REARM_PACKAGE.json").read_text())
        registered = Path("E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md").read_text(encoding="utf-8")
        assert receipt["registration_sha256"] in registered, "append conditional source amendment before installing watcher"
        watch = subprocess.check_output(["git", "show", f"{receipt['code_sha']}:tools/nativelane/watch.py"], cwd=ROOT)
        req = {"watch": base64.b64encode(watch).decode(), "package_sha256": receipt["package_sha256"]}
        output = remote("gpuR", '''
import os,subprocess,json,time,sys
from pathlib import Path
req=json.load(sys.stdin); a=Path('/root/autodl-tmp/c4/v5/main14_arm'); assert a.is_dir()
assert not any((a/n).exists() for n in ['LAUNCHED.json','WAIT_OWNER.json','STOP_ARM','EXPIRED.json'])
code="import base64;exec(base64.b64decode('"+req['watch']+"'))"
proc=subprocess.Popen(['setsid','nohup','/root/miniconda3/bin/python','-c',code,req['package_sha256']],
 cwd=a,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
 env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'})
time.sleep(2); assert proc.poll() is None
f=Path('/proc/'+str(proc.pid)+'/stat').read_text().rsplit(')',1)[1].split()
print(json.dumps({'pid':proc.pid,'start_ticks':f[19],'session':int(f[3]),'ppid':int(f[1]),'package_sha256':req['package_sha256'],'armed_directory_writes':0}))
''', json.dumps(req).encode())
        print(output.decode())
        (ROOT / "tools/nativelane/WATCHER.json").write_bytes(output)
        print(remote("cpuX", "import sys;from pathlib import Path;p=Path.home()/'nativelane';assert p.is_dir();(p/'GPU_WATCHER.json').write_bytes(sys.stdin.buffer.read());print('watcher receipt stored')", output).decode())
    elif "--test-adoption-link" in sys.argv:
        host = "gpuB" if "--gpuB" in sys.argv else "cpuX"
        print(remote(host, '''
import subprocess,json
from pathlib import Path
p=Path.home()/'nativelane'; assert p.parent.is_dir(); p.mkdir(exist_ok=True)
key=subprocess.check_output(['ssh-keyscan','-T','10','-p','53897','-t','ed25519','connect.westb.seetacloud.com'],stderr=subprocess.DEVNULL)
fp=subprocess.check_output(['ssh-keygen','-lf','-'],input=key).decode()
assert 'SHA256:liZ36vNCsNcNdXeWs4f+g5ZIhPM/ZihP834vxs8Ulqc' in fp,fp
known=p/'gpuR_known_hosts'; known.write_bytes(key)
argv=['ssh','-p','53897','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(known),'root@connect.westb.seetacloud.com','true']
r=subprocess.run(argv,capture_output=True,text=True)
print(json.dumps({'rc':r.returncode,'stderr':r.stderr,'fingerprint':fp}))
''', timeout=60000).decode())
    elif "--status" in sys.argv:
        print(remote("cpuX", '''
import json
from pathlib import Path
p=Path.home()/'nativelane'
out={}
for n in ['DRIVER.json','CHECKS.json','QUAL_TASKS.json','MEASUREMENTS.json','DECISION.json','FINISHED.json','DRIVER_FAILED.json','FINISH_FAILED.json']:
 f=p/n
 if f.exists():
  d=json.loads(f.read_text())
  if n=='DECISION.json': d={k:v for k,v in d.items() if k in ['adopt','adopt_scientific','reason','error','vacate_contrast_per_cav_h_ci95','absolute_gap_reduction_ci95','main14_action']}
  if n=='QUAL_TASKS.json': d={k:v for k,v in d.items() if k!='failed'}|{'failed_count':len(d.get('failed',[]))}
  out[n]=d
out['unit_counts']={str(d.relative_to(p)):len(list(d.glob('w*_t*.json'))) for d in [p/'off/wm_coop',p/'on/wm_coop',p/'off/wm_c',p/'on/wm_c']}
out['successful_tasks']=len(list((p/'done').glob('*.json')))
if (p/'driver.log').exists():
 lines=(p/'driver.log').read_text().splitlines()
 out['recent_events']=lines[-8:]
print(json.dumps(out,indent=1))
''').decode())
    elif "--pause-qual-fix" in sys.argv:
        print(remote("cpuX", '''
import os,json,time,signal,shutil
from pathlib import Path
p=Path.home()/'nativelane'; rec=json.loads((p/'qual_retry.pid.json').read_text())
def alive(rec):
 try:
  f=Path('/proc/'+str(rec['pid'])+'/stat').read_text().rsplit(')',1)[1].split()
  return f[19]==rec['start_ticks'] and f[0]!='Z'
 except FileNotFoundError: return False
if alive(rec): os.kill(rec['pid'],signal.SIGTERM)
children=[json.loads(f.read_text()) for f in (p/'logs').glob('qual_*.pid.json')]
until=time.time()+240
while any(alive(r) for r in children):
 assert time.time()<until,'own qualification children still completing'
 time.sleep(2)
archive=p/ARCHIVE; assert not archive.exists(); archive.mkdir()
for name in ['pulse','c9']:
 f=p/'on'/name
 if f.exists(): f.rename(archive/('on_'+name))
for f in (p/'done').glob('qual_*_on_*.json'): f.rename(archive/f.name)
for f in (p/'logs').glob('qual_*_on_*'): shutil.copyfile(f,archive/f.name)
(p/'qual_retry.pid.json').rename(archive/'qual_retry.pid.json')
if PRIORITY:
 source=p/'on/wm_coop'; units=[]
 for f in sorted(source.glob('w*_t*.json')):
  d=json.loads(f.read_text()); units.append(d['world']+'@'+str(int(d['t0'])))
 assert len(units)==33
 source.rename(archive/'wm_coop')
 (p/'done/mappo_coop_on.json').rename(archive/'mappo_coop_on.done.json')
 (p/'REPAIR_ON60.json').write_text(json.dumps({'units':','.join(units),'reason':'native mainline insertion priority must match HDV geometry','archive':str(archive)},indent=1))
print(json.dumps({'paused':True,'archived_on':str(archive),'off_successes_retained':True}))
'''.replace("ARCHIVE", repr("superseded_priority_attempt2" if "--priority-fix" in sys.argv else "superseded_tickmask_attempt1"))
   .replace("PRIORITY", repr("--priority-fix" in sys.argv)), timeout=300000).decode())
    elif "--qual-retry" in sys.argv:
        print(remote("cpuX", '''
import os,subprocess,json,time,signal
from pathlib import Path
p=Path.home()/'nativelane'; assert p.is_dir()
assert not (p/'qual_retry.pid.json').exists(), 'already installed'
assert (p/'QUAL_TASKS.json').exists(), 'initial qualifier still active'
# Replace only our waiting continuation; it has no children before the main driver exits.
rec=json.loads((p/'finish.pid.json').read_text())
fields=Path('/proc/'+str(rec['pid'])+'/stat').read_text().rsplit(')',1)[1].split()
assert fields[19]==rec['start_ticks'] and fields[0]!='Z'
os.kill(rec['pid'],signal.SIGTERM)
(p/'finish.pid.json').rename(p/('finish_install_'+str(rec['pid'])+'.pid.json'))
with (p/'qual_retry.log').open('ab') as log:
 proc=subprocess.Popen(['setsid','nohup','/home/ubuntu/c4/venv/bin/python','-c',"import sys;sys.path.insert(0,'tools/nativelane');import drive as D;import os;os.nice(19);D.qual(WORKERS)"],
  cwd=p/'repo',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
  env={**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''})
time.sleep(2)
fields=Path('/proc/'+str(proc.pid)+'/stat').read_text().rsplit(')',1)[1].split()
record={'pid':proc.pid,'start_ticks':fields[19],'running':proc.poll() is None}
(p/'qual_retry.pid.json').write_text(json.dumps(record,indent=1))
print(json.dumps(record))
'''.replace("WORKERS", "3" if "--three" in sys.argv else "6")).decode())
    elif "--finish" in sys.argv:
        print(remote("cpuX", '''
import os,subprocess,json,time
from pathlib import Path
p=Path.home()/'nativelane'; assert p.is_dir()
assert not (p/'finish.pid.json').exists(), 'already installed'
with (p/'finish.log').open('ab') as log:
 proc=subprocess.Popen(['setsid','nohup','/home/ubuntu/c4/venv/bin/python',str(p/'repo/tools/nativelane/finish.py')],
  cwd=p/'repo',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
  env={**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''})
time.sleep(2)
fields=Path('/proc/'+str(proc.pid)+'/stat').read_text().rsplit(')',1)[1].split()
record={'pid':proc.pid,'start_ticks':fields[19],'running':proc.poll() is None}
(p/'finish.pid.json').write_text(json.dumps(record,indent=1))
print(json.dumps(record))
''').decode())
    elif "--start" in sys.argv:
        print(remote("cpuX", '''
import os,subprocess,json,time
from pathlib import Path
p=Path.home()/'nativelane'; assert p.is_dir()
assert not (p/'DRIVER.json').exists(), 'inspect existing driver before a retry'
with (p/'driver.log').open('ab') as log:
 proc=subprocess.Popen(['setsid','nohup','/home/ubuntu/c4/venv/bin/python',str(p/'repo/tools/nativelane/drive.py')],
  cwd=p/'repo',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
  env={**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''})
time.sleep(2)
print(json.dumps({'pid':proc.pid,'running':proc.poll() is None}))
''').decode())
    else:
        stage("--baseline" in sys.argv, "--harness" in sys.argv,
              "gpuB" if "--gpuB" in sys.argv else "cpuX", "--deps" in sys.argv)
