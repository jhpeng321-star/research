"""Allowlisted NATIVELANE source staging; authorised cpuX only."""
from __future__ import annotations

import io
import subprocess
import tarfile
import sys
from pathlib import Path

from remote import remote

ROOT = Path(__file__).resolve().parents[2]


def stage(baseline=False, harness=False):
    # Export only executable source/configs; never traverse data or assignment files.
    names = subprocess.check_output([
        "git", "ls-files", "--cached", "--others", "--exclude-standard", "--",
        "src", "traffic_shepherd", "tools", "scripts", "output", "tests/unit", "configs", "pyproject.toml"], cwd=ROOT).decode().splitlines()
    import re
    bad = re.compile(r"sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype", re.I)
    if baseline:
        names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "01f590e9", "--",
                                        "src", "configs", "tools/loop14_checks/loop14c_bitid.py"], cwd=ROOT).decode().splitlines()
    if harness:
        names = [n for n in names if n.startswith("tools/nativelane/")]
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
    print(remote("cpuX", '''
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
    if "--status" in sys.argv:
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
(p/'finish.pid.json').rename(p/'finish_install1.pid.json')
with (p/'qual_retry.log').open('ab') as log:
 proc=subprocess.Popen(['setsid','nohup','/home/ubuntu/c4/venv/bin/python','-c',"import sys;sys.path.insert(0,'tools/nativelane');import drive as D;import os;os.nice(19);D.qual()"],
  cwd=p/'repo',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
  env={**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''})
time.sleep(2)
fields=Path('/proc/'+str(proc.pid)+'/stat').read_text().rsplit(')',1)[1].split()
record={'pid':proc.pid,'start_ticks':fields[19],'running':proc.poll() is None}
(p/'qual_retry.pid.json').write_text(json.dumps(record,indent=1))
print(json.dumps(record))
''').decode())
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
        stage("--baseline" in sys.argv, "--harness" in sys.argv)
