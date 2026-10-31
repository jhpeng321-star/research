"""Short SSH operations: metadata-only exposure, immutable deploy, receipt export."""
import argparse
import base64
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

from .design import FORBIDDEN, checked, sha, worlds
from .storage import write

ROOT = '/root/c1switch20261031'
EVIDENCE = Path('tools/c1switch/evidence')
REG = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')


def ssh(host, code, timeout=60):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', host,
                             '/root/miniconda3/bin/python -'], input=code, text=True,
                            encoding='utf-8', capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr+result.stdout)
    return json.loads(result.stdout)


def audit():
    """Exact registry seed identities/ranges; corpus identities; remote names only."""
    seeds = {w['seed'] for w in worlds('step0')+worlds('main')}
    hits = []
    # Umbrella development allocation is not an experiment allocation.
    umbrellas = {(965184, 969999), (966000, 969999), (966000, 969699)}
    for n, line in enumerate(REG.read_text(encoding='utf-8').splitlines(), 1):
        if 'S-C1SWITCH' in line:
            continue
        tokens = set(int(s) for s in re.findall(r'(?<![\w.])(?:[sw])?(\d{6})(?![\w.])', line))
        matches = sorted(tokens & seeds)
        ranges = []
        for a, b in re.findall(r'(?<![\w.])(\d{6})\s*(?:-|–|—|\.\.)\s*(\d{3,6})(?!\w)', line):
            lo, hi = int(a), int(b) if len(b) == 6 else int(a[:-len(b)]+b)
            if (lo, hi) not in umbrellas and any(lo <= s <= hi for s in seeds):
                ranges.append([lo, hi])
        if matches or ranges:
            hits.append(dict(line=n, seeds=matches, ranges=ranges, text=line))
    corpus_path = Path('E:/agent-work/c4-c1gap/docs/paper_results/c1gap/exposure.json')
    corpus = json.loads(corpus_path.read_text())
    ids = set(int(x) for x in re.findall(r'[sw](\d{6})(?=[mp_/\"]|$)', json.dumps(corpus)))
    hosts = {}
    for host in ('gpuR', 'gpuA'):
        code = f'''import json, os, re
from pathlib import Path
bad = re.compile({FORBIDDEN.pattern!r}, re.I)
seeds = {seeds!r}
hits = []; count = 0
for base in ['/root']:
 for here, dirs, files in os.walk(base):
  dirs[:] = [n for n in dirs if not bad.search(str(Path(here)/n)) and not n.startswith('.') and n not in ('miniconda3','miniconda','c1switch20261031')]
  for name in dirs+files:
   path = str(Path(here)/name)
   if bad.search(path): continue
   count += 1
   if any(int(x) in seeds for x in re.findall(r'(?<![0-9])(?:[sw])?(969[0-9]{{3}})(?![0-9])', name)):
    hits.append(path)
print(json.dumps(dict(hits=hits, safe_names_checked=count)))
'''
        hosts[host] = ssh(host, code, timeout=90)
    report = dict(registration_sha256=sha(REG), registry_conflicts=hits,
                  corpus_sha256=sha(corpus_path), corpus_seed_conflicts=sorted(ids & seeds), hosts=hosts)
    report['pass'] = not hits and not (ids & seeds) and not any(h['hits'] for h in hosts.values())
    write(EVIDENCE/'EXPOSURE.json', report)
    print(json.dumps(report, indent=2))


def deploy():
    report = json.loads((EVIDENCE/'EXPOSURE.json').read_text())
    if not report['pass']:
        raise ValueError('exposure not cleared')
    buf = io.BytesIO()
    sources = {}
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        paths = sorted(Path('tools/c1switch').glob('*.py'))+[Path('tests/test_c1switch.py'), Path('docs/paper_results/C1SWITCH_DESIGN.md')]
        for path in paths:
            checked(path)
            tar.add(path, arcname=path.as_posix())
            sources[path.as_posix()] = sha(path)
        path = Path('traffic_shepherd/c4/branching.py')
        tar.add(path, arcname='engine/branching.py')
        sources['engine/branching.py'] = sha(path)
    raw = base64.b64encode(buf.getvalue()).decode()
    result = ssh('gpuR', f'''import base64, hashlib, io, json, tarfile
from pathlib import Path
root=Path({ROOT!r})
assert root.parent.is_dir()
root.mkdir(exist_ok=False)
stage=root/'pytools'; stage.mkdir()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({raw!r})),mode='r:gz') as t:
 t.extractall(stage, filter='data')
sources={sources!r}
assert all(hashlib.sha256((stage/n).read_bytes()).hexdigest()==h for n,h in sources.items())
(root/'STAGED.json').write_text(json.dumps(sources,indent=2))
(root/'EXPOSURE.json').write_text(json.dumps({report!r},indent=2))
print(json.dumps(dict(staged=True,files=len(sources),root=str(root))))
''')
    write(EVIDENCE/'STAGED.json', sources)
    print(json.dumps(result))


def preflight():
    result = ssh('gpuR', f'''import json, os, sys
os.nice(19); os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PATH']='/root/miniconda3/bin:'+os.environ['PATH']
sys.path.insert(0,{ROOT!r}+'/pytools')
from tools.c1switch.sumo import setup
setup()
from tools.c1switch.design import guard, worlds
for w in worlds('step0')+worlds('main'): guard(w)
print(json.dumps(dict(pass_=True,guarded=36,load=os.getloadavg())))
''')
    write(EVIDENCE/'PREFLIGHT.json', result)
    print(json.dumps(result))


def recorded():
    # WMDATA's already-exported cumulative receipts are available locally.
    rows, hashes = [], {}
    base = Path('E:/agent-work/c4-wmcalib/tools/wmcalib/data')
    cells = [(p, d) for p in (.05,.1,.2,.4) for d in (1900,2400,2700)]
    corners = [(.05,1900),(.05,2700),(.4,1900),(.4,2700)]
    for i in range(40):
        p,d = cells[i%12] if i<36 else corners[i-36]
        for arm in ('native','ctrl0','mappo','pi','circles'):
            for clock in (120,300,600):
                path = base/f's{969400+i}_{arm}'/f'cumulative_{clock}.json'
                value = json.loads(checked(path).read_text())
                hashes[str(path)] = sha(path)
                rows.append(dict(dataset='WMDATA', seed=969400+i, penetration=p, demand=d,
                                 arm=arm, clock_s=clock, team_delay_vs=value['team_delay_vs'],
                                 initial_state_sha256=value['initial_state_sha256']))
    result = ssh('gpuR', '''import hashlib, json
from pathlib import Path
rows=[]; hashes={}
cells=[(.2,2400),(.2,2700),(.4,2400),(.4,2700)]
other=[(.05,1900),(.05,2400),(.05,2700),(.1,1900),(.1,2400),(.1,2700),(.2,1900),(.4,1900),(.05,2700),(.1,2700)]
for i in range(50):
 p,d=cells[i%4] if i<40 else other[i-40]
 for arm in ['native','ctrl0','mappo','pi','circles']:
  for clock in [120,300,600]:
   path=Path('/root/wmdata2/jobs')/f's{969550+i}_{arm}'/f'cumulative_{clock}.json'
   raw=path.read_bytes(); v=json.loads(raw)
   hashes[str(path)]=hashlib.sha256(raw).hexdigest()
   rows.append(dict(dataset='WMDATA2',seed=969550+i,penetration=p,demand=d,arm=arm,clock_s=clock,team_delay_vs=v['team_delay_vs'],initial_state_sha256=v['initial_state_sha256']))
print(json.dumps(dict(rows=rows,hashes=hashes)))
''')
    rows += result['rows']
    hashes.update(result['hashes'])
    write(EVIDENCE/'RECORDED.json', dict(rows=rows, hashes=hashes))
    print(json.dumps(dict(rows=len(rows), worlds=90, new_sim_s=0, source_files=len(hashes))))


def cpu():
    result = ssh('gpuR', f'''import contextlib, io, json, os, sys
os.nice(19); os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PATH']='/root/miniconda3/bin:'+os.environ['PATH']
sys.path.insert(0,{ROOT!r}+'/pytools')
from tools.c1switch.sumo import setup
setup()
import pytest
log=io.StringIO()
with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
 code=pytest.main(['-q',{ROOT!r}+'/pytools/tests/test_c1switch.py'])
print(json.dumps(dict(returncode=code,output=log.getvalue())))
''')
    write(EVIDENCE/'CPU_REMOTE.json', result)
    print(json.dumps(result))


def canary():
    result = ssh('gpuR', f'''import json, os, subprocess, sys, time
from pathlib import Path
root=Path({ROOT!r}); assert root.is_dir()
assert not (root/'CANARY_DISPATCH.json').exists()
assert not (root/'STOP').exists()
assert os.getloadavg()[0] <= 20, 'yield to priority track'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PATH='/root/miniconda3/bin:'+os.environ['PATH'])
with (root/'canary.log').open('x') as log:
 proc=subprocess.Popen([sys.executable,'-B','-u','-m','tools.c1switch.runner','canary'],cwd=root/'pytools',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.nice(19))
 row=dict(pid=proc.pid,unix=time.time(),mode='canary',max_sim_s=480)
 (root/'CANARY_DISPATCH.json').write_text(json.dumps(row,indent=2))
print(json.dumps(row))
''')
    write(EVIDENCE/'CANARY_DISPATCH.json', result)
    print(json.dumps(result))


def status():
    result = ssh('gpuR', f'''import json
from pathlib import Path
root=Path({ROOT!r})
names=['CANARY_DISPATCH.json','STOP','accounting.jsonl','canary.log', 'jobs/canary_s969100/START.json','jobs/canary_s969100/result.json','jobs/canary_s969100/failure.json','jobs/canary_s969100/initialization_failure.json']
out={{}}
for n in names:
 p=root/n
 if p.exists():
  text=p.read_text(); out[n]=text if n.endswith('.log') or n.endswith('.jsonl') else json.loads(text)
print(json.dumps(out))
''')
    write(EVIDENCE/'CANARY_STATUS.json', result)
    print(json.dumps(result, indent=2))


def step0():
    path = Path('tools/c1switch/step0.py')
    raw = base64.b64encode(path.read_bytes()).decode()
    result = ssh('gpuR', f'''import base64, hashlib, json, os, subprocess, sys, time
from pathlib import Path
root=Path({ROOT!r}); stage=root/'pytools'; assert stage.is_dir()
assert json.loads((root/'jobs/canary_s969100/result.json').read_text())['pass']
assert not (root/'STEP0_DISPATCH.json').exists() and not (root/'STOP').exists()
target=stage/'tools/c1switch/step0.py'; assert not target.exists()
data=base64.b64decode({raw!r}); assert hashlib.sha256(data).hexdigest()=={sha(path)!r}
target.write_bytes(data)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PATH='/root/miniconda3/bin:'+os.environ['PATH'])
with (root/'step0.log').open('x') as log:
 proc=subprocess.Popen([sys.executable,'-B','-u','-m','tools.c1switch.step0'],cwd=stage,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.nice(19))
 row=dict(pid=proc.pid,unix=time.time(),source_sha256={sha(path)!r},max_sim_s=58680,workers=1)
 (root/'STEP0_DISPATCH.json').write_text(json.dumps(row,indent=2))
print(json.dumps(row))
''')
    write(EVIDENCE/'STEP0_DISPATCH.json', result)
    print(json.dumps(result))


def progress():
    result = ssh('gpuR', f'''import json
from pathlib import Path
root=Path({ROOT!r})
out={{}}
for name in ['STEP0_STATUS.json','STOP','step0.log','accounting.jsonl']:
 p=root/name
 if p.exists():
  text=p.read_text()
  if name=='accounting.jsonl':
   rows=[json.loads(x) for x in text.splitlines()];out[name]=dict(charged_sim_s=sum(r['sim_s'] for r in rows),segments=len(rows))
  else: out[name]=text if name.endswith('.log') else json.loads(text)
out['completed']=[]
out['decisions']={{}}
for seed in range(969100,969112):
 job=root/'jobs'/f'step0_s{{seed}}'
 if (job/'result.json').exists(): out['completed'].append(json.loads((job/'result.json').read_text()))
 out['decisions'][str(seed)]=len(list(job.glob('decision_*.json'))) if job.is_dir() else 0
 if (job/'failure.json').exists():out['failure']=json.loads((job/'failure.json').read_text())
print(json.dumps(out))
''')
    write(EVIDENCE/'STEP0_PROGRESS.json', result)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('audit','deploy','preflight','recorded','cpu','canary','status','step0','progress'))
    a = p.parse_args()
    globals()[a.action]()


if __name__ == '__main__':
    main()
