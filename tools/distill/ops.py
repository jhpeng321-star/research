"""One-shot immutable staging/dispatch and explicit read-only collection."""
import argparse
import json
from pathlib import Path
import pickle
import re

import numpy as np

from tools.coopbandit.ops import remote
from tools.coopbandit.protocol import FORBIDDEN, digest, sha, checked
from tools.coopplan.ops import sources as inherited_sources
from .data import LOCAL, KEYS, load_exam, arrays, planner_records, save, read
from .live import panel, budget, approval

REG = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
OUT = LOCAL/'docs/paper_results/distill_live'
FIT = LOCAL/'docs/paper_results/distill_v1'
ROOT = '/root/autodl-tmp/distill20261101_v1'
HOST = 'gpuA'


def export():
    from .model import predict_values, choose as original_choose
    from .portable import export as convert, predict, choose
    if not OUT.parent.is_dir():
        raise ValueError('output parent')
    OUT.mkdir(exist_ok=True)
    freeze = read(FIT/'FREEZE.json')
    if sha(FIT/'policy.pkl') != freeze['policy_sha256']:
        raise ValueError('original frozen pickle hash')
    with (FIT/'policy.pkl').open('rb') as f:
        payload = pickle.load(f)
    policy = convert(payload)
    records,_ = load_exam()
    x,_,av,_ = arrays(records)
    dev,_ = planner_records()
    x = np.concatenate([x,np.array([[r['features'][k] for k in KEYS] for r in dev])])
    av = np.concatenate([av,np.array([[o in r['available'] for o in policy['options']] for r in dev])])
    old = predict_values(payload['models'],x)
    new = predict(policy,x)
    np.testing.assert_allclose(new,old,atol=1e-8,rtol=1e-10)
    expected = original_choose(old,av,payload['errors'],payload['confidence'])
    np.testing.assert_array_equal(choose(policy,new,av)[0],expected)
    save(OUT/'POLICY.json',policy)
    save(OUT/'PARITY.json',dict(x=x.tolist(),available=av.tolist(),predictions=old.tolist(),
        selected=expected.tolist(),original_policy_sha256=freeze['policy_sha256'],
        max_absolute_error=float(abs(new-old).max()),states=len(x)))
    save(OUT/'PANEL.json',panel())
    save(OUT/'EXPORT.json',dict(policy_sha256=sha(OUT/'POLICY.json'),
         panel_sha256=digest(panel()),budget=budget(),original_policy_sha256=freeze['policy_sha256']))
    print(json.dumps(read(OUT/'EXPORT.json')))


def inspect():
    for host in ('gpuA','cpuQ','gpuR'):
        print(remote(host,"import os,shutil,json; print(json.dumps(dict(load=os.getloadavg(),cpus=os.cpu_count(),disk=shutil.disk_usage('.').free)))",timeout=30))


def manifest():
    print(json.dumps(dict(design_sha256=sha(LOCAL/'docs/paper_results/DISTILL_LIVE_DESIGN.md'),
        panel_sha256=digest(panel()),policy_sha256=sha(OUT/'POLICY.json'),budget=budget())))


def audit():
    text = checked(REG).read_text(encoding='utf-8')
    seeds = [w['seed'] for w in panel()]
    exact = re.compile(r'(?<![0-9a-fA-F.])(?:s|w)?(?:'+'|'.join(map(str,seeds))+r')(?![0-9a-fA-F])')
    for line in text.splitlines():
        if 'S-DISTILL LIVE REGISTERED BEFORE DISPATCH' in line:
            continue
        if exact.search(line):
            raise ValueError('existing registered exposure: '+line)
        for a,b in re.findall(r'(?<![0-9a-fA-F.])(\d{6})\s*[-–~]+\s*(\d{3,6})(?![0-9a-fA-F])',line):
            b = a[:6-len(b)]+b
            if any(int(a) <= s <= int(b) for s in seeds):
                raise ValueError('existing registered seed range exposure')
    results = []
    for host in ('cpuQ','gpuA','gpuR'):
        home = '/home/ubuntu' if host == 'cpuQ' else '/root'
        result = json.loads(remote(host,f'''import os,re,json,pathlib
bad=re.compile({FORBIDDEN.pattern!r},re.I);pattern=re.compile({exact.pattern!r})
hits=[];count=0
for parent,dirs,files in os.walk({home!r}):
 dirs[:]=[d for d in dirs if d not in ('.git','__pycache__','miniconda3','.cache','.npm','.rustup','.local') and not bad.search(str(pathlib.Path(parent)/d)) and not (pathlib.Path(parent)/d).is_symlink()]
 for name in dirs+files:
  p=pathlib.Path(parent)/name
  if bad.search(str(p)):continue
  count+=1
  if pattern.search(name):hits.append(str(p))
print(json.dumps(dict(host={host!r},safe_names=count,hits=hits,pass_=not hits)))
''',timeout=180))
        if not result['pass_']:
            raise ValueError('safe-name exposure collision: '+str(result))
        results.append(result)
    result = dict(pass_=True,hosts=results,panel_sha256=digest(panel()),registration_sha256=sha(REG))
    save(OUT/'EXPOSURE.json',result)
    print(json.dumps(result))


def stage():
    line = approval(checked(REG).read_text(encoding='utf-8'))
    files = {'pytools/'+k:v for k,v in inherited_sources().items()}
    for name in ('__init__.py','data.py','portable.py','live.py','test_live.py'):
        files['pytools/tools/distill/'+name] = (LOCAL/'tools/distill'/name).read_text(encoding='utf-8')
    artifacts = {}
    for name in ('POLICY.json','PARITY.json','PANEL.json','EXPOSURE.json'):
        artifacts[name] = (OUT/name).read_text(encoding='utf-8')
    artifacts['DISTILL_LIVE_DESIGN.md'] = (LOCAL/'docs/paper_results/DISTILL_LIVE_DESIGN.md').read_text(encoding='utf-8')
    # Hash bytes exactly as transferred (local Git may normalize textual line endings).
    import hashlib
    digest_text = lambda s:hashlib.sha256(s.encode()).hexdigest()
    binding = dict(runtime='/root/autodl-tmp/c4/speedhead20261030/e3final',
        checkpoint='/root/autodl-tmp/c4/speedhead20261030/mappospeed_final169.pt',
        deps=['/root/autodl-tmp/c4/speedhead20261030/pythondeps'],workers=4,load_max=16,
        budget=budget(),panel=panel(),sources={k:digest_text(v) for k,v in files.items()},
        artifacts={k:digest_text(v) for k,v in artifacts.items()},
        design_sha256=digest_text(artifacts['DISTILL_LIVE_DESIGN.md']))
    files.update(artifacts)
    files.update({'BINDING.json':json.dumps(binding),'APPROVAL.json':json.dumps(dict(line=line))})
    result = json.loads(remote(HOST,f'''import pathlib,json
root=pathlib.Path({ROOT!r});assert root.parent.is_dir();root.mkdir()
files=json.loads({json.dumps(files)!r})
for name,text in files.items():
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:f.write(text)
for name in ('jobs','states','reservations'):(root/name).mkdir()
print(json.dumps(dict(host={HOST!r},root=str(root),files=len(files),new_sumo_sim_s=0)))
'''))
    save(OUT/'STAGED.json',dict(**result,binding=binding))
    print(json.dumps(result))


def cpu():
    result = json.loads(remote(HOST,f'''import os,sys,pathlib,json,unittest
from unittest.mock import patch
root=pathlib.Path({ROOT!r});assert root.is_dir();os.nice(19)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DISTILL_TEST_ROOT=str(root))
sys.path[:0]=[str(root/'pytools'),'/root/autodl-tmp/c4/speedhead20261030/pythondeps']
from tools.distill.live import setup,guard,panel
setup(root)
with patch('subprocess.Popen',side_effect=AssertionError('SUMO disabled')):
 for w in panel():guard(w)
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tools.distill.test_live'))
from tools.coopplan.storage import write
receipt=dict(pass_=result.wasSuccessful(),tests=result.testsRun,skipped=len(result.skipped),guards=24,new_sumo_sim_s=0)
assert receipt['pass_'] and not receipt['skipped'];write(root/'CPU.json',receipt)
print(json.dumps(receipt))
''',timeout=180))
    save(OUT/'CPU.json',result);print(json.dumps(result))


def launch():
    result = json.loads(remote(HOST,f'''import pathlib,sys,os,json,subprocess
root=pathlib.Path({ROOT!r});assert root.is_dir()
assert json.loads((root/'CPU.json').read_text())['pass_']
assert not (root/'LAUNCH_ONCE.json').exists() and not (root/'DISPATCH_ONCE.json').exists()
with (root/'LAUNCH_ONCE.json').open('x') as f:f.write(json.dumps(dict(parent=os.getpid())))
import sumo
home=str(pathlib.Path(sumo.__file__).parent)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 PYTHONPATH=str(root/'pytools')+':/root/autodl-tmp/c4/speedhead20261030/pythondeps',SUMO_HOME=home,PATH=home+'/bin:'+os.environ['PATH'])
log=(root/'LIVE.log').open('x')
p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.distill.live',str(root),'--workers','4'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
print(json.dumps(dict(host={HOST!r},root=str(root),pid=p.pid,command='tools.distill.live',new_sumo_sim_s=0)))
'''))
    save(OUT/'DISPATCH.json',result);print(json.dumps(result))


def status(collect=False):
    result = json.loads(remote(HOST,f'''import pathlib,json
root=pathlib.Path({ROOT!r});records=[];incomplete=[];closed=[]
for w in json.loads((root/'PANEL.json').read_text()):
 seed=w['seed'];base=root/'jobs'/f's{{seed}}_mappo'/'RESULT.json';student=root/'jobs'/f's{{seed}}_student'/'RESULT.json'
 if base.exists() and student.exists():records.append(dict(world=w,baseline=json.loads(base.read_text()),student=json.loads(student.read_text())))
for job in (root/'jobs').iterdir():
 p=job/'CLOSE.json'
 if p.exists():closed.append(dict(id=job.name,**json.loads(p.read_text())))
 else:incomplete.append(job.name)
charges=[json.loads(s) for s in (root/'CHARGES.jsonl').read_text().splitlines()] if (root/'CHARGES.jsonl').exists() else []
report=dict(complete=(root/'COMPLETE.json').exists(),stopped=(root/'STOP.json').exists(),worlds=len(records),
 charged=sum(r['sim_s'] for r in charges),actual=sum(r['actual_sim_s'] for r in closed),incomplete=incomplete,
 mechanics=json.loads((root/'MECHANICS.json').read_text()) if (root/'MECHANICS.json').exists() else None,
 stop=json.loads((root/'STOP.json').read_text()) if (root/'STOP.json').exists() else None)
if {collect!r}:report.update(records=records,charges=charges,closed=closed,binding=json.loads((root/'BINDING.json').read_text()))
print(json.dumps(report))
''',timeout=180))
    if collect:
        if not result['complete'] or result['stopped'] or result['incomplete'] or result['worlds'] != 24:
            raise ValueError('full24paired panel required; status='+str({k:v for k,v in result.items() if k!='records'}))
        save(OUT/'RECEIPTS.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','charges','closed','binding')}))


def main():
    p = argparse.ArgumentParser();p.add_argument('mode',choices=('export','inspect','manifest','audit','stage','cpu','launch','status','collect'))
    a = p.parse_args()
    if a.mode in ('status','collect'):status(a.mode == 'collect')
    else:globals()[a.mode]()


if __name__ == '__main__':main()
