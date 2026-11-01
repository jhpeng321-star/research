"""CPU-only stage/audit/evidence commands. This module has NO launch command."""
import argparse
import json
from pathlib import Path
from tools.coopbandit import ops as ssh
from .protocol import checked, sha, worlds, registry_conflicts, REGISTRY, FORBIDDEN
from .storage import write

LOCAL = Path(__file__).resolve().parents[2]
ssh.PYTHON['gpuA'] = '/root/miniconda3/bin/python'


def sources():
    paths = list((LOCAL/'tools/coopnative').glob('*.py'))
    paths += [LOCAL/f'tools/{package}/{name}' for package,names in (
        ('coopfree',('__init__.py','policy.py','protocol.py','analysis.py')),
        ('coopbandit',('__init__.py','protocol.py'))) for name in names]
    paths += [Path('E:/agent-work/c4-coopbehav/tools/coopbehav')/name for name in ('__init__.py','detectors.py','io.py')]
    return {('tools/coopbehav/'+p.name if p.parent.name=='coopbehav' else p.relative_to(LOCAL).as_posix()):checked(p).read_text(encoding='utf-8') for p in paths}


def stage_cpu(root):
    files = sources()
    design = checked(LOCAL/'docs/paper_results/COOPNATIVE_DESIGN.md').read_text(encoding='utf-8')
    assert len(design.splitlines()) <= 70
    result = ssh.remote('cpuQ',f'''import os,sys,pathlib,json,time,hashlib,unittest
from unittest.mock import patch
root=pathlib.Path({root!r});assert root.parent.is_dir()
root.mkdir(exist_ok=True)
files=json.loads({json.dumps(files)!r})
for name,text in files.items():
 compile(text,name,'exec')
 p=root/'pytools'/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists(): assert p.read_text()==text,'immutable CPU stage'
 else: p.write_text(text)
p=root/'COOPNATIVE_DESIGN.md'
if p.exists(): assert p.read_text()=={ascii(design)}
else: p.write_text({ascii(design)})
os.nice(19)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path[:0]=[str(root/'pytools'),'/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
import torch
torch.set_num_threads(1)
begin=time.monotonic()
with patch('subprocess.Popen',side_effect=AssertionError('SUMO forbidden during CPU tests')):
 result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tools.coopnative.test_cpu'))
receipt=dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0,
 wall_s=time.monotonic()-begin,torch=torch.__version__,root=str(root),
 sources={{n:hashlib.sha256((root/'pytools'/n).read_bytes()).hexdigest() for n in files}},
 design_sha256=hashlib.sha256((root/'COOPNATIVE_DESIGN.md').read_bytes()).hexdigest())
print(json.dumps(receipt))
''',timeout=180)
    print(result)
    receipt = json.loads(result.strip().splitlines()[-1])
    write(LOCAL/'docs/paper_results/coopnative'/('CPU_'+Path(root).name+'.json'),receipt)
    if not receipt['pass_'] or receipt['skips']:
        raise ValueError('CPU suite failed; receipt preserved')


def audit(host):
    text = checked(REGISTRY).read_text(encoding='utf-8')
    conflicts = registry_conflicts(text)
    if conflicts:
        raise ValueError('registry seed collisions '+str(conflicts))
    seeds = [w['seed'] for role in ('train','check') for w in worlds(role)]
    home = '/home/ubuntu' if host=='cpuQ' else '/root'
    result = json.loads(ssh.remote(host,f'''import os,pathlib,re,json,time
forbidden=re.compile({FORBIDDEN.pattern!r},re.I)
seeds={seeds!r}
ident=re.compile(r'(?<![0-9a-fA-F.])(?:s|w)?(?:'+'|'.join(map(str,seeds))+r')(?![0-9a-fA-F])')
hits=[];count=0
for parent,dirs,files in os.walk({home!r}):
 dirs[:]=[d for d in dirs if d not in ('.git','__pycache__','miniconda3','.cache','.npm','.rustup','.local') and not forbidden.search(str(pathlib.Path(parent)/d)) and not (pathlib.Path(parent)/d).is_symlink()]
 for name in dirs+files:
  p=pathlib.Path(parent)/name
  if forbidden.search(str(p)): continue
  count+=1
  if ident.search(name): hits.append(str(p))
print(json.dumps(dict(host={host!r},safe_names=count,hits=hits,pass_=not hits,load=os.getloadavg(),unix=time.time())))
''',timeout=180))
    result.update(registration_sha256=sha(REGISTRY),registry_conflicts=conflicts,seeds=seeds,sim_s=0)
    write(LOCAL/f'docs/paper_results/coopnative/EXPOSURE_{host}.json',result)
    print(json.dumps(result))


def comparator():
    result = json.loads(ssh.remote('cpuQ','''import pathlib,json,hashlib
r=pathlib.Path('/home/ubuntu/mappospeed/production2')
p=r/'RUN_SUMMARY.json'
summary=json.loads(p.read_text())
for line in (r/'log.jsonl').open():
 episode=json.loads(line)
 if episode.get('type')=='episode': break
print(json.dumps(dict(summary=summary,episode=episode,summary_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),actor_sha256=hashlib.sha256((r/'policy_00169.pt').read_bytes()).hexdigest(),new_sim_s=0)))
'''))
    write(LOCAL/'docs/paper_results/coopnative/COMPARATOR.json',result)
    print(json.dumps(result))


def baselines():
    receipt_path=Path('E:/agent-work/c4-distill-1101/docs/paper_results/distill_live/RECEIPTS.json')
    receipt=json.loads(checked(receipt_path).read_text(encoding='utf-8'))
    expected={f'jobs/s{w["seed"]}_mappo/RESULT.json':receipt['result_hashes'][f'jobs/s{w["seed"]}_mappo/RESULT.json'] for w in worlds('dev')}
    ssh.PYTHON['gpuA']='/root/miniconda3/bin/python'
    result=json.loads(ssh.remote('gpuA',f'''import pathlib,json,hashlib
root=pathlib.Path('/root/autodl-tmp/distill20261101_v1')
expected=json.loads({json.dumps(expected)!r})
records=[]
for name,sha in expected.items():
 p=root/name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,'baseline byte mismatch'
 r=json.loads(p.read_text())
 records.append(dict(world=r['world'],arm='mappo',metrics=r['clocks'],source=str(p),sha256=sha))
print(json.dumps(dict(records=records,hashes=expected,new_sim_s=0)))
'''))
    result['receipt_sha256']=sha(receipt_path)
    write(LOCAL/'docs/paper_results/coopnative/MAPPO_DEV.json',result)
    print(json.dumps(dict(worlds=len(result['records']),new_sim_s=0,verified=True)))


def benchmark(host):
    # One CPU-only synthetic episode plus a PPO minibatch. No traffic engine imports.
    files=sources()
    root='/home/ubuntu/coopnative20261101_bench' if host=='cpuQ' else '/root/coopnative20261101_bench'
    if host=='gpuA': ssh.PYTHON['gpuA']='/root/miniconda3/bin/python'
    result=json.loads(ssh.remote(host,f'''import os,sys,pathlib,json,time
root=pathlib.Path({root!r});assert root.parent.is_dir()
root.mkdir(exist_ok=True)
files=json.loads({json.dumps(files)!r})
for name,text in files.items():
 p=root/'pytools'/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists(): assert p.read_text()==text
 else: p.write_text(text)
os.nice(19)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path.insert(0,str(root/'pytools'))
import torch
torch.set_num_threads(1)
from tools.coopnative.policy import Policy,encode
from tools.coopnative.learner import update
from tools.coopnative.test_cpu import rows,state,env
from tools.coopnative.execution import Controller
# A local safety module is needed only for the CPU timing, never a simulator.
runtime={'/home/ubuntu/e3final/runtime/src' if host=='cpuQ' else '/root/autodl-tmp/c4/speedhead20261030/e3final/runtime/src'!r}
sys.path.insert(0,runtime)
p=Policy();g=torch.Generator().manual_seed(1);r=rows();e=env(r);c=Controller()
begin=time.monotonic();steps=[]
for i in range(90):
 s=p.sample(encode(r),g);c.hold(s,i*10)
 for j in range(10): c.command(e,r,i*10+j)
 steps.append(dict(sample=s,reward=-100.))
episode_cpu_s=time.monotonic()-begin
optimizer=torch.optim.Adam(p.parameters(),lr=3e-4)
begin=time.monotonic();update(p,optimizer,[dict(steps=steps)],g,epochs=1)
ppo90_s=time.monotonic()-begin
print(json.dumps(dict(host={host!r},episode_cpu_s=episode_cpu_s,ppo90_s=ppo90_s,
 estimated_update2160_four_epochs_s=ppo90_s*24*4,sim_s=0,load=os.getloadavg(),torch=torch.__version__,
 note='synthetic sparse episode, no SUMO; not a traffic-throughput measurement')))
''',timeout=180))
    write(LOCAL/f'docs/paper_results/coopnative/TIMING_{host}.json',result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=('cpu','audit','comparator','baselines','benchmark'))
    parser.add_argument('--root')
    parser.add_argument('--host',default='cpuQ')
    args=parser.parse_args()
    if args.operation=='cpu': stage_cpu(args.root)
    elif args.operation=='audit': audit(args.host)
    elif args.operation=='comparator': comparator()
    elif args.operation=='baselines': baselines()
    else: benchmark(args.host)
