"""Explicit small remote operations; stage immutable source, audit, CPU, then launch."""
import argparse
import ast
import json
from pathlib import Path
import re

from tools.coopbandit.ops import remote, HOSTS as INHERITED, PYTHON
from .protocol import REGISTRATION, worlds, budget, approval, sha, digest, checked

LOCAL = Path(__file__).resolve().parents[2]
HOSTS = {h:dict(c,root=('/home/ubuntu' if h == 'cpuQ' else '/root/autodl-tmp')+
               '/coopplan20261101_v1',workers=4,load_max=(24 if h == 'cpuQ' else 16 if h == 'gpuA' else 112))
         for h,c in INHERITED.items()}


def inspect(host):
    from tools.coopbandit.ops import inspect as inherited_inspect
    return inherited_inspect(host)


def manifest(host=None):
    design = LOCAL/'docs/paper_results/COOPPLAN_DESIGN.md'
    lines = len(design.read_text(encoding='utf-8').splitlines())
    assert lines <= 70
    return dict(design_sha256=sha(design),design_lines=lines,budget=budget(),total=sum(budget().values()),
                panel_sha256=digest(worlds()),new_sumo_sim_s=0)


def audit(host):
    from tools.coopbandit.protocol import FORBIDDEN
    seeds = [w['seed'] for w in worlds()]
    text = checked(REGISTRATION).read_text(encoding='utf-8')
    collisions = []
    exact = re.compile(r'(?<![0-9a-fA-F.])(?:s|w)?(?:'+'|'.join(map(str,seeds))+r')(?![0-9a-fA-F])')
    for line in text.splitlines():
        if 'S-COOPPLAN PRECOMPUTE DESIGN REGISTERED' in line:
            continue
        if exact.search(line):
            collisions.append(line)
        for a,b in re.findall(r'(?<![0-9a-fA-F.])(\d{6})\s*[-–~]+\s*(\d{3,6})(?![0-9a-fA-F])',line):
            b = a[:6-len(b)]+b
            if any(int(a) <= seed <= int(b) for seed in seeds):
                collisions.append(line)
    if collisions:
        raise ValueError('registration exposure collision: '+str(collisions))
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
'''))
    if not result['pass_']:
        raise ValueError('safe filename exposure collision: '+str(result))
    result.update(registration_sha256=sha(REGISTRATION),panel_sha256=digest(worlds()))
    return result


def sources():
    files = {}
    for package in ('coopplan','coopbandit'):
        for p in sorted((LOCAL/'tools'/package).glob('*.py')):
            files[f'tools/{package}/{p.name}'] = p.read_text(encoding='utf-8')
    for name in ('__init__.py','backend.py','design.py','intervention.py','selection.py','storage.py'):
        p = Path('E:/agent-work/c4-leverprobe/tools/leverprobe')/name
        files['tools/leverprobe/'+name] = checked(p).read_text(encoding='utf-8')
    for package,names in [('coopfree',('__init__.py','policy.py','protocol.py')),
                          ('coopbehav',('__init__.py','io.py','detectors.py')),
                          ('wmdecide',('__init__.py','decision.py'))]:
        tree = 'c4-rb-decide' if package == 'wmdecide' else 'c4-'+package
        for name in names:
            p = Path('E:/agent-work')/tree/'tools'/package/name
            files[f'tools/{package}/{name}'] = checked(p).read_text(encoding='utf-8')
    p = Path('E:/agent-work/c4-entrant/tools/lrdiag/replay.py')
    text = checked(p).read_text(encoding='utf-8')
    node = next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name == 'breakdown')
    files['tools/coopplan/bic.py'] = ('"""Unmodified LRDIAG BIC source, origin SHA256 '+sha(p)+'"""\nimport numpy as np\n\n'+ast.get_source_segment(text,node)+'\n')
    return files


def stage(host):
    c = HOSTS[host]
    files = sources()
    design = (LOCAL/'docs/paper_results/COOPPLAN_DESIGN.md').read_text(encoding='utf-8')
    return remote(host,f'''import pathlib,json
root=pathlib.Path({c['root']!r});assert root.parent.is_dir();root.mkdir(exist_ok=True)
files=json.loads({json.dumps(files)!r})
for name,text in files.items():
 p=root/'pytools'/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_text()==text,('immutable source',name)
 else:p.write_text(text)
p=root/'COOPPLAN_DESIGN.md'
if p.exists():assert p.read_text()=={design!r}
else:p.write_text({design!r})
print(json.dumps(dict(host={host!r},root=str(root),files=len(files),sim_s=0)))
''')


def prepare(host):
    c = HOSTS[host]
    text = checked(REGISTRATION).read_text(encoding='utf-8')
    owner = approval(text)
    registered = next(s for s in text.splitlines() if 'S-COOPPLAN PRECOMPUTE DESIGN REGISTERED' in s)
    panel = [w for i,w in enumerate(worlds()) if i%3 == tuple(HOSTS).index(host)]
    return remote(host,f'''import pathlib,sys,os,json
root=pathlib.Path({c['root']!r});assert root.is_dir()
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path[:0]=[str(root/'pytools')]+{c['deps']!r}
from tools.coopplan.storage import write,read
from tools.coopplan.protocol import sha,budget
files=json.loads({json.dumps(list(sources()))!r})
binding=dict(host={host!r},runtime={c['runtime']!r},checkpoint={c['checkpoint']!r},deps={c['deps']!r},
 design_sha256=sha(root/'COOPPLAN_DESIGN.md'),sources={{name:sha(root/'pytools'/name) for name in files}},
 budget=budget(),workers={c['workers']},load_max={c['load_max']},free_checkpoint=None)
allocation=dict(worlds={panel!r},cap=4*58560+{2040 if host == 'cpuQ' else 0})
for name,value in [('BINDING.json',binding),('ALLOCATION.json',allocation),('APPROVAL.json',dict(owner_line={owner!r},registration_line={registered!r},design_sha256=binding['design_sha256']))]:
 if (root/name).exists():assert read(root/name)==value
 else:write(root/name,value)
for name in ('jobs','states'):(root/name).mkdir(exist_ok=True)
from tools.coopplan.plant import setup
setup(root)
from tools.coopplan.protocol import worlds,guard
for world in worlds():guard(world)
import sumo,subprocess
binary=pathlib.Path(sumo.__file__).parent/'bin/sumo';assert binary.is_file()
version=subprocess.run([str(binary),'--version'],capture_output=True,text=True,check=True).stdout;assert '1.27.1' in version
print(json.dumps(dict(host={host!r},source_count=len(files),design_sha256=binding['design_sha256'],guards=12,cap=allocation['cap'],nice=os.getpriority(os.PRIO_PROCESS,0),sim_s=0)))
''')


def cpu(host):
    c = HOSTS[host]
    return remote(host,f'''import sys,os,pathlib,json,unittest
from unittest.mock import patch
root=pathlib.Path({c['root']!r});assert root.is_dir();os.nice(19)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path[:0]=[str(root/'pytools')]+{c['deps']!r}
from tools.coopplan.plant import setup
setup(root)
with patch('subprocess.Popen',side_effect=AssertionError('SUMO disabled')):
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tools.coopplan.test_cpu'))
from tools.coopplan.predictor import SumoOraclePredictor,ObservableState
from tools.coopplan.planner import wmdecide_check,decide
from tools.coopplan.test_cpu import STATE,ACTIONS,toy
p=SumoOraclePredictor(lambda s,a,h:toy(a)).predict(STATE,ACTIONS,120,1)
assert wmdecide_check(p)['selected']==decide(p)['selected']
receipt=dict(host={host!r},pass_=result.wasSuccessful(),tests=result.testsRun,skipped=len(result.skipped),wmdecide_parity=True,sim_s=0)
print(json.dumps(receipt));assert receipt['pass_'] and not receipt['skipped']
from tools.coopplan.storage import write
write(root/'CPU.json',receipt)
''')


def launch(host,mechanics=False):
    c = HOSTS[host]
    mode = 'mechanics' if mechanics else 'runner'
    return remote(host,f'''import pathlib,sys,os,json,subprocess
root=pathlib.Path({c['root']!r});assert root.is_dir();sys.path[:0]={c['deps']!r}
receipt=json.loads((root/'CPU.json').read_text());assert receipt['pass_'] and receipt['sim_s']==0
import sumo
home=str(pathlib.Path(sumo.__file__).parent)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 PYTHONPATH=':'.join([str(root/'pytools')]+{c['deps']!r}),SUMO_HOME=home,PATH=home+'/bin:'+str(pathlib.Path(sys.executable).parent)+':'+os.environ.get('PATH',''))
log=(root/{(mode.upper()+'.log')!r}).open('x')
args=['nice','-n','19',sys.executable,'-B','-u','-m',{'tools.coopplan.'+mode!r},str(root)]
if {not mechanics!r}:args+=['--workers',{str(c['workers'])!r}]
p=subprocess.Popen(args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
print(json.dumps(dict(host={host!r},mode={mode!r},pid=p.pid,root=str(root))))
''')


def admit(host):
    """Copy the successful, already-paid global mechanics receipt; no physics."""
    origin = HOSTS['cpuQ']['root']
    receipt = json.loads(remote('cpuQ',f'''import pathlib,json
p=pathlib.Path({origin!r})/'MECHANICS.json';print(p.read_text())
'''))
    assert receipt['pass_'] and receipt['sim_s'] == 2040
    c = HOSTS[host]
    return remote(host,f'''import pathlib,sys,json
root=pathlib.Path({c['root']!r});sys.path.insert(0,str(root/'pytools'))
from tools.coopplan.storage import read,write
value=json.loads({json.dumps(receipt)!r})
p=root/'MECHANICS.json'
if p.exists():assert read(p)==value
else:write(p,value)
print(json.dumps(dict(host={host!r},mechanics_pass=True,mechanics_sim_s=2040,new_sim_s=0)))
''')


def status(host):
    c = HOSTS[host]
    return remote(host,f'''import pathlib,json,time
root=pathlib.Path({c['root']!r})
names=('STOP.json','MECHANICS.json','COMPLETE.json','DISPATCH_ONCE.json')
results={{name:json.loads((root/name).read_text()) for name in names if (root/name).exists()}}
if 'MECHANICS.json' in results:results['MECHANICS.json']={{k:v for k,v in results['MECHANICS.json'].items() if k in ('pass_','sim_s')}}
charges=[json.loads(s) for s in (root/'CHARGES.jsonl').read_text().splitlines()] if (root/'CHARGES.jsonl').exists() else []
closed=[];partial=[]
for p in (root/'jobs').iterdir():
 if (p/'CLOSE.json').exists():closed.append(json.loads((p/'CLOSE.json').read_text()))
 else:partial.append(p.name)
logs={{name:(root/name).read_text()[-12000:] for name in ('MECHANICS.log','RUNNER.log') if (root/name).exists()}}
print(json.dumps(dict(host={host!r},status=results,charged=sum(s['sim_s'] for s in charges),closed_actual=sum(s['actual_sim_s'] for s in closed),closed_jobs=len(closed),active_attempts=partial,logs=logs,unix=time.time())))
''')


def free_cpu(host):
    """CPU ABI test for the future FREE adapter; no checkpoint fitting or SUMO."""
    import inspect as introspection
    from .runner import learned_proposer
    source = introspection.getsource(learned_proposer)
    c = HOSTS[host]
    return remote(host,f'''import sys,os,pathlib,json
from unittest.mock import patch
root=pathlib.Path({c['root']!r});os.nice(19)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path[:0]=[str(root/'pytools')]+{c['deps']!r}
import torch
torch.set_num_threads(1)
from tools.coopfree.policy import SetPointPolicy
from tools.coopplan.predictor import ObservableState
from tools.coopplan.test_cpu import ROWS
namespace={{}};exec({source!r},namespace)
rows=tuple(ROWS)+(dict(vid='far',s_m=-90.,v_mps=20.,lane=0,is_cav=True),)
state=ObservableState(300,rows,'CPU-only',commands={{'a':0.,'b':0.,'far':0.}})
with patch('subprocess.Popen',side_effect=AssertionError('SUMO disabled')),patch('torch.load',return_value=dict(policy=SetPointPolicy().state_dict())):
 proposal=namespace['learned_proposer'](dict(free_checkpoint='CPU-only-synthetic-weights'))(state)
assert proposal['id']=='FREE' and set(proposal['members'])=={{'a','b'}}
assert set(proposal['targets'])=={{'a','b'}} and proposal['window'] in (10,20,30)
print(json.dumps(dict(host={host!r},free_adapter_abi=True,nearest_two_pruning=True,sim_s=0,synthetic_weights_only=True)))
''')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode',choices=('manifest','inspect','audit','stage','prepare','cpu','free_cpu','mechanics','admit','launch','status'))
    p.add_argument('--host',choices=tuple(HOSTS),default='cpuQ')
    a = p.parse_args()
    if a.mode == 'mechanics':print(launch(a.host,True))
    elif a.mode == 'launch':print(launch(a.host))
    else:print(globals()[a.mode](a.host))
