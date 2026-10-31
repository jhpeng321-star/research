"""Explicit short staging/audit/CPU operations; no automatic training dispatch."""
import argparse
import json
from pathlib import Path
from tools.coopbandit import ops as ssh
from .protocol import checked, sha, worlds, digest, budget, FORBIDDEN, approval_line
from .storage import write

LOCAL = Path(__file__).resolve().parents[2]
ROOT = '/home/ubuntu/coopfree20261101_build4'
REG = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
ssh.PYTHON['gpuA'] = '/root/miniconda3/bin/python'


def audit(host):
    import re
    seeds = [w['seed'] for w in worlds()+worlds(True)]
    text = checked(REG).read_text(encoding='utf-8')
    ownsha = sha(LOCAL/'docs/paper_results/COOPFREE_DESIGN.md')
    hits = []
    for line in text.splitlines():
        if 'S-COOPFREE BUILD READY' in line and ownsha in line:
            continue  # The exact source-bound task reservation, not any other use of the block.
        for a,b in re.findall(r'(?<![0-9a-fA-F.])(9\d{5})\s*[-–锟?]+\s*(\d{4,6})(?![0-9a-fA-F])',line):
            b = a[:6-len(b)]+b
            if any(int(a) <= s <= int(b) for s in seeds):
                hits.append(line)
        if re.search(r'(?<![0-9a-fA-F.])(?:s|w)?(?:'+'|'.join(map(str,seeds))+r')(?![0-9a-fA-F])',line):
            hits.append(line)
    if hits:
        raise ValueError('registration collision: '+str(hits))
    home = '/home/ubuntu' if host == 'cpuQ' else '/root'
    result = json.loads(ssh.remote(host,f'''import os,pathlib,re,json
forbidden=re.compile({FORBIDDEN.pattern!r},re.I)
ident=re.compile(r'(?:^|[^0-9a-fA-F.])(?:s|w)?(?:'+'|'.join(map(str,{seeds!r}))+r')(?![0-9a-fA-F])')
hits=[];count=0
for parent,dirs,files in os.walk({home!r}):
 dirs[:]=[d for d in dirs if d not in ('.git','__pycache__','miniconda3','.cache','.npm','.rustup','.local') and not forbidden.search(str(pathlib.Path(parent)/d)) and not (pathlib.Path(parent)/d).is_symlink()]
 for name in dirs+files:
  if forbidden.search(str(pathlib.Path(parent)/name)): continue
  count+=1
  if ident.search(name): hits.append(str(pathlib.Path(parent)/name))
print(json.dumps(dict(host={host!r},safe_names=count,hits=hits,pass_=not hits)))
'''))
    result.update(registration_sha256=sha(REG),worlds_sha256=digest(worlds()+worlds(True)))
    return result


def sources():
    files = {}
    for p in sorted((LOCAL/'tools/coopfree').glob('*.py')):
        files['tools/coopfree/'+p.name] = checked(p).read_text(encoding='utf-8')
    for name in ('__init__.py','protocol.py'):
        p = LOCAL/'tools/coopbandit'/name
        files['tools/coopbandit/'+name] = checked(p).read_text(encoding='utf-8')
    for name in ('__init__.py','backend.py','design.py','intervention.py','selection.py','storage.py'):
        p = Path('E:/agent-work/c4-leverprobe/tools/leverprobe')/name
        files['tools/leverprobe/'+name] = checked(p).read_text(encoding='utf-8')
    for name in ('__init__.py','detectors.py','io.py'):
        p = Path('E:/agent-work/c4-coopbehav/tools/coopbehav')/name
        files['tools/coopbehav/'+name] = checked(p).read_text(encoding='utf-8')
    return files


def stage(host,root):
    files = sources()
    design = checked(LOCAL/'docs/paper_results/COOPFREE_DESIGN.md').read_text(encoding='utf-8')
    return ssh.remote(host,f'''import pathlib,json,hashlib
root=pathlib.Path({root!r});assert root.parent.is_dir()
root.mkdir(exist_ok=True)
files=json.loads({json.dumps(files)!r})
design=json.loads({json.dumps(design)!r})
for name,text in files.items():
 p=root/'pytools'/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and p.read_text()!=text: raise ValueError('immutable stage '+name)
 if not p.exists(): p.write_text(text)
p=root/'COOPFREE_DESIGN.md'
if p.exists() and p.read_text()!=design: raise ValueError('immutable design')
if not p.exists(): p.write_text(design)
print(json.dumps(dict(root=str(root),files=len(files),sources={{n:hashlib.sha256((root/'pytools'/n).read_bytes()).hexdigest() for n in files}},design_sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
''')


def cpu(host,root):
    if host != 'cpuQ':
        raise ValueError('CPU build uses existing cpuQ frozen environment')
    return ssh.remote(host,f'''import sys,os,unittest,json,hashlib,time,pathlib
from unittest.mock import patch
os.nice(19)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path.insert(0,{root!r}+'/pytools')
from tools.leverprobe.backend import setup
checkpoint='/home/ubuntu/mappospeed/production2/policy_latest.pt'
setup('/home/ubuntu/e3final',checkpoint)
os.environ['COOPFREE_CPU_CHECKPOINT']=checkpoint
begin=time.monotonic()
with patch('subprocess.Popen',side_effect=AssertionError('SUMO forbidden during CPU build')):
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tools.coopfree.test_cpu'))
files=[p for p in (pathlib.Path({root!r})/'pytools').rglob('*.py') if '__pycache__' not in str(p)]
receipt=dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0,wall_s=time.monotonic()-begin,root={root!r},sources={{str(p.relative_to(pathlib.Path({root!r})/'pytools')):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
if receipt['pass_'] and not receipt['skips']:
 from tools.coopfree.storage import write
 write(pathlib.Path({root!r})/'CPU_APPROVED.json',receipt)
print(json.dumps(receipt))
assert result.wasSuccessful() and not result.skipped
''')


def bind(host,root,audits):
    """Never grants approval; binds an actual MAIN registry approval if one exists."""
    if host != 'cpuQ':
        raise ValueError('first owned worker uses frozen cpuQ runtime; other hosts need path/hash binding')
    text = checked(REG).read_text(encoding='utf-8')
    line = approval_line(text)
    # Smaller explicit cap needs a prospectively revised design, not silent overspending.
    import re
    cap = re.search(r'cap\s*([\d,]+)',line,re.I)
    if cap and int(cap[1].replace(',','')) < budget()['total']:
        raise ValueError('approval cap is below the proposed design; amend before binding')
    exposure = dict(pass_=all(r['pass_'] for r in audits),hosts=audits,
                    worlds_sha256=digest(worlds()+worlds(True)),registration_sha256=sha(REG))
    approval = dict(owner_line=line,design_sha256=None,registration_sha256=sha(REG))
    return ssh.remote(host,f'''import sys,pathlib,json,hashlib
root=pathlib.Path({root!r});assert root.is_dir()
sys.path.insert(0,str(root/'pytools'))
from tools.coopfree.storage import read,write
cpu=read(root/'CPU_APPROVED.json');assert cpu['pass_'] and cpu['sim_s']==0 and not cpu['skips']
sources=cpu['sources']
for name,expected in sources.items(): assert hashlib.sha256((root/'pytools'/name).read_bytes()).hexdigest()==expected
design_sha=hashlib.sha256((root/'COOPFREE_DESIGN.md').read_bytes()).hexdigest()
approval=json.loads({json.dumps(approval)!r});approval['design_sha256']=design_sha
binding=dict(runtime='/home/ubuntu/e3final',checkpoint='/home/ubuntu/mappospeed/production2/policy_latest.pt',
 design_sha256=design_sha,sources=sources,budget=json.loads({json.dumps(budget())!r}),host={host!r})
for name,value in [('BINDING.json',binding),('EXPOSURE.json',json.loads({json.dumps(exposure)!r})),('APPROVAL.json',approval)]:
 p=root/name
 if p.exists(): assert read(p)==value,'immutable binding'
 else: write(p,value)
print(json.dumps(dict(bound=True,design_sha256=design_sha,host={host!r})))
''')


def launch(host,root):
    if host != 'cpuQ':
        raise ValueError('first dispatch is one nice19 cpuQ worker')
    return ssh.remote(host,f'''import pathlib,subprocess,os,sys,json,hashlib
r=pathlib.Path({root!r});assert r.is_dir()
sys.path.insert(0,str(r/'pytools'))
from tools.coopfree.runner import verify
from tools.coopfree.storage import write
verify(r)
assert not (r/'DISPATCH.json').exists(),'already dispatched; inspect owned status, do not duplicate'
assert not (r/'STOP.json').exists(),'durable STOP'
import sumo
home=str(pathlib.Path(sumo.__file__).parent)
binary=str(pathlib.Path(home)/'bin/sumo')
version=subprocess.check_output([binary,'--version'],text=True)
assert 'Version 1.27.1' in version,version
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 PYTHONPATH=str(r/'pytools'),SUMO_HOME=home,PATH=home+'/bin:'+str(pathlib.Path(sys.executable).parent)+':'+os.environ.get('PATH',''))
log=(r/'SUPERVISOR.log').open('x')
p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopfree.runner',str(r),'--phase','pipeline'],
 cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
receipt=dict(host={host!r},root=str(r),pid=p.pid,start_ticks=pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],
 phase='pipeline',workers=1,sumo_version=version.splitlines()[0])
write(r/'DISPATCH.json',receipt)
print(json.dumps(receipt))
''')


def owned_status(host,root):
    return ssh.remote(host,f'''import pathlib,json,os
r=pathlib.Path({root!r});assert r.is_dir()
def read(p): return json.loads(p.read_text())
dispatch=read(r/'DISPATCH.json') if (r/'DISPATCH.json').exists() else None
alive=False
if dispatch:
 p=pathlib.Path('/proc/'+str(dispatch['pid'])+'/stat')
 alive=p.exists() and p.read_text().split()[21]==dispatch['start_ticks']
charges=[json.loads(s) for s in (r/'CHARGES.jsonl').read_text().splitlines()] if (r/'CHARGES.jsonl').exists() else []
closed=[];complete=[]
for row in charges:
 job=r/'jobs'/row['id']
 if (job/'CLOSE.json').exists(): closed.append(read(job/'CLOSE.json'))
 if (job/'RESULT.json').exists() and (job/'CLOSE.json').exists(): complete.append(row['id'])
print(json.dumps(dict(dispatch=dispatch,owned_alive=alive,charged=sum(x['sim_s'] for x in charges),
 actual_closed=sum(x['actual_sim_s'] for x in closed),complete=complete,
 updates=[p.name for p in r.glob('UPDATE_*.json')],
 stop=read(r/'STOP.json') if (r/'STOP.json').exists() else None,
 load=os.getloadavg())))
''')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode',choices=('inspect','audit','stage','cpu','approval','bind','launch','status'))
    p.add_argument('--host',default='cpuQ',choices=('cpuQ','gpuR','gpuA'))
    p.add_argument('--root',default=ROOT)
    p.add_argument('--receipt',type=Path)
    a = p.parse_args()
    checked(a.root)
    if a.mode == 'inspect':
        result = ssh.inspect(a.host)
    elif a.mode == 'audit':
        result = json.dumps(audit(a.host))
    elif a.mode == 'stage':
        result = stage(a.host,a.root)
    elif a.mode == 'approval':
        # Keep this lightweight on the laptop, which intentionally has no torch installed.
        import re
        lines = checked(REG).read_text(encoding='utf-8').splitlines()
        matches = [s for s in lines if re.search(r'S-COOPFREE BUDGET APPROVED\b(?![\'\"])',s)
                   and not re.search(r'await|pending|not approved|requires|must append|NO SUMO|proposed',s,re.I)]
        result = json.dumps(dict(approved=bool(matches),lines=matches,registration_sha256=sha(REG)))
    elif a.mode == 'bind':
        result = bind(a.host,a.root,[audit(h) for h in ('cpuQ','gpuR','gpuA')])
    elif a.mode == 'launch':
        result = launch(a.host,a.root)
    elif a.mode == 'status':
        result = owned_status(a.host,a.root)
    else:
        result = cpu(a.host,a.root)
    print(result)
    if a.receipt:
        write(a.receipt,dict(operation=a.mode,host=a.host,root=a.root,result=result,
                             design_sha256=sha(LOCAL/'docs/paper_results/COOPFREE_DESIGN.md')))


if __name__ == '__main__':
    main()
