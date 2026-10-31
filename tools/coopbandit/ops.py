"""Explicit, short SSH operations. All remote reads prune restricted paths."""
import argparse
import json
from pathlib import Path
import subprocess
from .protocol import checked, digest, sha, worlds, dev_worlds, FORBIDDEN

LOCAL = Path(__file__).resolve().parents[2]
REG = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
PROBE = Path('E:/agent-work/c4-leverprobe/tools/leverprobe')
ROOT = '/home/ubuntu/coopbandit20261031_v2'
PYTHON = dict(cpuQ='/home/ubuntu/miniconda3/bin/python', gpuR='/root/miniconda3/bin/python', gpuA='/root/miniconda3/bin/python')
HOSTS = {
    'cpuQ': dict(root=ROOT,runtime='/home/ubuntu/e3final',checkpoint='/home/ubuntu/mappospeed/production2/policy_latest.pt',
                 deps=[],load_max=24,workers=8),
    'gpuA': dict(root='/root/autodl-tmp/coopbandit20261031_v1',runtime='/root/autodl-tmp/c4/speedhead20261030/e3final',
                 checkpoint='/root/autodl-tmp/c4/speedhead20261030/mappospeed_final169.pt',
                 deps=['/root/autodl-tmp/c4/speedhead20261030/pythondeps'],load_max=16,workers=8),
    'gpuR': dict(root='/root/autodl-tmp/coopbandit20261031_v1',runtime='/root/autodl-tmp/c4/speedhead20261030/e3final',
                 checkpoint='/root/autodl-tmp/c4/speedhead20261030/mappospeed_final169.pt',
                 deps=['/root/autodl-tmp/c4/speedhead20261030/pythondeps'],load_max=20,workers=2),
}


def remote(host, source, timeout=180):
    code = 'import sys;exec(compile(sys.stdin.read(),\'<coopbandit-remote>\',\'exec\'))'
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', host, PYTHON[host]+' -c "'+code+'"'],
                       input=source, capture_output=True, text=True, encoding='utf-8', timeout=timeout)
    if r.returncode:
        raise RuntimeError(r.stdout+r.stderr)
    return r.stdout


def inspect(host):
    return remote(host, '''import os,json,pathlib,shutil,time
p=pathlib.Path('/sys/fs/cgroup/cpu.max')
print(json.dumps(dict(host=os.uname().nodename,load=os.getloadavg(),cpus=os.cpu_count(),
 quota=p.read_text() if p.exists() else None,disk_free=shutil.disk_usage('.').free,
 meminfo=pathlib.Path('/proc/meminfo').read_text().splitlines()[:3],unix=time.time())))
''')


def audit(host):
    import re
    seeds = [w['seed'] for w in worlds()+dev_worlds()]
    text = checked(REG).read_text(encoding='utf-8')
    hits = []
    for line in text.splitlines():
        ranges = re.findall(r'(?<![0-9a-fA-F.])(9\d{5})\s*[-–锟?]+\s*(9?\d{4,5})(?![0-9a-fA-F])', line)
        for a, b in ranges:
            if len(b) < 6:
                b = a[:6-len(b)]+b
            if any(int(a) <= s <= int(b) for s in seeds):
                hits.append(line)
        if re.search(r'(?<![0-9a-fA-F.])(?:s|w)?(?:'+'|'.join(map(str,seeds))+r')(?![0-9a-fA-F])',line):
            hits.append(line)
    if hits:
        raise ValueError('registration collision: '+str(hits))
    home = '/home/ubuntu' if host == 'cpuQ' else '/root'
    result = json.loads(remote(host, f'''import os,pathlib,re,json
forbidden=re.compile({FORBIDDEN.pattern!r},re.I)
ident=re.compile(r'(?:^|[^0-9a-fA-F.])(?:s|w)?(?:'+ '|'.join(map(str,{seeds!r}))+r')(?![0-9a-fA-F])')
hits=[];count=0
for parent,dirs,files in os.walk({home!r}):
 dirs[:]=[d for d in dirs if d not in ('.git','__pycache__','miniconda3','.cache','.npm','.rustup','.local') and not forbidden.search(str(pathlib.Path(parent)/d)) and not (pathlib.Path(parent)/d).is_symlink()]
 for name in dirs+files:
  if forbidden.search(str(pathlib.Path(parent)/name)): continue
  count+=1
  if ident.search(name): hits.append(str(pathlib.Path(parent)/name))
print(json.dumps(dict(host={host!r},safe_names=count,hits=hits,pass_=not hits)))
'''))
    result['registration_sha256'] = sha(REG)
    result['worlds_sha256'] = digest(worlds()+dev_worlds())
    return result


def stage(root, host='cpuQ'):
    checked(root)
    files = {}
    for p in sorted((LOCAL/'tools/coopbandit').glob('*.py')):
        files['tools/coopbandit/'+p.name] = p.read_text(encoding='utf-8')
    for name in ('__init__.py','backend.py','design.py','intervention.py','selection.py','storage.py'):
        files['tools/leverprobe/'+name] = (PROBE/name).read_text(encoding='utf-8')
    files['tools/coopbandit/probe_quantiles.json'] = (PROBE/'evidence/RECORDED_QUANTILES.json').read_text(encoding='utf-8')
    design = (LOCAL/'docs/paper_results/COOPBANDIT_DESIGN.md').read_text(encoding='utf-8')
    return remote(host, f'''import pathlib,json
root=pathlib.Path({root!r});assert root.parent.is_dir()
root.mkdir(exist_ok=True)
files=json.loads({json.dumps(files)!r})
for name,text in files.items():
 p=root/'pytools'/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and p.read_text()!=text: raise ValueError('immutable stage '+name)
 if not p.exists(): p.write_text(text)
p=root/'COOPBANDIT_DESIGN.md'
if p.exists() and p.read_text()!={design!r}: raise ValueError('immutable design')
if not p.exists(): p.write_text({design!r})
print(json.dumps(dict(root=str(root),files=len(files))))
''')


def prepare(host):
    config=HOSTS[host]
    root=config['root']
    print(stage(root,host))
    lines=checked(REG).read_text(encoding='utf-8').splitlines()
    owner=next(line for line in lines if '23:38 OWNER APPROVED' in line and 'S-COOPBANDIT' in line)
    registered=next(line for line in lines if 'S-COOPBANDIT PRECOMPUTE DESIGN REGISTERED' in line)
    return remote(host,f'''import pathlib,json,hashlib,sys,os,time
root=pathlib.Path({root!r}); assert root.is_dir()
os.nice(19)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path[:0]=[{root!r}+'/pytools']+{config['deps']!r}
from tools.coopbandit.storage import write,read,admission
from tools.coopbandit.protocol import sha,budget,digest,worlds
from tools.coopbandit.backend import setup
sources={{str(p.relative_to(root/'pytools')):sha(p) for p in (root/'pytools').rglob('*') if p.is_file() and p.suffix in ('.py','.json')}}
binding=dict(host={host!r},runtime={config['runtime']!r},checkpoint={config['checkpoint']!r},deps={config['deps']!r},
 design_sha256=sha(root/'COOPBANDIT_DESIGN.md'),sources=sources,budget=budget(),resources=dict(load_max={config['load_max']},workers={config['workers']}))
setup(binding)
import sumo,subprocess
binary=pathlib.Path(sumo.__file__).parent/'bin/sumo'
assert binary.is_file()
version=subprocess.run([str(binary),'--version'],capture_output=True,text=True,check=True).stdout
assert '1.27.1' in version
for name,value in [('BINDING.json',binding),('APPROVAL.json',dict(owner_line={owner!r},registration_line={registered!r},
 design_sha256=binding['design_sha256'],registration_sha256={sha(REG)!r}))]:
 if (root/name).exists(): assert read(root/name)==value
 else: write(root/name,value)
if {host!r}=='cpuQ':
 attempts={{f's{{w["seed"]}}_screen':dict(sim_s=1140,phase='data') for w in worlds()}}
 for cell in range(12):
  w=next(w for w in worlds() if w['cell']==cell)
  attempts[f's{{w["seed"]}}_repeat']=dict(sim_s=900,phase='data')
 allocation=dict(host={host!r},attempts=attempts)
 if not (root/'ALLOCATION.json').exists(): write(root/'ALLOCATION.json',allocation)
 else: assert read(root/'ALLOCATION.json')==allocation
admission(root)
print(json.dumps(dict(host={host!r},runtime_verified=True,checkpoint_verified=True,source_count=len(sources),
 design_sha256=binding['design_sha256'],nice=os.getpriority(os.PRIO_PROCESS,0),binary=str(binary),sim_s=0)))
''')


def launch(host,phase,workers=None):
    config=HOSTS[host];root=config['root']
    workers=workers or config['workers']
    return remote(host,f'''import pathlib,json,subprocess,os,sys
root=pathlib.Path({root!r});assert root.is_dir()
sys.path[:0]={config['deps']!r}
import sumo
sumo_home=str(pathlib.Path(sumo.__file__).parent)
from_env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 PYTHONPATH=':'.join([{root!r}+'/pytools']+{config['deps']!r}),SUMO_HOME=sumo_home,
 PATH=sumo_home+'/bin:'+str(pathlib.Path(sys.executable).parent)+':'+os.environ.get('PATH',''))
log=(root/{('SUPERVISOR_'+phase+'.log')!r}).open('x')
p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopbandit.runner',str(root),
 '--phase',{phase!r},'--workers',{str(workers)!r}],cwd=root,env=from_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
print(json.dumps(dict(host={host!r},phase={phase!r},pid=p.pid,root=str(root),workers={workers})))
''')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode',choices=('inspect','audit','stage','cpu','deps','prepare','launch-screen','launch-branches'))
    p.add_argument('--host',default='cpuQ',choices=tuple(PYTHON))
    p.add_argument('--root',default=ROOT)
    a = p.parse_args()
    if a.mode == 'inspect':
        print(inspect(a.host))
    elif a.mode == 'prepare':
        print(prepare(a.host))
    elif a.mode.startswith('launch-'):
        print(launch(a.host,a.mode.split('-')[1]))
    elif a.mode == 'audit':
        print(json.dumps(audit(a.host),indent=2))
    elif a.mode == 'stage':
        print(stage(a.root,a.host))
    elif a.mode == 'deps':
        print(remote('cpuQ',f'''import pathlib,subprocess,sys,json
root=pathlib.Path({a.root!r});assert root.is_dir()
dest=root/'deps';assert not dest.exists()
subprocess.run([sys.executable,'-m','pip','install','--target',str(dest),'--no-cache-dir','scikit-learn==1.7.2','numpy==1.26.4','scipy==1.15.3','joblib==1.5.2','threadpoolctl==3.6.0'],check=True)
print(json.dumps(dict(deps=str(dest),new_venv=False)))
''',timeout=240))
    else:
        print(stage(a.root))
        print(remote('cpuQ', f'''import sys,os,unittest,json
from unittest.mock import patch
os.nice(19)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path.insert(0,{a.root!r}+'/pytools')
from tools.leverprobe.backend import setup
setup('/home/ubuntu/e3final','/home/ubuntu/mappospeed/production2/policy_latest.pt')
import numpy
sys.path.insert(0,'/home/ubuntu/coopbandit20261031_cpu1/deps')
with patch('subprocess.Popen',side_effect=AssertionError('SUMO disabled')):
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tools.coopbandit.test_cpu'))
print(json.dumps(dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0)))
assert result.wasSuccessful() and not result.skipped
'''))


if __name__ == '__main__':
    main()
