"""Short explicit remote operations; immutable source staging and CPU receipts."""
import argparse
import json
from pathlib import Path
from tools.coopbandit import ops as ssh
from .protocol import checked,sha,budget,worlds,registry_conflicts,REGISTRY,FORBIDDEN,authority

LOCAL=Path(__file__).resolve().parents[2]
EVIDENCE=LOCAL/'docs/paper_results/coopmwm'
CPU_ROOT='/home/ubuntu/coopmwm20261101_cpu1'
RUN_ROOT='/home/ubuntu/coopmwm20261101_v1'
GPU_ROOT='/root/coopmwm20261101_v1'
ssh.PYTHON['gpuA']='/root/miniconda3/bin/python'


def sources():
    files={}
    for package,base in [('coopmwm',LOCAL),('coopnative',LOCAL),('coopfree',LOCAL),
                         ('coopbandit',LOCAL),('leverprobe',Path('E:/agent-work/c4-leverprobe')),
                         ('coopbehav',Path('E:/agent-work/c4-coopbehav')),
                         ('coopplan',Path('E:/agent-work/c4-coopplan')),
                         ('wmteach',Path('E:/agent-work/c4-wmteach'))]:
        for p in (base/'tools'/package).glob('*.py'):
            files[f'tools/{package}/{p.name}']=checked(p).read_text(encoding='utf-8')
    return files


def stage(host,root):
    files=sources()
    files['COOPMWM_DESIGN.md']=checked(LOCAL/'docs/paper_results/COOPMWM_DESIGN.md').read_text(encoding='utf-8')
    return ssh.remote(host,f'''import pathlib,json,hashlib
root=pathlib.Path({root!r});assert root.parent.is_dir()
root.mkdir(exist_ok=True)
files=json.loads({json.dumps(files)!r})
hashes={{}}
for name,text in files.items():
 p=root/('pytools/'+name if name.startswith('tools/') else name)
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_text()==text,'immutable source '+name
 else:p.write_text(text)
 hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps(dict(root=str(root),hashes=hashes)))
''',timeout=60)


def cpu(root):
    from tools.coopnative.storage import write
    staged=json.loads(stage('cpuQ',root))
    print(json.dumps(dict(root=root,source_files=len(staged['hashes']))))
    out=ssh.remote('cpuQ',f'''import os,sys,pathlib,json,time,unittest
from unittest.mock import patch
os.nice(19)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',COOPMWM_MAPPO='/home/ubuntu/mappospeed/production2/policy_00169.pt')
sys.path[:0]=[{root!r}+'/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
import torch;torch.set_num_threads(1)
start=time.monotonic()
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromName(n) for n in ['tools.coopnative.test_cpu','tools.coopmwm.test_cpu']])
with patch('subprocess.Popen',side_effect=AssertionError('physics forbidden during CPU tests')):
 result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(suite)
print(json.dumps(dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0,wall_s=time.monotonic()-start,root={root!r})))
''',timeout=90)
    print(out)
    receipt=json.loads(out.strip().splitlines()[-1])
    write(EVIDENCE/('CPU_'+Path(root).name+'.json'),receipt)
    if not receipt['pass_'] or receipt['skips']:raise ValueError('CPU tests failed; preserve receipt')


def audit():
    from .storage import write
    registry=checked(REGISTRY).read_text(encoding='utf-8');authority(registry)
    conflicts=registry_conflicts(registry)
    if conflicts:raise ValueError('seed conflicts '+str(conflicts))
    seeds=[w['seed'] for role in ('train','check') for w in worlds(role)]
    reports=[]
    for host,home in [('cpuQ','/home/ubuntu'),('gpuA','/root'),('gpuR','/root')]:
        result=json.loads(ssh.remote(host,f'''import os,pathlib,re,json,time
forbidden=re.compile({FORBIDDEN.pattern!r},re.I)
ident=re.compile(r'(?<![0-9a-fA-F.])(?:s|w)?(?:'+'|'.join(map(str,{seeds!r}))+r')(?![0-9a-fA-F])')
hits=[];count=0
for parent,dirs,files in os.walk({home!r}):
 dirs[:]=[d for d in dirs if d not in ('.git','__pycache__','miniconda3','.cache','.npm','.rustup','.local') and not forbidden.search(str(pathlib.Path(parent)/d)) and not (pathlib.Path(parent)/d).is_symlink()]
 for name in dirs+files:
  p=pathlib.Path(parent)/name
  if forbidden.search(str(p)):continue
  count+=1
  if ident.search(name):hits.append(str(p))
print(json.dumps(dict(host={host!r},safe_names=count,hits=hits,pass_=not hits,load=os.getloadavg(),unix=time.time())))
''',timeout=60))
        write(EVIDENCE/f'EXPOSURE_{host}.json',dict(result,registry_sha256=sha(REGISTRY),sim_s=0))
        reports.append(result)
        print(json.dumps(result))
    return dict(pass_=all(r['pass_'] for r in reports),reports=reports,conflicts=conflicts)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','stage','audit'])
    p.add_argument('--root',default=CPU_ROOT);p.add_argument('--host',default='cpuQ')
    a=p.parse_args()
    if a.mode=='cpu':cpu(a.root)
    elif a.mode=='audit':audit()
    else:
        r=json.loads(stage(a.host,a.root))
        print(json.dumps(dict(root=r['root'],files=len(r['hashes']))))
