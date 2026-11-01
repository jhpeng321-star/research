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
ssh.PYTHON['cpuP']='/home/ubuntu/miniconda3/bin/python'


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


def recover(host):
    home='/home/ubuntu' if host.startswith('cpu') else '/root'
    print(ssh.remote(host,f'''import pathlib,os,json,subprocess
home=pathlib.Path({home!r})
roots=[dict(path=str(p),receipts=[x.name for x in p.glob('*.json')]) for p in home.glob('coopmwm*') if p.is_dir()]
processes=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
  if 'tools.coopmwm.' in cmd:processes.append(dict(pid=int(p.name),cmd=cmd))
 except (OSError,PermissionError):pass
print(json.dumps(dict(host={host!r},load=os.getloadavg(),roots=roots,processes=processes)))
if {host!r}.startswith('gpu'):
 print(subprocess.check_output(['nvidia-smi'],text=True))
''',timeout=40))


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


def plan_seeds():
    return sorted({w['seed'] for i in range(3) for w in worlds('train',i)}|
                  {w['seed'] for r in ('check','final_test') for w in worlds(r)})


def audit_host(host):
    """Names only, no reading any simulation data; one short host call."""
    from .storage import write
    home='/home/ubuntu' if host.startswith('cpu') else '/root'
    source=f'''import os,pathlib,re,json,time
forbidden=re.compile({FORBIDDEN.pattern!r}+r'|hetdata_truth|vehicle_params|driver_params|vtype',re.I)
ident=re.compile(r'(?<![0-9a-fA-F.])(?:s|w)?(?:'+'|'.join(map(str,{plan_seeds()!r}))+r')(?![0-9a-fA-F])')
hits=[];count=0
for parent,dirs,files in os.walk({home!r}):
 dirs[:]=[d for d in dirs if d not in ('.git','__pycache__','miniconda3','.cache','.npm','.rustup','.local') and not forbidden.search(str(pathlib.Path(parent)/d)) and not (pathlib.Path(parent)/d).is_symlink()]
 for name in dirs+files:
  p=pathlib.Path(parent)/name
  if forbidden.search(str(p)):continue
  count+=1
  if ident.search(name):hits.append(str(p))
print(json.dumps(dict(host={host!r},safe_names=count,hits=hits,pass_=not hits,unix=time.time())))
'''
    result=json.loads(ssh.remote(host,source,timeout=60))
    write(EVIDENCE/f'FINAL_EXPOSURE_{host}.json',dict(result,seed_plan_sha256=__import__('tools.coopmwm.protocol',fromlist=['digest']).digest(plan_seeds()),sim_s=0))
    print(json.dumps(result))


def audit_protocols():
    """Explicit safe protocol/source paths, never a recursive content grep."""
    import os,re
    from .protocol import digest
    from .storage import write
    forbidden=re.compile(FORBIDDEN.pattern+r'|hetdata_truth|vehicle_params|driver_params|vtype',re.I)
    seeds=plan_seeds();hits=[];files={}
    reg=checked(REGISTRY);authority(reg.read_text(encoding='utf-8'))
    conflicts=registry_conflicts(reg.read_text(encoding='utf-8'),set(seeds))
    if conflicts:hits.append(dict(path=str(reg),lines=conflicts))
    patterns=('tools/*/protocol.py','tools/*/design.py','tools/*/registration*.json',
              'docs/paper_results/*DESIGN.md','*REGISTRATION*.md','protocol*.json')
    for base in Path('E:/agent-work').iterdir():
        if not base.is_dir() or forbidden.search(str(base)) or base.is_symlink():continue
        paths={p for pattern in patterns for p in base.glob(pattern)}
        for p in sorted(paths):
            if forbidden.search(str(p)) or not p.is_file():continue
            if 'c4-coopnative' in p.parts and ('coopmwm' in p.parts or p.name=='COOPMWM_DESIGN.md'):continue
            text=p.read_text(encoding='utf-8',errors='replace')
            # The seed0 block has already been prospectively reserved by the
            # explicitly superseded COOPNATIVE protocol. Fresh blocks must be absent.
            fresh=set(seeds)-set(range(986000,986084))
            lines=registry_conflicts(text,fresh);files[str(p)]=sha(p)
            if lines:hits.append(dict(path=str(p),lines=lines))
    result=dict(pass_=not hits,hits=hits,files_scanned=len(files),files_sha256=digest(files),registry_sha256=sha(reg),seed_plan_sha256=digest(seeds),sim_s=0)
    write(EVIDENCE/'FINAL_PROTOCOL_AUDIT.json',result);print(json.dumps(result))


def plan():
    from .protocol import digest
    from .storage import read,write
    audits=[read(EVIDENCE/f'FINAL_EXPOSURE_{h}.json') for h in ('cpuQ','gpuA','gpuR')]
    protocols=read(EVIDENCE/'FINAL_PROTOCOL_AUDIT.json')
    assert protocols['pass_'] and all(a['pass_'] for a in audits)
    panel=dict(worlds=worlds('final_test'),panel_sha256=digest(worlds('final_test')),
        access='SEALED_UNTIL_ALL_TRAINING_COMPLETE',run_count=1,reruns=0,
        arms=['MAPPO169','native','A_seed0_update24','B_seed0_update24'],
        precharged_sim_s=172800,warmup_s=300,control_s=900,
        checkpoint_rule='fixed final update24 of primary seed20261101; no outcome selection',
        analysis='paired whole-world bootstrap10000 RNG20261101 95CI: B-MAPPO/B-A/A-MAPPO J600/J900; per-cell descriptives; collisions/TTC/hard braking; B detector-first complete exact C_G twins')
    write(EVIDENCE/'FINAL_TEST_PANEL.json',panel)
    quote=dict(budget(),status='REGISTER_BEFORE_SUMO',new_sumo_sim_s=0,
        plan='three independent training seeds; 24updates per arm per seed; final24 fixed',
        checks='u4/8/12/16/20/24 each seed A+B, common12MAPPO controls once',
        dev='u4 primaryseed and u24 all3seeds, A+B; reuse24MAPPO cache',
        exposure_hosts=[a['host'] for a in audits],protocols=protocols,
        extra_host_audit=dict(cpuP='SSH connection closed; unavailable; not admitted for execution'),
        cpu=read(EVIDENCE/'CPU_coopmwm20261101_cpu6.json'),design_sha256=sha(LOCAL/'docs/paper_results/COOPMWM_DESIGN.md'))
    write(EVIDENCE/'BUDGET_QUOTE.json',quote)
    print(json.dumps(dict(budget=budget(),panel_sha256=panel['panel_sha256'],design_sha256=quote['design_sha256'],quote_sha256=sha(EVIDENCE/'BUDGET_QUOTE.json'))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','stage','audit','audit-host','audit-protocols','plan','recover'])
    p.add_argument('--root',default=CPU_ROOT);p.add_argument('--host',default='cpuQ')
    a=p.parse_args()
    if a.mode=='recover':recover(a.host)
    elif a.mode=='cpu':cpu(a.root)
    elif a.mode=='audit':audit()
    elif a.mode=='audit-host':audit_host(a.host)
    elif a.mode=='audit-protocols':audit_protocols()
    elif a.mode=='plan':plan()
    else:
        r=json.loads(stage(a.host,a.root))
        print(json.dumps(dict(root=r['root'],files=len(r['hashes']))))
