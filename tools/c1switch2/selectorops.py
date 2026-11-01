"""Final-only selector evaluation on the original disjoint CPU world partitions."""
import argparse
import json

from tools.c1switch.design import checked, sha
from tools.c1switch.storage import canonical, write
from . import operations
from .audit import registration_audit
from .design import REGISTRATION
from .gate import approval, cpu_authority
from .operations import EVIDENCE, env, ssh


def copy_fleet(spec):
    """Copy only the final receipt-pinned checkpoint and declared Python sources."""
    archive = ssh('cpuQ', f'''import base64,hashlib,json
from pathlib import Path
spec={spec!r};out={{}}
for name,path,digest in [('checkpoint.pt',spec['checkpoint'],spec['checkpoint_sha256']),('FROZEN_FINAL.json',spec['frozen_receipt'],spec['frozen_receipt_sha256'])]+[('peer/'+n,str(Path(spec['root'])/n),v) for n,v in spec['sources'].items()]:
 raw=Path(path).read_bytes();assert hashlib.sha256(raw).hexdigest()==digest
 out[name]=base64.b64encode(raw).decode()
print(json.dumps(out))
''', timeout=45)
    root = '/root/c1switch2_selector_fleet_'+canonical(spec)[:16]
    result = ssh('gpuR', f'''import base64,hashlib,json
from pathlib import Path
root=Path({root!r});assert root.parent.is_dir() and not root.exists(),'preserve existing selector copy'
root.mkdir();sources={{}}
for name,value in {archive!r}.items():
 path=root/name;path.parent.mkdir(parents=True,exist_ok=True);raw=base64.b64decode(value);path.write_bytes(raw)
 sources[name]=hashlib.sha256(raw).hexdigest()
print(json.dumps(dict(root=str(root),sources=sources)))
''', timeout=45)
    write(EVIDENCE/'SELECTOR_COPY_fleet_ppo_gpuR.json', result)
    return dict(spec, root=root+'/peer', checkpoint=root+'/checkpoint.pt',
                frozen_receipt=root+'/FROZEN_FINAL.json')


def copy_per_cav(spec, peers):
    entries = [('checkpoint.pt', spec['checkpoint'], spec['checkpoint_sha256']),
               ('FROZEN_FINAL.json', spec['frozen_receipt'], spec['frozen_receipt_sha256'])]
    for key, package in peers.items():
        entries += [(key+'/'+n, package['root']+'/'+n, v) for n, v in package['sources'].items()]
    archive = ssh('cpuQ', f'''import base64,hashlib,json
from pathlib import Path
out={{}}
for name,path,digest in {entries!r}:
 raw=Path(path).read_bytes();assert hashlib.sha256(raw).hexdigest()==digest
 out[name]=base64.b64encode(raw).decode()
print(json.dumps(out))
''', timeout=45)
    root = '/root/c1switch2_selector_cav_'+canonical(dict(spec=spec, peers=peers))[:16]
    result = ssh('gpuR', f'''import base64,hashlib,json
from pathlib import Path
root=Path({root!r});assert root.parent.is_dir() and not root.exists(),'preserve existing selector copy'
root.mkdir();sources={{}}
for name,value in {archive!r}.items():
 path=root/name;path.parent.mkdir(parents=True,exist_ok=True);raw=base64.b64decode(value);path.write_bytes(raw)
 sources[name]=hashlib.sha256(raw).hexdigest()
print(json.dumps(dict(root=str(root),sources=sources)))
''', timeout=45)
    write(EVIDENCE/'SELECTOR_COPY_cav_mappo_gpuR.json', result)
    peers = {key: dict(package, root=root+'/'+key) for key, package in peers.items()}
    return dict(spec, root=peers['shared']['root'], checkpoint=root+'/checkpoint.pt',
                frozen_receipt=root+'/FROZEN_FINAL.json'), peers


def prepare(host, resume_root=None, arm='fleet_ppo'):
    if arm not in ('fleet_ppo', 'cav_mappo'):
        raise ValueError('registered selector arm required')
    path = EVIDENCE/('SELECTOR_PREPARED_'+arm+'_'+host+'.json')
    if path.exists():
        raise ValueError('selector already prepared; inspect existing receipt')
    peers = None
    if arm == 'cav_mappo':
        from .selectorhandoff import load_per_cav
        spec, peers = load_per_cav()
    else:
        delivery = json.loads(checked('SELMAPPO_FINAL.json').read_text())
        spec = delivery['spec']
        if (delivery['host'] != 'cpuQ' or delivery['receipt']['iteration'] != 169 or
                delivery['receipt']['status'] != 'FROZEN_FINAL' or
                spec['checkpoint_sha256'] != delivery['receipt']['checkpoint_sha256']):
            raise ValueError('genuine peer final169 delivery required')
    if resume_root:
        # Reuse the already-passed immutable preflight after a CPU-only probe
        # setup error. No claims/jobs/charges may exist; do not rerun preflight.
        identity = resume_root.rsplit('_', 1)[-1]
        staged = json.loads((EVIDENCE/('STAGE_'+host+'_'+identity+'.json')).read_text())
        if staged['root'] != resume_root:
            raise ValueError('exact recorded zero-physics stage required')
        preflight = json.loads((EVIDENCE/('PREFLIGHT_'+host+'_'+resume_root.rsplit('/', 1)[-1]+'.json')).read_text())
    else:
        if host == 'gpuR':
            if arm == 'cav_mappo':
                spec, peers = copy_per_cav(spec, peers)
            else:
                spec = copy_fleet(spec)
        staged = operations.stage(host, selector=arm, selector_spec=spec, execution_peers=peers)
        preflight = operations.preflight(host, staged['root'])
    if preflight['status'] != 'PASS':
        raise ValueError('immutable selector CPU preflight failed; no physics')
    probe = ssh(host, f'''import json,os,sys,sysconfig
from pathlib import Path
root=Path({staged['root']!r});os.nice(19);os.environ.update({env(host, staged['root'])!r})
os.environ['PATH']=str(Path(sysconfig.get_paths()['purelib'])/'sumo/bin')+':'+os.environ['PATH']
assert not (root/'accounting.jsonl').exists() and not list((root/'jobs').iterdir())
assert not (root/'SELECTOR_DISPATCH_ONCE.json').exists()
sys.path.insert(0,str(root/'pytools'));os.chdir(root/'pytools')
from tools.c1switch2.selectors import Fleet,PerCav,package
from tools.c1switch2.hostcheck import belief
from tools.c1switch2.runtime import Runtime
binding=json.loads((root/'BINDING.json').read_text());Runtime(binding)
assert binding['enabled_arms']==[{arm!r}] and binding['cpu_host']=={host!r}
selector=(Fleet if {arm!r}=='fleet_ppo' else PerCav)(binding['selector_specs'][{arm!r}])
raw=belief(now=300)
if {arm!r}=='cav_mappo':
 import importlib
 shared=package(binding['execution_peers']['shared'],{{'runtime.py','environment.py','observations.py'}})
 mixed=package(binding['execution_peers']['per_cav'],{{'mixed.py'}})
 factory=importlib.import_module(mixed+'.mixed').make_mixed_fleet(importlib.import_module(shared+'.runtime'))
 assert callable(factory)
 selector.sync_incumbents({{r['vehicle_id']:'native' for r in raw['history'][-1]['self_reports']}})
choice=selector.select(raw)
assert (choice in binding['library']) if isinstance(choice,str) else all(v in binding['library'] for v in choice.values())
out=dict(status='PASS',iteration=169,checkpoint_sha256={spec['checkpoint_sha256']!r},choice=choice,new_sim_s=0,new_gpu_s=0)
with (root/'SELECTOR_CPU.json').open('x') as stream:json.dump(out,stream,indent=2)
print(json.dumps(out))
''', timeout=45)
    result = dict(staged, host=host, arm=arm, selector_probe=probe)
    write(path, result)
    print(json.dumps(result, indent=2))


def dispatch(host, workers, arm='fleet_ppo'):
    if arm not in ('fleet_ppo', 'cav_mappo'):
        raise ValueError('registered selector arm required')
    prepared = json.loads((EVIDENCE/('SELECTOR_PREPARED_'+arm+'_'+host+'.json')).read_text())
    text = checked(REGISTRATION).read_text(encoding='utf-8')
    audit = registration_audit(REGISTRATION)
    if not approval(text) or not cpu_authority(text) or audit['hits']:
        raise ValueError('current selector authority/registry absent')
    marker = 'FINAL FLEET' if arm == 'fleet_ppo' else 'FINAL PER-CAV'
    if not any('S-C1SWITCH2 '+marker+' EVALUATION REGISTERED' in line and
               prepared['selector_probe']['checkpoint_sha256'] in line for line in text.splitlines()):
        raise ValueError('final checkpoint evaluation must be prospectively registered')
    archive = json.loads((EVIDENCE/'CANARY_BINDING_c1switch2_run_4ad4ece01d5a9ac1.json').read_text())
    frozen = {n: v for n, v in archive['BINDING.json']['sources'].items()
              if n.startswith('tools/c1switch/') or n == 'engine/branching.py'}
    exposures = {h: sha(EVIDENCE/('EXPOSURE_'+h+'_v3.json')) for h in ('cpuQ', 'gpuA', 'gpuR')}
    if not all(json.loads((EVIDENCE/('EXPOSURE_'+h+'_v3.json')).read_text())['pass'] for h in exposures):
        raise ValueError('retained exposure audit failed')
    if not 1 <= workers <= (14 if host == 'cpuQ' else 3):
        raise ValueError('selector worker cap; reserve one gpuR slot for original WM oracles')
    root = prepared['root']
    result = ssh(host, f'''import hashlib,json,os,shutil,subprocess,sys,sysconfig,time
from pathlib import Path
root=Path({root!r});assert root.name.startswith('c1switch2_run_')
assert os.getloadavg()[0]<{28 if host == 'cpuQ' else 20} and shutil.disk_usage(root).free>=2*1024**3,'yield CPU/disk pressure'
binding=json.loads((root/'BINDING.json').read_text())
assert binding['dispatch_phase']=='selector' and binding['enabled_arms']==[{arm!r}] and binding['cpu_host']=={host!r}
assert hashlib.sha256((root/'REGISTRATION.md').read_bytes()).hexdigest()==binding['registration_sha256']
assert all(hashlib.sha256((root/'pytools'/n).read_bytes()).hexdigest()==v for n,v in binding['sources'].items())
assert all(binding['sources'][n]==v for n,v in {frozen!r}.items()),'frozen v7 changed'
assert json.loads((root/'PREFLIGHT.json').read_text())['status']=='PASS'
assert json.loads((root/'SELECTOR_CPU.json').read_text())['status']=='PASS'
assert not (root/'STOP').exists() and not (root/'accounting.jsonl').exists() and not list((root/'jobs').iterdir())
for entry in os.scandir(root.parent):
 if entry.name.startswith('c1switch2_run_') and entry.is_dir(follow_symlinks=False):
  for name in ('SELECTOR_DISPATCH_ONCE.json','MODEL_DISPATCH_ONCE.json'):
   claim=Path(entry.path,name)
   if claim.exists():assert json.loads(claim.read_text())['arm']!={arm!r},'selector arm already claimed'
sys.path.insert(0,str(root/'pytools'));os.chdir(root/'pytools');os.nice(19);os.environ.update({env(host, root)!r})
os.environ['PATH']=str(Path(sysconfig.get_paths()['purelib'])/'sumo/bin')+':'+os.environ['PATH']
from tools.c1switch.storage import canonical
from tools.c1switch2.selectors import Fleet,PerCav,package
from tools.c1switch2.gate import REQUIRED,CPU_REQUIRED,admission
from tools.c1switch2.design import local_budget
from tools.c1switch2.runtime import Runtime
Runtime(binding)
(Fleet if {arm!r}=='fleet_ppo' else PerCav)(binding['selector_specs'][{arm!r}])
if {arm!r}=='cav_mappo':
 for package_spec in binding['execution_peers'].values():package(package_spec,{{'__init__.py'}})
needed=CPU_REQUIRED+({arm!r},)
evidence={{k:dict(status='PASS' if k in needed else 'PENDING',binding_sha256=canonical(binding),source='immutable CPU/source/WALL/canary/exposure/final169 selector validation') for k in REQUIRED}}
evidence.update(registry_audit={audit!r},exposure_receipt_sha256={exposures!r},mechanics_sha256={archive['canary_sha256']!r})
assert admission((root/'REGISTRATION.md').read_text(),binding,evidence)['dispatch_authorized']
with (root/'GATES.json').open('x') as stream:json.dump(evidence,stream,indent=2)
claim=dict(arm={arm!r},host={host!r},root=str(root),planned_sim_s=local_budget({host!r})[{arm!r}],unix=time.time())
with (root/'SELECTOR_DISPATCH_ONCE.json').open('x') as stream:
 json.dump(claim,stream,indent=2);stream.flush();os.fsync(stream.fileno())
environment=dict(os.environ,**{env(host, root)!r},PYTHONPATH=str(root/'pytools'))
environment['PATH']=str(Path(sysconfig.get_paths()['purelib'])/'sumo/bin')+':'+environment['PATH']
with (root/'SELECTOR.log').open('x') as log:
 process=subprocess.Popen([sys.executable,'-B','-u','-m','tools.c1switch2.supervisor','--root',str(root),'--host',{host!r},'--workers',{str(workers)!r}],cwd=root/'pytools',env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
out=dict(claim,pid=process.pid,workers={workers!r},new_gpu_s=0)
with (root/'DISPATCH_SELECTOR.json').open('x') as stream:json.dump(out,stream,indent=2)
print(json.dumps(out))
''', timeout=60)
    write(EVIDENCE/('SELECTOR_DISPATCH_'+arm+'_'+host+'.json'), result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'dispatch'))
    parser.add_argument('--host', choices=('cpuQ', 'gpuR'), required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--resume-root')
    parser.add_argument('--arm', choices=('fleet_ppo','cav_mappo'), default='fleet_ppo')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.host, args.resume_root, args.arm)
    else:
        dispatch(args.host, args.workers, args.arm)
