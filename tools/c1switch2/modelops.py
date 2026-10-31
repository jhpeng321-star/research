"""Independent frozen-model arm admission, synthetic parity and durable two-GPU relay."""
import argparse
import copy
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

from tools.c1switch.design import checked, sha
from tools.c1switch.storage import canonical, exclusive, write

from .design import CONTROLLERS, REGISTRATION, assigned, worlds
from .gate import approval, cpu_authority
from .hostcheck import belief
from .operations import EVIDENCE, env, ssh


def parity_requests(engine):
    raw = belief()
    continued = copy.deepcopy(raw)
    continued['controller_state'].update(incumbent='wall', wall=dict(schema='c1switch2.wall/1',
        started_s=300., last_s=319., cooldown_s=None, members=['a', 'b'], speed_mps=17.,
        phase=1, assembly_since_s=300., assembled_at_s=301.))
    rows = []
    for name, payload in [('fresh', raw), ('continued_wall', continued)]:
        request = dict(protocol=engine['protocol'], id='parity_'+name, belief=payload,
                       binding=engine, K=8, horizon_s=60)
        rows.append(dict(request, request_sha256=canonical(request)))
    return rows


def compare_parity(results):
    for name in ('fresh', 'continued_wall'):
        a, b = (results[host][name]['samples'] for host in ('gpuA', 'gpuR'))
        if set(a) != set(CONTROLLERS) or set(a) != set(b) or any(len(a[c]) != 8 or len(b[c]) != 8 for c in a):
            raise ValueError('six-way K8 parity shape mismatch')
        if any(not math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-6)
               for c in a for x, y in zip(a[c], b[c])):
            raise ValueError('cross-GPU sample parity failed')
    for host in ('gpuA', 'gpuR'):
        if results[host]['fresh']['samples']['wall'] == results[host]['continued_wall']['samples']['wall']:
            raise ValueError('GPU incumbent WALL continuation missing')


def parity(config, submit=False):
    config = json.loads(checked(config).read_text())
    requests = parity_requests(config['engine'])
    results = {}
    for host, root in config['gpus'].items():
        results[host] = ssh(host, f'''import json,sys
from pathlib import Path
root=Path({root!r});sys.path.insert(0,str(root/'pytools'))
from tools.c1switch.storage import canonical,write
assert root.name.startswith('c1switch2_gpu_')
assert not (root/'rpc/STOP').exists(),'service stopped'
ready=json.loads((root/'rpc/READY.json').read_text());assert ready['binding']=={config['engine']!r}
out={{}}
for name,request in zip(('fresh','continued_wall'),{requests!r}):
 key=request['request_sha256'];path=root/'rpc'/(key+'.request.json')
 if {submit!r} and not path.exists():write(path,request)
 finish=root/'rpc'/(key+'.result.json')
 if finish.exists():
  row=json.loads(finish.read_text());assert row['request_sha256']==key and row['binding']==ready['binding']
  out[name]=row
print(json.dumps(out))
''', timeout=30)
    if any(set(rows) != {'fresh', 'continued_wall'} for rows in results.values()):
        print(json.dumps(dict(status='PENDING', received={h: list(v) for h, v in results.items()})))
        return None
    compare_parity(results)
    out = dict(status='PASS', model=config['engine']['model'], engine=config['engine'],
               numerical_tolerance=dict(rtol=1e-6, atol=1e-6), results=results,
               gpu_roots=config['gpus'], new_sim_s=0)
    path = EVIDENCE/('GPU_PARITY_'+config['engine']['model']+'.json')
    if path.exists():
        if json.loads(path.read_text()) != out:
            raise ValueError('preserve previous parity receipt')
    else:
        write(path, out)
    print(json.dumps(dict(status='PASS', path=str(path), sha256=sha(path))))
    return out


def dispatch(host, root, model, workers):
    from .audit import registration_audit
    text = checked(REGISTRATION).read_text(encoding='utf-8')
    audit = registration_audit(REGISTRATION)
    if not approval(text) or not cpu_authority(text) or audit['hits']:
        raise ValueError('current model authority/registry absent')
    parity_path = EVIDENCE/('GPU_PARITY_'+model+'.json')
    p = json.loads(parity_path.read_text())
    if p['status'] != 'PASS':
        raise ValueError('real both-GPU parity missing')
    compare_parity(p['results'])
    canary = EVIDENCE/'CANARY_BINDING_c1switch2_run_4ad4ece01d5a9ac1.json'
    archive = json.loads(canary.read_text())
    frozen = {n: v for n, v in archive['BINDING.json']['sources'].items()
              if n.startswith('tools/c1switch/') or n == 'engine/branching.py'}
    exposures = {h: json.loads((EVIDENCE/('EXPOSURE_'+h+'_v3.json')).read_text())['pass']
                 for h in ('cpuQ', 'gpuA', 'gpuR')}
    if not all(exposures.values()) or not 1 <= workers <= (14 if host == 'cpuQ' else 4):
        raise ValueError('exposure/worker admission')
    result = ssh(host, f'''import hashlib,json,os,subprocess,sys,sysconfig,time
from pathlib import Path
root=Path({root!r});assert root.name.startswith('c1switch2_run_')
binding=json.loads((root/'BINDING.json').read_text());assert binding['enabled_arms']==[{model!r}] and binding['dispatch_phase']=='model'
assert binding['cpu_host']=={host!r} and binding['inference_specs'][{model!r}]['engine']=={p['engine']!r}
assert hashlib.sha256((root/'REGISTRATION.md').read_bytes()).hexdigest()==binding['registration_sha256']
assert all(hashlib.sha256((root/'pytools'/n).read_bytes()).hexdigest()==v for n,v in binding['sources'].items())
assert all(binding['sources'][n]==v for n,v in {frozen!r}.items()),'v7 changed'
assert json.loads((root/'PREFLIGHT.json').read_text())['status']=='PASS'
assert not (root/'STOP').exists() and not (root/'accounting.jsonl').exists() and not list((root/'jobs').iterdir())
assert os.getloadavg()[0]<{28 if host == 'cpuQ' else 20},'yield spare CPU priority'
for entry in os.scandir(root.parent):
 if entry.name.startswith('c1switch2_run_') and entry.is_dir(follow_symlinks=False):
  claim=Path(entry.path,'MODEL_DISPATCH_ONCE.json')
  if claim.exists():assert json.loads(claim.read_text())['arm']!={model!r},'model panel already claimed'
sys.path.insert(0,str(root/'pytools'));os.chdir(root/'pytools')
from tools.c1switch.storage import canonical
from tools.c1switch2.gate import REQUIRED,CPU_REQUIRED,admission
from tools.c1switch2.design import local_budget
needed=CPU_REQUIRED+('inference_'+{model!r},'gpuA_parity','gpuR_parity')
evidence={{k:dict(status='PASS' if k in needed else 'PENDING',binding_sha256=canonical(binding),source='CPU/frozen canary/preallocation exposure/currentregistry/real GPU parity') for k in REQUIRED}}
evidence['gpu_parity_sha256']={sha(parity_path)!r};evidence['registry_audit']={audit!r};evidence['mechanics_sha256']={archive['canary_sha256']!r}
assert admission((root/'REGISTRATION.md').read_text(),binding,evidence)['dispatch_authorized']
for route in binding['inference_specs'][{model!r}]['routes'].values():(root/route).mkdir(parents=True)
with (root/'GATES.json').open('x') as stream:json.dump(evidence,stream,indent=2)
claim=dict(arm={model!r},host={host!r},root=str(root),planned_sim_s=local_budget({host!r})[{model!r}],unix=time.time())
with (root/'MODEL_DISPATCH_ONCE.json').open('x') as stream:
 json.dump(claim,stream,indent=2);stream.flush();os.fsync(stream.fileno())
environment=os.environ.copy();environment.update({env(host, root)!r})
environment['PYTHONPATH']=str(root/'pytools');environment['PATH']=str(Path(sysconfig.get_paths()['purelib'])/'sumo/bin')+':'+environment['PATH']
with (root/'MODEL.log').open('x') as log:
 process=subprocess.Popen([sys.executable,'-B','-u','-m','tools.c1switch2.supervisor','--root',str(root),'--host',{host!r},'--workers',{str(workers)!r}],cwd=root/'pytools',env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.nice(19))
out=dict(claim,pid=process.pid,workers={workers!r})
with (root/'DISPATCH_MODEL.json').open('x') as stream:json.dump(out,stream,indent=2)
print(json.dumps(out))
''', timeout=45)
    write(EVIDENCE/('MODEL_DISPATCH_'+model+'_'+host+'.json'), result)
    print(json.dumps(result, indent=2))


def transfer(cpu, root, gpu, gpu_root, model):
    route = root+'/rpc/'+model+'/'+gpu
    requests = ssh(cpu, f'''import json
from pathlib import Path
root=Path({route!r});out={{}}
for path in sorted(root.glob('*.request.json')):
 if not path.with_name(path.name.replace('.request.','.result.')).exists():out[path.name]=json.loads(path.read_text())
print(json.dumps(out))
''', timeout=30)
    if not requests:
        return 0
    if any(not re.fullmatch(r'[0-9a-f]{64}\.request\.json', n) for n in requests):
        raise ValueError('invalid relay identity')
    results = ssh(gpu, f'''import json,sys
from pathlib import Path
root=Path({gpu_root!r});sys.path.insert(0,str(root/'pytools'))
from tools.c1switch.storage import canonical,write
out={{}}
for name,row in {requests!r}.items():
 key=row['request_sha256'];body=dict(row);body.pop('request_sha256');assert canonical(body)==key and name==key+'.request.json'
 path=root/'rpc'/name
 if path.exists():assert json.loads(path.read_text())==row
 else:write(path,row)
 finish=root/'rpc'/(key+'.result.json')
 if finish.exists():out[finish.name]=json.loads(finish.read_text())
print(json.dumps(out))
''', timeout=30)
    if results:
        ssh(cpu, f'''import json,sys
from pathlib import Path
root=Path({root!r});sys.path.insert(0,str(root/'pytools'))
from tools.c1switch.storage import write
route=Path({route!r})
for name,row in {results!r}.items():
 request=json.loads((route/name.replace('.result.','.request.')).read_text())
 assert row['request_sha256']==request['request_sha256'] and row['binding']==request['binding']
 path=route/name
 if path.exists():assert json.loads(path.read_text())==row
 else:write(path,row)
print(json.dumps(dict(count={len(results)})))
''', timeout=30)
    return len(results)


def relay(config_path):
    config_path = checked(config_path)
    config = json.loads(config_path.read_text())
    local = EVIDENCE/('RELAY_'+config['engine']['model'])
    with exclusive(local):
        if (local/'START.json').exists():
            raise ValueError('relay already claimed; inspect own process before recovery')
        import os
        write(local/'START.json', dict(pid=os.getpid(), config_sha256=sha(config_path), unix=time.time()))
        while True:
            try:
                statuses = {}
                for cpu, root in config['cpus'].items():
                    count = sum(assigned(w['seed']) == cpu for w in worlds())
                    statuses[cpu] = ssh(cpu, f'''import json
from pathlib import Path
root=Path({root!r})
print(json.dumps(dict(stop=json.loads((root/'STOP').read_text()) if (root/'STOP').exists() else None,episodes=len(list((root/'jobs').glob('*__episode/PHYSICS.json'))),expected={count},complete=(root/'supervisor/COMPLETE.json').exists())))
''', timeout=30)
                done = all(s['episodes'] == s['expected'] for s in statuses.values())
                failed = any(s['stop'] for s in statuses.values())
                gpu_status = {}
                for gpu, root in config['gpus'].items():
                    gpu_status[gpu] = ssh(gpu, f'''import json
from pathlib import Path
root=Path({root!r});path=root/'rpc/COMPLETE'
if {done or failed!r} and not path.exists():
 with path.open('x') as stream:json.dump(dict(reason='all model episodes ended or CPU STOP'),stream)
print(json.dumps(dict(stop=json.loads((root/'rpc/STOP').read_text()) if (root/'rpc/STOP').exists() else None,finished=json.loads((root/'rpc/GPU_FINISHED.json').read_text()) if (root/'rpc/GPU_FINISHED.json').exists() else None)))
''', timeout=30)
                if any(s['stop'] for s in gpu_status.values()):
                    for cpu, root in config['cpus'].items():
                        ssh(cpu, f'''import json
from pathlib import Path
path=Path({root!r})/'STOP'
if not path.exists():
 with path.open('x') as stream:json.dump(dict(reason='owned model GPU service STOP',gpu_status={gpu_status!r}),stream)
print(json.dumps(dict(stopped=True)))
''', timeout=30)
                    failed = True
                transferred = 0
                if not done and not failed:
                    for cpu, root in config['cpus'].items():
                        for gpu, gpu_root in config['gpus'].items():
                            transferred += transfer(cpu, root, gpu, gpu_root, config['engine']['model'])
                write(local/'STATUS.json', dict(unix=time.time(), cpus=statuses, gpus=gpu_status, transferred=transferred))
                if failed or (done and all(s['finished'] for s in gpu_status.values())):
                    write(local/'COMPLETE.json', dict(unix=time.time(), episodes_done=done, failed=failed, gpus=gpu_status))
                    return
            except Exception as exc:
                # Re-send immutable RPC bytes only; never re-spawn physics or GPU services.
                write(local/'TRANSPORT_ERROR.json', dict(unix=time.time(), error=repr(exc)))
            time.sleep(1)


def relay_start(config):
    config = checked(config).resolve()
    model = json.loads(config.read_text())['engine']['model']
    path = EVIDENCE/('RELAY_DISPATCH_'+model+'.json')
    if path.exists():
        raise ValueError('relay already dispatched')
    assert EVIDENCE.is_dir()
    with (EVIDENCE/('RELAY_'+model+'.log')).open('x') as log:
        process = subprocess.Popen([sys.executable, '-B', '-u', '-m', 'tools.c1switch2.modelops',
                                    'relay', '--config', str(config)], stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS)
    out = dict(pid=process.pid, config=str(config), config_sha256=sha(config), unix=time.time())
    write(path, out)
    print(json.dumps(out))


def prepare(config_path):
    from .operations import stage, preflight
    config = json.loads(checked(config_path).read_text())
    model = config['engine']['model']
    parity_receipt = json.loads((EVIDENCE/('GPU_PARITY_'+model+'.json')).read_text())
    if parity_receipt['engine'] != config['engine']:
        raise ValueError('GPU parity engine differs from proposed task binding')
    compare_parity(parity_receipt['results'])
    result = dict(config, cpus={})
    path = EVIDENCE/(model.upper()+'_EXECUTION_BOUND.json')
    if path.exists():
        raise ValueError('preserve existing prepared execution')
    inference = dict(status='FROZEN_READY', engine=config['engine'],
                     routes={gpu: 'rpc/'+model+'/'+gpu for gpu in ('gpuA', 'gpuR')})
    for host in ('cpuQ', 'gpuR'):
        staged = stage(host, model=model, inference=inference)
        if preflight(host, staged['root'])['status'] != 'PASS':
            raise ValueError('model CPU preflight failed before dispatch')
        result['cpus'][host] = staged['root']
    write(path, result)
    print(json.dumps(dict(path=str(path), sha256=sha(path), cpus=result['cpus'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('parity-submit', 'parity-collect', 'dispatch', 'relay', 'relay-start', 'prepare'))
    parser.add_argument('--config')
    parser.add_argument('--host', choices=('cpuQ', 'gpuR'))
    parser.add_argument('--root')
    parser.add_argument('--model', choices=('wm', 'bb'))
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.command.startswith('parity-'):
        parity(args.config, submit=args.command == 'parity-submit')
    elif args.command == 'dispatch':
        dispatch(args.host, args.root, args.model, args.workers)
    elif args.command == 'relay':
        relay(args.config)
    elif args.command == 'prepare':
        prepare(args.config)
    else:
        relay_start(args.config)
