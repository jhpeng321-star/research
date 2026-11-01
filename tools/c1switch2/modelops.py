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


def parity(config, submit=False, receipt_path=None):
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
    path = receipt_path or EVIDENCE/('GPU_PARITY_'+config['engine']['model']+'.json')
    if path.exists():
        if json.loads(path.read_text()) != out:
            raise ValueError('preserve previous parity receipt')
    else:
        write(path, out)
    print(json.dumps(dict(status='PASS', path=str(path), sha256=sha(path))))
    return out


def dispatch(host, root, model, workers, parity_path=None):
    from .gpuhold import require_available
    require_available()
    from .audit import registration_audit
    text = checked(REGISTRATION).read_text(encoding='utf-8')
    audit = registration_audit(REGISTRATION)
    if not approval(text) or not cpu_authority(text) or audit['hits']:
        raise ValueError('current model authority/registry absent')
    parity_path = parity_path or EVIDENCE/('GPU_PARITY_'+model+'.json')
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
    result = ssh(host, f'''import hashlib,json,os,shutil,subprocess,sys,sysconfig,time
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
assert shutil.disk_usage(root).free>=2*1024**3,'yield disk pressure before model claim'
for entry in os.scandir(root.parent):
 if entry.name.startswith('c1switch2_run_') and entry.is_dir(follow_symlinks=False):
  claim=Path(entry.path,'MODEL_DISPATCH_ONCE.json')
  if claim.exists():assert json.loads(claim.read_text())['arm']!={model!r},'model panel already claimed'
sys.path.insert(0,str(root/'pytools'));os.chdir(root/'pytools')
from tools.c1switch.storage import canonical
from tools.c1switch2.gate import REQUIRED,CPU_REQUIRED,admission
from tools.c1switch2.relocation import execution_budget
needed=CPU_REQUIRED+('inference_'+{model!r},'gpuA_parity','gpuR_parity')
evidence={{k:dict(status='PASS' if k in needed else 'PENDING',binding_sha256=canonical(binding),source='CPU/frozen canary/preallocation exposure/currentregistry/real GPU parity') for k in REQUIRED}}
evidence['gpu_parity_sha256']={sha(parity_path)!r};evidence['registry_audit']={audit!r};evidence['mechanics_sha256']={archive['canary_sha256']!r}
assert admission((root/'REGISTRATION.md').read_text(),binding,evidence)['dispatch_authorized']
for route in binding['inference_specs'][{model!r}]['routes'].values():(root/route).mkdir(parents=True)
with (root/'GATES.json').open('x') as stream:json.dump(evidence,stream,indent=2)
claim=dict(arm={model!r},host={host!r},root=str(root),planned_sim_s=execution_budget(binding)[{model!r}],unix=time.time())
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


def collect(host, root, model):
    result = ssh(host, f'''import json,time
from pathlib import Path
root=Path({root!r});assert root.name.startswith('c1switch2_run_')
dispatch=json.loads((root/'DISPATCH_MODEL.json').read_text());proc=Path('/proc',str(dispatch['pid']))
ledger=root/'accounting.jsonl';rows=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
out=dict(host={host!r},root=str(root),arm={model!r},unix=time.time(),pid=dispatch['pid'],pid_state=proc.joinpath('stat').read_text().split()[2] if proc.exists() else 'EXITED',charged_sim_s=sum(r['sim_s'] for r in rows),closed_actual_sim_s=0,episodes=0,decisions=0,requests=0,results=0)
out['complete']=(root/'supervisor/COMPLETE.json').exists();out['stop']=json.loads((root/'STOP').read_text()) if (root/'STOP').exists() else None
for job in (root/'jobs').iterdir():
 close=job/'CLOSE.json'
 if close.exists():out['closed_actual_sim_s']+=json.loads(close.read_text())['actual_sim_s']
 out['episodes']+=(job/'PHYSICS.json').exists();out['decisions']+=len(list(job.glob('pick_j*.json')))
for gpu in ('gpuA','gpuR'):
 route=root/'rpc'/{model!r}/gpu;out['requests']+=len(list(route.glob('*.request.json')));out['results']+=len(list(route.glob('*.result.json')))
if out['stop']:
 out['log']=(root/'MODEL.log').read_text();failed=root/'logs'/(out['stop'].get('id','')+'.log')
 if failed.is_file():out['failed_log']=failed.read_text()
print(json.dumps(out))
''', timeout=30)
    path = EVIDENCE/('MODEL_PROGRESS_'+model+'_'+host+'_'+str(int(result['unix']))+'.json')
    write(path, result)
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


def relay_cycle(config):
    """Isolate host transport failures so one offline CPU cannot hide another's STOP."""
    statuses, gpu_status, errors = {}, {}, {}
    for cpu, root in config['cpus'].items():
        if 'relocation' in config:
            from .relocation import plan
            if config['relocation'] != plan() or set(config['cpus']) != {'gpuR'}:
                raise ValueError('unexpected relay placement')
            count = len(worlds())
        else:
            count = sum(assigned(w['seed']) == cpu for w in worlds())
        try:
            statuses[cpu] = ssh(cpu, f'''import json
from pathlib import Path
root=Path({root!r})
print(json.dumps(dict(stop=json.loads((root/'STOP').read_text()) if (root/'STOP').exists() else None,episodes=len(list((root/'jobs').glob('*__episode/PHYSICS.json'))),expected={count},complete=(root/'supervisor/COMPLETE.json').exists())))
''', timeout=30)
        except Exception as exc:
            errors[cpu] = repr(exc)
            statuses[cpu] = dict(unknown=True, stop=None, episodes=None, expected=count, complete=False)
    done = all(not s.get('unknown') and s['episodes'] == s['expected'] for s in statuses.values())
    failed = any(s['stop'] for s in statuses.values())
    for gpu, root in config['gpus'].items():
        try:
            gpu_status[gpu] = ssh(gpu, f'''import json
from pathlib import Path
root=Path({root!r});path=root/'rpc/COMPLETE'
if {done or failed!r} and not path.exists():
 with path.open('x') as stream:json.dump(dict(reason='all model episodes ended or CPU STOP'),stream)
print(json.dumps(dict(stop=json.loads((root/'rpc/STOP').read_text()) if (root/'rpc/STOP').exists() else None,finished=json.loads((root/'rpc/GPU_FINISHED.json').read_text()) if (root/'rpc/GPU_FINISHED.json').exists() else None)))
''', timeout=30)
        except Exception as exc:
            errors[gpu] = repr(exc)
    if any(s['stop'] for s in gpu_status.values()):
        for cpu, root in config['cpus'].items():
            try:
                ssh(cpu, f'''import json
from pathlib import Path
path=Path({root!r})/'STOP'
if not path.exists():
 with path.open('x') as stream:json.dump(dict(reason='owned model GPU service STOP',gpu_status={gpu_status!r}),stream)
print(json.dumps(dict(stopped=True)))
''', timeout=30)
            except Exception as exc:
                errors[cpu+'/stop'] = repr(exc)
        failed = True
    transferred = 0
    if not done and not failed:
        for cpu, root in config['cpus'].items():
            if statuses[cpu].get('unknown'):
                continue
            for gpu, gpu_root in config['gpus'].items():
                if gpu not in gpu_status:
                    continue
                try:
                    transferred += transfer(cpu, root, gpu, gpu_root, config['engine']['model'])
                except Exception as exc:
                    errors[cpu+'/'+gpu] = repr(exc)
    return dict(unix=time.time(), cpus=statuses, gpus=gpu_status, transferred=transferred,
                transport_errors=errors, episodes_done=done, failed=failed)


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
                state = relay_cycle(config)
                write(local/'STATUS.json', state)
                if state['failed'] or (state['episodes_done'] and
                        set(state['gpus']) == set(config['gpus']) and
                        all(s['finished'] for s in state['gpus'].values())):
                    write(local/'COMPLETE.json', state)
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
    return path


def wait_for_release(after, local):
    """Transport failure while observing releases spends nothing and may be polled again."""
    while True:
        states = {}
        try:
            for host, root in after['gpus'].items():
                states[host] = ssh(host, f'''import json,subprocess
from pathlib import Path
root=Path({root!r})
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
print(json.dumps(dict(wm_finished=(root/'rpc/GPU_FINISHED.json').exists(),free_mib=free)))
''', timeout=30)
        except Exception as exc:
            write(local/'WAIT_TRANSPORT_ERROR.json', dict(unix=time.time(), error=repr(exc)))
            time.sleep(60)
            continue
        write(local/'WAIT.json', dict(unix=time.time(), states=states))
        if all(s['wm_finished'] and s['free_mib'] >= 8*1024 for s in states.values()):
            return
        time.sleep(60)


def follow(config_path, after_config, *, resume_wait=False):
    """One unused BB arm after WM releases its services; no scientific retry path."""
    from .gpuops import start
    config_path, after_config = checked(config_path), checked(after_config)
    config, after = (json.loads(p.read_text()) for p in (config_path, after_config))
    if config['engine']['model'] != 'bb' or after['engine']['model'] != 'wm':
        raise ValueError('registered WM-then-unused-BB sequence required')
    local = EVIDENCE/'BB_FOLLOW'
    with exclusive(local):
        if (local/'START.json').exists() and not resume_wait:
            raise ValueError('BB follower already claimed')
        import os
        if resume_wait:
            claim = json.loads((local/'WAIT_RESUME_ONCE.json').read_text())
            if claim['config_sha256'] != sha(config_path) or claim['after_sha256'] != sha(after_config):
                raise ValueError('wait correction configuration mismatch')
            with (local/'WAIT_RESUME_STARTED.json').open('x') as stream:
                json.dump(dict(pid=os.getpid(), unix=time.time(), claim_sha256=sha(local/'WAIT_RESUME_ONCE.json')), stream)
        else:
            write(local/'START.json', dict(pid=os.getpid(), config_sha256=sha(config_path),
                  after_sha256=sha(after_config), unix=time.time()))
        wait_for_release(after, local)
        try:
            for host, root in config['gpus'].items():
                start(host, root)
            # READY is a construction gate; no main physics while it is absent.
            while True:
                ready = []
                for host, root in config['gpus'].items():
                    ready.append(ssh(host, f'''import json
from pathlib import Path
root=Path({root!r});assert not (root/'rpc/STOP').exists(),'BB service STOP'
print(json.dumps(dict(ready=(root/'rpc/READY.json').exists())))
''', timeout=30)['ready'])
                if all(ready):
                    break
                time.sleep(2)
            parity(config_path, submit=True)
            while parity(config_path) is None:
                time.sleep(5)
            bound = prepare(config_path)
            cpus = json.loads(bound.read_text())['cpus']
            dispatch('cpuQ', cpus['cpuQ'], 'bb', 4)
            dispatch('gpuR', cpus['gpuR'], 'bb', 1)
            relay(bound)
            write(local/'COMPLETE.json', dict(unix=time.time()))
        except BaseException as exc:
            write(local/'STOP.json', dict(unix=time.time(), error=repr(exc)))
            for host, root in config['gpus'].items():
                ssh(host, f'''import json
from pathlib import Path
root=Path({root!r});path=root/'rpc/COMPLETE'
if not path.exists():
 with path.open('x') as stream:json.dump(dict(reason='BB follower terminal STOP'),stream)
print(json.dumps(dict(owned_services_released=True)))
''', timeout=30)
            raise


def follow_start(config, after_config):
    config, after_config = checked(config).resolve(), checked(after_config).resolve()
    path = EVIDENCE/'BB_FOLLOW_DISPATCH.json'
    if path.exists():
        raise ValueError('BB follower already dispatched')
    with (EVIDENCE/'BB_FOLLOW.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-B', '-u', '-m', 'tools.c1switch2.modelops',
                    'follow', '--config', str(config), '--after-config', str(after_config)],
                    stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS)
    out = dict(pid=process.pid, config=str(config), after_config=str(after_config), unix=time.time())
    write(path, out)
    print(json.dumps(out))


def resume_follow_wait(config_path, after_config):
    """One explicit zero-spend observer correction; never retries claimed BB science."""
    from .watch import local_process
    config_path, after_config = checked(config_path).resolve(), checked(after_config).resolve()
    config = json.loads(config_path.read_text())
    local = EVIDENCE/'BB_FOLLOW'
    if (local/'WAIT_RESUME_ONCE.json').exists():
        raise ValueError('wait correction already claimed; never duplicate')
    original = json.loads((local/'START.json').read_text())
    if (config['engine']['model'] != 'bb' or original['pid'] != 43404 or
            original['config_sha256'] != sha(config_path) or original['after_sha256'] != sha(after_config)):
        raise ValueError('exact original BB waiting allocation required')
    if local_process(original['pid'])['state'] != 'EXITED':
        raise ValueError('original owned follower has not exited')
    log = EVIDENCE/'BB_FOLLOW.log'
    if ('RuntimeError: Connection closed by ' not in log.read_text() or
            (local/'STOP.json').exists() or (local/'COMPLETE.json').exists() or
            (local/'DISPATCH.lock').exists() or (EVIDENCE/'BB_EXECUTION_BOUND.json').exists()):
        raise ValueError('only diagnosed pre-dispatch transport exit may resume')
    proofs = {}
    for host, root in config['gpus'].items():
        proofs[host] = ssh(host, f'''import json,os
from pathlib import Path
root=Path({root!r});assert root.name=='c1switch2_gpu_bb_caeab9eaa303f93c'
assert not (root/'DISPATCH.json').exists() and not (root/'rpc/GPU_STARTED.json').exists()
for entry in os.scandir(root.parent):
 if entry.name.startswith('c1switch2_run_') and entry.is_dir(follow_symlinks=False):
  p=Path(entry.path,'MODEL_DISPATCH_ONCE.json')
  if p.exists():assert json.loads(p.read_text())['arm']!='bb','BB science already claimed'
print(json.dumps(dict(root=str(root),bb_gpu_claimed=False,bb_physics_claimed=False,new_gpu_s=0,new_sim_s=0)))
''', timeout=30)
    claim = dict(unix=time.time(), config_sha256=sha(config_path), after_sha256=sha(after_config),
                 predecessor_sha256=sha(local/'START.json'), error_sha256=sha(log), proofs=proofs)
    with (local/'WAIT_RESUME_ONCE.json').open('x') as stream:
        json.dump(claim, stream, indent=2)
    with (EVIDENCE/'BB_FOLLOW_WAIT_RESUME.log').open('x') as stream:
        process = subprocess.Popen([sys.executable, '-B', '-u', '-m', 'tools.c1switch2.modelops',
            'follow-resume-wait', '--config', str(config_path), '--after-config', str(after_config)],
            stdout=stream, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS)
    out = dict(pid=process.pid, unix=time.time(), claim_sha256=sha(local/'WAIT_RESUME_ONCE.json'))
    write(EVIDENCE/'BB_FOLLOW_WAIT_RESUME_DISPATCH.json', out)
    print(json.dumps(out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('parity-submit', 'parity-collect', 'dispatch', 'collect', 'relay', 'relay-start',
                                           'prepare', 'follow', 'follow-start', 'resume-wait-start',
                                           'follow-resume-wait'))
    parser.add_argument('--config')
    parser.add_argument('--after-config')
    parser.add_argument('--host', choices=('cpuQ', 'gpuR'))
    parser.add_argument('--root')
    parser.add_argument('--model', choices=('wm', 'bb'))
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.command.startswith('parity-'):
        parity(args.config, submit=args.command == 'parity-submit')
    elif args.command == 'dispatch':
        dispatch(args.host, args.root, args.model, args.workers)
    elif args.command == 'collect':
        collect(args.host, args.root, args.model)
    elif args.command == 'relay':
        relay(args.config)
    elif args.command == 'prepare':
        prepare(args.config)
    elif args.command == 'follow':
        follow(args.config, args.after_config)
    elif args.command == 'follow-start':
        follow_start(args.config, args.after_config)
    elif args.command == 'resume-wait-start':
        resume_follow_wait(args.config, args.after_config)
    elif args.command == 'follow-resume-wait':
        follow(args.config, args.after_config, resume_wait=True)
    else:
        relay_start(args.config)
