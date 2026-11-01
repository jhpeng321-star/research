"""One registered MAIN12:14 BB startup continuation; original physics stays unclaimed."""
import argparse
import base64
import io
import json
from pathlib import Path
import tarfile
import time

from tools.c1switch.design import checked, sha
from tools.c1switch.storage import canonical, write
from . import bbrelocate, gpuops, modelops
from .audit import registration_audit
from .design import REGISTRATION
from .gate import approval, cpu_authority
from .gpuallocation import AUTHORITY, PREDECESSOR, PRIOR, allocation_cap
from .operations import EVIDENCE, ssh

PLAN = EVIDENCE/'BB_CONTINUATION_PLAN.json'
BOUND = EVIDENCE/'BB_CONTINUATION_BOUND.json'
PARITY = EVIDENCE/'GPU_PARITY_bb_continuation.json'
CLAIM = EVIDENCE/'BB_CONTINUATION_DISPATCH_ONCE.json'


def authority():
    text = checked(REGISTRATION).read_text(encoding='utf-8')
    lines = [line for line in text.splitlines() if AUTHORITY in line]
    if not lines or not approval(text) or not cpu_authority(text) or registration_audit(REGISTRATION)['hits']:
        raise ValueError('explicit current MAIN12:14 exception and clean registry required')
    return text, lines[-1]


def proof(config):
    out = {}
    for host in ('gpuA', 'gpuR'):
        out[host] = ssh(host, f'''import hashlib,json,os
from pathlib import Path
old=Path({PREDECESSOR!r});finished=old/'rpc/GPU_FINISHED.json';row=json.loads(finished.read_text())
assert row['elapsed_s']=={PRIOR[host]!r} and not Path('/proc',str(row['pid'])).exists()
assert json.loads((old/'DISPATCH.json').read_text())['pid']==row['pid']
for entry in os.scandir(old.parent):
 if entry.is_dir(follow_symlinks=False) and entry.name.startswith('c1switch2_run_'):
  claim=Path(entry.path,'MODEL_DISPATCH_ONCE.json')
  if claim.exists():assert json.loads(claim.read_text())['arm']!='bb','BB physics already claimed'
 if entry.is_dir(follow_symlinks=False) and entry.name.startswith('c1switch2_gpu_bb_') and entry.path!=str(old):
  assert not Path(entry.path,'DISPATCH.json').exists(),'BB continuation already consumed'
print(json.dumps(dict(finished=row,finished_sha256=hashlib.sha256(finished.read_bytes()).hexdigest(),sources=json.loads((old/'STAGED.json').read_text()))))
''', timeout=30)
    root = config['cpus']['gpuR']
    out['cpu'] = ssh('gpuR', f'''import json
from pathlib import Path
root=Path({root!r})
assert not (root/'MODEL_DISPATCH_ONCE.json').exists() and not (root/'GATES.json').exists()
assert not (root/'accounting.jsonl').exists() and not list((root/'jobs').iterdir())
assert json.loads((root/'PREFLIGHT.json').read_text())['status']=='PASS'
print(json.dumps(dict(root=str(root),new_sim_s=0)))
''', timeout=30)
    return out


def plan():
    if PLAN.exists():
        raise ValueError('preserve existing continuation plan')
    _, line = authority()
    config = json.loads((EVIDENCE/'BB_EXECUTION_BOUND.json').read_text())
    proofs = proof(config)
    names = set(proofs['gpuA']['sources']) | set(proofs['gpuR']['sources'])
    names.add('tools/c1switch2/gpuallocation.py')
    sources = {name: sha(checked(name)) for name in sorted(names)}
    value = dict(schema='S-C1SWITCH2/BB-MAIN1214-v1', authority=line,
        predecessor_root=PREDECESSOR, prior_gpu_s=PRIOR, proofs=proofs,
        sources=sources, config_sha256=sha(EVIDENCE/'BB_EXECUTION_BOUND.json'),
        sim_s_cap=79800, gpu_s_cap=10800, remaining_gpu_s={h: 5400-v for h, v in PRIOR.items()},
        cpu_root=config['cpus']['gpuR'], engine=config['engine'])
    write(PLAN, value)
    print(json.dumps(dict(path=str(PLAN),plan_sha256=canonical(value),remaining_gpu_s=value['remaining_gpu_s'])))


def prepare():
    if BOUND.exists():
        raise ValueError('preserve continuation preparation')
    text, _ = authority()
    value = json.loads(PLAN.read_text()); digest = canonical(value)
    if not any('S-C1SWITCH2 BB CONTINUATION REGISTERED' in line and digest in line for line in text.splitlines()):
        raise ValueError('prospective exact-plan registration required')
    config = json.loads((EVIDENCE/'BB_EXECUTION_BOUND.json').read_text())
    if sha(EVIDENCE/'BB_EXECUTION_BOUND.json') != value['config_sha256']:
        raise ValueError('original binding changed')
    proofs = proof(config)
    if proofs != value['proofs'] or any(sha(checked(n)) != v for n, v in value['sources'].items()):
        raise ValueError('immutable source/predecessor proof changed')
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for name in value['sources']:
            tar.add(checked(name), arcname=name)
    encoded = base64.b64encode(buf.getvalue()).decode()
    root = '/root/c1switch2_gpu_bb_cont_'+digest[:16]
    allocations = {}
    for host in ('gpuA', 'gpuR'):
        allocations[host] = dict(schema=value['schema'], host=host, authority=value['authority'],
            plan_sha256=digest, predecessor_root=PREDECESSOR, prior_gpu_s=PRIOR[host],
            remaining_gpu_s=value['remaining_gpu_s'][host], finished=proofs[host]['finished'],
            finished_sha256=proofs[host]['finished_sha256'])
        allocation_cap(config['engine'], allocations[host])
        staged = ssh(host, f'''import base64,hashlib,io,json,shutil,tarfile
from pathlib import Path
root=Path({root!r});old=Path({PREDECESSOR!r})
assert root.parent.is_dir() and not root.exists(),'preserve existing continuation stage'
root.mkdir();(root/'pytools').mkdir();(root/'rpc').mkdir()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({encoded!r})),mode='r:gz') as tar:tar.extractall(root/'pytools',filter='data')
assert all(hashlib.sha256((root/'pytools'/n).read_bytes()).hexdigest()==v for n,v in {value['sources']!r}.items())
for name in ('bb.pt','physics.json'):shutil.copyfile(old/name,root/name)
(root/'STAGED.json').write_text(json.dumps({value['sources']!r},indent=2))
(root/'ENGINE.json').write_text(json.dumps({config['engine']!r},indent=2))
(root/'GPU_ALLOCATION.json').write_text(json.dumps({allocations[host]!r},indent=2))
(root/'CONTINUATION_REGISTRATION.md').write_text({text!r})
print(json.dumps(dict(host={host!r},model='bb',root=str(root),engine={config['engine']!r},new_gpu_s=0,new_sim_s=0)))
''', timeout=90)
        write(EVIDENCE/('GPU_STAGE_'+host+'_'+Path(root).name+'.json'), staged)
        if gpuops.cpu_probe(host, root, 'bb')['status'] != 'PASS':
            raise ValueError('continuation real frozen CPU probe failed; no allocation')
    result = dict(config, gpus={h: root for h in ('gpuA', 'gpuR')}, allocations=allocations,
                  continuation_plan_sha256=digest)
    write(BOUND, result)
    print(json.dumps(dict(status='CPU_PREPARED_NO_DISPATCH',config=str(BOUND),root=root)))


def resources(config):
    states, ready = bbrelocate.resources(config)
    disk = ssh('gpuR', '''import json,shutil
print(json.dumps(dict(free_bytes=shutil.disk_usage('/root').free)))''', timeout=30)
    states['disk'] = disk
    return states, ready and disk['free_bytes'] >= 2*1024**3


def start():
    from .gpuhold import require_available
    require_available()
    text, _ = authority()
    config = json.loads(BOUND.read_text())
    if config['continuation_plan_sha256'] != canonical(json.loads(PLAN.read_text())):
        raise ValueError('plan changed')
    if config['continuation_plan_sha256'] not in text:
        raise ValueError('exact registration absent')
    if CLAIM.exists():
        raise ValueError('continuation is one-use; never retry')
    states, ready = resources(config)
    if not ready:
        print(json.dumps(dict(status='WAIT_RESOURCES_NO_CLAIM',states=states)))
        return False
    proofs = proof(config)
    with CLAIM.open('x') as stream:
        json.dump(dict(unix=time.time(),config_sha256=sha(BOUND),states=states,proofs=proofs),stream,indent=2)
    started = []
    try:
        for host, root in config['gpus'].items():
            gpuops.start(host,root,config['allocations'][host]);started.append((host,root))
        deadline = time.monotonic()+300
        while True:
            ready = [ssh(host, f'''import json
from pathlib import Path
root=Path({root!r});assert not (root/'rpc/STOP').exists()
print(json.dumps(dict(ready=(root/'rpc/READY.json').exists())))''', timeout=30)['ready']
                for host,root in config['gpus'].items()]
            if all(ready):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('continuation GPU startup deadline')
            time.sleep(2)
        modelops.parity(BOUND,submit=True,receipt_path=PARITY)
        while modelops.parity(BOUND,receipt_path=PARITY) is None:
            if time.monotonic() >= deadline:
                raise TimeoutError('continuation GPU parity deadline')
            time.sleep(5)
        # Three BB workers plus one fleet worker fit the four-slot task gpuR cap.
        modelops.dispatch('gpuR',config['cpus']['gpuR'],'bb',3,parity_path=PARITY)
        modelops.relay_start(BOUND)
        write(EVIDENCE/'BB_CONTINUATION_RUNNING.json',dict(unix=time.time(),config_sha256=sha(BOUND)))
    except BaseException as exc:
        write(EVIDENCE/'BB_CONTINUATION_STOP.json',dict(unix=time.time(),error=repr(exc),started=started))
        for host,root in started:
            ssh(host,f'''import json
from pathlib import Path
path=Path({root!r})/'rpc/COMPLETE'
if not path.exists():
 with path.open('x') as stream:json.dump(dict(reason='terminal MAIN1214 continuation startup'),stream)
print(json.dumps(dict(released=True)))''',timeout=30)
        raise
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('plan','prepare','start'))
    command = parser.parse_args().command
    {'plan':plan,'prepare':prepare,'start':start}[command]()
