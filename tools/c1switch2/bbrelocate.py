"""One unused BB allocation on gpuR CPUs after MAIN's 11:30 priority window."""
import argparse
from datetime import datetime
import json
import time

from tools.c1switch.design import checked, sha
from tools.c1switch.storage import canonical, write

from . import gpuops, modelops, operations, relocation
from .design import REGISTRATION
from .gate import approval, cpu_authority
from .operations import EVIDENCE, ssh

PLAN_PATH = EVIDENCE/'BB_RELOCATION_PLAN.json'
BOUND_PATH = EVIDENCE/'BB_EXECUTION_BOUND.json'


def publish_plan():
    value = relocation.plan()
    if PLAN_PATH.exists():
        if json.loads(PLAN_PATH.read_text()) != value:
            raise ValueError('preserve placement plan')
    else:
        write(PLAN_PATH, value)
    print(json.dumps(dict(plan=value, plan_sha256=canonical(value))))


def prove_unused(config):
    if (EVIDENCE/'BB_FOLLOW/STOP.json').exists():
        raise ValueError('terminal BB science failure cannot be retried')
    proofs = {}
    for host, root in config['gpus'].items():
        proofs[host] = ssh(host, f'''import json,os
from pathlib import Path
root=Path({root!r});assert root.name=='c1switch2_gpu_bb_caeab9eaa303f93c'
assert not (root/'DISPATCH.json').exists() and not (root/'rpc/GPU_STARTED.json').exists(),'BB GPU already consumed'
for entry in os.scandir(root.parent):
 if entry.name.startswith('c1switch2_run_') and entry.is_dir(follow_symlinks=False):
  path=Path(entry.path,'MODEL_DISPATCH_ONCE.json')
  if path.exists():assert json.loads(path.read_text())['arm']!='bb','BB science already claimed'
print(json.dumps(dict(root=str(root),bb_gpu_s=0,bb_sim_s=0)))
''', timeout=30)
    return proofs


def prepare():
    if BOUND_PATH.exists():
        raise ValueError('BB already bound; preserve existing preparation')
    move = json.loads(checked(PLAN_PATH).read_text())
    probe = dict(relocation=move, cpu_host='gpuR', enabled_arms=['bb'], dispatch_phase='model')
    relocation.validate(probe, checked(REGISTRATION).read_text(encoding='utf-8'))
    config = json.loads((EVIDENCE/'BB_EXECUTION.json').read_text())
    proofs = prove_unused(config)
    inference = dict(status='FROZEN_READY', engine=config['engine'],
                     routes={gpu: 'rpc/bb/'+gpu for gpu in ('gpuA', 'gpuR')})
    staged = operations.stage('gpuR', model='bb', inference=inference, relocation=move)
    if operations.preflight('gpuR', staged['root'])['status'] != 'PASS':
        raise ValueError('relocated BB CPU gate failed; no GPU/physics started')
    result = dict(config, cpus=dict(gpuR=staged['root']), relocation=move,
                  unused_proofs=proofs, relocation_plan_sha256=canonical(move))
    write(BOUND_PATH, result)
    print(json.dumps(dict(status='CPU_PREPARED_NO_DISPATCH', config=str(BOUND_PATH),
                          root=staged['root'], new_gpu_s=0, new_sim_s=0)))


def resources(config):
    states = {}
    for host in config['gpus']:
        states[host] = ssh(host, '''import json,os,subprocess
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
print(json.dumps(dict(free_mib=free,load=os.getloadavg()[0])))
''', timeout=30)
    return states, (all(s['free_mib'] >= 8192 for s in states.values()) and states['gpuR']['load'] < 20)


def start():
    config = json.loads(checked(BOUND_PATH).read_text())
    if config['relocation'] != relocation.plan():
        raise ValueError('exact registered unused BB plan required')
    text = checked(REGISTRATION).read_text(encoding='utf-8')
    relocation.validate(dict(relocation=config['relocation'], cpu_host='gpuR',
                        enabled_arms=['bb'], dispatch_phase='model'), text)
    if not approval(text) or not cpu_authority(text):
        raise ValueError('current owner/MAIN authority required before GPU allocation')
    if time.time() < datetime.fromisoformat(config['relocation']['not_before']).timestamp():
        raise ValueError('priority WM-PHYS window; not before11:30')
    claim_path = EVIDENCE/'BB_RELOCATION_DISPATCH_ONCE.json'
    if claim_path.exists():
        raise ValueError('BB placement already claimed; never retry')
    if set(config['cpus']) != {'gpuR'}:
        raise ValueError('exact placement required')
    # Resource refusal is before any allocation/claim and may be observed again.
    states, ready = resources(config)
    if not ready:
        print(json.dumps(dict(status='WAIT_RESOURCES_NO_CLAIM', states=states)))
        return
    proofs = prove_unused(config)
    with claim_path.open('x') as stream:
        json.dump(dict(unix=time.time(), config_sha256=sha(BOUND_PATH), proofs=proofs, states=states), stream, indent=2)
    started = []
    try:
        for host, root in config['gpus'].items():
            gpuops.start(host, root)
            started.append((host, root))
        deadline = time.monotonic()+300
        while True:
            ready = []
            for host, root in config['gpus'].items():
                ready.append(ssh(host, f'''import json
from pathlib import Path
root=Path({root!r});assert not (root/'rpc/STOP').exists(),'BB service stopped'
print(json.dumps(dict(ready=(root/'rpc/READY.json').exists())))
''', timeout=30)['ready'])
            if all(ready):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('BB readiness exceeded five-minute allocation gate')
            time.sleep(2)
        modelops.parity(BOUND_PATH, submit=True)
        while modelops.parity(BOUND_PATH) is None:
            if time.monotonic() >= deadline:
                raise TimeoutError('BB parity exceeded five-minute allocation gate')
            time.sleep(5)
        modelops.dispatch('gpuR', config['cpus']['gpuR'], 'bb', 4)
        modelops.relay_start(BOUND_PATH)
        operations.archive_binding('gpuR', config['cpus']['gpuR'])
        write(EVIDENCE/'BB_RELOCATION_RUNNING.json', dict(unix=time.time(), cpus=config['cpus'],
              config_sha256=sha(BOUND_PATH), prior_sim_s=0, sim_s_cap=79800, gpu_s_cap=10800))
    except BaseException as exc:
        write(EVIDENCE/'BB_RELOCATION_STOP.json', dict(unix=time.time(), error=repr(exc), started=started))
        for host, root in started:
            ssh(host, f'''import json
from pathlib import Path
path=Path({root!r})/'rpc/COMPLETE'
if not path.exists():
 with path.open('x') as stream:json.dump(dict(reason='terminal unused-BB placement startup failure'),stream)
print(json.dumps(dict(released=True)))
''', timeout=30)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('plan', 'prepare', 'start'))
    command = parser.parse_args().command
    if command == 'plan':
        publish_plan()
    elif command == 'prepare':
        prepare()
    else:
        start()
