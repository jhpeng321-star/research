"""Admit the already-qualified incident replacement; never reuse the failed slot."""
import argparse
import json

from . import ops
from .storage import read, write
from .protocol import REGISTRY, checked, sha
from .warm_deploy import ROOT

SLOT = 'gpuA@incident1'
ROUTES = '/home/ubuntu/coopmwm20261102_incident_routes1'
SOURCE = '/home/ubuntu/coopmwm20261102_verifycpu9'


def register():
    from .incident_ops import gpu_status, GPU, service_status
    from .hotadd import qualified
    target = ops.EVIDENCE/'INCIDENT_REPLACEMENT_ADMISSION_PLAN.json'
    assert not target.exists(), 'adopt existing replacement admission plan'
    status = gpu_status()
    owner = read(ops.EVIDENCE/'INCIDENT0944_GPU_DISPATCH.json')
    live_owner = status['receipts']['SPOOL_OWNER.json']
    assert all(live_owner[k] == owner[k] for k in ('pid', 'start_ticks', 'boot_id'))
    assert qualified(status, live_owner)
    service = service_status()
    assert SLOT not in service['slots'] and 'gpuA@0' in service['disabled']
    stage = read(ops.EVIDENCE/'INCIDENT0944_GPU_STAGE.json')
    value = dict(slot=SLOT, root=GPU, expected_owner=live_owner,
                 search_sha256=stage['binding']['hashes']['pytools/tools/coopmwm/mppi_search.py'],
                 deadline_unix=stage['binding']['deadline_unix'], status=status,
                 original_service=service, new_sumo_sim_s=0,
                 invariant='adopt qualified live759743; never launch a new worker or reuse disabled gpuA@0; preserve all previous assignments/claims and gates; unchanged search/horizons/CRN/deadline; no new SUMO reservation')
    write(target, value)
    h = sha(target)
    line = '\nS-COOPMWM INCIDENT REPLACEMENT ADMISSION REGISTERED planSHA'+h+': '+value['invariant']+'; new slot '+SLOT+' uses fresh cpuQ-only restricted RPC routes, all8 retained exact-zero gates plus PID/boot/start identity checked before admission. Existing3healthy slots remain. Zero new SUMO allocation/physics retry.\n'
    with checked(REGISTRY).open('a', encoding='utf-8') as f:
        f.write(line)
    write(ops.EVIDENCE/'INCIDENT_REPLACEMENT_ADMISSION_REGISTRATION.json', dict(plan_sha256=h, line=line))
    print(json.dumps(dict(registered=True, slot=SLOT, plan_sha256=h, new_sumo_sim_s=0)))


def authorize():
    from . import warm_remote_ops as w
    from .incident_ops import GPU
    from .exact_ops import GPU_R
    assert (ops.EVIDENCE/'INCIDENT_REPLACEMENT_ADMISSION_REGISTRATION.json').exists()
    assert not (ops.EVIDENCE/'INCIDENT_REPLACEMENT_RPC_AUTHORIZATION.json').exists(), 'adopt existing routes'
    w.ROUTES = ROUTES
    w.authorize('INCIDENT_REPLACEMENT_RPC_AUTHORIZATION.json', roots={'gpuA': GPU, 'gpuR': GPU_R})


def authorize_v2():
    """Expose all eight existing gates through the already-qualified RPC wrapper."""
    from . import warm_remote_ops as w
    from .incident_ops import GPU
    from .exact_ops import GPU_R
    assert (ops.EVIDENCE/'INCIDENT_REPLACEMENT_ADMISSION_REGISTRATION.json').exists()
    target = ops.EVIDENCE/'INCIDENT_REPLACEMENT_RPC_AUTHORIZATION_V2.json'
    assert not target.exists(), 'adopt existing eight-gate routes'
    cpu = read(ops.EVIDENCE/'SECOND_CPU_QUALIFICATION.json')
    assert cpu['pass_'] and cpu['skips'] == cpu['sim_s'] == 0
    wrapper = ops.checked(ops.LOCAL/'tools/coopmwm/hotadd_rpc.py').read_text(encoding='utf-8')
    import hashlib
    assert hashlib.sha256(wrapper.encode()).hexdigest() == cpu['staged']['hashes']['pytools/tools/coopmwm/hotadd_rpc.py']
    line = '\nS-COOPMWM MAIN1424 REPLACEMENT RPC ENGINEERING AMENDMENT: all8 startup gates already exact-zero on owned gpuA759743; legacy restricted warm_rpc exposes only6. Install original CPU26-qualified byte-identical hotadd_rpc wrapper, fresh restricted routes2 key on cpuQ only; expose both committed gate receipts without rerunning inference or changing numerical source, search, deadlines, existing claims or physics. Original routes/failed attempts retained. Zero new SUMO allocation.\n'
    with checked(REGISTRY).open('a', encoding='utf-8') as f:
        f.write(line)
    write(ops.EVIDENCE/'INCIDENT_REPLACEMENT_RPC_V2_REGISTRATION.json', dict(line=line, cpu=cpu, new_sumo_sim_s=0))
    for host, root in (('gpuA', GPU), ('gpuR', GPU_R)):
        ops.stage(host, root, {'tools/coopmwm/hotadd_rpc.py': wrapper})
    w.ROUTES = '/home/ubuntu/coopmwm20261102_incident_routes2'
    w.authorize(target.name, roots={'gpuA': GPU, 'gpuR': GPU_R}, rpc_module='tools.coopmwm.hotadd_rpc')


def admit():
    plan = read(ops.EVIDENCE/'INCIDENT_REPLACEMENT_ADMISSION_PLAN.json')
    reg = read(ops.EVIDENCE/'INCIDENT_REPLACEMENT_ADMISSION_REGISTRATION.json')
    endpoints = read(ops.EVIDENCE/'INCIDENT_REPLACEMENT_RPC_AUTHORIZATION_V2.json')['endpoints']
    endpoint = dict(next(e for e in endpoints if e['slot'] == 'gpuA@0'), slot=SLOT)
    helper_sha = read(ops.EVIDENCE/'SECOND_CPU_QUALIFICATION.json')['staged']['hashes']['pytools/tools/coopmwm/hotadd.py']
    value = json.loads(ops.ssh.remote('cpuQ', f'''import pathlib,json,sys,time
r=pathlib.Path({ROOT!r});sys.path.insert(0,{SOURCE!r}+'/pytools')
import tools.coopmwm,hashlib
helper=pathlib.Path('/home/ubuntu/coopmwm20261102_hotadd1/pytools/tools/coopmwm/hotadd.py')
assert hashlib.sha256(helper.read_bytes()).hexdigest()=={helper_sha!r}
tools.coopmwm.__path__.append(str(helper.parent))
from tools.coopmwm.warm_broker import Remote,worker_endpoints
from tools.coopmwm.hotadd import qualified,GATES
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.warm_health import assert_running
plan=json.loads({json.dumps(plan)!r});reg=json.loads({json.dumps(reg)!r});endpoint=json.loads({json.dumps(endpoint)!r})
assert_running(r,'B')
original=json.loads((r/'WARM_REMOTE_BINDING.json').read_text())
assert plan['search_sha256']==original['search_sha256'] and plan['deadline_unix']==original['deadline_unix']
p=r/'WARM_REMOTE_ENDPOINT_ADMISSION_gpuA_INCIDENT1.json'
if p.exists():
 v=json.loads(p.read_text());assert v['amendment_sha256']==reg['plan_sha256'] and v['endpoints']==[endpoint]
else:
 assert endpoint['slot'] not in worker_endpoints(r,original,{{}})
 assert 'gpuA@0' in json.loads((r/'INCIDENT_RETIRED_JOBS.json').read_text())['disabled_slots']
 remote=Remote(r,endpoint);status=remote.raw(dict(action='status'))
 assert qualified(status,plan['expected_owner']);remote.verify()
 assert time.time()<original['deadline_unix']
 with (r/'AUTHORITY.md').open('a') as f:f.write(reg['line'])
 v=dict(amendment_sha256=reg['plan_sha256'],search_sha256=plan['search_sha256'],endpoints=[endpoint],
        gates={{n:status['receipts'][n] for n in GATES}},owner=plan['expected_owner'],unix=time.time(),new_sumo_sim_s=0)
 immutable(p,v)
print(json.dumps(dict(admitted=True,slot=endpoint['slot'],owner=v['owner'],unix=v['unix'],
                    admission_sha256=__import__('hashlib').sha256(p.read_bytes()).hexdigest(),new_sumo_sim_s=0)))''', timeout=60))
    target = ops.EVIDENCE/'INCIDENT_REPLACEMENT_ADMITTED.json'
    if target.exists():
        assert read(target) == value
    else:
        write(target, value)
    print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('register', 'authorize', 'authorize_v2', 'admit'))
    globals()[parser.parse_args().mode]()
