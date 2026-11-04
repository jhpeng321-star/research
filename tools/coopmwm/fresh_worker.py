"""Fresh-process qualification only; immutable binding, worker-owned evidence.

Unlike legacy admission receipts, a fresh binding starts pending. The first four
warm_service gates permit entry to plan_service, not READY. All eight exact gates
from this same process are required to publish FRESH_QUALIFICATION.json.
"""
import os
import sys
from .hotadd import GATES
from .protocol import sha
from .storage import read, write

MODE = 'fresh_startup_v1'
RECEIPT = 'FRESH_QUALIFICATION.json'
START = 'FRESH_STARTUP.json'


def pending(hashes):
    return dict(mode=MODE, pass_=False, status='pending', required=list(GATES),
                receipts={}, source_hashes=dict(hashes), receipt=RECEIPT,
                new_sumo_sim_s=0)


def environment(binding):
    """Check package metadata only; never initializes CUDA or runs inference."""
    import torch
    pin = binding['worker_env']
    actual = dict(python_version='.'.join(map(str, sys.version_info[:2])),
                  torch_version=str(torch.__version__), cuda_version=torch.version.cuda)
    if any(actual[k] != pin[k] for k in actual):
        raise ValueError(f'fresh worker environment mismatch: {actual} != {pin}')
    if os.path.realpath(sys.executable) != os.path.realpath(pin['python_executable']):
        raise ValueError('fresh worker interpreter mismatch')
    if any(os.environ.get(k) != v for k, v in pin['variables'].items()):
        raise ValueError('fresh worker environment variables mismatch')
    return dict(actual, host=pin['host'], python_executable=sys.executable,
                variables=dict(pin['variables']))


def begin(root, binding):
    if binding['qualification'] != pending(binding['hashes']):
        raise ValueError('fresh qualification must start pending, never inherited')
    if 'adopted_startup_gates' in binding or any((root/n).exists() for n in GATES+(RECEIPT, START)):
        raise ValueError('fresh worker refuses existing startup evidence')
    for name, h in binding['hashes'].items():
        if sha(root/name) != h:
            raise ValueError('immutable GPU source mismatch '+name)
    write(root/START, dict(pid=os.getpid(), binding_sha256=sha(root/'GPU_BINDING.json'),
                          environment=environment(binding), new_sumo_sim_s=0))


def qualification(root, binding, names=GATES):
    """Aggregate ONLY receipts produced since begin() by this very process."""
    start = read(root/START)
    if start['pid'] != os.getpid() or start['binding_sha256'] != sha(root/'GPU_BINDING.json'):
        raise ValueError('fresh startup process/binding changed')
    receipts = {}
    for name in names:
        gate = read(root/name)
        if gate.get('pid') != os.getpid() or gate.get('pass_') is not True or \
                gate.get('max_cost_error') != 0.:
            raise ValueError('missing/nonexact fresh-worker gate '+name)
        receipts[name] = gate
    complete = set(receipts) == set(GATES)
    return dict(pending(binding['hashes']), pass_=complete,
                status='passed' if complete else 'running', receipts=receipts,
                pid=os.getpid(), binding_sha256=start['binding_sha256'],
                environment=start['environment'])


def finish(root, binding):
    result = qualification(root, binding)
    write(root/RECEIPT, result)
    return result
