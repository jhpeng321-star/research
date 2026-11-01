"""Prospective MAIN07:07 placement of the wholly unused BB arm; no WM recovery."""
from tools.c1switch.storage import canonical

from .design import PROTOCOL, assigned, budget, local_budget, worlds


def plan():
    return dict(protocol=PROTOCOL, arm='bb', host='gpuR',
                seeds=[w['seed'] for w in worlds()], prior_sim_s=0,
                sim_s=budget()['bb'], gpu_s=10800,
                not_before='2026-11-01T11:30:00+08:00')


def validate(binding, registration=None):
    move = binding.get('relocation')
    if move is None:
        return False
    if (move != plan() or binding.get('enabled_arms') != ['bb'] or
            binding.get('cpu_host') != 'gpuR' or binding.get('dispatch_phase') != 'model'):
        raise ValueError('only the exact unused BB placement is registered')
    if registration is not None:
        if ('07:07 MAIN DECISION' not in registration or
                not any('S-C1SWITCH2 BB RELOCATION REGISTERED' in line and canonical(move) in line
                        for line in registration.splitlines())):
            raise ValueError('MAIN authority and exact prospective placement registration required')
    return True


def execution_host(binding, seed):
    return 'gpuR' if validate(binding) else assigned(seed)


def execution_budget(binding):
    if validate(binding):
        return {arm: budget()[arm] if arm == 'bb' else 0 for arm in budget()}
    return local_budget(binding['cpu_host'], binding.get('include_phys') is True)
