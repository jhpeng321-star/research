"""The single MAIN12:14 BB exception debits its immutable predecessor per card."""
import math

AUTHORITY = 'S-C1SWITCH2 BB startup restart AUTHORISED as a registered continuation exception'
PREDECESSOR = '/root/c1switch2_gpu_bb_caeab9eaa303f93c'
PRIOR = {'gpuA': 21.381694240495563, 'gpuR': 21.642283007502556}


def allocation_cap(binding, allocation=None):
    from .design import GPU_SECONDS
    cap = GPU_SECONDS[binding['arm']]/2
    if binding['service_gpu_cap_s'] != cap or binding.get('prior_gpu_s', 0) != 0:
        raise ValueError('unregistered GPU engine allocation')
    if allocation is None:
        return cap
    host = allocation.get('host')
    if (binding['arm'] != 'bb' or host not in PRIOR or
            allocation.get('schema') != 'S-C1SWITCH2/BB-MAIN1214-v1' or
            allocation.get('predecessor_root') != PREDECESSOR or
            AUTHORITY not in allocation.get('authority', '') or
            not allocation.get('plan_sha256')):
        raise ValueError('only registered MAIN12:14 BB continuation permitted')
    prior = allocation.get('prior_gpu_s')
    remaining = allocation.get('remaining_gpu_s')
    if (prior != PRIOR[host] or allocation['finished']['elapsed_s'] != prior or
            not isinstance(remaining, (int, float)) or not math.isfinite(remaining) or
            remaining != cap-prior or not 0 < remaining < cap):
        raise ValueError('retain exact per-card predecessor debit; no refund or top-up')
    return remaining
