"""CPU-only standard recovery admission; no dispatch, refund, or STOP removal.

Reconstruct the full registered panel instead of considering only currently queued
jobs. Existing successful episode/branch physics is credited exactly once; every
abandoned prefix remains in the charged ledger. Shared branch/episode recovery is
not assumed. A future implementation needs separate scientific validation.
"""
import json
import math

from .continuation import TOTAL
from .design import BUDGET, CAP, CONTROLLERS, measurement_indices, worlds
from .episode import ARMS
from .ops import EVIDENCE
from .storage import canonical, write


def expected_jobs():
    jobs = {}
    for arm in ARMS:
        for world in worlds('main'):
            stem = f'{arm}_s{world["seed"]}'
            jobs[stem+'_episode'] = dict(arm=arm, sim_s=900)
            if arm in ('wm', 'bb'):
                for j in measurement_indices(world['seed']):
                    for controller in CONTROLLERS:
                        jobs[f'{stem}_j{j}_{controller}'] = dict(arm=arm, sim_s=390+60*j)
    return jobs


def plan(audit):
    cpu, gpu = audit['cpuQ'], audit['gpuR']
    rows = cpu['accounting']
    expected = expected_jobs()
    jobs = cpu['jobs']
    if len({x['id'] for x in jobs}) != len(jobs):
        raise ValueError('duplicate job receipt')
    if len({x['id'] for x in rows}) != len(rows):
        raise ValueError('duplicate ledger reservation')
    if any(not math.isfinite(x['sim_s']) or x['sim_s'] < 0 for x in rows):
        raise ValueError('invalid ledger debit')
    if any(x['id'] not in expected for x in jobs):
        raise ValueError('unregistered job receipt')
    if any(cpu['own_pids'].values()) or any(g['owned_pid_alive'] for g in gpu['arms'].values()):
        raise ValueError('owned process still alive; inspection only')
    credits = {arm:0 for arm in ARMS}
    abandoned = {arm:0 for arm in ARMS}
    complete, partial = [], []
    for job in jobs:
        close = job['close']
        if close is None:
            raise ValueError('unclosed job; actual simulation requires reconciliation')
        actual, charged = close['actual_sim_s'], close['charged_sim_s']
        if (not math.isfinite(actual) or actual < 0 or
                actual != charged or charged != job['charged_sim_s']):
            raise ValueError('job actual/charged mismatch')
        debit = sum(x['sim_s'] for x in rows if x['id'].startswith(job['id']+'_'))
        if debit != charged:
            raise ValueError('job/ledger debit mismatch')
        want = expected[job['id']]
        if actual > want['sim_s']:
            raise ValueError('run exceeds its registered duration')
        if job['result_present']:
            if actual != want['sim_s']:
                raise ValueError('completed run has incomplete accounting')
            if job['id'].endswith('_episode') and job['picks'] != list(range(10)):
                raise ValueError('completed episode lacks all ten picks')
            complete.append(job['id'])
            credits[want['arm']] += want['sim_s']
        else:
            partial.append(job['id'])
            abandoned[want['arm']] += actual
    charged = sum(x['sim_s'] for x in rows)
    closed = 47010+12450+sum(x['close']['actual_sim_s'] for x in jobs)
    if charged != closed:
        raise ValueError('global actual/charged mismatch')
    per_arm = {}
    for arm in ARMS:
        used = sum(x['sim_s'] for x in rows if x['id'].startswith(arm+'_'))
        if used != credits[arm]+abandoned[arm]:
            raise ValueError('unattributed arm debit')
        remaining = BUDGET[arm]-credits[arm]
        per_arm[arm] = dict(charged_sim_s=used, reused_complete_sim_s=credits[arm],
                            abandoned_sim_s=abandoned[arm], remaining_sim_s=remaining,
                            projected_total_sim_s=used+remaining)
    projected = charged+sum(x['remaining_sim_s'] for x in per_arm.values())
    if projected != TOTAL+sum(abandoned.values()):
        raise ValueError('registered plan reconstruction mismatch')
    reasons = []
    if projected > CAP:
        reasons.append(f'standard full-rerun recovery exceeds SUMO cap by {projected-CAP:g} sim-s')
    gpu_budget = {}
    for arm, cap in [('wm',10800), ('bb',3600)]:
        service = gpu['arms'][arm]
        if 'GPU_FINISHED.json' not in service:
            raise ValueError('GPU allocation lacks terminal receipt')
        used = service['GPU_FINISHED.json']['cumulative_gpu_s']
        if not math.isfinite(used) or used < 0:
            raise ValueError('invalid GPU debit')
        picks = sum(len(x['picks']) for x in jobs if x['id'].startswith(arm+'_') and x['id'].endswith('_episode'))
        remaining_picks = 160-picks
        gpu_budget[arm] = dict(cap_s=cap, cumulative_gpu_s=used,
                               remaining_gpu_s=max(0,cap-used), uncommitted_picks=remaining_picks)
        if remaining_picks and used >= cap:
            reasons.append(f'{arm} GPU hard cap exhausted with {remaining_picks} uncommitted picks')
    # Passing numerical budgets would not constitute authority to bypass STOP or
    # reuse results without their original live-prefix and source binding checks.
    return dict(status='BLOCKED' if reasons else 'REQUIRES_VALIDATED_RECOVERY',
                dispatch_authorized=False, reasons=reasons, per_arm=per_arm,
                gpu=gpu_budget, charged_sim_s=charged, actual_closed_sim_s=closed,
                projected_total_sim_s=projected, hard_cap_sim_s=CAP,
                complete_jobs=complete, partial_jobs=partial,
                evidence_sha256=canonical(audit), new_sim_s=0, new_gpu_s=0)


def main():
    audit = json.loads((EVIDENCE/'V7_INTERRUPTION_AUDIT.json').read_text())
    result = plan(audit)
    write(EVIDENCE/'V7_RECOVERY_ADMISSION.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('complete_jobs','partial_jobs')},indent=2))


if __name__ == '__main__':
    main()
