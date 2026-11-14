"""Pure, fail-closed cumulative accounting for the registered clean3 restart.

This module does not stop a run, reserve a charge, or authorize a dispatch. Final
admission must be rebuilt from the actual stopped ledgers; a running-prefix quote
is diagnostic only. Existing plan_fix work consumes its protected reservation.
"""
from copy import deepcopy

from . import clean2_budget, rerun_cpus
from .exact_grade import _seal, verify_sealed
from .protocol import digest

CAP = 6_000_000
G_LEVER_CAP = 230_400
OWNER_SCOPE = 'clean1+clean2+plan_fix+clean3 (incl. G1 certification)'
SCHEMA = 'coopmwm.clean3.cumulative_quote.v2'


def _approval(value):
    if (value.get('owner_approval') != 'yes, approve.'
            or type(value.get('total_cap_sim_s')) is not int
            or value['total_cap_sim_s'] != CAP
            or value.get('scope') != OWNER_SCOPE):
        raise ValueError('registered cumulative clean3 owner approval required')


def _prior(rows, name):
    seen, spent, plan = set(), 0, 0
    for row in rows:
        ident = row['id']
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ValueError('unique nonempty charge id required: '+name)
        seen.add(ident)
        amount = clean2_budget._count(row['sim_s'], name+' charge')
        spent += amount
        if row['phase'] == 'plan_fix':
            if name != 'clean1':
                raise ValueError('plan_fix belongs only to the clean1 ledger')
            plan += amount
    return dict(rows=len(rows), canonical_prefix_sha256=digest(rows),
                charged_sim_s=spent, plan_fix_sim_s=plan)


def _lesson_costs(quotes):
    expected = {f'lesson_u{u:03}': u for u in rerun_cpus.SCHEDULE['B']['lessons']}
    if set(quotes) != set(expected):
        raise ValueError('complete native lesson schedule quote required')
    costs = {}
    for name, update in expected.items():
        row = quotes[name]
        verify_sealed(row, 'receipt_sha256')
        if (row.get('kind') != 'lesson' or row.get('status') != 'dry_run'
                or row.get('update') != update or row.get('new_sim_s') != 0):
            raise ValueError('zero-physics native lesson quote required')
        manifest = row['branch_manifest']
        if (not manifest or len(manifest) != row['branches']
                or digest(manifest) != row['quote_sha256']):
            raise ValueError('complete native lesson manifest required')
        for i, branch in enumerate(manifest):
            window = branch['window']
            if (branch['id'] != f'{name}_{i:04}' or window.get('truncated') is not False
                    or not 0 <= window['start_s'] < window['end_s'] <= 1200):
                raise ValueError('full original lesson window and unique branch id required')
            # Native replay includes its entire prefix, not just the suffix.
            from .sumo_verify import branch_sim_s
            if clean2_budget._count(branch['sim_s'], 'lesson sim_s') != branch_sim_s(window['end_s']):
                raise ValueError('full-prefix native lesson cost required')
        costs[name] = sum(b['sim_s'] for b in manifest)
        if costs[name] != row['quoted_sim_s']:
            raise ValueError('native lesson quote sum mismatch')
    return costs


def prospective(*, clean1_rows, clean2_rows, owner_budget, diagnostic_quotes,
                gate_manifest, lesson_quotes):
    """Quote the unchanged full design, retaining overbudget results as evidence.

    The G allocations use the complete registered bounds, without an unqualified
    two-actor/reuse discount. Final accepted G adapters and smoke still require
    qualification; a fit here alone is never permission to simulate.
    """
    _approval(owner_budget)
    prior_ledgers = {name: _prior(rows, name) for name, rows in
                     (('clean1', clean1_rows), ('clean2', clean2_rows))}
    plan = prior_ledgers['clean1']['plan_fix_sim_s']
    if plan > clean2_budget.PLAN_FIX_CAP:
        raise ValueError('plan_fix exceeds its registered phase cap')
    costs = clean2_budget._diagnostic_costs(diagnostic_quotes)
    lessons = _lesson_costs(lesson_quotes)
    if not gate_manifest:
        raise ValueError('complete final-runtime gate manifest required')
    seen, gate = set(), 0
    for branch in gate_manifest:
        ident = branch['id']
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ValueError('unique nonempty safety gate branch id required')
        seen.add(ident)
        if branch['window'].get('truncated') is not False:
            raise ValueError('full-window safety gate required')
        amount = clean2_budget._count(branch['sim_s'], 'gate sim_s')
        if amount == 0:
            raise ValueError('nonempty safety gate branch required')
        gate += amount
    phases = deepcopy(rerun_cpus.budget(track2=True)['phases'])
    phases.update(teacher_verify=0, group_credit_verify=0, augment_verify=0,
        track2_smoke=20_000, gate_bank_verification_ext=gate,
        early_probes=costs['probe_u002']+costs['probe_u004'],
        isolated_replicated=sum(costs[f'replicated_u{u:03}'] for u in (6, 12, 24)),
        v2v_lessons=max(phases['v2v_lessons'], sum(lessons.values())),
        g1_certification=G_LEVER_CAP, g4_synergy_credit=G_LEVER_CAP)
    for name, amount in phases.items():
        clean2_budget._count(amount, name)
    prior = sum(row['charged_sim_s'] for row in prior_ledgers.values())+clean2_budget.PARITY
    reserved = clean2_budget.PLAN_FIX_CAP-plan
    allocation = sum(phases.values())
    bound = prior+reserved+allocation
    return _seal(dict(schema=SCHEMA, owner_budget_sha256=digest(owner_budget),
        owner_cap_sim_s=CAP, prior_ledgers=prior_ledgers,
        prior_parity_sim_s=clean2_budget.PARITY, prior_charged_sim_s=prior,
        plan_fix_unspent_reserved_sim_s=reserved, phases=phases,
        allocated_sim_s=allocation, diagnostic_costs=costs,
        lesson_costs=lessons, lesson_quotes=deepcopy(lesson_quotes),
        diagnostic_quotes=deepcopy(diagnostic_quotes), gate_manifest=deepcopy(gate_manifest),
        cumulative_upper_bound_sim_s=bound, margin_sim_s=CAP-bound,
        fits_current_prefix=bound <= CAP,
        status='FITS_CURRENT_PREFIX' if bound <= CAP else 'BLOCKED_BUDGET',
        training_updates=24, arms=['A', 'B', 'R'], start_updates=dict(A=0, B=0, R=0),
        gate_attempts=1, gate_environment_retries=0,
        launch_authorized=False), 'receipt_sha256')


def admit(quote, *, clean1_rows, clean2_rows, owner_budget, accepted_levers,
          clean2_stop_sha256, sealed_updates):
    """Recompute against final ledger snapshots before a chain may charge.

    The caller separately verifies STOP/process identities and source/smoke/gate
    bindings. The stop digest and actual sealed update indices bind that evidence
    here. This accounting admission is not a smoke/gate or launch receipt.
    """
    verify_sealed(quote, 'receipt_sha256')
    if set(accepted_levers) != {'G1', 'G2', 'G4'}:
        raise ValueError('all G1/G2/G4 acceptance receipts required')
    for name, sha in accepted_levers.items():
        clean2_budget._sha(sha, name+' acceptance')
    clean2_budget._sha(clean2_stop_sha256, 'clean2 durable owner stop')
    if (set(sealed_updates) != {'A', 'B', 'R'}
            or any(type(u) is not int or not 0 <= u <= 24 for u in sealed_updates.values())):
        raise ValueError('actual sealed A/B/R boundary required')
    rebuilt = prospective(clean1_rows=clean1_rows, clean2_rows=clean2_rows,
        owner_budget=owner_budget, diagnostic_quotes=quote['diagnostic_quotes'],
        gate_manifest=quote['gate_manifest'], lesson_quotes=quote['lesson_quotes'])
    if quote != rebuilt:
        raise ValueError('clean3 quote differs from current ledgers or full registered design')
    if not rebuilt['fits_current_prefix']:
        raise ValueError('complete clean3 quote exceeds cumulative registered cap')
    return _seal(dict(schema='coopmwm.clean3.accounting_admission.v1',
        quote_sha256=quote['receipt_sha256'], owner_budget_sha256=digest(owner_budget),
        accepted_levers=deepcopy(accepted_levers), clean2_stop_sha256=clean2_stop_sha256,
        stopped_at_updates=deepcopy(sealed_updates), budget=dict(total=CAP-rebuilt['prior_charged_sim_s']
            -rebuilt['plan_fix_unspent_reserved_sim_s'], phases=deepcopy(rebuilt['phases']),
            allocated=rebuilt['allocated_sim_s'], unallocated=rebuilt['margin_sim_s']),
        cumulative_upper_bound_sim_s=rebuilt['cumulative_upper_bound_sim_s'],
        smoke_and_gate_required=True, launch_authorized=False), 'receipt_sha256')
