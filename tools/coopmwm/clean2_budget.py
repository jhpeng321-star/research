"""Disjoint remaining-cap accounting for the owner-approved clean2 restart.

Plan-fix owns a phase in clean1's ledger. Reserve its entire remaining approval,
not a second copy of its spent rows. Diagnostic costs come from full native dry
branch manifests; no horizon, branch, arm or replication count is reduced.
"""
from copy import deepcopy
import re

from .exact_grade import _seal, verify_sealed
from .protocol import digest

CAP = 5_000_000
PARITY = 960
PLAN_FIX_CAP = 600_000
SCHEMA = 'coopmwm.clean2.remaining_cap.v1'


def _count(value, label):
    if type(value) is not int or value < 0:
        raise ValueError('nonnegative integer required: '+label)
    return value


def _sha(value, label):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('SHA256 required: '+label)
    return value


def _diagnostic_costs(quotes):
    expected = {'probe_u002': ('probe', 2, 3), 'probe_u004': ('probe', 4, 3),
                **{f'replicated_u{u:03}': ('replicated', u, 10) for u in (6, 12, 24)}}
    if set(quotes) != set(expected):
        raise ValueError('complete u2/u4 and N5 B/R u6/u12/u24 dry quotes required')
    costs = {}
    for name, (kind, update, cells) in expected.items():
        row = quotes[name]
        verify_sealed(row, 'receipt_sha256')
        if (row.get('status') != 'dry_run' or row.get('update') != update
                or row.get('kind') != kind or row.get('cells') != cells
                or row.get('new_sim_s') != 0):
            raise ValueError('incomplete or measured diagnostic quote: '+name)
        manifest = row['branch_manifest']
        if not manifest or len(manifest) != row['branches']:
            raise ValueError('complete full-window branch manifest required')
        if digest(manifest) != row['quote_sha256']:
            raise ValueError('diagnostic manifest SHA mismatch')
        ids = set()
        for branch in manifest:
            if branch['id'] in ids or branch['window'].get('truncated'):
                raise ValueError('duplicate or truncated diagnostic branch')
            ids.add(branch['id'])
            if _count(branch['sim_s'], 'branch sim_s') == 0:
                raise ValueError('nonempty full-window branch required')
        costs[name] = sum(b['sim_s'] for b in manifest)
        if costs[name] != row['quoted_sim_s']:
            raise ValueError('full diagnostic quote sum mismatch')
    return costs


def build(*, clean1_rows, clean1_ledger_sha256, clean1_stop_sha256,
          accepted_fixes, diagnostic_quotes, gate_manifest, canonical_budget):
    """Pure prospective registration. Every spent/void row remains charged."""
    if set(accepted_fixes) != {f'F{i}' for i in range(1, 7)}:
        raise ValueError('all F1-F6 acceptance receipts required')
    for name, value in accepted_fixes.items():
        _sha(value, name)
    _sha(clean1_ledger_sha256, 'clean1 ledger')
    _sha(clean1_stop_sha256, 'clean1 durable STOP')
    seen, spent, plan_spent = set(), 0, 0
    for row in clean1_rows:
        ident = row['id']
        if ident in seen:
            raise ValueError('duplicate clean1 charge id')
        seen.add(ident)
        amount = _count(row['sim_s'], 'clean1 charge')
        spent += amount
        if row['phase'] == 'plan_fix':
            plan_spent += amount
    if plan_spent > PLAN_FIX_CAP:
        raise ValueError('plan_fix exceeds approved cap')
    # Existing plan-fix spend is already inside spent. Only its unspent portion
    # is an additional reservation. Concurrent later appends consume this reserve.
    prior = spent + PARITY + (PLAN_FIX_CAP-plan_spent)
    remaining = CAP-prior
    if remaining < 0:
        raise ValueError('no global remaining budget')
    costs = _diagnostic_costs(diagnostic_quotes)
    if not gate_manifest or len({b['id'] for b in gate_manifest}) != len(gate_manifest):
        raise ValueError('unique full final-runtime gate manifest required')
    gate = 0
    for branch in gate_manifest:
        if branch['window'].get('truncated'):
            raise ValueError('truncated safety gate refused')
        gate += _count(branch['sim_s'], 'gate sim_s')
    if gate <= 0:
        raise ValueError('nonempty gate required')
    phases = deepcopy(canonical_budget['phases'])
    phases.update(teacher_verify=0, group_credit_verify=0, augment_verify=0,
        track2_smoke=20_000, gate_bank_verification_ext=2*gate,
        early_probes=sum(costs[f'probe_u{u:03}'] for u in (2, 4)),
        isolated_replicated=sum(costs[f'replicated_u{u:03}'] for u in (6, 12, 24)))
    allocated = sum(phases.values())
    if allocated > remaining:
        raise ValueError('complete clean2 quote exceeds disjoint remaining approved cap')
    budget = dict(total=remaining, phases=phases, allocated=allocated,
                  unallocated=remaining-allocated)
    return _seal(dict(schema=SCHEMA, owner_quote='restart', owner_cap_sim_s=CAP,
        clean1_ledger_sha256=clean1_ledger_sha256, clean1_stop_sha256=clean1_stop_sha256,
        clean1_rows=len(clean1_rows), clean1_spent_sim_s=spent,
        clean1_non_plan_fix_sim_s=spent-plan_spent, prior_parity_sim_s=PARITY,
        plan_fix_cap_sim_s=PLAN_FIX_CAP, plan_fix_spent_in_clean1_sim_s=plan_spent,
        plan_fix_unspent_reserved_sim_s=PLAN_FIX_CAP-plan_spent,
        protected_prior_sim_s=prior, accepted_fixes=deepcopy(accepted_fixes),
        diagnostic_quotes=deepcopy(diagnostic_quotes), gate_manifest=deepcopy(gate_manifest),
        canonical_budget=deepcopy(canonical_budget), budget=budget,
        cumulative_upper_bound_sim_s=prior+allocated,
        rule='all clean1 charges including voids plus parity, all remaining approved plan_fix, and clean2; plan_fix counted once; no horizon/schedule reduction'),
        'receipt_sha256')


def checked(cfg):
    """Recompute rather than trusting a mutable summary or advertised total."""
    value = cfg['clean2_budget']
    verify_sealed(value, 'receipt_sha256')
    if value.get('schema') != SCHEMA or value.get('owner_quote') != 'restart':
        raise ValueError('registered owner clean2 restart required')
    # Recreate conservative prior rows without retaining/opening old job data.
    synthetic = [dict(id='clean1_non_plan_fix', phase='prior', sim_s=value['clean1_non_plan_fix_sim_s']),
                 dict(id='clean1_plan_fix', phase='plan_fix', sim_s=value['plan_fix_spent_in_clean1_sim_s'])]
    rebuilt = build(clean1_rows=synthetic,
        clean1_ledger_sha256=value['clean1_ledger_sha256'],
        clean1_stop_sha256=value['clean1_stop_sha256'], accepted_fixes=value['accepted_fixes'],
        diagnostic_quotes=value['diagnostic_quotes'], gate_manifest=value['gate_manifest'],
        canonical_budget=cfg['budget'])
    for key in ('budget', 'protected_prior_sim_s', 'cumulative_upper_bound_sim_s',
                'clean1_spent_sim_s', 'prior_parity_sim_s', 'plan_fix_cap_sim_s',
                'plan_fix_unspent_reserved_sim_s', 'canonical_budget'):
        if value[key] != rebuilt[key]:
            raise ValueError('clean2 remaining-cap registration differs: '+key)
    return deepcopy(value)
