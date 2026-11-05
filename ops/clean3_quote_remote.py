"""Read-only clean3 full-design quote; no root creation or simulation dispatch.

Run through the accepted F6 clean2 transport with the qualified research env.
All costs are retained from the full native clean2 manifests, except that the
new owner's once-only gate has no retry allocation. G1/G4 reserve their complete
registered four-actor bounds rather than assuming two-actor discounts.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from tools.coopmwm import clean2_budget
from tools.coopmwm.exact_grade import verify_sealed
from tools.coopmwm.protocol import digest
from tools.coopmwm.storage import ledger


clean1 = Path('/home/ubuntu/coopmwm20261105_clean1')
clean2 = Path('/home/ubuntu/coopmwm20261105_clean2')
cfg = json.loads((clean2/'CONFIG.json').read_text())
previous = clean2_budget.checked(cfg)
quotes = previous['diagnostic_quotes']
costs = clean2_budget._diagnostic_costs(quotes)
gate_manifest = previous['gate_manifest']
gate_sim_s = sum(row['sim_s'] for row in gate_manifest)
phases = dict(previous['budget']['phases'])
phases['gate_bank_verification_ext'] = gate_sim_s
phases['g1_certification'] = 230400
phases['g4_synergy_credit'] = 230400
assert phases['track2_smoke'] == 20000
assert phases['early_probes'] == costs['probe_u002']+costs['probe_u004']
assert phases['isolated_replicated'] == sum(costs[f'replicated_u{u:03}'] for u in (6, 12, 24))
assert all(type(value) is int and value >= 0 for value in phases.values())
ledgers = {}
for name, root in (('clean1', clean1), ('clean2', clean2)):
    rows = ledger(root)
    assert len({row['id'] for row in rows}) == len(rows)
    assert all(type(row['sim_s']) is int and row['sim_s'] >= 0 for row in rows)
    ledgers[name] = dict(rows=len(rows), canonical_prefix_sha256=digest(rows),
        charged_sim_s=sum(row['sim_s'] for row in rows),
        plan_fix_sim_s=sum(row['sim_s'] for row in rows if row['phase'] == 'plan_fix'))
plan_spent = ledgers['clean1']['plan_fix_sim_s']
assert 0 <= plan_spent <= 600000
prior = sum(row['charged_sim_s'] for row in ledgers.values())+960
protected = 600000-plan_spent
allocation = sum(phases.values())
monitors = []
for u in range(1, 25):
    path = clean2/f'LEARNING_MONITOR_u{u:03}.json'
    if path.is_file():
        row = json.loads(path.read_text())
        verify_sealed(row, 'receipt_sha256')
        monitors.append(dict(update=u, receipt_sha256=row['receipt_sha256']))
print(json.dumps(dict(schema='coopmwm.clean3.prospective_quote.v1',
    timestamp=datetime.now(timezone.utc).isoformat(), read_only=True,
    status='FITS_CURRENT_PREFIX' if prior+protected+allocation <= 6000000 else 'BLOCKED_BUDGET',
    owner_budget=dict(owner_approval='yes, approve.', total_cap_sim_s=6000000,
        scope='clean1+clean2+plan_fix+clean3 (incl. G1 certification)'),
    prior_ledgers=ledgers, prior_parity_sim_s=960, prior_charged_sim_s=prior,
    plan_fix_unspent_reserved_sim_s=protected,
    available_after_prior_and_plan_fix_sim_s=6000000-prior-protected,
    phases=phases, allocated_sim_s=allocation,
    cumulative_upper_bound_at_observed_prefix_sim_s=prior+protected+allocation,
    remaining_margin_at_observed_prefix_sim_s=6000000-prior-protected-allocation,
    fits_current_prefix=prior+protected+allocation <= 6000000,
    training_updates=24, arms=['A', 'B', 'R'], start_updates=dict(A=0, B=0, R=0),
    diagnostic_costs=costs, diagnostic_quotes=quotes, gate_manifest=gate_manifest,
    gate_attempts=1, gate_environment_retries=0,
    smoke_actual_existing_quote=cfg['clean2_smoke_reporting_quote'],
    g1=dict(contexts_per_arm=8, arms=['B','R'], seeds=2,
        preferred_two_actor_sim_s=153600, worst_case_sim_s=230400, reuse_discount_sim_s=0),
    g4=dict(max_starts_per_arm=8, arms=['B','R'], seeds=2,
        preferred_two_actor_sim_s=153600, worst_case_sim_s=230400, reuse_discount_sim_s=0),
    inherited_config_file_sha256=hashlib.sha256((clean2/'CONFIG.json').read_bytes()).hexdigest(),
    observed_monitors=monitors,
    qualification_and_launch_authorized=False,
    finalization_requirements=['G1/G2/G4 acceptance and integrated qualification',
        'Recompute against final stopped clean2 ledger; current running prefix is not final',
        'Assert complete quote plus prior charges plus protected plan_fix <= registered cap',
        'Qualify production G1/G4 adapters and all fix/G receipt checks before any physics',
        'One detached chain only; preserve clean1 and clean2; no retry or weakened check']),
    sort_keys=True, allow_nan=False))
