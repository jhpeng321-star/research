"""Actual F1-F5 smoke evidence; every admission uses measured component output."""
from collections import Counter
from pathlib import Path
import math

from . import launch_production as production, launch_training as training
from . import track2_backend as p1, track2_monitor as p3
from .exact_grade import _seal


def update_evidence(arm, episodes, learning, envelope, metrics):
    if arm == 'A':
        return dict(arm='A', cooperation_layer=False)
    counts = Counter()
    examples = []
    for episode in episodes:
        for step in episode['steps']:
            sample = step['sample']
            record = sample.get('coop_layer', {})
            if 'base_action' not in record:
                continue
            counts['recorded_samples'] += 1
            members = sample['context']['members']
            for i, vid in enumerate(members):
                base = record['base_action'][i][0]
                correction = record['applied_correction'][i][0]
                actual = sample['own_action'][i][0]
                if not all(math.isfinite(x) for x in (base, correction, actual)) or not math.isclose(
                        actual-base, correction, rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError('actual sampled command does not match additive actuation receipt')
                offset = sample['context'].get('move_offsets', {}).get(vid, 0.)
                if offset:
                    counts['nonzero_library_member_ticks'] += 1
                    if math.isclose(correction, offset, rel_tol=1e-6, abs_tol=1e-6):
                        counts['full_strength_zero_residual_member_ticks'] += 1
            for row in step.get('library_actuation_audit', []):
                offset = row.get('library_move_offset_mps2', 0.)
                if not offset:
                    continue
                if not math.isclose(row['requested_raw']-row['without_library_raw'], offset,
                                    rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError('library offset differs from actual raw request delta')
                measured = row['issued_safe']-row['without_library_issued_safe']
                if not math.isclose(measured,row['library_issued_delta_mps2'],rel_tol=1e-6,abs_tol=1e-6):
                    raise ValueError('inconsistent safety-boundary actuation receipt')
                counts['audited_library_member_ticks'] += 1
                if abs(measured)>1e-6:
                    counts['physically_changed_issued_commands'] += 1
                    if len(examples)<16:
                        examples.append(dict(episode=episode['id'], time_s=step['time_s'],
                            vid=row['vid'], library_offset_mps2=offset,
                            issued_safe=row['issued_safe'], without_library_issued_safe=row['without_library_issued_safe']))
    bank = learning.get('bank_trigger', {})
    return dict(arm=arm, actuation=dict(counts), actuation_examples=examples,
        proposals=metrics.get('move_proposals'),
        trigger_breadth={k:bank.get(k) for k in ('examples','contexts','family_counts','family_mass','rule')},
        safety_reward=envelope['receipt'])


def reporting(cfg):
    """One prospectively fixed extra state per B/R, full group/solo/base branches.

    This is charged smoke evidence, not a scheduled probe, statistical verdict or
    replacement for the original N5 u6/u12/u24 grid.
    """
    from .protocol import digest
    root = Path(cfg['run_root']); artifacts=root/'smoke'
    backend = training.MonitorBackend(cfg, root=artifacts)
    native = production.backend_config(cfg, root=artifacts, smoke=True)
    first = backend.registration['replicates'][0]
    states = backend.resolve_states(first)[:1]
    common = production.checkpoint_weights(native,'A',0)
    prepared=[]; jobs=[]
    for arm in ('B','R'):
        job=dict(id=f"clean2_smoke{cfg.get('smoke_attempt',1)}_reporting_{arm}",
            arm=arm,update=1,replicate=0,world_seed=first['world_seed'],**p3.diagnostic_identity(states))
        current=production.checkpoint_weights(native,arm,1)
        expanded=p3.isolated_expansion(job,states,current,common)
        prepared.append((job,expanded))
        jobs.extend(dict(id=s['verification_id'],sim_s=s['sim_s']) for _,pairs in expanded for s,_ in pairs)
    quote=cfg['clean2_smoke_reporting_quote']
    if sum(j['sim_s'] for j in jobs)!=quote['sim_s'] or len(jobs)!=len(quote['branches']):
        raise ValueError('prospective full-window reporting-smoke quote differs')
    training.write(root/'SMOKE_REPORTING_STARTED.json',dict(config_sha256=digest(cfg),jobs=jobs,
        prospective_quote=quote,rule='one fixed state; unchanged full window; B/R group/solos/common base; no criterion'))
    try:
        accounting,records=p3._run_wave(artifacts,backend,jobs,prepared,'track2_smoke')
    finally:
        backend.close()
    reports=p3._metric_reports(records)
    if set(reports)!={'delta_all_new_veh_s','policy_cooperation_gain_veh_s'} or any(
            set(row['arm_means'])!={'B','R'} or any(not math.isfinite(v) for v in row['arm_means'].values())
            for row in reports.values()):
        raise ValueError('both actual labelled reporting quantities required')
    safety={k:sum(row['safety'][k] for row in records) for k in p3.SAFETY}
    if safety['collisions'] or safety['teleports']:
        training.write(root/'PAUSE_SAFETY.json',dict(reason='clean2 reporting smoke safety failure',safety=safety))
        raise ValueError('smoke requires zero collisions/teleports')
    receipt=_seal(dict(status='COMPLETE',config_sha256=digest(cfg),records=records,
        metric_reports=reports,accounting=accounting,safety=safety,
        inference='descriptive one-state smoke only; registered u24 synergy criterion unchanged'), 'receipt_sha256')
    training.write(root/'SMOKE_REPORTING_PASS.json',receipt)
    return receipt


def validate_smoke(cfg, updates, reporting_receipt=None):
    from . import bank_breadth
    root=Path(cfg['run_root']);arms={u['arm']:u for u in updates};pools={}
    for arm in ('B','R'):
        update=arms[arm];e=update['clean2_fix_receipts'];counts=e['actuation']
        if (counts.get('nonzero_library_member_ticks',0)<=0 or
                counts.get('full_strength_zero_residual_member_ticks')!=counts.get('nonzero_library_member_ticks') or
                counts.get('audited_library_member_ticks',0)<=0 or
                counts.get('physically_changed_issued_commands',0)<=0):
            raise ValueError('F1 full-strength move must change actual issued commands')
        proposals=e['proposals']
        if not proposals or proposals.get('decisions',0)<=0 or proposals.get('selected_non_none',0)<=0:
            raise ValueError('F2 actual proposals and infeasible-rate receipt required')
        expected=proposals['rejected_non_none']/proposals['selected_non_none']
        if proposals.get('infeasible_rate')!=expected:
            raise ValueError('F2 actual infeasible-rate arithmetic differs')
        native=production.backend_config(cfg,root=root/'smoke',smoke=True)
        pool=bank_breadth.practice_pool(native,arm)
        ids={row['context_id'] for row in pool}
        selected={row['breadth']['context_id'] for row in update['practice']['episodes'] if row.get('breadth')}
        if not pool or not selected or not selected<=ids or not e['trigger_breadth']['contexts']:
            raise ValueError('F3 actual verified pool/selected-practice/trigger contexts required')
        pools[arm]=dict(available_unique_contexts=len(ids),context_ids=sorted(ids),
            selected_context_ids=sorted(selected),family_counts=dict(Counter(row['family'] for row in pool)))
        reward=e['safety_reward']
        if (not reward.get('close_call') or reward.get('move_exposed_cav_ticks',0)<=0 or
                'close_call_penalty_sum' not in reward or reward['close_call_penalty_sum']>0):
            raise ValueError('F4 measured exposed safety-term rows required')
    if arms['B']['clean2_fix_receipts']['trigger_breadth']['family_mass']!=arms['R']['clean2_fix_receipts']['trigger_breadth']['family_mass']:
        raise ValueError('F3 B/R guidance family distribution differs')
    if reporting_receipt is None:
        # Precheck before spending the extra reporting branches. This is never
        # a published pass and cannot satisfy launch/release admission.
        return dict(status='COMPONENT_PRECHECK_ONLY',practice_pools=pools)
    receipt=_seal(dict(status='FIX_RECEIPTS_PASS',arms={a:arms[a]['clean2_fix_receipts'] for a in ('B','R')},
        practice_pools=pools,reporting_receipt_sha256=reporting_receipt['receipt_sha256'],
        metric_reports=reporting_receipt['metric_reports']), 'receipt_sha256')
    training.write(root/'SMOKE_FIX_RECEIPTS_PASS.json',receipt)
    return receipt
