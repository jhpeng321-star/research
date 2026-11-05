"""Hard-coded clean3 G-lever smoke receipt checks (B and R identical rules)."""
import json
from pathlib import Path

from .exact_grade import _seal, verify_sealed


def validate_levers(cfg, updates, fixes=None):
    from . import bank_breadth, composite_health, coordination_opportunities as co
    from . import executable_labels as labels, g4_runtime as g4, group_lane
    from . import launch_production as production, launch_training as training
    root = Path(cfg['run_root'])
    arms = {u['arm']: u for u in updates}
    out = {}
    # Safety rule registered before launch: B/R composite report-only, A full.
    for arm in ('B', 'R'):
        if not composite_health.report_only(root, arm):
            raise ValueError('clean3 registered B/R report-only safety rule missing: '+arm)
    # G1: certified-label gate bound exactly when the go/no-go receipt says GO.
    go = training.read(Path(cfg['clean3_restart']['g1_root'])/'CLEAN3_GO_NOGO.json')
    verify_sealed(go, 'receipt_sha256')
    if go['receipt_sha256'] != cfg['clean3_restart']['g1_go_nogo_sha256']:
        raise ValueError('G1 go/no-go receipt changed')
    if go['decision'] == 'GO':
        manifest = labels.read_manifest(cfg['executable_label_manifest'])
        if (manifest['manifest_sha256'] != go['manifest_sha256'] or manifest['admitted'] <= 0
                or cfg.get('executable_label_runtime_sha256') != go['runtime_sha256']):
            raise ValueError('G1 GO requires the bound admitted certificate manifest')
    elif cfg.get('executable_label_manifest'):
        raise ValueError('G1 NO-GO must not bind a certificate manifest')
    out['G1'] = dict(decision=go['decision'], admitted=go['executable_joint_labels_ci95_low_positive'],
                     bound=bool(cfg.get('executable_label_manifest')))
    # G2: actual opportunity-graph receipts from smoke trigger decisions. The
    # registration is validated on the native config (it owns trigger_candidates).
    native = production.backend_config(cfg)
    if co.settings(native) is None:
        raise ValueError('G2 coordination opportunities must be enabled')
    out['G2'] = {}
    for arm in ('B', 'R'):
        proposals = arms[arm]['metrics'].get('move_proposals') or {}
        row = proposals.get('coordination_opportunities')
        if (not row or row.get('rule') != co.RULE or row.get('decisions', 0) <= 0
                or row.get('offered_joint_rate') is None):
            raise ValueError('G2 measured opportunity receipt required: '+arm)
        out['G2'][arm] = {k: row[k] for k in ('decisions', 'offered_joint', 'offered_joint_rate',
                                              'graph_edges', 'disjoint_pairs', 'event_decisions')}
    # G4: TRAIN-only by accepted design; binding and full phase must be live from u1.
    if not g4.enabled(native, 1) or native['budget']['phases'].get('g4_synergy_credit') != 230400:
        raise ValueError('G4 sparse synergy credit must be bound from u1 with its 230400 phase')
    if g4.enabled(production.backend_config(cfg, root=root/'smoke', smoke=True), 1, smoke=True):
        raise ValueError('G4 must stay disabled inside smoke (TRAIN-only sampling)')
    out['G4'] = dict(first_update=cfg['g4_synergy_credit']['first_update'],
                     phase_sim_s=native['budget']['phases']['g4_synergy_credit'])
    # G5: lane families in the live proposal vocabulary of both arms.
    if not group_lane.enabled(cfg) or len(group_lane.families(cfg)) != 7:
        raise ValueError('G5 seven-family lane vocabulary required')
    out['G5'] = {}
    for arm in ('B', 'R'):
        proposed = (arms[arm]['metrics'].get('move_proposals') or {}).get('proposed_by_family') or {}
        if not {'yield_lane', 'open_gap_merge'} <= set(proposed):
            raise ValueError('G5 lane families missing from actual proposals: '+arm)
        out['G5'][arm] = {f: proposed[f] for f in ('yield_lane', 'open_gap_merge')}
    # Owner 05:05 mined-plan exposure (a)-(d) for B (and R with its own examples) + fairness.
    if not cfg.get('mined_moves'):
        raise ValueError('owner rule: clean3 launches only with admitted mined moves')
    from .clean3_mined import smoke_check
    out['mined'] = smoke_check(cfg, updates, root/'smoke')
    if fixes is None:
        return dict(status='LEVER_PRECHECK_ONLY', **out)
    receipt = _seal(dict(status='LEVER_RECEIPTS_PASS', levers=out,
        fix_receipts_sha256=fixes['receipt_sha256'],
        smoke_updates={a: arms[a]['update_sha256'] for a in arms}), 'receipt_sha256')
    training.write(root/'SMOKE_LEVER_RECEIPTS_PASS.json', receipt)
    return receipt
