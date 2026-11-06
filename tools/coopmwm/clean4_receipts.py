"""Hard-coded clean4 (wz3to2) smoke lever receipt checks; B and R identical rules.

Registered before the clean4 smoke (N4, REGISTRATION 2026-11-06): every component ON must
produce its own measured receipt in the one-update smoke:
  safety  - B/R composite report-only owner rule present (A keeps the full rule);
  G2      - coordination-opportunity receipts with decisions > 0 in B and R;
  G4      - sparse synergy credit bound from u1 with its 230,400 phase; disabled inside smoke;
  G5      - the seven-family lane vocabulary is live (descriptive per-family proposal counts);
  W1      - V2V wall protocol on (range 200 m) and its initiate family proposed in B and R;
  W2      - pressure/release/negatives settings on, identical for B and R;
  N2      - lane-drop moves on: closing_pair or abreast_wall proposed in B and R, lane-drop
            initiate family proposed when V2V is on, RSU head trained (records > 0) when RSU is on;
  N9      - every registered ON family (when merged) is in the live vocabulary of B and R;
  mined   - Plan A guidance rows: owner exposure checks (a)-(d) (clean3_mined.smoke_check);
  map     - the process map is the registered wz3to2 descriptor / staged net.
Nothing here changes training; it reads receipts only.
"""
from pathlib import Path

from .exact_grade import _seal, verify_sealed


def _proposed(update):
    return (update['metrics'].get('move_proposals') or {}).get('proposed_by_family') or {}


def validate_levers(cfg, updates, fixes=None):
    from . import composite_health, coordination_opportunities as co, g4_runtime as g4, group_lane
    from . import launch_production as production, launch_training as training, wall_w2, clean4_restart as c4
    root = Path(cfg['run_root'])
    arms = {u['arm']: u for u in updates}
    out = {}
    c4.check_map(cfg)
    out['map'] = dict(map=cfg['newmap']['map'], descriptor_sha256=cfg['newmap']['descriptor_sha256'],
                      base_controller=cfg['newmap']['base_controller'], cell=cfg['newmap']['cell'])
    for arm in ('B', 'R'):
        if not composite_health.report_only(root, arm):
            raise ValueError('clean4 registered B/R report-only safety rule missing: '+arm)
    native = production.backend_config(cfg)
    if co.settings(native) is None:
        raise ValueError('G2 coordination opportunities must be enabled')
    out['G2'] = {}
    for arm in ('B', 'R'):
        row = (arms[arm]['metrics'].get('move_proposals') or {}).get('coordination_opportunities')
        if not row or row.get('rule') != co.RULE or row.get('decisions', 0) <= 0 or row.get('offered_joint_rate') is None:
            raise ValueError('G2 measured opportunity receipt required: '+arm)
        out['G2'][arm] = {k: row[k] for k in ('decisions', 'offered_joint', 'offered_joint_rate', 'graph_edges',
                                              'disjoint_pairs', 'event_decisions')}
    if not g4.enabled(native, 1) or native['budget']['phases'].get('g4_synergy_credit') != 230400:
        raise ValueError('G4 sparse synergy credit must be bound from u1 with its 230400 phase')
    if g4.enabled(production.backend_config(cfg, root=root/'smoke', smoke=True), 1, smoke=True):
        raise ValueError('G4 must stay disabled inside smoke (TRAIN-only sampling)')
    out['G4'] = dict(first_update=cfg['g4_synergy_credit']['first_update'],
                     phase_sim_s=native['budget']['phases']['g4_synergy_credit'])
    if not group_lane.enabled(cfg) or len(group_lane.families(cfg)) != 7:
        raise ValueError('G5 seven-family lane vocabulary required')
    out['G5'] = {arm: {f: _proposed(arms[arm]).get(f, 0) for f in ('yield_lane', 'open_gap_merge')}
                 for arm in ('B', 'R')}
    wall = cfg.get('v2v_wall') or {}
    if not wall.get('enabled') or float(wall.get('range_m', 0)) != c4.V2V_RANGE_M:
        raise ValueError('W1 V2V wall initiate/join must be on with the registered 200 m range')
    out['W1'] = {}
    for arm in ('B', 'R'):
        proposed = _proposed(arms[arm])
        if proposed.get('wall_initiate', 0) <= 0:
            raise ValueError('W1 wall_initiate never proposed: '+arm)
        out['W1'][arm] = {f: proposed.get(f, 0) for f in ('wall_initiate', 'wall_join')}
    w2 = wall_w2.settings(cfg)
    if not (w2['wall_pressure_features'] and w2['wall_release'] and w2['wall_negative_labels']):
        raise ValueError('W2 pressure/release/negative labels must be on')
    out['W2'] = {k: w2[k] for k in ('wall_pressure_features', 'wall_release', 'wall_negative_labels', 'wall_negatives')}
    ld = cfg.get('lane_drop') or {}
    if not (ld.get('enabled') and ld.get('moves')):
        raise ValueError('N2 lane-drop moves must be on')
    out['N2'] = {}
    for arm in ('B', 'R'):
        proposed = _proposed(arms[arm])
        joint = proposed.get('closing_pair', 0)+proposed.get('abreast_wall', 0)
        if joint <= 0:
            raise ValueError('N2 lane-drop joint move never proposed: '+arm)
        row = {f: proposed.get(f, 0) for f in ('closing_pair', 'abreast_wall', 'lane_drop_initiate', 'lane_drop_join')}
        initiate = sum(v for f, v in proposed.items() if f.startswith('lane_drop_initiate'))
        if ld.get('v2v') and initiate <= 0:
            raise ValueError('N2 lane-drop V2V initiate never proposed: '+arm)
        row['lane_drop_initiate_all'] = initiate
        if ld.get('rsu'):
            rsu = arms[arm]['learning'].get('rsu')
            if not rsu or not rsu.get('decisions', 0):
                raise ValueError('N2 RSU head has no recorded decisions: '+arm)
            row['rsu'] = {k: rsu.get(k) for k in ('decisions', 'accepted', 'exact_kl') if k in rsu}
        out['N2'][arm] = row
    families = cfg.get('nm_families') or {}
    if families.get('enabled'):
        out['N9'] = {}
        for arm in ('B', 'R'):
            proposed = _proposed(arms[arm])
            live = [f for f in families.get('families', ()) if f in proposed]
            if sorted(live) != sorted(families.get('families', ())):
                raise ValueError('N9 registered families missing from the live vocabulary: '+arm)
            out['N9'][arm] = {f: proposed[f] for f in live}
    if not cfg.get('mined_moves'):
        raise ValueError('clean4 launches only with Plan A guidance rows')
    from .clean3_mined import smoke_check
    out['mined'] = smoke_check(cfg, updates, root/'smoke')
    out['guidance_rows'] = cfg['clean4_restart']['guidance_rows']
    if fixes is None:
        return dict(status='LEVER_PRECHECK_ONLY', **out)
    receipt = _seal(dict(status='LEVER_RECEIPTS_PASS', levers=out, fix_receipts_sha256=fixes['receipt_sha256'],
        smoke_updates={a: arms[a]['update_sha256'] for a in arms}), 'receipt_sha256')
    training.write(root/'SMOKE_LEVER_RECEIPTS_PASS.json', receipt)
    verify_sealed(receipt, 'receipt_sha256')
    return receipt
