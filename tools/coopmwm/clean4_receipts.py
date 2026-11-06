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
    if (cfg.get('newmap') or {}).get('run') in ('clean5a', 'clean5b'):
        return validate_clean5a(cfg, updates, fixes)
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
            raise ValueError('RSU-initiated suggestions must be OFF in clean4 (MAIN registration)')
        out['N2'][arm] = row
    # MAIN: N9 not merged / OFF; group_hold is not a runtime proposal family.
    for arm in ('B', 'R'):
        offered = [f for f, n in _proposed(arms[arm]).items() if n and f in c4.FAMILIES_OFF]
        if offered:
            raise ValueError('OFF family offered at runtime: '+arm+' '+repr(offered))
    if cfg.get('nm_families'):
        raise ValueError('N9 families must be absent from the clean4 config')
    if not cfg.get('mined_moves'):
        raise ValueError('clean4 launches only with Plan A guidance rows')
    out['mined'] = mined_check(cfg, updates, root/'smoke')
    out['supervision'] = supervision_check(arms)
    out['guidance_rows'] = cfg['clean4_restart']['guidance_rows']
    if fixes is None:
        return dict(status='LEVER_PRECHECK_ONLY', **out)
    receipt = _seal(dict(status='LEVER_RECEIPTS_PASS', levers=out, fix_receipts_sha256=fixes['receipt_sha256'],
        smoke_updates={a: arms[a]['update_sha256'] for a in arms}), 'receipt_sha256')
    training.write(root/'SMOKE_LEVER_RECEIPTS_PASS.json', receipt)
    verify_sealed(receipt, 'receipt_sha256')
    return receipt


FORBIDDEN_POSITIVE = ('none', 'wall', 'ease_off', 'wall_initiate', 'wall_join')


def _lane_drop_family(label, family):
    return label == family or label == 'lane_drop_initiate:'+family


def supervision_check(arms):
    """LEARNAUDIT pre-launch assert on the smoke's real bank supervision: B positive contexts cover BOTH
    lane-drop families (direct or via the N2 initiate protocol) and no positive context is a legacy label;
    R has exactly zero guidance contexts (option a)."""
    from collections import Counter
    out = {}
    for arm in ('B', 'R'):
        contexts = arms[arm]['learning']['bank_trigger'].get('contexts') or []
        positives = Counter(c['family'] for c in contexts if c.get('family') != 'wall_negative')
        out[arm] = dict(positive_family_counts=dict(positives), contexts=len(contexts))
    b = out['B']['positive_family_counts']
    for family in ('closing_pair', 'abreast_wall'):
        if not any(_lane_drop_family(label, family) for label in b):
            raise ValueError('LEARNAUDIT: B supervision lacks lane-drop family '+family+': '+repr(b))
    bad = sorted(label for label in b if label in FORBIDDEN_POSITIVE)
    if bad:
        raise ValueError('LEARNAUDIT: B positive contexts carry legacy labels (projection): '+repr(bad))
    if out['R']['contexts'] != 0:
        raise ValueError('option (a): R must have zero guidance contexts')
    return out


def mined_check(cfg, updates, artifacts):
    """Owner exposure (a)-(d) for B's Plan A rows (as clean3), every row bound with ZERO legacy projection
    (exact N2 LaneDropMove, ON family), and R zero guidance (option a: no rows, no practice starts)."""
    from . import bank_breadth, launch_production as production, lane_drop, clean4_restart as c4
    from .clean3_mined import examples, observability, rows
    native = production.backend_config(cfg, root=artifacts, smoke=True)
    arms = {u['arm']: u for u in updates}
    if any(r['arm'] != 'B' for r in rows(native)):
        raise ValueError('option (a): clean4 guidance rows are B only')
    found, wanted = examples(native, 'B')
    if not wanted or [e for e in found if not e['executable']] or             {(e['verification_id'], e['plan_sha256']) for e in found} != wanted:
        raise ValueError('(a) every admitted Plan A row must load as an executable B bank example')
    projected = []
    def consume(*, cfg, receipt, example, plan, pairs, trace):
        move = bank_breadth.mined_move(plan)
        if not isinstance(move, lane_drop.LaneDropMove) or move.family not in c4.FAMILIES_ON:
            projected.append((receipt['verification_id'], type(move).__name__, getattr(move, 'family', None)))
    bank_breadth.visit(native, 'B', consume)
    if projected:
        raise ValueError('LEARNAUDIT zero legacy projection violated: '+repr(projected[:5]))
    pool = bank_breadth.practice_pool(native, 'B')
    demo_ctx = {r['context_id']: r for r in pool if r.get('demonstrate')}
    contexts = {(c['verification_id'], c['example_id']) for c in arms['B']['learning']['bank_trigger'].get('contexts', [])}
    targets = {(e['verification_id'], e['example_id']) for e in found}
    every = {(r['verification_id'], r['example_id']): r['context_id']
             for r in bank_breadth.practice_pool(native, 'B', deduplicate=False) if r.get('mined')}
    pool_ids = {r['context_id'] for r in pool}
    if not targets <= set(every) or any(every[t] not in pool_ids for t in targets)             or len({every[t] for t in targets}) != len(targets):
        raise ValueError('(b) Plan A seed situations missing from the B practice pool')
    started = [e for e in arms['B']['practice']['episodes']
               if (e.get('breadth') or {}).get('context_id') in demo_ctx and demo_ctx[e['breadth']['context_id']].get('mined')]
    if not started:
        raise ValueError('(b) smoke practice did not start from a Plan A seed situation: B')
    demos = arms['B']['clean3_mined_usage']['demonstrations']
    passed = [d for d in demos if d['executed'].get('passed')]
    if not passed:
        raise ValueError('(c) no demonstrated Plan A move executed its verified profile: B')
    if not {d['family'] for d in passed} & set(native.get('demonstrate_families') or ()):
        raise ValueError('(c) demonstrated family must be a Plan A family: B')
    credit = arms['B']['clean3_mined_usage']['g4_credit']
    if not credit:
        raise ValueError('(d) G4-format measured synergy rows missing for demonstrated moves: B')
    r_found, r_wanted = examples(native, 'R')
    r_practice = [e for e in arms['R']['practice']['episodes'] if e.get('breadth')]
    if r_found or r_wanted or bank_breadth.practice_pool(native, 'R') or r_practice:
        raise ValueError('option (a): R must have zero guidance rows, pool and practice starts')
    return dict(B=dict(examples=len(found), observability=observability(found, contexts),
                       imitation_targets=len(targets & contexts), practice_starts=len(started),
                       demonstrations=[{k: d[k] for k in ('id', 'family', 'executed')} for d in demos],
                       g4_credit=len(credit)),
                R=dict(examples=0, practice_starts=0, rule=c4.R_GUIDANCE))


CLEAN5A_ALLOWED = ('abreast_wall', 'group_hold', 'lane_drop_initiate', 'lane_drop_join')
CLEAN5B_PROGRAMS = ('free_program_initiate', 'free_program_join')      # LCEXEC per-CAV program options (clean5b)


def validate_clean5a(cfg, updates, fixes=None):
    """clean5a smoke receipts (registered before the clean5a smoke): map/base binding, safety rule, G2, G4 + G4LD
    extension, N2-only FULLPOL slate (solo/pair hold + abreast wall offered in B and R, no legacy/W1/N9 option),
    EV-BC guidance fitted for B and absent for R, frozen A (no PPO), no verified bank rows, trunk binding."""
    from . import composite_health, coordination_opportunities as co, g4_runtime as g4, group_lane
    from . import launch_production as production, launch_training as training, clean4_restart as c4
    from . import sparse_synergy_credit as ssc, frozen_trunk, clean5a_restart as c5
    from .bank_breadth import ev_only
    root = Path(cfg['run_root'])
    arms = {u['arm']: u for u in updates}
    run = cfg['newmap']['run']
    allowed = CLEAN5A_ALLOWED+(CLEAN5B_PROGRAMS if run == 'clean5b' else ())
    out = dict(run=run)
    c4.check_map(cfg)
    if run == 'clean5b':
        # clean5b (MAIN 23:40/23:48): LCEXEC programs offered to B and R identically, never BC/EV targets,
        # G4 off for programs (checked below via the clean5a G4LD families), LC runtime bound in this process.
        from . import lc_families, lc_runtime, clean5b_restart as c5b
        lc = cfg.get('lc_families') or {}
        if (not lc_runtime.enabled() or lc_families.settings(cfg) is None or lc.get('cap') != c5b.LC_CAP
                or (lc.get('generator_args') or {}).get('templates') != []):
            raise ValueError('clean5b requires the LC runtime and lc_families cap 6 with no templates (B = R)')
        if (cfg.get('clean5_ev') or {}).get('hold_lane_pin') is not True:
            raise ValueError('clean5b launches with clean5_ev.hold_lane_pin ON (MAIN 2026-11-07)')
        out['lc_families'] = dict(cap=lc['cap'], generator=lc.get('generator'), templates=0, hold_lane_pin=True)
    from . import fullpol_fair, fullpol_ev as _fe
    fair = fullpol_fair.settings(cfg)
    if fair is not None:
        # clean5b-F (owner 04:0x): wall_h banned; EV targets fairness-eligible only; reward term from the logger
        if (run != 'clean5b' or not fair.get('reward') or not fair.get('ev_filter')
                or fair.get('solo_variants') != ['wall_g']):
            raise ValueError('clean5b-F needs the registered fair slot (reward + ev_filter, wall_g solos only)')
        out['fair'] = dict(rule=fair['rule'], threshold_s=fair['threshold_s'], parity=sorted(_fe.FAIR_PARITY))
    base = cfg['newmap']['base']
    if (frozen_trunk.file_sha256(base['trunk']) != base['trunk_sha256'] or cfg['checkpoint'] != base['trunk']
            or frozen_trunk.check_process(cfg) is not None):
        raise ValueError('clean5a frozen base binding differs (trunk path/sha or an unexpected trunk shim)')
    out['base'] = dict(trunk_sha256=base['trunk_sha256'], residual_sha256=base['residual_sha256'])
    for arm in ('B', 'R'):
        if not composite_health.report_only(root, arm):
            raise ValueError('registered B/R report-only safety rule missing: '+arm)
    native = production.backend_config(cfg)
    if co.settings(native) is None:
        raise ValueError('G2 coordination opportunities must be enabled')
    if not g4.enabled(native, 1) or native['budget']['phases'].get('g4_synergy_credit') != 230400:
        raise ValueError('G4 must be bound from u1 with its 230400 phase')
    ext = ssc.extension(native)
    if ext is None or ext['families'] != sorted(c5.G4_FAMILIES):
        raise ValueError('G4LD extension must be bound with the registered clean5a families')
    if g4.enabled(production.backend_config(cfg, root=root/'smoke', smoke=True), 1, smoke=True):
        raise ValueError('G4 must stay disabled inside smoke (TRAIN-only sampling)')
    if not cfg.get('g4_start_cap') or ssc.start_cap(native) is None:
        raise ValueError('clean5a G4 budget-derived start cap must be bound')
    out['G4'] = dict(phase_sim_s=230400, extension=ext['families'], start_cap=cfg['g4_start_cap']['rule'])
    if not group_lane.enabled(cfg) or len(group_lane.families(cfg)) != 7:
        raise ValueError('G5 seven-family lane vocabulary must stay registered')
    if not ev_only(cfg) or cfg.get('mined_moves'):
        raise ValueError('clean5a must run the registered EV-only guidance mode without mined rows')
    out['slate'], out['ev_bc'] = {}, {}
    for arm in ('B', 'R'):
        proposed = {f: n for f, n in _proposed(arms[arm]).items() if n}
        base_family = {f.split(':', 1)[0]: n for f, n in proposed.items()}
        illegal = sorted(f for f in base_family if f not in allowed)
        if illegal:
            raise ValueError(run+' slate offered an unregistered option: '+arm+' '+repr(illegal))
        if run == 'clean5b' and not base_family.get('free_program_initiate'):
            raise ValueError('clean5b slate must offer LCEXEC free_program moves: '+arm+' '+repr(proposed))
        if not base_family.get('group_hold') or not (base_family.get('abreast_wall') or base_family.get('lane_drop_initiate')):
            raise ValueError('clean5a slate must offer group_hold and abreast_wall moves: '+arm+' '+repr(proposed))
        row = (arms[arm]['metrics'].get('move_proposals') or {}).get('coordination_opportunities')
        if not row or row.get('decisions', 0) <= 0:
            raise ValueError('G2 measured opportunity receipt required: '+arm)
        out['slate'][arm] = proposed
        fit = arms[arm]['learning'].get('fullpol') or {}
        if arm == 'B' and fit.get('records', 0) <= 0:
            raise ValueError('B must fit EV-BC records in the smoke')
        if arm == 'R' and fit.get('records', 0) != 0:
            raise ValueError('R is the no-guidance arm: zero EV-BC records')
        out['ev_bc'][arm] = {k: fit.get(k) for k in ('records', 'ev_bc', 'report_bc_kl', 'report_kl_unlabelled')}
        if run == 'clean5b':
            # programs are never BC/EV targets: every fitted record is an EV-BC solo-hold record
            ev_rec = fit.get('ev_bc') or {}
            fams = list(ev_rec.get('families') or [])
            ok_family = ((lambda f: f in _fe.FAIR_PARITY) if fair is not None
                         else (lambda f: __import__('re').fullmatch(r'wall_[gh]_solo[0-2]', f) is not None))
            if (fit.get('bank') not in (None, {}) or fit.get('records', 0) != ev_rec.get('records', 0)
                    or any(not ok_family(f) for f in fams)):
                raise ValueError('clean5b BC/EV targets must be solo holds only (no program BC): '+arm)
    if cfg.get('imitation_epochs') != c5.IMITATION_EPOCHS:
        raise ValueError(f'clean5a registered imitation epochs = {c5.IMITATION_EPOCHS}')
    # MAIN ~21:1x registered pace: EV-BC 60 steps; trigger-PPO KL guard = per-CAV-row mean (B and R identically)
    from . import fullpol_ev
    ev = fullpol_ev.settings(cfg)
    if ev.get('ev_bc_steps') != fullpol_ev.EV_BC_STEPS or ev.get('ppo_kl') != 'row_mean':
        raise ValueError('clean5a registered pace: clean5_ev.ev_bc_steps 60 and ppo_kl row_mean')
    out['pace'] = dict(ev_bc_steps=ev['ev_bc_steps'], ppo_kl=ev['ppo_kl'])
    for arm in ('B', 'R'):
        receipt = arms[arm]['learning'].get('fullpol_ppo_kl') or {}
        if receipt.get('rule') != 'row_mean':
            raise ValueError('trigger-PPO row-mean KL receipt required: '+arm)
        out['pace'][arm] = {k: receipt.get(k) for k in ('final_kl', 'accepted', 'rejected')}
    for arm in ('B', 'R'):   # ONE trust region: legacy bank imitation is off (0 contexts, 0 steps)
        learning = arms[arm]['learning']
        if (learning['bank'].get('pairs_total', 0) or learning['bank'].get('optimizer_steps', 0)
                or learning['bank_trigger'].get('contexts')):
            raise ValueError('legacy bank imitation must be OFF in clean5a: '+arm)
    if fair is not None:
        # the fairness reward read REAL logger output: A wrote its group file, B/R attached groups and were penalised
        a_log = arms['A'].get('fair_vehicle_log') or {}
        if not a_log.get('worlds') or not Path(a_log.get('path', '/nonexistent')).is_file():
            raise ValueError('clean5b-F: A per-vehicle group summaries missing in the smoke')
        out['fair']['A_worlds'] = a_log['worlds']
        for arm in ('B', 'R'):
            log = arms[arm].get('fair_vehicle_log') or {}
            receipt = arms[arm]['learning'].get('fullpol_fair') or {}
            if not log.get('episodes') or receipt.get('rule') != fullpol_fair.RULE or not receipt.get('episodes'):
                raise ValueError('clean5b-F: fairness reward receipt from logger output required: '+arm)
            out['fair'][arm] = dict(episodes=log['episodes'], unpaired=log['unpaired'],
                                    harm_veh_s=receipt.get('total_harm_veh_s'))
    ppo = arms['A']['learning'].get('ppo') or {}
    if ppo.get('status') != 'frozen_base_no_update':
        raise ValueError('A must be the frozen base evaluation arm (no PPO)')
    out['A'] = ppo
    if fixes is None:
        return dict(status='LEVER_PRECHECK_ONLY', **out)
    receipt = _seal(dict(status='LEVER_RECEIPTS_PASS', levers=out, fix_receipts_sha256=fixes['receipt_sha256'],
        smoke_updates={a: arms[a]['update_sha256'] for a in arms}), 'receipt_sha256')
    training.write(root/'SMOKE_LEVER_RECEIPTS_PASS.json', receipt)
    return receipt
