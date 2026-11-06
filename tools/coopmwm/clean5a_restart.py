"""Clean5a: speed-only expected-value cooperation on wz3to2 (MAIN GO 2026-11-06 19:54). Zero-physics prepare.

Design (REGISTRATION 19:54/19:57, FAIRAUDIT final, MAIN decisions to N4):
* base = MAPPO u0 FROZEN (trunk d9042cc5 + residual A_checkpoint_000 5ea22964); every arm binds this same base
  by path+sha. A = the frozen base evaluated as an arm every update (registered frozen_control_arm, no PPO);
  only B and R train their cooperation layer/trigger on that base.
* learner = FULLPOL clean5_ev (N2-only slate: abreast_wall + group_hold solo/pair), EV-BC ON from the S-REGIMEU0
  (u0 rows per cell) / S-REGIME (native fallback) guidance doc - arm B rows only; R gets no guidance terms
  (no-guidance arm). No verified bank/practice rows in either arm (registered EV-only guidance mode).
* G4 sparse synergy credit + G4LD extension (abreast_wall, group_hold, lane_drop_initiate); practice_offer ON
  (0.5; acts only on exact-plan demonstration slots, of which clean5a has none - disclosed).
* imitation epochs 20 (KL cap 0.045 unchanged); 6 staged training cells; 24 updates; u2/u4 probes,
  u6/u12/u24 replicated test (registration = the frozen Plan A u0-base contexts, metadata order, 19:22 rule);
  final evaluation = N6 launcher (claims 3a + 3c), in-run paired evaluations OFF.
* budget = the owner-approved 4M clean-training envelope (BUDGET_CLEAN4.json) after clean4's actual spend; the
  budget block keeps the runtime key ``clean4_budget`` (same envelope) with run='clean5a'.
* safety: A full rule; B/R composite report-only; any collision or teleport pauses.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

from .exact_grade import _seal
from .protocol import digest
from . import clean4_restart as c4

RUN = 'clean5a'
ROOT = '/home/ubuntu/coopmwm20261107_clean5a'
TRUNK = '/home/ubuntu/mappospeed/production2/policy_00169.pt'
TRUNK_SHA256 = 'd9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca'
CELL_KEYS = ('D2400_p40', 'D2400_p60', 'D2800_p40', 'D2800_p60', 'D3200_p40', 'D3200_p60')   # 6 staged cells
TRAIN_SEEDS = c4.TRAIN_SEEDS            # 989220-989267, 16 per update, rotating thirds
FINAL_EVAL_CEILING = 1_200_000          # MAIN 19:57: v2v_final cap (claims 3a + 3c ~1,020,960)
G4_FAMILIES = ['abreast_wall', 'group_hold', 'lane_drop_initiate']
IMITATION_EPOCHS = 20
QUOTE_FILE = 'CLEAN5A_QUOTE.json'
HOST_RECEIPTS = dict(
    cpuZ=dict(ready='/home/ubuntu/CPUZ_READY.json'),
    cpuV=dict(s0='/home/ubuntu/cpuv_setup_20261105/S0_READY.json', parity='/home/ubuntu/cpuv_parity_20261105/PARITY_PASS.json'))
DESIGN = ('A = frozen u0 base evaluated as an arm (no PPO); B/R identical code (FULLPOL clean5_ev N2-only slate, '
          'lane_drop features/moves/V2V 100 m, G2, G4 + G4LD extension, G5 vocabulary, practice offer); only the '
          'guidance differs: B = regime EV-BC (S-REGIMEU0 u0 rows, native fallback), R = none.')


def worlds_by_update():
    """16 TRAIN worlds per update over the 6 staged cells (cell rotates with the update), rotating seed thirds."""
    from . import map_scenarios as ms
    cells = {c['cell_key']: c for c in ms.cells()}
    out = {}
    for u in range(1, 25):
        offset = ((u-1) % 3)*16
        rows = []
        for k, seed in enumerate(TRAIN_SEEDS[offset:offset+16]):
            cell = cells[CELL_KEYS[(k+u) % len(CELL_KEYS)]]
            rows.append(ms.world(seed, cell['demand'], cell['penetration']))
        out[str(u)] = rows
    return out


def host_receipt(host):
    from . import launch_training as training
    spec = HOST_RECEIPTS[host]
    if host == 'cpuZ':
        ready = training.read(spec['ready'])
        if not (ready.get('pass_') is True and ready.get('status') == 'READY' and ready.get('host') == 'cpuZ'
                and ready['env']['tree_manifest']['mismatches_vs_cpuV'] == 0
                and ready['code_roots']['mismatches_vs_cpuV'] == 0):
            raise ValueError('cpuZ READY receipt (byte-identical to cpuV) required')
        return dict(host='cpuZ', receipt=spec['ready'], sha256=hashlib.sha256(Path(spec['ready']).read_bytes()).hexdigest(),
                    inventory_sha256=ready['env']['tree_manifest']['canonical_sha256_cpuZ'])
    setup, parity = training.read(spec['s0']), training.read(spec['parity'])
    if not setup['pass_'] or not parity['pass_']:
        raise ValueError('reuse qualified environment and exact SUMO parity')
    return dict(host='cpuV', receipt=spec['s0'], sha256=hashlib.sha256(Path(spec['s0']).read_bytes()).hexdigest(),
                inventory_sha256=setup['inventory_sha256'])


def check_guidance(cfg):
    """FAIRAUDIT: ASSERT (not just record) the guidance sources - FULLPOL EV-BC on, regime guidance doc with
    SHA-cited sources, rows for arm B only (R = the no-guidance arm), no verified bank rows."""
    from . import fullpol_ev, fullpol_learn
    slot = fullpol_ev.settings(cfg)
    learn = fullpol_learn.learn_settings(cfg)
    if slot is None or learn is None or not learn['ev_bc'] or learn.get('ev_credit'):
        raise ValueError('clean5a requires FULLPOL clean5_ev with EV-BC on and EV credit off')
    doc = slot.get('regime_guidance')
    source = (doc or {}).get('source') or {}
    sources = source.get('sources') or []
    if (not isinstance(doc, dict) or doc.get('regime_rule') != 'cell' or not sources
            or any(not isinstance(s, dict) or len(str(s.get('sha256', ''))) != 64 for s in sources)
            or len(str(source.get('sha256', ''))) != 64):
        raise ValueError('clean5a regime guidance must cite its S-REGIME/S-REGIMEU0 sources by sha256')
    rows = doc.get('rows') or []
    if not rows or any(r.get('arm') != 'B' for r in rows):
        raise ValueError('clean5a guidance rows must exist and be arm B only (R = no-guidance arm)')
    regimes = sorted({r['regime'] for r in rows})
    missing = sorted(set(CELL_KEYS)-set(regimes))
    if missing:
        raise ValueError('clean5a guidance must cover every training cell: missing '+repr(missing))
    mappo_cells = sorted({r['regime'] for r in rows if r.get('baseline') == 'mappo'})
    if cfg.get('mined_moves') or cfg.get('replay_sources'):
        raise ValueError('clean5a has no verified bank/mined guidance rows')
    return dict(rows=len(rows), table_sha256=source['sha256'], regimes=regimes, mappo_relative_cells=mappo_cells,
                native_fallback_cells=sorted(set(regimes)-set(mappo_cells)),
                sources=[dict(name=s.get('name'), sha256=s['sha256'], baseline=s.get('baseline')) for s in sources])


def keys(*, desc, guidance_doc, provenance):
    """Common B = R keys for clean5a (registered values)."""
    from . import lane_drop, fullpol_ev, sparse_synergy_credit as ssc, practice_offer, bank_breadth
    from .track2_backend import FROZEN_A_RULE
    out = {}
    out.update(lane_drop.config_keys(provenance=provenance, descriptor=desc, features=True, moves_on=True,
                                     v2v_range_m=c4.V2V_RANGE_M, member_cap=3))
    out.update(fullpol_ev.recommended(provenance=provenance, regime_guidance=guidance_doc))
    out['g4_extension'] = ssc.extension_keys(provenance=provenance, families=G4_FAMILIES)
    out['g4_start_cap'] = ssc.start_cap_keys(provenance=provenance)   # 12/9/8 by arity from the 230,400 cap
    out['practice_offer'] = practice_offer.config_keys(fraction=0.5, provenance=provenance)
    out['imitation_epochs'] = IMITATION_EPOCHS
    out['frozen_control_arm'] = FROZEN_A_RULE
    out['guidance_mode'] = bank_breadth.EV_ONLY_RULE
    return out


def prepare(source, *, contexts, guidance_doc, quote_doc, approval, prior_runtime_sim_s, host, provenance,
            root=ROOT, clean2=c4.CLEAN2, u0=c4.U0, u0_sha256=c4.U0_SHA256, trunk=TRUNK, trunk_sha256=TRUNK_SHA256):
    """Zero physics. contexts = the frozen Plan A u0-base B contexts (registration, metadata order)."""
    import torch
    from . import composite_health, launch_preparation as stage, launch_clean2 as c2
    from . import launch_production as production, launch_qualification as qualification
    from . import launch_training as training, rerun_cpus as rc, track2_backend as p1, g5_runtime
    from . import map_descriptor as md, frozen_trunk
    from .bank_breadth import ev_only
    from .track2_monitor import STATE_ORDER_RULE
    root, source, clean2 = Path(root), Path(source), Path(clean2)
    for path in (root, source, clean2):
        rc.guard_path(path); rc.guard_path(path.resolve())
    if root.exists():
        raise ValueError('new clean5a root required; never overwrite')
    if hashlib.sha256(Path(u0).read_bytes()).hexdigest() != u0_sha256:
        raise ValueError('staged MAPPO u0 residual identity differs')
    if frozen_trunk.file_sha256(trunk) != trunk_sha256:
        raise ValueError('frozen MAPPO trunk identity differs')
    desc = md.current()
    if desc.name != 'wz3to2' or desc.network_file_sha256 != c4.NET_FILE_SHA256:
        raise ValueError('process map must be the staged wz3to2 (COOPMWM_MAP_SPEC)')
    if not g5_runtime.enabled():
        raise ValueError('clean5a requires the bound G5 runtime env (COOPMWM_G5_RUNTIME=1)')
    host_rec = host_receipt(host)
    source_manifest = training.read(source/'SOURCE_MANIFEST.json')
    for name, sha in source_manifest['files'].items():
        if rc.sha(source/name) != sha:
            raise ValueError('immutable clean5a source differs: '+name)
    cfg = c4.levers(training.read(clean2/'CONFIG.json'), lane_drop_keys={}, v2v_keys={}, w2_keys={})
    cfg.update(run_root=str(root), accounting_root=str(root), bank_source_root=str(root),
        source_root=str(source), source_manifest_sha256=digest(source_manifest),
        gate_attempt=1, smoke_attempt=1, resume_update=0, A_resume_update=0,
        worlds_by_update=worlds_by_update(), replay_sources={}, paired_evaluation=False, extra_final_baselines=[],
        checkpoint=trunk)
    cfg.update(keys(desc=desc, guidance_doc=guidance_doc, provenance=provenance))
    cfg['control_arm'] = dict(arm='A', root=str(root), update=24)
    cfg['a_results']['root'] = str(root)
    cfg['single_ledger']['root'] = str(root)
    cfg['augmentation_prelaunch']['ledger_root'] = str(root)
    cfg['plan_fix_ready'] = str(root/'PLAN_FIX_DISABLED.json')
    cfg.pop('clean2_restart'); cfg.pop('clean2_budget')
    cfg['safety_reference'] = dict(by_world={}, unmatched_zero_event_benchmark=True,
        source='clean5a: no registered wz3to2 MAPPO check panel; unmatched TRAIN zero-event benchmark (conservative)')
    cfg['newmap'] = dict(run=RUN, map='wz3to2', base_controller='mappo_frozen', cell=list(CELL_KEYS),
        eval_panel_sha256=c4.EVAL_PANEL_SHA256, descriptor_sha256=desc.digest, network_file_sha256=c4.NET_FILE_SHA256,
        map_spec=c4.MAP_SPEC, final_eval_ceiling_sim_s=FINAL_EVAL_CEILING, v2v_range_m=c4.V2V_RANGE_M, rsu=False,
        base=dict(trunk=trunk, trunk_sha256=trunk_sha256, residual=u0, residual_sha256=u0_sha256,
                  rule='every arm (A/B/R, BASE, decision branches) binds this same frozen base by path+sha'),
        host=host_rec, design=DESIGN, practice_offer_note='practice_offer ON (0.5) for the guidance treatment; it acts '
        'only on exact-plan demonstration slots, of which clean5a has none (EV-BC only), so it is inert - disclosed')
    cfg['registered_state_order'] = STATE_ORDER_RULE
    cfg['g5_runtime'] = dict(flag=g5_runtime.FLAG, base=str(g5_runtime.BASE_REL),
        base_sha256=g5_runtime.BASE_SHA256, overlay_sha256=g5_runtime.OVERLAY_SHA256,
        rule='accepted G5 env change bound onto the frozen e3final env; wz3to2 bound by map_runtime.install')
    if cfg['plan_augment']['enabled'] or cfg.get('launch_augmentation_decision') != rc.AUGMENTATION_OFF_DECISION:
        raise ValueError('owner symmetric augmentation-OFF design required')
    if not ev_only(cfg):
        raise ValueError('clean5a requires the registered EV-only guidance mode')
    guidance = check_guidance(cfg)
    rc.check_pins(cfg)
    budget = c4.budget(quote_doc=quote_doc, approval=approval, prior_runtime_sim_s=prior_runtime_sim_s)
    budget = _seal(dict({k: v for k, v in budget.items() if k != 'receipt_sha256'}, run=RUN, quote_file=QUOTE_FILE),
                   'receipt_sha256')
    rule = training.read(clean2/rc.RULE_FILE)
    rc.init_root(root, amendment_shas=[rule['amendment_sha256']], track2=True)
    training.write(root/'PREPARATION_STARTED.json', dict(unix=time.time(), source_root=str(source), run=RUN,
        guidance=guidance, host=host_rec, new_sim_s=0))
    for name, value in ((QUOTE_FILE, quote_doc), ('BUDGET_CLEAN4.json', approval)):
        (root/name).write_text(json.dumps(value, indent=1, sort_keys=True), encoding='utf-8')
    stage.copy_equal(clean2/rc.RULE_FILE, root/rc.RULE_FILE)
    safety_receipts = {}
    for arm, rule_name, pause_name in (('B', composite_health.B_RULE, composite_health.B_PAUSE),
                                       ('R', composite_health.R_RULE, composite_health.R_PAUSE)):
        if not composite_health.report_only(clean2, arm):
            raise ValueError('registered owner report-only rule missing: '+arm)
        for name in (rule_name, pause_name):
            stage.copy_equal(clean2/name, root/name)
            safety_receipts[name] = rc.sha(root/name)
        with (root/'AUTHORITY.md').open('a', encoding='utf-8') as f:
            f.write('\nclean5a owner safety rule for '+arm+': composite report-only (inherited owner rule); any '
                    'collision/teleport pauses. A keeps the full rule.\n')
    for arm in ('B', 'R'):
        (root/'replay_bank'/arm).mkdir(parents=True)
    for arm in ('A', 'B', 'R'):
        stage.copy_equal(u0, root/f'{arm}_checkpoint_000.pt')
        if rc.sha(root/f'{arm}_checkpoint_000.pt') != cfg['common_checkpoint_sha256']:
            raise ValueError('MAPPO u0 identity differs')
    cfg['clean4_budget'] = budget
    cfg['clean4_restart'] = dict(owner_quote=RUN, clean2_root=str(clean2), inherited_safety_receipts=safety_receipts,
        inherited_safety=c4.SAFETY_RULE, source_commit=source_manifest['commit'], guidance=guidance,
        registration_contexts=len(contexts))
    native = production.backend_config(cfg)
    for arm in ('B', 'R'):
        torch.manual_seed(cfg['training_seed'])
        runtime = p1.ArmRuntime(dict(native, initial_sidecars_required=False), arm, 1)
        with (root/f'{arm}_track2_000.pt').open('xb') as f:
            torch.save(c4.sidecar_state(cfg, arm, runtime), f)
    registration = c4.registration_from_contexts(contexts, root)
    training.write(root/'ISOLATED_REGISTRATION.json', registration)
    quotes = c2.full_diagnostic_quotes(cfg)
    gate_pairs = qualification.gate_pairs(cfg)
    gate_manifest = [dict(id=s['verification_id'], sim_s=s['sim_s'], window=s['window'],
        snapshot_sha256=s['snapshot_sha256']) for s, _ in gate_pairs]
    gate_cost = sum(s['sim_s'] for s in gate_manifest)
    phases = budget['phases']
    if gate_cost > phases['gate_bank_verification_ext']:
        raise ValueError('full-window gate exceeds its phase')
    diag = {k: q['quoted_sim_s'] for k, q in quotes.items()}
    if (diag['probe_u002']+diag['probe_u004'] > phases['early_probes'] or
            sum(diag[f'replicated_u{u:03}'] for u in (6, 12, 24)) > phases['isolated_replicated']):
        raise ValueError('native diagnostic manifests exceed clean5a phases')
    reporting = [s for s in gate_manifest if __import__('re').match(r'clean1_gate_[BR]_s000_', s['id'])]
    if not reporting or cfg['smoke']['total_sim_s']+sum(s['sim_s'] for s in reporting) > 20000:
        raise ValueError('full first-state B/R smoke reporting plus episodes exceeds smoke cap')
    cfg['clean2_smoke_reporting_quote'] = dict(selection='first prospectively registered state, B/R group/solos/common-base; full windows; descriptive only',
        branches=reporting, sim_s=sum(s['sim_s'] for s in reporting),
        complete_smoke_sim_s=cfg['smoke']['total_sim_s']+sum(s['sim_s'] for s in reporting))
    cfg['clean4_restart'].update(diagnostic_quotes=diag, gate_sim_s=gate_cost,
                                 lesson_quotes={f'lesson_u{u:03}': 0 for u in (6, 12, 18, 24)})
    actual = set(root/name for name in safety_receipts)
    with rc._ReadRecorder([root], Path('/home/ubuntu')) as recorder:
        native = production.backend_config(cfg)
        for arm in ('B', 'R'):
            runtime = p1.ArmRuntime(native, arm, 1)
            sup = production.bank_move_supervision(runtime)
            if sup.get('contexts'):
                raise ValueError('clean5a has no verified bank supervision by design')
    actual.update(recorder.paths); actual.update(recorder.lexical)
    for row in registration['replicates']:
        for item in row['states']:
            actual.update(Path(item[k+'_path']) for k in ('snapshot', 'receipt'))
    actual.update(Path(p) for p in (u0, trunk))
    entries = {}
    for path in sorted(actual):
        rc.guard_path(path); rc.guard_path(path.resolve())
        if not path.is_file():
            raise ValueError('required clean5a reader input missing: '+str(path))
        entries[str(path)] = dict(sha256=rc.sha(path), bytes=path.stat().st_size, resolved_path=str(path.resolve()))
    manifest = _seal(dict(schema='coopmwm.clean5a.reader_staging.v1', entries=entries, host=host_rec), 'manifest_sha256')
    training.write(root/'STAGING_MANIFEST.json', manifest)
    cfg.update(staging_manifest_sha256=rc.sha(root/'STAGING_MANIFEST.json'),
               isolated_registration_sha256=rc.sha(root/'ISOLATED_REGISTRATION.json'))
    training.write(root/'CLEAN5A_BUDGET.json', budget)
    training.write(root/'CONFIG.json', cfg)
    approval_doc = dict(approved=True, owner_quote=RUN, config_sha256=digest(cfg),
        source_manifest_sha256=cfg['source_manifest_sha256'], budget_receipt_sha256=budget['receipt_sha256'],
        created=datetime.now().astimezone().isoformat(),
        authority='owner 14:58 4M clean-training envelope (BUDGET_CLEAN4.json); MAIN clean5a GO 19:54 (REGISTRATION); '
                  'launch requires FULLPOL CPU suites + N5 review + N11 integration PASS + qualification + smoke + gate')
    training.write(root/'OWNER_APPROVAL.json', approval_doc)
    training.write(root/'AUGMENTATION_DISABLED.json', _seal(dict(config_sha256=digest(cfg),
        admitted=dict(B=0, R=0), owner_decision=rc.AUGMENTATION_OFF_DECISION), 'receipt_sha256'))
    with (root/'AUTHORITY.md').open('a', encoding='utf-8') as f:
        f.write('\n'+json.dumps(dict(owner_approval=approval_doc, budget=budget, quote_sha256=quote_doc['receipt_sha256'],
            newmap=cfg['newmap'], eval_panel=c4.EVAL_PANEL, design=DESIGN, guidance=guidance,
            g4=dict(extension=cfg['g4_extension'], start_cap=cfg['g4_start_cap'], disclosed='budget-derived start caps 12/9/8 (from the 230,400 cap); 1-in-16 '
                    'sampling and the 10,000 veh-s scale are inherited/hand-set G4 registrations'),
            imitation=dict(epochs=IMITATION_EPOCHS, kl_cap=0.045),
            seeds='TRAIN worlds 989220-989267 over 6 staged cells (16/update); final panel 989600-989695 + 3c (N6)',
            criterion=rc.criterion_document(), reporting=cfg['clean2_reporting_definition'],
            paired_evaluation='in-run u8/u24 OFF; final = N6 launcher (claims 3a/3c), debited from v2v_final',
            registered_state_order=c4.STATE_ORDER_RULE_TEXT, safety=c4.SAFETY_RULE), sort_keys=True)+'\n')
    training.checked_config(root/'CONFIG.json')
    qualification.verify_inputs(cfg)
    training.write(root/'PREPARED.json', dict(status='STAGED_NOT_LAUNCHED', run=RUN, config_sha256=digest(cfg),
        files=len(entries), remaining_cap_sim_s=budget['remaining_sim_s'], guidance=guidance, new_sim_s=0))
    return cfg
