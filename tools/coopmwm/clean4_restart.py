"""Clean4: wz3to2 lane-drop training run (A/B/R from frozen speed-only MAPPO u0, 24 updates).

Zero-physics preparation and the clean4 budget binding. Physics happens only in the existing
smoke/gate/training/diagnostic code paths (launch_qualification / launch_training), driven by the
one-use ops/clean4_chain.py. Design (MAIN/owner 2026-11-06, REGISTRATION.md 14:37-16:50):

* map wz3to2 (N1 descriptor, net raw sha b6277898), cell D3200_p60, base = frozen speed-only MAPPO u0
  (N1 sanity 16:22), 1200-s episodes, control 300-1200 s;
* A = plain MAPPO PPO; B and R identical code (G2 opportunities, G4 sparse synergy credit, G5 lanes,
  W1 V2V wall initiate/join, W2 pressure/release/negatives, N2 lane-drop moves + V2V + RSU) - only the
  guidance source differs: B = Plan A SUMO-verified examples, R = Plan A re-verified random-procedure
  examples (same per-move procedure; counts reported, never padded);
* safety: A full rule; B/R composite report-only; any collision or teleport pauses;
* replicated synergy test u6/u12/u24 on registered TRAIN states (Plan A contexts, metadata order);
* final paired evaluation is N6's launcher on the registered final panel (3acb2b56..., 96 worlds
  989600-989695, D3200_p60), debited from v2v_final; the in-run u8/u24 paired evaluations are OFF.
"""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

from .exact_grade import _seal, verify_sealed
from .protocol import digest

ROOT = '/home/ubuntu/coopmwm20261106_clean4_r1'
SKELETON = '/home/ubuntu/coopmwm20261106_clean4'          # N10 staging (inputs/ read-only)
CLEAN2 = '/home/ubuntu/coopmwm20261105_clean2'            # template CONFIG + owner safety rule files
MAP_SPEC = SKELETON+'/inputs/map/wz3to2/MAP_SPEC.json'
NET_FILE_SHA256 = 'b62778986347e8c21bca90d7ddcc8d9faeb288290a4dbdc3f53ee64ae8c21dfa'
U0 = SKELETON+'/inputs/mappo_u0/A_checkpoint_000.pt'
U0_SHA256 = '5ea2296470c2af2855f9e292b03eeaa62bafdfec08146712d7347d7ce117d734'
SCHEMA = 'coopmwm.clean4.budget.v1'
OWNER_CAP = 12_400_000
APPROVED = 4_000_000
CELL_KEY = 'D3200_p60'
EVAL_PANEL_SHA256 = '3acb2b565fcb61f4978834b3ba47f0b574fe2d5da7ca028210ef75830c19a130'
EVAL_PANEL = dict(sha256=EVAL_PANEL_SHA256, worlds='96 worlds 989600-989695', cell=CELL_KEY, episode_s=1200,
                  owner='N6 ops/nm_finaleval.py (c4/nm-finaleval)', phase='v2v_final', ceiling_sim_s=672_000)
# Training worlds: 48 fresh staged TRAIN seeds in the clean4 cell, 16 per update, rotating thirds
# (clean3 batch rule); no mining/sanity seed (989200-989216 N3/PLANGEN, 989351+ LOWCAV, 989400+ N8,
# 989596-989599 sanity).
TRAIN_SEEDS = tuple(range(989220, 989268))
EPISODES_PER_UPDATE = 16
SAFETY_RULE = ('A: full original rule. B and R: composite TTC/critical comparison reported each update, no '
               'pause. Any collision or teleport in any arm pauses. (owner 2026-11-06 02:44 "1"; clean4 spec 14:37)')
FAMILIES_ON = ('closing_pair', 'abreast_wall', 'lane_balance', 'early_merge', 'zipper_order')
FAMILIES_OFF = ('platoon_headway', 'harmonize_rows', 'discharge_sync')
FAMILY_RULE = ('MAIN 2026-11-06: B/R move families ON = abreast walls, closing-lane pairs (N2) and, if N9 is '
               'merged, lane_balance / early_merge / zipper_order; OFF = platoon_headway, harmonize_rows, '
               'discharge_sync. The clean3 vocabulary (wall/easing/gap_open/ramp_assist/G5 lane families) stays '
               'on unchanged for B and R (WM lane 2 = closing lane plays the ramp role).')
V2V_RANGE_M = 200.0


def worlds_by_update():
    from . import map_scenarios as ms
    cell = next(c for c in ms.cells() if c['cell_key'] == CELL_KEY)
    out = {}
    for u in range(1, 25):
        offset = ((u-1) % 3)*EPISODES_PER_UPDATE
        out[str(u)] = [ms.world(s, cell['demand'], cell['penetration'])
                       for s in TRAIN_SEEDS[offset:offset+EPISODES_PER_UPDATE]]
    return out


# ---------------------------------------------------------------- budget

def budget(*, quote_doc, approval, prior_runtime_sim_s):
    """Sealed clean4 budget registration bound to the owner-approved quote (BUDGET_CLEAN4.json)."""
    from . import clean4_budget as b
    bound = b.approved(approval, quote_doc)
    phases = dict(quote_doc['runtime_phases'])
    if any(type(v) is not int or v < 0 for v in phases.values()):
        raise ValueError('integer phase allocations required')
    # The clean4 root ledger may spend at most the approved envelope minus the phases charged on
    # their own ledgers (base sanity, Plan A) - their allocations are protected as prior.
    remaining = APPROVED-prior_runtime_sim_s
    if sum(phases.values()) > remaining:
        raise ValueError('clean4 runtime phases exceed the approved envelope after sanity/Plan A')
    return _seal(dict(schema=SCHEMA, owner_cap_sim_s=OWNER_CAP, approved_sim_s=APPROVED,
        quote_sha256=quote_doc['receipt_sha256'], quote_total_sim_s=quote_doc['total_training_sim_s'],
        owner_approval=bound['owner_approval'], phases=phases, own_ledger_prior_sim_s=prior_runtime_sim_s,
        protected_prior_sim_s=OWNER_CAP-APPROVED+prior_runtime_sim_s, remaining_sim_s=remaining,
        smoke_gate=quote_doc['smoke_gate'],
        rule='owner 14:58 "yes, approve the 4M budget"; re-quotes <=4M with >=15% per-phase margin '
             '(BUDGET_CLEAN4.json requote_rule); smoke+gate inside newmap2_smoke <=40,000'), 'receipt_sha256')


def checked(cfg):
    value = cfg['clean4_budget']
    verify_sealed(value, 'receipt_sha256')
    if (value['schema'] != SCHEMA or value['owner_cap_sim_s'] != OWNER_CAP or value['approved_sim_s'] != APPROVED
            or sum(value['phases'].values()) > value['remaining_sim_s']):
        raise ValueError('clean4 budget registration differs')
    return deepcopy(value)


def native_budget(cfg):
    """Budget block for launch_production.backend_config (global cap semantics, like clean3)."""
    value = checked(cfg)
    return (dict(total=OWNER_CAP, phases=deepcopy(value['phases']), remaining_cap_sim_s=value['remaining_sim_s']),
            value['protected_prior_sim_s'])


def verify_previous(cfg):
    """Release-time audit: sealed budget, owner approval copy, map binding unchanged."""
    from . import clean4_budget as b
    bound = checked(cfg)
    root = Path(cfg['run_root'])
    quote_doc = json.loads((root/'CLEAN4_QUOTE.json').read_text(encoding='utf-8'))
    approval = json.loads((root/'BUDGET_CLEAN4.json').read_text(encoding='utf-8'))
    b.approved(approval, quote_doc)
    if quote_doc['receipt_sha256'] != bound['quote_sha256']:
        raise ValueError('clean4 quote changed after registration')
    check_map(cfg)
    return bound


def check_map(cfg):
    from . import map_descriptor as md
    desc = md.current()
    newmap = cfg['newmap']
    if desc.name != newmap['map'] or desc.digest != newmap['descriptor_sha256']:
        raise ValueError('process map differs from the clean4 registered wz3to2 descriptor')
    if getattr(desc, 'network_file_sha256', None) != NET_FILE_SHA256:
        raise ValueError('clean4 requires the staged wz3to2 net b6277898')
    return desc


# ---------------------------------------------------------------- guidance rows

def admitted_rows(paths, cells):
    """Plan A re-verified rows (B and R) for training: training cells only, own run_root bank."""
    from .map_runtime import registered_train_world
    rows, rejected = [], []
    for path in paths:
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            world = row.get('world') or {}
            key = (world.get('demand'), world.get('penetration'))
            if key not in cells:
                rejected.append(dict(row=row, reason='cell outside the clean4 training cell set'))
            elif not registered_train_world(world):
                rejected.append(dict(row=row, reason='world is not a registered wz3to2 TRAIN world'))
            else:
                rows.append(row)
    return rows, rejected


def registration_from_contexts(contexts, root):
    """ISOLATED_REGISTRATION from Plan A B decision contexts, by METADATA only (frozen move-list order):
    worlds ordered by descending context count then seed; first world up to 3 states, the next four
    one state; the registered group is the frozen move's group (never the outcome-chosen plan).

    contexts: [dict(seed, snapshot_path, receipt_path, group_plan)] in frozen order."""
    import torch
    from collections import defaultdict
    from . import rerun_cpus as rc, launch_preparation as stage
    from .track2_monitor import diagnostic_identity
    by_seed = defaultdict(list)
    for c in contexts:
        members = list(c['group_plan']['members'])
        if 2 <= len(members) <= 4 and len(set(members)) == len(members):
            by_seed[int(c['seed'])].append(c)
    ordered = sorted(by_seed, key=lambda s: (-len(by_seed[s]), s))
    if len(ordered) < 5 or len(by_seed[ordered[0]]) < 3:
        raise ValueError('five distinct TRAIN world seeds and three first-world states required')
    rows = []
    for index, seed in enumerate(ordered[:5]):
        states, files = [], []
        for number, c in enumerate(by_seed[seed][:3 if index == 0 else 1]):
            snapshot = torch.load(c['snapshot_path'], weights_only=False, map_location='cpu')
            original = json.loads(Path(c['receipt_path']).read_text(encoding='utf-8'))
            if snapshot['episode'].get('role') != 'train' or snapshot['episode']['world']['seed'] != seed:
                raise ValueError('registered state must be a TRAIN snapshot of its world')
            receipt = dict(original, chosen_plan=c['group_plan'],
                           clean4_registration=dict(original_receipt_sha256=hashlib.sha256(
                               Path(c['receipt_path']).read_bytes()).hexdigest(),
                               rule='frozen Plan A move group (metadata), not the outcome-chosen plan'))
            directory = Path(root)/'registered_states'/f'r{index:02}_s{number:02}'
            directory.mkdir(parents=True)
            stage.copy_equal(c['snapshot_path'], directory/'SNAPSHOT.pt')
            (directory/'RECEIPT.json').write_text(json.dumps(receipt, sort_keys=True), encoding='utf-8')
            files.append(dict(snapshot_path=str(directory/'SNAPSHOT.pt'), snapshot_file_sha256=rc.sha(directory/'SNAPSHOT.pt'),
                              receipt_path=str(directory/'RECEIPT.json'), receipt_file_sha256=rc.sha(directory/'RECEIPT.json')))
            states.append((snapshot, receipt))
        rows.append(dict(replicate=index, world_seed=seed, states=files, **diagnostic_identity(states)))
    registration = dict(role='train', metric=rc.ISOLATED_TEST['metric'], replicates=rows,
        selection='clean4: Plan A B contexts (frozen move-list order); five TRAIN world seeds by descending '
                  'context count then seed; first world three states, others first state; no outcomes',
        final_evaluation=dict(role='external', panel=EVAL_PANEL))
    rc.isolated_jobs(registration)
    return registration


def sidecar_state(cfg, arm, runtime):
    """Initial B/R u0 sidecar exactly as ArmRuntime.save_checkpoint writes it (incl. the N2 RSU optimizer).
    Optional prepare-time warm start (BOOST-PRE, default OFF): cfg['coop_warmstart'] delegates to
    tools.coopmwm.coop_warmstart.sidecar_state (B and R each from their own rows, equal effort)."""
    if cfg.get('coop_warmstart'):
        from . import coop_warmstart
        return coop_warmstart.sidecar_state(cfg, arm, runtime)
    state = dict(arm=arm, update=0, worker=runtime.policy.worker_state(),
                 layer_optimizer=runtime.policy.layer.optimizer.state_dict(),
                 trigger_optimizer=runtime.policy.trigger_optimizer.state_dict())
    if getattr(runtime.policy, 'rsu', None) is not None:
        state['rsu_optimizer'] = runtime.policy.rsu_optimizer.state_dict()
    return state


# ---------------------------------------------------------------- configuration

def levers(cfg, *, lane_drop_keys, v2v_keys, w2_keys, nm_families=None):
    """Common B/R switches on top of clean3's G2/G4/G5 (A unaffected by construction)."""
    from .clean3_restart import levers as clean3_levers, G4_CAP
    cfg = clean3_levers(cfg)
    cfg['g4_budget'] = dict(phase_id='g4_synergy_credit', allocation_sim_s=G4_CAP, debits={},
                            provenance='CLEAN4_QUOTE.json g4_synergy_credit 230400 (registered hard cap)')
    cfg.update(deepcopy(lane_drop_keys))
    cfg.update(deepcopy(v2v_keys))
    cfg.update(deepcopy(w2_keys))
    if nm_families:
        cfg.update(deepcopy(nm_families))
    return cfg


def prepare(source, *, plan_a, quote_doc, approval, prior_runtime_sim_s, lane_drop_keys, v2v_keys, w2_keys,
            nm_families=None, warmstart=None, root=ROOT, clean2=CLEAN2):
    """Zero physics. plan_a = dict(rows=[B/R jsonl paths], contexts=[...], move_list_sha256, potential_sha256,
    negatives=<path|None>, cells=[(demand, share), ...])."""
    import torch
    from . import composite_health, launch_preparation as stage, launch_clean2 as c2
    from . import launch_production as production, launch_qualification as qualification
    from . import launch_training as training, practice_starts as practice
    from . import rerun_cpus as rc, track2_backend as p1, bank_breadth, g5_runtime
    from . import map_descriptor as md
    root, source, clean2 = Path(root), Path(source), Path(clean2)
    for path in (root, source, clean2):
        rc.guard_path(path); rc.guard_path(path.resolve())
    if root.exists():
        raise ValueError('new clean4 root required; never overwrite')
    if hashlib.sha256(Path(U0).read_bytes()).hexdigest() != U0_SHA256:
        raise ValueError('staged MAPPO u0 identity differs')
    desc = md.current()
    if desc.name != 'wz3to2' or desc.network_file_sha256 != NET_FILE_SHA256:
        raise ValueError('process map must be the staged wz3to2 (COOPMWM_MAP_SPEC)')
    if not g5_runtime.enabled():
        raise ValueError('clean4 requires the bound G5 runtime env (COOPMWM_G5_RUNTIME=1)')
    setup = training.read('/home/ubuntu/cpuv_setup_20261105/S0_READY.json')
    parity = training.read('/home/ubuntu/cpuv_parity_20261105/PARITY_PASS.json')
    if not setup['pass_'] or not parity['pass_']:
        raise ValueError('reuse qualified environment and exact SUMO parity')
    source_manifest = training.read(source/'SOURCE_MANIFEST.json')
    for name, sha in source_manifest['files'].items():
        if rc.sha(source/name) != sha:
            raise ValueError('immutable clean4 source differs: '+name)
    rows, rejected = admitted_rows(plan_a['rows'], {tuple(c) for c in plan_a['cells']})
    arms = {r['arm'] for r in rows}
    if 'B' not in arms:
        raise ValueError('no admitted B guidance rows: no launch')
    cfg = levers(training.read(clean2/'CONFIG.json'), lane_drop_keys=lane_drop_keys, v2v_keys=v2v_keys,
                 w2_keys=w2_keys, nm_families=nm_families)
    cfg.update(run_root=str(root), accounting_root=str(root), bank_source_root=str(root),
        source_root=str(source), source_manifest_sha256=digest(source_manifest),
        gate_attempt=1, smoke_attempt=1, resume_update=0, A_resume_update=0,
        worlds_by_update=worlds_by_update(), replay_sources={}, paired_evaluation=False,
        extra_final_baselines=[])
    cfg['control_arm'] = dict(arm='A', root=str(root), update=24)
    cfg['a_results']['root'] = str(root)
    cfg['single_ledger']['root'] = str(root)
    cfg['augmentation_prelaunch']['ledger_root'] = str(root)
    cfg['plan_fix_ready'] = str(root/'PLAN_FIX_DISABLED.json')
    cfg.pop('clean2_restart'); cfg.pop('clean2_budget')
    cfg['safety_reference'] = dict(by_world={}, unmatched_zero_event_benchmark=True,
        source='clean4: no registered wz3to2 MAPPO check panel; unmatched TRAIN zero-event benchmark (conservative)')
    cfg['newmap'] = dict(map='wz3to2', base_controller='mappo_u0', cell=CELL_KEY, eval_panel_sha256=EVAL_PANEL_SHA256,
        descriptor_sha256=desc.digest, network_file_sha256=NET_FILE_SHA256, map_spec=MAP_SPEC,
        sanity='/home/ubuntu/clean4_sanity_e512b059_r2/SANITY_SUMMARY.json (MAPPO PASS, REGISTRATION 16:22)',
        family_rule=FAMILY_RULE, v2v_range_m=V2V_RANGE_M)
    cfg['g5_runtime'] = dict(flag=g5_runtime.FLAG, base=str(g5_runtime.BASE_REL),
        base_sha256=g5_runtime.BASE_SHA256, overlay_sha256=g5_runtime.OVERLAY_SHA256,
        rule='accepted G5 env change bound onto the frozen e3final env; wz3to2 bound by map_runtime.install')
    if cfg['plan_augment']['enabled'] or cfg.get('launch_augmentation_decision') != rc.AUGMENTATION_OFF_DECISION:
        raise ValueError('owner symmetric augmentation-OFF design required')
    if warmstart:
        cfg['coop_warmstart'] = deepcopy(warmstart)
    rc.check_pins(cfg)
    clean4_budget = budget(quote_doc=quote_doc, approval=approval, prior_runtime_sim_s=prior_runtime_sim_s)
    rule = training.read(clean2/rc.RULE_FILE)
    rc.init_root(root, amendment_shas=[rule['amendment_sha256']], track2=True)
    training.write(root/'PREPARATION_STARTED.json', dict(unix=time.time(), source_root=str(source),
        plan_a=deepcopy({k: plan_a[k] for k in ('rows', 'move_list_sha256', 'potential_sha256', 'cells')}),
        rejected_rows=len(rejected), new_sim_s=0))
    for name, value in (('CLEAN4_QUOTE.json', quote_doc), ('BUDGET_CLEAN4.json', approval)):
        (root/name).write_text(json.dumps(value, indent=1, sort_keys=True), encoding='utf-8')
    # Guidance rows (B and R) copied into the root; ingestion reads them through their own banks.
    with (root/'CLEAN4_GUIDANCE_ROWS.jsonl').open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True)+'\n')
    cfg['mined_moves'] = str(root/'CLEAN4_GUIDANCE_ROWS.jsonl')
    cfg['demonstrate_families'] = sorted({r['family'] for r in rows})
    if plan_a.get('negatives'):
        stage.copy_equal(plan_a['negatives'], root/'WALL_NEGATIVES.jsonl')
        cfg['wall_negatives'] = str(root/'WALL_NEGATIVES.jsonl')
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
            f.write('\nclean4 owner safety rule for '+arm+': composite report-only (inherited owner rule); any '
                    'collision/teleport pauses. A keeps the full rule.\n')
    for arm in ('B', 'R'):
        (root/'replay_bank'/arm).mkdir(parents=True)
    for arm in ('A', 'B', 'R'):
        stage.copy_equal(U0, root/f'{arm}_checkpoint_000.pt')
        if rc.sha(root/f'{arm}_checkpoint_000.pt') != cfg['common_checkpoint_sha256']:
            raise ValueError('MAPPO u0 identity differs')
    cfg['clean4_budget'] = clean4_budget
    cfg['clean4_restart'] = dict(owner_quote='clean4', clean2_root=str(clean2), inherited_safety_receipts=safety_receipts,
        inherited_safety=SAFETY_RULE, source_commit=source_manifest['commit'], plan_a=deepcopy(plan_a),
        guidance_rows=dict(B=sum(r['arm'] == 'B' for r in rows), R=sum(r['arm'] == 'R' for r in rows),
                           rejected=len(rejected)))
    native = production.backend_config(cfg)
    for arm in ('B', 'R'):
        torch.manual_seed(cfg['training_seed'])
        runtime = p1.ArmRuntime(dict(native, initial_sidecars_required=False), arm, 1)
        state = sidecar_state(cfg, arm, runtime)
        with (root/f'{arm}_track2_000.pt').open('xb') as f:
            torch.save(state, f)
    registration = registration_from_contexts(plan_a['contexts'], root)
    training.write(root/'ISOLATED_REGISTRATION.json', registration)
    quotes = c2.full_diagnostic_quotes(cfg)
    gate_pairs = qualification.gate_pairs(cfg)
    gate_manifest = [dict(id=s['verification_id'], sim_s=s['sim_s'], window=s['window'],
        snapshot_sha256=s['snapshot_sha256']) for s, _ in gate_pairs]
    gate_cost = sum(s['sim_s'] for s in gate_manifest)
    phases = clean4_budget['phases']
    if gate_cost > phases['gate_bank_verification_ext']:
        raise ValueError('full-window gate exceeds its newmap2_smoke share')
    diag = {k: q['quoted_sim_s'] for k, q in quotes.items()}
    if (diag['probe_u002']+diag['probe_u004'] > phases['early_probes'] or
            sum(diag[f'replicated_u{u:03}'] for u in (6, 12, 24)) > phases['isolated_replicated']):
        raise ValueError('native diagnostic manifests exceed clean4 phases')
    reporting = [s for s in gate_manifest if '_s000_' in s['id']]
    if not reporting or cfg['smoke']['total_sim_s']+sum(s['sim_s'] for s in reporting) > 20000:
        raise ValueError('full first-state B/R smoke reporting plus episodes exceeds smoke cap')
    if cfg['smoke']['total_sim_s']+sum(s['sim_s'] for s in reporting)+gate_cost > 40000:
        raise ValueError('smoke + gate exceed newmap2_smoke 40,000')
    cfg['clean2_smoke_reporting_quote'] = dict(selection='first prospectively registered state, B/R group/solos/common-base; full windows; descriptive only',
        branches=reporting, sim_s=sum(s['sim_s'] for s in reporting),
        complete_smoke_sim_s=cfg['smoke']['total_sim_s']+sum(s['sim_s'] for s in reporting))
    cfg['clean4_restart'].update(diagnostic_quotes=diag, gate_sim_s=gate_cost)
    actual = set(root/name for name in safety_receipts)
    summaries, pools = {}, {}
    with rc._ReadRecorder([root], Path('/home/ubuntu')) as recorder:
        native = production.backend_config(cfg)
        for arm in ('B', 'R'):
            with p1._bank_view(native) as bank:
                pairs, summary = bank.bank_pairs(root, arm, cfg['training_seed'], sources=(), v2v=True,
                    v2v_intent=cfg, imitation_bank_filter='cooperative_only', imitation_target_space='effective')
                summaries[arm] = summary
            pool = bank_breadth.practice_pool(native, arm)
            pools[arm] = [dict(context_id=r['context_id'], family=r['family'],
                verification_id=r['verification_id']) for r in pool]
            for row in pool:
                practice.load_snapshot(root, arm, row)
        if not pools['B']:
            raise ValueError('nonempty verified B practice pool required')
        from .launch_evaluation import lesson_quotes
        lessons = {k: q['quoted_sim_s'] for k, q in lesson_quotes(cfg).items()}
        if sum(lessons.values()) > phases['v2v_lessons']:
            raise ValueError('native lesson schedule exceeds its clean4 phase: '+json.dumps(lessons))
        cfg['clean4_restart']['lesson_quotes'] = lessons
        supervision = {}
        for arm in ('B', 'R'):
            runtime = p1.ArmRuntime(native, arm, 1)
            supervision[arm] = len(production.bank_move_supervision(runtime)['contexts'])
        if not supervision['B']:
            raise ValueError('nonempty B trigger supervision required')
        cfg['clean4_restart']['supervision_contexts'] = supervision
    actual.update(recorder.paths); actual.update(recorder.lexical)
    actual.add(Path(cfg['mined_moves']))
    for row in registration['replicates']:
        for item in row['states']:
            actual.update(Path(item[k+'_path']) for k in ('snapshot', 'receipt'))
    expanded, closure = production.staging_closure([p for p in actual if
        p.parent.name in ('pending', 'decisions') or p.name == 'RESULT.json' or p.suffix == '.pt'])
    actual.update(expanded)
    entries = {}
    for path in sorted(actual):
        rc.guard_path(path); rc.guard_path(path.resolve())
        if not path.is_file():
            raise ValueError('required clean4 reader input missing: '+str(path))
        entries[str(path)] = dict(sha256=rc.sha(path), bytes=path.stat().st_size, resolved_path=str(path.resolve()))
    manifest = _seal(dict(schema='coopmwm.clean4.reader_staging.v1', entries=entries, bank_summaries=summaries,
        practice_pools=pools, closure=closure, s0_manifest_sha256=setup['inventory_sha256']), 'manifest_sha256')
    training.write(root/'STAGING_MANIFEST.json', manifest)
    cfg.update(staging_manifest_sha256=rc.sha(root/'STAGING_MANIFEST.json'),
               isolated_registration_sha256=rc.sha(root/'ISOLATED_REGISTRATION.json'))
    training.write(root/'CLEAN4_BUDGET.json', clean4_budget)
    training.write(root/'CONFIG.json', cfg)
    approval_doc = dict(approved=True, owner_quote='clean4', config_sha256=digest(cfg),
        source_manifest_sha256=cfg['source_manifest_sha256'], budget_receipt_sha256=clean4_budget['receipt_sha256'],
        created=datetime.now().astimezone().isoformat(),
        authority='owner 14:58 "yes, approve the 4M budget. can you launch as much in parallel as possible to build '
                  'this and make sure everything works. try to have it as fast as possible" (BUDGET_CLEAN4.json); '
                  'launch requires the registered potential gate + smoke + safety gate')
    training.write(root/'OWNER_APPROVAL.json', approval_doc)
    training.write(root/'AUGMENTATION_DISABLED.json', _seal(dict(config_sha256=digest(cfg),
        admitted=dict(B=0, R=0), owner_decision=rc.AUGMENTATION_OFF_DECISION), 'receipt_sha256'))
    with (root/'AUTHORITY.md').open('a', encoding='utf-8') as f:
        f.write('\n'+json.dumps(dict(owner_approval=approval_doc, budget=clean4_budget, quote_sha256=quote_doc['receipt_sha256'],
            newmap=cfg['newmap'], eval_panel=EVAL_PANEL, family_rule=FAMILY_RULE,
            design='A plain MAPPO; B/R identical code (G2, G4, G5, W1 V2V initiate/join range 200 m, W2 pressure/release/'
                   'negatives, N2 lane-drop moves/V2V/RSU, N9 families if merged); only the guidance source differs',
            guidance=cfg['clean4_restart']['guidance_rows'], plan_a=cfg['clean4_restart']['plan_a'],
            seeds='TRAIN worlds 989220-989267 (cell D3200_p60, 16/update, rotating thirds); guidance rows from '
                  'registered TRAIN worlds (989000-989099 MAIN 16:45 + 989200-989599); final panel 989600-989695 (N6)',
            criterion=rc.criterion_document(), reporting=cfg['clean2_reporting_definition'],
            paired_evaluation='in-run u8/u24 OFF; final = N6 launcher on panel 3acb2b56..., debited from v2v_final',
            safety=SAFETY_RULE), sort_keys=True)+'\n')
    training.checked_config(root/'CONFIG.json')
    qualification.verify_inputs(cfg)
    training.write(root/'PREPARED.json', dict(status='STAGED_NOT_LAUNCHED', config_sha256=digest(cfg),
        files=len(entries), remaining_cap_sim_s=clean4_budget['remaining_sim_s'],
        guidance=cfg['clean4_restart']['guidance_rows'], new_sim_s=0))
    return cfg
