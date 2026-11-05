"""Owner-approved clean3 restart from MAPPO u0 with G1/G2/G4/G5 levers.

Zero-physics preparation, budget registration, G1 certification runner and
mined-move admission. Physics happens only in the explicit G1 batch (its own
ledger) and the existing smoke/gate/training code paths. Clean1/clean2 remain
intact evidence. No horizon, window, schedule or criterion is changed.
"""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import json
import math
import os
import time

from .exact_grade import _seal, verify_sealed
from .protocol import digest

ROOT = '/home/ubuntu/coopmwm20261106_clean3'
CLEAN2 = '/home/ubuntu/coopmwm20261105_clean2'
CLEAN1 = '/home/ubuntu/coopmwm20261105_clean1'
G1_ROOT = '/home/ubuntu/coopmwm20261106_clean3_g1cert'
MINING_ROOT = '/home/ubuntu/coopmwm20261106_s_mining'
SCHEMA = 'coopmwm.clean3.restart_budget.v1'

# Owner 2026-11-06 03:00 "approve": plan_fix reservation released, cap 7.3M.
CAP = 7_300_000
MINING = 500_000
PARITY = 960
G1_CAP = 230_400
G4_CAP = 230_400
# CLEAN3_QUOTE.json (full u1-u24, >=15% margin per phase). v2v_train includes
# practice references (clean2 charged them to v2v_train); v2v_final includes
# the descriptive lane-on MAPPO aecd7d70 baseline (96 x 1200 x 1.15).
PHASES = dict(v2v_train=1_988_400, v2v_lessons=40_800, early_probes=52_800,
    isolated_replicated=165_600, v2v_interim=398_400, v2v_final=531_600,
    gate_bank_verification_ext=18_000, track2_smoke=24_000,
    teacher_verify=0, group_credit_verify=0, augment_verify=0)
QUOTE_TOTAL = 3_680_400
LANE_ON_BASELINE = dict(name='mappo_laneon_aecd7d70', commit='aecd7d70', episodes=96,
    role='extra frozen final-evaluation baseline; descriptive only; primary comparator remains speed-only MAPPO',
    authority='owner 2026-11-06 01:16 "keep it as the baseline for now..."; MAIN 01:16')
SAFETY_RULE = ('A: full original rule. B and R: composite TTC/critical comparison reported each '
    'update, no pause. Any collision or teleport in any arm pauses. (owner 2026-11-06 02:44 "1")')
G1_GO_RULE = ('GO iff >=1 admitted executable joint label (synergy CI95 low>0 on pooled and both '
    'seeds, safety holds) in BOTH B and R with identical admitted family sets; then the '
    'certificate manifest gates B/R bank lessons. Otherwise NO-GO: manifest kept as diagnostic, not bound.')

assert sum(PHASES.values())+G1_CAP+G4_CAP == QUOTE_TOTAL


def _count(value, label):
    if type(value) is not int or value < 0:
        raise ValueError('nonnegative integer required: '+label)
    return value


def build(*, clean1_rows, clean2_rows, g1_rows, mining_rows, prefixes, accepted):
    """Pure registration. Every prior/void row stays charged; mining reserved in full."""
    spent = {}
    for name, rows in (('clean1', clean1_rows), ('clean2', clean2_rows), ('g1', g1_rows),
                       ('mining', mining_rows)):
        ids = [r['id'] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate charge id in '+name)
        spent[name] = sum(_count(r['sim_s'], name) for r in rows)
    if spent['g1'] > G1_CAP:
        raise ValueError('G1 certification exceeded its registered cap')
    if spent['mining'] > MINING:
        raise ValueError('mining exceeded its separate 500000 phase')
    if set(accepted) != {'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'G1', 'G2', 'G4', 'G5'}:
        raise ValueError('F1-F6 and G1/G2/G4/G5 acceptance receipts required')
    # Quote assertion (MAIN 02:43): full quote <= cap - prior - mining.
    before = spent['clean1']+spent['clean2']+PARITY
    if before+MINING+QUOTE_TOTAL > CAP:
        raise ValueError('CLEAN3 full quote exceeds remaining owner cap')
    prior = before+spent['g1']+MINING
    remaining = CAP-prior
    if sum(PHASES.values())+G4_CAP > remaining:
        raise ValueError('clean3 phases exceed remaining owner cap after G1/mining')
    return _seal(dict(schema=SCHEMA, owner_cap_sim_s=CAP, spent=spent, prior_parity_sim_s=PARITY,
        mining_reserved_sim_s=MINING, plan_fix_reserved_sim_s=0, protected_prior_sim_s=prior,
        remaining_sim_s=remaining, phases=dict(PHASES), g4_cap_sim_s=G4_CAP, g1_cap_sim_s=G1_CAP,
        quote_total_sim_s=QUOTE_TOTAL, prefixes=deepcopy(prefixes), accepted=deepcopy(accepted),
        rows=dict(clean1=len(clean1_rows), clean2=len(clean2_rows), g1=len(g1_rows)),
        rule='clean1+clean2+parity+G1 actual+mining 500000 reserved; plan_fix released (owner 03:00); '
             'every phase sized for u1-u24 with >=15% margin'), 'receipt_sha256')


def checked(cfg):
    value = cfg['clean3_budget']
    verify_sealed(value, 'receipt_sha256')
    synthetic = {name: [dict(id=name, sim_s=value['spent'][name])] if value['spent'][name] else []
                 for name in ('clean1', 'clean2', 'g1', 'mining')}
    rebuilt = build(clean1_rows=synthetic['clean1'], clean2_rows=synthetic['clean2'],
        g1_rows=synthetic['g1'], mining_rows=synthetic['mining'], prefixes=value['prefixes'],
        accepted=value['accepted'])
    for key in ('phases', 'protected_prior_sim_s', 'remaining_sim_s', 'owner_cap_sim_s',
                'spent', 'quote_total_sim_s'):
        if rebuilt[key] != value[key]:
            raise ValueError('clean3 budget registration differs: '+key)
    return deepcopy(value)


def native_budget(cfg):
    """Budget block for launch_production.backend_config (global cap semantics)."""
    value = checked(cfg)
    return (dict(total=value['owner_cap_sim_s'], phases=deepcopy(value['phases']),
                 remaining_cap_sim_s=value['remaining_sim_s']), value['protected_prior_sim_s'])


# ---------------------------------------------------------------- previous runs

def _alive(owner):
    path = Path('/proc')/str(owner['pid'])/'stat'
    fields = path.read_text().split() if path.exists() else []
    return bool(fields) and fields[2] != 'Z' and fields[21] == owner['start_ticks']


def stopped_clean2(root=CLEAN2):
    from . import rerun_cpus as rc, launch_training as training
    root = Path(root)
    for name in ('STOP.json', 'STOPPED_BY_OWNER.json'):
        if not (root/name).is_file():
            raise ValueError('preserved clean2 durable owner stop required: '+name)
    launched = training.read(root/'LAUNCHED.json')
    for role, owner in launched['owners'].items():
        if _alive(owner):
            raise ValueError('clean2 owner still alive: '+role)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if str(root/'CONFIG.json').encode() in args:
            raise ValueError('unrecorded clean2 runtime owner: '+proc.name)
    return dict(stop=rc.sha(root/'STOP.json'), owner_stop=rc.sha(root/'STOPPED_BY_OWNER.json'))


def ledgers():
    from .storage import ledger
    return {name: ledger(Path(root)) for name, root in
            (('clean1', CLEAN1), ('clean2', CLEAN2), ('mining', MINING_ROOT))}


def g1_rows(root=G1_ROOT):
    from .storage import ledger
    return ledger(Path(root)) if (Path(root)/'CHARGES.jsonl').is_file() else []


def verify_previous(cfg):
    """Release-time audit: preserved stops and ledger prefixes; nothing new outside caps."""
    bound = checked(cfg)
    restart = cfg['clean3_restart']
    stops = stopped_clean2(restart['clean2_root'])
    if stops != restart['clean2_stop_sha256']:
        raise ValueError('clean2 durable stop changed since clean3 registration')
    current = ledgers()
    for name in ('clean1', 'clean2'):
        size = bound['rows'][name]
        if len(current[name]) != size or digest(current[name]) != bound['prefixes'][name]:
            raise ValueError(name+' ledger changed after clean3 registration')
    if sum(r['sim_s'] for r in current['mining']) > MINING:
        raise ValueError('mining exceeded its separate phase')
    if sum(r['sim_s'] for r in g1_rows(restart['g1_root'])) != bound['spent']['g1']:
        raise ValueError('G1 certification ledger changed after registration')
    return bound


# ---------------------------------------------------------------- G1 batch

def g1_config(source_root, work):
    """Clean2 design + clean3 lever switches, bank = read-only view of clean1's."""
    from . import launch_training as training, launch_preparation as stage, rerun_cpus as rc
    from .launch_production import backend_config
    source_manifest = training.read(Path(source_root)/'SOURCE_MANIFEST.json')
    cfg = training.read(Path(CLEAN2)/'CONFIG.json')
    cfg = levers(cfg)
    work = Path(work)
    for key in ('run_root', 'accounting_root', 'bank_source_root'):
        cfg[key] = str(work)
    cfg['plan_fix_ready'] = str(work/'PLAN_FIX_DISABLED.json')
    for key in ('control_arm', 'a_results', 'single_ledger'):
        cfg[key]['root'] = str(work)
    cfg['augmentation_prelaunch']['ledger_root'] = str(work)
    cfg.pop('clean2_budget'); cfg.pop('clean2_restart')
    cfg.pop('g4_synergy_credit'); cfg.pop('g4_budget')
    rc.check_pins(cfg)
    return backend_config(cfg), digest(source_manifest)


def levers(cfg):
    """Common B/R G2/G4/G5 switches (A unaffected by construction)."""
    from . import coordination_opportunities as co, sparse_synergy_credit as credit
    cfg = deepcopy(cfg)
    cfg['coordination_opportunities'] = dict(enabled=True, rule=co.RULE,
        provenance='MAIN 2026-11-05 23:15 G2 (CLEAN2_GROUP_COOP_RESEARCH rank 2); accepted c4/grp-opportunity')
    cfg['g4_synergy_credit'] = dict(enabled=True, registration=credit.registration(),
        phase_id='g4_synergy_credit', first_update=1)
    cfg['g4_budget'] = dict(phase_id='g4_synergy_credit', allocation_sim_s=G4_CAP, debits={},
        provenance='CLEAN3_QUOTE.json g4_synergy_credit 230400 (registered hard cap); owner cap 7.3M 03:00')
    cfg['group_lane_moves'] = True
    return cfg


def certify(source_root, *, work=G1_ROOT, workers=16):
    """Run the registered G1 batch once; write manifest and go/no-go. No retry."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    import types
    import fcntl
    from . import g1_certification as g1, executable_labels as labels
    from . import launch_preparation as stage, launch_training as training, rerun_cpus as rc
    from tools.coopnative import storage as old
    work = Path(work)
    if work.exists():
        raise ValueError('G1 work root exists; inspect, never replay')
    rc.init_root(work)
    for arm in ('B', 'R'):
        stage.readonly_bank_view(Path(CLEAN1)/'replay_bank'/arm, work/'replay_bank'/arm)
    cfg, runtime_sha256 = g1_config(source_root, work)
    selected, info = g1.prepare(cfg, runtime_sha256)
    training.write(work/'G1_PREPARED.json', dict(runtime_sha256=runtime_sha256,
        candidates=info['candidates'], selected=info['selected'], quote=info['quote'],
        diagnostics=info['diagnostics'], selection=info['selection'], unix=time.time()))
    jobs = [item['job'] for item in selected]
    pairs = []
    for item in selected:
        width = len(item['job']['lesson']['macro']['members'])+2
        for i, branch in enumerate(item['job']['branches']):
            pairs.append((branch, item['snapshots'][i//width]))
    charges = [dict(id='g1_'+b['branch_sha256'], sim_s=b['sim_s']) for b, _ in pairs]
    if sum(c['sim_s'] for c in charges) > G1_CAP:
        raise ValueError('G1 batch exceeds registered cap')
    if charges:
        reserve = types.FunctionType(old.reserve.__code__, dict(old.reserve.__globals__,
            CAP=G1_CAP, CAPS=dict(g1_certification=G1_CAP)))
        with (work/'ACCOUNTING.claim').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            reserve(work, charges, 'g1_certification')
    results = {}
    if pairs:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {b['branch_sha256']: pool.submit(g1.run_branch, b, s, work) for b, s in pairs}
            for key, future in futures.items():
                try:
                    results[key] = future.result(timeout=7200)
                except Exception as error:  # retained as diagnostic evidence, never retried
                    results[key] = dict(branch_sha256=key, status='error', error=repr(error))
    entries = [dict(job=job, results=[results[b['branch_sha256']] for b in job['branches']
                                      if results.get(b['branch_sha256'], {}).get('status') == 'graded'])
               for job in jobs]
    manifest = labels.manifest(entries)
    training.write(work/'G1_MANIFEST.json', manifest)
    training.write(work/'G1_RAW_RESULTS.json', dict(results=results))
    return go_nogo(manifest, work=work, runtime_sha256=runtime_sha256,
                   charged=sum(c['sim_s'] for c in charges), branches=len(pairs))


def go_nogo(manifest, *, work, runtime_sha256, charged, branches):
    from . import launch_training as training
    admitted = {'B': set(), 'R': set()}
    counts = {'B': 0, 'R': 0}
    for row in manifest['entries']:
        if row['admission']['admitted']:
            lesson = row['job']['lesson']
            counts[lesson['arm']] += 1
            admitted[lesson['arm']].add(lesson['macro']['family'])
    go = counts['B'] > 0 and counts['R'] > 0 and admitted['B'] == admitted['R']
    value = _seal(dict(schema='coopmwm.clean3.g1_go_nogo.v1', decision='GO' if go else 'NO-GO',
        executable_joint_labels_ci95_low_positive=manifest['admitted'], admitted_by_arm=counts,
        admitted_families={a: sorted(v) for a, v in admitted.items()},
        diagnostics=manifest['diagnostics'], jobs=len(manifest['entries']), branches=branches,
        charged_sim_s=charged, manifest_path=str(Path(work)/'G1_MANIFEST.json'),
        manifest_sha256=manifest['manifest_sha256'], runtime_sha256=runtime_sha256,
        rule=G1_GO_RULE, unix=time.time()), 'receipt_sha256')
    training.write(Path(work)/'CLEAN3_GO_NOGO.json', value)
    return value


# ---------------------------------------------------------------- mined moves

def admit_mined(path, cfg, out_path):
    """Keep only rows that pass native F3/G5 admission, families common to B and R.

    Writes the admitted subset to out_path (never edits the producer's file).
    Returns a receipt; an empty admission means the current bank only.
    """
    from . import bank_breadth, group_lane_bank, launch_training as training
    from .rerun_cpus import sha
    path = Path(path)
    rows = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()] \
        if path.is_file() else []
    usable, rejected = [], []
    for row in rows:
        single = Path(str(out_path)+'.probe')
        try:
            single.write_text(json.dumps(row)+'\n', encoding='utf-8')
            local = dict(cfg, mined_moves=str(single))
            if row.get('run_root') and not Path(row['run_root']).is_absolute():
                row = dict(row, run_root=str((path.parent/row['run_root']).resolve()))
                single.write_text(json.dumps(row)+'\n', encoding='utf-8')
            group_lane_bank.inputs(local)
            families = set()
            def consume(*, cfg, receipt, example, plan, pairs, trace):
                from .group_lane import from_plan
                if receipt['verification_id'] == row['verification_id'] and digest(plan) == row['plan_sha256']:
                    families.add(from_plan(plan).family)
            bank_breadth.visit(local, row['arm'], consume)
            if not families:
                raise ValueError('no admitted native example')
            usable.append(dict(row, _families=sorted(families)))
        except (ValueError, KeyError, TypeError, OSError) as error:
            rejected.append(dict(row=row, reason=str(error)))
        finally:
            if single.exists():
                single.unlink()
    support = {a: {f for r in usable if r['arm'] == a for f in r['_families']} for a in ('B', 'R')}
    common = support['B'] & support['R']
    kept = [r for r in usable if set(r['_families']) <= common]
    dropped = [dict(row=r, reason='family not verified in both B and R (same-distribution rule)')
               for r in usable if not set(r['_families']) <= common]
    if kept:
        with Path(out_path).open('x', encoding='utf-8') as f:
            for r in kept:
                f.write(json.dumps({k: v for k, v in r.items() if k != '_families'}, sort_keys=True)+'\n')
    return dict(source=str(path), source_sha256=sha(path) if path.is_file() else None,
        rows=len(rows), admitted=len(kept), admitted_path=str(out_path) if kept else None,
        families=sorted(common), rejected=rejected+dropped, unix=time.time())


# ---------------------------------------------------------------- preparation

def prepare(source, *, root=ROOT, clean1=CLEAN1, clean2=CLEAN2, g1_root=G1_ROOT,
            mined=None, laneon=None):
    import torch
    from . import composite_health, launch_preparation as stage, launch_clean2 as c2
    from . import launch_production as production, launch_qualification as qualification
    from . import launch_training as training, practice_starts as practice
    from . import rerun_cpus as rc, track2_backend as p1, bank_breadth
    root, source, clean1, clean2 = map(Path, (root, source, clean1, clean2))
    for path in (root, source, clean1, clean2):
        rc.guard_path(path); rc.guard_path(path.resolve())
    if root.exists():
        raise ValueError('new clean3 root required; never overwrite')
    stops = stopped_clean2(clean2)
    setup = training.read('/home/ubuntu/cpuv_setup_20261105/S0_READY.json')
    parity = training.read('/home/ubuntu/cpuv_parity_20261105/PARITY_PASS.json')
    if not setup['pass_'] or not parity['pass_'] or parity['charged_sim_s'] != PARITY:
        raise ValueError('reuse qualified environment and exact SUMO parity')
    source_manifest = training.read(source/'SOURCE_MANIFEST.json')
    for name, sha in source_manifest['files'].items():
        if rc.sha(source/name) != sha:
            raise ValueError('immutable clean3 source differs: '+name)
    accepted = {}
    for file in ('FIX_ACCEPTANCE.json', 'G_ACCEPTANCE.json'):
        for task, entry in training.read(source/file).items():
            if rc.sha(source/entry['path']) != entry['sha256']:
                raise ValueError('acceptance SHA differs: '+task)
            accepted[task] = entry['sha256']
    go = training.read(Path(g1_root)/'CLEAN3_GO_NOGO.json')
    verify_sealed(go, 'receipt_sha256')
    if go['runtime_sha256'] != digest(source_manifest):
        raise ValueError('G1 certification must bind this exact clean3 source')
    original = training.read(clean2/'CONFIG.json')
    cfg = levers(original)
    cfg.update(run_root=str(root), accounting_root=str(root), bank_source_root=str(root),
        source_root=str(source), source_manifest_sha256=digest(source_manifest),
        gate_attempt=1, smoke_attempt=1, resume_update=0, A_resume_update=0)
    cfg['control_arm'] = dict(arm='A', root=str(root), update=24)
    cfg['a_results']['root'] = str(root)
    cfg['single_ledger']['root'] = str(root)
    cfg['augmentation_prelaunch']['ledger_root'] = str(root)
    # Owner 02:44: plan-fix superseded; its outputs never enter clean3.
    cfg['plan_fix_ready'] = str(root/'PLAN_FIX_DISABLED.json')
    restart = cfg.pop('clean2_restart'); cfg.pop('clean2_budget')
    if go['decision'] == 'GO':
        cfg['executable_label_manifest'] = go['manifest_path']
        cfg['executable_label_runtime_sha256'] = go['runtime_sha256']
    if mined and mined.get('admitted'):
        cfg['mined_moves'] = mined['admitted_path']
    cfg['extra_final_baselines'] = [dict(LANE_ON_BASELINE, **(laneon or {}))]
    if cfg['plan_augment']['enabled'] or cfg.get('launch_augmentation_decision') != rc.AUGMENTATION_OFF_DECISION:
        raise ValueError('owner symmetric augmentation-OFF design required')
    rc.check_pins(cfg)
    current = ledgers()
    g1 = g1_rows(g1_root)
    prefixes = {name: digest(current[name]) for name in ('clean1', 'clean2')}
    budget = build(clean1_rows=current['clean1'], clean2_rows=current['clean2'], g1_rows=g1,
        mining_rows=current['mining'], prefixes=prefixes, accepted=accepted)
    rule = training.read(clean2/rc.RULE_FILE)
    rc.init_root(root, amendment_shas=[rule['amendment_sha256']], track2=True)
    training.write(root/'PREPARATION_STARTED.json', dict(unix=time.time(), source_root=str(source),
        accepted=accepted, clean2_stop=stops, g1_go_nogo_sha256=go['receipt_sha256'],
        mined=mined, new_sim_s=0))
    stage.copy_equal(clean2/rc.RULE_FILE, root/rc.RULE_FILE)
    safety_receipts = {}
    for arm, rule_name, pause_name in (('B', composite_health.B_RULE, composite_health.B_PAUSE),
                                       ('R', composite_health.R_RULE, composite_health.R_PAUSE)):
        if not composite_health.report_only(clean2, arm):
            raise ValueError('registered owner report-only rule missing: '+arm)
        for name in (rule_name, pause_name):
            stage.copy_equal(clean2/name, root/name)
            safety_receipts[name] = rc.sha(root/name)
        amendment = training.read(clean2/rule_name)['amendment_sha256']
        with (root/'AUTHORITY.md').open('a', encoding='utf-8') as f:
            f.write('\nclean3 owner safety rule (02:44 "1") for '+arm+': composite report-only, '
                    'amendmentSHA'+amendment+'; any collision/teleport pauses. A keeps the full rule.\n')
    for arm in ('B', 'R'):
        stage.readonly_bank_view(clean1/'replay_bank'/arm, root/'replay_bank'/arm)
    for arm in ('A', 'B', 'R'):
        stage.copy_equal(clean1/f'{arm}_checkpoint_000.pt', root/f'{arm}_checkpoint_000.pt')
        if rc.sha(root/f'{arm}_checkpoint_000.pt') != cfg['common_checkpoint_sha256']:
            raise ValueError('MAPPO u0 identity differs')
    cfg['clean3_budget'] = budget
    cfg['clean3_restart'] = dict(owner_quote='clean3', clean1_root=str(clean1), clean2_root=str(clean2),
        clean2_stop_sha256=stops, g1_root=str(g1_root), g1_go_nogo=go['decision'],
        g1_go_nogo_sha256=go['receipt_sha256'], inherited_safety_receipts=safety_receipts,
        inherited_safety=SAFETY_RULE, source_commit=source_manifest['commit'],
        mined_moves=mined, plan_fix='superseded and released (owner 02:44/03:00)')
    native = production.backend_config(cfg)
    for arm in ('B', 'R'):
        torch.manual_seed(cfg['training_seed'])
        runtime = p1.ArmRuntime(dict(native, initial_sidecars_required=False), arm, 1)
        state = dict(arm=arm, update=0, worker=runtime.policy.worker_state(),
            layer_optimizer=runtime.policy.layer.optimizer.state_dict(),
            trigger_optimizer=runtime.policy.trigger_optimizer.state_dict())
        with (root/f'{arm}_track2_000.pt').open('xb') as f:
            torch.save(state, f)
    registration = c2._registration(clean1, root)
    quotes = c2.full_diagnostic_quotes(cfg)
    gate_pairs = qualification.gate_pairs(cfg)
    gate_manifest = [dict(id=s['verification_id'], sim_s=s['sim_s'], window=s['window'],
        snapshot_sha256=s['snapshot_sha256']) for s, _ in gate_pairs]
    gate_cost = sum(s['sim_s'] for s in gate_manifest)
    if gate_cost > PHASES['gate_bank_verification_ext']:
        raise ValueError('full-window gate exceeds its clean3 phase')
    diag = {k: q['quoted_sim_s'] for k, q in quotes.items()}
    if (diag['probe_u002']+diag['probe_u004'] > PHASES['early_probes'] or
            sum(diag[f'replicated_u{u:03}'] for u in (6, 12, 24)) > PHASES['isolated_replicated']):
        raise ValueError('native diagnostic manifests exceed clean3 phases')
    reporting = [s for s in gate_manifest if '_s000_' in s['id']]
    if not reporting or cfg['smoke']['total_sim_s']+sum(s['sim_s'] for s in reporting) > 20000:
        raise ValueError('full first-state B/R smoke reporting plus episodes exceeds smoke cap')
    cfg['clean2_smoke_reporting_quote'] = dict(selection='first prospectively registered state, B/R group/solos/common-base; full windows; descriptive only',
        branches=reporting, sim_s=sum(s['sim_s'] for s in reporting),
        complete_smoke_sim_s=cfg['smoke']['total_sim_s']+sum(s['sim_s'] for s in reporting))
    cfg['clean3_restart'].update(diagnostic_quotes=diag, gate_sim_s=gate_cost)
    # Real final readers; every selected input must exist and be staged.
    actual = set(root/name for name in safety_receipts)
    summaries, pools = {}, {}
    from .launch_preparation import LEGACY
    with rc._ReadRecorder([root, clean1, LEGACY], Path('/home/ubuntu')) as recorder:
        native = production.backend_config(cfg)
        for arm in ('B', 'R'):
            with p1._bank_view(native) as bank:
                pairs, summary = bank.bank_pairs(root, arm, cfg['training_seed'],
                    sources=cfg.get('replay_sources', {}).get(arm, ()), v2v=True,
                    v2v_intent=cfg, imitation_bank_filter='cooperative_only',
                    imitation_target_space='effective')
                if not pairs:
                    raise ValueError('nonempty verified effective bank required')
                summaries[arm] = summary
            pool = bank_breadth.practice_pool(native, arm)
            if not pool:
                raise ValueError('nonempty verified practice pool required: '+arm)
            pools[arm] = [dict(context_id=r['context_id'], family=r['family'],
                verification_id=r['verification_id']) for r in pool]
            for row in pool:
                practice.load_snapshot(root, arm, row)
        bank_breadth.matched_pools(native)
        from .launch_evaluation import lesson_quotes
        lessons = {k: q['quoted_sim_s'] for k, q in lesson_quotes(cfg).items()}
        if len(lessons) != 4 or sum(lessons.values()) > PHASES['v2v_lessons']:
            raise ValueError('native lesson schedule exceeds its clean3 phase: '+json.dumps(lessons))
        cfg['clean3_restart']['lesson_quotes'] = lessons
        for arm in ('B', 'R'):
            runtime = p1.ArmRuntime(native, arm, 1)
            supervision = production.bank_move_supervision(runtime)
            if not supervision['contexts']:
                raise ValueError('nonempty trigger supervision required: '+arm)
    actual.update(recorder.paths); actual.update(recorder.lexical)
    for extra in filter(None, (cfg.get('executable_label_manifest'), cfg.get('mined_moves'))):
        actual.add(Path(extra))
    for row in registration['replicates']:
        for item in row['states']:
            actual.update(Path(item[k+'_path']) for k in ('snapshot', 'receipt'))
    prior_staging = training.read(clean1/'STAGING_MANIFEST.json')
    verify_sealed(prior_staging, 'manifest_sha256')
    actual.update(Path(p) for p in prior_staging['entries'])
    known_unavailable = {Path(p) for p in prior_staging.get('unresolved_nonselected_snapshot_attempts', [])}
    known_unavailable.update(root/p.relative_to(clean1) for p in tuple(known_unavailable)
                             if p.is_relative_to(clean1))
    unavailable = {p for p in actual if p in known_unavailable and not p.is_file()}
    actual.difference_update(unavailable)
    expanded, closure = production.staging_closure([p for p in actual if
        p.parent.name in ('pending', 'decisions') or p.name == 'RESULT.json' or p.suffix == '.pt'])
    actual.update(expanded)
    entries = {}
    for path in sorted(actual):
        rc.guard_path(path); rc.guard_path(path.resolve())
        if not path.is_file():
            raise ValueError('required clean3 reader input missing: '+str(path))
        entries[str(path)] = dict(sha256=rc.sha(path), bytes=path.stat().st_size,
                                 resolved_path=str(path.resolve()))
    for name, entry in prior_staging['entries'].items():
        if name in entries and entries[name]['sha256'] != entry['sha256']:
            raise ValueError('preserved staged evidence changed: '+name)
    manifest = _seal(dict(schema='coopmwm.clean3.reader_staging.v1', entries=entries,
        bank_summaries=summaries, practice_pools=pools, inherited_staging_sha256=prior_staging['manifest_sha256'],
        closure=closure, unresolved_nonselected_snapshot_attempts=sorted(map(str, unavailable)),
        s0_manifest_sha256=setup['inventory_sha256']), 'manifest_sha256')
    training.write(root/'STAGING_MANIFEST.json', manifest)
    cfg.update(staging_manifest_sha256=rc.sha(root/'STAGING_MANIFEST.json'),
        isolated_registration_sha256=rc.sha(root/'ISOLATED_REGISTRATION.json'))
    training.write(root/'CLEAN3_BUDGET.json', budget)
    training.write(root/'CONFIG.json', cfg)
    approval = dict(approved=True, owner_quote='clean3', config_sha256=digest(cfg),
        source_manifest_sha256=cfg['source_manifest_sha256'],
        budget_receipt_sha256=budget['receipt_sha256'], accepted=accepted,
        created=datetime.now().astimezone().isoformat(),
        authority=('MAIN 23:39 owner "when the lever finishes, just stop the current run and rerun it"; '
                   '02:13 "yes, do that"; 02:44 "use your own usage from now on. and 1"; '
                   '03:00 "approve" (cap 7.3M, plan_fix released)'))
    training.write(root/'OWNER_APPROVAL.json', approval)
    training.write(root/'AUGMENTATION_DISABLED.json', _seal(dict(config_sha256=digest(cfg),
        admitted=dict(B=0, R=0), owner_decision=rc.AUGMENTATION_OFF_DECISION), 'receipt_sha256'))
    with (root/'AUTHORITY.md').open('a', encoding='utf-8') as f:
        f.write('\n'+json.dumps(dict(owner_approval=approval, budget=budget,
            quote='CLEAN3_QUOTE.json total 3680400 (full u1-u24, >=15% margin per phase, incl. lane-on MAPPO final baseline)',
            cap='7,300,000 total; prior clean1+clean2+parity+G1 actual; mining 500,000 reserved; plan_fix released',
            design='A plain MAPPO; B/R identical code (F1-F6 + G1 certified lessons when GO, G2 opportunities, '
                   'G4 sparse measured synergy credit, G5 lane moves, verified mined moves); only guidance source differs',
            g1=dict(decision=go['decision'], rule=G1_GO_RULE, receipt_sha256=go['receipt_sha256']),
            mined_moves=mined, extra_final_baselines=cfg['extra_final_baselines'],
            criterion=rc.criterion_document(), reporting=cfg['clean2_reporting_definition'],
            seeds='original registered TRAIN worlds only; full windows; u2/u4 probes; N5 u6/u12/u24; paired u8/u24',
            safety=SAFETY_RULE, clean1_clean2_preserved=True), sort_keys=True)+'\n')
    stopped_clean2(clean2)
    training.checked_config(root/'CONFIG.json')
    qualification.verify_inputs(cfg)
    training.write(root/'PREPARED.json', dict(status='STAGED_NOT_LAUNCHED', config_sha256=digest(cfg),
        files=len(entries), remaining_cap_sim_s=budget['remaining_sim_s'],
        g1=go['decision'], mined_admitted=(mined or {}).get('admitted', 0), new_sim_s=0))
    return cfg
