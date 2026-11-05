"""Zero-physics preparation of the owner-approved u0 restart, preserving clean1."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import argparse
import json
import os
import time

import torch

from . import clean2_budget, composite_health, launch_preparation as stage
from . import launch_production as production, launch_qualification as qualification
from . import launch_training as training, practice_starts as practice
from . import replay_bank, rerun_cpus as rc, track2_backend as p1, track2_monitor as p3
from .exact_grade import _seal, verify_sealed
from .protocol import digest


ROOT = '/home/ubuntu/coopmwm20261105_clean2'
CLEAN1 = '/home/ubuntu/coopmwm20261105_clean1'


def restart_config(original, root, source, source_manifest):
    """Keep registered design; reset only root/source identities and attempt IDs."""
    cfg = deepcopy(original)
    cfg.update(run_root=str(root), accounting_root=str(root), bank_source_root=str(root),
        source_root=str(source), source_manifest_sha256=digest(source_manifest),
        gate_attempt=1, smoke_attempt=1, resume_update=0, A_resume_update=0)
    cfg['control_arm'] = dict(arm='A', root=str(root), update=24)
    cfg['a_results']['root'] = str(root)
    cfg['single_ledger']['root'] = str(root)
    cfg['augmentation_prelaunch']['ledger_root'] = str(root)
    if cfg['plan_augment']['enabled'] or cfg.get('launch_augmentation_decision') != rc.AUGMENTATION_OFF_DECISION:
        raise ValueError('owner symmetric augmentation-OFF design required')
    rc.check_pins(cfg)
    return cfg


def stopped_clean1(root):
    root = Path(root)
    if not (root/'STOP.json').is_file() or not (root/'STOPPED_BY_OWNER_CLEAN2_RESTART.json').is_file():
        raise ValueError('preserved clean1 durable owner restart stop required')
    owners = training.read(root/'RESUMED_1434.json')['owners']
    for role, owner in owners.items():
        path = Path('/proc')/str(owner['pid'])/'stat'
        fields = path.read_text().split() if path.exists() else []
        if fields and fields[2] != 'Z' and fields[21] == owner['start_ticks']:
            raise ValueError('clean1 owner still alive: '+role)
    # Also refuse an unrecorded duplicate owner of this exact configuration.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (str(root/'CONFIG.json').encode() in args and any(x in args for x in
                (b'tools.coopmwm.launch_training', b'tools.coopmwm.launch_recovery'))):
            raise ValueError('unrecorded clean1 runtime owner: '+proc.name)
    return rc.sha(root/'STOP.json')


def _registration(clean1, root):
    reg = training.read(clean1/'ISOLATED_REGISTRATION.json')
    for row in reg['replicates']:
        for i, item in enumerate(row['states']):
            target = root/'registered_states'/f"r{row['replicate']:02}_s{i:02}"
            target.mkdir(parents=True)
            for key, name in (('snapshot', 'SNAPSHOT.pt'), ('receipt', 'RECEIPT.json')):
                source = Path(item[key+'_path'])
                if rc.sha(source) != item[key+'_file_sha256']:
                    raise ValueError('preserved registered diagnostic state changed')
                stage.copy_equal(source, target/name)
                item[key+'_path'] = str(target/name)
    training.write(root/'ISOLATED_REGISTRATION.json', reg)
    return reg


class QuoteBackend(training.MonitorBackend):
    """Same complete native expansion, with u0 tensors; no future update claim."""
    def weights(self, arm, update):
        return (production.checkpoint_weights(self.native, arm, 0),
                production.checkpoint_weights(self.native, 'A', 0))


def full_diagnostic_quotes(cfg):
    quotes = {}
    for u in (2, 4, 6, 12, 24):
        backend = QuoteBackend(cfg, dry=True)
        options = production.monitor_config(cfg, root=Path(cfg['run_root'])/'prospective_quotes')
        options.update(isolated_registration=backend.registration, monitor_backend=backend,
                       monitor_dry_run=True)
        try:
            kind = 'probe' if u in (2, 4) else 'replicated'
            quotes[f'{kind}_u{u:03}'] = (p3.probe if kind == 'probe' else p3.replicated)(options, u)
        finally:
            backend.close()
    return quotes


def preflight_banks(cfg, clean1):
    """Real new-code readers/CE on disposable in-memory u0 models, no physics.

    This discovers unsupported B/R family support and evidence gaps before a new
    root or simulation attempt. Existing u0 bytes and clean1 policy state stay
    untouched; no optimization result from this engineering check is deployed.
    """
    from . import bank_breadth
    borrowed = deepcopy(cfg)
    for key in ('run_root','accounting_root','bank_source_root'):
        borrowed[key] = str(clean1)
    borrowed['control_arm']['root'] = str(clean1)
    borrowed['a_results']['root'] = str(clean1)
    borrowed['single_ledger']['root'] = str(clean1)
    borrowed['augmentation_prelaunch']['ledger_root'] = str(clean1)
    native = production.backend_config(borrowed)
    files = [clean1/f'{a}_{kind}_000.pt' for a in ('B','R') for kind in ('checkpoint','track2')]
    before = {str(p):rc.sha(p) for p in files}
    result = {}
    for arm in ('B','R'):
        runtime = p1.ArmRuntime(native,arm,1)
        supervision = production.bank_move_supervision(runtime)
        pool = bank_breadth.practice_pool(native,arm)
        if not pool or not supervision['contexts']:
            raise ValueError('nonempty verified breadth and trigger contexts required before physics')
        result[arm] = dict(trigger=supervision,practice_contexts=[dict(context_id=r['context_id'],
            family=r['family']) for r in pool])
    if before != {str(p):rc.sha(p) for p in files}:
        raise ValueError('read-only preflight changed preserved u0 state')
    return dict(arms=result,models_deployed=False,new_sim_s=0,preserved_u0_files=before)


def prepare(source, *, root=ROOT, clean1=CLEAN1):
    root, source, clean1 = map(Path, (root, source, clean1))
    for path in (root, source, clean1):
        rc.guard_path(path); rc.guard_path(path.resolve())
    if root == clean1 or root.exists():
        raise ValueError('new clean2 root required; inspect any previous preparation, never overwrite')
    stop_sha = stopped_clean1(clean1)
    setup = training.read('/home/ubuntu/cpuv_setup_20261105/S0_READY.json')
    parity = training.read('/home/ubuntu/cpuv_parity_20261105/PARITY_PASS.json')
    if not setup['pass_'] or not parity['pass_'] or parity['charged_sim_s'] != 960:
        raise ValueError('reuse qualified environment and exact SUMO parity')
    source_manifest = training.read(source/'SOURCE_MANIFEST.json')
    for name, sha in source_manifest['files'].items():
        if rc.sha(source/name) != sha:
            raise ValueError('immutable clean2 source differs: '+name)
    acceptance = training.read(source/'FIX_ACCEPTANCE.json')
    if set(acceptance) != {f'F{i}' for i in range(1, 7)}:
        raise ValueError('all six accepted fix receipts required')
    accepted_shas = {}
    for task, entry in acceptance.items():
        path = source/entry['path']
        if rc.sha(path) != entry['sha256']:
            raise ValueError('fix acceptance SHA differs: '+task)
        accepted_shas[task] = entry['sha256']
    cfg = restart_config(training.read(clean1/'CONFIG.json'), root, source, source_manifest)
    recipe = training.read(source/'F4_CONTINUATION_RECIPE.json')
    if recipe.get('schema') != 'coopmwm-f4-continuation-recipe-v1' or set(recipe['config_patch']) != {'f4_close_call'}:
        raise ValueError('accepted fixed F4 continuation recipe required')
    cfg.update(deepcopy(recipe['config_patch']))
    handoff = clean1/'plan_fix/PLAN_FIX_READY.json'
    if handoff.is_file():
        cfg['plan_fix_ready'] = str(handoff)
    preflight = preflight_banks(cfg,clean1)
    rule = training.read(clean1/rc.RULE_FILE)
    rc.init_root(root, amendment_shas=[rule['amendment_sha256']], track2=True)
    training.write(root/'PREPARATION_STARTED.json', dict(unix=time.time(),
        owner_quote='restart', source_root=str(source), accepted_fixes=accepted_shas,
        clean1_stop_sha256=stop_sha, new_sim_s=0))
    training.write(root/'BANK_PREFLIGHT.json', preflight)
    stage.copy_equal(clean1/rc.RULE_FILE, root/rc.RULE_FILE)
    # These are inherited owner authorities, not new clean2 safety failures.
    for arm, rule_name, pause_name in (('B', composite_health.B_RULE, composite_health.B_PAUSE),
                                       ('R', composite_health.R_RULE, composite_health.R_PAUSE)):
        if not composite_health.report_only(clean1, arm):
            raise ValueError('existing registered owner safety rule missing: '+arm)
        for name in (rule_name, pause_name):
            stage.copy_equal(clean1/name, root/name)
        amendment = training.read(clean1/rule_name)['amendment_sha256']
        with (root/'AUTHORITY.md').open('a', encoding='utf-8') as f:
            f.write('\nInherited exact owner '+arm+' composite report-only authority amendmentSHA'+amendment+
                '; original clean1 pause evidence copied solely to verify its authority; any collision/teleport still pauses.\n')
    for arm in ('B', 'R'):
        stage.readonly_bank_view(clean1/'replay_bank'/arm, root/'replay_bank'/arm)
    for arm in ('A', 'B', 'R'):
        stage.copy_equal(clean1/f'{arm}_checkpoint_000.pt', root/f'{arm}_checkpoint_000.pt')
        if rc.sha(root/f'{arm}_checkpoint_000.pt') != cfg['common_checkpoint_sha256']:
            raise ValueError('MAPPO u0 identity differs')
    native = production.backend_config(cfg)
    for arm in ('B', 'R'):
        torch.manual_seed(cfg['training_seed'])
        runtime = p1.ArmRuntime(dict(native, initial_sidecars_required=False), arm, 1)
        state = dict(arm=arm, update=0, worker=runtime.policy.worker_state(),
            layer_optimizer=runtime.policy.layer.optimizer.state_dict(),
            trigger_optimizer=runtime.policy.trigger_optimizer.state_dict())
        with (root/f'{arm}_track2_000.pt').open('xb') as f:
            torch.save(state, f)
    registration = _registration(clean1, root)
    quotes = full_diagnostic_quotes(cfg)
    gate_pairs = qualification.gate_pairs(cfg)
    gate_manifest = [dict(id=s['verification_id'], sim_s=s['sim_s'], window=s['window'],
        snapshot_sha256=s['snapshot_sha256']) for s, _ in gate_pairs]
    reporting = [s for s in gate_manifest if '_s000_' in s['id']]
    if not reporting or cfg['smoke']['total_sim_s']+sum(s['sim_s'] for s in reporting) > 20000:
        raise ValueError('full first-state B/R smoke reporting plus episodes exceeds smoke cap')
    cfg['clean2_smoke_reporting_quote'] = dict(selection='first prospectively registered state, B/R group/solos/common-base; full windows; descriptive only',
        branches=reporting, sim_s=sum(s['sim_s'] for s in reporting),
        complete_smoke_sim_s=cfg['smoke']['total_sim_s']+sum(s['sim_s'] for s in reporting))
    from .storage import ledger
    rows = ledger(clean1)
    # Lock-consistent prefix SHA: a concurrent plan-fix append may occur between
    # reads, so bind the exact canonical readback of this public prefix too.
    prefix = digest(rows)
    budget = clean2_budget.build(clean1_rows=rows, clean1_ledger_sha256=prefix,
        clean1_stop_sha256=stop_sha, accepted_fixes=accepted_shas,
        diagnostic_quotes=quotes, gate_manifest=gate_manifest, canonical_budget=cfg['budget'])
    cfg['clean2_budget'] = budget
    cfg['clean2_restart'] = dict(owner_quote='restart', clean1_root=str(clean1),
        clean1_ledger_prefix_sha256=prefix, source_commit=source_manifest['commit'],
        inherited_safety='A full original; B/R composite report-only; any collision/teleport pauses')
    # Run the actual final reader paths, including accepted breadth hooks. Every
    # selected input must exist and be covered by the new root's own manifest.
    actual = set()
    summaries = {}
    with rc._ReadRecorder([root, clean1, stage.LEGACY], Path('/home/ubuntu')) as recorder:
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
                for row in practice.examples(root, arm):
                    practice.load_snapshot(root, arm, row)
    actual.update(recorder.paths); actual.update(recorder.lexical)
    for row in registration['replicates']:
        for item in row['states']:
            actual.update(Path(item[k+'_path']) for k in ('snapshot', 'receipt'))
    prior_staging = training.read(clean1/'STAGING_MANIFEST.json')
    verify_sealed(prior_staging, 'manifest_sha256')
    actual.update(Path(p) for p in prior_staging['entries'])
    entries = {}
    for path in sorted(actual):
        rc.guard_path(path); rc.guard_path(path.resolve())
        if not path.is_file():
            raise ValueError('required clean2 reader input missing: '+str(path))
        entries[str(path)] = dict(sha256=rc.sha(path), bytes=path.stat().st_size,
                                 resolved_path=str(path.resolve()))
    for name, entry in prior_staging['entries'].items():
        if entries[name]['sha256'] != entry['sha256']:
            raise ValueError('preserved staged evidence changed: '+name)
    manifest = _seal(dict(schema='coopmwm.clean2.reader_staging.v1', entries=entries,
        bank_summaries=summaries, inherited_staging_sha256=prior_staging['manifest_sha256'],
        s0_manifest_sha256=setup['inventory_sha256']), 'manifest_sha256')
    training.write(root/'STAGING_MANIFEST.json', manifest)
    cfg.update(staging_manifest_sha256=rc.sha(root/'STAGING_MANIFEST.json'),
        isolated_registration_sha256=rc.sha(root/'ISOLATED_REGISTRATION.json'))
    training.write(root/'CLEAN2_BUDGET.json', budget)
    training.write(root/'CONFIG.json', cfg)
    approval = dict(approved=True, owner_quote='restart', config_sha256=digest(cfg),
        source_manifest_sha256=cfg['source_manifest_sha256'],
        budget_receipt_sha256=budget['receipt_sha256'], accepted_fixes=accepted_shas,
        created=datetime.now().astimezone().isoformat(),
        authority='REGISTRATION owner15:21 fix all before training; owner15:23 restart from u0 after all fixes+smoke+gate')
    training.write(root/'OWNER_APPROVAL.json', approval)
    training.write(root/'AUGMENTATION_DISABLED.json', _seal(dict(config_sha256=digest(cfg),
        admitted=dict(B=0, R=0), owner_decision=rc.AUGMENTATION_OFF_DECISION), 'receipt_sha256'))
    with (root/'AUTHORITY.md').open('a', encoding='utf-8') as f:
        f.write('\n'+json.dumps(dict(owner_approval=approval,
            owner_registration_quote='2026-11-05 15:23 Owner (chat, verbatim): "restart"; new root clean2 after F1-F6 + smoke + gate; budget = remaining within the 5M cap.',
            budget=budget, design='A plain MAPPO; B/R same accepted F1-F6 execution/reward/breadth, guidance differs; dense WM and WM credit OFF; augmentation OFF both',
            criterion=rc.criterion_document(), monitor=cfg['learning_monitor'],
            seeds='original registered TRAIN worlds only; full windows, u2/u4 probes; N5 u6/u12/u24; paired u8/u24',
            safety=cfg['clean2_restart']['inherited_safety'], clean1_preserved=True), sort_keys=True)+'\n')
    stopped_clean1(clean1)
    training.checked_config(root/'CONFIG.json')
    qualification.verify_inputs(cfg)
    training.write(root/'PREPARED.json', dict(status='STAGED_NOT_LAUNCHED', config_sha256=digest(cfg),
        files=len(entries), remaining_cap_sim_s=budget['budget']['total'],
        complete_allocation_sim_s=budget['budget']['allocated'], new_sim_s=0))
    return cfg


def verify_previous(cfg):
    """Admission-time audit: only reserved plan-fix may have appended in clean1."""
    from .storage import ledger
    previous = Path(cfg['clean2_restart']['clean1_root'])
    bound = clean2_budget.checked(cfg)
    if stopped_clean1(previous) != bound['clean1_stop_sha256']:
        raise ValueError('clean1 durable stop changed since clean2 registration')
    rows = ledger(previous)
    size = bound['clean1_rows']
    if len(rows) < size or digest(rows[:size]) != bound['clean1_ledger_sha256']:
        raise ValueError('preserved clean1 ledger prefix changed')
    if any(row['phase'] != 'plan_fix' for row in rows[size:]):
        raise ValueError('unexpected clean1 simulation after clean2 budget registration')
    if sum(row['sim_s'] for row in rows if row['phase'] == 'plan_fix') > clean2_budget.PLAN_FIX_CAP:
        raise ValueError('concurrent plan_fix cap exceeded')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', required=True)
    args = p.parse_args()
    print(digest(prepare(args.source_root)))


if __name__ == '__main__':
    main()
