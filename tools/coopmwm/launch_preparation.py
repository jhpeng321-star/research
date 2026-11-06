"""Zero-physics clean1 staging, real-reader closure and prospective authority."""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import shutil

import torch

from . import launch_production as production, rerun_cpus as rc, track2_backend as p1
from . import practice_starts as ps, replay_bank as rb, protocol
from .exact_grade import _seal
from .launch_training import write, read
from .protocol import digest


LEGACY = Path('/home/ubuntu/coopmwm20261104_v2v_coop3')


def readonly_bank_view(source, target, *, resume=False):
    """New writable metadata directories; existing evidence linked read-only.

    Never symlink an entire bank (practice plans and new admissions belong to
    this experiment). Original evidence paths/hashes remain authoritative.
    """
    source, target = Path(source), Path(target)
    if target.exists() and not resume:
        raise ValueError('bank view already exists')
    target.mkdir(parents=True, exist_ok=resume)
    for current, dirs, files in os.walk(source, followlinks=True):
        relative = Path(current).relative_to(source)
        directory = target/relative
        directory.mkdir(parents=True, exist_ok=True)
        allowed = []
        for name in dirs:
            try:
                rc.guard_path(Path(current)/name)
                rc.guard_path((Path(current)/name).resolve())
            except ValueError:
                continue
            if name in ('scenarios', 'claims', 'requests', 'responses', '__pycache__'):
                continue
            allowed.append(name)
        dirs[:] = allowed
        for name in files:
            if name.startswith(('PRACTICE_PLAN_', 'LESSON_LIST_')) or name.endswith('.claim'):
                continue
            path = Path(current)/name
            try:
                rc.guard_path(path); rc.guard_path(path.resolve())
            except ValueError:
                continue
            if not path.is_file():
                continue
            destination = directory/name
            if destination.exists() or destination.is_symlink():
                if not resume or destination.resolve() != path.resolve():
                    raise ValueError('unfinished bank view differs from original source')
            else:
                destination.symlink_to(path.resolve())


def _snapshot_registration(root, cfg):
    """Freeze states by metadata only, before any clean1 outcome is measured."""
    contexts = defaultdict(list)
    selected_inputs = set()
    unresolved_nonselected = set()
    links = ps._links(root, 'B')
    decisions = rb.bank_decisions(root, 'B', sources=cfg.get('replay_sources', {}).get('B', ()))
    for receipt in decisions:
        group = (receipt.get('chosen_plan') or {}).get('members', ())
        if not 2 <= len(group) <= 4 or receipt.get('window', {}).get('truncated'):
            continue
        # Missing snapshot provenance for historical unconfirmed LS labels is
        # disclosed in S0. Such labels are not diagnostic states. Bank imitation
        # still reads all admitted tapes, independently traced below.
        try:
            with rc._ReadRecorder([root], Path('/home/ubuntu')) as recorder:
                snapshot = rb._decision_snapshot(root, 'B', receipt, links)
        except (FileNotFoundError, StopIteration, KeyError):
            unresolved_nonselected.update(p for p in set(recorder.paths)|set(recorder.lexical)
                                          if not p.is_file())
            continue
        seed = snapshot['episode']['world']['seed']
        from .map_runtime import primary_train_world  # N1: merge 986000-986071 or the map's TRAIN registry
        if snapshot['episode'].get('role') != 'train' or not primary_train_world(snapshot['episode']['world']):
            raise ValueError('diagnostics must use original registered primary TRAIN worlds')
        contexts[seed].append((snapshot, receipt, set(recorder.paths)|set(recorder.lexical)))
    ordered = sorted(contexts, key=lambda s: (-len(contexts[s]), s))
    if len(ordered) < 5 or len(contexts[ordered[0]]) < 3:
        raise ValueError('five paired TRAIN worlds and three first-world probe states required')
    from .track2_monitor import diagnostic_identity
    rows = []
    for index, seed in enumerate(ordered[:5]):
        entries = contexts[seed][:3 if index == 0 else 1]
        states, files = [], []
        for number, (snapshot, receipt, inputs) in enumerate(entries):
            selected_inputs.update(inputs)
            directory = root/'registered_states'/f'r{index:02}_s{number:02}'
            directory.mkdir(parents=True, exist_ok=True)
            path = directory/'SNAPSHOT.pt'
            if path.exists():
                from .sumo_verify import snapshot_sha
                if snapshot_sha(torch.load(path, weights_only=False)) != snapshot_sha(snapshot):
                    raise ValueError('unfinished registered snapshot differs')
            else:
                with path.open('xb') as stream:
                    torch.save(snapshot, stream)
            receipt_path = directory/'RECEIPT.json'
            stage_write(receipt_path, receipt)
            files.append(dict(snapshot_path=str(path), snapshot_file_sha256=rc.sha(path),
                receipt_path=str(receipt_path), receipt_file_sha256=rc.sha(receipt_path)))
            states.append((snapshot, receipt))
        rows.append(dict(replicate=index, world_seed=seed, states=files, **diagnostic_identity(states)))
    worlds = protocol.worlds('train', 0)+protocol.worlds('train', 1)[:24]
    registration = dict(role='train', metric=rc.ISOLATED_TEST['metric'], replicates=rows,
        selection='five original TRAIN worlds ordered by descending resolvable multi-CAV context count, then seed; first world three states, others first state; no clean1 outcomes',
        final_evaluation=dict(role='train', worlds=[dict(world_id=str(w['seed']), world=w) for w in worlds]))
    rc.isolated_jobs(registration)
    rc.final_world_registration(registration)
    return registration, selected_inputs, unresolved_nonselected


def successful_reader_inputs(recorder, selected_inputs, unresolved_nonselected):
    """Only caught, nonselected missing-context attempts may leave the closure.

    A required bank/practice read fails before reaching this boundary. A selected
    diagnostic dependency may never be excluded. The historical unavailable
    contexts remain disclosed in the final staging receipt.
    """
    actual = set(recorder.paths)|set(recorder.lexical)|set(selected_inputs)
    unresolved = set(unresolved_nonselected)
    if unresolved & set(selected_inputs) or any(p.is_file() for p in unresolved):
        raise ValueError('a selected/present reader dependency cannot be marked unavailable')
    for path in actual-unresolved:
        if not path.is_file():
            raise ValueError('required real-reader input missing: '+str(path))
    return actual-unresolved


def stage_write(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError('unfinished preparation receipt differs: '+str(path))
    else:
        write(path, value)


def copy_equal(source, destination):
    if Path(destination).exists():
        if rc.sha(source) != rc.sha(destination):
            raise ValueError('unfinished staged copy differs: '+str(destination))
    else:
        shutil.copyfile(source, destination)


def prepare(source_root, *, root='/home/ubuntu/coopmwm20261105_clean1', resume=False):
    root, source = Path(rc.guard_path(root)), Path(rc.guard_path(source_root))
    setup = read('/home/ubuntu/cpuv_setup_20261105/S0_READY.json')
    parity = read('/home/ubuntu/cpuv_parity_20261105/PARITY_PASS.json')
    if not setup['pass_'] or not parity['pass_'] or parity['charged_sim_s'] != 960:
        raise ValueError('reuse qualified cpuV environment and exact parity before staging')
    if root.exists() and not resume:
        raise ValueError('new clean1 root required; never repeat preparation')
    if resume:
        rc.own_ledger(root, track2=True)
        if ((root/'PREPARED.json').exists() or (root/'OWNER_APPROVAL.json').exists()
                or (root/'CHARGES.jsonl').read_text().strip() or (root/'updates').exists()):
            raise ValueError('only an unapproved zero-charge unfinished preparation may resume')
        write(root/f'PREPARATION_RECOVERY_{datetime.now().strftime("%H%M%S")}.json',
            dict(source_root=str(source), reason='explicit zero-physics preparation recovery', new_sim_s=0))
    base = read(LEGACY/'V2V_RUNTIME.json')['config']
    common = LEGACY/'B_checkpoint_000.pt'
    cfg = rc.runtime_config(base, root=root, reference_root=LEGACY,
        source_root=source, common_checkpoint_sha256=rc.sha(common), track2=True)
    cfg.update(launch_optimization=deepcopy(production.OPTIMIZATION),
        launch_contingency=deepcopy(production.CONTINGENCY),
        initial_sidecars_required=True, load_max=40, augmentation_layout='own_bank',
        augmentation_accounting_scheme='native_directory_v2',
        augmentation_recipe_version=2,
        training_seed=20261101,
        worlds_by_update={str(u): protocol.batch_worlds(u)[:16] for u in range(1, 25)},
        augmentation_prelaunch=dict(provenance='owner MAIN 02:30: SUMO-verified children only; before clean1 launch',
            allocation_sim_s=rc.VERIFICATION['augmentation_total_sim_s'], phase='augment_verify', ledger_root=str(root)))
    from .launch_augmentation import PROPOSAL_SETTINGS
    cfg['augmentation_proposals'] = deepcopy(PROPOSAL_SETTINGS)
    source_manifest = read(source/'SOURCE_MANIFEST.json')
    for name, sha in source_manifest['files'].items():
        if rc.sha(source/name) != sha:
            raise ValueError('deployed source SHA mismatch')
    cfg['source_manifest_sha256'] = digest(source_manifest)
    rule = read(LEGACY/rc.RULE_FILE)
    if not resume:
        rc.init_root(root, amendment_shas=[rule['amendment_sha256']], track2=True)
    for arm in ('B', 'R'):
        readonly_bank_view(rb.bank_root(LEGACY, arm), rb.bank_root(root, arm), resume=resume)
    for arm in ('A', 'B', 'R'):
        copy_equal(common, root/f'{arm}_checkpoint_000.pt')
        if rc.sha(root/f'{arm}_checkpoint_000.pt') != cfg['common_checkpoint_sha256']:
            raise ValueError('common MAPPO u0 SHA mismatch')
    copy_equal(LEGACY/rc.RULE_FILE, root/rc.RULE_FILE)
    native = production.backend_config(cfg)
    for arm in ('B', 'R'):
        torch.manual_seed(cfg['training_seed'])
        # This is the actual P1 build used by update1, with an explicit u0
        # sidecar so gate/smoke/training never initialise different heads.
        runtime = p1.ArmRuntime(dict(native, initial_sidecars_required=False), arm, 1)
        state = dict(arm=arm, update=0, worker=runtime.policy.worker_state(),
            layer_optimizer=runtime.policy.layer.optimizer.state_dict(),
            trigger_optimizer=runtime.policy.trigger_optimizer.state_dict())
        path = root/f'{arm}_track2_000.pt'
        if path.exists():
            old = torch.load(path, weights_only=False)
            if old['arm'] != arm or old['update'] != 0:
                raise ValueError('unfinished initial sidecar identity mismatch')
            runtime.policy.restore_worker(old['worker'])
        else:
            with path.open('xb') as stream:
                torch.save(state, stream)
    with rc._ReadRecorder([root, LEGACY], Path('/home/ubuntu')) as recorder:
        summaries = {}
        for arm in ('B', 'R'):
            with p1._bank_view(native) as bank:
                pairs, summary = bank.bank_pairs(root, arm, cfg['training_seed'],
                    sources=cfg.get('replay_sources', {}).get(arm, ()), v2v=True,
                    v2v_intent=cfg, imitation_bank_filter='cooperative_only', imitation_target_space='effective')
                if not pairs:
                    raise ValueError('nonempty own verified effective-imitation bank required')
                summaries[arm] = summary
                # Snapshot loading follows the REAL practice sampler's eligible
                # set. Its dependencies are required even when no smoke chooses it.
                for row in ps.examples(root, arm):
                    ps.load_snapshot(root, arm, row)
        registration, selected_inputs, unresolved_nonselected = _snapshot_registration(root, cfg)
    actual = successful_reader_inputs(recorder, selected_inputs, unresolved_nonselected)
    # P4 expands only snapshot reader roots. Historical metadata reads do not
    # imply that every rejected/SLIM candidate tape or unusable LS snapshot is
    # opened by training. Preserve all actual metadata reads in the SHA inventory.
    closure_roots = [p for p in actual if p.parent.name in ('pending', 'decisions') or
                     p.name == 'RESULT.json' or p.suffix == '.pt']
    expanded, closure = production.staging_closure(closure_roots)
    actual.update(expanded)
    for p in (common, LEGACY/rc.RULE_FILE):
        actual.add(p)
    entries = {}
    for path in sorted(actual):
        rc.guard_path(path); rc.guard_path(path.resolve())
        if not path.is_file():
            raise ValueError('required reader input missing during final staging')
        entries[str(path)] = dict(sha256=rc.sha(path), bytes=path.stat().st_size,
                                  resolved_path=str(path.resolve()))
    manifest = _seal(dict(schema='coopmwm.clean1.reader_staging.v1', entries=entries,
        closure=closure, bank_summaries=summaries, s0_manifest_sha256=setup['inventory_sha256'],
        unresolved_nonselected_snapshot_attempts=sorted(map(str, unresolved_nonselected))), 'manifest_sha256')
    write(root/'STAGING_MANIFEST.json', manifest)
    write(root/'ISOLATED_REGISTRATION.json', registration)
    cfg.update(staging_manifest_sha256=rc.sha(root/'STAGING_MANIFEST.json'),
               isolated_registration_sha256=rc.sha(root/'ISOLATED_REGISTRATION.json'))
    allocations = production.backend_config(cfg)['budget']['phases']
    quote = dict(base=rc.quote(track2=True), prior_parity_sim_s=960,
        contingency=production.CONTINGENCY, allocations=allocations,
        cumulative_upper_bound_sim_s=sum(allocations.values())+960, owner_cap_sim_s=5000000,
        maximum_smoke_sim_s=20000, host='cpuV')
    assert quote['cumulative_upper_bound_sim_s'] <= 5000000
    write(root/'PRODUCTION_QUOTE.json', quote)
    write(root/'CONFIG.json', cfg)
    approval = dict(approved=True, config_sha256=digest(cfg), source_manifest_sha256=cfg['source_manifest_sha256'],
        quote_sha256=rc.sha(root/'PRODUCTION_QUOTE.json'), owner_quote='approve the cut',
        authority='REGISTRATION.md owner 01:39 cap5M including smoke, 02:28 redesign, 03:33 S-LAUNCH finishes wiring; current S-LAUNCH launch/smoke/gate instruction',
        created=datetime.now().astimezone().isoformat())
    write(root/'OWNER_APPROVAL.json', approval)
    with (root/'AUTHORITY.md').open('a', encoding='utf-8') as stream:
        stream.write('\n'+json.dumps(dict(owner_approval=approval, quote=quote,
            config_sha256=digest(cfg), design=dict(arms=['A','B','R'], dense_teacher=False,
            wm_group_credit=False, real_observable_team_reward=cfg['track2_reward'],
            augmentation='only native all8 SUMO-admitted children, once prelaunch',
            practice_weighting=cfg['practice_weighting'], optimization=cfg['launch_optimization']),
            training_worlds='protocol.worlds(train,0/1) only; online training index0; no evaluation/forbidden seeds',
            criterion=rc.criterion_document(), learning_monitor=cfg['learning_monitor'],
            safety='zero collisions and teleports gate; per-update paired A composite before optimization',
            outcome_boundary='P2 post-state speed-deficit occupancy: measured disappearance from public SUMO observations ends that CAV lifetime with zero terminal occupancy reward; no missing-frame fallback',
            prior_charges='parity960 lives in cpuv_parity_20261105, subtracted from cumulative cap exactly once; no cpuQ ledger'),
            sort_keys=True)+'\n')
    write(root/'PREPARED.json', dict(status='STAGED_NOT_LAUNCHED', config_sha256=digest(cfg),
        quote=quote, files=len(entries), new_sim_s=0))
    return cfg
