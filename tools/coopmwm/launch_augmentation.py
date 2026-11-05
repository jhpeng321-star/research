"""Once-only SUMO-qualified T2G proposals, own-ledger all8 admission.

After the 02:30 redesign, WM scores cannot admit a plan. The accepted T2G
perturbation operator proposes variants; native full-window all8/SUMO decides
every bank/practice admission. Failed/absent grades never become examples.
"""
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from . import launch_production as production, rerun_cpus as rc, track2_backend as p1
from . import replay_bank as rb, practice_starts as ps, plan_augment, bank_v2v, moves
from .exact_grade import _seal, verify_sealed
from .launch_training import write, read, health
from .protocol import digest


PROPOSAL_SETTINGS = dict(parameter_fraction=.01, partner_swap_probability=0., attempts_per_plan=2,
    target_variants_per_arm=4, continuation_variants_per_arm=6, parent_cycles=3,
    source_reader='complete same-arm verified effective training bank, not practice-only subset',
    observation='decision-second direct sensing clipped to 100m; no relayed/global inputs',
    provenance='clean1 proposed small-perturbation radius; same B/R, fixed count by registered recipe, measured prefix retained; full nonzero native-plan bank including none-labelled templates; no WM admission')


def template_parameters(plan):
    """Fit source-template coordinates without inventing an online move label.

    A nearest Move of none does NOT mean the verified native plan has zero
    offsets: the move library only has negative boxes, while native plans can
    accelerate or oscillate. Fit either signed box as an offline normalization
    coordinate and preserve the original signed waveform in T2G perturbation.
    The online move vocabulary/decoder and every SUMO admission gate stay fixed.
    """
    label = moves.from_plan(plan)
    if label.family != 'none':
        return dict(onset_s=label.onset, hold_s=label.hold, depth=label.depth)
    residual = np.asarray(plan['perturbation_profile'], float)
    if not np.isfinite(residual).all() or not np.any(residual):
        return None
    actors = tuple(range(len(plan['members'])))
    fits = [moves._project_window(sign*residual, actors) for sign in (1., -1.)]
    parameters, _ = max(fits, key=lambda value:value[1])
    return dict(zip(('onset_s','hold_s','depth'), parameters))


def full_bank_sources(cfg, arm):
    """Freeze every eligible source from the real, same-arm training-bank reader."""
    from .sumo_verify import snapshot_sha
    from .launch_preparation import stage_write
    root, native = Path(cfg['run_root']), production.backend_config(cfg)
    suffix = '_v3' if cfg.get('augmentation_recipe_version') == 3 else ''
    path = root/f'AUGMENTATION_PARENTS_{arm}{suffix}.json'
    if path.exists():
        binding = read(path);verify_sealed(binding, 'parent_bank_sha256')
        if (binding['arm'] != arm or 'augmentationParentBankSHA'+binding['parent_bank_sha256']
                not in (root/'AUTHORITY.md').read_text()):
            raise ValueError('foreign frozen augmentation parent bank')
        if any(rc.sha(entry['snapshot_path']) != entry['base_snapshot_file_sha256'] or
               digest(entry['plan']) != entry['source_plan_sha256'] for entry in binding['parents']):
            raise ValueError('frozen training-bank source identity changed')
        return binding['parents']
    entries, coverage = [], []
    with p1._bank_view(native) as bank:
        own = {r['example_id']:r for r in ps.examples(root, arm)}
        links = ps._links(root, arm)
        def consume(*, receipt, example, plan, pairs, trace):
            if not pairs or example is None:
                return
            coverage.append(dict(example_id=example['example_id'], members=len(plan['members'])))
            if links.get(receipt['verification_id'], {}).get('augmentation'):
                return  # once-only augmentation never recursively augments its own children
            if not 2 <= len(plan['members']) <= 4:
                return
            parameters = template_parameters(plan)
            if parameters is None:
                return
            base = rb._decision_snapshot(root, arm, receipt, links)
            seed_index = int(example['seed_index'])
            snapshot = (rb.replicate(base, receipt['verification_id']) if seed_index == 2 else
                rb.replicate(base, receipt['verification_id']+f'#s{seed_index}') if seed_index > 2 else base)
            if (snapshot['episode'].get('role') != 'train' or snapshot['episode'].get('arm') != arm or
                    not 986000 <= snapshot['episode']['world']['seed'] <= 986071):
                raise ValueError('same-arm registered TRAIN source required')
            frames = bank_v2v.trace_frames(trace)
            context = proposal_context(plan, frames, snapshot['t0'])
            if context is None:
                return
            parent = own.get(example['example_id'])
            if parent is None:
                state = 'augparent_'+arm+'_'+digest(dict(example_id=example['example_id']))[:16]
                target = rb.bank_root(root, arm)/'snapshots'/(state+'.pt')
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if snapshot_sha(torch.load(target, weights_only=False, map_location='cpu')) != snapshot_sha(base):
                        raise ValueError('imported parent snapshot changed')
                else:
                    with target.open('xb') as stream:torch.save(base, stream)
                parent = dict(verification_id=receipt['verification_id'], example_id=example['example_id'],
                    seed_index=seed_index, salt=receipt['verification_id'], state_id=state, fallback=False)
                stage_write(rb.bank_root(root, arm)/'decisions'/(receipt['verification_id']+'.json'),
                    dict(verification_id=receipt['verification_id'], state_id=state, policy_update=3,
                         augmentation_parent_only=True, source_receipt_sha256=receipt['receipt_sha256']))
            base_path = rb.bank_root(root, arm)/'snapshots'/(parent['state_id']+'.pt')
            entries.append(dict(row=parent, base_snapshot_file_sha256=rc.sha(base_path),
                snapshot_sha256=snapshot_sha(snapshot), source_receipt={k:v for k,v in receipt.items() if not k.startswith('_')},
                source_plan_sha256=digest(plan), plan=deepcopy(plan), context=context,
                snapshot_path=str(base_path), parameters=parameters))
        bank.bank_pairs(root, arm, cfg['training_seed'], sources=cfg.get('replay_sources', {}).get(arm, ()),
            v2v=True, v2v_intent=cfg, imitation_bank_filter='cooperative_only',
            imitation_target_space='effective', example_consumer=consume)
    if not entries:
        raise ValueError('full same-arm training bank has no observable eligible augmentation parents')
    binding = _seal(dict(arm=arm, reader='replay_bank.bank_pairs:cooperative_only:effective',
        coverage=coverage, parents=entries), 'parent_bank_sha256')
    write(path, binding)
    with (root/'AUTHORITY.md').open('a') as stream:
        stream.write('\nDerived same-arm augmentation source bank augmentationParentBankSHA'+binding['parent_bank_sha256']+'\n')
    return entries


def example_plan(example, receipt):
    """The exact reviewed bank-reader fallback for legacy per-seed overlays."""
    plan = example.get('plan')
    if plan is None:
        plan = receipt.get('chosen_plan')
    if not isinstance(plan, dict) or not plan.get('members') or 'profile' not in plan:
        raise ValueError('source example needs its own or legacy verified chosen plan')
    return plan


def proposal_context(plan, frames, time):
    """An acting CAV's current own sensing is legal even at t300.

    No partner packet is fabricated when the stored tape begins at t300. Only
    directly measured vehicles within that ego's 100m sensor reach enter T2G.
    Full six-member legacy plans are not silently shrunk to the new 1..4 API.
    """
    if not 2 <= len(plan['members']) <= 4:
        return None
    frame = frames.get(time)
    if frame is None or frame.get('time_s') != time:
        return None
    ego, visible = None, None
    for candidate in plan['members']:
        own = next((r for r in frame['vehicles'] if r['vid'] == candidate), None)
        if own is None:
            continue
        sensed = [r for r in frame['vehicles'] if abs(r['s_m']-own['s_m']) <= 100.]
        if set(plan['members']) <= {r['vid'] for r in sensed if r['is_cav']}:
            ego, visible = candidate, sensed
            break
    if ego is None:
        return None
    rows = [dict(vid=r['vid'], is_cav=r['is_cav'], s_m=r['s_m'], v_mps=r['v_mps'],
        lane=r['lane'], allowed_speed_mps=31.29, observed_at_s=time) for r in visible]
    centre = np.asarray(plan['profile'])-np.asarray(plan['perturbation_profile'])
    return dict(ego_id=ego, t_s=time, rows=rows,
        policy_traj={v: centre[i].tolist() for i,v in enumerate(plan['members'])})


class MergeEncoder:
    def encode(self, obs):
        from .track2_reward import _sensor_packet
        from .merge_features import merge_features, MergeFeaturesConfig
        own = obs['own_sensing']
        packet = _sensor_packet(own['vehicles'], own['ego_id'], obs['time_s'])
        return merge_features(packet, own['ego_id'], now_s=obs['time_s'],
                              cfg=MergeFeaturesConfig(True)).tolist()


def proposals(cfg, arm):
    if cfg.get('augmentation_recipe_version') in (2, 3):
        return full_bank_proposals(cfg, arm)
    if cfg.get('augmentation_proposals') != PROPOSAL_SETTINGS:
        raise ValueError('approval-bound common B/R augmentation settings required')
    root, native = Path(cfg['run_root']), production.backend_config(cfg)
    cache = root/'augmentation_sources'/f'PROPOSALS_{arm}.json'
    if cache.exists():
        frozen = read(cache); verify_sealed(frozen, 'proposal_batch_sha256')
        if (frozen['arm'] != arm or frozen['settings'] != PROPOSAL_SETTINGS or
                frozen['training_seed'] != cfg['training_seed'] or len(frozen['proposals']) != 2):
            raise ValueError('frozen augmentation proposal registration changed')
        for item in frozen['proposals']:
            path = Path(item['snapshot_path'])
            rc.guard_path(path)
            if not path.resolve().is_relative_to((root/'augmentation_sources').resolve()) or rc.sha(path) != item['snapshot_file_sha256']:
                raise ValueError('attempted augmentation snapshot changed; never overwrite/replay')
        return frozen['proposals']
    output, seen = [], set()
    with p1._bank_view(native):
        confirmations = rb.confirmations(root, arm)
        decisions = {r['verification_id']:r for r in rb.bank_decisions(root, arm,
            sources=cfg.get('replay_sources', {}).get(arm, ()))}
        parents = ps.examples(root, arm)
        for row in parents:
            if len(output) == 2:
                break
            record = confirmations[row['verification_id']]
            example = next(e for e in record['per_seed_v1']['examples'] if e['example_id'] == row['example_id'])
            plan = example_plan(example, decisions[row['verification_id']])
            if not 2 <= len(plan['members']) <= 4:
                continue
            label = moves.from_plan(plan)
            if label.family == 'none':
                continue
            snapshot = ps.load_snapshot(root, arm, row)
            trace = Path(example['tape']['branches_dir'])/example['tape']['verification_id']/example['tape']['branch']['branch']/'observable.jsonl.gz'
            frames = bank_v2v.trace_frames(trace)
            t = snapshot['t0']
            context = proposal_context(plan, frames, t)
            if context is None:
                continue
            settings = dict(parameter_fraction=PROPOSAL_SETTINGS['parameter_fraction'],
                partner_swap_probability=PROPOSAL_SETTINGS['partner_swap_probability'],
                variety=dict(plan_augment.variety.VARIETY_REGISTERED), bounds=(-3., 1.2))
            parameters = dict(onset_s=label.onset, hold_s=label.hold, depth=label.depth)
            for attempt in range(PROPOSAL_SETTINGS['attempts_per_plan']):
                if len(output) == 2:
                    break
                candidate = plan_augment._perturb(plan, parameters, context, settings,
                    np.random.default_rng(cfg['training_seed']+len(output)+attempt))
                if candidate is None:
                    continue
                modified, perturbed = candidate
                fingerprint = digest(modified)
                if fingerprint in seen:
                    continue
                ident = f'clean1_aug_{arm}_{len(output):02}'
                directory = root/'augmentation_sources'/ident
                directory.mkdir(parents=True, exist_ok=True)
                path = directory/'SNAPSHOT.pt'
                if path.exists():
                    from .sumo_verify import snapshot_sha
                    if snapshot_sha(torch.load(path, map_location='cpu', weights_only=False)) != snapshot_sha(snapshot):
                        raise ValueError('attempted augmentation snapshot changed; never overwrite/replay')
                else:
                    with path.open('xb') as stream:
                        torch.save(snapshot, stream)
                output.append(dict(id=ident, proposal_id=ident, source_id=row['example_id'],
                    parent_example_id=row['example_id'], arm=arm, seed=snapshot['episode']['world']['seed'],
                    plan=modified, source_plan_sha256=digest(plan), parameters=perturbed,
                    snapshot_path=str(path), snapshot_file_sha256=rc.sha(path),
                    window=dict(start_s=t, end_s=1200, truncated=False)))
                seen.add(fingerprint)
    if len(output) != 2:
        raise ValueError('two distinct observable cooperative variants from the own bank required per arm')
    write(cache, _seal(dict(arm=arm, settings=PROPOSAL_SETTINGS, training_seed=cfg['training_seed'],
        originating_config_sha256=digest(cfg), proposals=output), 'proposal_batch_sha256'))
    return output


def full_bank_proposals(cfg, arm):
    """Fixed symmetric four-candidate slate; preserve any registered prefix."""
    from .sumo_verify import snapshot_sha
    root = Path(cfg['run_root'])
    version = cfg.get('augmentation_recipe_version')
    cache = root/'augmentation_sources'/f'PROPOSALS_{arm}_v{version}.json'
    target = PROPOSAL_SETTINGS['continuation_variants_per_arm'] if version == 3 else PROPOSAL_SETTINGS['target_variants_per_arm']
    if cfg.get('augmentation_proposals') != PROPOSAL_SETTINGS:
        raise ValueError('full-bank augmentation registration required')
    if cache.exists():
        frozen = read(cache);verify_sealed(frozen, 'proposal_batch_sha256')
        if frozen['settings'] != PROPOSAL_SETTINGS or frozen['training_seed'] != cfg['training_seed']:
            raise ValueError('frozen full-bank proposal registration changed')
        output = frozen['proposals']
    else:
        output = []
        resume_path = root/'AUGMENTATION_RESUME.json'
        resume = read(resume_path) if resume_path.exists() else {}
        authorized = resume.get('previous_proposal_batches', {}).get(arm, {})
        previous = Path(authorized.get('path', root/'augmentation_sources'/f'PROPOSALS_{arm}.json'))
        if previous.exists():
            verify_sealed(resume, 'receipt_sha256')
            if authorized.get('file_sha256') != rc.sha(previous):
                raise ValueError('previous measured proposal prefix needs explicit adoption binding')
            old = read(previous);verify_sealed(old, 'proposal_batch_sha256')
            output = deepcopy(old['proposals'])
        if version == 3 and len(output) != 4:
            raise ValueError('six-candidate continuation requires the four measured candidates per arm as immutable prefix')
        parents = full_bank_sources(cfg, arm)
        seen = {digest(p['plan']) for p in output}
        for cycle in range(PROPOSAL_SETTINGS['parent_cycles']):
            for source in parents:
                row, plan = source['row'], source['plan']
                have = sum(p['parent_example_id'] == row['example_id'] for p in output)
                if have >= (cycle+1)*PROPOSAL_SETTINGS['attempts_per_plan']:
                    continue
                snapshot = ps.load_snapshot(root, arm, row)
                if snapshot_sha(snapshot) != source['snapshot_sha256']:
                    raise ValueError('source example CRN snapshot changed before augmentation')
                settings = dict(parameter_fraction=PROPOSAL_SETTINGS['parameter_fraction'],
                    partner_swap_probability=PROPOSAL_SETTINGS['partner_swap_probability'],
                    variety=dict(plan_augment.variety.VARIETY_REGISTERED), bounds=(-3., 1.2))
                for attempt in range(PROPOSAL_SETTINGS['attempts_per_plan']):
                    if len(output) == target:break
                    candidate = plan_augment._perturb(plan, source['parameters'], source['context'], settings,
                        np.random.default_rng(cfg['training_seed']+len(output)+attempt+cycle*target))
                    if candidate is None:continue
                    modified, parameters = candidate
                    if digest(modified) in seen:continue
                    ident = f'clean1_aug_{arm}_{len(output):02}'
                    directory = root/'augmentation_sources'/ident;directory.mkdir(parents=True, exist_ok=True)
                    path = directory/'SNAPSHOT.pt'
                    if path.exists():
                        if snapshot_sha(torch.load(path, weights_only=False, map_location='cpu')) != snapshot_sha(snapshot):
                            raise ValueError('frozen proposal source snapshot changed')
                    else:
                        with path.open('xb') as stream:torch.save(snapshot, stream)
                    output.append(dict(id=ident, proposal_id=ident, source_id=row['example_id'],
                        parent_example_id=row['example_id'], arm=arm, seed=snapshot['episode']['world']['seed'],
                        plan=modified, source_plan_sha256=source['source_plan_sha256'], parameters=parameters,
                        snapshot_path=str(path), snapshot_file_sha256=rc.sha(path),
                        window=dict(start_s=snapshot['t0'], end_s=1200, truncated=False)))
                    seen.add(digest(modified))
                if len(output) == target:break
            if len(output) == target:break
        if len(output) != target:
            raise ValueError('complete fixed distinct observable variant slate required per arm')
        write(cache, _seal(dict(arm=arm, settings=PROPOSAL_SETTINGS, training_seed=cfg['training_seed'],
            proposals=output), 'proposal_batch_sha256'))
    if len(output) != target or any(p['arm'] != arm or not 2 <= len(p['plan']['members']) <= 4 or
            not Path(p['snapshot_path']).resolve().is_relative_to((root/'augmentation_sources').resolve()) or
            rc.sha(p['snapshot_path']) != p['snapshot_file_sha256'] for p in output):
        raise ValueError('full-bank frozen slate identity changed')
    return output


def slate_quote(cfg, slates):
    """Conservative complete all8 quote; all already-measured outcomes retained."""
    from .storage import ledger
    from . import sumo_verify as sv
    root = Path(cfg['run_root'])
    count = (PROPOSAL_SETTINGS['continuation_variants_per_arm'] if cfg.get('augmentation_recipe_version') == 3
             else PROPOSAL_SETTINGS['target_variants_per_arm'])
    if set(slates) != {'B','R'} or any(len(entries) != count for entries in slates.values()):
        raise ValueError('matched fixed B/R all8 slate required')
    spent = sum(r['sim_s'] for r in ledger(root) if r['phase'] == 'augment_verify')
    pending, completed = 0, []
    for arm, entries in slates.items():
        for item in entries:
            snap = torch.load(item['snapshot_path'], weights_only=False, map_location='cpu')
            if snap['episode'].get('arm') != arm or snap['episode'].get('role') != 'train':
                raise ValueError('same-arm TRAIN snapshot required in all8 slate quote')
            vid = sv.verification_id(snap['episode']['id'], snap['t0'], [item['plan']])
            path = rb.bank_root(root, arm)/'confirmations_all8'/(vid+'.json')
            if path.exists():
                record = read(path);verify_sealed(record, 'confirmation_sha256')
                completed.append(record['confirmation_sha256'])
            else:
                # First+second seeds, two later-harm checks, and each seed's
                # policy+all solos. Every branch charged at full1200 ceiling.
                pending += (10+2*len(item['plan']['members']))*1200
    ceiling = cfg['augmentation_prelaunch']['allocation_sim_s']
    if spent+pending > ceiling or ceiling > 200000:
        raise ValueError('complete fixed all8 slate exceeds unchanged prelaunch allocation')
    return dict(per_arm=count, already_charged_sim_s=spent, additional_upper_bound_sim_s=pending,
        total_upper_bound_sim_s=spent+pending, allocation_sim_s=ceiling, completed_receipts=completed)


def _install(root, arm, payload, result):
    """Add actual native receipts/links, retaining every original evidence path."""
    bank = rb.bank_root(root, arm)
    record = result['receipt']
    if not result['examples']:
        return []
    child = rb.bank_root(Path(result['examples'][0]['practice_root']), arm)
    verify_sealed(record, 'confirmation_sha256')
    if record.get('bank_rule') != 'per_seed_v1' or record.get('verify_mode') != 'all8_both_seeds':
        raise ValueError('native all8 per-seed admission required')
    # Merge the child view additively. A duplicate path with different bytes
    # refuses; branch tapes and seals stay in their original immutable tree.
    for source in (child.rglob('*') if child != bank else ()):
        if 'scenarios' in source.relative_to(child).parts:
            continue
        rc.guard_path(source)
        if not source.is_file() or source.suffix not in ('.json', '.gz', '.pt'):
            continue
        relative = source.relative_to(child)
        if relative.parts[0] in ('snapshots', 'decisions'):
            continue
        target = bank/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if rc.sha(target) != rc.sha(source):
                raise ValueError('augmentation import conflicts with existing bank evidence')
        else:
            target.symlink_to(source.resolve())
    state_id = payload['id']+'_source'
    source_snapshot = child/'snapshots'/(result['examples'][0]['practice_row']['state_id']+'.pt')
    target = bank/'snapshots'/(state_id+'.pt')
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        if rc.sha(target) != rc.sha(source_snapshot):
            raise ValueError('augmented decision source snapshot differs')
    else:
        target.symlink_to(source_snapshot.resolve())
    from .sumo_verify import snapshot_sha
    source_snapshot_sha = snapshot_sha(torch.load(source_snapshot, weights_only=False, map_location='cpu'))
    from .launch_preparation import stage_write
    stage_write(bank/'decisions'/(record['verification_id']+'.json'), dict(
        verification_id=record['verification_id'], state_id=state_id, policy_update=0,
        augmentation=dict(parent_example_id=payload['parent_example_id'],
                          source_snapshot_sha256=source_snapshot_sha,
                          confirmation_sha256=record['confirmation_sha256'])))
    # Seed-specific tape receipts are authoritative; initial rejected seed1 can
    # coexist with an admitted seed2. Only successful admitted examples enter.
    rows = []
    for example in result['examples']:
        admission = example['augmentation_admission']
        receipt = admission['verification_receipt']
        rows.append(dict(verification_id=record['verification_id'], example_id=admission['example_id'],
            parent_example_id=payload['parent_example_id'], state_id=state_id,
            seed_index=example['practice_row']['seed_index'], snapshot_sha256=example['snapshot_identity_sha256'],
            source_snapshot_sha256=source_snapshot_sha,
            wm_cooperative=True, sumo_evidence=dict(receipt_path='',
                receipt_sha256=receipt['receipt_sha256'], verification_id=receipt['verification_id'],
                plan_sha256=digest(receipt['chosen_plan']),
                charge_ids=[f'{receipt["verification_id"]}_{b["branch"]}' for b in receipt['branches']])))
    # Derive receipt paths from the verified tape projection rather than an
    # assumed generation directory (seed2/extended paths are different).
    by_example = {e['example_id']: e for e in record['per_seed_v1']['examples']}
    for row in rows:
        tape = by_example[row['example_id']]['tape']
        path = Path(tape['branches_dir']).parent/'verifications'/(tape['verification_id']+'.json')
        candidates = [p for p in bank.rglob(path.name) if p.is_file() and rc.sha(p) == rc.sha(path)]
        if len(candidates) != 1:
            raise ValueError('unique own-bank augmented tape receipt required')
        row['sumo_evidence']['receipt_path'] = candidates[0].relative_to(bank).as_posix()
        from .augmentation_charges import recorded_binding
        actual_receipt = read(candidates[0])
        if actual_receipt['receipt_sha256'] != row['sumo_evidence']['receipt_sha256']:
            raise ValueError('augmented own-seed receipt changed during ledger binding')
        row['sumo_evidence'].update(recorded_binding(root, candidates[0], actual_receipt))
    return rows


def run(cfg):
    from .rerun_tracks_augment import verify
    root, native = Path(cfg['run_root']), production.backend_config(cfg)
    health(root)
    if cfg.get('augmentation_accounting_scheme') != 'native_directory_v2':
        raise ValueError('registered directory/CRN-qualified augmentation accounting required')
    if (root/'AUGMENTATION_READY.json').exists():
        ready = read(root/'AUGMENTATION_READY.json'); verify_sealed(ready, 'receipt_sha256')
        if ready['config_sha256'] != digest(cfg):
            adoption = read(root/'AUGMENTATION_REBIND.json');verify_sealed(adoption, 'receipt_sha256')
            if (adoption['new_config_sha256'] != digest(cfg) or
                    adoption['augmentation_file_sha256'] != rc.sha(root/'AUGMENTATION_READY.json') or
                    'augmentationRebindSHA'+adoption['receipt_sha256'] not in (root/'AUTHORITY.md').read_text() or
                    any(rc.sha(path) != sha for path,sha in adoption['immutable_inputs'].items())):
                raise ValueError('once-only augmentation belongs to another approved runtime')
        return ready
    started = root/'AUGMENTATION_STARTED.json'
    if started.exists():
        resume = read(root/'AUGMENTATION_RESUME.json')
        verify_sealed(resume, 'receipt_sha256')
        if (resume['new_config_sha256'] != digest(cfg) or
                resume['original_start_sha256'] != rc.sha(started) or
                ('augmentationResumeSHA'+resume['receipt_sha256']) not in (root/'AUTHORITY.md').read_text()):
            raise ValueError('explicit registered partial augmentation recovery required')
        write(root/('AUGMENTATION_RESUME_STARTED_'+resume['receipt_sha256'][:12]+'.json'),
            dict(resume_sha256=resume['receipt_sha256'], config_sha256=digest(cfg)))
    else:
        write(started, dict(config_sha256=digest(cfg)))
    if cfg.get('augmentation_recipe_version') not in (2, 3):
        raise ValueError('complete same-arm bank augmentation recipe required')
    slates = {arm:proposals(cfg, arm) for arm in ('B','R')}
    quote = slate_quote(cfg, slates)
    write(root/('AUGMENTATION_FIXED_SLATE_QUOTE_'+digest(cfg)[:12]+'.json'), quote)
    admissions, counts, receipts = {}, {}, []
    for arm in ('B', 'R'):
        runtime = p1.ArmRuntime(dict(native, update=1), arm, 1)
        backend = SimpleNamespace(models={arm:runtime.base},
            components={arm:dict(merge_features=MergeEncoder())})
        rows, accepted = [], []
        for payload in slates[arm]:
            spent = 0
            def execute(pairs, suffix):
                nonlocal spent
                costs = sum(int(s['sim_s']) for s, _ in pairs)
                if spent+costs > rc.GATE_SIM_S:
                    raise ValueError('full-window all8 proposal exceeds registered allowance')
                if any(s['window']['end_s'] < payload['window']['end_s'] for s, _ in pairs):
                    raise ValueError('augmentation cannot shorten its original full window')
                from .augmentation_charges import job_id
                jobs = [dict(id=job_id(s), sim_s=int(s['sim_s'])) for s, _ in pairs]
                p1._reserve(native, jobs, 'augment_verify')
                with ProcessPoolExecutor(max_workers=16, mp_context=multiprocessing.get_context('spawn')) as pool:
                    futures = [pool.submit(production.run_branch, s, snapshot, native,
                        Path(s['t2c_artifact_root'])) for s, snapshot in pairs]
                    results = [f.result(timeout=3600) for f in futures]
                spent += costs
                return results
            result = verify(backend, payload, cfg, execute)
            rows.extend(_install(root, arm, payload, result))
            accepted.extend(result['examples'])
            receipts.append(dict(arm=arm, proposal=payload, charged_sim_s=spent,
                                  admitted=len(result['examples']), receipt=result['receipt']))
        if not accepted:
            raise ValueError('no SUMO-admitted augmented examples; smoke cannot claim augmentation')
        from .practice_weighting import SCHEMA
        record = _seal(dict(schema=SCHEMA, arm=arm, rows=rows,
            prelaunch=cfg['augmentation_prelaunch']), 'practice_augmented_sha256')
        target = rb.bank_root(root, arm)/'PRACTICE_AUGMENTED.json'
        if target.exists() and read(target) != record:
            previous = read(target);verify_sealed(previous,'practice_augmented_sha256')
            if previous['prelaunch'] != record['prelaunch'] or any(row not in rows for row in previous['rows']):
                raise ValueError('completed augmentation evidence may only be extended additively')
            history = target.parent/'augmentation_history';history.mkdir(exist_ok=True)
            target.rename(history/(previous['practice_augmented_sha256']+'.json'))
        from .launch_preparation import stage_write
        stage_write(target, record)
        # Real consumer validation is mandatory before declaring readiness.
        from .practice_weighting import augmented_examples
        with p1._bank_view(native):
            augmented_examples(root, arm, ps.examples(root, arm))
        counts[arm], admissions[arm] = len(accepted), rows
    ready = _seal(dict(status='SUMO_ALL8_ADMITTED', config_sha256=digest(cfg),
        admitted=counts, practice_rows=admissions, receipts=receipts), 'receipt_sha256')
    write(root/'AUGMENTATION_READY.json', ready)
    return ready
