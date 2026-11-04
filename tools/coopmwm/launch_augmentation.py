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


PROPOSAL_SETTINGS = dict(parameter_fraction=.01, partner_swap_probability=0.,
    provenance='clean1 proposed small-perturbation radius; same B/R, no WM admission, SHA-bound before outcomes')


class MergeEncoder:
    def encode(self, obs):
        from .track2_reward import _sensor_packet
        from .merge_features import merge_features, MergeFeaturesConfig
        own = obs['own_sensing']
        packet = _sensor_packet(own['vehicles'], own['ego_id'], obs['time_s'])
        return merge_features(packet, own['ego_id'], now_s=obs['time_s'],
                              cfg=MergeFeaturesConfig(True)).tolist()


def proposals(cfg, arm):
    if cfg.get('augmentation_proposals') != PROPOSAL_SETTINGS:
        raise ValueError('approval-bound common B/R augmentation settings required')
    root, native = Path(cfg['run_root']), production.backend_config(cfg)
    output, seen = [], set()
    with p1._bank_view(native):
        confirmations = rb.confirmations(root, arm)
        parents = ps.examples(root, arm)
        for row in parents:
            if len(output) == 2:
                break
            record = confirmations[row['verification_id']]
            example = next(e for e in record['per_seed_v1']['examples'] if e['example_id'] == row['example_id'])
            plan = example['plan']
            if plan['id'] in seen:
                continue
            label = moves.from_plan(plan)
            if label.family == 'none':
                continue
            snapshot = ps.load_snapshot(root, arm, row)
            trace = Path(example['tape']['branches_dir'])/example['tape']['verification_id']/example['tape']['branch']['branch']/'observable.jsonl.gz'
            frames = bank_v2v.trace_frames(trace)
            t = snapshot['t0']
            frame = frames.get(t-1)
            if frame is None:
                continue
            # Source plan geometry is clipped to actual direct sensing. No
            # all-road table is passed to T2G or a policy/scorer.
            ego = plan['members'][0]
            own = next((r for r in frame['vehicles'] if r['vid'] == ego), None)
            if own is None:
                continue
            rows = [dict(vid=r['vid'], is_cav=r['is_cav'], s_m=r['s_m'], v_mps=r['v_mps'],
                lane=r['lane'], allowed_speed_mps=31.29, observed_at_s=t-1)
                for r in frame['vehicles'] if abs(r['s_m']-own['s_m']) <= 100.]
            if not set(plan['members']) <= {r['vid'] for r in rows if r['is_cav']}:
                continue
            centre = np.asarray(plan['profile'])-np.asarray(plan['perturbation_profile'])
            context = dict(ego_id=ego, t_s=t, rows=rows,
                policy_traj={v: centre[i].tolist() for i,v in enumerate(plan['members'])})
            settings = dict(parameter_fraction=PROPOSAL_SETTINGS['parameter_fraction'],
                partner_swap_probability=PROPOSAL_SETTINGS['partner_swap_probability'],
                variety=dict(plan_augment.variety.VARIETY_REGISTERED), bounds=(-3., 1.2))
            parameters = dict(onset_s=label.onset, hold_s=label.hold, depth=label.depth)
            candidate = plan_augment._perturb(plan, parameters, context, settings,
                np.random.default_rng(cfg['training_seed']+len(output)))
            if candidate is None:
                continue
            modified, perturbed = candidate
            ident = f'clean1_aug_{arm}_{len(output):02}'
            directory = root/'augmentation_sources'/ident
            directory.mkdir(parents=True)
            path = directory/'SNAPSHOT.pt'
            with path.open('xb') as stream:
                torch.save(snapshot, stream)
            output.append(dict(id=ident, proposal_id=ident, source_id=row['example_id'],
                parent_example_id=row['example_id'], arm=arm, seed=snapshot['episode']['world']['seed'],
                plan=modified, source_plan_sha256=digest(plan), parameters=perturbed,
                snapshot_path=str(path), snapshot_file_sha256=rc.sha(path),
                window=dict(start_s=t, end_s=1200, truncated=False)))
            seen.add(plan['id'])
    if len(output) != 2:
        raise ValueError('two distinct own-bank observable cooperative plan proposals required per arm')
    return output


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
    write(bank/'decisions'/(record['verification_id']+'.json'), dict(
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
    return rows


def run(cfg):
    from .rerun_tracks_augment import verify
    root, native = Path(cfg['run_root']), production.backend_config(cfg)
    health(root)
    if (root/'AUGMENTATION_READY.json').exists():
        ready = read(root/'AUGMENTATION_READY.json'); verify_sealed(ready, 'receipt_sha256')
        if ready['config_sha256'] != digest(cfg):
            raise ValueError('once-only augmentation belongs to another approved runtime')
        return ready
    write(root/'AUGMENTATION_STARTED.json', dict(config_sha256=digest(cfg)))
    admissions, counts, receipts = {}, {}, []
    for arm in ('B', 'R'):
        runtime = p1.ArmRuntime(dict(native, update=1), arm, 1)
        backend = SimpleNamespace(models={arm:runtime.base},
            components={arm:dict(merge_features=MergeEncoder())})
        rows, accepted = [], []
        for payload in proposals(cfg, arm):
            spent = 0
            def execute(pairs, suffix):
                nonlocal spent
                costs = sum(int(s['sim_s']) for s, _ in pairs)
                if spent+costs > rc.GATE_SIM_S:
                    raise ValueError('full-window all8 proposal exceeds registered allowance')
                if any(s['window']['end_s'] < payload['window']['end_s'] for s, _ in pairs):
                    raise ValueError('augmentation cannot shorten its original full window')
                jobs = [dict(id=f'{s["verification_id"]}_{s["branch"]}', sim_s=int(s['sim_s'])) for s, _ in pairs]
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
        write(rb.bank_root(root, arm)/'PRACTICE_AUGMENTED.json', record)
        # Real consumer validation is mandatory before declaring readiness.
        from .practice_weighting import augmented_examples
        with p1._bank_view(native):
            augmented_examples(root, arm, ps.examples(root, arm))
        counts[arm], admissions[arm] = len(accepted), rows
    ready = _seal(dict(status='SUMO_ALL8_ADMITTED', config_sha256=digest(cfg),
        admitted=counts, practice_rows=admissions, receipts=receipts), 'receipt_sha256')
    write(root/'AUGMENTATION_READY.json', ready)
    return ready
