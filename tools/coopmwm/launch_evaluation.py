"""One diagnostic owner for lessons and T1D shared paired evaluation."""
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

import torch

from . import launch_production as production, rerun_cpus as rc
from . import track2_backend as p1
from .exact_grade import _seal


def episode(spec, cfg, work):
    """An immutable final policy, full 1200s window and registered TRAIN world."""
    from . import accel_plant, protocol
    from .v2v_runner import patched
    world = spec['world']
    allowed = [w for index in (0, 1) for w in protocol.worlds('train', index)]
    if world not in allowed:
        raise ValueError('evaluation only accepts originally registered TRAIN worlds')
    path = Path(cfg['run_root'])/f'{spec["arm"]}_checkpoint_{cfg["evaluation_update"]:03}.pt'
    if rc.sha(path) != spec['checkpoint']['file_sha256']:
        raise ValueError('paired checkpoint changed after registration')
    weights = production.checkpoint_weights(cfg, spec['arm'], cfg['evaluation_update'])
    holder = {}
    def loader(values):
        policy = production.load_policy(values, cfg)
        holder['policy'] = policy
        return policy
    def hook(controller, job, prefixes):
        policy = holder['policy']
        if isinstance(policy, p1.EpisodePolicy):
            original = controller.apply
            def apply(rows, commands, now):
                policy.observe(rows, now, command_commitments=controller.active_plans)
                return original(rows, commands, now)
            controller.apply = apply
    job = dict(id=Path(work).name, path=str(work), world=world, role='train', arm=spec['arm'],
        weights=weights, runtime=cfg['runtime'], checkpoint=cfg['checkpoint'],
        noise_seed=world['seed'], deterministic=True, guided=False, v2v=True, sim_s=1200,
        warm_history=cfg['warm_history'], accel_bounds=cfg['accel_bounds'])
    # protocol's original registered TRAIN universe includes both 986xxx and
    # 987xxx. The narrower accel wrapper only included training index0. These
    # explicit evaluation jobs use the original guards, not a new seed bypass.
    with patched(accel_plant, dict(guard=protocol.guard, install_seed_scope=protocol.install_seed_scope)):
        result = accel_plant.episode(job, controller_hook=hook, policy_loader=loader)
    from .rerun_tracks_backend import safety
    vector = safety(result, Path(work)/'observable.jsonl.gz')
    return dict({k: deepcopy(spec[k]) for k in ('arm', 'world_id', 'world', 'checkpoint',
        'window', 'final_worlds_sha256')}, status='graded',
        J_veh_s=result['totals']['team_delay_vs'], safety=vector)


def _wave(cfg, pairs, phase, prefix, *, fixed=False):
    from .launch_training import health
    health(cfg['run_root'])
    jobs = [dict(id=f'{prefix}_{i:04}', sim_s=int(spec['sim_s'])) for i, (spec, _) in enumerate(pairs)]
    p1._reserve(production.backend_config(cfg), jobs, phase)
    native = production.backend_config(cfg)
    with ProcessPoolExecutor(max_workers=16, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(episode, spec, native,
            Path(cfg['run_root'])/'evaluation'/job['id']) if fixed else pool.submit(
            production.run_branch, spec, snapshot, native, Path(cfg['run_root'])/'evaluation'/job['id'])
            for job, (spec, snapshot) in zip(jobs, pairs)]
        results = [future.result(timeout=3600) for future in futures]
    if any(r.get('status') != 'graded' for r in results):
        raise ValueError('incomplete scheduled check; never drop/impute/retry')
    return results


def scheduled(cfg, update):
    from .launch_training import write, read
    from . import replay_bank as rb, sumo_verify as sv, practice_starts as ps
    root = Path(cfg['run_root'])
    if update in rc.SCHEDULE['B']['lessons']:
        target = root/f'evaluation/LESSON_u{update:03}.json'
        if not target.exists():
            native = production.backend_config(cfg)
            with p1._bank_view(native):
                frozen = rb.lesson_list(root, 'B', cfg.get('replay_sources', {}).get('B', ()))
                decisions = {r['verification_id']: r for r in rb.bank_decisions(root, 'B',
                    sources=cfg.get('replay_sources', {}).get('B', ()))}
                links, pairs = ps._links(root, 'B'), []
                weights = production.checkpoint_weights(native, 'B', update)
                for vid in frozen['verification_ids']:
                    receipt = decisions[vid]
                    snap = rb._decision_snapshot(root, 'B', receipt, links)
                    snap.update(weights=weights, policy=dict(state_dict_sha256=sv._state_sha(weights)))
                    pairs.append((rb.lesson_policy_spec(snap, receipt['chosen_plan'], receipt), snap))
            results = _wave(cfg, pairs, 'v2v_lessons', f'lesson_u{update:03}')
            write(target, _seal(dict(status='graded', update=update, results=results), 'receipt_sha256'))
    if update not in (8, 24):
        return
    target = root/f'evaluation/PAIRED_u{update:03}.json'
    if target.exists():
        return read(target)
    from . import planner_eval
    registration = rc.final_world_registration(read(root/'ISOLATED_REGISTRATION.json'))
    contract = deepcopy(cfg['final_evaluation'])
    if update == 8:
        contract.update(update=8, phase='v2v_interim')
    checkpoints = {}
    for arm in contract['arms']:
        path = root/f'{arm}_checkpoint_{update:03}.pt'
        checkpoints[arm] = dict(path=str(path), file_sha256=rc.sha(path), update=update,
            components=contract['checkpoint_components'][arm])
    # Interim uses the same registered panel and estimator, with immutable u8
    # weights explicitly identified. It is descriptive; u24 alone is final.
    pairs = planner_eval.prepare_track2(registration=registration, checkpoints=checkpoints, contract=contract)
    execution = deepcopy(cfg)
    execution['evaluation_update'] = update
    phase = 'v2v_interim' if update == 8 else 'v2v_final'
    # _wave translates its input but preserves this explicit update metadata.
    results = _wave(execution, pairs, phase, f'paired_u{update:03}', fixed=True)
    analysis = planner_eval.analyse_track2(registration=registration, checkpoints=checkpoints,
        records=results, contract=contract)
    value = _seal(dict(update=update, status='graded', registration=registration,
        checkpoints=checkpoints, results=results, analysis=analysis), 'receipt_sha256')
    write(target, value)
    return value
