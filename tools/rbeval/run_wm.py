"""Resumable K8 WM2/WM-PHYS inference from registered observable starts.

Use the existing interpreter/runtime, no simulator or training. Run identity is
written before inference. The reference helpers are AST snapshots from the
registered WMCALIB/SPEEDHEAD producers, not imported executable entrypoints.
"""
import argparse
import copy
from dataclasses import replace
import gc
import inspect
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

if __package__:
    from .core import (ARMS, HERE, HS, SEEDS, attach_events, digest, load, reference,
                       reduce_trajectory, safe, save_new)
else:
    from core import (ARMS, HERE, HS, SEEDS, attach_events, digest, load, reference,
                      reduce_trajectory, safe, save_new)

BASE = Path('/root/autodl-tmp/c4')
DEFAULT_RUNTIME = BASE/'setupaudit20261030/fix'
DATA_ROOT = Path('/root/wmcalib')


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--runtime-root', default=str(DEFAULT_RUNTIME))
    p.add_argument('--wm', default=str(BASE/'wmcombined_wm2/wm_s.pt'))
    p.add_argument('--wm-config', default=str(BASE/'restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml'))
    p.add_argument('--model-name', default='wm2')
    p.add_argument('--device', default='cuda')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--memory-gib', type=float, default=5)
    p.add_argument('--max-tasks', type=int)
    p.add_argument('--endpoint-only', action='store_true', help='separate60s endpoint contract; retain600s padding/RNG prefix')
    p.add_argument('--switches', help='JSON object of explicit model-attribute overrides')
    return p.parse_args()


def main(model_loader=None, identity_sources=(), future_builder=None,
         worlds=None, future_indices=None):
    """Optional INT loader has the strict load_members(checkpoints,configs,device)
    interface. It returns ([frozen_model], JSON receipt); its Python source and
    explicitly supplied recipe/helper files are bound into the run identity.
    Live WM2 uses the original independently staged source, never overwritten.
    """
    args = arguments()
    panel_worlds = SEEDS if worlds is None else tuple(worlds)
    futures = tuple(range(8)) if future_indices is None else tuple(future_indices)
    assert panel_worlds and len(set(panel_worlds)) == len(panel_worlds) and set(panel_worlds) <= set(SEEDS)
    assert futures and len(set(futures)) == len(futures) and set(futures) <= set(range(8))
    if args.endpoint_only:
        assert args.model_name != 'wm2' and 'hdv60' in args.model_name.lower(), 'separate endpoint model identity required'
    rollout_steps = 120 if args.endpoint_only else 1200
    if model_loader is not None:
        assert args.model_name != 'wm2', 'custom construction requires explicit candidate identity'
        assert callable(model_loader)
    if future_builder is not None:
        assert args.model_name != 'wm2' and callable(future_builder)
    import fcntl
    assert 0 <= args.shard < args.shards
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19
    runtime = safe(args.runtime_root)
    sys.path[:0] = [str(runtime/'src'), str(runtime)]
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        C4_ALLOWED_ROOTS='/root/autodl-tmp/c4:/root/wmcalib:/root/rbeval20261031',
        C4_EXECUTION_PARITY='2', C4_SUMO_LEADER_RULE='1',
        C4_EXECUTION_DRIVER=str(runtime/'configs/execparity_cav_idm_v1.json'),
        WMS_WMSPEED_DEDUPE_MOBIL='1', WMS_WMSPEED_MOBIL_LAZY_SAFE='1',
        WMS_SPEED_PIVOT_DIAG_OFF='1', WMS_SPEED_FAST_MERGE_LANE='1',
        WMS_SPEED_FAST_CAV_DRIVE='1', WMS_SPEED_FAST_LONG_HISTORY='1', WMS_SPEED_GRAPH_CVAR='1')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(
            args.memory_gib*2**30/torch.cuda.get_device_properties(0).total_memory)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    from wms.control.bridge import make_cav_controller
    from wms.control.contract import MERGE_S1_M, SPEED_LIMIT_MPS
    from wms.control.counterfactual import native_fn
    from wms.control.dreamer import DreamerConfig, _entry_process, with_entry_slots
    from wms.control.entrant_mix import make_sampler
    from wms.control.futures import FuturesConfig, SampledFutures, lane_uniform_draws
    from wms.control.nativeinit import pair_merge_streams
    from wms.control.policy import load_policy_components
    from wms.control.mappo.eval_bridge import load_policy_fn
    from wms.control.policy14_cli import load_members
    from wms.control.safety import safe_accel
    from wms.control.start_states import StartBatch
    from wms.control.train_cli import GroupSignal
    from wms.model.entries import build_entrance_history, _counts_counter
    from wms.model.cav import WM_LANE_TARGET_KEY, WM_MERGE_KEY
    from wms.physics.batched import road_end_exit_bound
    from wms.train.cav_drive import TimelineControllers
    from wms.sim_bridge import load_controller_dependencies

    namespace = dict(globals(), **locals(), ACTORS=[])
    for key in ('motion', 'worker', 'costs'):
        reference(key, namespace)
    frozen = SimpleNamespace(**namespace)
    namespace.update(frozen=frozen, ROOT=DATA_ROOT, DATA=DATA_ROOT/'data')
    reference('starts', namespace)
    out = safe(args.out)
    assert out.parent.is_dir()
    out.mkdir(exist_ok=True)
    lock = (out/'WORKER.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pid = os.getpid()
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    start_ticks = Path(f'/proc/{pid}/stat').read_text().split()[21]
    save_new(out/f'PROCESS_{pid}.json', dict(pid=pid, boot_id=boot, start_ticks=start_ticks,
        argv=sys.argv, started_unix=time.time()))
    torch.manual_seed(965140)
    if args.model_name in ('wm2', 'wm2_hdv60'):
        assert digest(args.wm) == 'b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436'
    models, receipt = (model_loader or load_members)([args.wm], [args.wm_config], args.device)
    assert len(models) == 1
    model = models[0]
    if model_loader is not None:
        assert not model.training and all(not p.requires_grad for p in model.parameters()), 'inference-only frozen model required'
    switches = load(args.switches) if args.switches else {}
    for key, value in switches.items():
        assert hasattr(model, key), key
        setattr(model, key, value)
    candidate = BASE/'setupaudit20261030/policy_000190.pt'
    payload = torch.load(candidate, map_location='cpu', weights_only=False)
    cfg = replace(DreamerConfig(**payload['config']), horizon=600, entrant_learned_mix=True)
    assert cfg.future_samples == 8
    model.cav_exec_cap = cfg.cav_exec_cap
    batches, provenance = namespace['make_starts_wmdata'](model)
    cp, ce, _, _ = load_policy_components(candidate, device=args.device)
    torch.nn.init.zeros_(cp.residual_head.weight)
    torch.nn.init.zeros_(cp.residual_head.bias)
    cp.eval()
    ce = ce.encoder if isinstance(ce, GroupSignal) else ce
    ce.eval()
    own = BASE/'speedhead20261030'
    binding = load(own/'EXPORT_BOUND.json')
    assert digest(binding['path']) == binding['sha256']
    mp = load_policy_fn(binding['path'], device=args.device)
    mp.policy.eval()
    mp.group_encoder.eval()
    assert mp.native_lanes and not mp.has_lane_head
    deps = load_controller_dependencies(own/'e3final/runtime')
    dependencies = (deps.config, deps.circles_type, deps.direct_target, deps.neighbor_type,
                    deps.frame_type, deps.pi_type, deps.u_report_type, deps.pi_params_type)
    source_paths = [Path(__file__), HERE/'core.py', HERE/'references.json',
                    runtime/'configs/execparity_cav_idm_v1.json']
    source_paths += [safe(p) for p in identity_sources]
    if args.endpoint_only:
        source_paths += [HERE.parent/'rbint/endpoints.py', HERE.parent/'rbint/observations.py']
    if future_builder is not None:
        source_paths.append(safe(inspect.getfile(future_builder)))
    if model_loader is not None:
        source_paths.append(safe(inspect.getfile(model_loader)))
    # Bind all Python implementation files in the selected runtime, not just top modules.
    source_paths += sorted((runtime/'src/wms').rglob('*.py'))
    source_paths += [Path(inspect.getfile(x)) for x in
                     (deps.circles_type, deps.direct_target, deps.pi_type, deps.pi_params_type)]
    sources = {str(safe(p)): digest(p) for p in source_paths}
    reals = load(HERE/'real_observables.json')['episodes']
    real = {(e['seed'], e['arm']): e for e in reals}
    for p in provenance:
        s = p['metadata']['seed']
        if s in SEEDS:
            assert all(real[s, arm]['initial_sha256'] == p['metadata']['initial_state_sha256'] for arm in ARMS)
    identity = dict(schema='wmphys-hdv60/1' if args.endpoint_only else 'rbeval/1',
        K=8, horizon=60 if args.endpoint_only else 600, t0=300, arms=ARMS,
        worlds=panel_worlds, shard=args.shard, shards=args.shards, model=args.model_name,
        wm=receipt, wm_sha256=digest(args.wm), config_sha256=digest(args.wm_config),
        candidate_sha256=digest(candidate), export_sha256=binding['sha256'],
        sources=sources, provenance=provenance, switches=switches, device=args.device,
        real_receipt_sha256=digest(HERE/'real_observables.json'), mix=True,
        native_lanes=True, seed=965140, update=190)
    if future_indices is not None:
        identity['future_indices'] = futures
    if model_loader is not None:
        identity['construction_loader_sha256'] = digest(inspect.getfile(model_loader))
    # Round-trip tuple/list normalization before exact resume identity comparison.
    identity = json.loads(json.dumps(identity))
    if (out/'IDENTITY.json').exists():
        assert load(out/'IDENTITY.json') == identity, 'Run identity changed; preserve this run'
    else:
        save_new(out/'IDENTITY.json', identity)
    identity_sha = digest(out/'IDENTITY.json')
    native = native_fn('head', a_max=1.2)
    if getattr(model, '_wmphys_native_model', 'registered_head') in ('sumo_physics', 'sumo_uncommanded'):
        from wms.model.sumo_native import native_physics_accel
        native = native_physics_accel

    def controllers(slots):
        def native_call(state):
            mask = state['present'].bool() & state['is_cav'].bool()
            state[WM_MERGE_KEY] = mask & (state['lane'] == 2)
            lane = torch.where(state['lane'] != 2, state[WM_LANE_TARGET_KEY], state['lane'])
            if getattr(model, '_wmphys_native_model', None) == 'sumo_uncommanded':
                # Dispatcher payload is ignored for these physically delegated
                # slots; do not consult the retired commanded-driver mirror.
                return torch.stack((state['s_m']+.5*state['v_mps'], lane,
                    state['v_mps'], torch.zeros_like(state['v_mps'])), -1)
            return frozen.native_background_motion(state, lane, native)
        calls = [native_call, frozen.callback_for('ctrl0', cp, ce, mp, native),
                 frozen.callback_for('mappo', cp, ce, mp, native)]
        for arm in ARMS[3:]:
            ids = torch.full((1, 601, slots), 2 if arm == 'pi' else 1, dtype=torch.long)
            timeline = TimelineControllers(model, dict(controller=ids,
                acceleration_mps2=torch.zeros_like(ids, dtype=torch.float32),
                target_speed_mps=torch.zeros_like(ids, dtype=torch.float32)), dependencies=dependencies)
            def baseline(state, timeline=timeline):
                mask = state['present'].bool() & state['is_cav'].bool()
                state[WM_MERGE_KEY] = mask & (state['lane'] == 2)
                lane = torch.where(state['lane'] != 2, state[WM_LANE_TARGET_KEY], state['lane'])
                return frozen.parity_motion(state, lane, timeline(state)[..., 3])
            calls.append(baseline)
        def callback(state):
            outputs, merges = [], []
            for i, call in enumerate(calls):
                row = {k: v[i:i+1] if torch.is_tensor(v) and v.ndim and v.shape[0] == len(ARMS) else v
                       for k, v in state.items()}
                outputs.append(call(row))
                merges.append(row[WM_MERGE_KEY])
            state[WM_MERGE_KEY] = torch.cat(merges)
            return torch.cat(outputs)
        callback.native_lane_mask = lambda state: state['present'].bool() & state['is_cav'].bool()
        if getattr(model, '_wmphys_native_model', None) == 'sumo_uncommanded':
            def uncommanded_mask(state):
                mask = torch.zeros_like(state['present'], dtype=torch.bool)
                mask[0] = state['present'][0].bool() & state['is_cav'][0].bool()
                return mask
            callback.native_uncommanded_mask = uncommanded_mask
        return callback

    def move(x):
        if torch.is_tensor(x):
            return x.to(args.device)
        return {k: move(v) for k, v in x.items()} if isinstance(x, dict) else x

    tasks = 0
    selected = [s for i, s in enumerate(panel_worlds) if i % args.shards == args.shard]
    if args.model_name != 'wm2':
        # Administrative order only: registered panel/partition/RNG stay fixed.
        # Complete the reserved eight worlds first for the owner's04:00 interim.
        selected.sort(key=lambda s: (s < 969432, s))
    by_seed = {b.metadata[0]['seed']: (b, i) for i, b in enumerate(batches)}
    for seed in selected:
        source, bi = by_seed[seed]
        for fi in futures:
            target = out/f'w{seed}_k{fi}.json'
            if target.exists():
                old = load(target)
                assert old['identity_sha256'] == identity_sha and old['seed'] == seed and old['future'] == fi
                assert len(old['episodes']) == 5
                if args.endpoint_only:
                    assert old['schema'] == 'wmphys-hdv60/1' and all('endpoint60' in e for e in old['episodes'])
                else:
                    assert all(len(e['bins']['upstream']['speed']) == 66 for e in old['episodes'])
                continue
            assert not (out/'STOP').exists(), 'cooperative stop'
            while os.getloadavg()[0] > 24:
                assert not (out/'STOP').exists()
                time.sleep(30)
            begin = time.monotonic()
            torch.manual_seed(965140)
            b = with_entry_slots(copy.deepcopy(source), 600, 1920,
                torch.Generator().manual_seed(1_000_003*191), queue_state=True)
            b.state, b.rollout_kwargs = move(b.state), move(b.rollout_kwargs)
            plan = (SampledFutures(FuturesConfig.from_trainer(cfg), 190).plan(
                models, b.state, b.rollout_kwargs, 1200)[0] if future_builder is None else None)
            initial = frozen.expand(b.state, len(ARMS))
            pair_merge_streams(initial, torch.zeros(len(ARMS), dtype=torch.long, device=args.device), 1, 190+fi)
            initial['exit_s_m'] = torch.full_like(initial['s_m'], road_end_exit_bound(model.physics))
            # Snapshot the t0 state before in-place admission/model updates.
            start = {k: initial[k].detach().cpu().numpy().copy() for k in ('s_m', 'v_mps', 'lane', 'present')}
            future_receipt = None
            if future_builder is not None:
                kw, entries, future_receipt = future_builder(model=model, batch=b,
                    initial=initial, future=fi, seed=seed, frozen=frozen)
                process = entries.process
            else:
                kw = frozen.expand({**b.rollout_kwargs, **plan[fi]}, len(ARMS))
                lanes = lane_uniform_draws(models, 1, 1200, initial['s_m'].shape[1],
                    190+1000003*fi if fi else 190, device=args.device)
                if lanes is not None:
                    kw['lane_uniform'] = frozen.expand(lanes, len(ARMS))
                assert not model.physics.entry_learned, 'use a separately frozen adapter for learned insertion'
                process = _entry_process(b.entry_spec, [0]*len(ARMS), initial['s_m'].shape[1],
                    cfg.entry_insert_rule, None, couple_draws=True, sample_index=fi)
                entries = frozen.Entries(process)
                kw['entries'] = entries
                kw['entrant_driver_sampler'] = make_sampler(model, initial, kw, 190+1000003*fi, [0]*len(ARMS))
            cb = controllers(initial['s_m'].shape[1])
            with torch.no_grad():
                if getattr(model, '_wmphys_departure_phase', 'before_movement') == 'after_movement_and_lanes':
                    from tools.rbint.departure import rollout_after_movement
                    roll = rollout_after_movement(model, initial, n_steps=rollout_steps, cav_controller=cb, **kw)
                else:
                    roll = model.rollout(initial, n_steps=rollout_steps, cav_controller=cb, **kw)
                assert torch.isfinite(roll['speeds']).all()
                score = None if args.endpoint_only else frozen.costs(roll, entries, b.entry_spec['k0'])
                if future_builder is not None:
                    future_receipt['population_audit'] = entries.audit_population(initial, roll)
            arrays = {k: roll[k].detach().cpu().numpy() for k in ('positions', 'speeds', 'lanes', 'masks')}
            episodes = []
            for i, arm in enumerate(ARMS):
                if args.endpoint_only:
                    prefix = [np.concatenate((start[s][i:i+1], arrays[r][i, :119]), axis=0)
                        for s, r in (('s_m', 'positions'), ('v_mps', 'speeds'), ('lane', 'lanes'), ('present', 'masks'))]
                    prefix_bins = reduce_trajectory(300+np.arange(120)*.5, *prefix)
                    episodes.append(dict(seed=seed, future=fi, arm=arm,
                        initial_sha256=source.metadata[0]['initial_state_sha256'],
                        prefix60_bins={reg:dict(speed=value['speed'][:6], post_counts=value['post_counts'][:6])
                                       for reg,value in prefix_bins.items()}))
                    continue
                pos, speed, lane, present = [np.concatenate((start[s][i:i+1], arrays[r][i, :-1]), axis=0)
                    for s, r in (('s_m', 'positions'), ('v_mps', 'speeds'), ('lane', 'lanes'), ('present', 'masks'))]
                bins = reduce_trajectory(300+np.arange(1200)*.5, pos, speed, lane, present)
                bins = attach_events(bins, real[seed, arm])
                clocks = {str(h): dict(team_delay_vs=score[str(h)]['real_J'][i],
                    road_delay_vs=score[str(h)]['real_road'][i], source_wait_vs=score[str(h)]['real_source'][i]) for h in HS}
                episodes.append(dict(seed=seed, future=fi, arm=arm,
                    initial_sha256=source.metadata[0]['initial_state_sha256'], bins=bins, clocks=clocks))
            if args.endpoint_only or getattr(model, '_wmphys_variant', None) in ('A', 'B', 'C', 'FULL'):
                from tools.rbint.endpoints import attach_endpoints60
                attach_endpoints60(episodes, arrays, seed, b.entry_spec['k0'])
            assert all(digest(p) == sha for p, sha in sources.items()), 'source changed during task'
            task = dict(schema='wmphys-hdv60/1' if args.endpoint_only else 'rbeval/1',
                identity_sha256=identity_sha, seed=seed, future=fi,
                arms=ARMS, episodes=episodes, costs=score, seconds=time.monotonic()-begin,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if args.device.startswith('cuda') else None)
            if future_receipt is not None:
                task['future_sampling'] = future_receipt
            save_new(target, task)
            print('DONE', seed, fi, round(task['seconds'], 2), flush=True)
            del roll, entries, process, kw, cb, b, initial, plan, arrays, start
            gc.collect()
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()
            tasks += 1
            if args.max_tasks and tasks >= args.max_tasks:
                return
    if not (out/'COMPLETE.json').exists():
        save_new(out/'COMPLETE.json', dict(tasks=len(futures)*len(selected), identity_sha256=identity_sha,
            hashes={f'w{s}_k{k}.json': digest(out/f'w{s}_k{k}.json') for s in selected for k in futures}))


if __name__ == '__main__':
    main()
