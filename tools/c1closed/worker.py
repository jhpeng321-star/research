"""Frozen-model inference on observed WMDATA starts only; no outcome inputs.

Runs beside the immutable SPEEDHEAD runtime. Task outputs contain horizon
snapshots and additive integrals, not recorded future trajectories.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import inspect
import json
import os
import sys
import time
from types import MethodType

os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
RUNTIME = Path('/root/autodl-tmp/c4/speedhead20261030')
sys.path.insert(0, str(RUNTIME / 'tools/speedhead'))
import wm_gpu_wmcalib as source
frozen = source.frozen
import numpy as np
import torch
from aggregates import fields, merge_queue

ROOT = Path('/root/autodl-tmp/c4/c1closed20261031')
HORIZONS = (10, 30, 60, 90, 120, 180, 300)
ARMS = ('native', 'ctrl0', 'mappo', 'pi', 'circles')
HASHES = {
    'wm2': 'b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436',
    'bbtraj': 'afaaa32cb15079014d1a9e08f78b26e74fd341298d1aa0292a24009d9f783742',
    'bbid': 'e12c81c008e3bb29e366db5597061754e0fa56092dbe5a995a9baa1fbfcce61d',
    'historical': '87ed02e0707e5e4eefdd2dbb0fd282f834c1288776f7f395b41ab4d13826aaef',
}
BAD = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
       'hetdata_truth', 'vtype')


def audit(event, args):
    if event == 'subprocess.Popen':
        raise PermissionError('inference may not launch subprocesses')
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0]).lower()
        if any(x in path for x in BAD) or path.endswith(('.xml', '.xml.gz')):
            raise PermissionError('excluded input path')
        # Inference must never open recorded future labels/tapes.
        if any(x in path for x in ('traj_base.npz', '/labels/', '/episode/episode.npz')):
            raise PermissionError('future outcome access from inference')


def write(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=1, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu(x):
    return x.detach().cpu().numpy()


class Entries:
    """Autonomous admissions on EACH arm; no reference-arm birth substitution."""
    def __init__(self, process):
        self.process = process
        self.states = []
        self.queues = []

    def insert(self, state):
        # Left-boundary integration before the admission clock advances .5 s.
        self.states.append({k: cpu(state[k]).copy() for k in
                            ('s_m', 'v_mps', 'lane', 'present', 'is_cav')})
        self.queues.append(self.process.backlog().copy())
        return self.process.insert(state)


def callbacks(model, cp, ce, mp, slots, deps, arms, plant='common'):
    from wms.control.continuous import project_native
    from wms.model.cav import WM_LANE_TARGET_KEY, WM_MERGE_KEY
    from wms.train.cav_drive import TimelineControllers
    native = frozen.native_fn('head', a_max=1.2)

    class Plant:
        physics = model.physics

        def physics_accel(self, state):
            # Shared external CAV plant, including for physics-free BB-traj.
            return project_native(state, native(state))

    def native_cb(state):
        mask = state['present'].bool() & state['is_cav'].bool()
        state[WM_MERGE_KEY] = mask & (state['lane'] == 2)
        lane = torch.where(state['lane'] != 2,
                           state.get(WM_LANE_TARGET_KEY, state['lane']), state['lane'])
        return frozen.parity_motion(state, lane, Plant().physics_accel(state))

    calls = []
    for arm in arms:
        if arm == 'native':
            calls.append(native_cb)
        elif arm in ('ctrl0', 'mappo'):
            calls.append(frozen.callback_for(arm, cp, ce, mp, native))
        else:
            ids = torch.full((1, 302, slots), 2 if arm == 'pi' else 1, dtype=torch.long)
            timeline = TimelineControllers(Plant(), dict(controller=ids,
                acceleration_mps2=torch.zeros_like(ids, dtype=torch.float32),
                target_speed_mps=torch.zeros_like(ids, dtype=torch.float32)), dependencies=deps)

            def baseline(state, timeline=timeline):
                mask = state['present'].bool() & state['is_cav'].bool()
                state[WM_MERGE_KEY] = mask & (state['lane'] == 2)
                lane = torch.where(state['lane'] != 2,
                    state.get(WM_LANE_TARGET_KEY, state['lane']), state['lane'])
                accel = timeline(state)[..., 3]
                return frozen.parity_motion(state, lane, accel)

            calls.append(baseline)

    def callback(state):
        if plant == 'common':
            state[WM_LANE_TARGET_KEY] = state['lane']
        out, merges = [], []
        for i, call in enumerate(calls):
            row = {k: v[i:i+1] if torch.is_tensor(v) and v.ndim and
                   v.shape[0] == len(arms) else v for k, v in state.items()}
            out.append(call(row))
            merges.append(row[WM_MERGE_KEY])
        state[WM_MERGE_KEY] = torch.cat(merges)
        result = torch.cat(out)
        if plant == 'common':
            # Frozen C1FAIR common-plant limitation: explicit gap-based ramp
            # merging, no ordinary CAV lane changes. Both learned HDV models
            # receive identical external CAV code/plant; no new BB architecture.
            from wms.control.bridge import _merge_lane
            cav = state['present'].bool() & state['is_cav'].bool()
            result[..., 1] = torch.where(cav, _merge_lane(state, cav), result[..., 1])
            state[WM_MERGE_KEY] = torch.zeros_like(cav)
        return result

    if plant == 'production':
        callback.native_lane_mask = lambda st: st['present'].bool() & st['is_cav'].bool()
    return callback


def extract(roll, entries, hs, ids):
    values = {}
    pos, speed, mask, lane = (cpu(roll[k]) for k in ('positions', 'speeds', 'masks', 'lanes'))
    assert all(np.isfinite(x).all() for x in (pos, speed))
    before = {k: np.stack([s[k] for s in entries.states], 1) for k in entries.states[0]}
    present = before['present'].astype(bool)
    cost = .5 * np.maximum(0., 1. - before['v_mps'] / 31.29) * present
    regions = (before['s_m'] < 0, (before['s_m'] >= 0) & (before['s_m'] <= 75.44),
               before['s_m'] > 75.44)
    per_region = np.stack([(cost * r).sum(-1) for r in regions], -1)
    queues = np.stack(entries.queues, 1)
    if queues.ndim == 3:
        queues = queues.sum(-1)
    # Slots never recycle: continuous-identity crossing brackets are unambiguous.
    after_s, before_s = pos, before['s_m']
    cross = present & mask & (before_s < 75.44) & (after_s >= 75.44) & (lane < 2)
    q = merge_queue(before_s, before['v_mps'], present)
    endpoint_q = merge_queue(pos, speed, mask.astype(bool))
    grid = fields(before_s, before['v_mps'], present) if pos.shape[1] >= 20 else None
    for h in hs:
        n = 2*h
        road = per_region[:, :n].sum(1)
        source_j = .5*queues[:, :n].sum(1)
        values[str(h)] = dict(source=source_j.tolist(), upstream=road[:, 0].tolist(),
            merge=road[:, 1].tolist(), discharge=road[:, 2].tolist(),
            J=(source_j+road.sum(-1)).tolist(), crossings=cross[:, :n].sum((1, 2)).tolist(),
            merge_queue=endpoint_q[:, n-1].tolist(), mean_merge_queue=q[:, :n].mean(1).tolist(),
            fields={k: np.asarray(v)[:, :h//10].tolist() for k, v in grid.items()} if grid else None,
            endpoints=[dict(ids=ids, s=pos[i, n-1, :len(ids)].tolist(),
                v=speed[i, n-1, :len(ids)].tolist(),
                present=mask[i, n-1, :len(ids)].tolist()) for i in range(pos.shape[0])])
        assert np.allclose(np.asarray(values[str(h)]['J']), source_j+road.sum(-1), atol=1e-6)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=HASHES, required=True)
    parser.add_argument('--world', type=int, required=True)
    parser.add_argument('--future', type=int, default=0)
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--plant', choices=('common', 'production'), default='common')
    args = parser.parse_args()
    assert 0 <= args.world < 40 and 0 <= args.future < (8 if args.model == 'wm2' else 1)
    assert args.seconds in (2, 10, 300)
    assert args.plant == 'common' or args.model == 'wm2'
    sys.addaudithook(audit)
    torch.set_num_threads(1)
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19
    out = ROOT / args.out
    assert out.is_dir()
    begin = time.monotonic()
    device = args.device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' if device == 'cuda' else ''
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(3 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(965140)
    models, receipt = frozen.load_members([str(frozen.WM_PATH)], [str(frozen.WM_CONFIG)], device)
    reference = models[0]
    assert digest(frozen.WM_PATH) == HASHES['wm2']
    if args.model == 'wm2':
        model = reference
        model_path = frozen.WM_PATH
    else:
        from wms.model.checkpoint import load_checkpoint
        from wms.config.train import TrainConfig
        kinds = {'bbtraj': 'blackbox_traj', 'bbid': 'blackbox_id', 'historical': 'structured'}
        model_path = ROOT / 'ckpt' / args.model / 'wm_s.pt'
        assert digest(model_path) == HASHES[args.model]
        loaded = load_checkpoint(model_path, device=device, strict=True,
            model_kind=kinds[args.model], config_factory=TrainConfig.from_legacy_flat)
        model = loaded.model.eval()
    payload = torch.load(frozen.CANDIDATE, map_location='cpu', weights_only=False)
    cfg = frozen.replace(frozen.DreamerConfig(**payload['config']), horizon=300,
                         entrant_learned_mix=True)
    model.cav_exec_cap = cfg.cav_exec_cap
    if args.model == 'bbid':
        # Original C1 ID-conditioned baseline learns unforced CAV motion; it
        # never received the later learned-driver execution ceiling.
        from wms.model.cav import set_exec_cap
        set_exec_cap(model, 'off')
    if args.plant == 'common' and args.model != 'bbid':
        def shared_cav_cap(self, out, state, driver_params):
            from wms.control.alwayscmd import inferred_cap
            return inferred_cap(state)
        model._exec_cap = MethodType(shared_cav_cap, model)
    batches, provenance = source.make_starts_wmdata(model)
    b0 = batches[args.world]
    seed = b0.metadata[0]['seed']
    assert seed == 969400+args.world
    root = source.DATA / 'jobs' / f's{seed}_ctrl0'
    st = torch.load(root/'start.pt', map_location='cpu', weights_only=False)
    with np.load(root/'start_history.npz', allow_pickle=False) as raw:
        keep = torch.where(st['state']['present'][0])[0].numpy()
        ids = raw['vid_index'][-1, keep].astype(int).tolist()
    assert len(ids) == len(set(ids)) and min(ids) >= 0
    cp, ce, _, _ = frozen.load_policy_components(frozen.CANDIDATE, device=device)
    torch.nn.init.zeros_(cp.residual_head.weight)
    torch.nn.init.zeros_(cp.residual_head.bias)
    cp.eval()
    ce = ce.encoder if isinstance(ce, frozen.GroupSignal) else ce
    ce.eval()
    binding = json.loads((RUNTIME/'EXPORT_BOUND.json').read_text())
    assert digest(binding['path']) == binding['sha256'] == 'd9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca'
    mp = frozen.load_policy_fn(binding['path'], device=device)
    mp.policy.eval()
    mp.group_encoder.eval()
    assert mp.native_lanes and not mp.has_lane_head
    deps = source.load_controller_dependencies(RUNTIME/'e3final/runtime')
    dep_tuple = (deps.config, deps.circles_type, deps.direct_target, deps.neighbor_type,
                 deps.frame_type, deps.pi_type, deps.u_report_type, deps.pi_params_type)
    arms = ARMS if args.model != 'bbid' else ('native', 'pi', 'circles')
    future = args.future
    b = frozen.with_entry_slots(copy.deepcopy(b0), 300, 1920,
        torch.Generator().manual_seed(1_000_003*191), queue_state=True)
    b.state = source.move(b.state, device)
    b.rollout_kwargs = source.move(b.rollout_kwargs, device)
    steps = 2*args.seconds
    if args.model == 'wm2':
        plan = frozen.SampledFutures(frozen.FuturesConfig.from_trainer(cfg), 190).plan(
            models, b.state, b.rollout_kwargs, steps)[0]
        kwargs = frozen.expand({**b.rollout_kwargs, **plan[future]}, len(arms))
    else:
        kwargs = frozen.expand(b.rollout_kwargs, len(arms))
    initial = frozen.expand(b.state, len(arms))
    frozen.pair_merge_streams(initial, torch.zeros(len(arms), dtype=torch.long, device=device), 1, 190+future)
    initial['exit_s_m'] = torch.full_like(initial['s_m'], frozen.road_end_exit_bound(reference.physics))
    lanes = frozen.lane_uniform_draws([model], 1, steps, initial['s_m'].shape[1],
        190+1000003*future if future else 190, device=device)
    if lanes is not None:
        kwargs['lane_uniform'] = frozen.expand(lanes, len(arms))
    process = frozen._entry_process(b.entry_spec, [0]*len(arms), initial['s_m'].shape[1],
        cfg.entry_insert_rule, None, couple_draws=True, sample_index=future)
    entries = Entries(process)
    kwargs['entries'] = entries
    if args.model == 'wm2':
        kwargs['entrant_driver_sampler'] = frozen.make_sampler(model, initial, kwargs,
            190+1000003*future, [0]*len(arms))
    callback = callbacks(reference, cp, ce, mp, initial['s_m'].shape[1], dep_tuple, arms, args.plant)
    if args.model == 'bbid':
        from wms.model import controller_index
        kwargs['controller_id'] = torch.tensor([controller_index(a if a != 'circles' else 'damping') for a in arms], device=device)
        callback = None
    hs = [h for h in HORIZONS if h <= args.seconds] or [2]
    target = out/f'{args.model}_{args.plant}_w{args.world}_k{future}_h{args.seconds}.json'
    assert not target.exists(), 'write-once task; existing output must be verified'
    with torch.no_grad():
        roll = model.rollout(initial, n_steps=steps, cav_controller=callback, **kwargs)
        values = extract(roll, entries, hs, ids)
    identity = dict(model_sha256=digest(model_path), source_sha256=digest(__file__),
        aggregates_sha256=digest(Path(__file__).with_name('aggregates.py')),
        worker_sha256=digest(frozen.__file__), wmdata_helper_sha256=digest(source.__file__),
        wm_config_sha256=digest(frozen.WM_CONFIG), mappo_sha256=binding['sha256'],
        provenance=provenance[args.world], wm_receipt=receipt,
        dependencies={str(p): digest(p) for p in
            [inspect.getfile(deps.pi_type), inspect.getfile(deps.circles_type),
              inspect.getfile(source.TimelineControllers), inspect.getfile(frozen.make_sampler)]})
    identity['runtime_closure'] = {str(p): digest(p) for p in
        sorted((Path(inspect.getfile(type(reference))).parents[1]).rglob('*.py'))
        if not any(x in str(p).lower() for x in BAD)}
    identity['candidate_sha256'] = digest(frozen.CANDIDATE)
    identity['execution_driver_sha256'] = digest(os.environ['C4_EXECUTION_DRIVER'])
    result = dict(schema='c1closed/1', model=args.model, world=args.world, seed=seed,
        future=future, arms=list(arms), seconds=args.seconds, values=values,
        identity=identity, pid=os.getpid(), runtime_seconds=time.monotonic()-begin,
        device=device, mix=args.model == 'wm2', slots=initial['s_m'].shape[1],
        steps=steps, original_ids=ids, original_cav=cpu(b0.state['is_cav'])[0].tolist(),
        callback_contract='feedback_on_predicted_states' if callback else 'legacy_learned_controller_id',
        source_queue_initial=entries.queues[0].tolist(), plant=args.plant)
    write(target, result)
    print(json.dumps({k: result[k] for k in ('model','world','future','steps','runtime_seconds','pid','slots')}), flush=True)


if __name__ == '__main__':
    main()
