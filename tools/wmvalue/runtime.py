"""WM2 frozen-runtime adapter. Imported only by the explicit remote runner/audit.

Environment setup precedes torch/wms imports. No simulator is instantiated.
"""
import copy
import os
import sys
from pathlib import Path

BASE = Path('/root/autodl-tmp/c4')
FIX = BASE/'setupaudit20261030/fix'
sys.path[:0] = [str(BASE/'speedhead20261030/tools/speedhead'), str(FIX/'src'),
                str(FIX), str(FIX/'tools/setupaudit')]

import wm_worker as frozen  # noqa: E402

# wm_worker defaults to CPU. The caller alone authorizes CUDA after nvidia-smi.
os.environ['C4_ALLOWED_ROOTS'] = '/root/autodl-tmp/c4:/root/wmcalib:/root/wmvalue20261031'
import numpy as np  # noqa: E402
import torch  # noqa: E402
from measure import native_background_motion  # noqa: E402
from wms.model.cav import WM_LANE_TARGET_KEY, WM_MERGE_KEY  # noqa: E402
from wms.train.cav_drive import TimelineControllers  # noqa: E402
from wms.sim_bridge import load_controller_dependencies  # noqa: E402

from .audit import ARMS  # noqa: E402
from .reconstruct import flow_at  # noqa: E402
from .restart import align_history, causal_queue_prior, entry_records, imagined_observation  # noqa: E402


def move(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move(v, device) for k, v in x.items()}
    return x


def load(device):
    torch.manual_seed(965140)
    assert frozen.digest(frozen.WM_PATH) == frozen.WM
    # Original config symlink now dangles; private LF-normalized byte-identical
    # copy, with its inheritance chain. Never repair or modify a peer runtime.
    config = Path(__file__).resolve().parents[2]/'configs/c4_wm_loop14_r16_ft.toml'
    assert frozen.digest(config) == 'f95f7c4cbb501ccb85bb3968e29f3d6ca53db4ecef5f586225f96df1f48193e4'
    models, receipt = frozen.load_members([str(frozen.WM_PATH)], [str(config)], device)
    model = models[0]
    payload = torch.load(frozen.CANDIDATE, map_location='cpu', weights_only=False)
    cfg = frozen.replace(frozen.DreamerConfig(**payload['config']), horizon=600,
                         entrant_learned_mix=True)
    assert cfg.future_samples == 8 and not model.physics.entry_learned
    model.cav_exec_cap = cfg.cav_exec_cap
    cp, ce, _, _ = frozen.load_policy_components(frozen.CANDIDATE, device=device)
    torch.nn.init.zeros_(cp.residual_head.weight)
    torch.nn.init.zeros_(cp.residual_head.bias)
    ce = ce.encoder if isinstance(ce, frozen.GroupSignal) else ce
    cp.eval()
    ce.eval()
    import json
    binding = json.loads((frozen.OWN/'EXPORT_BOUND.json').read_text())
    assert frozen.digest(Path(binding['path'])) == binding['sha256']
    mp = frozen.load_policy_fn(binding['path'], device=device)
    mp.policy.eval()
    mp.group_encoder.eval()
    assert mp.native_lanes and not mp.has_lane_head
    deps = load_controller_dependencies(frozen.OWN/'e3final/runtime')
    dependencies = (deps.config, deps.circles_type, deps.direct_target, deps.neighbor_type,
                    deps.frame_type, deps.pi_type, deps.u_report_type, deps.pi_params_type)
    native = frozen.native_fn('head', a_max=1.2)
    return dict(models=models, model=model, cfg=cfg, cp=cp, ce=ce, mp=mp, native=native,
                dependencies=dependencies, receipt=receipt, binding=binding)


def make_batch(p, columns, model, queue300, previous_commands=None):
    raw = align_history(p, columns)
    k = len(columns)
    state = {name: torch.from_numpy(value.copy())[None] for name, value in dict(
        s_m=raw['kin'][-1, :k, 0], v_mps=raw['kin'][-1, :k, 4],
        a_mps2=raw['kin'][-1, :k, 5], length_m=raw['kin'][-1, :k, 6],
        lane=raw['lane'][-1, :k], present=raw['mask'][-1, :k], is_cav=raw['cav'][-1, :k]).items()}
    if previous_commands is not None:
        state['cmd_accel_mps2'] = torch.tensor([[previous_commands.get(n, 0.) for n in columns]],
                                             dtype=torch.float32)
        state['lane_intent'] = torch.zeros((1, k))
    kw = {}
    kw['history'], kw['history_mask'] = frozen.history_tensors(raw, history_len=int(model.history_len))
    if model.long_history:
        kw['driver_history'], kw['driver_history_mask'] = frozen.history_tensors(
            raw, model.history_s, model.history_stride_s)
    kw = {key: value[:, :k].clone() for key, value in kw.items()}
    records = entry_records(p)
    t0 = float(p['time_s'][-1])
    events = np.array([[r['time'], int(r['lane'] == 2)] for r in records]).reshape(-1, 2)
    recent = [r for r in records if r['time'] >= t0-60]
    gaps = [r for r in recent if r['gap'] is not None]
    counter = frozen._counts_counter(events[:, 0], events[:, 1].astype(int), t0)
    hist = frozen.build_entrance_history(counter=counter,
        times=np.array([r['time'] for r in recent]),
        entrance=np.array([int(r['lane'] == 2) for r in recent]),
        speed=np.array([r['v_mps'] for r in recent]),
        local_speed=np.array([r['local_speed'] for r in recent]),
        lane=np.array([r['lane'] for r in recent]),
        gap=np.array([r['gap'] for r in gaps]),
        gap_entrance=np.array([int(r['lane'] == 2) for r in gaps]), t0_s=t0)
    hist.cav_share = np.array([np.mean([r['is_cav'] for r in recent if int(r['lane'] == 2) == e])
                               if any(int(r['lane'] == 2) == e for r in recent) else 0.
                               for e in (0, 1)])
    batch = frozen.StartBatch(state, kw, [dict(t0=t0)],
        torch.tensor(hist.to_features(), dtype=torch.float32)[None],
        entry_queue=torch.tensor(causal_queue_prior(records, queue300, t0))[None])
    return batch, events


def replay_baseline(timeline, p, columns, arm):
    """Restore PI/CIRCLES memory from real prefix, never reset at late decision."""
    if p is None:
        return
    t0 = int(p['time_s'][-1])
    names = {n: j for j, n in enumerate(columns)}
    action = 'pi-external-u' if arm == 'pi' else 'damping'
    for tick in range(600, 2*t0, 2):
        # Retain all recorded neighbours at this historical tick, including exits.
        slots = np.flatnonzero(p['present'][tick])
        extra = [j for j in slots if p['names'][int(p['vid_index'][tick, j])] not in names]
        count = len(columns)+len(extra)
        frame = {k: torch.zeros((1, count), dtype=torch.float32)
                 for k in ('s_m', 'v_mps', 'a_mps2', 'length_m')}
        frame['length_m'].fill_(5.)
        frame['lane'] = torch.full((1, count), -1, dtype=torch.long)
        frame['present'] = torch.zeros((1, count), dtype=torch.bool)
        frame['is_cav'] = torch.zeros((1, count), dtype=torch.bool)
        extra_index = {j: len(columns)+i for i, j in enumerate(extra)}
        for j in slots:
            name = p['names'][int(p['vid_index'][tick, j])]
            col = names[name] if name in names else extra_index[j]
            for k in ('s_m', 'v_mps', 'a_mps2', 'lane', 'is_cav'):
                frame[k][0, col] = p[k][tick, j].item()
            frame['present'][0, col] = True
        for col in range(len(columns)):
            vid = f'0:{col}'
            if bool(frame['present'][0, col] & frame['is_cav'][0, col]) and frame['lane'][0, col] >= 0:
                timeline._new_command(action, vid, 0, col, int(tick/2-300), frame)
            else:
                timeline._invalidate(vid)
    timeline.step_index = 2*(t0-300)  # keep the original controller timestamp clock


def controllers(rt, slots, prefixes=None, columns=None):
    model = rt['model']
    native = rt['native']
    calls = {}
    for arm in ('ctrl0', 'mappo'):
        calls[arm] = frozen.callback_for(arm, rt['cp'], rt['ce'], rt['mp'], native)
    for arm in ('pi', 'circles'):
        ids = torch.full((1, 601, slots), 2 if arm == 'pi' else 1, dtype=torch.long)
        timeline = TimelineControllers(model, dict(controller=ids,
            acceleration_mps2=torch.zeros_like(ids, dtype=torch.float32),
            target_speed_mps=torch.zeros_like(ids, dtype=torch.float32)), dependencies=rt['dependencies'])
        if prefixes is not None:
            replay_baseline(timeline, prefixes[arm], columns, arm)

        def baseline(state, timeline=timeline):
            mask = state['present'].bool() & state['is_cav'].bool()
            state[WM_MERGE_KEY] = mask & (state['lane'] == 2)
            lane = torch.where(state['lane'] != 2, state[WM_LANE_TARGET_KEY], state['lane'])
            return frozen.parity_motion(state, lane, timeline(state)[..., 3])
        calls[arm] = baseline

    def native_call(state):
        mask = state['present'].bool() & state['is_cav'].bool()
        state[WM_MERGE_KEY] = mask & (state['lane'] == 2)
        lane = torch.where(state['lane'] != 2, state[WM_LANE_TARGET_KEY], state['lane'])
        return native_background_motion(state, lane, native)
    calls['native'] = native_call

    def callback(state):
        outputs, merges = [], []
        for i, arm in enumerate(ARMS):
            row = {k: v[i:i+1] if torch.is_tensor(v) and v.ndim and v.shape[0] == len(ARMS) else v
                   for k, v in state.items()}
            outputs.append(calls[arm](row))
            merges.append(row[WM_MERGE_KEY])
        state[WM_MERGE_KEY] = torch.cat(merges)
        return torch.cat(outputs)
    callback.native_lane_mask = lambda state: state['present'].bool() & state['is_cav'].bool()
    return callback


class Recorder:
    """One pre-motion frame per step; end-step source queue, imaginary entry events."""
    def __init__(self, process):
        self.process = process
        self.frames, self.queues, self.births = [], [], []

    def insert(self, state):
        mask = self.process.insert(state)
        self.frames.append({k: state[k].detach().clone() for k in
                            ('s_m', 'lane', 'v_mps', 'a_mps2', 'present', 'is_cav')})
        self.queues.append(self.process.pending.copy())
        self.births.append(mask.detach().clone())
        return mask


def score(roll, recorder, prefix_events, t0):
    """Full cost clocks and sensed terminal X, without any real future state."""
    from .core import features
    pre = torch.stack([f['present'] for f in recorder.frames], 1).bool()
    post = roll['masks'].bool()
    before_v = torch.stack([f['v_mps'] for f in recorder.frames], 1)
    speed = torch.where(post, roll['speeds'], before_v)
    road = .5*torch.where(pre | post, (1-speed/31.29).clamp_min(0), 0).double().sum(-1)
    queues = np.stack(recorder.queues, 1)
    source = .5*torch.as_tensor(queues.sum(-1), device=road.device, dtype=torch.float64)
    clock = (road+source).cumsum(1).cpu().numpy()
    windows = {}
    for h in (60, 90):
        tick = 2*h-1
        if tick >= clock.shape[1]:
            continue
        x = []
        for row in range(len(ARMS)):
            # CAV identity is invariant in a virgin slot; exited presence is masked.
            frame = {k: v[row].detach().cpu().numpy() for k, v in recorder.frames[tick].items()}
            for key, rk in (('s_m', 'positions'), ('lane', 'lanes'), ('v_mps', 'speeds'),
                            ('a_mps2', 'accel'), ('present', 'masks')):
                frame[key] = roll[rk][row, tick].detach().cpu().numpy()
            events = [prefix_events[row]]
            for j in range(tick+1):
                born = recorder.births[j][row].detach().cpu().numpy()
                lane = recorder.frames[j]['lane'][row].detach().cpu().numpy()
                if born.any():
                    events.append(np.column_stack([np.full(born.sum(), t0+(j+1)*.5), lane[born] == 2]))
            events = np.concatenate(events)
            observed = imagined_observation(frame, events, t0+h)
            # Explicitly assert boundary counts survived the real/imagined splice.
            assert np.array_equal(observed['flow_counts'], flow_at(events, t0+h))
            x.append(features(observed))
        windows[str(h)] = dict(J=clock[:, tick].tolist(), x=np.stack(x).tolist())
    return dict(J=clock[:, -1].tolist(), windows=windows,
                road=road.sum(1).cpu().tolist(), source=source.sum(1).cpu().tolist(),
                steps=clock.shape[1])


def rollout(rt, batches, events, t0, fi, device, prefixes=None, columns=None, seconds=None):
    horizon = int(900-t0) if seconds is None else seconds
    padded = []
    plans = []
    for batch in batches:
        b = frozen.with_entry_slots(copy.deepcopy(batch), 900-t0, 1920,
            torch.Generator().manual_seed(1_000_003*191), queue_state=True)
        padded.append(b)
    # Equal padding is mandatory for identity CRN and concatenation. Rates differ
    # across late-arm observations, so pad all to the largest physical slot count.
    max_slots = max(b.state['s_m'].shape[1] for b in padded)
    from wms.control.dreamer import _pad_slots, _PAD_FILL
    for b in padded:
        k = b.state['s_m'].shape[1]
        if k != max_slots:
            b.state = {key: _pad_slots(v, k, max_slots-k, _PAD_FILL.get(key, 0)) for key, v in b.state.items()}
            b.rollout_kwargs = {key: _pad_slots(v, k, max_slots-k) for key, v in b.rollout_kwargs.items()}
            b.entry_spec['pad'] += max_slots-k
        b.state = move(b.state, device)
        b.rollout_kwargs = move(b.rollout_kwargs, device)
        plans.append(frozen.SampledFutures(frozen.FuturesConfig.from_trainer(rt['cfg']), 190).plan(
            rt['models'], b.state, b.rollout_kwargs, int(2*(900-t0)))[0][fi])
    def cat(values):
        if torch.is_tensor(values[0]):
            return torch.cat(values)
        if isinstance(values[0], dict):
            return {k: cat([v[k] for v in values]) for k in values[0]}
        assert all(v == values[0] for v in values)
        return values[0]
    initial = cat([b.state for b in padded])
    kw = cat([{**b.rollout_kwargs, **p} for b, p in zip(padded, plans)])
    frozen.pair_merge_streams(initial, torch.zeros(5, dtype=torch.long, device=device), 1, 190+fi)
    initial['exit_s_m'] = torch.full_like(initial['s_m'], frozen.road_end_exit_bound(rt['model'].physics))
    lanes = frozen.lane_uniform_draws(rt['models'], 1, int(2*(900-t0)), max_slots,
        190+1000003*fi if fi else 190, device=device)
    if lanes is not None:
        kw['lane_uniform'] = frozen.expand(lanes, 5)
    spec = dict(histories=[b.entry_spec['histories'][0] for b in padded],
                queues=[b.entry_spec['queues'][0] for b in padded],
                seeds=[b.entry_spec['seeds'][0] for b in padded],
                k0=padded[0].entry_spec['k0'])
    assert len({b.entry_spec['k0'] for b in padded}) == 1
    process = frozen._entry_process(spec, list(range(5)), max_slots,
        rt['cfg'].entry_insert_rule, None, couple_draws=True, sample_index=fi)
    recorder = Recorder(process)
    kw['entries'] = recorder
    kw['entrant_driver_sampler'] = frozen.make_sampler(rt['model'], initial, kw, 190+1000003*fi, [0]*5)
    # Keep long-prefix plans identical; consume only the requested prefix.
    if 'accel_noise' in kw:
        kw['accel_noise'] = kw['accel_noise'][:, :2*horizon]
    if 'lane_uniform' in kw:
        kw['lane_uniform'] = kw['lane_uniform'][:, :2*horizon]
    cb = controllers(rt, max_slots, prefixes, columns)
    torch.manual_seed(965140)
    with torch.no_grad():
        roll = rt['model'].rollout(initial, n_steps=2*horizon, cav_controller=cb, **kw)
        assert torch.isfinite(roll['speeds']).all()
        result = score(roll, recorder, events, t0)
    result['slots'] = max_slots
    result['mix_assigned'] = kw['entrant_driver_sampler'].assigned
    return result
