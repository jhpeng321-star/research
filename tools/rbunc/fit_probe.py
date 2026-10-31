"""CPU FIT-only source-attribution probe; deterministic legacy physics reference.

This is a sampler engineering diagnostic, not the integrated WM-PHYS efficacy
panel. Frozen motion encoder + deterministic IDM/MOBIL, all learned corrections
absent. Only native observable starts 969400..969431 are admitted. No SUMO.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

BAD = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
       'hetdata_truth', 'vehicle_params', 'driver_params', 'vtype')
BASE = Path('/home/ubuntu/wmdata20261031/jobs')
CHECKPOINT = Path('/home/ubuntu/e3final/inputs/wm_s_best.pt')
CHECKPOINT_SHA = '5b2569418bf651b8ef2c3a2abdc110ac6860c32dd92dca8cdf3b2f48b1c8cbb2'


def safe(path):
    path = Path(path)
    if any(word in str(path.resolve()).lower() for word in BAD):
        raise ValueError('prohibited input path')
    return path


def digest(path):
    return hashlib.sha256(safe(path).read_bytes()).hexdigest()


def install_guard():
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            safe(os.fsdecode(args[0]))
    sys.addaudithook(guard)


def prepare(world):
    from wms.control.hidden_state import Population
    from wms.data.history import long_history_features
    from wms.model.checkpoint import load_checkpoint
    from wms.model.entries import EntranceQueue, build_entrance_history
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel
    from wms.physics.config import PhysicsConfig
    root = safe(BASE/f's{world}_native')
    st = torch.load(root/'start.pt', map_location='cpu', weights_only=False)
    assert st['t0'] == 300
    with np.load(root/'start_history.npz', allow_pickle=False) as z:
        raw = {key: z[key] for key in ('kin', 'mask', 'target_mask')}
    assert len(raw['kin']) == 241  # genuine pre-t0 t180..300
    assert digest(CHECKPOINT) == CHECKPOINT_SHA
    loaded = load_checkpoint(CHECKPOINT)
    source = loaded.model.eval()
    kin, pres = torch.tensor(raw['kin']), torch.tensor(raw['mask']).bool()
    features, mask = long_history_features(kin, pres,
        torch.tensor(raw['target_mask']).bool(), pres[-1], history_s=source.history_s,
        history_stride_s=source.history_stride_s)
    history = dict(driver_history=features[None], driver_history_mask=mask[None])
    # Infer BEFORE any t0 padding so the population excludes padding by construction.
    with torch.inference_mode():
        _, _, params = source.encode_drivers(st['state'], **history)
    keep = st['state']['present'][0].nonzero().flatten()
    fields = ('s_m', 'v_mps', 'a_mps2', 'lane', 'length_m', 'present', 'is_cav', 'junction')
    state = {key: st['state'][key][:, keep].clone() for key in fields if key in st['state']}
    history = {key: value[:, keep].clone() for key, value in history.items()}
    params = {key: value[:, keep].clone() for key, value in params.items()}
    population = Population.from_motion(params, state, history['driver_history_mask'])
    # Reference physics architecture has no trainable correction modules in use.
    # It calls the untouched frozen source encoder through its original history API.
    model = WmSModel(ModelSpec(rollout_mode='driver_params'), physics=PhysicsConfig()).eval()
    model.encode_drivers = source.encode_drivers
    model.driver_encoder = source.driver_encoder
    events = st['past_entries']
    assert all(0 < e['time'] <= 300 for e in events)
    entrances = np.array([int(e['lane'] == 2) for e in events])
    inserted = np.bincount(entrances, minlength=2)
    backlog = np.array(st['observed_queue_history'][-1][1:], dtype=float)
    assert backlog.sum() == st['queue']
    rates = (inserted + backlog) / 300
    cav_shares = np.array([np.mean([e['is_cav'] for e in events if int(e['lane'] == 2) == en])
                           if inserted[en] else 0. for en in (0, 1)])
    lane1 = np.mean([e['lane'] == 1 for e in events if e['lane'] != 2])
    eh = build_entrance_history(counter=np.zeros((2, 30)))
    eh.rate, eh.headway_cv, eh.cav_share, eh.lane_p1 = rates, np.ones(2), cav_shares, float(lane1)
    queue = EntranceQueue(backlog, rates, cav_share=cav_shares)
    # Padding is a computational allocation, not a behavioral parameter. Report
    # no-slot refusals, and do not silently treat capacity-constrained runs as valid.
    pad = 128
    for key, value in list(state.items()):
        state[key] = torch.cat((value, torch.zeros((1, pad), dtype=value.dtype)), -1)
    state['length_m'][:, -pad:] = 5.
    for key, value in list(history.items()):
        shape = (1, pad, *value.shape[2:])
        history[key] = torch.cat((value, torch.zeros(shape, dtype=value.dtype)), 1)
    from wms.physics.batched import road_end_exit_bound
    state['exit_s_m'] = torch.full_like(state['s_m'], road_end_exit_bound(model.physics))
    receipt = dict(world=world, t0=300, history_s=source.history_s,
        observed_history_counts=history['driver_history_mask'][:, :len(keep)].sum(-1).tolist(),
        checkpoint_sha256=CHECKPOINT_SHA, initial_sha256=st['initial_state_sha256'],
        inputs={str(root/n): digest(root/n) for n in ('start.pt', 'start_history.npz')},
        population_size=len(population.hdv[0]), demand_rates=rates.tolist(),
        backlog=backlog.tolist(), cav_shares=cav_shares.tolist(), padded_slots=pad,
        physics='legacy deterministic IDM/MOBIL attribution reference; NOT integrated WM-PHYS')
    return model, state, history, population, eh, queue, receipt


class RecordEntries:
    def __init__(self, process):
        self.process = process
        self.queue = []
        self.states = []
    def __getattr__(self, name):
        return getattr(self.process, name)
    def insert(self, state):
        self.queue.append(self.process.pending.copy())  # left boundary
        new = self.process.insert(state)
        self.states.append(state['present'].clone())
        return new


def native_controller(physics):
    from wms.physics.accel import baseline_accel
    from wms.physics.state import state_kin
    from wms.model.driver_encoder import PARAMETER_NAMES
    def callback(state):
        encoded = state['inferred_driver_parameters']
        params = {name: encoded[..., i] for i, name in enumerate(PARAMETER_NAMES)}
        accel = baseline_accel(state_kin(state), state['present'], physics, params)
        v = (state['v_mps'] + physics.dt_s * accel).clamp_min(0)
        dt = torch.where(accel < 0,
            (state['v_mps'] / (-accel).clamp_min(torch.finfo(accel.dtype).eps)).clamp(0, physics.dt_s),
            torch.full_like(accel, physics.dt_s))
        s = state['s_m'] + state['v_mps'] * dt + .5 * accel * dt.square()
        return torch.stack((s, state['lane'].float(), v, accel), -1)
    return callback


def run(model, state, history, future, horizon):
    kw = dict(future.kwargs)
    process = kw['entries']
    entries = RecordEntries(process)
    kw['entries'] = entries
    initial = {k: v.clone() for k, v in state.items()}
    before_rng = torch.get_rng_state().clone()
    with torch.inference_mode():
        roll = model.rollout(initial, n_steps=2*horizon,
            cav_controller=native_controller(model.physics), **history, **kw)
    assert torch.equal(before_rng, torch.get_rng_state()), 'physics consumed global RNG'
    assert torch.isfinite(roll['speeds']).all() and torch.isfinite(roll['positions']).all()
    assert process.stats[0].no_slot == 0 and process.stats[0].dropped_pending == 0
    assert (process.pending[0] == np.rint(process.queues[0].backlog).astype(int) +
            process.due_total[0] - np.array(process.stats[0].per_entrance)).all()
    # Physical post-step road delay plus pre-step source rectangle, explicit contract.
    masks = torch.stack(entries.states, 1)
    road = float(.5 * torch.where(masks, (1-roll['speeds']/31.29).clamp_min(0), 0).sum())
    source = float(.5 * np.stack(entries.queue).sum())
    tape_hash = hashlib.sha256(b''.join(roll[k].cpu().numpy().tobytes()
        for k in ('positions', 'speeds', 'lanes', 'masks'))).hexdigest()
    return dict(road_delay_vs=road, source_wait_vs=source, team_delay_vs=road+source,
        arrivals=process.due_total[0].tolist(), inserted=process.stats[0].inserted,
        no_slot=process.stats[0].no_slot, tape_sha256=tape_hash,
        final_queue=process.pending[0].tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--world', type=int, required=True)
    p.add_argument('--horizon', type=int, default=120)
    p.add_argument('--sampling-only', action='store_true')
    args = p.parse_args()
    assert args.world in range(969400, 969432) and args.horizon in (60, 120, 300, 600)
    assert args.out.is_dir()
    os.nice(19)
    torch.set_num_threads(1)
    install_guard()
    begin = time.monotonic()
    from wms.control.hidden_state import plan_hidden_state_futures
    model, state, history, population, eh, queue, receipt = prepare(args.world)
    rows, driver_spread, request_spread = {}, {}, {}
    modes = [('both', True, True)]
    if args.world == 969400:
        modes += [('drivers', True, False), ('arrivals', False, True), ('fixed', False, False)]
    for name, drivers, arrivals in modes:
        while os.getloadavg()[0] > 24:
            time.sleep(10)
        futures = plan_hidden_state_futures(model, state, history,
            future_sampling='hidden_state', n_samples=8, seed=20261031, row_keys=[args.world],
            population=population, entrance_histories=[eh], queues=[queue],
            sample_present=drivers, sample_arrivals=arrivals)
        present_hdv = state['present'] & ~state['is_cav']
        driver_spread[name] = {param: float(torch.stack([
            f.kwargs['driver_params'][param][present_hdv] for f in futures]).std(0).mean())
            for param in futures[0].kwargs['driver_params']}
        # Request-only streams at all target horizons, without reading future demand.
        request_spread[name] = []
        for f in copy.deepcopy(futures):
            process = f.kwargs['entries']
            clocks = {}
            for step in range(1200):
                process._emit_due(0)
                if step + 1 in (240, 600, 1200):
                    clocks[str((step+1)//2)] = process.due_total[0].tolist()
            request_spread[name].append(clocks)
        rows[name] = []
        if not args.sampling_only:
            for f in futures:
                result = run(model, state, history, f, args.horizon)
                rows[name].append(result)
                print('DONE', args.world, name, f.receipt['future'], result['team_delay_vs'], flush=True)
            if name == 'fixed':
                assert len({r['tape_sha256'] for r in rows[name]}) == 1
        receipt[name] = futures[0].receipt
    # Factual outcome is read ONLY after the sampling/rollout; never used for settings.
    real_path = safe(BASE/f's{args.world}_native'/f'cumulative_{args.horizon}.json')
    real = json.loads(real_path.read_text()) if args.horizon != 60 else None
    output = dict(schema='rbunc_fit/1', receipt=receipt, horizon=args.horizon,
        modes=rows, driver_sd=driver_spread, requests=request_spread,
        real=None if real is None else {k: real[k] for k in
            ('team_delay_vs', 'road_delay_vs', 'source_wait_vs', 'arrivals')},
        real_sha256=None if real is None else digest(real_path), seconds=time.monotonic()-begin,
        source_sha256=digest(__file__), sampler_sha256=digest(
            Path(__file__).resolve().parents[2]/'src/wms/control/hidden_state.py'))
    target = args.out/f'w{args.world}_h{args.horizon}.json'
    with target.open('x') as f:
        json.dump(output, f, indent=1, allow_nan=False)
    print('COMPLETE', args.world, round(output['seconds'], 2), flush=True)


if __name__ == '__main__':
    main()
