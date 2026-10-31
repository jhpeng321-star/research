"""FIT-only merge diagnostics. Never scans data roots or opens latent driver files.

Two contracts: recorded-state one-step mechanics, and autonomous t0-cohort
rollout (no entrants). The latter is deliberately not a fleet-delay qualification.
Can also score an integrator's full predictions via --prediction-dir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from wms.model.spec import ModelSpec
from wms.model.entries import ENTRY_S_M
from wms.model.wms import WmSModel
from wms.model.sumo_insertion import insertion_speed
from wms.physics.config import PhysicsConfig
from wms.physics.junction_merge import apply_merge
from wms.physics.state import state_kin

BAD = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
       'hetdata_truth', 'vehicle_params', 'driver_params', 'vtype')
ALLOW = {'episode.json', 'time_s.npy', 'veh_kin.npy', 'veh_mask.npy', 'vid.npy', 'is_cav.npy'}


def safe(path):
    path = Path(path).resolve()
    if any(bad in str(path).lower() for bad in BAD):
        raise ValueError(f'prohibited path: {path}')
    return path


def summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {'n': 0, 'mean': None, 'q10': None, 'q50': None, 'q90': None}
    return dict(n=len(values), mean=float(values.mean()),
                **dict(zip(('q10', 'q50', 'q90'), map(float, np.quantile(values, [.1, .5, .9])))))


def load_episode(root, seed, arm):
    if not 969400 <= seed <= 969431:
        raise ValueError('FIT allowlist is exactly 969400..969431')
    root = safe(root)
    record = json.loads(safe(root / f'world_{seed}.json').read_text())[arm]
    ep = safe(root / 'episodes' / record['episode'])
    meta = json.loads((ep / 'episode.json').read_text())
    if meta['seed'] != seed or meta['arm'] != arm:
        raise ValueError('episode identity mismatch')
    def load(name):
        if name not in ALLOW:
            raise ValueError('observable field not allowlisted')
        return np.load(safe(ep / name), mmap_mode='r', allow_pickle=False)
    arrays = {name: load(name) for name in ('time_s.npy', 'veh_kin.npy', 'veh_mask.npy', 'vid.npy')}
    return ep, meta, arrays


def make_state(kin, mask):
    kin = torch.as_tensor(np.array(kin, copy=True), dtype=torch.float32).unsqueeze(0)
    return dict(s_m=kin[..., 0], v_mps=kin[..., 4], a_mps2=kin[..., 5],
                length_m=kin[..., 6], lane=kin[..., 1:4].argmax(-1),
                present=torch.as_tensor(np.array(mask, copy=True), dtype=torch.bool).unsqueeze(0),
                is_cav=torch.zeros(kin.shape[:2], dtype=torch.bool), exit_s_m=553.33)


def events(pos, lane, mask, times):
    rows = []
    for j in range(1, len(times)):
        ids = np.flatnonzero(mask[j - 1] & mask[j] & (lane[j - 1] == 2) & (lane[j] == 1))
        rows.extend(dict(slot=int(i), time_s=float(times[j]), position_m=float(pos[j, i])) for i in ids)
    return rows


def score_trajectories(real, pred, times, dt, label):
    """Same slot IDs required; compare first ramp->mainline events and delay."""
    re = events(real['s'], real['lane'], real['mask'], times)
    pe = events(pred['s'], pred['lane'], pred['mask'], times)
    rm = {r['slot']: r for r in re}
    pm = {r['slot']: r for r in pe}
    common = sorted(rm.keys() & pm.keys())
    def delay(x):
        # Named diagnostic: ramp-lane normalized speed-loss vehicle-time.
        active = x['mask'][:-1] & (x['lane'][:-1] == 2)
        loss = np.clip(1 - x['v'][:-1] / 22.22, 0, 1)
        return (active * loss).sum(axis=0) * dt
    rd, pd = delay(real), delay(pred)
    risk = np.any((real['mask'] & (real['lane'] == 2)) | (pred['mask'] & (pred['lane'] == 2)), axis=0)
    return dict(contract=label, real_merges=len(re), predicted_merges=len(pe), matched=len(common),
        missed_slots=sorted(rm.keys() - pm.keys()), extra_slots=sorted(pm.keys() - rm.keys()),
        time_error_s=summary([pm[i]['time_s'] - rm[i]['time_s'] for i in common]),
        position_error_m=summary([pm[i]['position_m'] - rm[i]['position_m'] for i in common]),
        real_merge_positions_m=summary([r['position_m'] for r in re]),
        predicted_merge_positions_m=summary([r['position_m'] for r in pe]),
        ramp_speed_loss_veh_s=dict(real=float(rd.sum()), predicted=float(pd.sum()),
            real_per_vehicle=summary(rd[risk]), predicted_per_vehicle=summary(pd[risk])),
        insertion_counts=dict(real=int((real['mask'].any(0) & ~real['mask'][0]).sum()),
                              predicted=int((pred['mask'].any(0) & ~pred['mask'][0]).sum())),
        real_events=re, predicted_events=pe)


@torch.no_grad()
def validate(root, seed, arm, horizon, stride, prediction_dir=None):
    ep, meta, arrays = load_episode(root, seed, arm)
    times = np.asarray(arrays['time_s.npy'])
    kin = arrays['veh_kin.npy']
    mask = np.asarray(arrays['veh_mask.npy'], dtype=bool) & (kin[..., 1:4].sum(-1) > 0)
    t0 = int(meta['t0_index'])
    dt = float(times[t0 + 1] - times[t0])
    end = min(len(times) - 1, t0 + round(horizon / dt))
    if end - t0 != round(horizon / dt):
        raise ValueError('recorded tape shorter than requested horizon')
    cfg = PhysicsConfig(merge_model='sumo', dt_s=dt, sumo_fidelity=True)
    # Conditional mechanics: keep only currently present cars, no data padding.
    tp = fp = fn = tn = 0
    first_accept = {}
    births = accepted_births = 0
    lanes = kin[..., 1:4].argmax(-1)
    a, b, tau, mg, _ = cfg.idm()
    params = dict(accel=a, decel=b, time_headway=tau, min_gap=mg, v0=cfg.v0_mps)
    for j in range(t0, end, stride):
        slots = np.flatnonzero(mask[j])
        if not len(slots):
            continue
        s = make_state(kin[j, slots], mask[j, slots])
        k = state_kin(s)
        # SUMO LC runs on moved positions; constant-acceleration motion is causal.
        v_next = (k[..., 4] + k[..., 5] * dt).clamp_min(0)
        k[..., 0] += .5 * (k[..., 4] + v_next) * dt
        k[..., 4] = v_next
        ramp = s['lane'] == 2
        k[..., 0] = torch.where(ramp, k[..., 0].clamp_max(75.44), k[..., 0])
        proposal = apply_merge(k, s['present'], s['lane'], s['present'], cfg)
        actual = (lanes[j, slots] == 2) & (lanes[j + 1, slots] == 1) & mask[j + 1, slots]
        predicted = (proposal[0].numpy() == 1) & (lanes[j, slots] == 2)
        at_risk = lanes[j, slots] == 2
        tp += int((actual & predicted).sum()); fn += int((actual & ~predicted).sum())
        fp += int((~actual & predicted & at_risk).sum()); tn += int((~actual & ~predicted & at_risk).sum())
        for i in slots[predicted]:
            first_accept.setdefault(int(i), float(times[j + 1]))
        # Insertion service test at each observed birth, without class/assignment
        # data. Other lanes' occupants use the same causal extrapolation.
        for i in np.flatnonzero(mask[j + 1] & ~mask[:j + 1].any(axis=0)):
            lane = int(lanes[j + 1, i])
            if lane not in (0, 1, 2):
                continue
            entry = ENTRY_S_M[lane]
            length = float(kin[j + 1, i, 6])  # observable geometry, never behavior
            front = entry + length + .1
            occupants = np.flatnonzero(lanes[j, slots] == lane)
            if len(occupants):
                rear = min(float(k[0, z, 0] - k[0, z, 6]) for z in occupants)
                gap = rear - front
                leader = float(k[0, min(occupants, key=lambda z: float(k[0, z, 0])), 4])
            else:
                gap, leader = 1e9, 0.
            births += 1
            speed = insertion_speed(min(cfg.v0_mps, 22.22 if lane == 2 else 31.29), gap, leader, params, dt)
            accepted_births += int(speed is not None)
    current_slots = np.flatnonzero(mask[t0])
    real = dict(s=np.asarray(kin[t0:end + 1, current_slots, 0]),
                v=np.asarray(kin[t0:end + 1, current_slots, 4]),
                lane=np.asarray(lanes[t0:end + 1, current_slots]),
                mask=np.asarray(mask[t0:end + 1, current_slots]))
    s = make_state(kin[t0, current_slots], mask[t0, current_slots])
    # Population-only physics, no random network/encoder outputs are consumed.
    model = WmSModel(ModelSpec(), physics=cfg).eval()
    result = model.rollout(s, n_steps=end - t0)
    pred = dict(s=result['positions'][0].numpy(), v=result['speeds'][0].numpy(),
                lane=result['lanes'][0].numpy(), mask=result['masks'][0].numpy())
    # WM result includes post-step states only; prepend the exact t0 snapshot.
    for name in pred:
        pred[name] = np.concatenate((real[name][:1], pred[name]), axis=0)
    cohort = score_trajectories(real, pred, times[t0:end + 1], dt, 'autonomous_t0_cohort_no_entrants_all_native')
    complete = None
    if prediction_dir is not None:
        file = safe(Path(prediction_dir) / f'{seed}_{arm}.npz')
        with np.load(file, allow_pickle=False) as f:
            full = {key: f[key] for key in ('s', 'v', 'lane', 'mask')}
            if not np.array_equal(f['time_s'], times[t0:end + 1]):
                raise ValueError('prediction clock differs')
            if not np.array_equal(f['vid'], arrays['vid.npy']):
                raise ValueError('prediction identity binding differs')
        expected = dict(s=np.asarray(kin[t0:end + 1, :, 0]), v=np.asarray(kin[t0:end + 1, :, 4]),
                        lane=np.asarray(lanes[t0:end + 1]), mask=mask[t0:end + 1])
        if any(full[k].shape != expected[k].shape for k in full):
            raise ValueError('full prediction shape/clock mismatch')
        complete = score_trajectories(expected, full, times[t0:end + 1], dt, 'integrator_full_population')
    return dict(seed=seed, arm=arm, episode=str(ep), t0_s=float(times[t0]), horizon_s=horizon,
        episode_sha256=hashlib.sha256((ep / 'episode.json').read_bytes()).hexdigest(),
        conditional_recorded_state=dict(stride=stride, tp=tp, fp=fp, fn=fn, tn=tn,
            observed_births=births, admitted_at_observed_birth=accepted_births,
            first_accepted_times_s=first_accept), cohort=cohort, full_population=complete)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[969400])
    p.add_argument('--arm', choices=['native', 'ctrl0', 'mappo', 'pi', 'circles'], default='native')
    p.add_argument('--horizon', type=float, default=300.)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--prediction-dir')
    p.add_argument('--output')
    p.add_argument('--inspect', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.stride < 1:
        p.error('stride must be positive')
    if args.inspect:
        ep, meta, arrays = load_episode(args.root, args.seeds[0], args.arm)
        print(json.dumps(dict(episode=str(ep), metadata_keys=sorted(meta),
                              arrays={k: list(v.shape) for k, v in arrays.items()})))
        return
    start = time.monotonic()
    rows = [validate(args.root, seed, args.arm, args.horizon, args.stride, args.prediction_dir)
            for seed in args.seeds]
    report = dict(task='S-RB-MERGE', runtime_s=time.monotonic() - start,
        fit_seed_allowlist=[969400, 969431], rows=rows,
        limitations=['Recorded-state diagnostics condition on observed neighbors each tick.',
            'Cohort rollout excludes entrants and treats all vehicles as native; not full fleet J.',
            '562.4 vs 1265.4 veh-s references are historical, not matched to this panel.',
            'Observed-birth admission is recall-only, not autonomous insertion-count calibration.',
            'MOBIL mainline proposals until LC2013 integration; no training or simulation used.'])
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        output = safe(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
