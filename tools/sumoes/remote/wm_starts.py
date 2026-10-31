"""WMDATA CTRL0 job -> WM StartBatch, and the CTRL0-path WM controller callback.

Verbatim logic of the frozen S-SPEEDHEAD wm_worker (make_starts / callback_for 'ctrl0'),
re-pointed at one S-WMDATA ``s<seed>_ctrl0`` job directory (observable t300 state,
<=t300 history, observed queue history; no scenario/type files).
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def history_tensors(raw, history_s=None, history_stride_s=1., history_len=10):
    # Verbatim S-SPEEDHEAD wm_worker featuriser (same as traffic_shepherd.c4.wms_glue).
    from wms.data.history import long_history_features
    from wms.model.history import history_step_features
    kin = torch.as_tensor(raw['kin'])
    pres = torch.as_tensor(raw['mask'])
    target = torch.as_tensor(raw['target_mask'])
    if history_s is not None:
        f, m = long_history_features(kin, pres, target, pres[-1], history_s=history_s,
                                     history_stride_s=history_stride_s, dt_s=.5)
        return f[None], m[None]
    same = torch.ones_like(pres)
    for j in range(len(pres) - 2, -1, -1):
        same[j] = same[j + 1] & target[j]
    valid = same & pres
    gap = torch.full(pres.shape, 250.)
    for t in range(len(pres)):
        lane = kin[t, :, 1:4].argmax(-1)
        s = kin[t, :, 0]
        ds = s[None, :] - s[:, None]
        ok = valid[t][None, :] & (lane[None, :] == lane[:, None]) & (ds > 0)
        gap[t] = torch.where(ok, ds - kin[t, :, 6][None, :],
                             torch.full_like(ds, 250.)).min(-1).values.clamp(.3, 250.)
    f = history_step_features(kin, gap)
    n = min(len(pres), history_len)
    out = torch.zeros(kin.shape[1], history_len, f.shape[-1])
    mask = torch.zeros(kin.shape[1], history_len, dtype=torch.bool)
    out[:, -n:] = f[-n:].transpose(0, 1)
    mask[:, -n:] = valid[-n:].transpose(0, 1)
    return out[None], mask[None]


def make_start(model, root):
    from wms.control.start_states import StartBatch
    from wms.model.entries import build_entrance_history, _counts_counter
    result = json.loads((root / 'result.json').read_text())
    for name in ('start.pt', 'start_history.npz'):
        assert digest(root / name) == result['output_hashes'][name]
    st = torch.load(root / 'start.pt', map_location='cpu', weights_only=False)
    with np.load(root / 'start_history.npz', allow_pickle=False) as f:
        raw = {k: f[k] for k in ('kin', 'mask', 'target_mask')}
    assert st['t0'] == 300 and st['initial_state_sha256'] == result['metrics']['initial_state_sha256']
    state = st['state']
    keep = torch.where(state['present'][0])[0]
    kw = {}
    kw['history'], kw['history_mask'] = history_tensors(raw, history_len=int(model.history_len))
    if model.long_history:
        kw['driver_history'], kw['driver_history_mask'] = history_tensors(
            raw, model.history_s, model.history_stride_s)
    state = {k: v[:, keep].clone() for k, v in state.items()}
    kw = {k: v[:, keep].clone() for k, v in kw.items()}
    events = st['past_entries']
    assert all(0 < e['time'] <= 300 for e in events)
    times = np.array([e['time'] for e in events])
    entrances = np.array([int(e['lane'] == 2) for e in events])
    counter = _counts_counter(times, entrances, 300.)
    selected = [e for e in events if e['time'] >= 240]
    en = np.array([int(e['lane'] == 2) for e in selected])
    speeds = np.array([e['v_mps'] for e in selected])
    local, gaps, gap_en = [], [], []
    for e in selected:
        j = int(round((e['time'] - 180.) * 2))
        kin, pres = raw['kin'][j], raw['mask'][j]
        lane = kin[:, 1:4].argmax(-1)
        idx = np.where(pres & (lane == e['lane']))[0]
        idx = idx[np.argsort(kin[idx, 0])]
        local.append(max(float(kin[idx[:3], 4].mean()) if len(idx) else e['v_mps'], 1.))
        ahead = idx[kin[idx, 0] > e['s_m']]
        if len(ahead):
            gap = float(kin[ahead[0], 0] - e['s_m'] - 5.)
            if gap > 0:
                gaps.append(gap)
                gap_en.append(int(e['lane'] == 2))
    history = build_entrance_history(counter=counter, times=np.array([e['time'] for e in selected]),
                                     entrance=en, speed=speeds, local_speed=np.array(local),
                                     lane=np.array([e['lane'] for e in selected]),
                                     gap=np.array(gaps), gap_entrance=np.array(gap_en), t0_s=300.)
    history.cav_share = np.array([np.mean([e['is_cav'] for e in selected if int(e['lane'] == 2) == a])
                                  if np.any(en == a) else 0. for a in (0, 1)])
    q = np.array(st['observed_queue_history'][-1][1:], dtype=float)
    assert q.sum() == st['queue']
    ins = np.bincount(entrances, minlength=2)
    demand = (ins + q) / 300.
    shares = np.array([np.mean([e['is_cav'] for e in events if int(e['lane'] == 2) == a])
                       if ins[a] else 0. for a in (0, 1)])
    entry_queue = torch.tensor(np.r_[q, demand, shares], dtype=torch.float32)[None]
    job = result['job']
    meta = dict(world_id=f"w{job['seed']}m{job['demand']}p{int(job['penetration'] * 100):02d}t300",
                split='TRAIN', seed=job['seed'], t0=300,
                initial_state_sha256=st['initial_state_sha256'])
    return StartBatch(state, kw, [meta], torch.tensor(history.to_features(), dtype=torch.float32)[None],
                      entry_queue=entry_queue)


def callback(cp, ce, native):
    """S-SPEEDHEAD callback_for('ctrl0') with the given residual policy (head = theta)."""
    from wms.control.bridge import make_cav_controller
    from wms.control.safety import safe_accel
    from wms.control.train_cli import GroupSignal

    def deterministic(obs, signal, **kw):
        return cp(obs, signal, deterministic=True, **kw)
    deterministic.accel_max = cp.accel_max
    deterministic.residual_on_native = bool(getattr(cp, 'residual_on_native', False))
    bridge = make_cav_controller(deterministic, GroupSignal(ce), safe_accel, native=native,
                                 native_accel_max=1.2, symmetric_exec_cap=True,
                                 mask_forced_merge=True, native_ramp_merge='wm_law',
                                 native_lane_change=True)

    def run(state):
        return bridge(state)
    run.native_lane_mask = bridge.native_lane_mask
    return run
