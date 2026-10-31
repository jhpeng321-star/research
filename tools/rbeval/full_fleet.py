"""FIT native full-fleet diagnostic; motion-only population settings, no training.

Run against an immutable integrated runtime with --runtime. Future observations
are read only after rollout for scoring; demand/backlog are t0 detector values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

BAD = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
       'hetdata_truth', 'vehicle_params', 'driver_params', 'vtype')


def safe(path):
    path = Path(path).resolve()
    if any(word in str(path).lower() for word in BAD):
        raise ValueError(f'prohibited path: {path}')
    return path


def digest(path):
    return hashlib.sha256(safe(path).read_bytes()).hexdigest()


def stats(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return dict(n=0, mean=None, q10=None, q50=None, q90=None)
    return dict(n=len(a), mean=float(a.mean()),
                **dict(zip(('q10', 'q50', 'q90'), map(float, np.quantile(a, [.1, .5, .9])))))


def road_delay(before, after, born, dt):
    """BranchTracker boundary convention: no birth charge; exit uses prior speed."""
    speed = np.where(after['present'], after['v_mps'], before['v_mps'])
    return dt * np.where(before['present'] & ~born, np.maximum(0., 1-speed/31.29), 0.)


def body_overlap_count(s, lane, length, present):
    """Geometric diagnostic, not SUMO's collision-action/collisionMinGap log."""
    count = 0
    worst = 0.
    for ln in (0, 1, 2):
        ids = np.flatnonzero(present & (lane == ln))
        ids = ids[np.argsort(s[ids])]
        gaps = s[ids[1:]]-length[ids[1:]]-s[ids[:-1]]
        count += int((gaps < 0).sum())
        if len(gaps):
            worst = min(worst, float(gaps.min()))
    return count, worst


def transfer_lc_memory(previous, current):
    """Pre-t0 packed slots reorder/recycle; LC memory follows observable identity."""
    from wms.physics.lc2013 import STATE_KEYS
    if previous is None:
        return
    old = {int(v): i for i, v in enumerate(previous['vehicle_id'][0])
           if previous['present'][0, i]}
    ids = current['vehicle_id'][0].tolist()
    index = torch.tensor([[old.get(int(v), 0) for v in ids]])
    found = torch.tensor([[int(v) in old for v in ids]]) & current['present']
    for key in STATE_KEYS:
        if key in previous:
            current[key] = previous[key].gather(-1, index).clone()
    if 'lc2013_present' in current:
        current['lc2013_present'] &= found


def warm_lc_history(model, arrays, t0, names, state, slots):
    from wms.physics.lc2013 import STATE_KEYS, lane_change
    from wms.physics.state import state_kin
    previous = None
    times = arrays['time_s']
    for tick in np.flatnonzero((times >= times[t0]-60) & (times < times[t0])):
        kin = np.array(arrays['veh_kin'][tick], copy=True)
        mask = arrays['veh_mask'][tick].astype(bool) & (kin[:, 1:4].sum(-1) > 0)
        ids = arrays['vid'][tick] if arrays['vid'].ndim == 2 else arrays['vid']
        current = make_state(kin, mask, ids, arrays['is_cav'][tick].astype(bool), names, slots)
        transfer_lc_memory(previous, current)
        lc = lane_change(state_kin(current), current['present'], model.physics,
                         memory=current, identity=current['vehicle_id'], decision_mask=current['present'])
        previous = {**current, **{k: lc[k] for k in STATE_KEYS if k in lc}}
    transfer_lc_memory(previous, state)


def merge_trace(state, physics, time_s):
    """Ten-second diagnostic of the two front ramp cars, never a control input."""
    from wms.physics.junction_merge import merge_geometry
    from wms.physics.state import state_kin
    geometry = merge_geometry(state_kin(state), state['present'], physics)
    ramp = torch.where(state['present'][0] & (state['lane'][0] == 2))[0]
    ramp = ramp[state['s_m'][0, ramp].argsort(descending=True)][:2]
    rows = []
    for slot in ramp.tolist():
        def value(key):
            v = float(geometry[key][0, slot])
            return v if np.isfinite(v) else None
        row = dict(time_s=time_s, identity=int(state['vehicle_id'][0, slot]),
            s_m=float(state['s_m'][0, slot]), v_mps=float(state['v_mps'][0, slot]),
            is_cav=bool(state['is_cav'][0, slot]), safe=bool(geometry['safe'][0, slot]),
            gap_leader_m=value('gap_l'), gap_follower_m=value('gap_f'),
            required_leader_m=value('lead_need'), required_follower_m=value('follow_need'),
            blocked_leader=bool(geometry['blocked_l'][0, slot]),
            blocked_follower=bool(geometry['blocked_f'][0, slot]))
        for label, has, index in (('leader', 'has_l', 'leader'), ('follower', 'has_f', 'follower')):
            i = int(geometry[index][0, slot])
            row[label] = (dict(identity=int(state['vehicle_id'][0, i]),
                               v_mps=float(state['v_mps'][0, i]), s_m=float(state['s_m'][0, i]))
                          if geometry[has][0, slot] else None)
        if 'lc2013_lookahead' in state:
            row['lookahead_mps'] = float(state['lc2013_lookahead'][0, slot])
        rows.append(row)
    return rows


class ObservedEntries:
    """Preserve the process and SourceLedger, record IDs/origins and exact waiting.

    Observing _headway reads an already scheduled due time and makes no new draw.
    This is panel2.py's reported-boundary integral, split by entrance.
    """
    def __init__(self, process, state):
        from wms.model.source_ledger import SourceLedger
        self.process = process
        self.ledgers = [SourceLedger(int(n)) for n in process.pending[0]]
        self.ids = state['vehicle_id'][0].numpy().copy()
        self.origin = state['ramp_origin'][0].numpy().copy()
        self.next_id = -2
        self.previous = None
        self.born = None
        self.due_area = np.zeros(2)
        self.source = np.zeros(2)
        self.births = np.zeros(2, dtype=int)
        self.last_births = []
        old_headway = process._headway

        def headway(b, e):
            if process._started[b]:
                self.due_area[e] += max(0., process.now[b]-process.t_next[b, e])
            return old_headway(b, e)

        process._headway = headway
        old_emit = process._emit_due

        def emit(b):
            due = old_emit(b)
            self.due = np.bincount(due, minlength=2)
            return due

        process._emit_due = emit

    def __getattr__(self, name):
        return getattr(self.process, name)

    def insert(self, state):
        pending = self.process.pending[0].copy()
        self.due_area[:] = 0
        new = self.process.insert(state)
        born = new[0].numpy()
        lanes = state['lane'][0].numpy()
        admitted = np.array([(born & (lanes != 2)).sum(), (born & (lanes == 2)).sum()])
        for e in range(2):
            if self.ledgers[e].advance(int(self.due[e]), int(admitted[e])) != self.process.pending[0, e]:
                raise AssertionError('source ledger differs from pending queue')
        self.source = self.process.dt*pending + self.due_area
        self.births += admitted
        self.last_births = []
        for slot in np.flatnonzero(born):
            self.ids[slot] = self.next_id
            self.next_id -= 1
            self.origin[slot] = lanes[slot] == 2
            self.last_births.append(dict(identity=int(self.ids[slot]), entrance=int(self.origin[slot])))
        state['vehicle_id'] = torch.as_tensor(self.ids[None].copy())
        state['ramp_origin'] = torch.as_tensor(self.origin[None].copy())
        self.previous = {k: state[k][0].numpy().copy() for k in
                         ('s_m', 'v_mps', 'present', 'lane')}
        self.length = state['length_m'][0].numpy().copy()
        self.born = born.copy()
        return new


def make_state(kin, mask, vid, cav, names, slots):
    if mask.sum() > slots:
        raise ValueError('initial cohort exceeds capacity')
    active = np.flatnonzero(mask)
    n = len(active)
    k = np.zeros((slots, 7), dtype=np.float32)
    k[:, 6] = 5.
    k[:n] = kin[active]
    ids = np.full(slots, -1, dtype=np.int64)
    ids[:n] = vid[active]
    present = np.arange(slots) < n
    ramp_origin = np.zeros(slots, dtype=bool)
    ramp_origin[:n] = [names[int(i)].startswith('ramp.') for i in ids[:n]]
    to = lambda x: torch.as_tensor(x[None].copy())
    return dict(s_m=to(k[:, 0]), v_mps=to(k[:, 4]), a_mps2=to(k[:, 5]),
                length_m=to(k[:, 6]), lane=to(k[:, 1:4].argmax(-1)),
                present=to(present), is_cav=to(np.r_[cav[active], np.zeros(slots-n, bool)]),
                vehicle_id=to(ids), ramp_origin=to(ramp_origin), exit_s_m=553.33)


def native_controller():
    # Exact native executor from INT recipe.py / EVAL run_wm.py, not a new policy.
    from wms.control.counterfactual import native_fn
    from wms.control.continuous import project_native
    from wms.model.cav import WM_MERGE_KEY, WM_LANE_TARGET_KEY
    native = native_fn('head', a_max=1.2)

    def callback(state):
        mask = state['present'].bool() & state['is_cav'].bool()
        state[WM_MERGE_KEY] = mask & (state['lane'] == 2)
        lane = torch.where(state['lane'] != 2, state.get(WM_LANE_TARGET_KEY, state['lane']), state['lane'])
        speed = (state['v_mps'] + .5*project_native(state, native(state))).clamp_min(0.)
        return torch.stack((state['s_m'] + .5*speed, lane, speed,
                            (speed-state['v_mps'])/.5), -1)

    callback.native_lane_mask = lambda s: s['present'].bool() & s['is_cav'].bool()
    return callback


def observable_source(job):
    """Existing WMDATA t0 detector contract (wm_gpu_wmcalib.py:79-115).

    Converted FIT views lack queue arrays. The identity-bound observable start
    records pre-t0 insertions and entrance queues; no assignment/request XML read.
    """
    from wms.model.entries import EntranceQueue
    path = safe(job/'start.pt')
    receipt = json.loads(safe(job/'result.json').read_text())
    sha = digest(path)
    if sha != receipt['output_hashes']['start.pt']:
        raise ValueError('observable start hash differs')
    start = torch.load(path, map_location='cpu', weights_only=False)
    if start['t0'] != 300 or start['initial_state_sha256'] != receipt['metrics']['initial_state_sha256']:
        raise ValueError('observable start clock/identity differs')
    events = start['past_entries']
    if not all(0 < e['time'] <= 300 for e in events):
        raise ValueError('noncausal entrance observations')
    if start['observed_queue_history'][-1][0] > 300:
        raise ValueError('future queue observation')
    q = np.array(start['observed_queue_history'][-1][1:], dtype=float)
    if q.sum() != start['queue'] or q.shape != (2,):
        raise ValueError('queue observation differs')
    entrance = np.array([int(e['lane'] == 2) for e in events])
    inserted = np.bincount(entrance, minlength=2)
    rates = (inserted+q)/300.
    shares = np.array([np.mean([e['is_cav'] for e in events if int(e['lane'] == 2) == a])
                       if inserted[a] else 0. for a in (0, 1)])
    queue = EntranceQueue(q, rates, cav_share=shares)
    return queue, dict(start_sha256=sha, initial_state_sha256=start['initial_state_sha256'],
                       source='observed pre-t0 insertions plus t0 queue, divided by 300s')


@torch.no_grad()
def validate(root, seed, horizon, slots, stream, target_dir, cf_model='legacy', warm_lc=False,
             trace_merge=False, native_model='head'):
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel
    from wms.physics.config import PhysicsConfig
    from wms.model.entries import EntryConfig, TorchEntryProcessV2

    if not 969400 <= seed <= 969431:
        raise ValueError('FIT allowlist is exactly 969400..969431')
    record = json.loads(safe(root / f'world_{seed}.json').read_text())['native']
    ep = safe(root / 'episodes' / record['episode'])
    meta = json.loads((ep / 'episode.json').read_text())
    if meta['seed'] != seed or meta['arm'] != 'native':
        raise ValueError('episode identity differs')
    allowed = ('time_s', 'veh_kin', 'veh_mask', 'vid', 'is_cav', 'counter')
    arrays = {n: np.load(safe(ep / (n+'.npy')), mmap_mode='r', allow_pickle=False) for n in allowed}
    t0 = int(meta['t0_index'])
    dt = float(arrays['time_s'][t0+1]-arrays['time_s'][t0])
    if dt != .5 or float(arrays['time_s'][t0]) != 300.:
        raise ValueError('unexpected FIT clock')
    tape = safe(Path(record['job']) / 'traj_base.npz')
    if digest(tape) != meta['source']['traj_sha256']:
        raise ValueError('observable tape hash differs')
    # Identity->entrance names only. No future kinematics used to generate predictions.
    with np.load(tape, allow_pickle=False) as f:
        names = json.loads(str(f['meta']))['vids']
    k0 = np.array(arrays['veh_kin'][t0], copy=True)
    mask = arrays['veh_mask'][t0].astype(bool) & (k0[:, 1:4].sum(-1) > 0)
    vid = arrays['vid'][t0] if arrays['vid'].ndim == 2 else arrays['vid']
    state = make_state(k0, mask, vid, arrays['is_cav'][t0].astype(bool), names, slots)
    options = dict(merge_model='sumo', sumo_fidelity=True, dt_s=dt)
    if cf_model != 'legacy':
        options.update(cf_model=cf_model, sumo_cf_emergency_decel_mps2=9.)
    cfg = PhysicsConfig(**options)
    # CF wiring may arrive in a later INT commit. Bind/report the actual selector.
    model = WmSModel(ModelSpec(lane_model='lc2013', lane_head='off'), physics=cfg).eval()
    model.requires_grad_(False)
    model.cav_exec_cap = 'idm'
    def reject_learned(module, inputs):
        raise AssertionError('parameterized module executed in population physics diagnostic')
    for module in model.modules():
        if module is not model and any(True for _ in module.parameters(recurse=False)):
            module.register_forward_pre_hook(reject_learned)
    if model.physics.emergency_braking:
        raise ValueError('unexpected extra emergency envelope')
    if warm_lc:
        warm_lc_history(model, arrays, t0, names, state, slots)
    queue, source_receipt = observable_source(safe(record['job']))
    rates = queue.demand_rate
    counter = np.array(arrays['counter'][t0])
    process = TorchEntryProcessV2(counter, dt_s=dt, row_seeds=[seed+stream], couple_draws=True,
        cfg=EntryConfig(conserve_queue=True, insert_rule='sumo_like_v2'), queues=[queue])
    entries = ObservedEntries(process, state)
    initial = int(mask.sum())
    if native_model == 'sumo':
        if cf_model != 'sumo_idm':
            raise ValueError('physics-native validation requires the SUMO CF runtime')
        from wms.model.sumo_native import native_physics_controller
        cb = native_physics_controller(dt_s=dt)
    else:
        cb = native_controller()
    # No encoder inputs: population-only construction; SUMO path bypasses heads.
    clock = np.zeros(2)
    source = np.zeros(2)
    per_vehicle = {}
    merges = []
    births = []
    exits = 0
    max_present = initial
    overlap_samples = 0
    worst_overlap_gap = 0.
    last = state
    history = history_mask = None
    trace = merge_trace(state, cfg, 300.) if trace_merge else []
    start = time.monotonic()
    # One-step rollout keeps the integrated carry/reset interfaces intact. Retain
    # the returned final state, rather than reconstructing and losing LC memory.
    for tick in range(round(horizon/dt)):
        if tick % 40 == 0 and Path('STOP').exists():
            raise RuntimeError('cooperative STOP requested')
        roll = model.rollout(last, n_steps=1, k0=tick, entries=entries, cav_controller=cb,
                             history=history, history_mask=history_mask)
        after = {k: roll[r][0, 0].numpy() for k, r in
                 (('s_m', 'positions'), ('v_mps', 'speeds'), ('lane', 'lanes'), ('present', 'masks'))}
        before = entries.previous
        if any(not np.isfinite(after[k]).all() for k in ('s_m', 'v_mps', 'lane')):
            raise AssertionError('nonfinite physics')
        if roll['raw_residual_mps2'].count_nonzero():
            raise AssertionError('learned correction emitted')
        disappeared = before['present'] & ~after['present']
        if np.any(disappeared & (after['s_m'] <= 553.33)):
            raise AssertionError('vehicle disappeared before road exit')
        exits += int(disappeared.sum())
        active = int(after['present'].sum())
        if initial+int(entries.births.sum())-exits != active:
            raise AssertionError('road population not conserved')
        if any(s.no_slot or s.dropped_pending for s in process.stats):
            raise AssertionError('slot pressure or discarded demand: enlarge capacity in new version')
        max_present = max(max_present, active)
        overlaps, worst_gap = body_overlap_count(after['s_m'], after['lane'],
                                                 entries.length, after['present'])
        overlap_samples += overlaps
        worst_overlap_gap = min(worst_overlap_gap, worst_gap)
        charge = road_delay(before, after, entries.born, dt)
        for e in range(2):
            clock[e] += charge[entries.origin == bool(e)].sum()
        source += entries.source
        for slot in np.flatnonzero(before['present']):
            identity = int(entries.ids[slot])
            per_vehicle[identity] = per_vehicle.get(identity, 0.) + float(charge[slot])
        changed = before['present'] & after['present'] & (before['lane'] == 2) & (after['lane'] == 1)
        merges.extend(dict(identity=int(entries.ids[s]), time_s=300+(tick+1)*dt,
                           position_m=float(after['s_m'][s])) for s in np.flatnonzero(changed))
        births.extend(dict(**b, time_s=300+tick*dt) for b in entries.last_births)
        last = roll['state']
        history, history_mask = roll['history'], roll['history_mask']
        if trace_merge and (tick+1) % round(10/dt) == 0:
            trace.extend(merge_trace(last, cfg, 300+(tick+1)*dt))
    # Read shared recorded outputs only after autonomous prediction is complete.
    def targets(name):
        with safe(target_dir / name).open(newline='') as f:
            return [r for r in csv.DictReader(f) if int(r['seed']) == seed and r['arm'] == 'native']
    delay = [r for r in targets('ramp_delay_world.csv') if float(r['horizon_s']) == horizon]
    if len(delay) != 1:
        raise ValueError('missing/duplicate recorded target clock')
    real_merges = [r for r in targets('ramp_merges.csv') if float(r['t_merge_s']) <= 300+horizon]
    insertions = targets('insertions_10s.csv')
    real_insertions = [sum(int(r['count']) for r in insertions if r['entrance'] == e
                           and float(r['t_start_s']) < 300+horizon) for e in ('mainline', 'ramp')]
    rm = {int(r['vid']): r for r in real_merges}
    pm = {r['identity']: r for r in merges if r['identity'] >= 0}
    common = sorted(rm.keys() & pm.keys())
    predicted = clock+source
    return dict(seed=seed, arm='native', horizon_s=horizon, stream=stream, runtime_s=time.monotonic()-start,
        initial=initial, max_present=max_present, exits=exits,
        cf_model=getattr(cfg, 'cf_model', 'legacy'), lane_model=model.lane_model, native_model=native_model,
        settings='population_default_no_encoder', lc_memory='observable_60s_then_persistent' if warm_lc else 'cold_t0_then_persistent',
        body_overlap=dict(pair_samples=overlap_samples, worst_bumper_gap_m=worst_overlap_gap),
        demand=dict(rate=rates.tolist(), initial_backlog=queue.backlog.tolist(),
                    cav_share=queue.cav_share.tolist(), inserted=entries.births.tolist(),
                    due=[l.due for l in entries.ledgers], pending=process.pending[0].tolist(),
                    conserved=True, no_slot=0, dropped=0, recorded_insertions=real_insertions),
        delay=dict(predicted_mainline_veh_s=float(predicted[0]), predicted_ramp_veh_s=float(predicted[1]),
                   ramp_road_veh_s=float(clock[1]), ramp_source_veh_s=float(source[1]), recorded=delay[0]),
        merges=dict(recorded=len(real_merges), predicted=len(merges), t0_matched=len(common),
                    t0_time_error_s=stats([pm[i]['time_s']-float(rm[i]['t_merge_s']) for i in common]),
                    t0_position_error_m=stats([pm[i]['position_m']-float(rm[i]['s_merge_m']) for i in common]),
                    t0_time_abs_error_s=stats([abs(pm[i]['time_s']-float(rm[i]['t_merge_s'])) for i in common]),
                    t0_position_abs_error_m=stats([abs(pm[i]['position_m']-float(rm[i]['s_merge_m'])) for i in common]),
                    predicted_positions_m=stats([m['position_m'] for m in merges]), events=merges),
        births=births, merge_front_trace=trace, road_delay_per_identity={str(k): v for k, v in per_vehicle.items()},
        episode_sha256=digest(ep/'episode.json'), tape_sha256=digest(tape), source_observations=source_receipt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--runtime', required=True)
    p.add_argument('--targets', required=True)
    p.add_argument('--output')
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--seeds', nargs='+', type=int, required=True)
    p.add_argument('--horizon', type=int, choices=(120, 300, 600), default=300)
    p.add_argument('--slots', type=int, default=320)
    p.add_argument('--stream', type=int, default=190)
    p.add_argument('--cf-model', choices=('legacy', 'sumo_idm'), default='legacy')
    p.add_argument('--warm-lc', action='store_true')
    p.add_argument('--trace-merge', action='store_true')
    p.add_argument('--native-model', choices=('head', 'sumo'), default='head')
    args = p.parse_args()
    if len(args.seeds) != len(set(args.seeds)) or any(not 969400 <= s <= 969431 for s in args.seeds):
        p.error('unique FIT seeds 969400..969431 required')
    runtime = safe(args.runtime)
    sys.path.insert(0, str(runtime/'src'))
    torch.set_num_threads(1)
    torch.manual_seed(965140)
    if args.preflight:
        rows = []
        for seed in args.seeds:
            record = json.loads(safe(Path(args.root)/f'world_{seed}.json').read_text())['native']
            queue, receipt = observable_source(safe(record['job']))
            ep = safe(Path(args.root)/'episodes'/record['episode'])
            for name in ('time_s', 'veh_kin', 'veh_mask', 'vid', 'is_cav', 'counter'):
                np.load(safe(ep/(name+'.npy')), mmap_mode='r', allow_pickle=False)
            rows.append(dict(seed=seed, source=receipt, rate=queue.demand_rate.tolist(),
                             queue=queue.backlog.tolist()))
        print(json.dumps(dict(preflight=True, rows=rows)))
        return
    if not args.output:
        p.error('--output is required for a rollout')
    out = safe(args.output)
    out.mkdir(exist_ok=False)
    paths = list((runtime/'src/wms').rglob('*.py'))
    paths += [runtime/'configs/native_cav_head_v5.json', runtime/'configs/execparity_cav_idm_v1.json']
    sources = {str(path.relative_to(runtime)): digest(path) for path in paths}
    identity = dict(owner='S-RB-MERGE', args=vars(args), sources=sources,
        validator_sha256=digest(__file__), pid=os.getpid(), started_unix=time.time(),
        targets={n: digest(safe(args.targets)/n) for n in ('ramp_delay_world.csv', 'ramp_merges.csv', 'insertions_10s.csv')})
    (out/'IDENTITY.json').write_text(json.dumps(identity, indent=2)+'\n')
    rows = []
    try:
        for seed in args.seeds:
            while hasattr(os, 'getloadavg') and os.getloadavg()[0] > 28:
                time.sleep(15)
            row = validate(safe(args.root), seed, args.horizon, args.slots, args.stream, safe(args.targets),
                           args.cf_model, args.warm_lc, args.trace_merge, args.native_model)
            if any(digest(runtime/name) != sha for name, sha in sources.items()):
                raise AssertionError('runtime changed during diagnostic')
            (out/f'{seed}.json').write_text(json.dumps(row, indent=2, allow_nan=False)+'\n')
            rows.append(row)
            print('DONE', seed, row['runtime_s'], row['delay']['predicted_ramp_veh_s'], flush=True)
        report = dict(rows=rows,
            recorded_ramp_veh_s=stats([float(r['delay']['recorded']['ramp_delay_vs']) for r in rows]),
            predicted_ramp_veh_s=stats([r['delay']['predicted_ramp_veh_s'] for r in rows]),
            limitations=['One population-default future, native arm only; not distributional qualification.',
                'Historical 562.4 vs 1265.4 is a different panel; recorded per-world clocks primary.',
                'Future arrival identities are synthetic; exact timing matches only t0 cohort.',
                'LC2013 memory is inferred only from observable pre-t0 traffic when --warm-lc is selected.',
                'CF selector reported per row; this is a merge/LC diagnostic, not final WM-PHYS.'])
        (out/'COMPLETE.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    except BaseException as exc:
        (out/'FAILED.json').write_text(json.dumps(dict(error=repr(exc), completed=[r['seed'] for r in rows]))+'\n')
        raise


if __name__ == '__main__':
    main()
