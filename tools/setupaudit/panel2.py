"""WM-only physical-J panel: closed-loop crossed controllers, paired K8 futures.

No trainer, optimizer, simulator, future observed trajectory or truth labels are used.
The scoring algebra is objective14's accrued cost, not the weighted training credit.
"""
from __future__ import annotations

import argparse
import copy
import gc
from dataclasses import replace
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from inspect_inputs import allowed
from measure import WM, MAPPO, digest, parity_motion, native_background_motion
from wms.control.bridge import make_cav_controller
from wms.control.contract import MERGE_S1_M, SPEED_LIMIT_MPS
from wms.control.counterfactual import native_fn
from wms.control.dreamer import DreamerConfig, _entry_process, with_entry_slots
from wms.control.entrant_mix import make_sampler
from wms.control.futures import FuturesConfig, SampledFutures, lane_uniform_draws
from wms.control.mappo.eval_bridge import load_policy_fn
from wms.control.nativeinit import pair_merge_streams
from wms.control.policy import load_policy_components
from wms.control.policy14_cli import load_members
from wms.control.safety import safe_accel
from wms.control.train_cli import GroupSignal
from wms.physics.batched import road_end_exit_bound

ACTORS = ['CTRL0', 'MAPPO', 'MAPPO_native_lanes', 'CTRL0_MAPPO_lanes', 'native',
          'MAPPO_CTRL0_admissions']


def expand(value, n):
    if torch.is_tensor(value):
        return value.expand(n, *value.shape[1:]).clone()
    if isinstance(value, dict):
        return {k: expand(v, n) for k, v in value.items()}
    return value


class Entries:
    """Record input cohorts/queues; optional CTRL0-conditioned admission intervention.

    Frozen means exogenous CTRL0 births, insertion states and source backlogs,
    not zero arrivals or free source waiting. Virgin entrant slots cannot be reused.
    """
    def __init__(self, process, frozen=False):
        self.process, self.frozen = process, frozen
        self.states, self.queues, self.pending = [], [], []
        self.new_masks, self.real_source = [], []
        if hasattr(process, '_headway'):
            original = process._headway

            def headway(b, e):
                # Original _emit_due advances a renewal at its exact continuous due
                # time. Observe that time before +=headway, without another RNG draw.
                if process._started[b]:
                    self.due_area[b] += max(0., process.now[b]-process.t_next[b, e])
                return original(b, e)

            process._headway = headway

    def insert(self, state):
        n = state['present'].shape[0]
        pending_before = self.process.pending.sum(-1).copy()
        self.due_area = np.zeros(n)
        if self.frozen or len(ACTORS) == n:
            shadow = {k: v.clone() if torch.is_tensor(v) else v for k, v in state.items()}
            for k, v in shadow.items():
                if torch.is_tensor(v) and v.ndim and v.shape[0] == n:
                    if self.frozen:
                        shadow[k] = expand(v[:1], n)
                    else:
                        shadow[k][-1] = v[0]
            mask = self.process.insert(shadow)
            for k in ('s_m', 'v_mps', 'a_mps2', 'lane', 'present', 'is_cav', 'length_m', 'entrant'):
                if k in shadow:
                    state[k] = torch.where(mask, shadow[k], state.get(k, torch.zeros_like(shadow[k])))
        else:
            mask = self.process.insert(state)
        self.states.append({k: state[k].clone() for k in
                            ('present', 'is_cav', 'lane', 's_m', 'v_mps', 'entrant') if k in state})
        self.queues.append(self.process.backlog().copy())
        self.pending.append(self.process.pending.copy())
        self.new_masks.append(mask.clone())
        self.real_source.append(.5*pending_before+self.due_area)
        return mask


def controller(candidate, mappo, native):
    """Use separate stateful bridges, selecting rows before the WM executes motion."""
    cp, ce, _ = candidate
    current = {}

    def deterministic(policy):
        def call(obs, signal, **kw):
            return policy(obs, signal, deterministic=True, **kw)
        call.accel_max = policy.accel_max
        call.residual_on_native = bool(getattr(policy, 'residual_on_native', False))
        return call

    common = dict(native=native, native_accel_max=1.2, symmetric_exec_cap=True,
                  mask_forced_merge=True, native_ramp_merge='wm_law', native_lane_change=True)
    ctrl = make_cav_controller(deterministic(cp), GroupSignal(ce), safe_accel, **common)
    bridges = []
    previous = None

    def callback(state):
        nonlocal previous
        from wms.control.alwayscmd import executed, inferred_cap
        from wms.model.cav import EXEC_CAP_KEY, WM_MERGE_KEY
        def rows(indices):
            return {k: v[indices] if torch.is_tensor(v) and v.ndim
                    and v.shape[0] == len(ACTORS) else v for k, v in state.items()}
        if not bridges:
            for indices, native_lanes in (([1, 3, 4, 5], False), ([2], True)):
                act = torch.ones_like(state['present'][indices])
                if not native_lanes:
                    act[2] = False
                bridges.append(make_cav_controller(deterministic(mappo.policy),
                    GroupSignal(mappo.group_encoder), safe_accel,
                    lane_rules=mappo.lane_rules, act_mask=act,
                    policy_ramp_merge=True, policy_native_lanes=native_lanes, **common))
        cstate = rows([0])
        cc = ctrl(cstate)
        commands, merges = [], []
        for indices, bridge in zip(([1, 3, 4, 5], [2]), bridges):
            ms = rows(indices)
            ms[EXEC_CAP_KEY] = inferred_cap(ms)
            # Hybrid lane controller observes the longitudinal commands actually executed.
            if previous is not None:
                ext = torch.zeros_like(ms['present'])
                if len(indices) == 4:
                    ext[1] = True
                ms.update(external_command_mask=ext, cmd_accel_mps2=previous[indices],
                          lane_intent=current['intent'][indices])
            mc = bridge(ms)
            applied = torch.maximum(-ms['v_mps'] / .5,
                                    torch.minimum(mc[..., 3], ms[EXEC_CAP_KEY]))
            mc = parity_motion(ms, mc[..., 1], applied)
            if len(indices) == 4:
                # Preserve the original WM-native cap, not the exact cap injected
                # exclusively into the absolute-actor bridge (measure.py:239).
                ns = rows([4])
                mc[2] = native_background_motion(ns, mc[2:3, :, 1], native)[0]
            commands.append(mc)
            merges.append(ms[WM_MERGE_KEY])
        hs = rows([3])
        hybrid = parity_motion(hs, commands[0][1:2, :, 1],
                               executed(hs, torch.zeros_like(hs['v_mps'])))[0]
        out = torch.stack((cc[0], commands[0][0], commands[1][0], hybrid, commands[0][2],
                           commands[0][3]))
        state[WM_MERGE_KEY] = torch.stack((cstate[WM_MERGE_KEY][0], merges[0][0],
                                          merges[1][0], merges[0][1], merges[0][2], merges[0][3]))
        current['intent'] = (out[..., 1] - state['lane']).clamp(-1, 1)
        previous = out[..., 3]
        mask = state['present'].bool() & state['is_cav'].bool()
        callback.requests.append(((out[..., 1] != state['lane']) & mask).sum(-1))
        tick = int(state.get('physics_step_index', 0))
        if tick and tick % 200 == 0:
            print('TICK', tick, flush=True)
        return out

    def native_lanes(state):
        mask = state['present'].bool() & state['is_cav'].bool() & (state['lane'] >= 0)
        which = torch.tensor([True, False, True, False, True, False], device=mask.device)
        return mask & which[:, None]

    callback.native_lane_mask = native_lanes
    callback.requests = []
    return callback


def costs(roll, entries, initial):
    """Exactly reward.py's current cohort / next end-step speed convention."""
    present = torch.stack([s['present'] for s in entries.states], 1).bool()
    cav = torch.stack([s['is_cav'] for s in entries.states], 1).bool()
    lane = torch.stack([s['lane'] for s in entries.states], 1)
    pos = torch.stack([s['s_m'] for s in entries.states], 1)
    road = .5 * torch.where(present, (1 - roll['speeds'] / SPEED_LIMIT_MPS).clamp_min(0), 0)
    source = .5 * torch.as_tensor(np.stack(entries.queues, 1), device=road.device, dtype=road.dtype)
    born = torch.arange(road.shape[-1], device=road.device) >= initial
    merge = (pos >= 0) & (pos <= MERGE_S1_M)
    ramp = (lane == 2) & ~merge
    columns = {'road': road.sum(-1), 'source': source,
               'HDV': (road * ~cav).sum(-1), 'CAV': (road * cav).sum(-1),
               'entrant': (road * born).sum(-1), 'initial': (road * ~born).sum(-1),
               'merge_zone': (road * merge).sum(-1), 'ramp': (road * ramp).sum(-1),
               'mainline': (road * ~(merge | ramp)).sum(-1)}
    columns['J'] = columns['road'] + source
    # Recompute BranchTracker/RequestLedger's *reported-boundary* convention on
    # these same WM trajectories: no road charge on the birth tick; exiting
    # vehicles carry their prior observed speed; source includes the insertion
    # tick and fractional waiting for requests due within it.
    new = torch.stack(entries.new_masks, 1)
    before_v = torch.stack([s['v_mps'] for s in entries.states], 1)
    real_v = torch.where(roll['masks'].bool(), roll['speeds'], before_v)
    real_road = .5*torch.where(present & ~new, (1-real_v/SPEED_LIMIT_MPS).clamp_min(0), 0).sum(-1)
    real_source = torch.as_tensor(np.stack(entries.real_source, 1), device=road.device, dtype=torch.float64)
    columns.update(real_road=real_road, real_source=real_source, real_J=real_road+real_source,
                   entrant_total=columns['entrant']+source)
    by_source = .5 * torch.as_tensor(np.stack(entries.pending, 1), device=road.device,
                                     dtype=road.dtype)
    columns.update(source_mainline=by_source[..., 0], source_ramp=by_source[..., 1],
                   stopped_ramp_seconds=.5*(present & (lane == 2) & (roll['speeds'] < .1)).sum(-1))
    # Qualification's different metric, explicitly kept out of the real-team comparison.
    t0_hdv = entries.states[0]['present'] & ~entries.states[0]['is_cav'] & ~born
    columns['t0_HDV_road'] = (road * t0_hdv[:, None]).sum(-1)
    result = {}
    for h in (90, 120, 300, 600):
        if road.shape[1] >= 2 * h:
            result[str(h)] = {k: v[:, :2*h].double().sum(1).cpu().tolist() for k, v in columns.items()}
    result['time_bins_30s'] = {k: torch.stack([v[:, t:t+60].double().sum(1)
                                 for t in range(0, road.shape[1], 60)], 1).cpu().tolist()
                              for k, v in columns.items()}
    result['terminal_queue_by_source'] = entries.pending[-1].tolist()
    result['births'] = [int((present[i].any(0) & born).sum()) for i in range(present.shape[0])]
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--starts', default='../starts24.pt')
    p.add_argument('--candidate', default='../policy_000190.pt')
    p.add_argument('--wm', default='/root/autodl-tmp/c4/wmcombined_wm2/wm_s.pt')
    p.add_argument('--wm-config', default='/root/autodl-tmp/c4/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml')
    p.add_argument('--mappo', default='/root/autodl-tmp/c4/mappores20261028/inputs/mappo_v5_final.pt')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--horizon', type=int, default=600)
    p.add_argument('--n-starts', type=int, default=24)
    p.add_argument('--futures', type=int, default=8)
    p.add_argument('--start-index', type=int, default=0)
    p.add_argument('--freeze', action='store_true')
    p.add_argument('--reuse')
    a = p.parse_args()
    assert all(allowed(x) for x in vars(a).values())
    assert digest(a.wm) == WM and digest(a.mappo) == MAPPO
    out = Path(a.out)
    assert not out.exists() and out.parent.is_dir()
    torch.set_num_threads(2)
    if a.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.85)
    torch.manual_seed(965140)
    models, receipt = load_members([a.wm], [a.wm_config], a.device)
    model = models[0]
    payload = torch.load(a.candidate, map_location='cpu', weights_only=False)
    cfg = replace(DreamerConfig(**payload['config']), horizon=a.horizon,
                  entrant_learned_mix=True)
    cp, ce, _, _ = load_policy_components(a.candidate, device=a.device)
    torch.nn.init.zeros_(cp.residual_head.weight)
    torch.nn.init.zeros_(cp.residual_head.bias)
    cp.eval()
    ce = ce.encoder if isinstance(ce, GroupSignal) else ce
    ce.eval()
    mp = load_policy_fn(a.mappo, device=a.device)
    mp.policy.eval()
    mp.group_encoder.eval()
    native = native_fn('head', a_max=1.2)
    model.cav_exec_cap = cfg.cav_exec_cap
    batches = torch.load(a.starts, map_location=a.device, weights_only=False)
    assert all(allowed(b.metadata) for b in batches)
    results = dict(schema='setupaudit2/1', pid=os.getpid(), gpu_fraction=.85,
                   actors=ACTORS, wm=receipt, starts_sha256=digest(a.starts),
                   candidate_sha256=digest(a.candidate), mappo_sha256=digest(a.mappo),
                   horizon=a.horizon, K=a.futures, mix=True, frozen=a.freeze, rows=[])
    import inspect
    import wms.model.entries as entries_module
    results['source_sha256'] = {name: digest(path) for name, path in
        [('panel2', __file__), ('bridge', inspect.getfile(make_cav_controller)),
         ('dreamer', inspect.getfile(_entry_process)), ('entries', entries_module.__file__)]}
    if a.reuse:
        old = json.loads(Path(a.reuse).read_text())
        for key in ('actors', 'wm', 'starts_sha256', 'candidate_sha256', 'mappo_sha256',
                    'horizon', 'K', 'mix', 'frozen', 'source_sha256'):
            assert results[key] == old[key], key
        results['rows'] = old['rows']
        results['prior_owned_pids'] = old.get('prior_owned_pids', []) + [old['pid']]
    print('OWN_PID', os.getpid(), flush=True)
    with torch.no_grad():
        tasks = [(bi, fi) for fi in range(a.futures) for bi in
                 range(a.start_index, min(len(batches), a.start_index+a.n_starts))]
        for bi, selected_future in tasks:
            b = copy.deepcopy(batches[bi])
            if cfg.entrant_control and b.entry_feat_control is not None:
                b = replace(b, entry_feat=b.entry_feat_control)
            b = with_entry_slots(b, a.horizon, max(960, int(3.2*a.horizon)),
                                 torch.Generator().manual_seed(1_000_003*191), queue_state=True)
            plan = SampledFutures(FuturesConfig.from_trainer(cfg), 190).plan(
                models, b.state, b.rollout_kwargs, a.horizon*2)[0]
            for fi in [selected_future]:
                if any(r['start'] == bi and r['future'] == fi for r in results['rows']):
                    continue
                begin = time.perf_counter()
                torch.manual_seed(965140)
                initial = expand(b.state, len(ACTORS))
                pair_merge_streams(initial, torch.zeros(len(ACTORS), dtype=torch.long,
                                                       device=a.device), 1, 190+fi)
                initial['exit_s_m'] = torch.full_like(initial['s_m'], road_end_exit_bound(model.physics))
                kw = expand({**b.rollout_kwargs, **plan[fi]}, len(ACTORS))
                lanes = lane_uniform_draws(models, 1, a.horizon*2, initial['s_m'].shape[1],
                                           190+1000003*fi if fi else 190, device=a.device)
                if lanes is not None:
                    kw['lane_uniform'] = expand(lanes, len(ACTORS))
                learned = None
                if model.physics.entry_learned:
                    from wms.model.entry_learned import load_learned_entry
                    learned = load_learned_entry()
                process = _entry_process(b.entry_spec, [0]*len(ACTORS), initial['s_m'].shape[1],
                    cfg.entry_insert_rule, learned, couple_draws=True, sample_index=fi)
                entries = Entries(process, a.freeze)
                kw['entries'] = entries
                kw['entrant_driver_sampler'] = make_sampler(model, initial, kw, 190+1000003*fi,
                                                            [0]*len(ACTORS))
                print('BEGIN', bi, fi, b.metadata, 'slots', initial['s_m'].shape, flush=True)
                cb = controller((cp, ce, None), mp, native)
                roll = model.rollout(initial, n_steps=a.horizon*2, cav_controller=cb, **kw)
                row = dict(start=bi, future=fi, metadata=b.metadata[0],
                           slots=initial['s_m'].shape[-1], seconds=time.perf_counter()-begin,
                           mix_assigned=kw['entrant_driver_sampler'].assigned,
                           lane_requests=torch.stack(cb.requests).sum(0).cpu().tolist(),
                           costs=costs(roll, entries, b.entry_spec['k0']))
                results['rows'].append(row)
                out.write_text(json.dumps(results, indent=2)+'\n')
                print('DONE', bi, fi, row['seconds'], row['costs'].get(str(a.horizon), {}), flush=True)
                # The due-time observer closes over the entry process. Release
                # that cycle explicitly between futures, including GPU cohorts.
                del roll, entries, kw, process, cb
                gc.collect()
    print('COMPLETE', a.out, flush=True)


if __name__ == '__main__':
    main()
