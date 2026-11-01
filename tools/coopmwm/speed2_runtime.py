"""Opt-in masked-prefix work reduction; full source and controller rosters survive.

Only never-used trailing slots are omitted from stateless geometry/model work.
The source's existing host insertion ledger supplies the high-water mark: no
extra device read, random draw, or changed admission capacity is necessary.
"""
from contextlib import contextmanager
from dataclasses import fields
import time
from unittest.mock import patch

import torch

from .fused import crop, restore


def trajectory_errors(expected, actual):
    """Compare every real cohort, including the movement that removes an exit."""
    if len(expected) != len(actual):
        raise ValueError('trajectory chunk count mismatch')
    errors = {key: 0. for key in ('positions','speeds','lanes','masks','accel')}
    for a,b in zip(expected,actual):
        if a['masks'].shape != b['masks'].shape:
            raise ValueError('trajectory roster/timeline mismatch')
        errors['masks'] = max(errors['masks'], float(not torch.equal(a['masks'],b['masks'])))
        mask = a['masks'].bool() | b['masks'].bool()
        mask = mask | torch.nn.functional.pad(mask[:,:-1],(0,0,1,0))
        for key in errors:
            if key != 'masks' and mask.any():
                errors[key] = max(errors[key],float((a[key]-b[key]).abs()[mask].max()))
    return errors


def qualify(rt, record, actions, K, H, width, memory=None):
    """Old full-roster vs optimized, once per newly bound persistent worker."""
    from .plan_wm import evaluate
    tapes = []
    original = rt.frozen.costs
    def capture(roll, entries, initial):
        tapes.append({k:roll[k].detach().cpu().clone() for k in
                      ('positions','speeds','lanes','masks','accel')})
        return original(roll, entries, initial)
    rt.frozen.costs = capture
    device = next(rt.model.parameters()).device
    def sync():
        if device.type == 'cuda':torch.cuda.synchronize(device)
    try:
        sync();start=time.perf_counter()
        old = evaluate(rt,record,actions,K,H,chunk_actions=width,trim=False,pad_actions=width,memory=memory)
        sync();old_wall=time.perf_counter()-start
        expected = tapes[:]
        tapes.clear()
        sync();start=time.perf_counter()
        new = evaluate(rt,record,actions,K,H,chunk_actions=width,trim=False,pad_actions=width,speed2=True,memory=memory)
        sync();new_wall=time.perf_counter()-start
        errors = trajectory_errors(expected,tapes)
    finally:
        rt.frozen.costs = original
    cost = max(abs(a['team_delay']-b['team_delay']) for key in old for a,b in zip(old[key],new[key]))
    draw_ids = all(a['draw_id']==b['draw_id'] for key in old for a,b in zip(old[key],new[key]))
    rank = lambda values: sorted(values,key=lambda a:(sum(v['team_delay'] for v in values[a]),a))
    ranking = rank(old)==rank(new)
    steps=K*width*((len(actions)+width-1)//width)*2*H
    return dict(pass_=cost<=.001 and draw_ids and ranking and max(errors.values())<=.001,
                max_cost_error=cost,trajectory_errors=errors,identical_candidate_ranking=ranking,
                identical_draw_ids=draw_ids,K=K,H=H,width=width,device=str(device),
                reference_wall_s=old_wall,optimized_wall_s=new_wall,speedup=old_wall/new_wall,
                reference_future_steps_per_s=steps/old_wall,optimized_future_steps_per_s=steps/new_wall,
                timing_includes_trajectory_capture=True,new_sumo_sim_s=0)


def bind_ledger(proxy, bind, restores=None):
    def tracked(rt, initial, kwargs, plan):
        result = bind(rt, initial, kwargs, plan)
        entries = result[1]
        process = entries.process
        original = process._commit
        if restores is not None:
            restores.append((process, original))

        def commit(state, work, new_mask):
            # The producer chooses these exact (row, slot) pairs on the host.
            # Retain exited/used slots too, so memories never change identity.
            if work.inserted:
                proxy.live_slots = max(proxy.live_slots, 1 + max(k for _, k in work.inserted))
            return original(state, work, new_mask)

        process._commit = commit
        return result
    return tracked


@contextmanager
def ledger(proxy):
    from tools.wmteach import marginal
    restores = []
    try:
        with patch.object(marginal, 'bind', bind_ledger(proxy, marginal.bind, restores)):
            yield
    finally:
        for process, original in restores:
            process._commit = original


@contextmanager
def compact_geometry(proxy):
    """Keep recurrent bridge tensors full-sized, compact only pure geometry.

    Network batch/slot dimensions, RSU reductions and recurrent state are kept
    exactly as inherited. There is still one batched actor forward for all rows.
    """
    from wms.control import bridge, lanes, safety, alwayscmd
    original_obs = bridge.build_observations
    original_feasible = lanes.lane_feasibility
    original_choices = lanes.apply_lane_choices
    original_safe = safety.safe_accel
    original_cap = alwayscmd.inferred_cap

    def geometry(state):
        capacity = state['present'].shape[1]
        n = min(proxy.live_slots, capacity)
        return crop(state, n, capacity), n, capacity

    def observations(state, report_age_s=None):
        compact, n, capacity = geometry(state)
        if report_age_s is not None:
            raise ValueError('speed2 qualified only for inherited zero-age reports')
        obs = original_obs(compact)
        return type(obs)(**{f.name: restore(getattr(obs, f.name), capacity) for f in fields(obs)})

    def feasible(state, ages, rules):
        compact, n, capacity = geometry(state)
        return restore(original_feasible(compact, ages[:, :n], rules), capacity)

    def choices(state, choice, lane_mask, rules, acting=None):
        compact, n, capacity = geometry(state)
        result = original_choices(compact, choice[:, :n], lane_mask[:, :n], rules,
                                  acting=None if acting is None else acting[:, :n])
        return (restore(result[0], capacity, state['lane']), restore(result[1], capacity))

    def safe(state, accel, **kw):
        compact, n, capacity = geometry(state)
        kw = {k: v[:, :n] if torch.is_tensor(v) and v.ndim == 2 else v for k, v in kw.items()}
        result = original_safe(compact, accel[:, :n], **kw)
        # HDVs are preserved by safety, including absent HDV padding.
        tail = torch.where(state['is_cav'], torch.zeros_like(accel), accel)
        return restore(result, capacity, tail)

    def cap(state):
        compact, n, capacity = geometry(state)
        return restore(original_cap(compact), capacity)

    with patch.object(bridge, 'build_observations', observations), \
         patch.object(lanes, 'lane_feasibility', feasible), \
         patch.object(lanes, 'apply_lane_choices', choices), \
         patch.object(safety, 'safe_accel', safe), \
         patch.object(alwayscmd, 'inferred_cap', cap):
        yield
