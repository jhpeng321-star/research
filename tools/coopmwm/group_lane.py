"""G5 observable joint lane macros, default OFF (``group_lane_moves=True``).

Reuses c4/lane-plans' decision_envelope (lane_feasibility/LaneRules) and
c4/lane-policy's lanes_tensor/lane_outcome + controller.lane_request ABI
(integration e5740395). No lane head, WM checkpoint or new driving threshold.
Only the existing env.step actuator issues changeLane, with LC_MODE_POLICY=513.

Members are ordered (mainline yielder, ramp CAV). yield_lane attempts 1->0 at
onset, with no acceleration correction. open_gap_merge eases the mainline CAV
for the registered hold, and attempts ramp 2->1 one control tick after onset. Attempts
are never retried/forced; changed geometry, missing coverage or safety refusal
means no lane command. The global look-ahead is untouched.
"""
import math

import numpy as np
import torch

from . import moves
from .merge_features import CONTROL_DT_S, SENSOR_RANGE_M

FAMILIES = ('yield_lane', 'open_gap_merge')
PROVENANCE = 'tracks_spec.md MAIN 00:30 owner G5 LANE-CHANGE GROUP MOVES'
OUTCOME_KEYS = ('issued', 'noop', 'refused_junction', 'refused_not_allowed',
                'refused_no_lane', 'deferred_edge_end', 'refused_not_policy')


def enabled(cfg):
    value = (cfg or {}).get('group_lane_moves', False)
    if type(value) is not bool:
        raise ValueError('group_lane_moves must be bool')
    return value


def families(cfg):
    return moves.FAMILIES + (FAMILIES if enabled(cfg) else ())


def envelope(state, member):
    """Lane-plans' envelope, restricted to an explicitly covered local packet.

    Unlike an offline full-scene envelope, absent coverage is NOT an empty lane.
    All sensed HDVs remain gap geometry, never actors. Malformed packets fail
    closed. Fresh sensing only; partner traffic outside this packet is not used.
    """
    rows = (state or {}).get('rows') or []
    by = {r['vid']: r for r in rows}
    ego = by.get((state or {}).get('self'))
    actor = by.get(member)
    if ego is None or actor is None or not actor.get('is_cav') or len(by) != len(rows):
        return None
    lane = actor['lane']
    if lane not in (1, 2) or lane-1 not in state.get('observed_lanes', ()):
        return None
    if abs(float(actor['s_m'])-float(ego['s_m'])) > SENSOR_RANGE_M:
        return None
    # No extra/global rows, NaNs, fractional lanes or unknown lengths accepted.
    if any(type(r.get('is_cav')) is not bool or r.get('lane') not in (0, 1, 2)
           or any(not isinstance(r.get(k), (int, float)) or isinstance(r[k], bool)
                  or not math.isfinite(r[k]) for k in ('s_m', 'v_mps', 'length_m'))
           or r['v_mps'] < 0 or r['length_m'] <= 0
           or abs(r['s_m']-ego['s_m']) > SENSOR_RANGE_M for r in rows):
        return None
    from wms.control.lanes import LaneRules, lane_feasibility
    packet = {k: torch.tensor([[r[k] for r in rows]])
              for k in ('s_m', 'v_mps', 'lane', 'is_cav', 'length_m')}
    packet['present'] = torch.ones((1, len(rows)), dtype=torch.bool)
    feasible = lane_feasibility(packet, None, LaneRules())[0, rows.index(actor)]
    return dict(lane=int(lane), left=bool(feasible[1]), right=bool(feasible[2]))


def candidate(family, state, own, partners, rng):
    """Two currently acquirable CAVs; local target-lane coverage is mandatory."""
    if family not in FAMILIES:
        raise ValueError('unknown G5 family')
    rows = [own]+list(partners)
    ramps = [r for r in rows if int(r['lane']) == moves._RAMP_LANE
             and moves._ZONE_S0 <= r['s_m'] <= moves._ZONE_S1]
    mains = [r for r in rows if int(r['lane']) == moves._MAIN_LANE
             and moves._ZONE_S0 <= r['s_m'] <= moves._ZONE_S1]
    link = float(moves.VARIETY_REGISTERED['variety_gap_link_m'])
    options = []
    for main in mains:
        for ramp in ramps:
            if own['vid'] not in (main['vid'], ramp['vid']) or abs(main['s_m']-ramp['s_m']) > link:
                continue
            actor = main if family == 'yield_lane' else ramp
            e = envelope(state, actor['vid'])
            # The gap-opening yielder is behind the merger; deceleration opens
            # space in front of it. Gap need not already be safe at proposal time.
            if e is not None and ((family == 'yield_lane' and e['left']) or
                    (family == 'open_gap_merge' and main['s_m'] < ramp['s_m'])):
                options.append((main['vid'], ramp['vid']))
    if not options:
        return None
    members = options[int(rng.integers(len(options)))]
    return moves.Move(family, members, *moves._sampled(rng))


def targets(move, elapsed):
    """Absolute WM targets; at most ONE attempt/member, like lane_plans.at."""
    if move.family == 'yield_lane' and elapsed == move.onset:
        return {move.members[0]: 0}
    if move.family == 'open_gap_merge' and elapsed == move.onset+int(CONTROL_DT_S):
        return {move.members[1]: 1}
    return {}


def safe_targets(requests, local_states, eligible_cavs):
    """Execution-time recheck using each commanded CAV's OWN sensed gaps."""
    out, blocked = {}, {}
    for vid, target in requests.items():
        e = envelope(local_states.get(vid), vid) if vid in eligible_cavs else None
        if e is None:
            blocked[vid] = 'missing_observable_gap_or_cav'
        elif type(target) is not int or target not in (0, 1) or e['lane']-1 != target:
            blocked[vid] = 'source_lane_changed'
        elif not e['left']:
            blocked[vid] = 'unsafe_gap_or_edge'
        else:
            out[vid] = target
    return out, blocked


def lanes_tensor(targets_by_id, env):
    """Reused lane-policy tensor contract; -1 slots remain native."""
    if not targets_by_id:
        return None
    ids = env.slot_vehicle_ids()
    tensor = torch.full((1, len(ids)), -1., dtype=torch.float32)
    placed = False
    for slot, vid in enumerate(ids):
        if vid in targets_by_id:
            tensor[0, slot] = float(targets_by_id[vid])
            placed = True
    return tensor if placed else None


def lane_outcome(counts):
    """Reused lane-policy tick aggregate; issued is never claimed executed."""
    counts = dict(counts or {})
    requested = sum(int(counts.get(k, 0)) for k in OUTCOME_KEYS)
    issued = int(counts.get('issued', 0))
    return dict(requested=requested, issued=issued, blocked=requested-issued,
                executed=int(counts.get('completed', 0)), counts=counts)


def controller_targets(controller, policy, rows, now):
    """Committed plans supersede samples; counterfactual branch release applies.

    No separate TraCI writer: the returned tensor enters the same env.step as
    acceleration. Replayed samples also recheck live local safety.
    """
    if not controller.events:
        return {}
    sample = controller.events[-1]['sample']
    committed_lane = any(e['plan'].get('g5_move') for e in controller.active_plans)
    # Original one-second samples are also explicit instructions on a CRN
    # prefix replay, whose native verifier carries only the frozen base policy.
    if not enabled(getattr(policy, 'cfg', None)) and not committed_lane and not sample.get('g5_lane_requests'):
        return {}
    requests = dict(sample.get('g5_lane_requests') or {})
    for entry in controller.active_plans:
        plan = entry['plan']
        if plan.get('g5_move'):
            move = from_plan(plan)
            for vid in move.members:
                requests.pop(vid, None)
            requests.update(targets(move, now-entry['start_s']))
    branch = controller.branch
    if branch and now >= branch['time_s']:
        release = branch.get('group', ()) if now >= branch.get('until', math.inf) else branch['revert']
        for vid in release:
            requests.pop(vid, None)
        if branch.get('mappo_background'):
            allowed = set(branch['group'])-set(branch['revert']) if now < branch['until'] else set()
            requests = {v: t for v, t in requests.items() if v in allowed}
    eligible = {r['vid'] for r in rows if r.get('is_cav')}
    local = getattr(policy, 'local_states', None)
    if local is None:
        # Native all8/VerifyController uses the frozen base policy. A bound G5
        # plan is its explicit opt-in; reconstruct only each actor's own sensing
        # with the same whitelist/radius as EpisodePolicy, never the full scene.
        local = {}
        for actor in rows:
            if actor['vid'] in requests and actor.get('is_cav'):
                sensed = [{k: r[k] for k in ('vid', 's_m', 'v_mps', 'lane', 'is_cav', 'length_m')}
                          for r in rows if abs(r['s_m']-actor['s_m']) <= SENSOR_RANGE_M]
                local[actor['vid']] = dict(self=actor['vid'], rows=sensed, observed_lanes=(0, 1, 2))
    safe, blocked = safe_targets(requests, local, eligible)
    if requests or 'g5_lane_requests' in sample:
        sample['g5_lane_requests'] = requests
        sample['g5_lane_gate'] = dict(safe=safe, blocked=blocked)
    return safe


def plan_fields(move, duration):
    """Canonical lane-plans compatible fields for mining/verification producers."""
    if move.family not in FAMILIES or type(duration) is not int or duration < move.onset+move.hold:
        raise ValueError('G5 plan must cover the full registered macro')
    actions = [[int(v in targets(move, t)) for t in range(duration)] for v in move.members]
    return dict(g5_move=move.to_dict(), lane_actions=actions, uses_lane_actions=True)


def from_plan(plan):
    """Exact executable macro binding; never project away a lane command."""
    if not isinstance(plan.get('g5_move'), dict):
        raise ValueError('lane-bearing plan requires an explicit executable g5_move')
    move = moves.Move.from_dict(plan.get('g5_move') or {})
    if move.family not in FAMILIES or list(move.members) != plan.get('members'):
        raise ValueError('G5 plan member/role binding mismatch')
    expected = plan_fields(move, plan['duration_s'])
    if plan.get('lane_actions') != expected['lane_actions'] or plan.get('uses_lane_actions') is not True:
        raise ValueError('G5 plan lane actions do not match executable macro')
    members, offsets = moves._plan_offsets(plan)
    wanted = np.array([[moves.offsets(move, t)[v] for t in range(plan['duration_s'])] for v in members])
    if not np.allclose(offsets, wanted, atol=1e-8, rtol=0):
        raise ValueError('G5 plan offsets do not match executable macro')
    return move


def verify_tape(plan, tape, t0):
    """Admission requires the macro's actual lane effect on the hash-bound tape.

    A positive acceleration-only verification is insufficient. Tick outcomes
    are actuator telemetry, and the actor's next sensed lane must confirm the
    requested change. Blocked/failed attempts stay diagnostics, never lessons.
    """
    move = from_plan(plan)
    elapsed = move.onset if move.family == 'yield_lane' else move.onset+int(CONTROL_DT_S)
    expected = targets(move, elapsed)
    steps = {int(s['time_s']): s for s in tape['steps']}
    before, after = steps.get(int(t0)+elapsed), steps.get(int(t0)+elapsed+1)
    if before is None or after is None:
        raise ValueError('G5 verified tape missing lane-effect transition')
    request = before['sample'].get('g5_lane_requests') or {}
    outcome = before.get('g5_lane_outcome') or {}
    pre = {r['vid']: r for r in before.get('rows', ())}
    post = {r['vid']: r for r in after.get('rows', ())}
    for vid, target in expected.items():
        if (request.get(vid) != target or outcome.get('issued', 0) < 1 or outcome.get('executed', 0) < 1
                or not pre.get(vid, {}).get('is_cav') or not post.get(vid, {}).get('is_cav')
                or pre[vid]['lane'] != target+1 or post[vid]['lane'] != target):
            raise ValueError('G5 verified tape lacks the actual commanded CAV lane effect')
