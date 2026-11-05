"""Versioned role schedule shared by WM and a future native mining executor.

This is deliberately NOT canonical G5 yield_lane: yield_then_merge has TWO lane
actors. No simulator is imported. MAIN must bind this exact record in its native
executor rather than feed it to moves.from_plan as a canonical G5 label.
"""
import copy
import math

SCHEMA = 'mining-role-lane-schedule-v1'
FAMILIES = ('yield_then_merge', 'open_gap_merge', 'g5_yield_lane')


def make_schedule(family, mainline, ramp, onset=0, hold=20, depth=.3,
                  *, ramp_offset=0.):
    if family not in FAMILIES:
        raise ValueError('unknown explicit role schedule family')
    lanes = ([dict(member=mainline, second_s=onset, source_lane=1, target_lane=0)]
             if family in ('yield_then_merge', 'g5_yield_lane') else [])
    if family != 'g5_yield_lane':
        lanes.append(dict(member=ramp, second_s=onset + 1, source_lane=2, target_lane=1))
    offsets = []
    if family == 'open_gap_merge':
        offsets.append(dict(member=mainline, onset_s=onset, hold_s=hold, accel_mps2=-depth))
    if ramp_offset:
        offsets.append(dict(member=ramp, onset_s=onset, hold_s=hold, accel_mps2=ramp_offset))
    result = dict(schema=SCHEMA, family=family, members=[mainline, ramp], onset_s=onset,
                  hold_s=hold, depth=depth, lane_requests=lanes, residual_boxes=offsets,
                  duration_s=onset + hold, control_dt_s=1, attempt_policy='once_no_retry',
                  lane_ownership_s=1)
    return validate_schedule(result)


def validate_schedule(s):
    fields = {'schema', 'family', 'members', 'onset_s', 'hold_s', 'depth', 'lane_requests',
              'residual_boxes', 'duration_s', 'control_dt_s', 'attempt_policy', 'lane_ownership_s'}
    if set(s) != fields or s['schema'] != SCHEMA or s['family'] not in FAMILIES:
        raise ValueError('explicit role schedule schema/whitelist')
    members = s['members']
    if (not isinstance(members, list) or len(members) != 2 or len(set(members)) != 2
            or any(not isinstance(v, str) or not v for v in members)):
        raise ValueError('ordered distinct (mainline, ramp) identities required')
    onset, hold, depth = s['onset_s'], s['hold_s'], s['depth']
    if (type(onset) is not int or not 0 <= onset <= 30
            or type(hold) is not int or not 20 <= hold <= 90
            or type(depth) not in (int, float) or not math.isfinite(depth) or not .3 <= depth <= 1.
            or s['duration_s'] != onset + hold or type(s['duration_s']) is not int
            or type(s['control_dt_s']) is not int or s['control_dt_s'] != 1
            or type(s['lane_ownership_s']) is not int or s['lane_ownership_s'] != 1
            or s['attempt_policy'] != 'once_no_retry'):
        raise ValueError('unregistered role timing/depth or ownership interval')
    expected = []
    if s['family'] in ('yield_then_merge', 'g5_yield_lane'):
        expected.append(dict(member=members[0], second_s=onset, source_lane=1, target_lane=0))
    if s['family'] != 'g5_yield_lane':
        expected.append(dict(member=members[1], second_s=onset + 1, source_lane=2, target_lane=1))
    if s['lane_requests'] != expected:
        raise ValueError('lane requests do not match exact role/timing semantics')
    seen = set()
    for box in s['residual_boxes']:
        if set(box) != {'member', 'onset_s', 'hold_s', 'accel_mps2'}:
            raise ValueError('residual box whitelist')
        v, a = box['member'], box['accel_mps2']
        if (v not in members or v in seen or type(a) not in (int, float) or not math.isfinite(a)
                or not .3 <= abs(a) <= 1. or box['onset_s'] != onset or box['hold_s'] != hold):
            raise ValueError('invalid/duplicate per-role residual box')
        if v == members[0] and (s['family'] != 'open_gap_merge' or a != -depth):
            raise ValueError('mainline box must be the exact registered open-gap action')
        seen.add(v)
    if s['family'] == 'open_gap_merge' and members[0] not in seen:
        raise ValueError('open_gap_merge must actually command its mainline actor')
    return s


def targets_at(schedule, second, active_members):
    validate_schedule(schedule)
    return {x['member']: (x['source_lane'], x['target_lane']) for x in schedule['lane_requests']
            if x['second_s'] == second and x['member'] in active_members}


def offsets_at(schedule, second, active_members):
    validate_schedule(schedule)
    return {b['member']: b['accel_mps2'] for b in schedule['residual_boxes']
            if b['member'] in active_members and b['onset_s'] <= second < b['onset_s'] + b['hold_s']}


def action_set(schedule):
    """One exact joint and its two member-release counterfactuals; no base mutation."""
    from .wm_contract import digest
    validate_schedule(schedule)
    ident = 'L' + digest(schedule)[:16]
    action = dict(id=ident, wm_role_schedule=copy.deepcopy(schedule),
                  wm_active_members=list(schedule['members']))
    return [action] + [dict(copy.deepcopy(action), id=ident + '_solo' + str(i),
                           wm_active_members=[v], solo_of=ident)
                       for i, v in enumerate(schedule['members'])]


def fixed_grid(mainline, ramp, *, onsets=(0, 10, 30), holds=(20, 60), depths=(.3, 1.)):
    """Small fixed slate, not outcome-derived; canonical G5 remains separately named."""
    if len(onsets) > 3 or len(holds) > 2 or len(depths) > 2:
        raise ValueError('bounded grid only')
    out = []
    for onset in onsets:
        out += action_set(make_schedule('yield_then_merge', mainline, ramp, onset))
        for hold in holds:
            for depth in depths:
                out += action_set(make_schedule('open_gap_merge', mainline, ramp, onset, hold, depth))
    return out


def native_safe_targets(schedule, second, active_members, local_states, eligible_cavs):
    """Native executor hook: same existing G5 observable safety, no TraCI writer.

    MAIN's additive native executor can combine this result with offsets_at and
    pass targets through the existing V2V/env.step lane actuator. This does not
    make this richer schedule admissible to the canonical G5 bank reader.
    """
    from tools.coopmwm.group_lane import safe_targets
    requests = {v: target for v, (_, target) in targets_at(schedule, second, active_members).items()}
    return safe_targets(requests, local_states, eligible_cavs)
