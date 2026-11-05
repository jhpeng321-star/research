"""F4 registration and observable, post-move close-call accounting (CPU only).

An event is one adjacent same-lane closing pair per control tick, matching the
existing replay-bank TTC counter. Only atomic, authorized sensor packets are
used: current and t-1 geometry must NEVER be combined to construct a pair.
Attribution is temporal, not a claimed move-versus-none causal estimate.
"""
from copy import deepcopy
import math

from . import merge_features as mf


REGISTRATION = dict(
    schema='coopmwm-f4-close-call-v1', threshold_s=1.5, weight=1.0,
    provenance='MAIN 2026-11-05 15:15 F4; implementation registration: '
               '1.0 speed-deficit reward-second per pair-event; fixed before tests, no tuning',
    threshold_provenance='MAIN 15:15 F4 TTC<1.5s; replay_bank.TTC_EVENT_S',
    arms=['B', 'R'],
    event_rule='adjacent same-lane closing pair per 1s control tick; strict TTC<1.5s',
    attribution='measurement strictly after first nonzero library offset of a currently active '
                'move containing the acting CAV; through commitment expiry; temporal, not counterfactual',
    aggregation='deduplicated observable pair count (own and neighbours), no global outcomes')


def registration():
    return deepcopy(REGISTRATION)


def settings(cfg):
    """Common B/R opt-in; absent/False retains the pre-F4 reward contract.

    Enabled recipes explicitly bind the fixed weight/provenance and event rule;
    there are no arm-specific or tuneable overrides.
    """
    slot = cfg.get('f4_close_call', False)
    if slot is False:
        return None
    if (not isinstance(slot, dict) or set(slot) != {'enabled', 'registration'}
            or slot['enabled'] is not True or slot['registration'] != REGISTRATION
            or type(slot['registration']['weight']) not in (int, float)):
        raise ValueError('F4 f4_close_call requires enabled=True and the fixed common registration')
    return registration()


def move_attribution(sample, time_s, history):
    """Per-episode history of first recorded library offsets, keyed by commitment.

    Active moves are accepted MoveExecutor commitments, not selected/rejected
    trigger proposals. A delayed onset has no attribution until its first
    nonzero offset. A later zero-offset tail retains attribution until expiry.
    No new time window or future observations are introduced.
    """
    result = {}
    offsets = sample['context'].get('move_offsets', {})
    for active in sample.get('track2', {}).get('active_moves', ()):
        move = active['move']
        start, until = active['start'], active['until']
        if not start <= time_s < until or move['family'] == 'none':
            continue
        members = tuple(move['members'])
        key = (start, until, move['family'], members)
        for vid in members:
            offset = offsets.get(vid, 0.)
            if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not math.isfinite(offset):
                raise ValueError('nonfinite F4 recorded move offset')
            if offset != 0.:
                history.setdefault((key, vid), time_s)
            first = history.get((key, vid))
            if first is not None and vid in sample['context']['members']:
                result[vid] = min(result.get(vid, first), first)
    return result


def close_calls(reports, visible, *, first_actuation_s):
    """Count pairs from authorized atomic packets with existing duplicate priority.

    ``visible`` is track2_reward's ID -> (row, timestamp) map. A lower-priority
    packet cannot override current sensing or splice a stale leader onto a
    current follower. Missing pair coverage is absent evidence, never global
    ground truth. Own sensing plus selected timestamped V2V define the domain.
    """
    events = []
    seen = set()
    if first_actuation_s is not None:
        if (type(first_actuation_s) not in (int, float)
                or not math.isfinite(first_actuation_s)):
            raise ValueError('F4 first actuation time must be finite')
        for report in reports:
            if report.timestamp_s <= first_actuation_s:
                continue
            lanes = {}
            for row in mf._local_rows(report):
                lanes.setdefault(row.lane, []).append(row)
            for rows in lanes.values():
                rows.sort(key=lambda row: (row.s_m, row.vehicle_id))
                for follower, leader in zip(rows, rows[1:]):
                    pair = (follower.vehicle_id, leader.vehicle_id)
                    if pair in seen or any(visible.get(r.vehicle_id) != (r, report.timestamp_s)
                                           for r in (follower, leader)):
                        continue
                    seen.add(pair)
                    closing = follower.v_mps - leader.v_mps
                    if closing <= 1e-6:
                        continue
                    gap = leader.s_m - follower.s_m - leader.length_m
                    ttc = max(0., gap) / closing
                    if ttc < REGISTRATION['threshold_s']:
                        events.append(dict(follower_id=pair[0], leader_id=pair[1],
                            timestamp_s=report.timestamp_s, gap_m=gap,
                            closing_speed_mps=closing, ttc_s=ttc))
    events.sort(key=lambda e: (e['follower_id'], e['leader_id']))
    return dict(registration=registration(), first_actuation_s=first_actuation_s,
                events=events, event_count=len(events),
                penalty=-REGISTRATION['weight'] * len(events))


def validate(component, *, cav_id, neighbour_ids, time_s, first_actuation_s):
    """Check P2 evidence/registration and arithmetic at the production PPO edge."""
    if component is None:
        if first_actuation_s is not None:
            raise ValueError('missing F4 close-call evidence after move actuation')
        return 0.  # Historical non-move CPU envelopes have no safety component.
    if (component.get('registration') != REGISTRATION
            or component.get('first_actuation_s') != first_actuation_s):
        raise ValueError('F4 close-call registration/attribution mismatch')
    domain, seen = {cav_id, *neighbour_ids}, set()
    for event in component['events']:
        pair = (event['follower_id'], event['leader_id'])
        if (pair[0] == pair[1] or not set(pair) <= domain or pair in seen
                or first_actuation_s is None):
            raise ValueError('F4 close-call pair outside observable post-move domain')
        seen.add(pair)
        numbers = [event[k] for k in ('timestamp_s', 'gap_m', 'closing_speed_mps', 'ttc_s')]
        if not all(type(v) in (int, float) and math.isfinite(v) for v in numbers):
            raise ValueError('nonfinite F4 close-call evidence')
        timestamp, gap, closing, ttc = numbers
        if (timestamp not in (time_s, time_s + mf.CONTROL_DT_S)
                or timestamp <= first_actuation_s or closing <= 1e-6
                or not 0. <= ttc < REGISTRATION['threshold_s']
                or not math.isclose(ttc, max(0., gap)/closing, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError('invalid F4 post-move TTC evidence')
    penalty = -REGISTRATION['weight'] * len(seen)
    if (type(component['event_count']) is not int or component['event_count'] != len(seen)
            or type(component['penalty']) not in (int, float) or component['penalty'] != penalty):
        raise ValueError('F4 close-call count/penalty mismatch')
    return penalty
