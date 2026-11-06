"""G4LD practice OFFER mode (default OFF): offer the verified move, do not force it.

LEARNAUDIT finding 2 (2026-11-06): a forced practice demonstration starts the verified plan
at t0 through VerifyController; every hold tick is then marked ``executed`` and leaves trigger
PPO, so the demonstration itself teaches the trigger nothing (its onset context is imitated by
the separate bank CE path, bank_breadth.observable_contexts, every update). In OFFER mode a
registered fraction of the practice slots that would demonstrate an exact N2 lane-drop plan
instead restore the same snapshot with NO forced plan ('policy continues'), and at the
current policy's first live decision (t0+1; t0 itself is the sealed snapshot command) the
verified move is placed in slot 1 of the offeree's slate. The trigger chooses; that recorded
stochastic decision is an ordinary PPO row (same slate/mask/logp contract), and with the V2V
protocol the other members get ordinary join offers at t+1.

Offeree and option follow lane_drop.protocol_contexts exactly (the CE label rule): with the
N2 protocol the initiator (pair_t: the anchor; else most members within the message range,
ties by role order) is offered ``lane_drop_initiate`` with the remaining onset/hold; without
it (or for group_hold) the joint LaneDropMove with the remaining schedule is offered to the
same CAV. Only observable quantities are used (own sensing + t-1 V2V positions).

Selection is outcome-blind and B = R: slot index i of update u is an offer slot iff
sha256([training_seed, u, i]) / 2**256 < fraction. The slate contract is unchanged when no
offer is pending (byte-identical OFF).
"""
from copy import deepcopy
from dataclasses import replace

from .protocol import digest

OFFER_RULE = 'g4ld-practice-offer-v1'


def settings(cfg):
    """None when OFF (absent/False); else the validated common B = R slot."""
    slot = cfg.get('practice_offer', False)
    if slot is False:
        return None
    if (not isinstance(slot, dict) or set(slot) != {'enabled', 'rule', 'fraction', 'provenance'}
            or slot['enabled'] is not True or slot['rule'] != OFFER_RULE
            or type(slot['fraction']) not in (int, float) or not 0. < float(slot['fraction']) <= 1.
            or not isinstance(slot['provenance'], str) or not slot['provenance'].strip()):
        raise ValueError('practice offer requires the registered rule, a fraction in (0, 1] and provenance')
    return deepcopy(slot)


def config_keys(*, fraction, provenance):
    return dict(enabled=True, rule=OFFER_RULE, fraction=float(fraction), provenance=provenance)


def offered(cfg, update, index):
    """Outcome-blind, arm-blind slot selection (B and R share slot indices)."""
    slot = settings(cfg)
    if slot is None:
        return False
    value = int(digest([int(cfg['training_seed']), int(update), int(index)]), 16)
    return value < float(slot['fraction'])*2**256


def mark_rows(cfg, update, rows):
    """Copies of the frozen practice rows with ``offer`` set on selected demonstrate slots.

    Only exact N2 lane-drop plans are offered; other demonstrations stay forced."""
    out = []
    for index, row in enumerate(rows):
        plan = (row.get('snapshot_lookup') or {}).get('plan') or {}
        if (row.get('mined') or row.get('demonstrate')) and plan.get('lane_drop_macro') is not None \
                and offered(cfg, update, index):
            row = dict(row, offer=True)
        out.append(row)
    return out


def offer_spec(plan, t0):
    """The practice meta entry (recorded in the episode receipt)."""
    return dict(plan=plan, plan_sha256=digest(plan), t0=int(t0), rule=OFFER_RULE)


def _initiator(move, candidates, positions, range_m):
    if move.anchor >= 0:
        anchor = move.members[move.anchor]
        return anchor if anchor in candidates else None

    def reach(v):
        own = positions.get(v, {})
        return sum(1 for m in move.members if m != v and m in own and v in own
                   and abs(float(own[m])-float(own[v])) <= range_m)
    return max(candidates, key=lambda v: (reach(v), -move.members.index(v))) if candidates else None


def options(policy, spec, context, now, free):
    """{offeree: option} for this decision second, or {} when not representable/free."""
    from . import lane_drop
    move = lane_drop.from_plan(spec['plan'])
    elapsed = int(now)-int(spec['t0'])
    onset, hold = max(0, move.onset-elapsed), move.hold-max(0, elapsed-move.onset)
    if elapsed < 0 or hold < lane_drop.HOLD_WINDOW_S[0]:
        return {}
    present = [v for v in move.members if v in context['members']]
    if not set(move.members) <= set(present) or not set(move.members) <= set(free):
        return {}
    protocol = getattr(policy, 'ld_protocol', None)
    range_m = protocol.range_m if protocol is not None else lane_drop.DEFAULT_V2V_RANGE_M
    ego = _initiator(move, [v for v in present if v in getattr(policy, 'visible', {})],
                     policy.execution_positions, range_m)
    if ego is None or not set(move.members) <= set(policy.visible[ego]):
        return {}
    from .track2_backend import TriggerMove
    if protocol is not None and move.family in lane_drop.KINDS:
        view = policy.lane_drop['view']
        lanes = dict(zip(move.members, move.lanes))
        option = lane_drop.LaneDropOption(
            'lane_drop_initiate', (ego,), move.family, onset, hold, move.depth, move.target_mps,
            move.gap_dv_mps, move.zip, lanes[ego], wall_id=f'{ego}@{int(now)}', initiator=ego,
            cap=lane_drop.kind_cap(move.family, view, protocol.member_cap), range_m=protocol.range_m,
            anchor=move.anchor >= 0)
        return {ego: TriggerMove(option)}
    if not policy.lane_drop or not policy.lane_drop['moves']:
        return {}
    return {ego: TriggerMove(replace(move, onset=onset, hold=hold))}
