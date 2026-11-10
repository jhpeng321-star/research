"""AMENDMENT 61 (S-COOPL2SPLIT60, registered before any outcome of this test): within-lane member pick on top of the
amendment-57 member-lane filter (tools/coopmwm/a57_partlane.py, UNCHANGED; a57_sha256 unchanged).

PART_L2A = only the lane-k member with the LOWER planned within-lane rank of its plan (role['rank'] == the minimum rank
among the plan's roles on lane k); PART_L2B = only the other lane-k member(s). Everything else (bank, executor,
trigger, full-group activation eligibility, the a57 lane filter) is unchanged; a dropped member keeps the U0 command,
as a non-member. Applied by the runner only when the checkpoint carries a61_member_pick + a61_sha256 (on top of
a57_member_lane); absent = the A57 path unchanged. No RNG, no clock.
"""
from __future__ import annotations

import hashlib
import inspect
import json

from . import oracle_plan as O

SCHEMA = 'coopmwm.a61_rankpick.v1'
PICKS = ('A', 'B')


def lane_roles(core, lane):
    """{vid: (rank, is_low)} of every active member planned on ``lane``; is_low = its rank is the lowest planned rank
    among its plan's roles on that lane."""
    out = {}
    for c in core.cores:
        for vid, m in c.active.items():
            r = m['role']
            if int(r['lane']) != int(lane):
                continue
            ranks = [int(x['rank']) for x in O.active_roles(c.s, m['act'].get('phase', 0)) if int(x['lane']) == int(lane)]
            out[vid] = (int(r['rank']), int(r['rank']) == min(ranks))
    return out


def picks_of(core, lane, pick):
    """Per activation: (family, t, members on the lane by planned rank, the picked members)."""
    out = []
    for c in core.cores:
        for act in c.acts:
            roles = O.active_roles(c.s, act.get('phase', 0))
            on = [(int(r['rank']), vid) for vid, r in zip(act['members'], roles) if int(r['lane']) == int(lane)]
            low = min((k for k, _ in on), default=None)
            got = [v for k, v in sorted(on) if (k == low) == (pick == 'A')]
            out.append(dict(family=c.s['name'], t=act['t'], on_lane=[v for _, v in sorted(on)], picked=got))
    return out


def apply_pick(hook, lane, pick):
    """Wrap a hook already filtered by a57_partlane.apply_filter(hook, lane): keep only the picked lane member(s)."""
    if pick not in PICKS:
        raise ValueError(f'A61: pick {pick!r} not in {PICKS}')
    if getattr(hook, 'a57_member_lane', None) != int(lane):
        raise ValueError('A61: the hook must carry the a57 member-lane filter of the same lane')
    lane = int(lane)
    core = hook.core
    step, receipts = core.step, hook.receipts
    tally = dict(kept_accel=0, dropped_accel=0, kept_lane=0, dropped_lane=0)
    commanded = set()

    def picked_step(rows, commands, t):
        accel, lanes = step(rows, commands, t)
        info = lane_roles(core, lane)
        stray = (set(accel) | set(lanes)) - set(info)
        if stray:
            raise RuntimeError(f'A61: commands for vehicles not planned on lane {lane}: {sorted(stray)}')
        keep = {v for v, (_, low) in info.items() if low == (pick == 'A')}
        ka = {v: a for v, a in accel.items() if v in keep}
        kl = {v: x for v, x in lanes.items() if v in keep}
        tally['kept_accel'] += len(ka)
        tally['dropped_accel'] += len(accel)-len(ka)
        tally['kept_lane'] += len(kl)
        tally['dropped_lane'] += len(lanes)-len(kl)
        commanded.update(ka)
        return ka, kl

    def picked_receipts():
        acts = picks_of(core, lane, pick)
        return dict(receipts(), a61_pick=dict(
            schema=SCHEMA, a61_sha256=a61_digest(), member_lane=lane, pick=pick, activations=len(acts),
            activations_two_lane_members=sum(1 for a in acts if len(a['on_lane']) >= 2),
            activations_fewer_than_two=sum(1 for a in acts if len(a['on_lane']) < 2),
            activations_no_picked_member=sum(1 for a in acts if not a['picked']),
            members_picked_total=sum(len(a['picked']) for a in acts),
            vehicles_with_accel_command=len(commanded), acts=acts, **tally))

    core.step = picked_step
    hook.receipts = picked_receipts
    hook.a61_member_pick = pick
    return hook


EXECUTED = ('lane_roles', 'picks_of', 'apply_pick')
CONSTANTS = ('SCHEMA', 'PICKS')


def a61_digest():
    import sys
    m = sys.modules[__name__]
    parts = [inspect.getsource(getattr(m, n)) for n in EXECUTED]
    parts.append(json.dumps({n: getattr(m, n) for n in CONSTANTS}, sort_keys=True, default=list))
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest()
