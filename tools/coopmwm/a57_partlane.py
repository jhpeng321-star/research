"""AMENDMENT 57 (S-COOPSYN60, registered before any outcome): member-lane filter for the A37 CELL_BANK executor.

PART_L<k> = the same frozen admitted bank, the same trigger and the same activation eligibility evaluated on the FULL
planned group (the a37_cellbank / oracle_plan executor runs UNCHANGED, sessions and member exclusivity included);
only the members whose PLANNED role lane (SUMO lane, role['lane']) equals k receive the plan's commands (hold
acceleration and the lane hold). Every other planned member gets no plan command: its desired entry stays the frozen
MAPPO U0 command, exactly as for a non-member.

The filter is applied at command time by wrapping hook.core.step of an a37_cellbank.make_hook hook. With no filter
(the runner sees no ``a57_member_lane`` in the checkpoint) nothing here is imported or called: the A42 path is
unchanged (a37_sha256 unchanged). No RNG, no clock.
"""
from __future__ import annotations

import hashlib
import inspect
import json

from . import oracle_plan as O

SCHEMA = 'coopmwm.a57_partlane.v1'
LANES = (0, 1, 2)


def planned_lanes(core):
    """{vid: planned SUMO lane} of every active member of every family session of an OracleCore."""
    return {vid: int(m['role']['lane']) for c in core.cores for vid, m in c.active.items()}


def members_on_lane(core, lane):
    """Per activation of every family core: (family, t, members, members whose planned role lane == lane)."""
    out = []
    for c in core.cores:
        for act in c.acts:
            roles = O.active_roles(c.s, act.get('phase', 0))
            on = [vid for vid, r in zip(act['members'], roles) if int(r['lane']) == int(lane)]
            out.append(dict(family=c.s['name'], t=act['t'], members=list(act['members']), on_lane=on))
    return out


def apply_filter(hook, lane):
    """Wrap an a37_cellbank.make_hook hook (CELL_BANK): only planned members on SUMO lane ``lane`` are commanded."""
    if int(lane) not in LANES:
        raise ValueError(f'A57: member lane {lane!r} not in {LANES}')
    lane = int(lane)
    core = hook.core
    step, receipts = core.step, hook.receipts
    tally = dict(kept_accel=0, dropped_accel=0, kept_lane=0, dropped_lane=0)
    commanded = set()

    def filtered_step(rows, commands, t):
        accel, lanes = step(rows, commands, t)
        planned = planned_lanes(core)
        stray = (set(accel) | set(lanes)) - set(planned)
        if stray:
            raise RuntimeError(f'A57: commands for vehicles outside any session {sorted(stray)}')
        keep_a = {v: a for v, a in accel.items() if planned[v] == lane}
        keep_l = {v: x for v, x in lanes.items() if planned[v] == lane}
        tally['kept_accel'] += len(keep_a)
        tally['dropped_accel'] += len(accel)-len(keep_a)
        tally['kept_lane'] += len(keep_l)
        tally['dropped_lane'] += len(lanes)-len(keep_l)
        commanded.update(keep_a)
        return keep_a, keep_l

    def filtered_receipts():
        acts = members_on_lane(core, lane)
        return dict(receipts(), a57_part=dict(
            schema=SCHEMA, a57_sha256=a57_digest(), member_lane=lane, activations=len(acts),
            activations_no_member_on_lane=sum(1 for a in acts if not a['on_lane']),
            members_on_lane_total=sum(len(a['on_lane']) for a in acts),
            members_planned_total=sum(len(a['members']) for a in acts),
            vehicles_with_accel_command=len(commanded), acts=acts, **tally))

    core.step = filtered_step
    hook.receipts = filtered_receipts
    hook.a57_member_lane = lane
    return hook


EXECUTED = ('planned_lanes', 'members_on_lane', 'apply_filter')
CONSTANTS = ('SCHEMA', 'LANES')


def a57_digest():
    import sys
    m = sys.modules[__name__]
    parts = [inspect.getsource(getattr(m, n)) for n in EXECUTED]
    parts.append(json.dumps({n: getattr(m, n) for n in CONSTANTS}, sort_keys=True, default=list))
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest()
