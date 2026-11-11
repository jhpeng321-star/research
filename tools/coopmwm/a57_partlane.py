"""AMENDMENT 57 (S-COOPSYN60, registered before any outcome): member-lane filter for the A37 CELL_BANK executor.

PART_L<k> = the same frozen admitted bank, the same trigger and the same activation eligibility evaluated on the FULL
planned group (the a37_cellbank / oracle_plan executor runs UNCHANGED, sessions and member exclusivity included);
only the members whose PLANNED role lane (SUMO lane, role['lane']) equals k receive the plan's commands (hold
acceleration and the lane hold). Every other planned member gets no plan command: its desired entry stays the frozen
MAPPO U0 command, exactly as for a non-member.

The filter is applied at command time by wrapping hook.core.step of an a37_cellbank.make_hook hook. With no filter
(the runner sees no ``a57_member_lane`` in the checkpoint) nothing here is imported or called: the A42 path is
unchanged (a37_sha256 unchanged). No RNG, no clock.

AMENDMENT 63 (S-STRICTBANK60): the filter takes a lane SET (``a57_member_lane`` = an int k as in A57, or a list of
SUMO lanes, e.g. [0, 1]); the members whose planned role lane is in the set are commanded. An int k behaves exactly as A57 (receipt member_lane = k). The receipt also lists the
planned lane of every vehicle that received an acceleration command (commanded_lanes, for the smoke check).
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


def lane_set(lane):
    """A63: an int k -> (k,) (the A57 form); a list / tuple / set of SUMO lanes -> the sorted distinct tuple."""
    if isinstance(lane, (list, tuple, set, frozenset)):
        out = tuple(sorted({int(x) for x in lane}))
    else:
        out = (int(lane),)
    if not out or any(x not in LANES for x in out):
        raise ValueError(f'A57: member lane(s) {lane!r} must be a non-empty subset of {LANES}')
    return out


def members_on_lane(core, lane):
    """Per activation of every family core: (family, t, members, members whose planned role lane is in the set)."""
    lanes = lane_set(lane)
    out = []
    for c in core.cores:
        for act in c.acts:
            roles = O.active_roles(c.s, act.get('phase', 0))
            on = [vid for vid, r in zip(act['members'], roles) if int(r['lane']) in lanes]
            out.append(dict(family=c.s['name'], t=act['t'], members=list(act['members']), on_lane=on))
    return out


def apply_filter(hook, lane):
    """Wrap an a37_cellbank.make_hook hook (CELL_BANK): only planned members on SUMO lane ``lane`` (an int, or a lane
    set for A63) are commanded."""
    lanes = lane_set(lane)
    lane = lanes[0] if len(lanes) == 1 else list(lanes)
    core = hook.core
    step, receipts = core.step, hook.receipts
    tally = dict(kept_accel=0, dropped_accel=0, kept_lane=0, dropped_lane=0)
    commanded = {}

    def filtered_step(rows, commands, t):
        accel, lane_cmds = step(rows, commands, t)
        planned = planned_lanes(core)
        stray = (set(accel) | set(lane_cmds)) - set(planned)
        if stray:
            raise RuntimeError(f'A57: commands for vehicles outside any session {sorted(stray)}')
        keep_a = {v: a for v, a in accel.items() if planned[v] in lanes}
        keep_l = {v: x for v, x in lane_cmds.items() if planned[v] in lanes}
        tally['kept_accel'] += len(keep_a)
        tally['dropped_accel'] += len(accel)-len(keep_a)
        tally['kept_lane'] += len(keep_l)
        tally['dropped_lane'] += len(lane_cmds)-len(keep_l)
        commanded.update({v: planned[v] for v in keep_a})
        return keep_a, keep_l

    def filtered_receipts():
        acts = members_on_lane(core, lane)
        return dict(receipts(), a57_part=dict(
            schema=SCHEMA, a57_sha256=a57_digest(), member_lane=lane, member_lanes=list(lanes), activations=len(acts),
            activations_no_member_on_lane=sum(1 for a in acts if not a['on_lane']),
            members_on_lane_total=sum(len(a['on_lane']) for a in acts),
            members_planned_total=sum(len(a['members']) for a in acts),
            vehicles_with_accel_command=len(commanded), commanded_lanes=dict(sorted(commanded.items())), acts=acts,
            **tally))

    core.step = filtered_step
    hook.receipts = filtered_receipts
    hook.a57_member_lane = lane
    return hook


EXECUTED = ('planned_lanes', 'lane_set', 'members_on_lane', 'apply_filter')
CONSTANTS = ('SCHEMA', 'LANES')


def a57_digest():
    import sys
    m = sys.modules[__name__]
    parts = [inspect.getsource(getattr(m, n)) for n in EXECUTED]
    parts.append(json.dumps({n: getattr(m, n) for n in CONSTANTS}, sort_keys=True, default=list))
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest()
