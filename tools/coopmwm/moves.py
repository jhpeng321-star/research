"""Cooperative move library: the shared move vocabulary (MAIN 2026-11-04, track 1 T1A).

A "move" = (family, member CAV ids (1..4 nearby CAVs incl. self), onset s,
hold s, depth) -> per-member per-second command offsets relative to the base
policy (MAPPO u0 / current policy), clamped by the existing feasibility bounds
(the registered CAV cap 1.2 m/s^2 and observable allowed-speed ceiling).
The speedMode 27 safe-speed enforcement stays in the existing executor.
The optional state clamp mirrors the plant's
``traffic_shepherd.damping.direct_target`` band -- ``lo = max(accel_min, -v/dt)``,
``hi = min(cap, (allowed - v)/dt)`` -- mirrored in ``feasible.issued_target``
and ``mppi_search.centre_feasibility``; it is applied to the FINAL command
(base + offset), so the feasible offset interval is [lo - base,
hi - base]. The executor's CommandOwnership clip stays the last
authority; this module never bypasses it.

Families are the ``variety.py`` families plus ``'none'``:
``('none', 'ease_off', 'wall', 'gap_open', 'ramp_assist')``. ``ease_off`` is
variety's free/smoothing direction projected onto a single member; ``wall``,
``gap_open`` and ``ramp_assist`` keep their variety semantics (wall = the
abreast members hold together upstream of the merge zone and release;
gap_open = the mainline CAV ahead of a ramp merger eases; ramp_assist = the
ramp CAV and the mainline CAV ahead coordinate an ease). Every number below is
registered: the sampling windows are ``variety.VARIETY_REGISTERED`` (the same
unsourced ranges the structured plan generators sample), the member arity cap
is ``v2v_obs.V2V_K`` (the K nearest V2V partner slots plus self) and the
sensing radius is ``v2v_obs.V2V_RANGE_M``; the CAV cap is the shared spec's
explicit 1.2 m/s^2 bound (no population/truth-file lookup).
No new hand-set behavioural threshold
is introduced; the only numeric cut in this module is the float-noise level
that separates a recorded plan offset from zero.

The move vocabulary shares ONE (onset, hold, depth) across members: a move's
offsets are ``-depth`` m/s^2 for its actors during ``[onset, onset+hold)``
seconds after the move starts and 0 elsewhere (release back to the policy).
Roles per family: ``ease_off``/``gap_open`` act through ``members[0]`` only
(the acting CAV; for ``gap_open`` the easing mainline CAV is stored first),
``wall``/``ramp_assist`` act through every member together. variety's
two-window ramp_assist projects onto this shared window (see ``from_plan``).

API (consumed by T1B planner_online, T2A trigger_policy, T2B dense_teacher):
    moves.FAMILIES
    moves.candidates(obs_state, rng, n) -> list[Move]
    moves.offsets(move, t[, state]) -> dict[cav_id, accel_offset]
    moves.from_plan(verified_plan) -> nearest Move

``obs_state`` is the acting CAV's OBSERVABLE view only: ``{'self': vid,
'rows': [observable rows with vid/s_m/v_mps/lane/is_cav]}`` (the same row
convention as ``mppi_search.group_components``; rows are already the sensing /
V2V partner set, and membership is further restricted to ``V2V_RANGE_M``).
``from_plan`` accepts the verified bank plan dicts exactly as
``replay_bank``/``sumo_verify`` receipts carry them (``chosen_plan`` and the
per_seed_v1 ``ex['plan']``): mppi_search/variety profile plans (with
``perturbation_profile`` or bare ``profile``) and legacy knots plans
(``plans.validate`` format). Untagged plans are family-classified from their
own offset structure; a ``variety_family`` tag wins when present.
"""
import math
from dataclasses import dataclass

import numpy as np

from .variety import VARIETY_REGISTERED
from .v2v_obs import V2V_K, V2V_RANGE_M

CONTROL_DT_S = 1.0   # one-second MAPPO control interval (the plant's decision tick)
CAV_ACCEL_CAP_MPS2 = 1.2   # tracks_spec.md, Move library feasibility bound
FAMILIES = ('none', 'ease_off', 'wall', 'gap_open', 'ramp_assist')
STRUCTURED = FAMILIES[1:]          # candidates() rotates these; 'none' is the caller's baseline
MOVE_MAX_MEMBERS = int(V2V_K)      # 1..4 nearby CAVs incl. self (the V2V partner-slot arity)
#: A recorded plan offset counts as active above float noise only (numerical,
#: not a behavioural threshold; same spirit as feasible.GATE_EPS_MPS).
ACTIVE_EPS = 1e-6

# Registered sampling windows / distances (variety.VARIETY_REGISTERED; the SAME
# ranges the structured plan generators sample -- no new hand-set values here).
_ONSET_LO, _ONSET_HI = (float(x) for x in VARIETY_REGISTERED['variety_onset_window_s'])
_HOLD_LO, _HOLD_HI = (float(x) for x in VARIETY_REGISTERED['variety_hold_window_s'])
_DEPTH_LO, _DEPTH_HI = (float(x) for x in VARIETY_REGISTERED['variety_depth_window'])
_RAMP_LANE = int(VARIETY_REGISTERED['variety_ramp_lane'])
_MAIN_LANE = int(VARIETY_REGISTERED['variety_mainline_lane'])
_ZONE_S0, _ZONE_S1 = (float(x) for x in VARIETY_REGISTERED['variety_merge_zone_m'])
_WALL_MIN_LANES = int(VARIETY_REGISTERED['variety_wall_min_lanes'])

_ROW_KEYS = ('vid', 's_m', 'v_mps', 'lane', 'is_cav')


def _in_window(value, lo, hi):
    return math.isfinite(float(value)) and float(lo) <= float(value) <= float(hi)


def _is_second(value):
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class Move:
    """One cooperative move of the shared vocabulary (see the module docstring)."""

    family: str
    members: tuple
    onset: int
    hold: int
    depth: float

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f'unknown move family {self.family!r}')
        members = tuple(self.members)
        object.__setattr__(self, 'members', members)
        if not 1 <= len(members) <= MOVE_MAX_MEMBERS or len(set(members)) != len(members):
            raise ValueError(f'move members must be 1..{MOVE_MAX_MEMBERS} distinct CAV ids')
        if isinstance(self.depth, bool) or not math.isfinite(float(self.depth)):
            raise ValueError('depth must be a finite number of m/s^2')
        if self.family == 'none':
            if len(members) != 1 or not _is_second(self.onset) or not _is_second(self.hold) \
                    or (self.onset, self.hold, float(self.depth)) != (0, 0, 0.0):
                raise ValueError("the 'none' move is the zero move: one member, onset/hold/depth 0")
            return
        if self.family == 'ease_off' and len(members) != 1:
            raise ValueError('ease_off acts through the single acting CAV')
        if self.family in ('wall', 'gap_open', 'ramp_assist') and len(members) < 2:
            raise ValueError(f'{self.family} is a cooperative move (at least two members)')
        if not _is_second(self.onset) or not _is_second(self.hold) \
                or not _in_window(self.onset, _ONSET_LO, _ONSET_HI) \
                or not _in_window(self.hold, _HOLD_LO, _HOLD_HI) \
                or not _in_window(self.depth, _DEPTH_LO, _DEPTH_HI):
            raise ValueError(f'parameters outside the registered variety windows '
                             f'onset {_ONSET_LO}..{_ONSET_HI} s, hold {_HOLD_LO}..{_HOLD_HI} s, '
                             f'depth {_DEPTH_LO}..{_DEPTH_HI} m/s^2')

    @property
    def actors(self):
        """Members the move commands away from the policy during the hold window."""
        if self.family in ('ease_off', 'gap_open'):
            return (self.members[0],)   # the acting CAV; gap_open stores the easing mainline CAV first
        if self.family == 'none':
            return ()
        return tuple(self.members)      # wall / ramp_assist: the members ease together

    def to_dict(self):
        """JSON-serialisable dict (bank labels persist this)."""
        return dict(family=self.family, members=list(self.members),
                    onset=int(self.onset), hold=int(self.hold), depth=float(self.depth))

    @classmethod
    def from_dict(cls, data):
        return cls(data['family'], tuple(data['members']), data['onset'],
                   data['hold'], data['depth'])


def none_move(vid):
    """The 'none' move: the base policy continues, no offset for anyone."""
    return Move('none', (vid,), 0, 0, 0.0)


# --------------------------------------------------------------------------- offsets

def offsets(move, t, state=None):
    """Per-member per-second command offsets (m/s^2) relative to the base policy.

    ``t`` is the integer second from the move's start; outside
    ``[onset, onset+hold)`` every offset is 0 (release back to the policy).
    ``state`` (optional) carries the observable second's ``base`` commands,
    ``v_mps``/``allowed`` speeds and the executor ``accel_bounds``; when they
    resolve for a member, its offset is clipped so that base + offset stays
    inside the ``damping.direct_target`` band (the registered CAV cap and
    observable allowed speed), mirroring ``feasible.issued_target``: an
    infeasible member-second (lo > hi) gets offset 0, an unresolvable member
    keeps the raw offset (the executor's clip remains the authority). With
    ``state['s_m']`` a wall releases all members as soon as any member reaches
    the merge zone; live executors should pass current observable state for
    this guard because the candidate's arrival estimate is constant-speed.
    """
    if not isinstance(move, Move):
        raise ValueError('offsets needs a moves.Move')
    if isinstance(t, float) and float(t).is_integer():
        t = int(t)
    if not isinstance(t, int) or isinstance(t, bool) or t < 0:
        raise ValueError('t must be a non-negative integer second')
    actors = move.actors
    active = move.family != 'none' and move.onset <= t < move.onset+move.hold
    if active and move.family == 'wall' and state is not None:
        positions = state.get('s_m') or {}
        if any(vid in positions and float(positions[vid]) >= _ZONE_S0 for vid in move.members):
            active = False
    raw = {v: (-float(move.depth) if active and v in actors else 0.0) for v in move.members}
    if state is None:
        return raw
    base = state.get('base') or {}
    speeds = state.get('v_mps') or {}
    allowed = state.get('allowed') or {}
    bounds = state.get('accel_bounds')
    out = {}
    for vid, off in raw.items():
        resolvable = (bounds is not None and vid in base and vid in speeds and vid in allowed
                      and all(math.isfinite(float(x)) for x in
                              (base[vid], speeds[vid], allowed[vid], bounds[0], bounds[1])))
        if not resolvable:
            out[vid] = off        # unknown member-second: never guess a clamp
            continue
        v = float(speeds[vid])
        lo = max(float(bounds[0]), -v/CONTROL_DT_S)
        hi = min(float(bounds[1]), CAV_ACCEL_CAP_MPS2,
                 (float(allowed[vid])-v)/CONTROL_DT_S)
        if lo > hi:
            out[vid] = 0.0        # infeasible second (direct_target returns None): no offset
            continue
        # Offset is added to the caller's original base, not to a silently
        # replaced base: clip(base + off) - base. The zero move must stay zero;
        # the existing executor clamps baseline-only seconds itself.
        out[vid] = (min(max(float(base[vid])+off, lo), hi)-float(base[vid])
                    if off else 0.0)
    return out


# --------------------------------------------------------------------------- candidates

def _row_checked(row):
    if not isinstance(row, dict) or any(k not in row for k in _ROW_KEYS):
        raise ValueError(f'observable row lacks the registered keys {_ROW_KEYS}')
    if not all(math.isfinite(float(row[k])) for k in ('s_m', 'v_mps', 'lane')) \
            or float(row['v_mps']) < 0:
        raise ValueError('observable position/lane must be finite and speed non-negative')
    return row


def _nearby(partner_rows, s_self):
    """Observable partners within the V2V sensing radius, nearest first."""
    ok = [r for r in partner_rows if abs(float(r['s_m'])-s_self) <= float(V2V_RANGE_M)]
    return sorted(ok, key=lambda r: (abs(float(r['s_m'])-s_self), r['vid']))


def _leader_first(rows):
    return sorted(rows, key=lambda r: (-float(r['s_m']), int(r['lane']), r['vid']))


def _zone_pools(cavs, self_row):
    """Ramp/mainline candidate rows inside the registered observable merge zone."""
    pools = {}
    for lane, key in ((_RAMP_LANE, 'ramps'), (_MAIN_LANE, 'mains')):
        rows = [r for r in cavs if int(r['lane']) == lane
                and _ZONE_S0 <= float(r['s_m']) <= _ZONE_S1]
        pools[key] = [r for r in rows if r['vid'] == self_row['vid']
                      or abs(float(r['s_m'])-float(self_row['s_m'])) <= float(V2V_RANGE_M)]
    return pools


def _sampled(rng, onset=None):
    """One (onset, hold, depth) draw from the registered variety windows."""
    if onset is None:
        onset = int(rng.uniform(_ONSET_LO, _ONSET_HI))
    onset = int(np.clip(int(onset), _ONSET_LO, _ONSET_HI))
    hold = int(rng.uniform(_HOLD_LO, _HOLD_HI))
    depth = float(rng.uniform(_DEPTH_LO, _DEPTH_HI))
    return onset, hold, depth


def _candidate(family, self_row, partners, rng):
    """One candidate Move of ``family`` for the acting CAV, or None (unavailable
    geometry: the rotation slot falls to the next family, as in variety)."""
    vid = self_row['vid']
    if family == 'ease_off':
        return Move('ease_off', (vid,), *_sampled(rng))
    if family == 'wall':
        span = float(VARIETY_REGISTERED['variety_wall_span_m'])
        near = [r for r in _nearby(partners, float(self_row['s_m']))
                if abs(float(r['s_m'])-float(self_row['s_m'])) <= span
                and float(r['s_m']) < _ZONE_S0]
        if float(self_row['s_m']) >= _ZONE_S0:
            return None
        if not near:
            return None
        rows = [self_row]+near[:MOVE_MAX_MEMBERS-1]
        if len({int(r['lane']) for r in rows}) < _WALL_MIN_LANES:
            return None
        onset, hold, depth = _sampled(rng)
        # Observable constant-speed arrival is conservative for this easing
        # move: release no later than the first member enters the merge zone.
        arrival = min(((_ZONE_S0-float(r['s_m']))/float(r['v_mps'])
                       if float(r['v_mps']) > 0 else math.inf) for r in rows)
        if arrival < _HOLD_LO:
            return None
        onset = min(onset, int(min(_ONSET_HI, arrival-_HOLD_LO)))
        hold = min(hold, int(min(_HOLD_HI, arrival-onset)))
        return Move('wall', tuple(r['vid'] for r in _leader_first(rows)), onset, hold, depth)
    pools = _zone_pools([self_row]+list(partners), self_row)
    ramps, mains = pools['ramps'], pools['mains']
    link = float(VARIETY_REGISTERED['variety_gap_link_m'])
    ramps = [r for r in ramps if abs(float(r['s_m'])-float(self_row['s_m'])) <= link]
    mains = [r for r in mains if abs(float(r['s_m'])-float(self_row['s_m'])) <= link]
    in_ramps = any(r['vid'] == vid for r in ramps)
    in_mains = any(r['vid'] == vid for r in mains)
    if family == 'gap_open':
        # The easing mainline CAV (members[0], the actor) + the ramp merger it
        # anticipates; the acting CAV is always one of the two. Onset mirrors
        # variety's predicted-arrival lead: floor(arrival - lead), arrival =
        # gap / merger speed, else the raw lead (variety.gap_open_candidate).
        if in_mains and ramps:
            eased, merger = self_row, ramps[int(rng.integers(0, len(ramps)))]
        elif in_ramps and mains:
            merger, eased = self_row, mains[int(rng.integers(0, len(mains)))]
        else:
            return None
        onset, hold, depth = _sampled(rng, onset=_gap_onset(eased, merger, rng))
        return Move('gap_open', (eased['vid'], merger['vid']), onset, hold, depth)
    if family == 'ramp_assist':
        # The mainline CAV ahead eases and the ramp CAV synchronises down with
        # it (shared window in the move vocabulary); the acting CAV is one of
        # the two (the ahead-most mainline CAV, as in variety.ramp_assist).
        if in_mains and ramps:
            ahead = max(mains, key=lambda r: (float(r['s_m']), r['vid']))
            if ahead['vid'] != vid:
                return None
            merger = ramps[int(rng.integers(0, len(ramps)))]
        elif in_ramps and mains:
            merger = self_row
            ahead = max(mains, key=lambda r: (float(r['s_m']), r['vid']))
        else:
            return None
        onset, hold, depth = _sampled(rng)
        members = _leader_first([ahead, merger])
        return Move('ramp_assist', tuple(r['vid'] for r in members), onset, hold, depth)
    raise ValueError(f'unhandled family {family!r}')


def _gap_onset(eased, merger, rng):
    gap = float(eased['s_m'])-float(merger['s_m'])
    arrival = gap/max(float(merger['v_mps']), 1e-6)
    lead = float(rng.uniform(_ONSET_LO, _ONSET_HI))
    return int(math.floor(arrival-lead)) if math.isfinite(arrival) and arrival > 1 else int(lead)


def candidates(obs_state, rng, n):
    """Up to ``n`` candidate moves for the acting CAV from its OBSERVABLE view.

    ``obs_state`` = ``{'self': vid, 'rows': [observable rows]}`` (rows carry
    ``vid/s_m/v_mps/lane/is_cav``; HDVs are never members). ``rng`` is a
    ``numpy.random.Generator``. Families rotate so the mix stays balanced
    (variety's rotation; an unavailable family's slot falls to the next),
    parameters are drawn from the registered variety windows, members are
    chosen only from CAVs inside the acting CAV's sensing range / V2V partners
    and always include the acting CAV. Wall members are upstream, within the
    registered span, with release at/before their observable constant-speed
    merge-zone arrival. The 'none' baseline is NOT included: callers roll it
    out separately (T1B). Exactly ``n`` moves (ease_off is always available).
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError('n must be a non-negative integer')
    if not isinstance(rng, np.random.Generator):
        raise ValueError('rng must be a numpy.random.Generator')
    rows = obs_state.get('rows')
    self_vid = obs_state.get('self')
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError('obs_state needs observable rows')
    rows = [_row_checked(r) for r in rows]
    by_vid = {r['vid']: r for r in rows}
    if len(by_vid) != len(rows):
        raise ValueError('observable rows must have distinct vehicle ids')
    if self_vid not in by_vid:
        raise ValueError('obs_state self must be one of the observable rows')
    self_row = by_vid[self_vid]
    if not self_row['is_cav']:
        raise ValueError('the acting CAV must be a CAV row')
    cavs = [r for r in rows if r['is_cav']]
    partners = [r for r in cavs if r['vid'] != self_vid]
    out, tries = [], 0
    while len(out) < n and tries < 8*n+8:
        tries += 1
        family = STRUCTURED[(tries-1) % len(STRUCTURED)]
        move = _candidate(family, self_row, partners, rng)
        if move is not None:
            out.append(move)
    return out


# --------------------------------------------------------------------------- from_plan

def _plan_offsets(plan):
    """(members, per-member per-second offsets) from a verified bank plan dict.

    Profile plans (mppi_search.build_plan / variety generators, as carried in
    sumo_verify receipts' ``chosen_plan`` and replay_bank per_seed_v1
    ``ex['plan']``) use ``perturbation_profile`` -- the recorded per-second
    command offsets vs the policy -- and fall back to the bare ``profile``
    minus its per-member median when the perturbation is absent. Legacy knots
    plans (``plans.validate`` format) are resampled per second at their knot
    stride (``plans.acceleration_at`` semantics) and taken relative to their
    per-member median baseline.
    """
    members = list(plan.get('members') or [])
    duration = plan.get('duration_s')
    if not members or len(set(members)) != len(members) \
            or not isinstance(duration, int) or isinstance(duration, bool) or duration <= 0:
        raise ValueError('verified plan needs members and a positive duration_s')
    if 'perturbation_profile' in plan:
        m = np.asarray(plan['perturbation_profile'], float)
    elif 'profile' in plan:
        m = np.asarray(plan['profile'], float)
        if m.ndim != 2:
            raise ValueError('plan profile must be a member x second matrix')
        m = m-np.median(m, axis=1, keepdims=True)
    elif 'knots' in plan:
        knots = np.asarray(plan['knots'], float)
        stride = float(plan.get('knot_s', 10.))
        if knots.ndim != 2 or knots.shape[0] != len(members) or knots.shape[1] < 2 \
                or not np.isfinite(knots).all() or not math.isfinite(stride) or stride <= 0:
            raise ValueError('invalid legacy knots or knot_s')
        times = np.arange(knots.shape[1], dtype=float)*stride
        seconds = np.arange(duration, dtype=float)
        m = np.stack([np.interp(seconds, times, row) for row in knots])
        m = m-np.median(m, axis=1, keepdims=True)
    else:
        raise ValueError('unsupported verified plan format (profile/perturbation_profile/knots)')
    if m.ndim != 2 or m.shape != (len(members), duration) or not np.isfinite(m).all():
        raise ValueError('plan offset matrix does not match members x duration_s')
    return members, m


def _member_windows(active):
    """Per-member (onset, hold) of the active seconds ([0, 0) when inactive)."""
    windows = []
    for row in active:
        cols = np.nonzero(row)[0]
        windows.append((int(cols[0]), int(cols[-1])+1-int(cols[0])) if len(cols) else (0, 0))
    return windows


def _project_window(m, actors):
    """Least-squares negative box on the registered one-second parameter grid.

    Zero padding charges for a move extending past the recorded plan, rather
    than silently shortening its hold. Depth is optimised analytically for
    each window; return None when continuing the policy is at least as close.
    """
    horizon = max(m.shape[1], int(_ONSET_HI+_HOLD_HI))
    total = np.zeros(horizon)
    total[:m.shape[1]] = -m[actors].sum(axis=0)
    prefix = np.r_[0., np.cumsum(total)]
    best, gain = None, 0.0
    for onset in range(int(_ONSET_LO), int(_ONSET_HI)+1):
        for hold in range(int(_HOLD_LO), int(_HOLD_HI)+1):
            summed = prefix[onset+hold]-prefix[onset]
            size = len(actors)*hold
            depth = float(np.clip(summed/size, _DEPTH_LO, _DEPTH_HI))
            improvement = 2*depth*summed-size*depth*depth
            if improvement > gain:
                best, gain = (onset, hold, depth), improvement
    return best


def from_plan(plan):
    """Project a verified bank plan onto the shared move vocabulary.

    Accept a bare plan, a sumo_verification receipt (``chosen_plan``), or an
    all8/per_seed_v1 example (its OWN ``plan``); certification is the caller's
    responsibility. Structured variety tags supply the family. Untagged
    negative offsets use one actor -> ease_off, aligned windows -> wall,
    otherwise ramp_assist. Up to four largest-braking-energy members survive.
    Gap-opening puts its actual easing actor first, regardless of plan order.

    Within that family/member projection, minimise squared command-offset
    error over registered integer onset/hold and continuous depth. Include
    zero padding beyond the recorded duration, and choose 'none' if no box
    improves on the policy. Positive offsets therefore never become braking.
    Bare profiles/legacy knots lack the policy reference: their per-member
    median is an explicit approximate reference (not recovered policy truth).
    """
    if not isinstance(plan, dict):
        raise ValueError('from_plan needs a verified plan dict')
    if isinstance(plan.get('plan'), dict):
        plan = plan['plan']
    elif isinstance(plan.get('chosen_plan'), dict):
        plan = plan['chosen_plan']
    members, m = _plan_offsets(plan)
    if plan.get('policy_duplicate'):
        return none_move(members[0])
    active = m < -ACTIVE_EPS
    if not active.any():
        return none_move(members[0])
    energy = np.square(np.minimum(m, 0.)).sum(axis=1)
    used_indices = np.flatnonzero(active.any(axis=1)).tolist()
    tag = plan.get('variety_family')
    if tag in ('wall', 'gap_open', 'ramp_assist') and len(members) >= 2:
        family = tag
    else:
        windows = _member_windows(active)
        used = [w for w, keep in zip(windows, active.any(axis=1)) if keep]
        aligned = len(used) >= 2 and len(set(used)) == 1
        family = 'wall' if aligned else ('ease_off' if len(used) == 1 else 'ramp_assist')
    if family == 'ease_off':
        keep = [max(used_indices, key=lambda i: (float(energy[i]), -i))]
    elif family == 'gap_open':
        actor = max(used_indices, key=lambda i: (float(energy[i]), -i))
        keep = [actor]+[i for i in range(len(members)) if i != actor][:MOVE_MAX_MEMBERS-1]
    else:
        pool = list(range(len(members))) if tag in ('wall', 'ramp_assist') else used_indices
        keep = sorted(sorted(pool, key=lambda i: (-float(energy[i]), i))[:MOVE_MAX_MEMBERS])
    actors = keep[:1] if family in ('ease_off', 'gap_open') else keep
    params = _project_window(m, actors)
    if params is None:
        return none_move(members[0])
    return Move(family, tuple(members[i] for i in keep), *params)
