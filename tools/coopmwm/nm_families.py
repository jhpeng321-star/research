"""N9: additional multi-CAV lane-drop move families for the cooperation layer (DEFAULT OFF).

Six families, each enabled separately (absent/False = off, byte-identical):

- ``platoon_headway``: 2-4 same-lane through-lane CAVs keep a registered tighter
  time headway with a CACC law (gap + relative speed + one-tick V2V feed-forward of
  the predecessor's t-1 packet). The law's acceleration is issued as the member's
  ABSOLUTE request through the existing single writer (committed move offset ->
  coop layer -> acceleration.requests -> CommandOwnership speed command v + a*1 s).
- ``early_merge``: closing-lane CAVs change to the adjacent lane early, one at a
  time front-first, once inside the registered upstream distance; adjacent-lane
  partner CAVs ease (-depth) while a pending merger is ahead within the link
  distance, leaving room.
- ``zipper_order``: closing-lane and adjacent-lane CAVs alternate in a registered
  fixed order (through_first / closing_first); the second of each pair eases until
  it is behind its designated predecessor; closing-lane members change lanes inside
  the registered zipper zone at the drop point.
- ``lane_balance``: adjacent-lane CAVs move to the far lane upstream (registered
  zone), front-first.
- ``discharge_sync``: queued through-lane CAVs at the front of the queue near the
  drop point start accelerating together at a common start second (registered
  lead) with a common acceleration bounded by a gap law.
- ``harmonize_rows``: two abreast rows (one member per (row, lane) slot) upstream
  track a common registered speed and release at the registered distance before
  the drop.

Lane changes are NEVER written here: lane requests enter ``sample['g5_lane_requests']``
and pass the G5 path (group_lane.controller_targets -> safe_targets envelope with
each CAV's OWN sensed gaps -> lanes_tensor -> env ``lane_actors``), CAVs only.

V2V initiate/join reuses the W1 pattern (v2v_wall): a free CAV initiates
``<family>_initiate`` from its own state; the initiator publishes a FamilyIntent
every second (visible only at t+1, within ``range_m``); a free CAV that receives it
and satisfies the family's LOCAL join rule gets ``<family>_join`` (one-shot event
decision on first receipt). Ownership: executor commitment (shared with every
other move) + one session per CAV + registered member cap + per-role/slot caps.

Trigger inputs are observable only: own sensor row (context x: s, v, lane), the
static lane-drop descriptor and the received t-1 message. B and R share one code
path (no arm input). HDVs are never members, offered options or commanded.
Map geometry comes only from :mod:`nm_map` (N1 stub until bound).
"""
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np
import torch

from . import moves, nm_map
from .merge_features import CONTROL_DT_S, SENSOR_RANGE_M, _valid_report
from .v2v_obs import V2V_RANGE_M

RULE = 'n9-nm-families-v1'
CONFIG_KEY = 'nm_families'
KINDS = ('initiate', 'join')
FAMILY_NAMES = ('platoon_headway', 'early_merge', 'zipper_order', 'lane_balance',
                'discharge_sync', 'harmonize_rows')
OPTION_FAMILIES = tuple(f'{n}_{k}' for n in FAMILY_NAMES for k in KINDS)
EXECUTABLE_SCHEMA = 'n9-nm-move-executable-v1'

ONSET = (moves._ONSET_LO, moves._ONSET_HI)
HOLD = (moves._HOLD_LO, moves._HOLD_HI)
DEPTH = (moves._DEPTH_LO, moves._DEPTH_HI)
PROBE = ('runs/active/C4-V7FT-20261017/newmap_probe/make_jobs.py sha256 fc3f668e '
         '(WALL_ZONE x_b-1800..x_b-400 m, WALL_SPREAD 60 m, wall v_w 15/22 m/s, merge max_s 10 s)')


@dataclass(frozen=True)
class Param:
    key: str
    lo: float = 0.
    hi: float = 0.
    kind: str = 'float'          # float | int | choice
    choices: tuple = ()

    def sample(self, rng):
        if self.kind == 'choice':
            return self.choices[int(rng.integers(len(self.choices)))]
        if self.kind == 'int':
            return int(rng.integers(int(self.lo), int(self.hi)+1))
        return float(rng.uniform(self.lo, self.hi))

    def valid(self, value):
        if self.kind == 'choice':
            return value in self.choices
        if self.kind == 'int':
            return type(value) is int and self.lo <= value <= self.hi
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and self.lo <= value <= self.hi)

    def normalised(self, value):
        if self.kind == 'choice':
            return self.choices.index(value)/max(1, len(self.choices)-1)
        return (float(value)-self.lo)/(self.hi-self.lo) if self.hi > self.lo else 0.

    def to_dict(self):
        return (dict(key=self.key, kind='choice', choices=list(self.choices)) if self.kind == 'choice'
                else dict(key=self.key, kind=self.kind, lo=self.lo, hi=self.hi))


@dataclass(frozen=True)
class Spec:
    name: str
    roles: tuple
    cap: tuple                   # registered (min, max) member cap
    params: tuple                # Param(...) sampled at initiation, inherited by joins
    constants: tuple             # ((key, value), ...) fixed registered constants
    depth: str                   # sampled | pinned_lo | pinned_hi
    depth_meaning: str
    lane_roles: tuple            # ((role, 'adjacent'|'far'), ...) roles that change lane
    absolute: bool               # profile is an absolute acceleration target (needs native)
    provenance: str

    def constant(self, key):
        return dict(self.constants)[key]

    def param(self, key):
        return next(p for p in self.params if p.key == key)

    def lane_target_role(self, role):
        return dict(self.lane_roles).get(role)


CACC = (('standstill_gap_m', 2.5), ('kp', .1), ('kd', .4), ('kff', .5))
CACC_PROVENANCE = ('CACC gap regulation after Milanes & Shladover 2014 (TR-C 48): a = kp*(gap - s0 - h*v) '
                   '+ kd*(v_pred - v) + kff*(v_pred - v_pred[t-1 V2V packet]); gains/h range chosen by the '
                   'N9 discrete-time string-stability check (1 s control tick, one-tick feed-forward delay): '
                   'max_w |Gamma(e^jw)| <= 1 for h in [1.1, 1.4] s (string_gain); kff = 0 is NOT string stable '
                   'there. s0 = SUMO default minGap 2.5 m. Tighter than the IDM textbook T = 1.5 s. '
                   'N9 registration for owner review.')

SPECS = {
    'platoon_headway': Spec(
        'platoon_headway', ('head', 'follower'), (2, 4),
        (Param('time_headway_s', 1.1, 1.4),), CACC, 'pinned_hi',
        'unused (pinned to the variety depth maximum); the CACC law has full authority, '
        'bounded only by the executor feasibility clip and the 1.2 m/s^2 CAV cap',
        (), True, CACC_PROVENANCE),
    'early_merge': Spec(
        'early_merge', ('merge', 'yield'), (2, 4),
        (Param('upstream_m', 400., 1800.), Param('sequence_gap_s', 2, 5, 'int'),
         Param('attempt_s', 3, 10, 'int')),
        (('link_m', float(moves.VARIETY_REGISTERED['variety_gap_link_m'])),), 'sampled',
        'yield-role easing depth (m/s^2 below the policy) while a pending merger is ahead within link_m',
        (('merge', 'adjacent'),), False,
        'upstream_m = probe wall zone '+PROBE+'; attempt_s upper bound = probe merge max_s 10 s; '
        'sequence_gap_s lower bound = two-second rule; link_m = variety_gap_link_m. N9 registration.'),
    'zipper_order': Spec(
        'zipper_order', ('closing', 'through'), (2, 4),
        (Param('zipper_zone_m', 50., 250.), Param('order_headway_s', 1., 2.),
         Param('attempt_s', 3, 10, 'int'), Param('order', kind='choice', choices=('through_first', 'closing_first'))),
        (('standstill_gap_m', 2.5),), 'sampled',
        'easing depth of a member not yet behind its designated predecessor',
        (('closing', 'adjacent'),), False,
        'zipper_zone_m upper bound ~ MUTCD merging taper L = W*S (12 ft, 65 mph ~ 238 m); order_headway_s '
        '1-2 s (two-second rule upper bound); attempt_s as early_merge. N9 registration.'),
    'lane_balance': Spec(
        'lane_balance', ('mover',), (1, 3),
        (Param('sequence_gap_s', 2, 5, 'int'), Param('attempt_s', 3, 10, 'int')),
        (('zone_far_m', 1800.), ('zone_near_m', 400.)), 'pinned_lo',
        'unused (pinned to the variety depth minimum); lane-only family',
        (('mover', 'far'),), False,
        'upstream zone = probe wall zone '+PROBE+'. N9 registration.'),
    'discharge_sync': Spec(
        'discharge_sync', ('sync',), (2, 4),
        (Param('queue_speed_mps', 2.24, 8.94), Param('zone_m', 50., 300.),
         Param('release_speed_mps', 15., 22.), Param('sync_lead_s', 2, 5, 'int')),
        (('time_headway_s', 1.4),)+CACC[:3], 'sampled',
        'common synchronised acceleration (m/s^2), bounded per member by the CACC gap law (no feed-forward)',
        (), True,
        'queue speed 5-20 mi/h (HCM queued-vehicle 5 mi/h lower bound); release speed = probe wall-speed '
        'range 15-22 m/s; sync_lead_s >= 2 s = t-1 message + join latency (onset = sync_lead_s); '
        'gap law = platoon CACC at its conservative h = 1.4 s. N9 registration.'),
    'harmonize_rows': Spec(
        'harmonize_rows', ('row0', 'row1'), (2, 6),
        (Param('row_speed_mps', 15., 22.), Param('row_spacing_m', 30., 90.)),
        (('row_span_m', 60.), ('zone_far_m', 1800.), ('release_m', 400.), ('track_time_s', 1.)), 'sampled',
        'speed-tracking acceleration bound |a| <= depth (probe wall a = clip((v_w - v)/1 s, -dmax, amax))',
        (), True,
        'row speed = probe wall v_w 15/22 m/s, zone/release/row span '+PROBE+'; row spacing 30-90 m '
        'around the probe 60 m spread. N9 registration.'),
}
ROLES = tuple(sorted({r for s in SPECS.values() for r in s.roles}))
PARAM_SLOTS = max(len(s.params) for s in SPECS.values())
GEOMETRY = ('to_drop_over_2km', 'own_lane_closing', 'own_lane_adjacent', 'own_lane_far', 'own_speed_over_30')
DESCRIPTOR = (tuple(f'is_{n}' for n in FAMILY_NAMES)+('is_initiate', 'is_join')
              + tuple(f'role_{r}' for r in ROLES)+('member_count_over_cap', 'gap_over_range')
              + tuple(f'param_{i}' for i in range(PARAM_SLOTS)))


def split(family):
    """'<name>_<kind>' -> (name, kind)."""
    for kind in KINDS:
        if family.endswith('_'+kind) and family[:-len(kind)-1] in SPECS:
            return family[:-len(kind)-1], kind
    raise ValueError(f'unregistered N9 family {family!r}')


def is_family(family):
    return isinstance(family, str) and family in OPTION_FAMILIES


def registry():
    """JSON registration table (N9_READY / owner review)."""
    out = {}
    for name, s in SPECS.items():
        out[name] = dict(roles=list(s.roles), member_cap=list(s.cap), params=[p.to_dict() for p in s.params],
                         constants=dict(s.constants), depth=s.depth, depth_meaning=s.depth_meaning,
                         onset_window_s=list(ONSET), hold_window_s=list(HOLD), depth_window=list(DEPTH),
                         lane_roles=dict(s.lane_roles), absolute_profile=s.absolute, provenance=s.provenance)
    return out


# ---------------------------------------------------------------- settings

def settings(cfg):
    """Resolved settings, or None when ``nm_families`` is absent/False (default)."""
    raw = (cfg or {}).get(CONFIG_KEY, False)
    if raw is False or raw is None:
        return None
    keys, optional = {'enabled', 'rule', 'provenance', 'map', 'families'}, {'range_m'}
    if (not isinstance(raw, dict) or not keys <= set(raw) or set(raw)-keys-optional
            or raw['enabled'] is not True or raw['rule'] != RULE
            or not isinstance(raw['provenance'], str) or not raw['provenance'].strip()):
        raise ValueError(f'{CONFIG_KEY} requires enabled=True, rule={RULE!r} and provenance')
    if (cfg or {}).get('group_lane_moves') is not True:
        raise ValueError(f'{CONFIG_KEY} requires group_lane_moves=True (feasible slates + G5 lane path)')
    geometry = nm_map.lane_drop(raw['map'])
    families = raw['families']
    if not isinstance(families, dict) or not families or set(families)-set(SPECS):
        raise ValueError(f'{CONFIG_KEY}.families must name registered families {FAMILY_NAMES}')
    caps = {}
    for name in FAMILY_NAMES:
        if name not in families:
            continue
        value = families[name]
        lo, hi = SPECS[name].cap
        if (not isinstance(value, dict) or set(value) != {'member_cap'}
                or type(value['member_cap']) is not int or not lo <= value['member_cap'] <= hi):
            raise ValueError(f'{name} needs member_cap registered in {lo}..{hi}')
        if SPECS[name].lane_roles and not geometry.g5_executable:
            raise ValueError(f'{name} lane changes are not G5-executable on map {geometry.name}')
        if name == 'lane_balance' and geometry.far_lane is None:
            raise ValueError('lane_balance needs a far through lane')
        caps[name] = value['member_cap']
    range_m = raw.get('range_m', V2V_RANGE_M)
    if isinstance(range_m, bool) or not isinstance(range_m, (int, float)) or not (
            math.isfinite(range_m) and range_m > 0):
        raise ValueError(f'{CONFIG_KEY}.range_m must be finite and positive')
    return dict(rule=RULE, provenance=raw['provenance'], map=geometry.name, geometry=geometry,
                caps=caps, families=tuple(caps), range_m=float(range_m))


def config_keys(*, provenance, map_name, families, range_m=None):
    """Exact common B/R binding: ``families`` = {family: member_cap}."""
    out = dict(enabled=True, rule=RULE, provenance=provenance, map=map_name,
               families={k: dict(member_cap=int(v)) for k, v in families.items()})
    if range_m is not None:
        out['range_m'] = float(range_m)
    return {CONFIG_KEY: out}


def _second(value):
    value = float(value)
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError('N9 families run on integer control seconds')
    return int(value)


def _params(spec, params):
    """Canonical sorted ((key, value), ...) after exact registered validation."""
    data = dict(params)
    if set(data) != {p.key for p in spec.params} or any(not p.valid(data[p.key]) for p in spec.params):
        raise ValueError(f'{spec.name} parameters outside the registered ranges')
    return tuple(sorted(data.items()))


def _check_depth(spec, depth):
    if isinstance(depth, bool) or not math.isfinite(float(depth)) or not DEPTH[0] <= float(depth) <= DEPTH[1]:
        raise ValueError('depth outside the registered variety depth window')
    if (spec.depth == 'pinned_lo' and float(depth) != DEPTH[0]) or (
            spec.depth == 'pinned_hi' and float(depth) != DEPTH[1]):
        raise ValueError(f'{spec.name} depth is pinned ({spec.depth})')


# ---------------------------------------------------------------- option / message

@dataclass(frozen=True)
class FamilyOption:
    """One per-CAV option; members is always (ego,). onset/hold relative to the decision second."""
    family: str
    members: tuple
    onset: int
    hold: int
    depth: float
    session_id: str
    initiator: str
    role: str
    params: tuple
    slot: object = None
    gap_m: float = 0.
    member_count: int = 1
    cap: int = 2
    range_m: float = V2V_RANGE_M

    def __post_init__(self):
        object.__setattr__(self, 'members', tuple(self.members))
        name, kind = split(self.family)
        spec = SPECS[name]
        object.__setattr__(self, 'params', _params(spec, self.params))
        if len(self.members) != 1 or not isinstance(self.session_id, str) or not self.session_id \
                or not isinstance(self.initiator, str) \
                or (kind == 'initiate') != (self.initiator == self.members[0]):
            raise ValueError('N9 option: exactly the acting CAV and a consistent initiator/session')
        if self.role not in spec.roles:
            raise ValueError(f'{name} role must be one of {spec.roles}')
        for key in ('onset', 'hold', 'member_count', 'cap'):
            if type(getattr(self, key)) is not int:
                raise ValueError(f'N9 option {key} must be an integer')
        if not moves._in_window(self.onset, *ONSET) or not moves._in_window(self.hold, *HOLD):
            raise ValueError('N9 option outside the registered variety onset/hold windows')
        _check_depth(spec, self.depth)
        if not spec.cap[0] <= self.cap <= spec.cap[1] or not 1 <= self.member_count <= self.cap \
                or (kind == 'join' and self.member_count >= self.cap):
            raise ValueError('N9 option membership outside the registered cap')
        if not (math.isfinite(self.gap_m) and math.isfinite(self.range_m) and self.range_m > 0):
            raise ValueError('N9 option geometry must be finite')

    @property
    def name(self):
        return split(self.family)[0]

    @property
    def kind(self):
        return split(self.family)[1]

    @property
    def onset_s(self):
        return self.onset

    @property
    def hold_s(self):
        return self.hold

    @property
    def actors(self):
        return self.members

    def param(self, key):
        return dict(self.params)[key]

    def to_dict(self):
        return dict(family=self.family, members=list(self.members), onset=int(self.onset),
                    hold=int(self.hold), depth=float(self.depth), session_id=self.session_id,
                    initiator=self.initiator, role=self.role, params=[list(p) for p in self.params],
                    slot=self.slot, gap_m=float(self.gap_m), member_count=int(self.member_count),
                    cap=int(self.cap), range_m=float(self.range_m))

    @classmethod
    def from_dict(cls, data):
        return cls(data['family'], tuple(data['members']), int(data['onset']), int(data['hold']),
                   float(data['depth']), data['session_id'], data['initiator'], data['role'],
                   tuple(tuple(p) for p in data['params']), data.get('slot'), float(data['gap_m']),
                   int(data['member_count']), int(data['cap']), float(data['range_m']))


def option_from_dict(data):
    """Recorded slate entry -> option (N9, else the W1/shared vocabulary)."""
    if is_family(data['family']):
        return FamilyOption.from_dict(data)
    from .v2v_wall import option_from_dict as w1
    return w1(data)


@dataclass(frozen=True)
class FamilyIntent:
    """Initiator broadcast at ``timestamp_s`` (visible only at t+1, within range)."""
    session_id: str
    family: str
    initiator: str
    timestamp_s: int
    source_s_m: float
    source_lane: int
    source_v_mps: float
    anchor_lane: int
    start_s: int
    onset: int
    hold: int
    depth: float
    params: tuple
    roster: tuple                 # ((vid, role, slot), ...) acknowledged by t-1
    cap: int
    released: bool

    @property
    def end_s(self):
        return self.start_s+self.onset+self.hold

    def to_dict(self):
        out = {k: getattr(self, k) for k in self.__dataclass_fields__}
        out.update(params=[list(p) for p in self.params], roster=[list(r) for r in self.roster],
                   source_s_m=float(self.source_s_m), source_v_mps=float(self.source_v_mps),
                   depth=float(self.depth))
        return out


# ---------------------------------------------------------------- local geometry

def _lane(row):
    return int(row['lane'])


def _nearest(rows, own, lane, ahead):
    """Own-sensing nearest vehicle in ``lane`` ahead (True) / behind (False), or None."""
    sign = 1. if ahead else -1.
    best = [r for r in rows if r['vid'] != own['vid'] and _lane(r) == lane
            and sign*(float(r['s_m'])-float(own['s_m'])) > 0
            and abs(float(r['s_m'])-float(own['s_m'])) <= SENSOR_RANGE_M]
    return min(best, key=lambda r: (abs(float(r['s_m'])-float(own['s_m'])), r['vid'])) if best else None


def cacc_accel(gap_m, v, v_pred, v_pred_prev, *, time_headway_s, standstill_gap_m, kp, kd, kff):
    """CACC acceleration (m/s^2) on the 1 s control tick; ``v_pred_prev`` = t-1 V2V packet or None."""
    a = kp*(float(gap_m)-standstill_gap_m-time_headway_s*float(v))+kd*(float(v_pred)-float(v))
    if v_pred_prev is not None:
        a += kff*(float(v_pred)-float(v_pred_prev))/CONTROL_DT_S
    return a


def string_gain(time_headway_s, *, kp=.1, kd=.4, kff=.5, points=4000):
    """Peak |Gamma(e^jw)| (follower speed / predecessor speed) of the discrete CACC loop.

    Model of the registered execution: 1 s ZOH acceleration, trapezoidal gap update,
    feed-forward from the predecessor's one-tick-old packet. <= 1 means string stable.
    """
    w = np.linspace(1e-4, np.pi, points)
    z = np.exp(1j*w)
    gap = .5*(1+z)/(z-1)
    gain = (kp*gap+kd+kff*(1-1/z))/((z-1)+kp*gap+kp*time_headway_s+kd)
    return float(np.abs(gain).max())


def geometry_block(x_row, geometry):
    """Observable trigger context from the recorded own sensor row x[:5] + static map."""
    s, v = float(x_row[0])*1000., float(x_row[1])*30.
    lanes = [float(x) for x in x_row[2:5]]
    one = lambda lane: 0. if lane is None else lanes[lane]    # noqa: E731
    return [max(-1., min(1., geometry.to_drop(s)/2000.)), one(geometry.closing_lane),
            one(geometry.adjacent_lane), one(geometry.far_lane), v/30.]


# ---------------------------------------------------------------- initiate / join rules

def _sample_common(spec, rng):
    params = {p.key: p.sample(rng) for p in spec.params}
    onset, hold, depth = moves._sampled(rng)
    if spec.depth == 'pinned_lo':
        depth = DEPTH[0]
    elif spec.depth == 'pinned_hi':
        depth = DEPTH[1]
    return params, onset, hold, depth


def initiate(name, own, sensed, geometry, *, now, cap, range_m, rng):
    """``<name>_initiate`` for the acting CAV from its OWN state/sensing, or None."""
    spec = SPECS[name]
    if not own.get('is_cav'):
        return None
    params, onset, hold, depth = _sample_common(spec, rng)
    lane, s, v = _lane(own), float(own['s_m']), float(own['v_mps'])
    lane_role, to_drop = geometry.lane_role(lane), geometry.to_drop(s)
    role, slot = None, None
    if name == 'platoon_headway':
        if lane_role not in ('adjacent', 'far'):
            return None
        lead, rear = _nearest(sensed, own, lane, True), _nearest(sensed, own, lane, False)
        if lead is not None and lead['is_cav']:
            role = 'follower'
        elif rear is not None and rear['is_cav']:
            role = 'head'
    elif name == 'early_merge':
        role = 'merge' if lane_role == 'closing' and to_drop > 0 else None
    elif name == 'zipper_order':
        if lane_role in ('closing', 'adjacent') and to_drop > params['zipper_zone_m']:
            role = 'closing' if lane_role == 'closing' else 'through'
    elif name == 'lane_balance':
        if lane_role == 'adjacent' and spec.constant('zone_near_m') <= to_drop <= spec.constant('zone_far_m'):
            role = 'mover'
    elif name == 'discharge_sync':
        lead = _nearest(sensed, own, lane, True)
        if (lane_role in ('adjacent', 'far') and v <= params['queue_speed_mps']
                and abs(to_drop) <= params['zone_m']
                and (lead is None or float(lead['v_mps']) > v)):
            role = 'sync'
            onset = params['sync_lead_s']
    elif name == 'harmonize_rows':
        near, far = spec.constant('release_m'), spec.constant('zone_far_m')
        if lane_role is not None and near < to_drop <= far and v > 0:
            arrival = (to_drop-near)/v
            if arrival >= HOLD[0]:
                onset = min(onset, int(min(ONSET[1], arrival-HOLD[0])))
                hold = min(hold, int(min(HOLD[1], arrival-onset)))
                role, slot = 'row0', lane
    if role is None:
        return None
    ego = own['vid']
    return FamilyOption(f'{name}_initiate', (ego,), int(onset), int(hold), float(depth),
                        session_id=f'{name}:{ego}@{now}', initiator=ego, role=role,
                        params=tuple(sorted(params.items())), slot=slot, cap=cap, range_m=range_m)


def join_role(msg, own, sensed, geometry, *, now):
    """(role, slot) for a receiver of ``msg`` from its OWN row/sensing + the message, or None."""
    spec = SPECS[msg.family]
    params = dict(msg.params)
    lane, s, v = _lane(own), float(own['s_m']), float(own['v_mps'])
    lane_role, to_drop = geometry.lane_role(lane), geometry.to_drop(s)
    roster = {vid: (role, slot) for vid, role, slot in msg.roster}
    counts = {}
    for role, _ in roster.values():
        counts[role] = counts.get(role, 0)+1
    if msg.family == 'platoon_headway':
        if lane != msg.anchor_lane or lane_role not in ('adjacent', 'far'):
            return None
        lead, rear = _nearest(sensed, own, lane, True), _nearest(sensed, own, lane, False)
        if lead is not None and lead['vid'] in roster:
            return 'follower', None
        if rear is not None and rear['vid'] in roster:
            return 'head', None
        return None
    if msg.family == 'early_merge':
        if to_drop <= 0:
            return None
        role = {'closing': 'merge', 'adjacent': 'yield'}.get(lane_role)
        return (role, None) if role else None
    if msg.family == 'zipper_order':
        role = {'closing': 'closing', 'adjacent': 'through'}.get(lane_role)
        if role is None or to_drop <= 0 or counts.get(role, 0) >= (msg.cap+1)//2:
            return None
        return role, None
    if msg.family == 'lane_balance':
        ok = lane_role == 'adjacent' and spec.constant('zone_near_m') <= to_drop <= spec.constant('zone_far_m')
        return ('mover', None) if ok else None
    if msg.family == 'discharge_sync':
        ok = (lane_role in ('adjacent', 'far') and v <= params['queue_speed_mps']
              and abs(to_drop) <= params['zone_m'])
        return ('sync', None) if ok else None
    if msg.family == 'harmonize_rows':
        if lane_role is None or not spec.constant('release_m') < to_drop <= spec.constant('zone_far_m'):
            return None
        source = float(msg.source_s_m)+float(msg.source_v_mps)*(now-msg.timestamp_s)
        half = spec.constant('row_span_m')/2.
        for role, centre in (('row0', source), ('row1', source-params['row_spacing_m'])):
            if abs(s-centre) <= half and (role, lane) not in set(roster.values()):
                return role, lane
        return None
    raise ValueError(f'unhandled family {msg.family!r}')


def lane_target(spec, role, geometry):
    which = spec.lane_target_role(role)
    return None if which is None else (geometry.adjacent_lane if which == 'adjacent' else geometry.far_lane)


# ---------------------------------------------------------------- slate helpers (feasible_proposals)

def slate_families(state):
    """Option families offered in a periodic slate (initiate + join of each enabled family)."""
    return tuple(f'{n}_{k}' for n in state['families'] for k in KINDS)


def join_families(state):
    return tuple(f'{n}_join' for n in state['families'])


def pool(family, state, own, rng):
    """Feasible options of one N9 option family for the acting CAV (local state only)."""
    name, kind = split(family)
    if name not in state['families']:
        return []
    if kind == 'join':
        return [o for o in state.get('offers', ()) if o.family == family and o.members == (own['vid'],)]
    if state.get('only_join'):
        return []
    option = initiate(name, own, state['sensed'], state['geometry'], now=state['now'],
                      cap=state['caps'][name], range_m=state['range_m'], rng=rng)
    return [] if option is None else [option]


def redraw(template, state, own, rng):
    """Slate entry: initiate re-samples its registered parameters, join is exact."""
    if template.kind == 'initiate':
        return initiate(template.name, own, state['sensed'], state['geometry'], now=state['now'],
                        cap=state['caps'][template.name], range_m=state['range_m'], rng=rng)
    return template


# ---------------------------------------------------------------- runtime protocol

class Protocol:
    """Per-episode N9 protocol + executor profiles (actuation ledger; never a policy input)."""

    def __init__(self, family_settings):
        self.settings = dict(family_settings)
        self.geometry = self.settings['geometry']
        self.range_m, self.caps = self.settings['range_m'], dict(self.settings['caps'])
        self.enabled = tuple(self.settings['families'])
        self.FAMILIES = tuple(f'{n}_{k}' for n in self.enabled for k in KINDS)
        self.reset()

    def reset(self):
        self.sessions = {}       # sid -> session ledger
        self.membership = {}     # vid -> sid
        self.board = {}          # sid -> FamilyIntent published at the last publish second
        self.offers = {}
        self.offered = {}
        self.release_latch = set()   # (vid, sid)
        self.native, self.native_time = {}, None
        self.rows, self.sensed, self.prev_reports, self.now = {}, {}, {}, None
        self.last_published, self.last_joins, self.last_commands = [], [], {}

    # ------------------------------------------------ inputs
    def observe_native(self, now, commands):
        """Each CAV's OWN native controller command this second (v2v_runtime hook)."""
        self.native_time = _second(now)
        self.native = {v: float(a) for v, a in (commands or {}).items() if math.isfinite(float(a))}

    def _expire(self, now):
        for sid in [k for k, s in self.sessions.items() if now >= s['start_s']+s['onset']+s['hold']]:
            for vid in self.sessions.pop(sid)['members']:
                if self.membership.get(vid) == sid:
                    del self.membership[vid]

    def receive(self, now, own_rows, sensed, previous_reports):
        """Offers from messages published at now-1 (radio range), own sensing, t-1 packets."""
        now = _second(now)
        self._expire(now)
        self.now = now
        self.rows = {v: dict(r) for v, r in own_rows.items() if r.get('is_cav')}
        self.sensed = {v: [dict(r) for r in rows] for v, rows in sensed.items() if v in self.rows}
        self.prev_reports = dict(previous_reports or {})
        for session in self.sessions.values():          # completions from members' own t-1 packets
            for vid, member in session['members'].items():
                report = self.prev_reports.get(vid)
                if (member['target'] is not None and vid not in session['completed']
                        and _valid_report(report, vid, float(now-1)) and int(report.source.lane) == member['target']):
                    session['completed'][vid] = now-1
        offers = {}
        for vid, own in self.rows.items():
            s, out = float(own['s_m']), []
            for msg in sorted(self.board.values(), key=lambda m: m.session_id):
                if msg.timestamp_s != now-CONTROL_DT_S or msg.initiator == vid:
                    continue
                if abs(msg.source_s_m-s) > self.range_m:
                    continue
                if msg.released:
                    if self.membership.get(vid) == msg.session_id:
                        self.release_latch.add((vid, msg.session_id))
                    continue
                if vid in self.membership or len(msg.roster) >= msg.cap:
                    continue
                begin = msg.start_s+msg.onset
                onset, hold = max(0, begin-now), msg.end_s-max(now, begin)
                if not moves._in_window(hold, *HOLD) or not moves._in_window(onset, *ONSET):
                    continue
                joined = join_role(msg, own, self.sensed.get(vid, ()), self.geometry, now=now)
                if joined is None:
                    continue
                out.append(FamilyOption(f'{msg.family}_join', (vid,), int(onset), int(hold), float(msg.depth),
                    session_id=msg.session_id, initiator=msg.initiator, role=joined[0], params=msg.params,
                    slot=joined[1], gap_m=float(msg.source_s_m-s), member_count=len(msg.roster),
                    cap=int(msg.cap), range_m=self.range_m))
            if out:
                offers[vid] = tuple(sorted(out, key=lambda o: (abs(o.gap_m), o.session_id, o.role)))
        self.offers = offers
        return offers

    def state(self, vid, now):
        """Local view attached to the CAV's candidate state (own sensing + received offers)."""
        return dict(now=_second(now), families=self.enabled, offers=self.offers.get(vid, ()),
                    caps=self.caps, range_m=self.range_m, geometry=self.geometry,
                    sensed=tuple(self.sensed.get(vid, ())))

    def events(self, members, free):
        return {v for v in members if v in free
                and any(o.session_id not in self.offered.get(v, ()) for o in self.offers.get(v, ()))}

    def mark_offered(self, ego, table):
        for move in table:
            if move is not None and is_family(getattr(move, 'family', None)) and move.kind == 'join':
                self.offered.setdefault(ego, set()).add(move.session_id)

    # ------------------------------------------------ arbitration
    def start(self, move, *, ego, executor, observable_cav_ids, now):
        """Atomic session ownership/cap/slot arbitration + executor commitment."""
        option = getattr(move, 'move', move)
        now = _second(now)
        if not isinstance(option, FamilyOption) or option.members != (ego,):
            raise ValueError('N9 option must act through the deciding CAV only')
        if option.name not in self.enabled:
            raise ValueError('N9 family not enabled')
        own = self.rows.get(ego)
        if ego in self.membership or own is None or not own.get('is_cav'):
            return False
        spec = SPECS[option.name]
        target = lane_target(spec, option.role, self.geometry)
        if option.kind == 'initiate':
            if option.session_id in self.sessions or option.session_id != f'{option.name}:{ego}@{now}' \
                    or option.cap != self.caps[option.name]:
                return False
            if not executor.start(move, ego=ego, observable_cav_ids=observable_cav_ids, now=now,
                                  until=now+option.onset+option.hold):
                return False
            self.sessions[option.session_id] = dict(
                family=option.name, initiator=ego, start_s=now, onset=option.onset, hold=option.hold,
                depth=float(option.depth), params=option.params, cap=option.cap, anchor_lane=_lane(own),
                members={ego: dict(t=now, role=option.role, slot=option.slot, target=target)},
                released=False, completed={}, attempts={}, latched=set())
            self.membership[ego] = option.session_id
            return True
        session = self.sessions.get(option.session_id)
        if session is None or session['released'] or len(session['members']) >= session['cap']:
            return False
        end = session['start_s']+session['onset']+session['hold']
        if (now+option.onset+option.hold != end or float(option.depth) != session['depth']
                or option.params != session['params'] or option.name != session['family']):
            raise ValueError('join option schedule/parameters differ from the session it names')
        roles = [(m['role'], m['slot']) for m in session['members'].values()]
        if option.name == 'harmonize_rows' and (option.role, option.slot) in roles:
            return False                                   # one member per (row, lane) slot
        if option.name == 'zipper_order' and sum(r == option.role for r, _ in roles) >= (session['cap']+1)//2:
            return False
        if not executor.start(move, ego=ego, observable_cav_ids=observable_cav_ids, now=now, until=end):
            return False
        session['members'][ego] = dict(t=now, role=option.role, slot=option.slot, target=target)
        self.membership[ego] = option.session_id
        self.last_joins.append(dict(session_id=option.session_id, vid=ego, time_s=now, role=option.role))
        return True

    def open_offline(self, option, now, anchor_lane):
        """Offline (bank) reconstruction of an accepted initiation; no executor."""
        spec = SPECS[option.name]
        self.sessions[option.session_id] = dict(
            family=option.name, initiator=option.initiator, start_s=_second(now), onset=option.onset,
            hold=option.hold, depth=float(option.depth), params=option.params, cap=option.cap,
            anchor_lane=int(anchor_lane),
            members={option.initiator: dict(t=_second(now), role=option.role, slot=option.slot,
                                            target=lane_target(spec, option.role, self.geometry))},
            released=False, completed={}, attempts={}, latched=set())
        self.membership[option.initiator] = option.session_id

    def admit_offline(self, option, now):
        session = self.sessions.get(option.session_id)
        vid = option.members[0]
        if (session is None or session['released'] or len(session['members']) >= session['cap']
                or vid in self.membership):
            return False
        session['members'][vid] = dict(t=_second(now), role=option.role, slot=option.slot,
                                       target=lane_target(SPECS[option.name], option.role, self.geometry))
        self.membership[vid] = option.session_id
        return True

    # ------------------------------------------------ broadcast
    def _released(self, session, source):
        spec, params = SPECS[session['family']], dict(session['params'])
        to_drop = self.geometry.to_drop(source.s_m)
        family = session['family']
        if family in ('early_merge', 'zipper_order'):
            return to_drop <= 0
        if family == 'lane_balance':
            return to_drop < spec.constant('zone_near_m')
        if family == 'discharge_sync':
            return float(source.v_mps) >= params['release_speed_mps']
        if family == 'harmonize_rows':
            return to_drop <= spec.constant('release_m')
        return False

    def publish(self, now, sources):
        """Initiators broadcast at ``now``; sources = {cav: own ObservedVehicle at now}."""
        now = _second(now)
        self._expire(now)
        board = {}
        for sid, s in sorted(self.sessions.items()):
            source = sources.get(s['initiator'])
            if s['released'] or source is None:
                continue
            released = self._released(s, source)
            roster = tuple((v, m['role'], m['slot']) for v, m in sorted(s['members'].items())
                           if v == s['initiator'] or m['t'] <= now-1)
            board[sid] = FamilyIntent(sid, s['family'], s['initiator'], now, float(source.s_m),
                int(source.lane), float(source.v_mps), int(s['anchor_lane']), s['start_s'], s['onset'],
                s['hold'], s['depth'], s['params'], roster, s['cap'], released)
            if released:
                s['released'] = True
        self.board = board
        self.last_published = [m.to_dict() for m in board.values()]
        return board

    # ------------------------------------------------ executor profiles
    def _packets(self, session, vid, now):
        """Teammates' own t-1 V2V packets received by ``vid`` (radio range)."""
        own = float(self.rows[vid]['s_m'])
        out = {}
        for mate in session['members']:
            report = self.prev_reports.get(mate)
            if (mate != vid and _valid_report(report, mate, float(now-1))
                    and abs(float(report.source.s_m)-own) <= self.range_m):
                out[mate] = report.source
        return out

    def _known(self, session, now):
        return {v for v, m in session['members'].items() if v == session['initiator'] or m['t'] <= now-1}

    def _sequence_ready(self, session, vid, now, role, lane, packets):
        """Front-first sequencing among same-role lane changers + registered gap/attempts."""
        params = dict(session['params'])
        own = self.rows[vid]
        for mate, src in packets.items():
            m = session['members'][mate]
            if (m['role'] == role and mate not in session['completed'] and int(src.lane) == lane
                    and float(src.s_m) > float(own['s_m'])):
                return 'sequence_wait'
        last = max(session['completed'].values(), default=None)
        if last is not None and now-last < params['sequence_gap_s']:
            return 'sequence_gap_wait'
        return None

    def _profile(self, session, vid, now):
        """(kind, value, lane_target, status); kind in (None, 'offset', 'absolute')."""
        family, spec = session['family'], SPECS[session['family']]
        params, member = dict(session['params']), session['members'][vid]
        own, sensed = self.rows[vid], self.sensed.get(vid, ())
        elapsed = now-session['start_s']
        if (vid, self.membership.get(vid)) in self.release_latch or vid in session['latched']:
            return None, 0., None, 'released'
        if not session['onset'] <= elapsed < session['onset']+session['hold']:
            return None, 0., None, 'outside_window'
        lane, s, v = _lane(own), float(own['s_m']), float(own['v_mps'])
        to_drop = self.geometry.to_drop(s)
        packets = self._packets(session, vid, now)
        depth = float(session['depth'])
        if family == 'platoon_headway':
            lead = _nearest(sensed, own, lane, True)
            if lane != session['anchor_lane'] or lead is None or not lead['is_cav'] \
                    or lead['vid'] not in self._known(session, now):
                return None, 0., None, 'no_member_predecessor'
            previous = packets.get(lead['vid'])
            a = cacc_accel(float(lead['s_m'])-s-float(lead['length_m']), v, float(lead['v_mps']),
                           None if previous is None else previous.v_mps,
                           time_headway_s=params['time_headway_s'], **dict(spec.constants))
            return 'absolute', a, None, 'cacc' if previous is not None else 'cacc_no_feedforward'
        if family in ('early_merge', 'lane_balance'):
            role = member['role']
            if role in ('merge', 'mover'):
                source_lane = self.geometry.closing_lane if role == 'merge' else self.geometry.adjacent_lane
                if lane != source_lane:
                    return None, 0., None, 'lane_done'
                if family == 'early_merge' and to_drop > params['upstream_m']:
                    return None, 0., None, 'upstream_wait'
                if family == 'lane_balance' and not (
                        spec.constant('zone_near_m') <= to_drop <= spec.constant('zone_far_m')):
                    return None, 0., None, 'outside_zone'
                wait = self._sequence_ready(session, vid, now, role, source_lane, packets)
                if wait:
                    return None, 0., None, wait
                if session['attempts'].get(vid, 0) >= params['attempt_s']:
                    return None, 0., None, 'attempts_exhausted'
                return None, 0., member['target'], 'lane_request'
            # early_merge yield role: leave room for a pending merger ahead within link_m.
            link = spec.constant('link_m')
            pending = [m for m, src in packets.items() if session['members'][m]['role'] == 'merge'
                       and int(src.lane) == self.geometry.closing_lane
                       and -float(own['length_m']) <= float(src.s_m)-s <= link]
            return ('offset', -depth, None, 'yield') if pending else (None, 0., None, 'no_pending_merger')
        if family == 'zipper_order':
            positions = {m: (float(src.s_m)+float(src.v_mps)*CONTROL_DT_S, float(src.length_m))
                         for m, src in packets.items()}
            positions[vid] = (s, float(own['length_m']))
            order = params['order']
            first = 'through' if order == 'through_first' else 'closing'
            second = 'closing' if first == 'through' else 'through'
            queues = {r: sorted((m for m in positions if session['members'][m]['role'] == r),
                                key=lambda m: (-positions[m][0], m)) for r in (first, second)}
            sequence = []
            for i in range(max(len(q) for q in queues.values())):
                sequence += [q[i] for q in (queues[first], queues[second]) if i < len(q)]
            index = sequence.index(vid)
            pred = sequence[index-1] if index else None
            foll = sequence[index+1] if index+1 < len(sequence) else None
            need = spec.constant('standstill_gap_m')+params['order_headway_s']*v
            behind = pred is None or positions[pred][0]-s-positions[pred][1] >= need
            ahead = foll is None or s-positions[foll][0]-float(own['length_m']) >= spec.constant('standstill_gap_m')
            offset = -depth if not behind else 0.
            if member['role'] == 'closing' and lane == self.geometry.closing_lane:
                pred_done = pred is None or session['members'][pred]['role'] != 'closing' \
                    or pred in session['completed']
                if (to_drop <= params['zipper_zone_m'] and behind and ahead and pred_done
                        and session['attempts'].get(vid, 0) < params['attempt_s']):
                    return ('offset' if offset else None), offset, member['target'], 'lane_request'
            return ('offset', offset, None, 'order_ease') if offset else (None, 0., None, 'in_order')
        if family == 'discharge_sync':
            if v >= params['release_speed_mps']:
                session['latched'].add(vid)
                return None, 0., None, 'released'
            lead = _nearest(sensed, own, lane, True)
            a = depth
            if lead is not None:
                gap = float(lead['s_m'])-s-float(lead['length_m'])
                a = min(a, cacc_accel(gap, v, float(lead['v_mps']), None, kff=0.,
                                      **{k: spec.constant(k) for k in ('time_headway_s', 'standstill_gap_m',
                                                                        'kp', 'kd')}))
            return 'absolute', a, None, 'sync'
        if family == 'harmonize_rows':
            if to_drop <= spec.constant('release_m'):
                session['latched'].add(vid)
                return None, 0., None, 'released'
            a = (params['row_speed_mps']-v)/spec.constant('track_time_s')
            return 'absolute', max(-depth, min(depth, a)), None, 'track'
        raise ValueError(f'unhandled family {family!r}')

    def commands(self, now, active, *, base_mean=None, reserved=()):
        """Per-member committed offsets + lane requests for this second.

        ``active`` = executor entries' (vid, option) of THIS protocol. ``base_mean()`` returns
        the frozen base policy mean residual by vid (called only for absolute profiles);
        the absolute request is ``native + base_mean + offset``, so offset = a - native -
        base_mean. Missing native command or controller-reserved members -> no offset, no lane.
        """
        now = _second(now)
        offsets, lanes, out = {}, {}, {}
        means = None
        for vid, option in active:
            session = self.sessions.get(option.session_id)
            if session is None or vid not in session['members'] or vid not in self.rows:
                out[vid] = dict(status='no_session', offset=0.)
                continue
            if vid in reserved:
                kind, value, lane, status = None, 0., None, 'controller_reserved'
            else:
                kind, value, lane, status = self._profile(session, vid, now)
            offset = 0.
            if kind == 'offset':
                offset = float(value)
            elif kind == 'absolute':
                if self.native_time != now or vid not in self.native:
                    status = 'native_unavailable'
                else:
                    if means is None:
                        means = base_mean() if base_mean is not None else {}
                    if vid not in means:
                        status = 'base_unavailable'
                    else:
                        offset = float(value)-self.native[vid]-float(means[vid])
            if not math.isfinite(offset):
                raise ValueError('nonfinite N9 offset')
            offsets[vid] = offset
            if lane is not None:
                lanes[vid] = int(lane)
                session['attempts'][vid] = session['attempts'].get(vid, 0)+1
            out[vid] = dict(session_id=option.session_id, family=session['family'],
                            role=session['members'][vid]['role'], status=status, offset=offset,
                            target=(float(value) if kind == 'absolute' else None), lane=lane)
        self.last_commands = out
        return offsets, lanes

    def receipt(self):
        out = dict(rule=RULE, map=self.geometry.name, published=self.last_published,
                   offers={v: [o.session_id for o in opts] for v, opts in sorted(self.offers.items())},
                   joins=list(self.last_joins), commands=dict(sorted(self.last_commands.items())),
                   sessions=[dict(session_id=k, family=s['family'], initiator=s['initiator'],
                                  members={v: m['role'] for v, m in sorted(s['members'].items())},
                                  completed=dict(sorted(s['completed'].items())), released=s['released'])
                             for k, s in sorted(self.sessions.items())])
        self.last_joins = []
        return out


# ---------------------------------------------------------------- trigger extension

class TriggerExtension:
    """N9 trigger inputs; every extra first-layer column is zero-initialised by TriggerPolicy."""

    wall = None
    aware = None

    def __init__(self, family_settings):
        self.nm = family_settings
        self.geometry = family_settings['geometry']
        self.families = tuple(f'{n}_{k}' for n in family_settings['families'] for k in KINDS)
        self.descriptor_width = len(DESCRIPTOR)
        self.context_width = len(GEOMETRY)

    def descriptor(self, move):
        family = None if move is None else (move['family'] if isinstance(move, dict) else move.family)
        if not is_family(family):
            return [0.]*self.descriptor_width
        get = (lambda k: move[k]) if isinstance(move, dict) else (lambda k: getattr(move, k))
        name, kind = split(family)
        spec, params = SPECS[name], dict(tuple(p) for p in get('params'))
        out = [float(n == name) for n in FAMILY_NAMES]+[float(kind == k) for k in KINDS]
        out += [float(get('role') == r) for r in ROLES]
        out += [int(get('member_count'))/int(get('cap')), max(-1., min(1., float(get('gap_m'))/float(get('range_m'))))]
        values = [p.normalised(params[p.key]) for p in spec.params]
        return out+values+[0.]*(PARAM_SLOTS-len(values))

    def context_features(self, context):
        rows = [geometry_block(x, self.geometry) for x in context['x']]
        out = torch.tensor(rows, dtype=torch.float32).reshape(len(rows), self.context_width)
        if not bool(torch.isfinite(out).all()):
            raise ValueError('nonfinite N9 trigger context')
        return out


def extension(cfg):
    value = settings(cfg)
    return None if value is None else TriggerExtension(value)


class Composite:
    """Several trigger extensions side by side (descriptor blocks, then context blocks)."""

    def __init__(self, parts):
        self.parts = tuple(parts)
        self.families = tuple(f for p in self.parts for f in p.families)
        if len(set(self.families)) != len(self.families):
            raise ValueError('extension option families overlap')
        self.descriptor_width = sum(p.descriptor_width for p in self.parts)
        self.context_width = sum(p.context_width for p in self.parts)
        self.wall = next((p.wall for p in self.parts if getattr(p, 'wall', None) is not None), None)
        self.aware = next((p.aware for p in self.parts if getattr(p, 'aware', None) is not None), None)
        self.nm = next((p.nm for p in self.parts if getattr(p, 'nm', None) is not None), None)

    def descriptor(self, move):
        return [x for p in self.parts for x in p.descriptor(move)]

    def context_features(self, context):
        blocks = [p.context_features(context) for p in self.parts if p.context_width]
        return torch.cat(blocks, 1) if blocks else torch.zeros((len(context['members']), 0))


def compose(*parts):
    """None/one part unchanged (byte-identical); otherwise a Composite."""
    parts = [p for p in parts if p is not None]
    return None if not parts else parts[0] if len(parts) == 1 else Composite(parts)


# ---------------------------------------------------------------- executable fields (G5 convention)

_FORBIDDEN = ('knots', 'knot_s', 'perturbation_knots', 'perturbation_canonical', 'g5_move', 'lane_actions')


def _matrix(value, shape, name):
    if not isinstance(value, (list, tuple)) or len(value) != shape[0] or any(
            not isinstance(row, (list, tuple)) or len(row) != shape[1]
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in row)
            for row in value):
        raise ValueError(f'N9 {name} must be a finite members x duration_s matrix')
    return np.asarray(value, float)


def nm_move(session, *, map_name, t0):
    """Executable session record (``nm_move``) from a protocol session ledger."""
    initiator = session['initiator']
    members = [initiator]+sorted((v for v in session['members'] if v != initiator),
                                 key=lambda v: (session['members'][v]['t'], v))
    return dict(schema=EXECUTABLE_SCHEMA, rule=RULE, family=session['family'], map=map_name,
                initiator=initiator, members=members, onset=int(session['onset']), hold=int(session['hold']),
                depth=float(session['depth']), params=dict(session['params']), cap=int(session['cap']),
                anchor_lane=int(session['anchor_lane']),
                roles={v: session['members'][v]['role'] for v in members},
                slots={v: session['members'][v]['slot'] for v in members},
                join_s={v: int(session['members'][v]['t'])-int(t0) for v in members})


def executable_fields(move, duration, *, policy_profile, profile, lane_targets):
    """Plan fields (G5 convention): absolute executed ``profile`` bound to the immutable
    ``policy_profile`` reference; ``perturbation_profile`` is exactly their difference;
    ``lane_targets`` (members x seconds, -1 = no request) are the G5 lane requests issued.
    The N9 laws are closed-loop, so ``profile`` is the EXECUTED tape, not a macro."""
    shape = (len(move['members']), int(duration))
    base = _matrix(policy_profile, shape, 'policy_profile')
    executed = _matrix(profile, shape, 'profile')
    lanes = _matrix(lane_targets, shape, 'lane_targets')
    plan = dict(members=list(move['members']), duration_s=int(duration), nm_move=deepcopy(move),
                policy_profile=base.tolist(), profile=executed.tolist(),
                perturbation_profile=(executed-base).tolist(),
                lane_targets=lanes.astype(int).tolist(), uses_lane_actions=bool((lanes >= 0).any()))
    from_plan(plan)
    return plan


def from_plan(plan, geometry=None):
    """Validate an N9 executable plan exactly; returns the ``nm_move`` dict."""
    move = plan.get('nm_move')
    if not isinstance(move, dict) or move.get('schema') != EXECUTABLE_SCHEMA or move.get('rule') != RULE:
        raise ValueError('N9 plan requires an explicit executable nm_move')
    if any(key in plan for key in _FORBIDDEN):
        raise ValueError('N9 plan carries a conflicting representation')
    name = move.get('family')
    if name not in SPECS:
        raise ValueError('unregistered N9 family')
    spec = SPECS[name]
    geometry = geometry or nm_map.lane_drop(move['map'])
    members = list(move['members'])
    if (plan.get('members') != members or not members or members[0] != move['initiator']
            or len(set(members)) != len(members) or not spec.cap[0] <= move['cap'] <= spec.cap[1]
            or len(members) > move['cap']):
        raise ValueError('N9 plan member/initiator/cap binding mismatch')
    _params(spec, move['params'].items())
    _check_depth(spec, move['depth'])
    if type(move['onset']) is not int or type(move['hold']) is not int or not moves._in_window(
            move['onset'], *ONSET) or not moves._in_window(move['hold'], *HOLD):
        raise ValueError('N9 plan onset/hold outside the registered windows')
    if set(move['roles']) != set(members) or any(r not in spec.roles for r in move['roles'].values()):
        raise ValueError('N9 plan roles must cover members with registered roles')
    if set(move['join_s']) != set(members) or move['join_s'][members[0]] != 0 or any(
            type(t) is not int or t < 0 for t in move['join_s'].values()):
        raise ValueError('N9 plan join seconds invalid')
    duration = plan.get('duration_s')
    if type(duration) is not int or duration <= 0:
        raise ValueError('N9 plan needs an integer duration_s')
    shape = (len(members), duration)
    base = _matrix(plan.get('policy_profile'), shape, 'policy_profile')
    executed = _matrix(plan.get('profile'), shape, 'profile')
    stored = _matrix(plan.get('perturbation_profile'), shape, 'perturbation_profile')
    if not np.allclose(executed-base, stored, atol=1e-8, rtol=0):
        raise ValueError('N9 profile does not equal policy reference plus perturbation')
    lanes = _matrix(plan.get('lane_targets'), shape, 'lane_targets')
    begin, end = move['onset'], move['onset']+move['hold']
    for i, vid in enumerate(members):
        target = lane_target(spec, move['roles'][vid], geometry)
        used = [t for t in range(duration) if lanes[i, t] >= 0]
        if used and (target is None or any(lanes[i, t] != target for t in used)
                     or any(t < move['join_s'][vid] or not begin <= t < end for t in used)
                     or len(used) > move['params'].get('attempt_s', 0)):
            raise ValueError('N9 plan lane requests outside the registered role/window/attempts')
        if any(lanes[i, t] not in (-1, target) for t in range(duration)):
            raise ValueError('N9 plan lane targets must be -1 or the role target lane')
    if plan.get('uses_lane_actions') is not bool((lanes >= 0).any()):
        raise ValueError('N9 plan uses_lane_actions mismatch')
    from .plans import acceleration_at
    for t in range(duration):
        got = acceleration_at(plan, t)
        if set(got) != set(members) or any(abs(got[v]-executed[i, t]) > 1e-8 for i, v in enumerate(members)):
            raise ValueError('N9 executor does not run the bound executable profile')
    return move


def lane_requests(plan, t):
    """{vid: WM target} recorded for plan second ``t`` (G5 replay input)."""
    move = from_plan(plan)
    return {v: int(plan['lane_targets'][i][t]) for i, v in enumerate(move['members'])
            if 0 <= t < plan['duration_s'] and plan['lane_targets'][i][t] >= 0}


def verify_tape(plan, tape, t0):
    """Admission check on a hash-bound verified tape (mirrors group_lane.verify_tape).

    Every covered member-second executed exactly the bound ``profile``; every member
    with recorded lane requests shows, at one of its request seconds, the G5 request,
    gate pass, per-vehicle actuator telemetry (issued + completed) and the next sensed
    lane equal to the target. Blocked/failed attempts stay diagnostics, never lessons.
    """
    move = from_plan(plan)
    members, duration = move['members'], plan['duration_s']
    steps = {int(s['time_s']): s for s in tape['steps']}
    covered = 0
    for t in range(duration):
        step = steps.get(int(t0)+t)
        if step is None:
            continue
        executed = (step.get('sample') or {}).get('absolute_accels') or {}
        for i, vid in enumerate(members):
            if vid not in executed or abs(float(executed[vid])-plan['profile'][i][t]) > 1e-8:
                raise ValueError('N9 verified tape did not execute the bound executable profile')
        covered += 1
    if covered < min(duration, move['onset']+move['hold']):
        raise ValueError('N9 verified tape does not cover the executed window')
    for i, vid in enumerate(members):
        seconds = [t for t in range(duration) if plan['lane_targets'][i][t] >= 0]
        if not seconds:
            continue
        target, ok = plan['lane_targets'][i][seconds[0]], False
        for t in seconds:
            before, after = steps.get(int(t0)+t), steps.get(int(t0)+t+1)
            if before is None or after is None:
                continue
            sample = before.get('sample') or {}
            own = ((before.get('g5_lane_outcome') or {}).get('vehicles') or {}).get(vid)
            pre = {r['vid']: r for r in before.get('rows', ())}
            post = {r['vid']: r for r in after.get('rows', ())}
            if ((sample.get('g5_lane_requests') or {}).get(vid) == target
                    and ((sample.get('g5_lane_gate') or {}).get('safe') or {}).get(vid) == target
                    and own == dict(target=target, status='issued', completed=True)
                    and pre.get(vid, {}).get('is_cav') and post.get(vid, {}).get('is_cav')
                    and pre[vid]['lane'] != target and post[vid]['lane'] == target):
                ok = True
                break
        if not ok:
            raise ValueError('N9 verified tape lacks the commanded CAV lane effect')
    return move


# ---------------------------------------------------------------- bank ingestion (initiate/join labels)

def protocol_contexts(policy, plan, pairs, frames, *, t0):
    """Initiate/join imitation contexts for one verified N9 example (mirrors W1).

    The plan's recorded initiator imitates ``<family>_initiate`` at its first usable
    observed second; every other member imitates ``<family>_join`` at the earliest
    second ITS OWN receive step produced an offer for the reconstructed t-1 message
    with the plan's role. Inputs: own sensing, own t-1 packets, received message only.
    """
    move = from_plan(plan, policy.nm_families.geometry)
    proto = policy.nm_families
    if move['family'] not in proto.enabled:
        return
    params = tuple(sorted(move['params'].items()))
    by_t = {}
    for sample, weight in sorted(pairs, key=lambda pair: int(pair[0]['intent_link']['t'])):
        by_t.setdefault(int(sample['intent_link']['t']), (sample, weight))
    reader = deepcopy(policy)
    reader.nm_families.reset()
    t0, initiator = int(t0), move['initiator']
    remaining, sid = [v for v in move['members'] if v != initiator], None
    cap = proto.caps[move['family']]
    if len(move['members']) > cap:
        return

    def build(source, keep):
        context = dict(source, members=list(keep),
                       x=[source['x'][source['members'].index(v)] for v in keep],
                       trigger_v2v={v: reader.local_intents[v] for v in keep})
        from .v2v_wall import attach_context
        return attach_context(context, reader, keep)

    for t in range(t0-1, t0+move['onset']+move['hold']):
        if t not in frames:
            continue
        reader.observe(frames[t]['vehicles'], t)
        if t < t0:
            continue
        elapsed = t-t0
        left_onset, left_hold = max(0, move['onset']-elapsed), move['hold']-max(0, elapsed-move['onset'])
        if left_hold < HOLD[0]:
            break
        pair = by_t.get(t)
        if sid is None:
            if pair is None or initiator not in pair[0]['context']['members'] \
                    or initiator not in reader.local_intents or initiator not in reader.nm_families.rows:
                continue
            sid = f'{move["family"]}:{initiator}@{t}'
            option = FamilyOption(f'{move["family"]}_initiate', (initiator,), left_onset, left_hold,
                                  float(move['depth']), session_id=sid, initiator=initiator,
                                  role=move['roles'][initiator], params=params,
                                  slot=move['slots'].get(initiator), cap=cap, range_m=proto.range_m)
            yield build(pair[0]['context'], [initiator]), option, reader, float(pair[1]), t
            reader.nm_families.open_offline(option, t, move['anchor_lane'])
        elif pair is not None and remaining:
            source = pair[0]['context']
            for vid in list(remaining):
                if vid not in source['members'] or vid not in reader.local_intents:
                    continue
                offer = next((o for o in reader.nm_families.offers.get(vid, ())
                              if o.session_id == sid and o.role == move['roles'][vid]), None)
                if offer is None or not reader.nm_families.admit_offline(offer, t):
                    continue
                remaining.remove(vid)
                yield build(source, [vid]), offer, reader, float(pair[1]), t
        if sid is not None:
            reader.nm_families.publish(t, {v: r.source for v, r in reader.reports.items()})
            if not remaining:
                break


def label_family(option):
    """Bank supervision family key (initiate/join of each N9 family)."""
    return option.family


def nominal_offsets(option, t):
    """Stateless executor stub (MoveExecutor.requests ABI; unused by the production path)."""
    return {v: 0. for v in option.members}


__all__ = ['RULE', 'FAMILY_NAMES', 'OPTION_FAMILIES', 'SPECS', 'settings', 'config_keys', 'FamilyOption',
           'FamilyIntent', 'Protocol', 'TriggerExtension', 'Composite', 'compose', 'extension', 'pool',
           'redraw', 'slate_families', 'join_families', 'executable_fields', 'from_plan', 'nm_move',
           'protocol_contexts', 'string_gain', 'cacc_accel', 'registry', 'option_from_dict']
