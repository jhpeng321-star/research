"""LCEXEC free_program: per-member programs of CAV primitives (clean5, DEFAULT OFF).

A free_program is the general cooperation option: every member CAV runs ONE member
program built from the CAV-only primitives the SUMO probes used (NEWMAP2, PLANGEN,
PLANGEN2, COOPKINDS, LOWCAV, LCMOVES):

* longitudinal (``lon``): ``base`` (base controller keeps the command), ``native`` (SUMO's
  own car-following, no speed command), ``speed`` (target speed: a = clip((v_w - v)/1 s,
  -dmax, amax)), ``track`` (reference slot: a = clip(kp*(mean ref s + off - s) +
  kv*(mean ref v - v), -dmax, amax), optional settle into ``speed``), ``match`` (target-lane
  mean speed of sensed vehicles within 100 m, until the lane change is done), ``accel``
  (constant acceleration);
* lateral (``lane``): a persistent lane request (target WM lane, armed from a second and/or
  a position, give-up position, request rule), re-issued every control second as
  ``changeLane(lane, duration_s)`` until the rule stops it;
* ``lc_mode`` (513 policy / 512 TraCI-only) and an own time gap ``tau`` set when the member
  activates and restored at release;
* release conditions in a registered order (hold end, release position, anchor, tau zone,
  partner left its lane, merger, fallback position).

Units/frame: WM frame of the N1 map descriptor (s in m, WM lanes 0 = far/left ... 2 =
closing lane on wz3to2). ``el`` = whole control seconds since the move start t0.

The executor (:class:`MemberExecutor`, :func:`step_members`) is a pure function of the
member's OWN observation (own row + own sensing within 100 m, or a teammate's own t-1 V2V
packet for references); it never reads truth, global metrics or another arm's state. Its
per-second output is a list of primitive commands ((kind, value) pairs); the env writer
(lc_runtime) turns them into the identical TraCI calls the probe Arm issued:

    ('lcMode', m)            -> vehicle.setLaneChangeMode(vid, m)   (first activation)
    ('tau', x)               -> vehicle.setTau(vid, x)              (first activation)
    ('accel', a)             -> vehicle.slowDown(vid, min(max(0, v + a*1 s), allowed), 1 s)
                                via the single longitudinal writer (CommandOwnership)
    ('changeLane', (L, d))   -> vehicle.changeLane(vid, sumo index of WM lane L on the CAV's
                                current edge, d)
    ('release', reason)      -> hand back: setSpeed(vid, -1) (native base) / base controller,
                                restore the saved lcMode and tau
A member that is active without an ('accel', a) command this second is longitudinally
``native`` (the probe issued no slowDown); ``base`` keeps the base controller's command.

Per-tick imitation labels: :func:`tick_label` (schema ``lcexec-tick-label-v1``).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import math

SCHEMA = 'lcexec-free-program-v1'
LABEL_SCHEMA = 'lcexec-tick-label-v1'
RULE = 'lcexec-free-program-v1'
LC_MODE_POLICY = 513             # probes + env LC_MODE_POLICY
LC_MODE_TRACI_ONLY = 512         # COOPKINDS/PLANGEN2 zip_merge
LANE_CHANGE_DURATION_S = 2.0     # every probe: changeLane(vid, lane, 2.0)
CONTROL_DT_S = 1.0
SENSING_M = 100.0                # own sensing radius (merge_features.SENSOR_RANGE_M; LCMOVES match rng)

LON_KINDS = ('base', 'native', 'speed', 'track', 'match', 'accel')
LANE_RULES = ('until_reached',   # LCMOVES prog: request while lane != target, latch done on reaching it
              'right_of_target',  # COOPKINDS lane_to: request while own lane is right of the target
              'from_lane')        # merge_early / zip_merge: request while in source_lane
MODE_ON = ('first_active',       # legacy roles: lcMode/tau at the first active second
           'on_action',          # LCMOVES prog: when the lane request arms or the speed action starts
           'never')              # pulse: no lcMode
RELEASES = ('hold_end', 'release_s', 'anchor', 'zone', 'partner_left', 'merged_zip', 'merger', 'fallback')

# Registered value sources (cite; FULLPOL's generator samples inside these, never outside).
PROBES = {
    'NEWMAP2': 'runs/active/C4-V7FT-20261017/newmap2/code/nm2_plan.py (VARIANTS/R_GRID; NEWMAP2 REGISTRATION)',
    'PLANGEN2': 'runs/active/C4-V7FT-20261017/plangen2/code/pg2_plan.py (VARIANTS, R_GRID, CK_B, CK_R)',
    'COOPKINDS': 'runs/active/C4-V7FT-20261017/coopkinds/code/ck_plan.py (B_PARAMS / R_GRID, registered 15:06)',
    'LCMOVES': 'runs/active/C4-V7FT-20261017/lcmoves/code/lcm_core.py role prog (S-LCMOVES registration)',
}
X_B_PROBE = 2696.0               # probe global x of the drop (end of SUMO up_0)


@dataclass(frozen=True)
class Range:
    lo: float
    hi: float
    values: tuple = ()           # registered grid (empty = continuous in [lo, hi])
    source: str = ''

    def ok(self, x):
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
            return False
        return self.lo - 1e-9 <= float(x) <= self.hi + 1e-9


# Union of the probe registrations (values observed in the registered grids). LCMOVES ranges
# are added from its registration when it lands (``extend_registry``); until then the prog
# role defaults (amax 1.0, kp 0.15, kv 0.6, lc_giveup X_B-300) are the registered values.
REGISTRY = {
    'v_w': Range(15., 22., (15., 18., 22.), 'NEWMAP2/PLANGEN2 VARIANTS v_w {15,22}, R_GRID {15,18,22}'),
    'dmax': Range(1., 1.5, (1., 1.5), 'NEWMAP2/PLANGEN2 dmax {1.0,1.5}; COOPKINDS zip_yield dmax 1.5'),
    'amax': Range(1., 1., (1.,), 'NEWMAP2 REL amax 1.0; COOPKINDS zip_yield amax 1.0; LCMOVES prog default'),
    'hold_s': Range(2, 300, (90, 120, 150, 180, 240, 300),
                    'NEWMAP2 hold {90,120,150,180}; COOPKINDS hold 240/300; dither pulse dur 2'),
    'tau': Range(.5, 2.4, (.5, .6, .8, 1.6, 2.0, 2.4),
                 'COOPKINDS ck_plan B_PARAMS/R_GRID: platoon/discharge tau B 0.6, R {0.5,0.6,0.8}; '
                 'early_merge room_tau B 2.0, R {1.6,2.0,2.4} (PLANGEN2 CK_B/CK_R)'),
    'kp': Range(.15, .15, (.15,), 'COOPKINDS zip_yield kp 0.15; LCMOVES track default'),
    'kv': Range(.6, .6, (.6,), 'COOPKINDS zip_yield kv 0.6; LCMOVES track default'),
    'track_off_m': Range(-35., 0., (-35., -25., -15.), 'COOPKINDS zipper gap {15,25,35} (off = -gap)'),
    'accel': Range(-1., -1., (-1.,), 'NEWMAP2 dither pulse a = -1.0'),
    'lane_duration_s': Range(2., 2., (2.,), 'probe changeLane duration 2.0 s'),
    'lc_mode': Range(512, 513, (512, 513), 'probe LC_MODE_POLICY 513 / LC_MODE_TRACI_ONLY 512'),
}


def extend_registry(key, rng):
    """Add/replace a registered range (LCMOVES registration landing); returns the old one."""
    if not isinstance(rng, Range) or not rng.source:
        raise ValueError('registry entries need a Range with a cited source')
    old = REGISTRY.get(key)
    REGISTRY[key] = rng
    return old


def _finite(x, name):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise ValueError(f'free_program {name} must be finite')
    return float(x)


def _opt(x, name):
    return None if x is None else _finite(x, name)


@dataclass(frozen=True)
class LaneReq:
    target: int                   # WM lane
    rule: str = 'until_reached'
    source_lane: int | None = None  # from_lane: the lane the member must still be in
    from_el: int = 0
    from_s: float | None = None
    giveup_s: float | None = None
    duration_s: float = LANE_CHANGE_DURATION_S

    def __post_init__(self):
        if type(self.target) is not int or self.target < 0 or self.rule not in LANE_RULES:
            raise ValueError('lane request needs an integer WM target and a registered rule')
        if self.rule == 'from_lane' and type(self.source_lane) is not int:
            raise ValueError('from_lane rule needs the source WM lane')
        if type(self.from_el) is not int or self.from_el < 0:
            raise ValueError('lane from_el must be a non-negative integer second')
        _opt(self.from_s, 'lane.from_s')
        _opt(self.giveup_s, 'lane.giveup_s')
        if not _finite(self.duration_s, 'lane.duration_s') > 0:
            raise ValueError('lane request duration must be positive')

    def to_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class Lon:
    kind: str = 'base'
    v_w: float | None = None
    amax: float = 1.
    dmax: float = 1.
    refs: tuple = ()
    off: float = 0.
    kp: float = .15
    kv: float = .6
    settle_gap: float | None = None
    from_el: int = 0              # LCMOVES sp_from
    a: float | None = None        # accel kind

    def __post_init__(self):
        object.__setattr__(self, 'refs', tuple(self.refs))
        if self.kind not in LON_KINDS:
            raise ValueError(f'lon kind must be one of {LON_KINDS}')
        for k in ('amax', 'dmax', 'off', 'kp', 'kv'):
            _finite(getattr(self, k), 'lon.'+k)
        if self.amax < 0 or self.dmax < 0:
            raise ValueError('amax/dmax are non-negative bounds')
        _opt(self.v_w, 'lon.v_w')
        _opt(self.settle_gap, 'lon.settle_gap')
        if self.kind == 'speed' and self.v_w is None:
            raise ValueError('speed lon needs v_w')
        if self.kind == 'track' and not self.refs:
            raise ValueError('track lon needs reference vehicles')
        if self.kind == 'accel' and self.a is None:
            raise ValueError('accel lon needs a')
        if type(self.from_el) is not int or self.from_el < 0:
            raise ValueError('lon from_el must be a non-negative integer second')

    def to_dict(self):
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out['refs'] = list(self.refs)
        return out


@dataclass(frozen=True)
class MemberProgram:
    """One member's program (probe role semantics, WM frame)."""
    lon: Lon = field(default_factory=Lon)
    lane: LaneReq | None = None
    start_el: int = 0
    hold_s: int | None = None
    release_s: float | None = None
    anchor: str | None = None
    anchor_release_s: float | None = None
    zone: tuple | None = None            # (s0, s1): inactive before s0, release after s1
    partner: str | None = None           # release when the partner left partner_lane
    partner_lane: int | None = None
    merger_lane: int | None = None       # merged_zip / merger release reference lane
    fallback_s: float | None = None
    lc_mode: int = LC_MODE_POLICY
    tau: float | None = None
    mode_on: str = 'first_active'
    releases: tuple = ('hold_end', 'release_s')
    role: str = 'prog'                   # label only (source role), never an executor input
    mode_logged: bool = False            # label only: the probe wrote the lcMode into its log

    def __post_init__(self):
        object.__setattr__(self, 'releases', tuple(self.releases))
        if self.zone is not None:
            object.__setattr__(self, 'zone', tuple(float(x) for x in self.zone))
            if len(self.zone) != 2 or not self.zone[0] <= self.zone[1]:
                raise ValueError('tau zone must be (s0 <= s1)')
        if not isinstance(self.lon, Lon) or (self.lane is not None and not isinstance(self.lane, LaneReq)):
            raise ValueError('member program needs Lon / LaneReq primitives')
        if type(self.start_el) is not int or self.start_el < 0:
            raise ValueError('start_el must be a non-negative integer second')
        if self.hold_s is not None and (type(self.hold_s) is not int or self.hold_s <= 0):
            raise ValueError('hold_s must be a positive integer')
        for k in ('release_s', 'anchor_release_s', 'fallback_s'):
            _opt(getattr(self, k), k)
        if self.tau is not None and not _finite(self.tau, 'tau') > 0:
            raise ValueError('tau must be positive')
        if self.lc_mode not in (LC_MODE_POLICY, LC_MODE_TRACI_ONLY) or self.mode_on not in MODE_ON:
            raise ValueError('lc_mode 513/512 and a registered mode_on rule')
        if any(r not in RELEASES for r in self.releases) or len(set(self.releases)) != len(self.releases):
            raise ValueError(f'releases must be distinct entries of {RELEASES}')
        need = {'anchor': ('anchor', 'anchor_release_s'), 'zone': ('zone',),
                'partner_left': ('partner', 'partner_lane'), 'merged_zip': ('merger_lane',),
                'merger': ('merger_lane',), 'fallback': ('fallback_s',)}
        for r in self.releases:
            if any(getattr(self, k) is None for k in need.get(r, ())):
                raise ValueError(f'release {r!r} needs {need[r]}')

    def to_dict(self):
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out['lon'] = self.lon.to_dict()
        out['lane'] = None if self.lane is None else self.lane.to_dict()
        out['releases'] = list(self.releases)
        out['zone'] = None if self.zone is None else list(self.zone)
        return out

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data['lon'] = Lon(**dict(data['lon'], refs=tuple(data['lon'].get('refs') or ())))
        data['lane'] = None if data.get('lane') is None else LaneReq(**data['lane'])
        data['releases'] = tuple(data.get('releases') or ())
        data['zone'] = None if data.get('zone') is None else tuple(data['zone'])
        return cls(**data)


@dataclass(frozen=True)
class FreeProgram:
    """A group program: members in join order (initiator first), one MemberProgram each.

    ``join_el`` = each member's join second relative to the initiator (0 for the initiator);
    member programs run on the MOVE clock (el since t0). ``relay`` = V2V join chaining
    (members re-broadcast the session so groups larger than one radio hop / 4 partner slots
    can fill, see lc_families)."""
    members: tuple
    programs: tuple                      # MemberProgram per member (same order)
    map: str = 'wz3to2'
    join_el: tuple = ()
    duration_s: int = 300
    source: dict = field(default_factory=dict)   # provenance (row id, probe, verification id)

    def __post_init__(self):
        object.__setattr__(self, 'members', tuple(self.members))
        object.__setattr__(self, 'programs', tuple(self.programs))
        object.__setattr__(self, 'join_el', tuple(self.join_el) or (0,)*len(self.members))
        if not self.members or len(set(self.members)) != len(self.members) or any(
                not isinstance(v, str) or not v for v in self.members):
            raise ValueError('free_program members must be distinct vehicle ids')
        if len(self.programs) != len(self.members) or any(not isinstance(p, MemberProgram) for p in self.programs):
            raise ValueError('one MemberProgram per member')
        if len(self.join_el) != len(self.members) or self.join_el[0] != 0 or any(
                type(t) is not int or t < 0 for t in self.join_el):
            raise ValueError('join_el: integer seconds, initiator 0')
        if type(self.duration_s) is not int or self.duration_s <= 0:
            raise ValueError('duration_s must be a positive integer')

    def program(self, vid):
        return self.programs[self.members.index(vid)]

    def to_dict(self):
        return dict(schema=SCHEMA, rule=RULE, map=self.map, members=list(self.members),
                    programs=[p.to_dict() for p in self.programs], join_el=list(self.join_el),
                    duration_s=int(self.duration_s), source=dict(self.source))

    @classmethod
    def from_dict(cls, data):
        if data.get('schema') != SCHEMA:
            raise ValueError('not a free_program record')
        return cls(tuple(data['members']), tuple(MemberProgram.from_dict(p) for p in data['programs']),
                   data.get('map', 'wz3to2'), tuple(data.get('join_el') or ()), int(data['duration_s']),
                   dict(data.get('source') or {}))


def check_registered(program, *, registry=None):
    """Registered-range check of the continuous parameters (generator contract).

    Returns a list of (member index, key, value) violations (empty = inside the registrations).
    Converted rows are exact copies of what the probes ran, so they pass by construction."""
    reg = REGISTRY if registry is None else registry
    out = []
    for i, p in enumerate(program.programs):
        checks = [('lc_mode', p.lc_mode)]
        if p.hold_s is not None:
            checks.append(('hold_s', p.hold_s))
        if p.tau is not None:
            checks.append(('tau', p.tau))
        if p.lane is not None:
            checks.append(('lane_duration_s', p.lane.duration_s))
        lon = p.lon
        if lon.kind in ('speed', 'track', 'match'):
            checks += [('amax', lon.amax), ('dmax', lon.dmax)]
        if lon.v_w is not None:
            checks.append(('v_w', lon.v_w))
        if lon.kind == 'track':
            checks += [('kp', lon.kp), ('kv', lon.kv), ('track_off_m', lon.off)]
        if lon.kind == 'accel':
            checks.append(('accel', lon.a))
        out += [(i, k, v) for k, v in checks if k in reg and not reg[k].ok(v)]
    return out


# ---------------------------------------------------------------- frame (N1 descriptor)

@dataclass(frozen=True)
class Frame:
    """Probe (SUMO global x, SUMO lane on the approach edge) <-> WM frame, from the N1 descriptor."""
    name: str
    edge: str                     # approach edge ('up' on wz3to2)
    anchor_x_m: float             # s = x - anchor_x_m
    drop_s_m: float
    approach_end_s_m: dict        # WM lane -> s where that approach lane ends
    sumo_to_wm: dict              # SUMO index on the approach edge -> WM lane
    closing_lane: int
    adjacent_lane: int
    far_lane: int | None

    def s(self, x):
        return float(x)-self.anchor_x_m

    def wm(self, sumo_lane):
        return self.sumo_to_wm[int(sumo_lane)]

    def sumo(self, wm_lane):
        return next(i for i, w in self.sumo_to_wm.items() if w == int(wm_lane))

    def on_approach(self, s_m, wm_lane):
        """Still on the approach edge (probe ``edge == 'up'``) in WM coordinates."""
        end = self.approach_end_s_m.get(int(wm_lane))
        return end is not None and float(s_m) < end


def frame_from_descriptor(desc):
    """Frame of an N1 lane-drop MapDescriptor (``map_descriptor.load``/``current``)."""
    if desc.kind != 'lane_drop':
        raise ValueError('free_program frames need a lane-drop descriptor')
    neck = desc.bottleneck
    edge, anchor = neck['edge'], float(neck['anchor_x_m'])
    drop_x = float(neck['drop_x_m'])
    table = {i: lane for (e, i), (lane, _) in desc.lane_table.items() if e == edge}
    ends = {}
    for i, lane in table.items():
        # lane_end_distance = drop_x - lane end x (TableSensor convention)
        ends[lane] = drop_x-float(desc.lane_end_distance[f'{edge}_{i}'])-anchor
    through = [int(x) for x in neck['through_lanes']]
    far = next((x for x in through if x != int(neck['adjacent_lane'])), None)
    return Frame(desc.name, edge, anchor, float(neck['drop_s_m']), ends, table, int(neck['closing_lane']),
                 int(neck['adjacent_lane']), far)


# ---------------------------------------------------------------- observation

@dataclass(frozen=True)
class View:
    """A vehicle as the member observes it (own sensing or a teammate's own t-1 packet)."""
    vid: str
    s_m: float
    lane: int
    v_mps: float
    on_approach: bool = True       # still on the lane-drop approach edge (SUMO 'up')
    is_cav: bool = False


@dataclass(frozen=True)
class Obs:
    """One member's observation at control second ``el`` (own row + own sensing)."""
    el: int
    own: View | None               # None = not live (left the network)
    sensed: tuple = ()             # Views within SENSING_M (own sensing), any type
    refs: dict = field(default_factory=dict)   # vid -> View (anchor / partner / track refs)

    def ref(self, vid):
        if vid in self.refs:
            return self.refs[vid]
        return next((r for r in self.sensed if r.vid == vid), None)


# ---------------------------------------------------------------- executor

class MemberExecutor:
    """Stateful per-member executor; ``step(obs)`` -> [(kind, value), ...] for this second.

    Mirrors the probe Arm.control semantics role by role (see the module docstring); the
    member program fields select the semantics, never the role label."""

    def __init__(self, vid, program):
        self.vid, self.p = vid, program
        self.active = False          # lcMode/tau applied (lc_on / tau_on)
        self.released, self.release_el, self.release_reason = False, None, None
        self.lane_done = None
        self.lane_left = None
        self.was_merger_lane = False
        self.settled = None
        self.last = []

    # ---- helpers
    def _ref(self, obs, vid):
        return obs.own if vid == self.vid else obs.ref(vid)

    def _release(self, el, why):
        self.released, self.release_el, self.release_reason = True, el, why
        return [('release', why)]

    def _activate(self, out):
        if self.p.mode_on != 'never' and not self.active:
            out.append(('lcMode', int(self.p.lc_mode)))
            if self.p.tau is not None:
                out.append(('tau', float(self.p.tau)))
            self.active = True

    def _law(self, obs, own, out, lane_ok):
        lon = self.p.lon
        if lon.kind in ('base', 'native') or obs.el < lon.from_el:
            return
        if lon.kind == 'match' and (lane_ok or not own.on_approach):
            return                       # speed matching only until the change is made
        if self.p.mode_on == 'on_action':
            self._activate(out)
        amax, dmax, v = float(lon.amax), float(lon.dmax), float(own.v_mps)
        if lon.kind == 'accel':
            out.append(('accel', float(lon.a)))
            return
        if lon.kind == 'track' and self.settled is None:
            refs = [self._ref(obs, r) for r in lon.refs]
            refs = [r for r in refs if r is not None]
            if not refs:
                self.settled = obs.el
            else:
                xr = sum(r.s_m for r in refs)/len(refs)+float(lon.off)
                vr = sum(r.v_mps for r in refs)/len(refs)
                err = xr-float(own.s_m)
                if lon.settle_gap is not None and lane_ok and abs(err) <= float(lon.settle_gap):
                    self.settled = obs.el
                else:
                    a = float(lon.kp)*err+float(lon.kv)*(vr-v)
                    out.append(('accel', max(-dmax, min(amax, a))))
                    return
        if lon.kind == 'match':
            target = self.p.lane.target
            speeds = [r.v_mps for r in obs.sensed if r.vid != self.vid and r.lane == target and r.on_approach
                      and abs(r.s_m-own.s_m) <= SENSING_M]
            if not speeds:
                return
            a = (sum(speeds)/len(speeds)-v)/CONTROL_DT_S
        else:
            if lon.v_w is None:
                return                   # track without a hold speed: no command after settling
            a = (float(lon.v_w)-v)/CONTROL_DT_S
        out.append(('accel', max(-dmax, min(amax, a))))

    def _lane(self, obs, own, out):
        q = self.p.lane
        if q is None:
            return
        if q.rule == 'until_reached':
            if self.lane_done is None:
                armed = obs.el >= q.from_el and (q.from_s is None or own.s_m >= q.from_s)
                if armed:
                    if self.p.mode_on == 'on_action':
                        self._activate(out)
                    if own.on_approach and own.lane == q.target:
                        self.lane_done = obs.el
                    elif own.on_approach and own.s_m < (q.giveup_s if q.giveup_s is not None else math.inf):
                        out.append(('changeLane', (int(q.target), float(q.duration_s))))
            elif own.on_approach and own.lane != q.target and self.lane_left is None:
                self.lane_left = obs.el
            return
        armed = obs.el >= q.from_el and (q.from_s is None or own.s_m >= q.from_s)
        if not armed or not own.on_approach:
            return
        if q.rule == 'right_of_target':
            # SUMO index below the target = WM index above it (WM 0 = leftmost)
            if own.lane > q.target and (q.giveup_s is None or own.s_m < q.giveup_s):
                out.append(('changeLane', (int(q.target), float(q.duration_s))))
        elif own.lane == q.source_lane and (q.giveup_s is None or own.s_m < q.giveup_s):
            out.append(('changeLane', (int(q.target), float(q.duration_s))))

    # ---- one control second
    def step(self, obs):
        out = []
        self.last = out
        if self.released:
            return out
        own = obs.own
        if own is None:                      # left the network: no TraCI command
            self.released, self.release_el, self.release_reason = True, obs.el, 'left'
            return out
        p = self.p
        if obs.el < p.start_el:
            return out
        s = float(own.s_m)
        for r in p.releases:
            if r == 'hold_end' and p.hold_s is not None and obs.el >= p.start_el+p.hold_s:
                return self._release(obs.el, 'hold_end')
            if r == 'release_s' and p.release_s is not None and s >= p.release_s:
                return self._release(obs.el, 'release_s')
            if r == 'anchor':
                a = self._ref(obs, p.anchor)
                if a is None or not a.on_approach or a.s_m >= p.anchor_release_s:
                    return self._release(obs.el, 'anchor')
            if r == 'zone':
                if s < p.zone[0]:
                    return out               # not yet in its zone: no command
                if s > p.zone[1]:
                    return self._release(obs.el, 'zone_end')
            if r == 'partner_left':
                q = self._ref(obs, p.partner)
                if q is None or not q.on_approach or q.lane != p.partner_lane:
                    return self._release(obs.el, 'partner_merged')
            if r == 'merged_zip' and (not own.on_approach or own.lane != p.merger_lane):
                return self._release(obs.el, 'merged')
            if r == 'merger':
                if own.on_approach and own.lane == p.merger_lane:
                    self.was_merger_lane = True
                elif self.was_merger_lane:
                    return self._release(obs.el, 'merged')
            if r == 'fallback' and s >= p.fallback_s:
                return self._release(obs.el, 'fallback')
        if p.mode_on == 'first_active':
            self._activate(out)
        self._lane(obs, own, out)
        lane_ok = p.lane is None or self.lane_done is not None
        self._law(obs, own, out, lane_ok)
        return out

    def state(self):
        return dict(active=self.active, released=self.released, release_el=self.release_el,
                    release_reason=self.release_reason, lane_done=self.lane_done, lane_left=self.lane_left,
                    settled=self.settled)


def step_members(executors, observations):
    """{vid: commands} for one second; ``observations`` = {vid: Obs} (members only)."""
    return {vid: ex.step(observations[vid]) for vid, ex in executors.items() if vid in observations}


def commanded(commands):
    """Split one member-second into (accel or None, lane (target, duration) or None, others)."""
    accel = next((v for k, v in commands if k == 'accel'), None)
    lane = next((v for k, v in commands if k == 'changeLane'), None)
    return accel, lane, [c for c in commands if c[0] not in ('accel', 'changeLane')]


# ---------------------------------------------------------------- per-tick labels

def tick_label(*, vid, t, el, program, executor, commands, session_id=None, source=None):
    """One per-CAV per-control-second imitation label (schema lcexec-tick-label-v1).

    ``lon``: the longitudinal primitive commanded this second ('accel' with ``accel_mps2``,
    'native' if active without a speed command, 'base' if not under the program).
    ``lane_target`` WM lane or -1; ``lc_mode``/``tau`` the member's active values (None when
    inactive); ``event`` in (None, 'activate', 'release', 'left')."""
    accel, lane, others = commanded(commands)
    kinds = [k for k, _ in others]
    released = 'release' in kinds
    event = 'release' if released else ('activate' if 'lcMode' in kinds or 'tau' in kinds else None)
    if executor.release_reason == 'left' and executor.release_el == el:
        event = 'left'
    under = executor.active or accel is not None or lane is not None
    lon = ('accel' if accel is not None else
           ('base' if program.lon.kind == 'base' or not under or released else 'native'))
    return dict(schema=LABEL_SCHEMA, vid=vid, t=int(t), el=int(el), session_id=session_id,
                role=program.role, lon=lon, accel_mps2=None if accel is None else float(accel),
                lane_target=-1 if lane is None else int(lane[0]),
                lane_duration_s=None if lane is None else float(lane[1]),
                lc_mode=int(program.lc_mode) if executor.active and not executor.released else None,
                tau=(float(program.tau) if program.tau is not None and executor.active
                     and not executor.released else None),
                event=event, release_reason=executor.release_reason if released else None,
                source=source)


__all__ = ['SCHEMA', 'LABEL_SCHEMA', 'RULE', 'REGISTRY', 'Range', 'extend_registry', 'LaneReq', 'Lon', 'Frame', 'frame_from_descriptor',
           'MemberProgram', 'FreeProgram', 'check_registered', 'View', 'Obs', 'MemberExecutor',
           'step_members', 'commanded', 'tick_label', 'LC_MODE_POLICY', 'LC_MODE_TRACI_ONLY',
           'LANE_CHANGE_DURATION_S']
