"""LCEXEC free_program runtime for the cooperation layer (clean5, DEFAULT OFF).

The ``free_program`` option family runs :mod:`lc_program` group programs through a V2V
initiate/join protocol inside :class:`track2_backend.EpisodePolicy` (key ``lc_families``;
absent/False = off and byte-identical everywhere).

Option families (per-CAV options, ``members == (ego,)``):

* ``free_program_initiate``: a free CAV proposes a group program TEMPLATE (a
  :class:`lc_program.FreeProgram` whose members are slots ``'@0'..'@n-1'`` with per-slot
  ``join_rules``) drawn by the registered generator from its OWN state and sensing; on
  acceptance it owns slot 0 and the session ``free_program:<ego>@<t>``.
* ``free_program_join``: a free CAV that RECEIVED the session intent (published at t-1,
  within ``range_m`` of the broadcaster) is offered the FIRST unfilled slot whose join rule
  it satisfies from its OWN row: ``lane`` (WM, or None = any) and ``ds`` = own s minus the
  initiator position carried in the message, extrapolated with the message speed.
  Offers are one-shot events (first receipt of each (session, slot)), like N9.

Join chaining (relay; groups larger than one radio hop / 4 partner slots): every
ACKNOWLEDGED member (join visible from t+1, i.e. joined at <= t-1) re-broadcasts the
session intent each second, so a CAV within ``range_m`` of ANY broadcaster receives it.
A relay carries the initiator position it knows: the initiator's own t-1 packet when it is
inside the relay's radio range, else the anchor of the copy the relay itself received
(at most ``MAX_ANCHOR_AGE_S`` old). Receivers dedupe copies by session id (freshest
anchor, then nearest broadcaster). The group cap is the template length <= settings cap.

Ownership: one session per CAV, executor commitment (``MoveExecutor.start``) until the
session end (initiate: now + template duration; join: the remaining duration, schedule
checked), slot free. ``'@k'`` references (track refs, anchor, partner) in every member
program are resolved to the vehicle id as slot k fills; unfilled slots stay missing
references (``Obs.ref`` -> None, lc_program semantics).

Execution (``commands``): per active member, :class:`lc_program.MemberExecutor` steps on
the move clock ``el = now - session start`` with an :class:`lc_program.Obs` built ONLY from
its own row, its own sensing (rows within 100 m, any vehicle type, geometry only) and
teammates' own t-1 V2V packets for references it does not sense. Lane requests pass an
own-sensing LANE GATE before they leave (adjacent +-1 only; the target lane must exist at
the CAV position: the closing lane only on the approach before the drop; outside the
LaneRules edge guard unless leaving the closing lane; ``LaneRules._target_gap_ok`` on the
CAV's own sensed rows; a same-lane 'hold' request passes without a gap check); refused
requests are dropped this second with a receipt reason. The surviving commands become the
env tick request (``env.lc_tick``, :mod:`lc_runtime`) via
:func:`lc_writer_harness.requests_from_commands`. Controller-reserved members get nothing.
Program members get move offset 0 (their longitudinal command is the exact program command).

Inputs are observable only; B and R run the identical code path (no arm input). HDVs are
never offered, members or commanded. The trigger extension (:class:`TriggerExtension`) is a
minimal zero-initialised descriptor; FULLPOL may supply its own extension later.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib
import json
import math

import torch

from . import lc_program as lp
from .merge_features import CONTROL_DT_S, SENSOR_RANGE_M, _valid_report

RULE = 'lcexec-lc-families-v1'
CONFIG_KEY = 'lc_families'
NAME = 'free_program'
KINDS = ('initiate', 'join')
PROGRAM_FAMILIES = tuple(f'{NAME}_{k}' for k in KINDS)
MAX_SLOTS = 6                       # lc_program.REGISTRY['cap'] upper bound
DEFAULT_RANGE_M = 100.0             # V2V radio (v2v_obs.V2V_RANGE_M)
MIN_JOIN_HOLD_S = 2                 # a join needs at least this many program seconds left
MAX_ANCHOR_AGE_S = 5                # relayed initiator anchors older than this are not re-broadcast
ROW_KEYS = ('vid', 's_m', 'v_mps', 'lane', 'is_cav', 'length_m')


# ---------------------------------------------------------------- settings

def _descriptor():
    from . import map_descriptor
    return map_descriptor.current()


def _resolve_generator(spec):
    if not isinstance(spec, str) or spec.count(':') != 1:
        raise ValueError(f"{CONFIG_KEY}.generator must be 'module:function'")
    module, name = spec.split(':')
    try:
        fn = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f'{CONFIG_KEY}.generator {spec!r} is not importable') from exc
    if not callable(fn):
        raise ValueError(f'{CONFIG_KEY}.generator {spec!r} is not callable')
    return fn


def settings(cfg):
    """Resolved settings, or None when ``lc_families`` is absent/False (default)."""
    raw = (cfg or {}).get(CONFIG_KEY, False)
    if raw is False or raw is None:
        return None
    keys, optional = {'enabled', 'rule', 'provenance', 'map', 'cap'}, {'range_m', 'generator'}
    if (not isinstance(raw, dict) or not keys <= set(raw) or set(raw)-keys-optional
            or raw['enabled'] is not True or raw['rule'] != RULE
            or not isinstance(raw['provenance'], str) or not raw['provenance'].strip()):
        raise ValueError(f'{CONFIG_KEY} requires enabled=True, rule={RULE!r} and provenance')
    if (cfg or {}).get('group_lane_moves') is not True:
        raise ValueError(f'{CONFIG_KEY} requires group_lane_moves=True (single-writer V2V lane runtime)')
    desc = _descriptor()
    if desc.kind != 'lane_drop' or desc.name != raw['map']:
        raise ValueError(f'{CONFIG_KEY}.map {raw["map"]!r} is not the bound lane-drop process map {desc.name!r}')
    cap = raw['cap']
    if type(cap) is not int or not 2 <= cap <= MAX_SLOTS or not lp.REGISTRY['cap'].ok(cap):
        raise ValueError(f'{CONFIG_KEY}.cap must be a registered integer in 2..{MAX_SLOTS}')
    range_m = raw.get('range_m', DEFAULT_RANGE_M)
    if isinstance(range_m, bool) or not isinstance(range_m, (int, float)) or not (
            math.isfinite(range_m) and range_m > 0):
        raise ValueError(f'{CONFIG_KEY}.range_m must be finite and positive')
    generator = raw.get('generator')
    propose = default_generator if generator is None else _resolve_generator(generator)
    return dict(rule=RULE, provenance=raw['provenance'], map=desc.name, frame=lp.frame_from_descriptor(desc),
                n_lanes=int(desc.frame.n_lanes), cap=cap, range_m=float(range_m), generator=generator,
                propose=propose)


def config_keys(*, provenance, map_name='wz3to2', cap, range_m=None, generator=None):
    """Exact common B/R binding."""
    out = dict(enabled=True, rule=RULE, provenance=provenance, map=map_name, cap=int(cap))
    if range_m is not None:
        out['range_m'] = float(range_m)
    if generator is not None:
        out['generator'] = generator
    return {CONFIG_KEY: out}


def _second(value):
    value = float(value)
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError('lc families run on integer control seconds')
    return int(value)


# ---------------------------------------------------------------- templates

def _slot_index(x):
    if isinstance(x, str) and x.startswith('@') and x[1:].isdigit():
        return int(x[1:])
    return None


def _program_refs(program):
    return tuple(program.lon.refs)+tuple(r for r in (program.anchor, program.partner) if r is not None)


def template_of(value, *, cap, frame, n_lanes):
    """Validated FreeProgram TEMPLATE (generator contract); ValueError otherwise."""
    program = value if isinstance(value, lp.FreeProgram) else lp.FreeProgram.from_dict(dict(value))
    n = len(program.members)
    if program.members != tuple(f'@{k}' for k in range(n)) or not 1 <= n <= cap:
        raise ValueError(f'free_program template members must be @0..@{cap-1} in slot order')
    if program.map != frame.name:
        raise ValueError(f'free_program template map {program.map!r} is not {frame.name!r}')
    rules = program.join_rules
    if n > 1 and len(rules) != n:
        raise ValueError('free_program template needs a join rule per slot')
    if any(r is not None and r.get('lane') is not None and r['lane'] not in range(n_lanes) for r in rules):
        raise ValueError('join rule lane outside the map lanes')
    for p in program.programs:
        for ref in _program_refs(p):
            k = _slot_index(ref)
            if k is not None and k >= n:
                raise ValueError(f'free_program template reference {ref!r} names no slot')
        if p.lane is not None and p.lane.target not in range(n_lanes):
            raise ValueError('free_program lane target outside the map lanes')
        grid = lp.REGISTRY['tau'].values
        if p.tau is not None and grid and not any(abs(float(p.tau)-float(x)) <= 1e-9 for x in grid):
            # env.lc_tick (lc_runtime) accepts only the registered tau grid values: fail at proposal time.
            raise ValueError(f'free_program tau {p.tau} is not a registered tau grid value {grid}')
    bad = lp.check_registered(program)
    if bad:
        raise ValueError(f'free_program template outside the registered ranges: {bad}')
    return program


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=list)


def template_sha256(template_dict):
    return hashlib.sha256(_canonical(template_dict).encode()).hexdigest()


def resolve(program, slots):
    """MemberProgram with '@k' references replaced by the slot-k vehicle id where filled."""
    def one(x):
        k = _slot_index(x)
        return x if k is None else slots.get(k, x)
    lon = replace(program.lon, refs=tuple(one(r) for r in program.lon.refs))
    return replace(program, lon=lon, anchor=None if program.anchor is None else one(program.anchor),
                   partner=None if program.partner is None else one(program.partner))


# Join windows of offline templates: own ds at t0 +- the registered settle window
# (S-LCMOVES wall_fill/rolling_wall settle within 25 m, lc_program.REGISTRY['settle_gap_m']).
JOIN_DS_HALF_M = float(lp.REGISTRY['settle_gap_m'].hi)


def template_from_program(free_program, frame, rows_at_t0):
    """Offline (FULLPOL bank BC): a concrete FreeProgram -> slot template.

    Slot k join_rule = member k's WM lane and its ds to the initiator at t0, widened to the
    registered window ds +- ``JOIN_DS_HALF_M`` (still containing it); member references
    (track refs, anchor, partner) become '@k'. ``rows_at_t0``: {vid: row} or [row]."""
    rows = rows_at_t0 if isinstance(rows_at_t0, dict) else {r['vid']: r for r in rows_at_t0}
    members = free_program.members
    if any(v not in rows for v in members):
        raise ValueError('template_from_program needs every member row at t0')
    slots = {v: f'@{k}' for k, v in enumerate(members)}
    s0 = float(rows[members[0]]['s_m'])
    rules = [None]
    for v in members[1:]:
        ds = float(rows[v]['s_m'])-s0
        rules.append(dict(lane=int(rows[v]['lane']), ds=(round(ds-JOIN_DS_HALF_M, 2), round(ds+JOIN_DS_HALF_M, 2))))
    programs = []
    for p in free_program.programs:
        swap = lambda x: slots.get(x, x)                # noqa: E731
        lon = replace(p.lon, refs=tuple(swap(r) for r in p.lon.refs))
        programs.append(replace(p, lon=lon, anchor=None if p.anchor is None else swap(p.anchor),
                                partner=None if p.partner is None else swap(p.partner)))
    if frame.name != free_program.map:
        raise ValueError('template frame/map mismatch')
    return lp.FreeProgram(tuple(f'@{k}' for k in range(len(members))), tuple(programs), free_program.map,
                          free_program.join_el, free_program.duration_s,
                          dict(free_program.source, template_of=list(members)), tuple(rules))


# ---------------------------------------------------------------- built-in generator (test fixture)

def default_generator(state, own, rng):
    """Small built-in template sampler (TEST FIXTURE default; FULLPOL supplies the real one).

    Draws ONLY lc_program.REGISTRY grid values: a solo speed hold, and (cap >= 2) a pair where
    slot 1 (same lane, behind by the registered track offset) tracks '@0' then holds v_w."""
    frame = state['frame']
    lane, s = int(own['lane']), float(own['s_m'])
    if not frame.on_approach(s, lane):
        return []
    pick = lambda key: lp.REGISTRY[key].values[int(rng.integers(len(lp.REGISTRY[key].values)))]  # noqa: E731
    v_w, dmax, hold = float(pick('v_w')), float(pick('dmax')), int(pick('hold_s'))
    head = lp.MemberProgram(lon=lp.Lon('speed', v_w=v_w, amax=1., dmax=dmax), hold_s=hold,
                            release_s=frame.drop_s_m, releases=('hold_end', 'release_s'), role='hold')
    out = [lp.FreeProgram(('@0',), (head,), frame.name, (0,), hold, dict(generator='lc_families.default'),
                          (None,))]
    if state['cap'] >= 2:
        off = float(pick('track_off_m'))
        follow = lp.MemberProgram(lon=lp.Lon('track', v_w=v_w, amax=1., dmax=dmax, refs=('@0',), off=off,
                                             kp=.15, kv=.6, settle_gap=25.), hold_s=hold,
                                  release_s=frame.drop_s_m, releases=('hold_end', 'release_s'), role='track')
        rule = dict(lane=lane, ds=(off-JOIN_DS_HALF_M, off+JOIN_DS_HALF_M))
        out.append(lp.FreeProgram(('@0', '@1'), (head, follow), frame.name, (0, 0), hold,
                                  dict(generator='lc_families.default'), (None, rule)))
    return out


# ---------------------------------------------------------------- option / message

def is_program_family(family):
    return isinstance(family, str) and family in PROGRAM_FAMILIES


@dataclass(frozen=True)
class ProgramOption:
    """One per-CAV free_program option; onset/hold relative to the decision second."""
    family: str
    members: tuple
    session_id: str
    initiator: str
    slot: int
    template: dict                 # FreeProgram.to_dict() of the TEMPLATE ('@k' members, join_rules)
    onset: int = 0
    hold: int = 1
    depth: float = 0.
    member_count: int = 1
    cap: int = 2
    gap_m: float = 0.
    range_m: float = DEFAULT_RANGE_M
    lane: int = -1                 # the acting CAV's WM lane when offered (descriptor only)

    self_validated_windows = True  # trigger_policy: registered program windows, not variety

    def __post_init__(self):
        object.__setattr__(self, 'members', tuple(self.members))
        if not is_program_family(self.family):
            raise ValueError(f'unregistered program family {self.family!r}')
        program = lp.FreeProgram.from_dict(dict(self.template))
        object.__setattr__(self, 'template', program.to_dict())
        n = len(program.members)
        if not program.is_template or program.members != tuple(f'@{k}' for k in range(n)):
            raise ValueError('program option needs a slot template (@0..@n-1)')
        if (len(self.members) != 1 or not isinstance(self.session_id, str) or not self.session_id
                or not isinstance(self.initiator, str)
                or (self.kind == 'initiate') != (self.initiator == self.members[0])):
            raise ValueError('program option: exactly the acting CAV and a consistent initiator/session')
        for key in ('slot', 'onset', 'hold', 'member_count', 'cap', 'lane'):
            if type(getattr(self, key)) is not int:
                raise ValueError(f'program option {key} must be an integer')
        if (self.kind == 'initiate') != (self.slot == 0) or not 0 <= self.slot < n:
            raise ValueError('program option slot: 0 for the initiator, an existing join slot otherwise')
        if self.onset != 0 or not 1 <= self.hold <= program.duration_s or float(self.depth) != 0.:
            raise ValueError('program option schedule: onset 0, 1 <= hold <= template duration, depth 0')
        if self.kind == 'initiate' and self.hold != program.duration_s:
            raise ValueError('initiate hold must equal the template duration')
        if (not 2 <= self.cap <= MAX_SLOTS or n > self.cap or not 1 <= self.member_count <= self.cap
                or (self.kind == 'join' and self.member_count >= n)):
            raise ValueError('program option membership outside the registered cap')
        if not (math.isfinite(self.gap_m) and math.isfinite(self.range_m) and self.range_m > 0):
            raise ValueError('program option geometry must be finite')

    def __hash__(self):
        return hash((self.family, self.members, self.session_id, self.slot))

    @property
    def name(self):
        return NAME

    @property
    def kind(self):
        return self.family[len(NAME)+1:]

    @property
    def onset_s(self):
        return self.onset

    @property
    def hold_s(self):
        return self.hold

    @property
    def actors(self):
        return self.members

    def program(self):
        return lp.FreeProgram.from_dict(self.template)

    def to_dict(self):
        return dict(family=self.family, members=list(self.members), session_id=self.session_id,
                    initiator=self.initiator, slot=int(self.slot), template=self.template,
                    onset=int(self.onset), hold=int(self.hold), depth=float(self.depth),
                    member_count=int(self.member_count), cap=int(self.cap), gap_m=float(self.gap_m),
                    range_m=float(self.range_m), lane=int(self.lane))

    @classmethod
    def from_dict(cls, data):
        return cls(data['family'], tuple(data['members']), data['session_id'], data['initiator'],
                   int(data['slot']), dict(data['template']), int(data['onset']), int(data['hold']),
                   float(data['depth']), int(data['member_count']), int(data['cap']), float(data['gap_m']),
                   float(data['range_m']), int(data.get('lane', -1)))


def is_program_option(move):
    return isinstance(getattr(move, 'move', move), ProgramOption)


def option_from_dict(data):
    """Recorded slate entry -> option (program families, else N9 -> N2 / W1 / shared vocabulary)."""
    if is_program_family(data['family']):
        return ProgramOption.from_dict(data)
    from .nm_families import option_from_dict as n9
    return n9(data)


@dataclass(frozen=True)
class ProgramIntent:
    """One broadcaster's copy of a session intent at ``timestamp_s`` (visible at t+1, within range)."""
    session_id: str
    initiator: str
    broadcaster: str
    timestamp_s: int
    source_s_m: float              # broadcaster's own position (radio range reference)
    source_lane: int
    source_v_mps: float
    anchor_s_m: float              # initiator position known to the broadcaster ...
    anchor_v_mps: float
    anchor_time_s: int             # ... at this second (extrapolate with anchor_v_mps)
    start_s: int
    duration_s: int
    template: dict
    roster: tuple                  # ((vid, slot), ...) acknowledged by t-1
    cap: int

    @property
    def end_s(self):
        return self.start_s+self.duration_s

    def anchor_at(self, now):
        return float(self.anchor_s_m)+float(self.anchor_v_mps)*(now-self.anchor_time_s)

    def to_dict(self):
        """Receipt summary (the template is identified by its sha256)."""
        out = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != 'template'}
        out.update(roster=[list(r) for r in self.roster], template_sha256=template_sha256(self.template))
        return out


# ---------------------------------------------------------------- slate helpers (feasible_proposals)

def slate_families(state):
    return PROGRAM_FAMILIES


def join_families(state):
    return (f'{NAME}_join',)


def _row(r):
    return {k: r[k] for k in ROW_KEYS}


def _propose(state, own, rng):
    """Validated templates from the registered generator (own state/sensing + static map only)."""
    view = dict(now=state['now'], map=state['map'], frame=state['frame'], cap=state['cap'],
                range_m=state['range_m'], sensed=tuple(_row(r) for r in state['sensed']),
                registry=lp.REGISTRY)
    out = state['propose'](view, _row(own), rng) or []
    return [template_of(t, cap=state['cap'], frame=state['frame'], n_lanes=state['n_lanes']) for t in out]


def _initiate(template, state, own):
    ego = own['vid']
    return ProgramOption(f'{NAME}_initiate', (ego,), f'{NAME}:{ego}@{state["now"]}', ego, 0, template.to_dict(),
                         0, int(template.duration_s), 0., 1, int(state['cap']), 0., float(state['range_m']),
                         int(own['lane']))


def pool(family, state, own, rng):
    """Feasible options of one program family for the acting CAV (local state only)."""
    if not is_program_family(family) or not own.get('is_cav'):
        return []
    if family.endswith('_join'):
        return [o for o in state.get('offers', ()) if o.family == family and o.members == (own['vid'],)]
    if state.get('only_join'):
        return []
    return [_initiate(t, state, own) for t in _propose(state, own, rng)]


def redraw(template, state, own, rng):
    """Slate entry: initiate re-draws a template (None if the generator has none), join is exact."""
    if template.kind == 'join':
        return template
    drawn = _propose(state, own, rng)
    if not drawn:
        return None
    return _initiate(drawn[int(rng.integers(len(drawn)))], state, own)


# ---------------------------------------------------------------- lane gate

def lane_gate(vid, target, rows, frame, n_lanes):
    """Own-sensing recheck of one program lane request -> (ok, reason).

    ``rows``: the CAV's OWN sensed rows (including itself). Adjacent +-1 only; the target
    lane must exist at the CAV position (closing lane only on the approach before the
    drop); LaneRules edge guard near the drop unless leaving the closing lane; target-lane
    gaps by the unchanged ``LaneRules._target_gap_ok``; a same-lane request ('hold')
    passes without a gap check. Malformed/absent sensing fails closed."""
    from wms.control.lanes import LaneRules, _target_gap_ok
    rules = LaneRules()
    by = {r.get('vid'): r for r in rows}
    ego = by.get(vid)
    if ego is None or ego.get('is_cav') is not True or len(by) != len(rows) or any(
            type(r.get('is_cav')) is not bool or r.get('lane') not in range(n_lanes)
            or any(isinstance(r.get(k), bool) or not isinstance(r.get(k), (int, float))
                   or not math.isfinite(r[k]) for k in ('s_m', 'v_mps', 'length_m'))
            or r['v_mps'] < 0 or r['length_m'] <= 0
            or abs(r['s_m']-ego['s_m']) > SENSOR_RANGE_M for r in rows):
        return False, 'missing_observable_gap_or_cav'
    source, s, v = int(ego['lane']), float(ego['s_m']), float(ego['v_mps'])
    if type(target) is not int:
        return False, 'not_adjacent'
    if target == source:
        return True, 'hold'
    if abs(target-source) != 1:
        return False, 'not_adjacent'
    if target not in range(n_lanes) or (target == frame.closing_lane and not frame.on_approach(s, target)):
        return False, 'no_target_lane'
    if source != frame.closing_lane and 0 <= frame.drop_s_m-s < v*CONTROL_DT_S+rules.edge_guard_m:
        return False, 'unsafe_gap_or_edge'
    tensor = lambda k, kind=float: torch.tensor([[kind(r[k]) for r in rows]])  # noqa: E731
    lane = tensor('lane', int)
    goal = lane.clone()
    index = rows.index(ego)
    goal[0, index] = target
    ok = _target_gap_ok(tensor('s_m'), tensor('v_mps'), tensor('length_m'),
                        torch.ones_like(lane, dtype=torch.bool), lane, goal, rules)[0, index]
    return (True, 'ok') if bool(ok) else (False, 'unsafe_gap_or_edge')


# ---------------------------------------------------------------- runtime protocol

class Protocol:
    """Per-episode free_program V2V protocol + member executors (never a policy input)."""

    FAMILIES = PROGRAM_FAMILIES

    def __init__(self, program_settings):
        self.settings = dict(program_settings)
        self.frame, self.n_lanes = self.settings['frame'], int(self.settings['n_lanes'])
        self.cap, self.range_m = int(self.settings['cap']), float(self.settings['range_m'])
        self.propose = self.settings['propose']
        self.reset()

    def reset(self):
        self.sessions = {}          # sid -> session ledger
        self.membership = {}        # vid -> sid
        self.board = {}             # (sid, broadcaster) -> ProgramIntent published at the last second
        self.inbox = {}             # vid -> {sid: ProgramIntent} received this second (deduped)
        self.offers, self.offered = {}, {}
        self.rows, self.sensed, self.prev_reports, self.now = {}, {}, {}, None
        self.last_published, self.last_joins, self.last_commands = [], [], {}
        self.last_labels, self.last_gate, self.last_request, self.expired = [], {}, {}, []

    # ------------------------------------------------ inputs
    def _expire(self, now):
        for sid in [k for k, s in self.sessions.items() if now >= s['start_s']+s['duration_s']]:
            session = self.sessions.pop(sid)
            receipts = {}
            for vid, ex in session['executors'].items():
                if not ex.released:
                    ex._release(now-session['start_s'], 'session_end')
                receipts[vid] = ex.receipt()
                if self.membership.get(vid) == sid:
                    del self.membership[vid]
            self.expired.append(dict(session_id=sid, end_s=now, receipts=receipts))

    def receive(self, now, own_rows, own_sensing, previous_reports):
        """Offers from intents published at now-1 (radio range of ANY broadcaster), own rows/sensing."""
        now = _second(now)
        self._expire(now)
        self.now = now
        self.rows = {v: _row(r) for v, r in own_rows.items() if r.get('is_cav') is True}
        self.sensed = {}
        for vid, own in self.rows.items():
            rows = [_row(r) for r in (own_sensing or {}).get(vid, ()) if r['vid'] != vid
                    and abs(float(r['s_m'])-float(own['s_m'])) <= SENSOR_RANGE_M]
            self.sensed[vid] = [own]+sorted(rows, key=lambda r: r['vid'])
        self.prev_reports = dict(previous_reports or {})
        inbox, offers = {}, {}
        for vid, own in sorted(self.rows.items()):
            s, copies = float(own['s_m']), {}
            for msg in self.board.values():
                if msg.timestamp_s != now-CONTROL_DT_S or msg.broadcaster == vid:
                    continue
                distance = abs(float(msg.source_s_m)-s)
                if distance > self.range_m:
                    continue
                key = (-msg.anchor_time_s, distance, msg.broadcaster)
                if msg.session_id not in copies or key < copies[msg.session_id][0]:
                    copies[msg.session_id] = (key, msg)
            inbox[vid] = {sid: m for sid, (_, m) in sorted(copies.items())}
            if vid in self.membership:
                continue
            out = []
            for sid, msg in inbox[vid].items():
                option = self._offer(msg, own, now)
                if option is not None:
                    out.append(option)
            if out:
                offers[vid] = tuple(sorted(out, key=lambda o: (abs(o.gap_m), o.session_id, o.slot)))
        self.inbox, self.offers = inbox, offers
        return offers

    def _offer(self, msg, own, now):
        vid = own['vid']
        template = lp.FreeProgram.from_dict(msg.template)
        roster = {v for v, _ in msg.roster}
        filled = {k for _, k in msg.roster}
        n = len(template.members)
        remaining = msg.end_s-now
        if (msg.initiator == vid or vid in roster or len(roster) >= min(msg.cap, n)
                or remaining < MIN_JOIN_HOLD_S):
            return None
        ds = float(own['s_m'])-msg.anchor_at(now)
        for k in range(1, n):
            rule = template.join_rules[k] if template.join_rules else None
            if k in filled or rule is None:
                continue
            if (rule.get('lane') is None or int(own['lane']) == rule['lane']) and \
                    float(rule['ds'][0]) <= ds <= float(rule['ds'][1]):
                return ProgramOption(f'{NAME}_join', (vid,), msg.session_id, msg.initiator, k, msg.template,
                                     0, int(remaining), 0., len(msg.roster), int(msg.cap), -ds, self.range_m,
                                     int(own['lane']))
        return None

    def state(self, vid, now):
        """Local view attached to the CAV's candidate state (own row/sensing + received offers)."""
        return dict(now=_second(now), families=PROGRAM_FAMILIES, offers=self.offers.get(vid, ()),
                    cap=self.cap, range_m=self.range_m, map=self.frame.name, frame=self.frame,
                    n_lanes=self.n_lanes, sensed=tuple(self.sensed.get(vid, ())), propose=self.propose)

    def events(self, members, free):
        return {v for v in members if v in free and any(
            (o.session_id, o.slot) not in self.offered.get(v, ()) for o in self.offers.get(v, ()))}

    def mark_offered(self, ego, table):
        for move in table:
            option = getattr(move, 'move', move)
            if isinstance(option, ProgramOption) and option.kind == 'join':
                self.offered.setdefault(ego, set()).add((option.session_id, option.slot))

    # ------------------------------------------------ arbitration
    def _executor(self, session, vid):
        slot = session['members'][vid]['slot']
        program = resolve(session['template'].programs[slot], session['slots'])
        return lp.MemberExecutor(vid, program)

    def _resolve_all(self, session):
        for vid, ex in session['executors'].items():
            ex.p = resolve(session['template'].programs[session['members'][vid]['slot']], session['slots'])

    def start(self, move, *, ego, executor, observable_cav_ids, now):
        """Atomic ownership (one session per CAV, slot free, schedule) + executor commitment."""
        option = getattr(move, 'move', move)
        now = _second(now)
        if not isinstance(option, ProgramOption) or option.members != (ego,):
            raise ValueError('program option must act through the deciding CAV only')
        own = self.rows.get(ego)
        if ego in self.membership or own is None:
            return False
        template = template_of(option.template, cap=self.cap, frame=self.frame, n_lanes=self.n_lanes)
        if option.kind == 'initiate':
            if (option.session_id in self.sessions or option.session_id != f'{NAME}:{ego}@{now}'
                    or option.cap != self.cap or option.hold != template.duration_s):
                return False
            if not executor.start(move, ego=ego, observable_cav_ids=observable_cav_ids, now=now,
                                  until=now+option.hold):
                return False
            session = dict(initiator=ego, start_s=now, duration_s=int(template.duration_s), cap=self.cap,
                           template=template, template_dict=template.to_dict(), slots={0: ego},
                           members={ego: dict(t=now, slot=0)}, executors={}, closed=False)
            session['executors'][ego] = self._executor(session, ego)
            self.sessions[option.session_id] = session
            self.membership[ego] = option.session_id
            return True
        session = self.sessions.get(option.session_id)
        if session is None or session['closed'] or option.slot in session['slots'] \
                or len(session['members']) >= min(session['cap'], len(template.members)):
            return False
        end = session['start_s']+session['duration_s']
        if now+option.hold != end or option.template != session['template_dict']:
            raise ValueError('join option schedule/template differ from the session it names')
        if not executor.start(move, ego=ego, observable_cav_ids=observable_cav_ids, now=now, until=end):
            return False
        session['slots'][option.slot] = ego
        session['members'][ego] = dict(t=now, slot=option.slot)
        session['executors'][ego] = self._executor(session, ego)
        self._resolve_all(session)
        self.membership[ego] = option.session_id
        self.last_joins.append(dict(session_id=option.session_id, vid=ego, time_s=now, slot=option.slot))
        return True

    def open_offline(self, option, now):
        """Offline (bank) reconstruction of an accepted initiation; no executor commitment."""
        template = template_of(option.template, cap=self.cap, frame=self.frame, n_lanes=self.n_lanes)
        session = dict(initiator=option.initiator, start_s=_second(now), duration_s=int(template.duration_s),
                       cap=int(option.cap), template=template, template_dict=template.to_dict(),
                       slots={0: option.initiator}, members={option.initiator: dict(t=_second(now), slot=0)},
                       executors={}, closed=False)
        session['executors'][option.initiator] = self._executor(session, option.initiator)
        self.sessions[option.session_id] = session
        self.membership[option.initiator] = option.session_id

    def admit_offline(self, option, now):
        session = self.sessions.get(option.session_id)
        vid = option.members[0]
        if (session is None or session['closed'] or option.slot in session['slots'] or vid in self.membership
                or len(session['members']) >= min(session['cap'], len(session['template'].members))):
            return False
        session['slots'][option.slot] = vid
        session['members'][vid] = dict(t=_second(now), slot=option.slot)
        session['executors'][vid] = self._executor(session, vid)
        self._resolve_all(session)
        self.membership[vid] = option.session_id
        return True

    # ------------------------------------------------ broadcast
    def _anchor(self, sid, session, broadcaster, source, now):
        if broadcaster == session['initiator']:
            return float(source.s_m), float(source.v_mps), now
        report = self.prev_reports.get(session['initiator'])
        if (_valid_report(report, session['initiator'], float(now-1))
                and abs(float(report.source.s_m)-float(source.s_m)) <= self.range_m):
            return float(report.source.s_m), float(report.source.v_mps), now-1
        copy = self.inbox.get(broadcaster, {}).get(sid)
        if copy is not None and now-copy.anchor_time_s <= MAX_ANCHOR_AGE_S:
            return float(copy.anchor_s_m), float(copy.anchor_v_mps), int(copy.anchor_time_s)
        return None

    def publish(self, now, sources):
        """Initiator + acknowledged members broadcast at ``now``; sources = {cav: own ObservedVehicle}."""
        now = _second(now)
        self._expire(now)
        board = {}
        for sid, s in sorted(self.sessions.items()):
            if s['closed']:
                continue
            acknowledged = sorted(((v, m['slot']) for v, m in s['members'].items()
                                   if v == s['initiator'] or m['t'] <= now-1), key=lambda r: r[1])
            for vid, _ in acknowledged:
                source = sources.get(vid)
                if source is None:
                    continue
                anchor = self._anchor(sid, s, vid, source, now)
                if anchor is None:
                    continue
                board[(sid, vid)] = ProgramIntent(sid, s['initiator'], vid, now, float(source.s_m),
                    int(source.lane), float(source.v_mps), anchor[0], anchor[1], int(anchor[2]), s['start_s'],
                    s['duration_s'], s['template_dict'], tuple(acknowledged), s['cap'])
        self.board = board
        self.last_published = [m.to_dict() for _, m in sorted(board.items())]
        return board

    # ------------------------------------------------ execution
    def _view_row(self, r):
        lane, s = int(r['lane']), float(r['s_m'])
        return lp.View(r['vid'], s, lane, float(r['v_mps']), self.frame.on_approach(s, lane), bool(r['is_cav']))

    def observation(self, sid, vid, now):
        """lc_program.Obs of one member: own row + own sensing + teammates' own t-1 packets."""
        session = self.sessions[sid]
        el = now-session['start_s']
        own = self.rows.get(vid)
        if own is None:
            return lp.Obs(el, None)
        sensed = tuple(self._view_row(r) for r in self.sensed.get(vid, ())[1:]
                       if abs(float(r['s_m'])-float(own['s_m'])) <= lp.SENSING_M)
        seen = {r.vid for r in sensed}
        refs = {}
        for ref in _program_refs(session['executors'][vid].p):
            if ref in seen or ref == vid or ref not in session['members']:
                continue
            report = self.prev_reports.get(ref)
            if (_valid_report(report, ref, float(now-1))
                    and abs(float(report.source.s_m)-float(own['s_m'])) <= self.range_m):
                src = report.source
                refs[ref] = lp.View(ref, float(src.s_m), int(src.lane), float(src.v_mps),
                                    self.frame.on_approach(src.s_m, src.lane), True)
        return lp.Obs(el, self._view_row(own), sensed, refs)

    def commands(self, now, active, *, reserved=()):
        """Per-member program commands for this second -> env tick request (``env.lc_tick``).

        ``active`` = executor entries' (vid, option) of THIS protocol (live context members).
        Returns the JSON request dict (only non-empty keys; {} when nothing is requested)."""
        from .lc_writer_harness import requests_from_commands
        now = _second(now)
        out, gate, labels, by_vid, stepped = {}, {}, [], {}, {}
        live = {}
        for vid, option in active:
            live[vid] = option
        for sid, session in sorted(self.sessions.items()):
            for vid, ex in sorted(session['executors'].items()):
                if vid not in live and vid not in self.rows and not ex.released:
                    ex.step(lp.Obs(now-session['start_s'], None))     # left the network: 'left'
        for vid, option in sorted(live.items()):
            session = self.sessions.get(option.session_id)
            if session is None or vid not in session['executors'] or vid not in self.rows:
                out[vid] = dict(session_id=option.session_id, status='no_session')
                continue
            if vid in reserved:
                out[vid] = dict(session_id=option.session_id, slot=session['members'][vid]['slot'],
                                status='controller_reserved', commands=[])
                continue
            ex = session['executors'][vid]
            obs = self.observation(option.session_id, vid, now)
            commands = list(ex.step(obs))
            status = 'released' if ex.released else ('active' if ex.active or commands else 'waiting')
            lane = next((c for c in commands if c[0] == 'changeLane'), None)
            if lane is not None:
                ok, reason = lane_gate(vid, int(lane[1][0]), self.sensed.get(vid, []), self.frame, self.n_lanes)
                if not ok:
                    commands = [c for c in commands if c[0] != 'changeLane']
                    gate[vid] = dict(target=int(lane[1][0]), reason=reason)
            by_vid[vid], stepped[vid] = commands, ex
            out[vid] = dict(session_id=option.session_id, slot=session['members'][vid]['slot'], el=obs.el,
                            status=status, commands=[[k, list(v) if isinstance(v, tuple) else v] for k, v in commands],
                            gate=gate.get(vid))
            labels.append(lp.tick_label(vid=vid, t=now, el=obs.el, program=ex.p, executor=ex, commands=commands,
                                        session_id=option.session_id,
                                        source=dict(kind=NAME, slot=session['members'][vid]['slot'])))
        for sid, session in self.sessions.items():
            ex = session['executors'].get(session['initiator'])
            if ex is not None and ex.released:
                session['closed'] = True                     # no further joins/broadcasts
        request = requests_from_commands(by_vid, stepped)
        tick = {}
        for key in ('accel', 'native', 'lanes', 'lane_mode', 'tau'):
            value = request[key]
            if value:
                tick[key] = list(value) if key == 'native' else dict(value)
        self.last_commands, self.last_gate, self.last_labels, self.last_request = out, gate, labels, tick
        return tick

    def receipt(self):
        out = dict(rule=RULE, map=self.frame.name, published=self.last_published,
                   offers={v: [[o.session_id, o.slot] for o in opts] for v, opts in sorted(self.offers.items())},
                   joins=list(self.last_joins), commands=dict(sorted(self.last_commands.items())),
                   gate_blocks=dict(sorted(self.last_gate.items())), labels=list(self.last_labels),
                   request=self.last_request, expired=list(self.expired),
                   sessions=[dict(session_id=k, initiator=s['initiator'], start_s=s['start_s'],
                                  end_s=s['start_s']+s['duration_s'], template_sha256=template_sha256(s['template_dict']),
                                  slots={str(i): v for i, v in sorted(s['slots'].items())}, closed=s['closed'],
                                  receipts={v: ex.receipt() for v, ex in sorted(s['executors'].items())})
                             for k, s in sorted(self.sessions.items())])
        self.last_joins, self.expired = [], []
        return out


def controller_requests(controller, now):
    """This tick's lc_tick request from the last policy sample (branch release applies)."""
    if not controller.events:
        return None
    request = controller.events[-1]['sample'].get('lc_tick')
    if not request:
        return None
    request = json.loads(json.dumps(request))
    branch = getattr(controller, 'branch', None)
    if branch and now >= branch['time_s']:
        # Same counterfactual-branch release as group_lane.controller_targets.
        release = set(branch.get('group', ()) if now >= branch.get('until', math.inf) else branch['revert'])
        allowed = None
        if branch.get('mappo_background'):
            allowed = set(branch['group'])-set(branch['revert']) if now < branch['until'] else set()

        def keep(vid):
            return vid not in release and (allowed is None or vid in allowed)
        request = {k: ([v for v in x if keep(v)] if k == 'native' else {v: y for v, y in x.items() if keep(v)})
                   for k, x in request.items()}
        request = {k: v for k, v in request.items() if v}
    return request or None


# ---------------------------------------------------------------- trigger extension

LON_KINDS = lp.LON_KINDS
DESCRIPTOR = (('is_program_initiate', 'is_program_join') + tuple(f'slot_{k}' for k in range(MAX_SLOTS))
              + ('member_count_over_cap', 'gap_over_range') + tuple(f'lon_{k}' for k in LON_KINDS)
              + ('lane_change_sign', 'v_w_norm', 'hold_norm', 'tau_present'))


def _norm(key, value):
    rng = lp.REGISTRY[key]
    if value is None or rng.hi <= rng.lo:
        return 0.
    return max(0., min(1., (float(value)-rng.lo)/(rng.hi-rng.lo)))


class TriggerExtension:
    """Minimal program-option descriptor; zero-initialised first-layer columns (TriggerPolicy).

    FULLPOL may supply its own extension later (same option families)."""

    wall = None
    aware = None
    nm = None

    def __init__(self, program_settings):
        self.lc = program_settings
        self.families = PROGRAM_FAMILIES
        self.descriptor_width = len(DESCRIPTOR)
        self.context_width = 0

    def descriptor(self, move):
        family = None if move is None else (move['family'] if isinstance(move, dict) else move.family)
        if not is_program_family(family):
            return [0.]*self.descriptor_width
        option = move if isinstance(move, ProgramOption) else (
            ProgramOption.from_dict(move) if isinstance(move, dict) else getattr(move, 'move', move))
        program = lp.FreeProgram.from_dict(option.template).programs[option.slot]
        out = [float(option.kind == 'initiate'), float(option.kind == 'join')]
        out += [float(option.slot == k) for k in range(MAX_SLOTS)]
        out += [option.member_count/option.cap, max(-1., min(1., option.gap_m/option.range_m))]
        out += [float(program.lon.kind == k) for k in LON_KINDS]
        sign = 0.
        if program.lane is not None and option.lane >= 0:
            sign = float((program.lane.target > option.lane)-(program.lane.target < option.lane))
        hold = program.hold_s if program.hold_s is not None else option.hold
        return out+[sign, _norm('v_w', program.lon.v_w), _norm('hold_s', hold), float(program.tau is not None)]

    def context_features(self, context):
        return torch.zeros((len(context['members']), 0))


def extension(cfg):
    value = settings(cfg)
    return None if value is None else TriggerExtension(value)


__all__ = ['RULE', 'CONFIG_KEY', 'PROGRAM_FAMILIES', 'settings', 'config_keys', 'ProgramOption', 'ProgramIntent',
           'Protocol', 'TriggerExtension', 'extension', 'pool', 'redraw', 'slate_families', 'join_families',
           'option_from_dict', 'is_program_option', 'is_program_family', 'template_of', 'template_from_program',
           'resolve', 'lane_gate', 'default_generator', 'controller_requests', 'DESCRIPTOR']
