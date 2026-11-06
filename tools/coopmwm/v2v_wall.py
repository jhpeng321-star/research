"""W1: V2V wall initiate/join protocol and wider coop-layer awareness (both DEFAULT OFF).

Motivation (COOP_PROBE_REPORT 2026-11-06, REGISTRATION observability limit): verified
3-4 CAV mainline walls show real group synergy, but in 4 of 5 mined examples no member
sees every other member inside its own observable slots (ego + 3 nearest CAVs within
sensing), so a decentralised trigger can never select the joint ``wall`` move.

(1) ``v2v_wall`` protocol (cooperation layer only; identical for B and R):

- INITIATE: at its own decision a free CAV upstream of the merge zone may choose the
  ``wall_initiate`` option, sampled from the registered variety onset/hold/depth windows
  and clipped to its observable constant-speed merge-zone arrival exactly like the
  ``wall`` move. Its trigger additionally sees its own merge-pressure block (the
  existing local t/t-1 ``merge_features``). Acceptance opens a wall session.
- BROADCAST: every second the initiator publishes a :class:`WallIntent` over the V2V
  intent channel: wall id, its own position (at t), target zone (the registered merge
  zone; release at its upstream edge), absolute start, onset/hold/depth, member count
  known at t (joins acknowledged by t-1), the registered member cap and a release flag.
  A message published at t is visible ONLY at t+1 (same t-1 rule as intent sharing).
- JOIN: a free CAV whose own current position is within ``range_m`` of the published
  initiator position receives a ``wall_join`` option in its trigger slate (one per
  received wall, while the remaining hold is >= the registered minimum and the
  member count is below the cap). The option executes the SAME absolute schedule
  (remaining onset/hold, same depth) through the existing MoveExecutor commitment and
  the layer's committed move offset, i.e. the single-writer command path. HDVs are
  never offered or commanded. First receipt of a wall triggers a one-shot event
  decision (non-periodic tick: only join options, everyone else ``none``).
- OWNERSHIP: a CAV belongs to at most one move (executor commitments and controller
  reservations mask it); session membership is capped by the registered
  ``member_cap`` (2..6). Same-tick joins are arbitrated in the existing stable
  decision order; a join is accepted only while the session is open and below cap.
- RELEASE: a member's offset stops at expiry, when its own position or a locally
  witnessed initiator position reaches the merge zone, or after it receives a
  release message (the initiator publishes ``released`` once on reaching the zone).

(2) ``coop_awareness`` (cooperation-layer inputs only): K in {4, 6} V2V publisher slots
within ``range_m`` (>= the 100 m V2V range): per slot gap/lane offset/speed/present plus
a merge-pressure block over those publishers' t-1 reports. Fed to the residual layer
(inserted before the committed-offset column) and to the trigger context. All new
columns are ZERO-initialised on top of the unchanged default initialisation, so an
on-but-untrained policy behaves exactly like the off policy. Candidate slates and the
frozen MAPPO base observation (x / v2v / rsu) are unchanged.

(3) Bank ingestion: with ``v2v_wall`` on, every verified wall-family example is taught
through the protocol: the initiator imitates ``wall_initiate`` from its own observation
at its decision second; every other member imitates ``wall_join`` at the earliest
second it actually received the reconstructed t-1 message (own position within range),
admitted in stable order up to the cap. Inputs are only that CAV's own sensing, its
local V2V packets and the received message; the initiator is the member reaching the
most other members within the message range (ties: plan member order). This is a
label-assignment rule over the verified plan, never a policy input.

Config (absent/False = off, byte-identical):
  v2v_wall = dict(enabled=True, member_cap=2..6, rule=WALL_RULE, provenance=str,
                  [range_m=float > 0, default V2V_RANGE_M])
  coop_awareness = dict(enabled=True, slots=4|6, range_m=float >= V2V_RANGE_M,
                        rule=AWARENESS_RULE, provenance=str)
"""
from copy import deepcopy
from dataclasses import dataclass, replace
import math

import torch
from torch import nn

from . import moves
from .merge_features import (CONTROL_DT_S, MERGE_DIM, MergeFeaturesConfig, _valid_report,
                             merge_features)
from .v2v_obs import V2V_RANGE_M

WALL_RULE = 'w1-v2v-wall-initiate-join-v1'
AWARENESS_RULE = 'w1-coop-awareness-v1'
FAMILIES = ('wall_initiate', 'wall_join')
MAX_MEMBER_CAP = 6
AWARENESS_SLOTS = (4, 6)
AWARENESS_SLOT_FEATURES = ('gap_m', 'lane_offset', 'v_mps', 'present')
WALL_DESCRIPTOR = ('is_initiate', 'is_join', 'initiator_gap_over_range',
                   'member_count_over_cap', 'open_slots_over_max_cap')
ZONE = (moves._ZONE_S0, moves._ZONE_S1)


# ---------------------------------------------------------------- settings

def _registered(raw, keys, optional, rule, name):
    if (not isinstance(raw, dict) or not set(keys) <= set(raw) or set(raw)-set(keys)-set(optional)
            or raw['enabled'] is not True or raw['rule'] != rule
            or not isinstance(raw['provenance'], str) or not raw['provenance'].strip()):
        raise ValueError(f'{name} requires enabled=True, the registered rule and provenance')


def settings(cfg):
    """Resolved protocol settings, or None when ``v2v_wall`` is absent/False (default)."""
    raw = (cfg or {}).get('v2v_wall', False)
    if raw is False or raw is None:
        return None
    _registered(raw, ('enabled', 'member_cap', 'rule', 'provenance'), ('range_m',), WALL_RULE, 'v2v_wall')
    cap = raw['member_cap']
    if type(cap) is not int or not 2 <= cap <= MAX_MEMBER_CAP:
        raise ValueError(f'v2v_wall member_cap must be a registered integer in 2..{MAX_MEMBER_CAP}')
    range_m = raw.get('range_m', V2V_RANGE_M)
    if isinstance(range_m, bool) or not isinstance(range_m, (int, float)) or not (
            math.isfinite(range_m) and range_m > 0):
        raise ValueError('v2v_wall range_m must be finite and positive')
    return dict(member_cap=cap, range_m=float(range_m), rule=WALL_RULE, provenance=raw['provenance'])


def awareness_settings(cfg):
    """Resolved awareness settings, or None when ``coop_awareness`` is absent/False."""
    raw = (cfg or {}).get('coop_awareness', False)
    if raw is False or raw is None:
        return None
    _registered(raw, ('enabled', 'slots', 'range_m', 'rule', 'provenance'), (), AWARENESS_RULE,
                'coop_awareness')
    if type(raw['slots']) is not int or raw['slots'] not in AWARENESS_SLOTS:
        raise ValueError(f'coop_awareness slots must be one of {AWARENESS_SLOTS}')
    range_m = raw['range_m']
    if isinstance(range_m, bool) or not isinstance(range_m, (int, float)) or not (
            math.isfinite(range_m) and range_m >= V2V_RANGE_M):
        raise ValueError(f'coop_awareness range_m must be finite and >= {V2V_RANGE_M}')
    return dict(slots=raw['slots'], range_m=float(range_m), rule=AWARENESS_RULE,
                provenance=raw['provenance'])


def awareness_width(aware):
    return 0 if aware is None else aware['slots']*len(AWARENESS_SLOT_FEATURES)+MERGE_DIM


def config_keys(*, provenance, member_cap=None, range_m=None, awareness_slots=None, awareness_range_m=None):
    """Exact binding keys for the common B/R config (either option may be enabled alone)."""
    out = {}
    if member_cap is not None:
        out['v2v_wall'] = dict(enabled=True, member_cap=int(member_cap), rule=WALL_RULE,
                               provenance=provenance, **({} if range_m is None else dict(range_m=float(range_m))))
    if awareness_slots is not None:
        out['coop_awareness'] = dict(enabled=True, slots=int(awareness_slots),
            range_m=float(V2V_RANGE_M if awareness_range_m is None else awareness_range_m),
            rule=AWARENESS_RULE, provenance=provenance)
    return out


def _second(value):
    value = float(value)
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError('V2V wall protocol runs on integer control seconds')
    return int(value)


# ---------------------------------------------------------------- option / message

@dataclass(frozen=True)
class WallOption:
    """One per-CAV wall-protocol option; members is always (ego,).

    onset/hold are RELATIVE to the decision second (a join carries the remaining part
    of the initiator's absolute schedule). gap_m/member_count come from the received
    t-1 message (0/1 for an initiate option)."""
    family: str
    members: tuple
    onset: int
    hold: int
    depth: float
    wall_id: str
    initiator: str
    gap_m: float = 0.
    member_count: int = 1
    cap: int = 2
    range_m: float = V2V_RANGE_M

    def __post_init__(self):
        object.__setattr__(self, 'members', tuple(self.members))
        if self.family not in FAMILIES or len(self.members) != 1:
            raise ValueError('wall option: registered family and exactly the acting CAV')
        if (not isinstance(self.wall_id, str) or not self.wall_id or not isinstance(self.initiator, str)
                or (self.family == 'wall_initiate' and self.initiator != self.members[0])
                or (self.family == 'wall_join' and self.initiator == self.members[0])):
            raise ValueError('wall option initiator/wall id mismatch')
        for key in ('onset', 'hold', 'member_count', 'cap'):
            if type(getattr(self, key)) is not int:
                raise ValueError(f'wall option {key} must be an integer')
        if (not moves._in_window(self.onset, moves._ONSET_LO, moves._ONSET_HI)
                or not moves._in_window(self.hold, moves._HOLD_LO, moves._HOLD_HI)
                or not moves._in_window(self.depth, moves._DEPTH_LO, moves._DEPTH_HI)):
            raise ValueError('wall option outside the registered variety windows')
        if not 2 <= self.cap <= MAX_MEMBER_CAP or not 1 <= self.member_count < self.cap:
            raise ValueError('wall option membership outside the registered cap')
        if not (math.isfinite(self.gap_m) and math.isfinite(self.range_m) and self.range_m > 0):
            raise ValueError('wall option geometry must be finite')

    @property
    def onset_s(self):
        return self.onset

    @property
    def hold_s(self):
        return self.hold

    @property
    def actors(self):
        return self.members

    def to_dict(self):
        return dict(family=self.family, members=list(self.members), onset=int(self.onset),
                    hold=int(self.hold), depth=float(self.depth), wall_id=self.wall_id,
                    initiator=self.initiator, gap_m=float(self.gap_m),
                    member_count=int(self.member_count), cap=int(self.cap), range_m=float(self.range_m))

    @classmethod
    def from_dict(cls, data):
        return cls(data['family'], tuple(data['members']), int(data['onset']), int(data['hold']),
                   float(data['depth']), data['wall_id'], data['initiator'], float(data['gap_m']),
                   int(data['member_count']), int(data['cap']), float(data['range_m']))


def is_option(move):
    return isinstance(getattr(move, 'move', move), WallOption)


def option_from_dict(data):
    """Recorded table entry -> option (wall protocol or shared move vocabulary)."""
    return WallOption.from_dict(data) if data['family'] in FAMILIES else moves.Move.from_dict(data)


@dataclass(frozen=True)
class WallIntent:
    """V2V wall message published at ``timestamp_s`` by the initiator (visible at t+1)."""
    wall_id: str
    initiator: str
    timestamp_s: int
    source_s_m: float
    source_lane: int
    start_s: int
    onset: int
    hold: int
    depth: float
    zone: tuple
    member_count: int
    cap: int
    released: bool

    @property
    def end_s(self):
        return self.start_s+self.onset+self.hold

    def to_dict(self):
        return dict(wall_id=self.wall_id, initiator=self.initiator, timestamp_s=self.timestamp_s,
                    source_s_m=float(self.source_s_m), source_lane=int(self.source_lane),
                    start_s=self.start_s, onset=self.onset, hold=self.hold, depth=float(self.depth),
                    zone=[float(z) for z in self.zone], member_count=self.member_count, cap=self.cap,
                    released=bool(self.released))


# ---------------------------------------------------------------- candidates

def _initiate(own, wall, rng):
    """Wall-initiate option for the acting CAV, or None (mirrors the wall move geometry)."""
    s, v = float(own['s_m']), float(own['v_mps'])
    if s >= moves._ZONE_S0:
        return None
    arrival = (moves._ZONE_S0-s)/v if v > 0 else math.inf
    if arrival < moves._HOLD_LO:
        return None
    onset, hold, depth = moves._sampled(rng)
    onset = min(onset, int(min(moves._ONSET_HI, arrival-moves._HOLD_LO)))
    hold = min(hold, int(min(moves._HOLD_HI, arrival-onset)))
    ego = own['vid']
    return WallOption('wall_initiate', (ego,), onset, hold, depth, wall_id=f'{ego}@{wall["now"]}',
                      initiator=ego, cap=wall['cap'], range_m=wall['range_m'])


def pool(family, wall, own, rng):
    """Feasible options of one protocol family for the acting CAV (local state only)."""
    if family == 'wall_initiate':
        if wall.get('only_join'):
            return []
        option = _initiate(own, wall, rng)
        return [] if option is None else [option]
    if family == 'wall_join':
        return [o for o in wall.get('offers', ()) if o.members == (own['vid'],)]
    raise ValueError(f'unregistered wall protocol family {family!r}')


def redraw(template, wall, own, rng):
    """Slate entry for a chosen family pool: initiate re-samples parameters, join is exact."""
    if template.family == 'wall_initiate':
        return _initiate(own, wall, rng)
    return template


# ---------------------------------------------------------------- runtime protocol

class Protocol:
    """Per-episode protocol state shared by the runtime (like the executor/boards).

    Policy inputs are only the t-1 board messages (via offers) and each CAV's own
    state. Sessions are the actuation arbitration ledger (ownership + cap), never
    a policy input."""
    FAMILIES = FAMILIES

    def __init__(self, wall_settings):
        self.settings = dict(wall_settings)
        self.cap, self.range_m = self.settings['member_cap'], self.settings['range_m']
        self.reset()

    def reset(self):
        self.sessions = {}      # wall_id -> dict(initiator, start_s, onset, hold, depth, members{vid: t}, released)
        self.membership = {}    # vid -> wall_id (own knowledge of its own membership)
        self.board = {}         # wall_id -> WallIntent published at the last publish second
        self.offers = {}        # vid -> tuple(WallOption) computed at the last receive
        self.offered = {}       # vid -> set(wall_id) already shown in a decision slate
        self.release_latch = set()   # (vid, wall_id) release messages received
        self.received_at = None
        self.last_published = []
        self.last_joins = []

    def _expire(self, now):
        for wall_id in [w for w, s in self.sessions.items()
                        if now >= s['start_s']+s['onset']+s['hold']]:
            for vid in self.sessions.pop(wall_id)['members']:
                if self.membership.get(vid) == wall_id:
                    del self.membership[vid]

    def receive(self, now, own_rows):
        """Offers from messages published at now-1; own_rows = {cav: own row at now}."""
        now = _second(now)
        self._expire(now)
        offers, self.received_at = {}, now
        for vid, own in own_rows.items():
            s = float(own['s_m'])
            out = []
            for msg in sorted(self.board.values(), key=lambda m: m.wall_id):
                if msg.timestamp_s != now-CONTROL_DT_S or msg.initiator == vid:
                    continue
                if abs(msg.source_s_m-s) > self.range_m:
                    continue   # radio range: the message is not received
                if msg.released:
                    if self.membership.get(vid) == msg.wall_id:
                        self.release_latch.add((vid, msg.wall_id))
                    continue
                if vid in self.membership or msg.member_count >= msg.cap or s >= ZONE[0]:
                    continue
                begin = msg.start_s+msg.onset
                onset, hold = max(0, begin-now), msg.end_s-max(now, begin)
                if hold < moves._HOLD_LO or not moves._in_window(hold, moves._HOLD_LO, moves._HOLD_HI):
                    continue
                out.append(WallOption('wall_join', (vid,), int(onset), int(hold), float(msg.depth),
                    wall_id=msg.wall_id, initiator=msg.initiator, gap_m=float(msg.source_s_m-s),
                    member_count=int(msg.member_count), cap=int(msg.cap), range_m=self.range_m))
            if out:
                offers[vid] = tuple(sorted(out, key=lambda o: (abs(o.gap_m), o.wall_id)))
        self.offers = offers
        return offers

    def state(self, vid, now):
        """Local protocol view attached to the CAV's candidate state."""
        return dict(now=_second(now), offers=self.offers.get(vid, ()), cap=self.cap, range_m=self.range_m)

    def events(self, members, free):
        """CAVs with a received wall not yet shown in any of their decision slates."""
        return {v for v in members if v in free
                and any(o.wall_id not in self.offered.get(v, ()) for o in self.offers.get(v, ()))}

    def mark_offered(self, ego, table):
        for move in table:
            if move is not None and getattr(move, 'family', None) == 'wall_join':
                self.offered.setdefault(ego, set()).add(move.wall_id)

    def start(self, move, *, ego, executor, observable_cav_ids, now):
        """Atomic protocol arbitration + executor commitment. Returns accepted."""
        option = getattr(move, 'move', move)
        now = _second(now)
        if not isinstance(option, WallOption) or option.members != (ego,):
            raise ValueError('wall protocol option must act through the deciding CAV only')
        if ego in self.membership:
            return False
        if option.family == 'wall_initiate':
            if option.wall_id in self.sessions or option.wall_id != f'{ego}@{now}':
                return False
            until = now+option.onset+option.hold
            if not executor.start(move, ego=ego, observable_cav_ids=observable_cav_ids, now=now, until=until):
                return False
            self.sessions[option.wall_id] = dict(initiator=ego, start_s=now, onset=option.onset,
                hold=option.hold, depth=float(option.depth), members={ego: now}, released=False)
            self.membership[ego] = option.wall_id
            return True
        session = self.sessions.get(option.wall_id)
        if session is None or session['released'] or len(session['members']) >= self.cap:
            return False
        end = session['start_s']+session['onset']+session['hold']
        if now+option.onset+option.hold != end or float(option.depth) != session['depth']:
            raise ValueError('join option schedule differs from the session it names')
        if not executor.start(move, ego=ego, observable_cav_ids=observable_cav_ids, now=now, until=end):
            return False
        session['members'][ego] = now
        self.membership[ego] = option.wall_id
        self.last_joins.append(dict(wall_id=option.wall_id, vid=ego, time_s=now))
        return True

    def open_offline(self, option, now):
        """Offline (bank) reconstruction of an accepted initiation; no executor."""
        self.sessions[option.wall_id] = dict(initiator=option.initiator, start_s=_second(now),
            onset=option.onset, hold=option.hold, depth=float(option.depth),
            members={option.initiator: _second(now)}, released=False)
        self.membership[option.initiator] = option.wall_id

    def admit_offline(self, option, now):
        session = self.sessions.get(option.wall_id)
        if (session is None or session['released'] or len(session['members']) >= self.cap
                or option.members[0] in self.membership):
            return False
        session['members'][option.members[0]] = _second(now)
        self.membership[option.members[0]] = option.wall_id
        return True

    def publish(self, now, sources):
        """Initiators broadcast at ``now``; sources = {cav: own ObservedVehicle at now}."""
        now = _second(now)
        self._expire(now)
        board = {}
        for wall_id, s in sorted(self.sessions.items()):
            source = sources.get(s['initiator'])
            if s['released'] or source is None:
                continue   # departed/missing initiator: no broadcast this second
            released = float(source.s_m) >= ZONE[0]
            known = 1+sum(1 for v, t in s['members'].items() if v != s['initiator'] and t <= now-1)
            board[wall_id] = WallIntent(wall_id, s['initiator'], now, float(source.s_m), int(source.lane),
                s['start_s'], s['onset'], s['hold'], s['depth'], ZONE, int(known), self.cap, released)
            if released:
                s['released'] = True
        self.board = board
        self.last_published = [m.to_dict() for m in board.values()]
        return board

    def offsets(self, option, t, positions, vid):
        """Committed offset for one member; t = seconds since its own commitment start."""
        t = _second(t)
        active = option.onset <= t < option.onset+option.hold
        if active and ((vid, option.wall_id) in self.release_latch
                       or any(float(positions[v]) >= ZONE[0] for v in {vid, option.initiator}
                              if v in positions)):
            active = False
        return {vid: -float(option.depth) if active else 0.}

    def receipt(self):
        out = dict(rule=WALL_RULE, published=self.last_published,
                   offers={v: [o.wall_id for o in opts] for v, opts in sorted(self.offers.items())},
                   joins=list(self.last_joins),
                   sessions=[dict(wall_id=w, initiator=s['initiator'], members=len(s['members']),
                                  released=s['released']) for w, s in sorted(self.sessions.items())])
        self.last_joins = []
        return out


# ---------------------------------------------------------------- trigger extension

class TriggerExtension:
    """Extra trigger inputs; every new first-layer column is zero-initialised.

    descriptor: WALL_DESCRIPTOR per candidate (wall protocol only).
    context: per ego [own merge-pressure block (wall)] + [awareness block (awareness)],
    read from the recorded context (``merge_features`` / ``coop_awareness``)."""

    def __init__(self, wall, aware):
        self.wall, self.aware = wall, aware
        self.families = FAMILIES if wall else ()
        self.descriptor_width = len(WALL_DESCRIPTOR) if wall else 0
        self.context_width = (MERGE_DIM if wall else 0)+awareness_width(aware)

    def descriptor(self, move):
        if not self.descriptor_width:
            return []
        family = None if move is None else (move['family'] if isinstance(move, dict) else move.family)
        if family not in FAMILIES:
            return [0.]*self.descriptor_width
        get = (lambda k: move[k]) if isinstance(move, dict) else (lambda k: getattr(move, k))
        cap, count = int(get('cap')), int(get('member_count'))
        gap = float(get('gap_m'))/float(get('range_m'))
        return [float(family == 'wall_initiate'), float(family == 'wall_join'),
                max(-1., min(1., gap)), count/cap, (cap-count)/MAX_MEMBER_CAP]

    def context_features(self, context):
        rows = []
        for vid in context['members']:
            row = []
            for key, on, width in (('merge_features', self.wall, MERGE_DIM),
                                   ('coop_awareness', self.aware, awareness_width(self.aware))):
                if not on:
                    continue
                block = (context.get(key) or {}).get(vid)
                if block is None or len(block) != width:
                    raise ValueError(f'trigger extension requires recorded local {key} for every member')
                row += [float(x) for x in block]
            rows.append(row)
        out = torch.tensor(rows, dtype=torch.float32).reshape(len(rows), self.context_width)
        if not bool(torch.isfinite(out).all()):
            raise ValueError('nonfinite trigger extension features')
        return out


def extension(cfg):
    wall, aware = settings(cfg), awareness_settings(cfg)
    return None if wall is None and aware is None else TriggerExtension(wall, aware)


def widen_columns(linear, insert_at, width):
    """Same Linear with ``width`` zero input columns inserted at ``insert_at``."""
    if width <= 0:
        return linear
    # The throwaway default init must not advance the caller's global CPU RNG.
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        weight = linear.weight.detach()
        zeros = weight.new_zeros((weight.shape[0], width))
        widened = nn.Linear(weight.shape[1]+width, weight.shape[0], bias=linear.bias is not None)
        widened.weight.copy_(torch.cat((weight[:, :insert_at], zeros, weight[:, insert_at:]), 1))
        if linear.bias is not None:
            widened.bias.copy_(linear.bias.detach())
    return widened


def widen_layer(layer, aware):
    """Insert zero awareness columns before the committed-offset column of the residual."""
    width = awareness_width(aware)
    if not width:
        return layer
    first = layer.residual[0]
    layer.residual[0] = widen_columns(first, first.in_features-1, width)
    layer.config = replace(layer.config, feature_dim=layer.config.feature_dim+width)
    layer.awareness_width = width
    return layer


# ---------------------------------------------------------------- awareness block

def awareness_block(own_report, ego, previous_reports, now, aware):
    """K nearest other CAV publishers (valid own t-1 packets within range) + merge pressure."""
    k, range_m = aware['slots'], aware['range_m']
    accepted = []
    valid_own = _valid_report(own_report, ego, float(now))
    if valid_own:
        own = own_report.source
        for vid, report in sorted((previous_reports or {}).items()):
            if vid == ego or not _valid_report(report, vid, float(now)-CONTROL_DT_S):
                continue
            gap = float(report.source.s_m)-float(own.s_m)
            if abs(gap) <= range_m:
                accepted.append((abs(gap), vid, report))
    accepted.sort(key=lambda e: (e[0], e[1]))
    slots = []
    for i in range(k):
        if i < len(accepted):
            src = accepted[i][2].source
            slots += [float(src.s_m)-float(own.s_m), float(src.lane-own.lane), float(src.v_mps), 1.]
        else:
            slots += [0., 0., 0., 0.]
    wide = merge_features(own_report if valid_own else None, ego, previous_reports, now_s=float(now),
                          partner_ids=[e[1] for e in accepted], cfg=MergeFeaturesConfig(True),
                          k=k, range_m=range_m)
    return slots+[float(x) for x in wide]


def missing_awareness(aware):
    """Explicitly missing block (no own packet): zero slots + merge missingness bits."""
    return [0.]*(aware['slots']*len(AWARENESS_SLOT_FEATURES))+[
        float(x) for x in merge_features(None, '_', now_s=0., cfg=MergeFeaturesConfig(True))]


# ---------------------------------------------------------------- bank ingestion

def attach_context(context, reader, members):
    """Trigger-extension inputs from the reader's own causal observation (no-op when off)."""
    ext = getattr(getattr(reader, 'trigger', None), 'extension', None)
    if ext is None:
        return context
    if ext.wall:
        context['merge_features'] = {v: list(reader.blocks[v]) for v in members}
    if ext.aware:
        context['coop_awareness'] = {v: list(reader.awareness_blocks[v]) for v in members}
    return context


def protocol_label(plan, label, cap):
    """(members, onset, hold, depth) of a verified wall plan, or None (not a wall)."""
    if label.family != 'wall':
        return None
    members, matrix = moves._plan_offsets(plan)
    if len(label.members) < len(members) <= cap:
        params, gain = moves._project_window(matrix, list(range(len(members))))
        if gain > 0:
            return tuple(members), *params
    return tuple(label.members), int(label.onset), int(label.hold), float(label.depth)


def protocol_contexts(policy, plan, label, pairs, frames, *, t0):
    """Initiate/join imitation contexts for one verified wall example (see module doc).

    Yields (context, option, reader, weight, time) like bank_breadth.observable_contexts.
    The reader replays every available second of the window with the SAME causal
    observer and protocol as online execution; a member is labelled ``wall_join`` only
    at a second where its own receive step produced the offer for this wall.
    """
    cfg = policy.wall.settings
    wall = protocol_label(plan, label, cfg['member_cap'])
    if wall is None:
        return
    members, onset, hold, depth = wall
    by_t = {}
    for sample, weight in sorted(pairs, key=lambda pair: int(pair[0]['intent_link']['t'])):
        by_t.setdefault(int(sample['intent_link']['t']), (sample, weight))
    reader = deepcopy(policy)
    reader.wall.reset()
    t0 = int(t0)
    initiator, wall_id, remaining = None, None, set()

    def build(source, keep):
        context = dict(source, members=list(keep),
                       x=[source['x'][source['members'].index(v)] for v in keep],
                       trigger_v2v={v: reader.local_intents[v] for v in keep})
        return attach_context(context, reader, keep)

    for t in range(t0-1, t0+onset+hold):
        if t not in frames:
            continue
        reader.observe(frames[t]['vehicles'], t)
        if t < t0:
            continue
        elapsed = t-t0
        left_onset, left_hold = max(0, onset-elapsed), hold-max(0, elapsed-onset)
        if left_hold < moves._HOLD_LO:
            break   # no initiate/join with less than the registered minimum hold remains
        pair = by_t.get(t)
        if initiator is None:
            if pair is None:
                continue
            source = pair[0]['context']
            rows = {r['vid']: r for r in frames[t]['vehicles']}
            candidates = [v for v in members if v in source['members'] and v in reader.local_intents]
            if not candidates:
                continue
            def reach(v):
                return sum(1 for m in members if m != v and m in rows
                           and abs(float(rows[m]['s_m'])-float(rows[v]['s_m'])) <= cfg['range_m'])
            initiator = max(candidates, key=lambda v: (reach(v), -members.index(v)))
            wall_id = f'{initiator}@{t}'
            option = WallOption('wall_initiate', (initiator,), left_onset, left_hold, depth,
                                wall_id=wall_id, initiator=initiator, cap=cfg['member_cap'],
                                range_m=cfg['range_m'])
            remaining = set(members)-{initiator}
            yield build(source, [initiator]), option, reader, float(pair[1]), t
            reader.wall.open_offline(option, t)
        elif pair is not None and remaining:
            source = pair[0]['context']
            for vid in sorted(remaining):
                if vid not in source['members'] or vid not in reader.local_intents:
                    continue
                offer = next((o for o in reader.wall.offers.get(vid, ()) if o.wall_id == wall_id), None)
                if offer is None or not reader.wall.admit_offline(offer, t):
                    continue
                remaining.discard(vid)
                yield build(source, [vid]), offer, reader, float(pair[1]), t
        if initiator is not None:
            reader.wall.publish(t, {v: r.source for v, r in reader.reports.items()})
            if not remaining:
                break
