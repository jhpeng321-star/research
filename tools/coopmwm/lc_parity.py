"""LCEXEC TraCI parity harness over ALL verified mined rows (CPU only: fake TraCI, no SUMO).

``run(row, program, frame=..., stream='log'|'random', seed=0)`` drives, on the SAME per-second state
stream (edge, SUMO lane, global x, v, allowed, live) of the members and every referenced vehicle:

* PROBE side: the verbatim ``class Arm`` of the row's source core (sha-pinned fixture copies in
  ``fixtures/lcexec/probes/<source>/``, loaded with importlib with their own ``population`` copy; no SUMO
  import), exactly as its ``run_branch`` did (``_positions`` of members + anchors + partners + track refs
  every whole second, then ``Arm.control``). LCMOVES rows use the S-LCMOVES reference executor
  ``lcm_families.Executor`` (fixture copies of lcm_families / lcm_core / lcm_plan).
* OUR side: one ``lc_program.MemberExecutor`` per member on ``Obs`` built from the same states (WM frame:
  s = frame.s(x), lane = frame.wm(sumo), on_approach = edge == 'up', sensed = vehicles within 100 m), ->
  ``lc_writer_harness.requests_from_commands`` -> ``Harness.tick`` (the REAL LCEXEC env subclass +
  CommandOwnership, native base, speedMode 27).

Both sides' TraCI writes are compared per (el, vid) as multisets: slowDown target (|d| <= 1e-9),
changeLane (SUMO index, duration), setLaneChangeMode (incl. restore), setTau (incl. restore),
setSpeed(-1), setSpeedMode. Documented normalisations only:

* N1: ``setSpeed(vid, -1)`` without a live speed command (no slowDown to vid at el-1 on that side) is a
  SUMO no-op: dropped on both sides;
* N2: our ``setSpeed(vid, -1)`` at the first non-commanded second after a commanded one (the owner hands
  the CAV back) == the probe's slowDown(.., 1 s) expiry (the probe issued nothing at el).

Every other difference is a mismatch. A probe changeLane the env writer did not issue is classified by
the writer's per-vehicle receipt of that tick: ``writer_guard`` (``deferred_edge_end``: edge-end deferral
within v*duration + 10 m of the lane end) or ``writer_refused`` (``refused_*``, e.g. a non-adjacent
target); anything else is ``mismatch``.

Streams. ``log``: reconstructed from the row so the probe reproduces the row's OWN log (issued accels:
NEWMAP/PLANGEN/LOWCAV/PLANGEN2/LCMOVES = profile on commanded seconds, COOPKINDS = its 'a' commands;
changeLane / lcMode / tau / laneHold commands, release_el and arm_events per member): speeds are chosen so
the law yields the logged accel (v = v_w - a when unsaturated, inside the saturation region otherwise;
zip_yield from the tracking law; LCMOVES ``match`` via one phantom HDV per matching second on the target
lane), lanes from the commands / member_lane_tracks, positions (0.01 m grid, speeds 1e-4 m/s: exactly the
probe's ``_positions`` rounding, so rounding is idempotent) from per-second interval constraints so every
release / arming condition triggers exactly at the logged second, following a nominal path from the
profile (x_t0/v_t0 when the row has them). ``log_check`` = the probe run on this stream reproduces the
row log (accel within 1e-4 = round(a, 4) plus float, commands exact, release_el exact, events exact).
``random``: seeded plausible random streams (members + references + extra HDVs, lanes change by one,
speeds 0-30 m/s, positions moving forward across every registered threshold, random initial lcMode/tau).
Dither vehicles (non-member pulses) are not part of the move: excluded on both sides.

CLI: ``python -m tools.coopmwm.lc_parity report OUT.json [--random N]`` (all fixture rows).
"""
from __future__ import annotations

from collections import Counter, defaultdict
import functools
import gzip
import hashlib
import json
import math
from pathlib import Path
import random as _random
import sys
import types

from . import lc_program as lp
from .lc_convert import (NotRepresentable, dither_ids, from_mined_row, member_specs, move_of, row_key, row_meta,
                         source_of)

FIX = Path(__file__).with_name('fixtures')/'lcexec'
ROW_FILES = ('NEWMAP_MOVES_B', 'NEWMAP_MOVES_R', 'PLANGEN_MOVES_B', 'PLANGEN_MOVES_R', 'PLANGEN2_MOVES_B',
             'PLANGEN2_MOVES_R', 'COOPKINDS_MOVES_B', 'LOWCAV_MOVES_B', 'LCMOVES_MOVES_B')
CORES = {'newmap2': 'nm2_core', 'plangen': 'nm2_core', 'lowcav': 'lc_core', 'plangen2': 'pg2_core',
         'coopkinds': 'ck_core', 'lcmoves': 'lcm_core'}
NM2_SOURCES = ('newmap2', 'plangen', 'lowcav')
COMPARED = ('slowDown', 'setSpeed', 'changeLane', 'setLaneChangeMode', 'setTau', 'setSpeedMode')
TOL = 1e-9
A_TOL = 1e-4 + 1e-9                 # probe logs round(a, 4)
ALLOWED = 31.29                     # probe SPEED_LIMIT (every lane of wz3to2)
V_MAX = 30.0
INF = 10**12
N_RANDOM = 3


# ---------------------------------------------------------------- fixtures
@functools.lru_cache(maxsize=None)
def manifest():
    return json.loads((FIX/'MANIFEST.json').read_text(encoding='utf-8'))


def _verified(rel, meta):
    raw = (FIX/rel).read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta['sha256']:
        raise ValueError(f'fixture {rel} does not match its MANIFEST sha256')
    return raw


@functools.lru_cache(maxsize=None)
def fixture_rows(name):
    """[(row index, row)] of one verified row file (gz copy; sha256 of the original bytes checked)."""
    meta = manifest()['rows'][name]
    raw = gzip.decompress((FIX/'rows'/f'{name}.jsonl.gz').read_bytes())
    if hashlib.sha256(raw).hexdigest() != meta['sha256']:
        raise ValueError(f'row fixture {name} does not match its MANIFEST sha256')
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if len(rows) != meta['rows']:
        raise ValueError(f'row fixture {name}: {len(rows)} rows, manifest {meta["rows"]}')
    return tuple(enumerate(rows))


def all_rows():
    """[(file stem, index, row)] over every non-empty verified row file."""
    return [(name, i, row) for name in ROW_FILES for i, row in fixture_rows(name)]


def _exec_module(name, rel):
    """Execute a sha-pinned .pysrc fixture as module ``name`` (registered in sys.modules)."""
    meta = manifest()['probes'][rel]
    raw = _verified(f'probes/{rel}', meta)
    mod = types.ModuleType(name)
    mod.__file__ = str(FIX/'probes'/rel)
    sys.modules[name] = mod
    exec(compile(raw, mod.__file__, 'exec'), mod.__dict__)
    return mod


@functools.lru_cache(maxsize=None)
def probe(source):
    """Verbatim probe code of one source: namespace(core, Arm, positions, families or None)."""
    core = CORES[source]
    saved_path = list(sys.path)
    saved_mods = {k: sys.modules.get(k) for k in ('population', 'lcm_core', 'lcm_plan')}
    before = set(sys.modules)
    try:
        pop = _exec_module(f'_lcexec_cv_{source}_population', f'{source}/population.pysrc')
        sys.modules['population'] = pop
        mod = _exec_module(f'_lcexec_cv_{source}_{core}', f'{source}/{core}.pysrc')
        fam = None
        if source == 'lcmoves':
            sys.modules['lcm_core'] = mod
            sys.modules['lcm_plan'] = _exec_module('_lcexec_cv_lcmoves_lcm_plan', 'lcmoves/lcm_plan.pysrc')
            fam = _exec_module('_lcexec_cv_lcmoves_lcm_families', 'lcmoves/lcm_families.pysrc')
    finally:
        sys.path[:] = saved_path
        for k, v in saved_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    sumo = {m for m in set(sys.modules)-before if m.split('.')[0] in ('traci', 'libsumo', 'sumolib')}
    if sumo:
        raise RuntimeError(f'probe fixture imported SUMO at load: {sorted(sumo)}')
    return types.SimpleNamespace(source=source, core=mod, Arm=mod.Arm, positions=mod._positions, X_B=mod.X_B,
                                 families=fam)


# ---------------------------------------------------------------- geometry helpers
class Geo:
    """wz3to2 probe geometry from the N1 descriptor (approach edge, junction, downstream edge)."""

    def __init__(self, desc, frame):
        self.frame = frame
        self.edge = frame.edge
        self.drop = float(desc.bottleneck['drop_x_m'])
        down = sorted({e for (e, _), _ in desc.lane_table.items() if e != frame.edge})
        if len(down) != 1:
            raise ValueError('expected one downstream edge')
        self.down = down[0]
        self.down_lanes = {lane: i for (e, i), (lane, _) in desc.lane_table.items() if e == self.down}
        self.down_x0 = min(x0 for (e, _), (_, x0) in desc.lane_table.items() if e == self.down)
        self.junction = next(lid.rsplit('_', 1)[0] for lid in desc.lane_end_distance if lid.startswith(':'))
        n_j = sum(1 for lid in desc.lane_end_distance if lid.startswith(':'))
        self.junction_lanes = n_j
        self.end = self.drop - float(desc.lane_end_distance[f'{self.down}_0'])
        self.n_up = len(frame.sumo_to_wm)

    def down_lane(self, up_lane):
        wm = self.frame.wm(up_lane)
        if wm not in self.down_lanes:                     # closing lane -> the adjacent through lane
            wm = self.frame.adjacent_lane
        return self.down_lanes[wm]

    def state(self, x_cm, up_lane, v, cav):
        x = x_cm/100.0
        if x < self.drop:
            edge, lane = self.edge, int(up_lane)
        elif x < self.down_x0:
            edge, lane = self.junction, min(self.down_lane(up_lane), self.junction_lanes-1)
        else:
            edge, lane = self.down, self.down_lane(up_lane)
        return dict(edge=edge, sumo_lane=lane, x=x, v=float(v), allowed=ALLOWED, is_cav=bool(cav))


def _grid(c):
    c100 = float(c)*100.0
    ci = round(c100)
    return (int(ci), True) if abs(c100-ci) < 1e-6 else (math.floor(c100), False)


def cm_lt(c):
    ci, on = _grid(c)
    return ci-1 if on else ci


def cm_le(c):
    return _grid(c)[0]


def cm_ge(c):
    ci, on = _grid(c)
    return ci if on else ci+1


def cm_gt(c):
    return _grid(c)[0]+1


class StreamError(ValueError):
    pass


class _Track:
    """Per-vehicle per-second requirements of a log stream (positions in integer cm)."""

    def __init__(self, vid, n, cav, geo):
        self.vid, self.n, self.cav = vid, n, cav
        self.lo = [0]*n
        self.hi = [cm_lt(geo.end)-1000]*n
        self.lane = [None]*n
        self.vpin = [None]*n
        self.vrng = [None]*n
        self.gone = None
        self.lane0 = 1
        self.x_t0 = None
        self.v_t0 = None
        self.acc = [0.0]*n
        self.why = defaultdict(list)

    def live(self, el):
        return self.gone is None or el < self.gone

    def ge(self, el, c, why=''):
        if 0 <= el < self.n:
            self.lo[el] = max(self.lo[el], cm_ge(c))
            self.why[el].append(f'>= {c} {why}')

    def gt(self, el, c, why=''):
        if 0 <= el < self.n:
            self.lo[el] = max(self.lo[el], cm_gt(c))
            self.why[el].append(f'> {c} {why}')

    def lt(self, el, c, why=''):
        if 0 <= el < self.n:
            self.hi[el] = min(self.hi[el], cm_lt(c))
            self.why[el].append(f'< {c} {why}')

    def le(self, el, c, why=''):
        if 0 <= el < self.n:
            self.hi[el] = min(self.hi[el], cm_le(c))
            self.why[el].append(f'<= {c} {why}')

    def need_lane(self, el, lane, why=''):
        if 0 <= el < self.n:
            if self.lane[el] is not None and self.lane[el] != lane:
                raise StreamError(f'{self.vid}: lane {self.lane[el]} and {lane} both required at el {el} ({why})')
            self.lane[el] = lane

    def pin_v(self, el, v):
        if 0 <= el < self.n:
            self.vpin[el] = v

    def range_v(self, el, lo, hi):
        if 0 <= el < self.n:
            self.vrng[el] = (lo, hi)


def _accel_pin(T, el, a, v_w, amax, dmax):
    """Speed that makes clip(v_w - v, -dmax, amax) == a (pinned) or its saturation region."""
    a = float(a)
    if a <= -dmax+TOL:
        T.range_v(el, v_w+dmax, V_MAX+1.0)
    elif a >= amax-TOL:
        T.range_v(el, 0.0, v_w-amax)
    else:
        T.pin_v(el, round(v_w-a, 4))


def _profile(row, i):
    prof = row.get('profile') or []
    return list(prof[i]) if i < len(prof) and prof[i] is not None else []


def _is_issued_value(a, amax, dmax):
    """Inferred issued accel in a 'issued while commanded, else measured' profile (round(a, 4) values)."""
    return a is not None and abs(a-round(a, 4)) < 1e-12 and -dmax-TOL <= a <= amax+TOL


def _row_commands(row, members):
    """Row command log [(el, vid, kind, value)] of members only (both probe formats)."""
    raw = row.get('commands')
    if raw is None:
        return []
    if isinstance(raw, dict):
        out = [(int(el), vid, kind, value) for vid, rows in raw.items() for el, kind, value in rows]
    else:
        out = [(int(el), vid, kind, value) for el, vid, kind, value in raw]
    return [c for c in out if c[1] in members]


def _release_reasons(row, members):
    out = {}
    for ev in row.get('arm_events') or ():
        el, vid, kind, why = ev
        if vid in members and kind in ('release', 'left'):
            out[vid] = (int(el), 'left' if kind == 'left' else why)
    return out


# ---------------------------------------------------------------- log stream
class LogStream:
    """The reconstructed log-consistent stream of one row (see the module docstring)."""

    def __init__(self, row, program, frame, desc, source):
        self.row, self.program, self.frame, self.source = row, program, frame, source
        self.geo = Geo(desc, frame)
        self.specs = member_specs(row)
        self.members = [m['id'] for m in self.specs]
        self.E = int(row['duration_s'])
        self.n = self.E+1
        self.cmds = _row_commands(row, set(self.members))
        self.reasons = _release_reasons(row, set(self.members))
        self.expected_accel = {}          # vid -> {el: a} logged issued accel
        self.match = {}                   # vid -> {el: target accel} (LCMOVES match seconds)
        self.notes = []
        T = self.tracks = {}
        for i, m in enumerate(self.specs):
            T[m['id']] = t = _Track(m['id'], self.n, True, self.geo)
            t.lane0 = int(m['lane'])
            t.x_t0, t.v_t0 = m.get('x_t0'), m.get('v_t0')
            prof = _profile(row, i)
            t.acc = [float(prof[el]) if el < len(prof) and prof[el] is not None else 0.0
                     for el in range(self.n)]
        self._own()
        self._cross()
        self._solve()

    # ---- helpers
    def _cl(self, vid, kind='changeLane'):
        return sorted({el for el, v, k, _ in self.cmds if v == vid and k == kind})

    def _window(self, vid, p):
        st = int(p.get('start_s', 0))
        rel = self.row.get('release_el', {}).get(vid)
        end = self.n if rel is None else int(rel)
        return st, rel, range(st, min(end, self.n))

    def _ref(self, vid):
        if vid not in self.tracks:
            raise StreamError(f'reference {vid!r} is not a member (non-member references not reconstructed)')
        return self.tracks[vid]

    # ---- pass 1: each member's own role
    def _own(self):
        for i, m in enumerate(self.specs):
            vid, role, p = m['id'], m['role'], m['params']
            T = self.tracks[vid]
            st, rel, act = self._window(vid, p)
            reason = self.reasons.get(vid, (None, None))[1]
            if rel is not None and reason is None:
                raise StreamError(f'{vid}: release_el without a release event')
            if reason == 'left':
                T.gone = rel
            if role == 'hold' and self.source != 'coopkinds':
                self._hold(i, vid, p, T, st, rel, act, reason)
            elif role in ('tau', 'merge_early', 'zip_merge', 'zip_yield', 'lane_to', 'prog'):
                self._generic(vid, p, T, st, rel, act, reason)
                getattr(self, '_'+role)(i, vid, p, T, st, rel, act, reason)
            else:
                raise StreamError(f'{vid}: no log-stream reconstruction for role {role!r} ({self.source})')

    def _hold(self, i, vid, p, T, st, rel, act, reason):
        rx = float(p.get('release_x', self.geo.drop))
        v_w, amax, dmax = float(p['v_w']), float(p['amax']), float(p['dmax'])
        acc = {}
        for el in act:
            T.lt(el, rx, 'hold active')
            if el < self.E:
                acc[el] = T.acc[el]
                _accel_pin(T, el, T.acc[el], v_w, amax, dmax)
        self.expected_accel[vid] = acc
        if reason == 'hold_end' and rel != st+int(p['hold_s']):
            self.notes.append(f'{vid}: hold_end at {rel} != start+hold_s')
        if reason == 'at_drop':
            T.ge(rel, rx, 'at_drop')
        elif reason == 'anchor':
            T.lt(rel, rx, 'anchor release (not at drop)')

    def _generic(self, vid, p, T, st, rel, act, reason):
        rx = p.get('release_x')
        if rx is not None:
            for el in act:
                T.lt(el, float(rx), 'release_x')
        if reason == 'release_x':
            T.ge(rel, float(rx), 'release_x')
        elif rx is not None and reason not in (None, 'left', 'hold_end'):
            T.lt(rel, float(rx), 'not release_x at release')
        if reason == 'hold_end' and rel != st+int(p.get('hold_s', 10**6)):
            self.notes.append(f'{vid}: hold_end at {rel} != start+hold_s')

    def _tau(self, i, vid, p, T, st, rel, act, reason):
        taus = self._cl(vid, 'tau')
        el_tau = taus[0] if taus else None
        zone = p.get('zone')
        if zone is not None:
            z0, z1 = float(zone[0]), float(zone[1])
            for el in act:
                if el_tau is None or el < el_tau:
                    T.lt(el, z0, 'before zone')
                else:
                    T.le(el, z1, 'in zone')
            if el_tau is not None:
                T.ge(el_tau, z0, 'zone entry')
            if reason == 'zone_end':
                T.gt(rel, z1, 'zone_end')
            elif rel is not None and reason not in ('release_x', 'left') and el_tau is not None:
                T.le(rel, z1, 'in zone at release')
        elif el_tau is not None and el_tau != st:
            self.notes.append(f'{vid}: tau without zone logged at {el_tau} != start {st}')

    def _lane_plan(self, T, act, cl, src_lane, tl, rel):
        """Lanes of a member whose changeLane seconds are logged (cl): source lane on command seconds,
        the start lane before the first, the target lane in gaps and after the last."""
        first = cl[0] if cl else None
        for el in act:
            if el in cl:
                T.need_lane(el, src_lane, 'changeLane logged')
            elif first is not None and el > first:
                T.need_lane(el, tl, 'after a request')
            else:
                T.need_lane(el, T.lane0, 'before the first request')
        if rel is not None and rel < self.n and first is not None and T.lane[rel] is None:
            T.need_lane(rel, tl, 'released after its requests')

    def _merge_early(self, i, vid, p, T, st, rel, act, reason):
        tl, trig = int(p.get('target_lane', 1)), float(p['trigger_x'])
        cl = self._cl(vid)
        self._lane_plan(T, act, cl, 0, tl, rel)
        for el in act:
            if el in cl:
                T.lt(el, self.geo.drop, 'on approach')
                T.ge(el, trig, 'triggered')
            elif T.lane[el] == 0:
                T.lt(el, trig, 'lane 0 without request')

    def _zip_merge(self, i, vid, p, T, st, rel, act, reason):
        mx, fb = float(p['merge_x']), float(p['fallback_x'])
        cl = set(self._cl(vid))
        for el in act:
            T.need_lane(el, 0, 'zip_merge active')
            T.lt(el, fb, 'before fallback')
            if el in cl:
                T.ge(el, mx, 'merge request')
            else:
                T.lt(el, mx, 'no merge request')
        if reason == 'merged':
            T.need_lane(rel, 1, 'merged')
            T.lt(rel, self.geo.drop, 'merged on approach')
        elif reason == 'fallback':
            T.need_lane(rel, 0, 'fallback in lane 0')
            T.ge(rel, fb, 'fallback')

    def _zip_yield(self, i, vid, p, T, st, rel, act, reason):
        acc = {}
        if self.source == 'coopkinds':
            acc = {el: float(v) for el, v2, k, v in self.cmds if v2 == vid and k == 'a'}
        else:
            acc = {el: T.acc[el] for el in act if el < self.E}
        self.expected_accel[vid] = acc

    def _lane_to(self, i, vid, p, T, st, rel, act, reason):
        tl, gx = int(p['target_lane']), float(p['giveup_x'])
        cl = self._cl(vid)
        src = T.lane0 if T.lane0 < tl else tl-1
        first = cl[0] if cl else None
        for el in act:
            if el in cl:
                T.need_lane(el, src, 'lane_to request')
                T.lt(el, self.geo.drop, 'on approach')
                T.lt(el, gx, 'before give-up')
            elif first is not None and el > first:
                T.need_lane(el, tl, 'reached')
            else:
                T.need_lane(el, T.lane0 if T.lane0 >= tl else tl, 'no request')
        if rel is not None and rel < self.n and T.lane[rel] is None:
            T.need_lane(rel, tl if cl else T.lane0, 'at release')

    def _prog(self, i, vid, p, T, st, rel, act, reason):
        for k in ('release_on_partner_merge', 'merger', 'track_ref'):
            if p.get(k):
                raise StreamError(f'{vid}: prog {k} not reconstructed in log streams')
        tracks = self.row.get('member_lane_tracks')
        if tracks is not None:
            track = tracks[i]
            for el in range(self.n):
                T.need_lane(el, int(track[min(el, len(track)-1)]), 'member_lane_tracks')
        tl = p.get('lc_target')
        lc_done = (self.row.get('lc_done') or {}).get(vid)
        x_b = self.geo.drop
        if tl is not None:
            gx = float(p.get('lc_giveup_x', x_b-300.0))
            lc_from = int(p.get('lc_from', 0))
            if p.get('lc_from_x') is not None:
                raise StreamError(f'{vid}: prog lc_from_x not reconstructed in log streams')
            cl, hold = set(self._cl(vid)), self._cl(vid, 'laneHold')
            h = hold[0] if hold else None
            for el in act:
                if el < lc_from:
                    continue
                if el in cl or el == h:
                    T.lt(el, x_b, 'prog request on approach')
                    T.lt(el, gx, 'prog request before give-up')
                elif (lc_done is None or el < lc_done or (h is None and el >= lc_done)
                      or (h is not None and lc_done <= el < h)):
                    T.ge(el, gx, 'armed but no logged request: past give-up')
        kind = p.get('speed')
        sp_from = int(p.get('sp_from', 0))
        amax, dmax = float(p.get('amax', 1.0)), float(p.get('dmax', 1.0))
        acc = {}
        for el in act:
            if not kind or el < sp_from or el >= self.E:
                continue
            a = T.acc[el]
            if kind == 'match':
                lc_ok = tl is None or (lc_done is not None and el >= lc_done)
                if not lc_ok and _is_issued_value(a, amax, dmax):
                    acc[el] = a
                    T.lt(el, x_b, 'matching on approach')
                    self.match.setdefault(vid, {})[el] = a
            elif kind == 'hold':
                acc[el] = a
                _accel_pin(T, el, a, float(p['v_w']), amax, dmax)
            elif kind == 'accel':
                acc[el] = a
            else:
                raise StreamError(f'{vid}: prog speed {kind!r} not reconstructed in log streams')
        self.expected_accel[vid] = acc

    # ---- pass 2: references between members (anchors, zip partners)
    def _cross(self):
        for m in self.specs:
            vid, role, p = m['id'], m['role'], m['params']
            st, rel, act = self._window(vid, p)
            reason = self.reasons.get(vid, (None, None))[1]
            if role == 'hold' and p.get('anchor') is not None:
                A, arx = self._ref(p['anchor']), float(p['anchor_release_x'])
                for el in act:
                    A.lt(el, self.geo.drop, f'anchor of {vid} on approach')
                    A.lt(el, arx, f'anchor of {vid} before its release x')
                if reason == 'anchor':
                    A.ge(rel, arx, f'anchor of {vid} passed')
            if role == 'zip_yield':
                P = self._ref(p['partner'])
                for el in act:
                    P.need_lane(el, 0, f'partner of {vid} in lane 0')
                    P.lt(el, self.geo.drop, f'partner of {vid} on approach')
                if reason == 'partner_merged':
                    if P.lane[rel] == 0:
                        P.ge(rel, self.geo.drop, f'partner of {vid} left the approach')
                    else:
                        P.need_lane(rel, 1, f'partner of {vid} merged')

    # ---- positions and speeds
    def _vnom(self, T):
        """Nominal speeds: pinned / saturation regions, else integrated profile accelerations."""
        n = T.n
        v = [None]*n
        pins = [el for el in range(n) if T.vpin[el] is not None]
        for el in pins:
            v[el] = T.vpin[el]
        if T.v_t0 is None and pins:                         # back-propagate from the first pinned second
            for el in range(pins[0]-1, -1, -1):
                v[el] = min(max(v[el+1]-T.acc[el], 0.0), V_MAX)
        elif v[0] is None:
            v[0] = float(T.v_t0) if T.v_t0 is not None else 20.0
        for el in range(1, n):
            if v[el] is None:
                v[el] = min(max(v[el-1]+T.acc[el-1], 0.0), V_MAX)
        for el in range(n):
            if T.vrng[el] is not None and T.vpin[el] is None:
                lo, hi = T.vrng[el]
                v[el] = min(max(v[el], lo), hi)
        return v

    def _positions(self, T, v):
        """Monotone positions inside the per-second tube [L, U]: the start follows the nominal path
        (scale/shift fit, x_t0 when known), then the TAUT STRING through the tube (piecewise-constant
        speed that changes only where a constraint forces it)."""
        n = T.n if T.gone is None else T.gone
        L, U = T.lo[:n], T.hi[:n]
        for el in range(1, n):
            L[el] = max(L[el], L[el-1])
        for el in range(n-2, -1, -1):
            U[el] = min(U[el], U[el+1])
        bad = [el for el in range(n) if L[el] > U[el]]
        if bad:
            el = bad[0]
            raise StreamError(f'{T.vid}: infeasible position at el {el}: {T.why.get(el)}')
        d = [0]
        for el in range(1, n):
            d.append(d[-1]+round(50.0*(max(v[el-1], 0.0)+max(v[el], 0.0))))
        x0 = None if T.x_t0 is None else round(float(T.x_t0)*100)
        start, scale = None, None
        for sc in (1.0, .8, .65, .5, .35, .25, .15, .1, .05, .02, 0.0):
            dd = [round(sc*x) for x in d]
            smin = max(L[el]-dd[el] for el in range(n))
            smax = min(U[el]-dd[el] for el in range(n))
            if smin <= smax:
                start, scale = ((smin+smax)//2 if x0 is None else min(max(x0, smin), smax)), sc
                break
        if start is None:
            start = x0 if x0 is not None else L[0]
            self.notes.append(f'{T.vid}: nominal path does not fit the tube (taut string from the start)')
        start = min(max(start, L[0]), U[0])
        end_slope = 100.0*max(v[n-1], 0.0)*(scale if scale is not None else 1.0)
        return _taut_string(L, U, start, end_slope), scale

    def _solve(self):
        self.x, self.v, self.lanes, self.scale = {}, {}, {}, {}
        yields = [m for m in self.specs if m['role'] == 'zip_yield']
        first = [vid for vid in self.tracks if vid not in {m['id'] for m in yields}]
        for vid in first:
            self._solve_track(vid)
        self._separate_tracks(first)
        for m in yields:
            self._zip_band(m)
            self._solve_track(m['id'])
            self._zip_speeds(m)
        self.extra = defaultdict(dict)           # el -> {vid: (x_cm, up lane, v, cav)} (phantoms)
        self._phantoms()
        self._repair_overlaps()
        self.states = []
        for el in range(self.n):
            st = {}
            for vid in self.members:
                if self.tracks[vid].live(el):
                    st[vid] = self.geo.state(self.x[vid][el], self.lanes[vid][el], self.v[vid][el], True)
            st.update({k: self.geo.state(*e) for k, e in self.extra.get(el, {}).items()})
            self.states.append(st)
        self.roster = {}
        for st in self.states:
            self.roster.update({v: s['is_cav'] for v, s in st.items()})

    def _solve_track(self, vid):
        T = self.tracks[vid]
        v = self._vnom(T)
        xs, s = self._positions(T, v)
        self.x[vid], self.scale[vid] = xs, s
        for el in range(len(xs)):
            if T.vpin[el] is not None:
                continue
            k = el if el+1 < len(xs) else el-1
            slope = (xs[k+1]-xs[k])/100.0 if 0 <= k < len(xs)-1 else v[el]
            if T.vrng[el] is not None:
                slope = min(max(slope, T.vrng[el][0]), T.vrng[el][1])
            v[el] = slope
        self.v[vid] = [round(min(max(vv, 0.0), V_MAX), 4) for vv in v]
        lanes, cur = [], T.lane0
        for el in range(T.n):
            if T.lane[el] is not None:
                cur = T.lane[el]
            lanes.append(cur)
        self.lanes[vid] = lanes

    def _separate_tracks(self, vids, min_gap=6.0):
        """SUMO keeps same-lane vehicles apart: a lane-constrained member never overlaps / passes another
        member in the same approach lane. Front-to-back (x at el 0), each member is re-solved with
        x <= x_ahead - min_gap at every second both are planned in that lane on the approach (seconds
        where it must already be past the drop excluded); a pair whose constraints are infeasible is
        left to the lane-relocation repair (noted)."""
        drop = cm_lt(self.geo.drop)
        cons = [v for v in vids if any(lane is not None for lane in self.tracks[v].lane)]
        order = sorted(cons, key=lambda v: -self.x[v][0])
        for i, b in enumerate(order):
            T = self.tracks[b]
            saved = (T.lo[:], T.hi[:])
            past = list(T.lo)                       # forward running max: must already be past the drop
            for el in range(1, len(past)):
                past[el] = max(past[el], past[el-1])
            changed = False
            for a in order[:i]:
                A = self.tracks[a]
                for el in range(min(len(self.x[a]), len(self.x[b]))):
                    if not (A.live(el) and T.live(el)) or self.x[a][el] > drop:
                        continue
                    if self.lanes[a][el] != self.lanes[b][el] or past[el] > drop:
                        continue
                    lim = self.x[a][el]-round(min_gap*100)
                    if T.hi[el] > lim:
                        T.hi[el] = lim
                        changed = changed or self.x[b][el] > lim
            if not changed:
                continue
            try:
                self._solve_track(b)
            except StreamError:
                T.lo, T.hi = saved
                self._solve_track(b)
                self.notes.append(f'{b}: same-lane separation infeasible (lane relocation repair only)')

    def _zip_band(self, m):
        """Position band of a zip_yield member around its partner in which some plausible own speed
        reproduces the logged tracking accel (a = kp*((xp - gap) - x) + kv*(vp - v), clipped)."""
        vid, p = m['id'], m['params']
        T, P = self.tracks[vid], p['partner']
        gap, kp, kv = float(p['gap']), float(p['kp']), float(p['kv'])
        dmax, amax = float(p['dmax']), float(p['amax'])
        vlo, vhi, mg = 1.0, V_MAX-1.0, 0.1
        for el, a in sorted(self.expected_accel[vid].items()):
            xp, vp = self.x[P][el]/100.0, self.v[P][el]
            if a <= -dmax+TOL:
                T.ge(el, round(xp-gap-(-dmax-mg-kv*(vp-vhi))/kp+mg, 2), 'zip_yield saturated -dmax')
            elif a >= amax-TOL:
                T.le(el, round(xp-gap-(amax+mg-kv*(vp-vlo))/kp-mg, 2), 'zip_yield saturated amax')
            else:
                err_lo, err_hi = (a-kv*(vp-vlo))/kp, (a-kv*(vp-vhi))/kp
                T.ge(el, round(xp-gap-err_hi+mg, 2), 'zip_yield band')
                T.le(el, round(xp-gap-err_lo-mg, 2), 'zip_yield band')
        vs = self.v[P]
        T.v_t0 = vs[0]
        T.acc = [vs[el+1]-vs[el] if el+1 < len(vs) else 0.0 for el in range(T.n)]

    def _zip_speeds(self, m):
        vid, p = m['id'], m['params']
        P = p['partner']
        gap, kp, kv = float(p['gap']), float(p['kp']), float(p['kv'])
        dmax, amax = float(p['dmax']), float(p['amax'])
        for el, a in sorted(self.expected_accel[vid].items()):
            xp, vp = self.x[P][el]/100.0, self.v[P][el]
            x = self.x[vid][el]/100.0
            base = kp*((xp-gap)-x)+kv*vp              # a = base - kv*v
            vnom = self.v[vid][el]
            if a <= -dmax+TOL:
                v = max(vnom, (base+dmax)/kv+.01)
            elif a >= amax-TOL:
                v = min(vnom, (base-amax)/kv-.01)
            else:
                v = (base-a)/kv
            v = round(v, 4)
            if not 0.0 <= v <= V_MAX+1.0:
                self.notes.append(f'{vid}: zip_yield speed {v} at el {el} outside the plausible range')
                v = round(min(max(v, 0.0), V_MAX+1.0), 4)
            self.v[vid][el] = v

    def _repair_overlaps(self, min_gap=6.0):
        """No same-lane overlap involving a CAV (SUMO never produces one; the env sensor refuses it):
        move a vehicle whose lane is unconstrained at that second to a free lane."""
        geo = self.geo
        specs = {m['id']: m for m in self.specs}
        match_lanes = defaultdict(list)                 # el -> [(lane, x_cm)] of matching members
        for vid, seconds in self.match.items():
            for el in seconds:
                match_lanes[el].append((int(specs[vid]['params']['lc_target']), self.x[vid][el]))
        for el in range(self.n):
            for _ in range(12):
                rows = []
                for vid in self.members:
                    if self.tracks[vid].live(el):
                        st = geo.state(self.x[vid][el], self.lanes[vid][el], 0.0, True)
                        rows.append((vid, st['edge'], st['sumo_lane'], self.x[vid][el], True))
                for pid, e in self.extra.get(el, {}).items():
                    st = geo.state(*e)
                    rows.append((pid, st['edge'], st['sumo_lane'], e[0], False))
                clash = _first_overlap(rows, min_gap)
                if clash is None:
                    break
                movable = [r for r in clash if r[0] in self.tracks and self.tracks[r[0]].lane[el] is None]
                done = False
                for vid, edge, lane, x, _ in movable:
                    for up in range(geo.n_up):
                        st = geo.state(x, up, 0.0, True)
                        if st['edge'] == edge and st['sumo_lane'] == lane:
                            continue
                        if any(abs(x-o[3]) < min_gap*100 for o in rows if o[0] != vid and o[1] == st['edge']
                               and o[2] == st['sumo_lane']):
                            continue
                        if st['edge'] == geo.edge and any(up == ml and abs(x-mx) <= 10100
                                                          for ml, mx in match_lanes.get(el, ())):
                            continue
                        self.lanes[vid][el] = up
                        done = True
                        break
                    if done:
                        break
                if not done:
                    self.notes.append(f'el {el}: unresolved same-lane overlap {[r[0] for r in clash]}')
                    break

    def _phantoms(self):
        """One phantom HDV per matching (member, second) on the target lane: the probe's _lane_speed /
        our sensed target-lane mean then yields the logged accel."""
        import numpy as np
        by_el = defaultdict(dict)
        for vid, seconds in self.match.items():
            for el, a in seconds.items():
                by_el[el][vid] = a
        specs = {m['id']: m for m in self.specs}
        for el, ms in sorted(by_el.items()):
            vids = sorted(ms)
            tl = {v: int(specs[v]['params']['lc_target']) for v in vids}
            want = {}
            for v in vids:
                p = specs[v]['params']
                amax, dmax = float(p.get('amax', 1.0)), float(p['dmax'])
                own = self.v[v][el]
                a = ms[v]
                if a <= -dmax+TOL:
                    if own < dmax+1.0:
                        own = self.v[v][el] = round(dmax+1.5, 4)
                    want[v] = round(own-dmax-0.5, 4)
                elif a >= amax-TOL:
                    if own+amax+0.5 > V_MAX:
                        own = self.v[v][el] = round(V_MAX-amax-1.0, 4)
                    want[v] = round(own+amax+0.5, 4)
                else:
                    want[v] = round(own+a, 4)
            pos = {v: self.x[v][el] for v in vids}
            fixed = {}
            for v in vids:
                xs = []
                for o in self.members:
                    if o == v or not self.tracks[o].live(el):
                        continue
                    xo = self.x[o][el]
                    if xo < cm_lt(self.geo.drop)+1 and self.lanes[o][el] == tl[v] and abs(xo-pos[v]) <= 10000:
                        xs.append(self.v[o][el])
                fixed[v] = xs
            ph = {v: pos[v] for v in vids}
            for _ in range(4):
                A = np.zeros((len(vids), len(vids)))
                b = np.zeros(len(vids))
                for r_, v in enumerate(vids):
                    inr = [c for c, k in enumerate(vids) if tl[k] == tl[v] and abs(ph[k]-pos[v]) <= 10000]
                    for c in inr:
                        A[r_, c] = 1.0
                    b[r_] = want[v]*(len(fixed[v])+len(inr))-sum(fixed[v])
                try:
                    u = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    u = None
                if u is not None and all(0.0 <= x <= 40.0 for x in u):
                    break
                top = cm_lt(self.geo.drop)
                ph = {v: (pos[v]+9500 if (k % 2 == 0 and pos[v]+9500 <= top) or pos[v]-9500 < 0 else pos[v]-9500)
                      for k, v in enumerate(vids)}
            else:
                raise StreamError(f'el {el}: no phantom speeds reproduce the logged match accels')
            for k, v in enumerate(vids):
                pid = f'~ph.{v}.{el}'
                self.extra[el][pid] = (int(ph[v]), tl[v], round(float(u[k]), 4), False)


def _taut_string(L, U, x0, end_slope):
    """Shortest non-decreasing path x[0..n-1] with x[0] = x0 inside L <= x <= U (funnel algorithm):
    piecewise-constant slope, bending only at binding bounds; the free end takes the slope closest to
    end_slope. Integer cm in, integer cm out."""
    n = len(L)
    xs = [0.0]*n
    xs[0] = float(x0)
    i, xi = 0, float(x0)
    while i < n-1:
        lo, hi, jlo, jhi = 0.0, math.inf, None, None
        bend = None
        for j in range(i+1, n):
            s_lo, s_hi = (L[j]-xi)/(j-i), (U[j]-xi)/(j-i)
            if s_lo > hi:
                bend = (jhi, hi)
                break
            if s_hi < lo:
                bend = (jlo, lo) if jlo is not None else (j, max(s_hi, 0.0))
                break
            if s_lo > lo:
                lo, jlo = s_lo, j
            if s_hi < hi:
                hi, jhi = s_hi, j
        if bend is None:
            slope = min(max(end_slope, lo), hi)
            for k in range(i+1, n):
                xs[k] = xi+slope*(k-i)
            break
        j, slope = bend
        for k in range(i+1, j+1):
            xs[k] = xi+slope*(k-i)
        i, xi = j, xs[j]
    out = []
    for k in range(n):
        x = min(max(int(round(xs[k])), L[k]), U[k])
        if out and x < out[-1]:
            x = out[-1]
        out.append(x)
    return out


def _first_overlap(rows, min_gap):
    """First same-(edge, lane) pair closer than min_gap m involving a CAV; rows (vid, edge, lane, x_cm, cav)."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r[1], r[2])].append(r)
    for key in sorted(groups):
        g = sorted(groups[key], key=lambda r: r[3])
        for a, b in zip(g, g[1:]):
            if (a[4] or b[4]) and b[3]-a[3] < min_gap*100:
                return (a, b)
    return None


# ---------------------------------------------------------------- random streams
def _thresholds(spec):
    p = spec['params']
    out = []
    for k in ('release_x', 'anchor_release_x', 'trigger_x', 'merge_x', 'fallback_x', 'giveup_x', 'lc_giveup_x',
              'lc_from_x'):
        if p.get(k) is not None:
            out.append(float(p[k]))
    if p.get('zone'):
        out += [float(z) for z in p['zone']]
    return out


class RandomStream:
    """Seeded plausible random stream: lanes change by one, speeds 0-30, positions move forward."""

    def __init__(self, row, program, frame, desc, source, seed, n_extra=8, min_gap=8.0):
        self.geo = geo = Geo(desc, frame)
        rng = _random.Random(f'lcexec-cv-random:{row_key(row)}:{seed}')
        specs = member_specs(row)
        self.members = [m['id'] for m in specs]
        refs = []
        for m in specs:
            p = m['params']
            refs += [p[k] for k in ('anchor', 'partner', 'release_on_partner_merge') if p.get(k)]
            refs += list(p.get('track_ref') or ())
        refs = [r for r in dict.fromkeys(refs) if r not in self.members]
        n = max(min(int(row['duration_s'])+5, 200), 60)
        self.n = n
        ths = [t for m in specs for t in _thresholds(m) if t < geo.end]
        lo_x = max(0.0, (min(ths) if ths else geo.drop-1500.0)-400.0)
        vids = [(v, True) for v in self.members]+[(v, False) for v in refs]
        vids += [(f'~hdv{k}', False) for k in range(n_extra)]
        lanes0 = {m['id']: int(m['lane']) for m in specs}
        self.settings = {v: dict(lc_mode=rng.choice((1621, 1557, 1365)), tau=rng.choice((1.0, 1.2, 0.8, 1.5)))
                         for v, cav in vids if cav}
        veh, member_x = {}, []
        for v, cav in vids:
            if v.startswith('~hdv') and member_x:
                x = rng.choice(member_x)+rng.uniform(-150.0, 150.0)
            else:
                x = rng.uniform(lo_x, geo.drop-60.0)
            x = min(max(x, 1.0), geo.drop-30.0)
            if cav:
                member_x.append(x)
            veh[v] = dict(x=x, lane=lanes0.get(v, rng.randrange(geo.n_up)), v=rng.uniform(3.0, V_MAX), cav=cav)
        end = geo.end-15.0
        self._push(veh, min_gap)
        self.states = []
        for el in range(n):
            st = {v: geo.state(round(d['x']*100), d['lane'], round(d['v'], 4), d['cav'])
                  for v, d in veh.items() if d['x'] < end}
            self._separate(st)
            self.states.append(st)
            for v, d in veh.items():
                d['v'] = min(max(d['v']+rng.choice((-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5))*rng.random(), 0.0), V_MAX)
                if rng.random() < 0.04:
                    d['v'] = rng.uniform(0.0, 4.0)
                d['x'] += d['v']
            for v, d in veh.items():
                if d['x'] < geo.drop-1.0 and rng.random() < 0.12:
                    lane = min(max(d['lane']+rng.choice((-1, 1)), 0), geo.n_up-1)
                    if all(o['lane'] != lane or abs(o['x']-d['x']) >= min_gap for w, o in veh.items()
                           if w != v and o['x'] < geo.drop):
                        d['lane'] = lane
            self._push(veh, min_gap)
        self.roster = {}
        for st in self.states:
            self.roster.update({v: s['is_cav'] for v, s in st.items()})
        self.notes = []

    def _push(self, veh, min_gap):
        """No same-(edge, lane) pair closer than min_gap m: push the leader forward (positions never
        move backwards; SUMO never produces overlapping vehicles and the env sensor refuses them)."""
        geo = self.geo
        for _ in range(8):
            groups = defaultdict(list)
            for v, d in veh.items():
                st = geo.state(round(d['x']*100), d['lane'], 0.0, d['cav'])
                groups[(st['edge'], st['sumo_lane'])].append(d)
            moved = False
            for g in groups.values():
                g.sort(key=lambda d: d['x'])
                for a, b in zip(g, g[1:]):
                    if b['x']-a['x'] < min_gap:
                        b['x'] = a['x']+min_gap
                        moved = True
            if not moved:
                return

    @staticmethod
    def _separate(st):
        """Keep every pairwise |dx| away from the 100 m sensing boundary (float frames differ by an ulp)."""
        items = sorted(st.items(), key=lambda kv: kv[1]['x'])
        for _ in range(3):
            moved = False
            for i, (a, sa) in enumerate(items):
                for b, sb in items[i+1:]:
                    if abs(abs(sb['x']-sa['x'])-lp.SENSING_M) < 0.015:
                        sb['x'] = round(sb['x']+0.03, 2)
                        moved = True
            if not moved:
                break


# ---------------------------------------------------------------- comparison
def _effective(calls, *, other=None):
    """Apply N1 (and N2 on our side when ``other`` = the probe calls)."""
    slowed = {(el, vid) for el, vid, kind, _ in calls if kind == 'slowDown'}
    out = []
    pk = None if other is None else {(el, vid) for el, vid, kind, _ in other if kind in ('slowDown', 'setSpeed')}
    for c in calls:
        el, vid, kind, value = c
        if kind == 'setSpeed' and float(value) == -1.0:
            if (el-1, vid) not in slowed:
                continue                                            # N1
            if pk is not None and (el, vid) not in slowed and (el, vid) not in pk:
                continue                                            # N2 (ours only)
        out.append(c)
    return out


def _same(kind, a, b):
    if kind == 'changeLane':
        return int(a[0]) == int(b[0]) and abs(float(a[1])-float(b[1])) <= TOL
    if kind in ('setLaneChangeMode', 'setSpeedMode'):
        return int(a) == int(b)
    return abs(float(a)-float(b)) <= TOL


def compare(probe_calls, our_calls, receipts=None):
    """Per-(el, vid) multiset comparison after N1/N2 -> list of mismatch dicts (classified)."""
    a = [c for c in _effective(probe_calls) if c[2] in COMPARED]
    b = [c for c in _effective(our_calls, other=probe_calls) if c[2] in COMPARED]
    ga, gb = defaultdict(list), defaultdict(list)
    for el, vid, kind, value in a:
        ga[(el, vid)].append((kind, value))
    for el, vid, kind, value in b:
        gb[(el, vid)].append((kind, value))
    out = []
    for key in sorted(set(ga) | set(gb), key=lambda k: (k[0], k[1])):
        pa, pb = list(ga.get(key, ())), list(gb.get(key, ()))
        for kind in sorted({k for k, _ in pa} | {k for k, _ in pb}):
            xa = [v for k, v in pa if k == kind]
            xb = [v for k, v in pb if k == kind]
            unmatched_b = list(xb)
            for va in xa:
                hit = next((i for i, vb in enumerate(unmatched_b) if _same(kind, va, vb)), None)
                if hit is None:
                    out.append(_classify(key, kind, va, None, receipts))
                else:
                    unmatched_b.pop(hit)
            for vb in unmatched_b:
                out.append(_classify(key, kind, None, vb, receipts))
    return out


def _classify(key, kind, probe_value, our_value, receipts):
    el, vid = key
    rec = dict(el=el, vid=vid, kind=kind, probe=_jsonable(probe_value), ours=_jsonable(our_value), cls='mismatch')
    status = ((receipts or {}).get(el) or {}).get(vid, {}).get('status')
    if kind == 'changeLane' and probe_value is not None and our_value is None and status:
        if status == 'deferred_edge_end':
            rec.update(cls='writer_guard', reason='env edge-end deferral (v*duration + 10 m of the lane end)')
        elif status.startswith('refused'):
            rec.update(cls='writer_refused', reason=f'env receipt {status}')
    return rec


def _jsonable(v):
    if isinstance(v, tuple):
        return list(v)
    return v


# ---------------------------------------------------------------- log check
def _log_check(row, stream, runner, source):
    members = stream.members
    diffs = []
    arm = runner
    rel = {m['id']: m.get('release_el') for m in arm.members}
    for vid in members:
        want = (row.get('release_el') or {}).get(vid)
        if rel.get(vid) != want:
            diffs.append(dict(vid=vid, what='release_el', probe=rel.get(vid), row=want))
    want_ev = sorted(tuple(e) for e in row.get('arm_events') or () if e[1] in members)
    got_ev = sorted(tuple(e) for e in arm.events if e[1] in members)
    if want_ev != got_ev:
        diffs.append(dict(what='arm_events', probe=[list(e) for e in got_ev if e not in want_ev][:6],
                          row=[list(e) for e in want_ev if e not in got_ev][:6]))
    E = stream.E
    worst = 0.0
    if source == 'coopkinds':
        want = sorted((int(el), vid, k, v) for el, vid, k, v in stream.cmds if el < E)
        got = sorted((int(e[0]), vid, e[1], e[2]) for vid in members for e in arm.issued.get(vid, ()) if e[0] < E)
        diffs += _cmp_logs(want, got)
        worst = _worst(want, got)
    else:
        if source in ('plangen2', 'lcmoves'):
            want = sorted((el, vid, k, v) for el, vid, k, v in stream.cmds if el < E)
            got = sorted((int(el), vid, k, v) for el, vid, k, v in arm.commands if vid in members and el < E)
            diffs += _cmp_logs(want, got)
        want = sorted((el, vid, 'a', a) for vid, acc in stream.expected_accel.items() for el, a in acc.items())
        got = sorted((int(e[0]), vid, 'a', e[1]) for vid in members for e in arm.issued.get(vid, ()) if e[0] < E)
        diffs += _cmp_logs(want, got)
        worst = _worst(want, got)
        if source == 'lcmoves':
            done = {m['id']: m.get('lc_done') for m in arm.members}
            for vid in members:
                if done.get(vid) != (row.get('lc_done') or {}).get(vid):
                    diffs.append(dict(vid=vid, what='lc_done', probe=done.get(vid), row=row['lc_done'].get(vid)))
    return dict(ok=not diffs, diffs=diffs[:20], n_diffs=len(diffs), accel_max_err=worst,
                n_accel=sum(len(a) for a in stream.expected_accel.values()), notes=list(stream.notes))


def _cmp_logs(want, got):
    out = []
    wa = defaultdict(list)
    for el, vid, k, v in want:
        wa[(el, vid, k)].append(v)
    ga = defaultdict(list)
    for el, vid, k, v in got:
        ga[(el, vid, k)].append(v)
    for key in sorted(set(wa) | set(ga), key=str):
        x, y = wa.get(key, []), ga.get(key, [])
        ok = len(x) == len(y) and all(
            (abs(float(p)-float(q)) <= A_TOL if key[2] == 'a' else p == q) for p, q in zip(sorted(x), sorted(y)))
        if not ok:
            out.append(dict(el=key[0], vid=key[1], what=key[2], row=x, probe=y))
    return out


def _worst(want, got):
    g = {(el, vid): v for el, vid, k, v in got if k == 'a'}
    errs = [abs(float(v)-float(g[(el, vid)])) for el, vid, k, v in want if k == 'a' and (el, vid) in g]
    return max(errs, default=0.0)


# ---------------------------------------------------------------- run
@functools.lru_cache(maxsize=None)
def _wz_desc():
    from . import map_descriptor as md
    from .lc_writer_harness import MAP_DIR
    return md.load(MAP_DIR)


def build_stream(row, program, *, frame, stream='log', seed=0, desc=None, source=None):
    desc = desc if desc is not None else _wz_desc()
    source = source or source_of(row)
    if stream == 'log':
        return LogStream(row, program, frame, desc, source)
    if stream == 'random':
        return RandomStream(row, program, frame, desc, source, seed)
    raise ValueError(f'unknown stream {stream!r}')


def probe_members(row):
    """Probe Arm member dicts (members only: dithers excluded; nm2 rows without a role = 'hold')."""
    return [dict(id=m['id'], role=m['role'], params=dict(m['params'])) for m in member_specs(row)]


def run(row, program, *, frame, stream='log', seed=0, desc=None, file=None, collect_labels=False, built=None):
    """Probe Arm vs our executors + real env writer on one stream (see the module docstring)."""
    from .lc_writer_harness import FakeTraci, Harness, requests_from_commands
    desc = desc if desc is not None else _wz_desc()
    source = source_of(row, file)
    st = built if built is not None else build_stream(row, program, frame=frame, stream=stream, seed=seed,
                                                       desc=desc, source=source)
    settings = getattr(st, 'settings', None) or {}
    h = Harness(frame, st.roster, desc=desc, settings=settings)
    api = FakeTraci(st.roster, lane_lengths=h.api.lane_lengths, settings=settings)
    P = probe(source)
    members = probe_members(row)
    cav_ids = frozenset(v for v, c in st.roster.items() if c)
    if P.families is not None:
        runner = P.families.Executor(api, members, cav_ids)
        arm = runner.arm

        def probe_step(el, live):
            runner.step(el)
    else:
        arm = P.Arm(api, members)
        refs = []
        for m in members:
            q = m['params']
            refs += [q[k] for k in ('anchor', 'partner', 'release_on_partner_merge') if q.get(k)]
            refs += list(q.get('track_ref') or ())
        track = list(dict.fromkeys([m['id'] for m in members]+refs))

        def probe_step(el, live):
            arm.control(el, live, P.positions(api, track, live))
    execs = {vid: lp.MemberExecutor(vid, program.program(vid)) for vid in program.members}
    probe_calls, our_calls, receipts, labels = [], [], {}, []
    t0 = int(row.get('t0') or 0)
    session = row.get('verification_id')
    for el, states in enumerate(st.states):
        hs = h.states(states)
        api.load(el, hs)
        n = len(api.calls)
        probe_step(el, set(states))
        probe_calls += api.calls[n:]
        before = {v: ex.released for v, ex in execs.items()}
        cmds = {v: ex.step(h.obs(el, v, states)) for v, ex in execs.items()}
        our_calls += h.tick(el, states, requests_from_commands(cmds, execs))
        receipts[el] = dict((h.last_info or {}).get('lc_lane_vehicles') or {})
        if collect_labels:
            for v in program.members:
                if not before[v]:
                    labels.append(lp.tick_label(vid=v, t=t0+el, el=el, program=program.program(v),
                                                executor=execs[v], commands=cmds.get(v) or [],
                                                session_id=session,
                                                source=None if file is None else Path(str(file)).name))
    mism = compare(probe_calls, our_calls, receipts)
    cls = Counter(m['cls'] for m in mism)
    out = dict(stream=stream, seed=seed, key=row_key(row), passed=not mism, mismatches=mism,
               classes=dict(cls), writer_only=bool(mism) and set(cls) <= {'writer_guard', 'writer_refused'},
               probe_calls=probe_calls, our_calls=our_calls, n_ticks=len(st.states),
               log_check=_log_check(row, st, arm, source) if stream == 'log' else None,
               receipts_count=dict(Counter(r['status'] for rr in receipts.values() for r in rr.values())))
    out['pass'] = out['passed']
    if collect_labels:
        out['labels'] = labels
    return out


# ---------------------------------------------------------------- report
def check_row(name, idx, row, *, n_random=N_RANDOM, desc=None):
    """Convert + log parity + n_random random parity of one fixture row -> compact record."""
    desc = desc if desc is not None else _wz_desc()
    frame = lp.frame_from_descriptor(desc)
    meta = row_meta(row, f'{name}.jsonl')
    rec = dict(file=name, index=idx, key=meta['key'], verification_id=meta['verification_id'],
               family=meta['family'], variant=meta['variant'], r_kind=meta['r_kind'], roles=meta['roles'],
               dither_excluded=meta['dither_excluded'])
    try:
        program = from_mined_row(row, desc, file=f'{name}.jsonl')
    except NotRepresentable as exc:
        rec.update(converted=False, refusal=exc.reason)
        return rec
    rec.update(converted=True, registered_violations=lp.check_registered(program))
    runs = []
    try:
        runs.append(run(row, program, frame=frame, stream='log', desc=desc, file=f'{name}.jsonl'))
    except StreamError as exc:
        runs.append(dict(stream='log', seed=0, passed=False, pass_=False, error=f'log stream: {exc}', mismatches=[],
                         classes={'stream_error': 1}, writer_only=False, log_check=dict(ok=False, error=str(exc))))
    for seed in range(n_random):
        runs.append(run(row, program, frame=frame, stream='random', seed=seed, desc=desc, file=f'{name}.jsonl'))
    log = runs[0]
    rec['log'] = dict(passed=log['passed'], writer_only=log['writer_only'], classes=log['classes'],
                      log_check=log['log_check'], error=log.get('error'),
                      mismatches=log['mismatches'][:12], n_mismatches=len(log['mismatches']),
                      n_ticks=log.get('n_ticks'), probe_calls=len(log.get('probe_calls') or ()),
                      our_calls=len(log.get('our_calls') or ()), receipts=log.get('receipts_count') or {},
                      kinds=dict(Counter(f"{m['cls']}:{m['kind']}" for m in log['mismatches'])),
                      probe_kinds=dict(Counter(c[2] for c in log.get('probe_calls') or ())),
                      notes=note_counts((log.get('log_check') or {}).get('notes') or ()))
    rec['random'] = [dict(seed=r['seed'], passed=r['passed'], writer_only=r['writer_only'], classes=r['classes'],
                          mismatches=r['mismatches'][:6], n_mismatches=len(r['mismatches']), n_ticks=r['n_ticks'],
                          probe_calls=len(r['probe_calls']), our_calls=len(r['our_calls']),
                          receipts=r.get('receipts_count') or {},
                          kinds=dict(Counter(f"{m['cls']}:{m['kind']}" for m in r['mismatches'])),
                          probe_kinds=dict(Counter(c[2] for c in r['probe_calls'])))
                     for r in runs[1:]]
    return rec


def note_counts(notes):
    """Log-stream reconstruction notes by category (plausibility receipts, not parity failures)."""
    out = Counter()
    for n in notes:
        if 'unresolved same-lane overlap' in n:
            out['overlap_seconds'] += 1
        elif 'separation infeasible' in n:
            out['separation_infeasible_tracks'] += 1
        elif 'does not fit the tube' in n:
            out['nominal_path_misfit_tracks'] += 1
        else:
            out['other'] += 1
    return dict(out)


def report(n_random=N_RANDOM, rows=None):
    rows = all_rows() if rows is None else rows
    per_file, records = {}, []
    for name, idx, row in rows:
        rec = check_row(name, idx, row, n_random=n_random)
        records.append(rec)
        f = per_file.setdefault(name, dict(rows=0, converted=0, parity_log_pass=0, parity_log_writer_only=0,
                                           log_check_pass=0, parity_random_pass=0,
                                           parity_random_writer_only=0, random_streams=0,
                                           random_streams_pass=0, refusals=[], registered_violations=0,
                                           mismatch_classes=Counter(), log_mismatch_kinds=Counter(),
                                           random_mismatch_kinds=Counter(),
                                           log_probe_call_kinds=Counter(), random_probe_call_kinds=Counter(),
                                           log_receipts=Counter(), random_receipts=Counter(),
                                           log_stream_notes=Counter(), dither_rows=0, dither_vehicles=0))
        f['rows'] += 1
        f['dither_rows'] += bool(rec['dither_excluded'])
        f['dither_vehicles'] += len(rec['dither_excluded'])
        if not rec['converted']:
            f['refusals'].append(dict(verification_id=rec['verification_id'], key=rec['key'],
                                      reason=rec['refusal']))
            continue
        f['converted'] += 1
        f['registered_violations'] += bool(rec['registered_violations'])
        log = rec['log']
        f['parity_log_pass'] += bool(log['passed'])
        f['parity_log_writer_only'] += bool(log['writer_only'])
        f['log_check_pass'] += bool((log['log_check'] or {}).get('ok'))
        rnd = rec['random']
        f['random_streams'] += len(rnd)
        f['random_streams_pass'] += sum(r['passed'] for r in rnd)
        f['parity_random_pass'] += all(r['passed'] for r in rnd)
        f['parity_random_writer_only'] += all(r['passed'] or r['writer_only'] for r in rnd) and not all(
            r['passed'] for r in rnd)
        for r in [log]+rnd:
            f['mismatch_classes'].update(r['classes'])
        f['log_mismatch_kinds'].update(log.get('kinds') or {})
        for r in rnd:
            f['random_mismatch_kinds'].update(r['kinds'])
        f['log_probe_call_kinds'].update(log.get('probe_kinds') or {})
        f['log_receipts'].update(log.get('receipts') or {})
        f['log_stream_notes'].update(log.get('notes') or {})
        for r in rnd:
            f['random_probe_call_kinds'].update(r['probe_kinds'])
            f['random_receipts'].update(r['receipts'])
    for f in per_file.values():
        for k, v in list(f.items()):
            if isinstance(v, Counter):
                f[k] = dict(v)
    writer = [dict(key=r['key'], stream='log', el=m['el'], vid=m['vid'], cls=m['cls'], reason=m.get('reason'))
              for r in records if r.get('converted') for m in r['log']['mismatches'] if m['cls'] != 'mismatch']
    other = [dict(key=r['key'], stream=s, mismatch=m)
             for r in records if r.get('converted')
             for s, ms in [('log', r['log']['mismatches'])]+[(f"random{x['seed']}", x['mismatches'])
                                                               for x in r['random']]
             for m in ms if m['cls'] == 'mismatch']
    totals = Counter()
    for f in per_file.values():
        for k in ('rows', 'converted', 'parity_log_pass', 'parity_log_writer_only', 'log_check_pass',
                  'parity_random_pass', 'parity_random_writer_only', 'random_streams', 'random_streams_pass',
                  'dither_rows', 'dither_vehicles'):
            totals[k] += f[k]
    rnd_cls = Counter()
    for f in per_file.values():
        rnd_cls.update(f['random_mismatch_kinds'])
    lcm = per_file.get('LCMOVES_MOVES_B', {})
    known = dict(
        persistent_lane_hold=dict(
            note=('LCMOVES prog re-issues changeLane(target, 2 s) every second while armed, also in the '
                  'target lane; lc_runtime (c4/lcexec-rt 40669eb6, merged) issues a current-lane request as '
                  "changeLane to the current SUMO index (receipt 'held')"),
            lcmoves_log_held_seconds=(lcm.get('log_receipts') or {}).get('held', 0),
            all_log_held_seconds=sum((f['log_receipts'] or {}).get('held', 0) for f in per_file.values())),
        writer_guard=dict(note='env edge-end deferral (v*duration + 10 m of the current lane end) drops a probe '
                               'changeLane near the drop (zip_merge / merge_early out of the closing lane)',
                          log_seconds=[w for w in writer if w['cls'] == 'writer_guard'],
                          random_seconds=sum(v for k, v in rnd_cls.items() if k.startswith('writer_guard'))),
        writer_refused=dict(note='env adjacent-only rule refuses a non-adjacent LCEXEC lane request (probe: '
                                 'changeLane(target) two lanes away, SUMO changes stepwise); random streams only',
                            log_seconds=[w for w in writer if w['cls'] == 'writer_refused'],
                            random_seconds=sum(v for k, v in rnd_cls.items() if k.startswith('writer_refused'))),
        dithers=dict(note='non-member pulse vehicles excluded from the program and from both parity sides',
                     rows=totals['dither_rows'], vehicles=totals['dither_vehicles']))
    return dict(schema='lcexec-parity-report-v1', n_random=n_random,
                normalisations=['N1 setSpeed(-1) without a live speed command dropped on both sides',
                                "N2 our setSpeed(-1) at the first non-commanded second == the probe's slowDown expiry"],
                fixtures=dict(manifest_schema=manifest()['schema'],
                              rows={k: v['sha256'] for k, v in manifest()['rows'].items()},
                              probes={k: v['sha256'] for k, v in manifest()['probes'].items()}),
                totals=dict(totals), known_items=known, per_file=per_file, writer_log_mismatches=writer[:200],
                n_writer_log_mismatches=len(writer), other_mismatches=other[:100],
                n_other_mismatches=len(other), rows=[_slim(r) for r in records])


def _slim(rec):
    out = {k: v for k, v in rec.items() if k not in ('log', 'random')}
    if 'log' in rec:
        lg = rec['log']
        out['log'] = dict(passed=lg['passed'], writer_only=lg['writer_only'], classes=lg['classes'],
                          log_check_ok=(lg['log_check'] or {}).get('ok'),
                          log_check_diffs=(lg['log_check'] or {}).get('diffs', [])[:4],
                          log_check_notes=(lg['log_check'] or {}).get('notes', [])[:4],
                          accel_max_err=(lg['log_check'] or {}).get('accel_max_err'), error=lg.get('error'),
                          n_ticks=lg['n_ticks'], probe_calls=lg['probe_calls'], our_calls=lg['our_calls'],
                          receipts=lg.get('receipts'), notes=lg.get('notes'))
        out['random'] = [dict(seed=r['seed'], passed=r['passed'], writer_only=r['writer_only'], classes=r['classes'])
                         for r in rec['random']]
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='python -m tools.coopmwm.lc_parity')
    ap.add_argument('cmd', choices=('report',))
    ap.add_argument('out')
    ap.add_argument('--random', type=int, default=N_RANDOM)
    a = ap.parse_args(argv)
    rep = report(n_random=a.random)
    Path(a.out).write_text(json.dumps(rep, indent=1, sort_keys=True, default=str)+'\n', encoding='utf-8')
    print(json.dumps(rep['totals'], sort_keys=True))
    return 0


__all__ = ['run', 'compare', 'build_stream', 'LogStream', 'RandomStream', 'StreamError', 'probe', 'all_rows',
           'fixture_rows', 'check_row', 'report', 'probe_members', 'manifest', 'dither_ids', 'move_of',
           'note_counts']

if __name__ == '__main__':
    raise SystemExit(main())
