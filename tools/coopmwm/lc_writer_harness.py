"""Reusable CPU harness for LCEXEC writer parity tests (no SUMO, no TraCI process).

:class:`Harness` drives the REAL env writer path of the LCEXEC env subclass
(``lc_runtime.lc_env_class``): ``CavSumoEnv.step`` -> ``_control_tick`` -> real
``CommandOwnership`` (speedMode 27) -> ``damping.direct_target`` -> ``slowDown`` / ``setSpeed``,
the real ``TableSensor`` frames, the real ``FrameBuilder``, the frozen ``_lane_control`` /
``_issue_lane_commands`` and the LCEXEC lane-mode / tau / lane extensions. Only TraCI
(:class:`FakeTraci`, which records every write) and the post-physics step are faked: SUMO does
not move vehicles; the post-physics table is the caller's ``after`` states (default: unchanged),
which feeds the completion telemetry exactly like a real physics step would.

``tick(el, states, requests)`` returns the TraCI writes ``[(el, vid, kind, value)]`` in issue
order; kinds ``slowDown`` (target speed), ``setSpeed`` (speed), ``changeLane`` ((sumo index,
duration)), ``setLaneChangeMode`` (mode), ``setTau`` (tau), ``setSpeedMode`` (mode). States are
``{vid: dict(edge, sumo_lane, x, v, allowed, is_cav[, pos, a, length])}`` with ``x`` the probe's
global x (lane position = x - segment start of the descriptor lane table; junction lanes: x - drop
x, the probe ``EDGE_OFFSET``); ``pos`` overrides the lane position (merge map).

:func:`requests_from_commands` turns ``lc_program`` executor outputs into the tick request dict.
The same :class:`FakeTraci` runs the probe ``Arm`` for the parity tests.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import importlib
import math
from pathlib import Path

import numpy as np

from . import lc_program as lp
from . import map_descriptor as md
from .lc_runtime import empty_request, lc_env_class

REPO = Path(__file__).resolve().parents[2]
MAP_DIR = REPO/'assets/research/newmap_candidates/wz3to2'
T0_S = 300.0
WRITES = ('slowDown', 'setSpeed', 'changeLane', 'setLaneChangeMode', 'setTau', 'setSpeedMode')
DEFAULT_LC_MODE = 1621               # SUMO default laneChangeMode
DEFAULT_TAU = 1.0


def default_descriptor():
    """The process map when it is a lane drop, else the registered wz3to2 map of this checkout."""
    desc = md.current()
    return desc if desc.kind == 'lane_drop' else md.load(MAP_DIR)


def lane_lengths_of(desc):
    """SUMO lane lengths of a lane-drop descriptor (straight lanes; junction lanes start at the drop)."""
    if desc.kind != 'lane_drop':
        return {}
    drop = float(desc.bottleneck['drop_x_m'])
    out = {}
    for (edge, i), (_, x0) in desc.lane_table.items():
        out[f'{edge}_{i}'] = drop - float(desc.lane_end_distance[f'{edge}_{i}']) - float(x0)
    for lane_id, dist in desc.lane_end_distance.items():
        if lane_id.startswith(':'):
            out[lane_id] = -float(dist)
    return out


class FakeTraci:
    """Recording TraCI stand-in (vehicle / lane / simulation domains used by the env and the probe Arm)."""

    def __init__(self, roster, *, lane_lengths=None, default_lane_length=None, settings=None):
        self.roster = dict(roster)                    # vid -> is_cav
        self.lane_lengths = dict(lane_lengths or {})
        self.default_lane_length = default_lane_length
        self.calls, self.raw = [], []
        self.el, self.time = None, T0_S
        self.states = {}
        self.lc_mode = {v: DEFAULT_LC_MODE for v in self.roster}
        self.tau = {v: DEFAULT_TAU for v in self.roster}
        self.speed_mode = {v: 27 if c else 31 for v, c in self.roster.items()}
        for v, s in (settings or {}).items():
            self.lc_mode[v] = int(s.get('lc_mode', self.lc_mode[v]))
            self.tau[v] = float(s.get('tau', self.tau[v]))
        api = self

        class _Vehicle:
            def getIDList(self):
                return list(api.states)

            def getSpeed(self, vid):
                return float(api.states[vid]['v'])

            def getAllowedSpeed(self, vid):
                return float(api.states[vid]['allowed'])

            def getRoadID(self, vid):
                return api.states[vid]['edge']

            def getLaneIndex(self, vid):
                return int(api.states[vid]['sumo_lane'])

            def getLanePosition(self, vid):
                return float(api.states[vid]['pos'])

            def getSpeedMode(self, vid):
                return api.speed_mode[vid]

            def setSpeedMode(self, vid, mode):
                api._write(vid, 'setSpeedMode', int(mode), (vid, mode))
                api.speed_mode[vid] = int(mode)

            def slowDown(self, vid, speed, duration):
                api._write(vid, 'slowDown', float(speed), (vid, speed, duration))

            def setSpeed(self, vid, speed):
                api._write(vid, 'setSpeed', float(speed), (vid, speed))

            def changeLane(self, vid, index, duration):
                api._write(vid, 'changeLane', (int(index), float(duration)), (vid, index, duration))

            def getLaneChangeMode(self, vid):
                return api.lc_mode[vid]

            def setLaneChangeMode(self, vid, mode):
                api._write(vid, 'setLaneChangeMode', int(mode), (vid, mode))
                api.lc_mode[vid] = int(mode)

            def getTau(self, vid):
                return api.tau[vid]

            def setTau(self, vid, tau):
                api._write(vid, 'setTau', float(tau), (vid, tau))
                api.tau[vid] = float(tau)

        class _Lane:
            def getLength(self, lane_id):
                if lane_id in api.lane_lengths:
                    return api.lane_lengths[lane_id]
                if api.default_lane_length is None:
                    raise KeyError(f'harness: no length for lane {lane_id!r}')
                return float(api.default_lane_length)

            def getLastStepVehicleIDs(self, lane_id):
                return [v for v, s in api.states.items() if f"{s['edge']}_{int(s['sumo_lane'])}" == lane_id]

        class _Simulation:
            def getTime(self):
                return api.time

            def getArrivedIDList(self):
                return []

        self.vehicle, self.lane, self.simulation = _Vehicle(), _Lane(), _Simulation()

    def _write(self, vid, kind, value, args):
        if vid not in self.states:
            raise KeyError(f'harness: TraCI write to a vehicle that is not live: {vid!r}')
        self.calls.append((self.el, vid, kind, value))
        self.raw.append((self.el, kind, args))

    def load(self, el, states, *, time_s=None):
        self.el = el
        self.time = T0_S + el if time_s is None else float(time_s)
        self.states = {v: dict(s) for v, s in states.items()}

    def simulationStep(self):
        self.time += 0.5

    def isLoaded(self):
        return True

    def close(self):
        pass


class _NoDamping:
    """CommandOwnership controller stand-in: the harness never hands a CAV to damping."""

    def __init__(self, config):
        self.config = config

    def decide(self, frame, *, now_s):
        raise RuntimeError('lc_writer_harness: damping fallback is not modelled (use fallback="native")')


@contextmanager
def bound_tables(desc, env_globals):
    """Temporarily bind ``desc``'s lane table / offsets into geometry and the env module namespace."""
    geometry = importlib.import_module('traffic_shepherd.c4.geometry')
    table, offsets = dict(desc.lane_table), tuple(desc.here_offsets)
    saved = (geometry.LANE_TABLE, geometry.HERE_OFFSETS, env_globals.get('LANE_TABLE'))
    if saved[0] == table and tuple(saved[1]) == offsets and saved[2] == table:
        yield
        return
    geometry.LANE_TABLE, geometry.HERE_OFFSETS, env_globals['LANE_TABLE'] = table, offsets, table
    try:
        yield
    finally:
        geometry.LANE_TABLE, geometry.HERE_OFFSETS, env_globals['LANE_TABLE'] = saved


class Harness:
    """Real env writer (LCEXEC subclass by default) on a fake TraCI api; see the module docstring."""

    def __init__(self, frame=None, vehicles=(), *, desc=None, env_module=None, env_class=None,
                 lane_lengths=None, default_lane_length=None, settings=None, k=None):
        self.desc = desc if desc is not None else default_descriptor()
        if frame is None and self.desc.kind == 'lane_drop':
            frame = lp.frame_from_descriptor(self.desc)
        if frame is not None and frame != lp.frame_from_descriptor(self.desc):
            raise ValueError('harness frame differs from the descriptor frame')
        self.frame = frame
        self.env_module = env_module if env_module is not None else importlib.import_module('traffic_shepherd.c4.env')
        base = self.env_module.CavSumoEnv
        cls = env_class if env_class is not None else lc_env_class(base, self.desc)
        self.env_globals = cls._issue_lane_commands.__globals__
        roster = dict(vehicles)
        if any(type(c) is not bool for c in roster.values()):
            raise ValueError('vehicles: {vid: is_cav (bool)}')
        lengths = lane_lengths_of(self.desc)
        lengths.update(lane_lengths or {})
        self.api = FakeTraci(roster, lane_lengths=lengths, default_lane_length=default_lane_length,
                             settings=settings)
        from traffic_shepherd.config import load_config
        from traffic_shepherd.ownership import CommandOwnership
        from traffic_shepherd.c4.sensing import TableSensor
        from traffic_shepherd.c4.state import FrameBuilder
        config = load_config()
        cav_ids = frozenset(v for v, c in roster.items() if c)
        env = object.__new__(cls)
        env.config = config
        env.api = self.api
        env.cav_ids = cav_ids
        env.k = int(k if k is not None else self.desc.k_slots)
        env.mode, env.warmup_s, env.end_s, env.done = 'direct', 0.0, 1e9, False
        env.fleet, env.pi_session, env._direct_pi, env._circles = None, None, False, None
        env.direct_accel_max_mps2 = float(self.env_module.DIRECT_ACCEL_MAX_MPS2)
        env.ownership = CommandOwnership(self.api, cav_ids, _NoDamping(config),
                                         cav_speed_mode=int(self.env_module.CAV_SPEED_MODE))
        env.sensor = TableSensor(self.api, config, dict(self.desc.lane_end_distance or {}))
        env.slow_sensor, env.sensor_check_every_s = None, 1
        env.builder = FrameBuilder(env.k, cav_ids)
        env.recorder = None
        env._previous_live, env._retired_cavs = set(), set()
        env._lc_original, env._lane_lengths, env._table = {}, {}, {}
        env.command_counts, env.fallback_counts = Counter(), Counter()
        env.lane_command_counts, env.totals, env.window_totals = Counter(), Counter(), Counter()
        env._last_actions, env._last_lane_counts = {}, {}
        env.last_entry_queue_steps, env.last_source_wait_steps = [], []
        env._after_physics = self._physics
        env.state = lambda: {}
        self.env = env
        self._after = None
        self._started = False
        self.last_info = None

    # ---- tables
    def position(self, s):
        if 'pos' in s:
            return float(s['pos'])
        edge = s['edge']
        if edge.startswith(':'):
            return float(s['x']) - float(self.desc.bottleneck['drop_x_m'])
        row = self.desc.lane_table.get((edge, int(s['sumo_lane'])))
        if row is None:
            raise ValueError(f'harness: lane {edge}_{s["sumo_lane"]} not in the {self.desc.name} lane table')
        return float(s['x']) - float(row[1])

    def states(self, states):
        out = {}
        for vid, s in states.items():
            if vid not in self.api.roster:
                raise ValueError(f'harness: {vid!r} is not in the roster')
            if bool(s.get('is_cav', self.api.roster[vid])) != self.api.roster[vid]:
                raise ValueError(f'harness: {vid!r} is_cav differs from the roster')
            out[vid] = dict(s, pos=self.position(s))
        return out

    def table(self, states):
        return {vid: dict(road_id=s['edge'], lane_id=f"{s['edge']}_{int(s['sumo_lane'])}",
                          lane_index=int(s['sumo_lane']), lane_position_m=float(s['pos']),
                          speed_mps=float(s['v']), acceleration_mps2=float(s.get('a', 0.0)),
                          allowed_speed_mps=float(s['allowed']), length_m=float(s.get('length', 5.0)),
                          leader=None, min_gap_m=2.5)
                for vid, s in sorted(states.items())}

    @contextmanager
    def bound(self):
        with bound_tables(self.desc, self.env_globals):
            yield

    def _physics(self, after, actions, command_time, commanded):
        env = self.env
        self.api.states = dict(self._after[0])
        env._table = self._after[1]
        env.frame = env.builder.build(after, env._table)
        return Counter(source_wait_vehicle_seconds=0.0)

    def _slots(self, values, fill):
        arr = np.full(self.env.k, fill, dtype=np.float64)
        for vid, x in (values or {}).items():
            slot = self.env.builder.pool.slot_of(vid)
            if slot is None:
                raise ValueError(f'harness: {vid!r} has no slot')
            arr[slot] = x
        return arr[None]

    # ---- one control second
    def tick(self, el, states, requests=None, *, base=None, lanes=None, lane_actors=None, after=None,
             fallback='native'):
        """Run ONE real env.step at control second ``el``; returns the TraCI writes in issue order."""
        env = self.env
        with self.bound():
            now = T0_S + int(el)
            cur = self.states(states)
            post = self.states(after) if after is not None else cur
            self.api.load(int(el), cur, time_s=now)
            env._table = self.table(cur)
            env.frame = env.builder.build(now, env._table)
            if not self._started:
                env._previous_live, self._started = set(cur), True
            self._after = (post, self.table(post))
            n = len(self.api.calls)
            kw = {}
            if lanes is not None:
                kw['lanes'] = self._slots(lanes, -1.0)
            if lane_actors is not None:
                kw['lane_actors'] = lane_actors
            if requests is not None:
                env.lc_tick(requests)
            _, info = env.step(self._slots(base, math.nan), fallback=fallback, **kw)
            self.last_info = info
            return list(self.api.calls[n:])

    # ---- member observation (lc_program.Obs) from harness states
    def view(self, vid, s):
        edge, sumo = s['edge'], int(s['sumo_lane'])
        f = self.frame
        if edge == f.edge:
            lane = f.wm(sumo)
        else:
            row = self.desc.lane_table.get((edge, sumo))
            lane = -1 if row is None else int(row[0])
        return lp.View(vid, f.s(s['x']), lane, float(s['v']), edge == f.edge, bool(s.get('is_cav', False)))

    def obs(self, el, vid, states):
        """Own row + own sensing (|ds| <= SENSING_M) + every other live vehicle as a reference."""
        views = {v: self.view(v, s) for v, s in states.items()}
        own = views.get(vid)
        if own is None:
            return lp.Obs(el, None)
        sensed = tuple(w for v, w in sorted(views.items()) if v != vid and abs(w.s_m - own.s_m) <= lp.SENSING_M)
        refs = {v: w for v, w in views.items() if v != vid}
        return lp.Obs(el, own, sensed, refs)


def requests_from_commands(commands_by_vid, executors):
    """Tick request dict from ``lc_program`` executor outputs of this second.

    Only ACTIVE (or commanding), non-released members contribute: ``accel`` for an ('accel', a)
    command; ``native`` for a member under its program without an accel command (its lon kind is
    not ``base``); ``lanes`` for a ('changeLane', (L, d)); ``lane_mode`` = program lc_mode and
    ``tau`` = program tau while the member is active. Released / left / not-yet-started members
    (and ``base``-kind longitudinal members) contribute nothing longitudinal; their absence makes
    the env restore the saved lane mode and tau and hand the CAV back to the base command."""
    out = empty_request()
    accel, native, lanes, modes, taus = {}, [], {}, {}, {}
    for vid, ex in sorted(executors.items()):
        commands = list(commands_by_vid.get(vid) or ())
        if ex.released:
            continue
        a, lane, _ = lp.commanded(commands)
        program = ex.p
        under = ex.active or a is not None or lane is not None
        if not under:
            continue
        if a is not None:
            accel[vid] = float(a)
        elif program.lon.kind != 'base':
            native.append(vid)
        if lane is not None:
            lanes[vid] = [int(lane[0]), float(lane[1])]
        if ex.active:
            modes[vid] = int(program.lc_mode)
            if program.tau is not None:
                taus[vid] = float(program.tau)
    out.update(accel=accel, native=tuple(native), lanes=lanes, lane_mode=modes, tau=taus)
    return out


def writes(calls, kinds=WRITES):
    return [c for c in calls if c[2] in kinds]


__all__ = ['Harness', 'FakeTraci', 'requests_from_commands', 'bound_tables', 'lane_lengths_of',
           'default_descriptor', 'writes', 'T0_S', 'WRITES']
