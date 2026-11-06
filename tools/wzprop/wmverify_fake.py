"""Zero-physics fake plant for S-WMVERIFY CPU tests and the dry run (NO SUMO; not a traffic model).

A deterministic toy stream on the wz3to2 frame (WM lanes 0/1/2 = SUMO up lanes 2/1/0, lane 2 closes at s = 75.44)
that exercises the harness hook protocol exactly as accel_plant/coopfree do: per whole second
``controller.apply(rows, commands, now)`` -> ``controller.lane_request(env)`` (G5 tensor stand-in + LCEXEC
``env.lc_tick`` stash) -> ``env.step`` -> ``controller.observe_lane(info)`` -> ``tick`` (totals series).
Numbers produced here are plumbing checks only, never evidence.
"""
from __future__ import annotations

import math
import random
from collections import Counter

DROP_S = 75.44
START_S = -2620.56
END_S = 600.0
V_FREE = 30.0
CLOSING = 2


class FakeController:
    """Base controller stand-in (v2v Controller protocol: apply, g5_lane_targets, lane_request/actors/observe)."""

    def __init__(self):
        self.g5_lane_targets = {}
        self.events = []
        self.lane_info = []

    def apply(self, rows, commands, now):
        self.g5_lane_targets = {}           # MAPPO u0 is speed-only: no base lane targets
        return {}

    def lane_request(self, env):
        return dict(self.g5_lane_targets) or None

    def lane_actors(self):
        return frozenset(self.g5_lane_targets)

    def observe_lane(self, info):
        self.lane_info.append(dict(info.get('lane_commands') or {}))


class FakeEnv:
    """Toy lane-drop stream with the LCEXEC lc_tick knob (CAV-only validation like lc_runtime.validate)."""

    def __init__(self, seed, share=0.6, rate=0.9, lc_runtime=True):
        self.rng = random.Random(f'wmverify-fake:{seed}')
        self.share, self.rate = float(share), float(rate)
        self.now = 0
        self.veh = {}                       # vid -> dict(s, v, lane, cav)
        self.n = 0
        self.stash = None
        self.lc_runtime = bool(lc_runtime)
        self.lc_mode = {}
        self.totals = Counter()
        self.done = False
        self._version, self._gap_cache = 0, None
        self.pv_req, self.pv_origin, self.pv_road = {}, {}, {}      # fake per-vehicle record (PerVehicle fields)
        if not lc_runtime:
            self.lc_tick = None

    # ---- observation
    def rows(self):
        return [dict(vid=v, s_m=round(d['s'], 3), v_mps=round(d['v'], 4), lane=d['lane'], is_cav=d['cav'],
                     length_m=5.0) for v, d in sorted(self.veh.items())]

    def commands(self):
        out = {}
        for vid, d in self.veh.items():
            if d['cav']:
                out[vid] = self._base_accel(vid, d)
        return out

    def _gaps(self):
        if self._gap_cache is None or self._gap_cache[0] != self._version:
            gaps, by = {}, {}
            for vid, d in self.veh.items():
                by.setdefault(d['lane'], []).append((d['s'], vid))
            for lane, xs in by.items():
                xs.sort()
                for i, (s, vid) in enumerate(xs):
                    gaps[vid] = xs[i + 1][0] - s if i + 1 < len(xs) else math.inf
            self._gap_cache = (self._version, gaps)
        return self._gap_cache[1]

    def _base_accel(self, vid, d):
        gap = self._gaps().get(vid, math.inf)
        v_des = V_FREE if gap > 120 else max(0.0, gap / 4.0)
        if d['lane'] == CLOSING and d['s'] > -150.0:
            v_des = min(v_des, 8.0)
        return max(-3.0, min(1.0, 0.6 * (v_des - d['v'])))

    # ---- LCEXEC knob
    def lc_tick(self, request):
        if self.stash is not None:
            raise ValueError('one lc_tick per env.step')
        unknown = set(request) - {'accel', 'native', 'lanes', 'lane_mode', 'tau'}
        if unknown:
            raise ValueError(f'unknown lc request keys {sorted(unknown)}')
        for key in ('accel', 'native', 'lanes', 'lane_mode', 'tau'):
            for vid in (request.get(key) or ()):
                d = self.veh.get(vid)
                if d is None or not d['cav']:
                    raise ValueError(f'lc request vehicle {vid!r} is not a live CAV (HDVs are never commanded)')
        self.stash = request
        return request

    # ---- one control second
    def step(self, desired, commands, g5_targets):
        stash, self.stash = self.stash, None
        info = dict(lane_commands=Counter(), lc_lane_vehicles={}, lc_lane_commands=Counter())
        clash = set(g5_targets or {}) & set((stash or {}).get('lanes') or {})
        if clash:
            raise ValueError(f'lane target from both LCEXEC and the G5 tensor: {sorted(clash)}')
        base = {vid: self._base_accel(vid, d) for vid, d in self.veh.items()}
        for vid, d in list(self.veh.items()):
            a = commands.get(vid, base[vid]) if d['cav'] else base[vid]
            if stash and vid in (stash.get('accel') or {}):
                d['v'] = max(0.0, min(V_FREE + 1.0, d['v'] + max(-3.0, min(1.0, float(stash['accel'][vid])))))
                continue                                               # LCEXEC program accel replaces the command
            if stash and vid in (stash.get('native') or ()):
                a = base[vid]
            req = desired.get(vid)
            if req is not None:
                if not d['cav']:
                    raise ValueError('HDV commanded')
                a = max(-3.0, min(a, float(req['requested_raw'])))      # base safety downstream (stand-in)
            d['v'] = max(0.0, min(V_FREE + 1.0, d['v'] + a))
        for vid, target in (g5_targets or {}).items():
            d = self.veh[vid]
            if target == d['lane']:
                info['lane_commands']['noop'] += 1
            elif abs(target - d['lane']) != 1 or target == CLOSING:
                info['lane_commands']['refused_not_allowed'] += 1
            else:
                d['lane'] = target
                info['lane_commands']['issued'] += 1
        for vid, (target, _dur) in ((stash or {}).get('lanes') or {}).items():
            d = self.veh[vid]
            if target == d['lane']:
                status, done = 'held', True
            elif abs(target - d['lane']) != 1:
                status, done = 'refused_not_allowed', False
            elif target == CLOSING and DROP_S - d['s'] <= d['v'] * 2.0 + 10.0:
                status, done = 'deferred_edge_end', False
            else:
                d['lane'], status, done = int(target), 'issued', True
            info['lc_lane_vehicles'][vid] = dict(target=int(target), status=status, completed=done)
            info['lc_lane_commands'][status] += 1
            if status == 'issued':
                info['lc_lane_commands']['completed'] += 1
        held = set(((stash or {}).get('lanes') or {})) | set(((stash or {}).get('lane_mode') or {}))
        for vid, d in list(self.veh.items()):
            d['s'] += d['v']
            if d['lane'] == CLOSING and d['s'] > -150.0 and vid not in held:
                d['lane'] = 1                                          # strategic merge out of the closing lane
            if d['lane'] == CLOSING and d['s'] >= DROP_S:
                d['lane'], d['v'] = 1, min(d['v'], 5.0)
            if d['s'] > END_S:
                del self.veh[vid]
        for lane in (0, 1, 2):
            if self.rng.random() < self.rate / 3.0:
                vid = f'v{self.n}'
                self.n += 1
                self.veh[vid] = dict(s=START_S, v=25.0, lane=lane, cav=self.rng.random() < self.share)
                self.pv_req[vid] = (float(self.now), bool(self.veh[vid]['cav']))
                self.pv_origin[vid] = ('up', 2 - lane)
        road = sum(max(0.0, V_FREE - d['v']) / V_FREE for d in self.veh.values())
        if 300 <= self.now < 1200:
            for vid, d in self.veh.items():
                self.pv_road[vid] = self.pv_road.get(vid, 0.0) + max(0.0, V_FREE - d['v']) / V_FREE
        self.totals['road_delay_vs'] += road
        self.totals['team_delay_vs'] += road
        self.now += 1
        self._version += 1
        return info


class FakeBackend:
    """``b`` of coopfree.tick: env.now + totals (Recorder.tick_wrapper reads both)."""

    def __init__(self, env):
        self.env = env
        self.totals = env.totals


def fake_tick(b, controller=None, trace=None):
    return []


def run_episode(hook, *, seed=989950, share=0.6, seconds=1200, lc_runtime=True, tick=None, record=None,
                warmup_s=300, keep_rows=False):
    """Drive one fake episode through the hook protocol; returns dict(totals, series, trace)."""
    env = FakeEnv(seed, share=share, lc_runtime=lc_runtime)
    controller = FakeController()
    hook(controller, dict(id='fake'), None)
    b = FakeBackend(env)
    ticker = tick if tick is not None else hook.tick_wrapper(fake_tick)
    trace = []
    for t in range(int(seconds)):
        if t < warmup_s:                                   # native warmup 0-300 s: no controller, no series
            env.step({}, env.commands(), None)
            continue
        rows = env.rows()
        commands = env.commands()
        desired = controller.apply(rows, commands, t)
        targets = controller.lane_request(env)
        if targets is not None:
            controller.lane_actors()
        info = env.step(desired, commands, targets)
        controller.observe_lane(info)
        ticker(b, controller, None)
        if record is not None:
            trace.append(dict(t=t, desired={k: (round(v['requested_raw'], 9), round(v['acceleration_residual'], 9))
                                            for k, v in desired.items()},
                              g5=dict(sorted((targets or {}).items())),
                              lc=info['lc_lane_vehicles']))
            if keep_rows:
                trace[-1]['rows'] = rows
    pv = getattr(hook, 'pv', None)
    if pv is not None:                       # fill the eval-only per-vehicle record from the fake stream
        pv.requests, pv.ins, pv.road, pv.origin = dict(env.pv_req), {v: t for v, (t, _c) in env.pv_req.items()}, \
            dict(env.pv_road), dict(env.pv_origin)
        pv.arrived_a, pv.error = set(), None
    return dict(totals=dict(env.totals), trace=trace, actual_sim_s=int(seconds))


def fake_episode_factory(share=0.6):
    """accel_plant.episode stand-in for the end-to-end worker test (uses coopfree.backend.tick as patched)."""
    def episode(job, *, controller_hook=None, policy_loader=None):
        from tools.coopfree import backend as cfb
        world = job.get('world') or {}
        lc = bool(getattr(controller_hook, 'needs_lc_runtime', False))
        res = run_episode(controller_hook, seed=int(world.get('seed', 0)), share=float(world.get('penetration', share)),
                          seconds=int(job.get('stop', 1200)), lc_runtime=lc, tick=cfb.tick)
        return dict(totals=res['totals'], actual_sim_s=res['actual_sim_s'], steps=[])
    return episode
