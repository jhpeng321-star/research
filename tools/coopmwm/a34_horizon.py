"""AMENDMENT 34 (evaluation-horizon robustness; MAIN 2026-11-10, registered before any amendment-33/34/35 outcome).

The sealed p20c episode of an arm (U0, LEARNED_TRIGGER, DUO_HOLD) is run IDENTICALLY up to t = 1,200 s (same plant,
same spec, same controllers, same vehicle log over [300, 1200)); then NO new arrivals and the simulation continues,
controllers unchanged, until every vehicle has left the network or t = 2,400 s (whichever first).

Mechanics (worker side, no frozen-runtime file is changed; every change takes effect at t >= 1,200 s only):
* the plant env's internal end time 1201 (reset horizon 901) is moved to 2401 (SUMO's own --end is already 2401;
  the 2,400 s scenario fixture holds demand to 1,800 s), so ``done`` never fires inside [300, 2400);
* at the start of control second t = 1,200 (ledger time 1,200, before the step) every requested trip with
  depart_s > 1,200 is withdrawn: removed from the RequestLedger (requests + due times; it never accrues source
  wait) and kept out of the network: a withdrawn trip pending insertion (``simulation.getPendingVehicles``) is
  removed before it departs; one that SUMO loads and inserts within the same 0.5 s physics step (route vehicles are
  loaded only at their departure step) is removed right after that step, before any reader/controller/ledger sees
  it (it never moves; queued trips requested earlier are tried for insertion first). Both are counted. Trips requested at <= 1,200 s that still wait at the entrance stay queued and
  enter when the plant lets them (their departure delay keeps accruing in J); if one never enters it is charged
  to the end (2,400 s) and listed separately;
* before each control second t >= 1,200: if every remaining requested trip has arrived and no vehicle is in the
  network, the episode ends at t (NetworkEmpty; nothing after t is simulated);
* the evaluation-only ground-truth vehicle log keeps [300, 1200) byte-for-byte in ``vehicles.jsonl.gz`` and writes
  [1200, end) to ``vehicles_ext.jsonl.gz`` (same row format, same vehicle index, headers for new vehicles).

J (record ``J_veh_s``) = the plant's team delay (speed deficit + source wait) over [300, end); ``a34.J_1200_veh_s``
= the same accumulator at t = 1,200 (must equal the sealed per-world J: plant identity).
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

ARRIVALS_END_S = 1200
END_S = 2400
WARMUP_S = 300
ORIGINAL_END_S = 1201.0                  # Backend reset: warmup 300 + horizon 901
ARMS = ('U0', 'LEARNED_TRIGGER', 'DUO_HOLD')
EXT_LOG = 'vehicles_ext.jsonl.gz'
RECEIPT_KEYS = ('arrivals_end_s', 'end_s', 't_end_s', 'end_reason', 'J_1200_veh_s', 'totals_1200', 'totals_end',
                'safety_1200', 'ttc_events_ext', 'future_trips', 'withdrawn', 'never_entered', 'on_road_at_end',
                'requested_by_1200', 'inserted', 'arrived', 'actual_sim_s', 'ext_log', 'removed_pending',
                'removed_at_insertion')


def a34_digest():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()   # host-independent


def horizon_field():
    """The spec field that routes an episode to this module (the sealed spec is otherwise unchanged)."""
    return dict(arrivals_end_s=ARRIVALS_END_S, end_s=END_S, a34_sha256=a34_digest())


class NetworkEmpty(Exception):
    """Raised before a control second t >= 1,200 when the network is empty: the episode ends at t."""


class Horizon:
    """Per-episode extension state (one episode per process at a time)."""

    def __init__(self, *, arrivals_end=ARRIVALS_END_S, end=END_S):
        self.arrivals_end, self.end = float(arrivals_end), float(end)
        self.b = None
        self.totals_1200 = None
        self.future = None                 # vid -> requested depart_s (> arrivals_end)
        self.withdrawn = set()
        self.removed_pending = self.removed_at_insertion = 0
        self.final = None
        self.log = None

    # -- called before every plant control second (cf.tick) -------------------------------------------------
    def before_tick(self, b):
        env = b.env
        if self.b is None:
            if abs(float(env.end_s)-ORIGINAL_END_S) > 1e-9:
                raise ValueError(f'A34: plant env end_s {env.end_s} is not the registered {ORIGINAL_END_S}')
            env.end_s = self.end+1.0       # == SUMO --end; done never fires inside [300, end)
            self.b = b
        now = float(env.now)
        if now < self.arrivals_end-1e-9:
            return
        if self.future is None:
            if abs(now-self.arrivals_end) > 1e-9:
                raise ValueError(f'A34: first extended control second is {now}, not {self.arrivals_end}')
            self.totals_1200 = dict(b.totals)
            self._trim(env)
        self._withdraw(env, now)
        if self._empty(env):
            self._finish(env, now, 'network_empty')
            raise NetworkEmpty(now)

    def after_tick(self, b):
        env = b.env
        if float(env.now) >= self.end-1e-9 and self.final is None:
            self._finish(env, float(env.now), 'end_s')

    # -- mechanics -----------------------------------------------------------------------------------------------
    def _trim(self, env):
        led = env.ledger
        if abs(float(led.time)-self.arrivals_end) > 1e-9:
            raise ValueError('A34: ledger time is not the arrivals end')
        self.future = {vid: float(r.depart_s) for vid, r in led.requests.items()
                       if float(r.depart_s) > self.arrivals_end+1e-9}
        if set(self.future) & (set(led.inserted) | set(led.arrived)):
            raise ValueError('A34: a trip requested after the arrivals end was already inserted')
        led.requests = {vid: r for vid, r in led.requests.items() if vid not in self.future}
        led.times = sorted(float(r.depart_s) for r in led.requests.values())
        original, future = led.advance, self.future

        def advance(now, departed, arrived):
            if any(v in future and v not in self.withdrawn for v in departed):
                raise ValueError('A34: a withdrawn trip was inserted and not removed')
            return original(now, [v for v in departed if v not in future], [v for v in arrived if v not in future])
        led.advance = advance
        api = env.api
        step0 = api.simulationStep

        def step(*args, **kwargs):                  # every physics step from 1,200 s on
            out = step0(*args, **kwargs)
            for vid in api.simulation.getDepartedIDList():
                if vid in future and vid not in self.withdrawn:
                    api.vehicle.remove(vid)
                    if vid in api.vehicle.getIDList():
                        raise ValueError(f'A34: withdrawn trip {vid} still in the network after removal')
                    self.withdrawn.add(vid)
                    self.removed_at_insertion += 1
            return out
        api.simulationStep = step                   # the plant Backend restores the original step on close

    def _withdraw(self, env, now):
        api = env.api
        for vid in sorted(str(v) for v in api.simulation.getPendingVehicles()):
            if vid in self.future and vid not in self.withdrawn:
                api.vehicle.remove(vid)
                self.withdrawn.add(vid)
                self.removed_pending += 1

    @staticmethod
    def _empty(env):
        led = env.ledger
        return len(led.arrived) == len(led.requests) and int(env.api.vehicle.getIDCount()) == 0

    def _finish(self, env, now, reason):
        led = env.ledger
        never = sorted((vid, float(r.depart_s), bool(getattr(r, 'is_cav', False)))
                       for vid, r in led.requests.items() if vid not in led.inserted)
        self.final = dict(t_end_s=float(now), end_reason=reason, totals_end=dict(self.b.totals),
                          never_entered=[dict(id=v, depart_s=d, cav=c) for v, d, c in never],
                          on_road_at_end=int(env.api.vehicle.getIDCount()),
                          requested_by_1200=len(led.requests), inserted=len(led.inserted), arrived=len(led.arrived))

    def receipt(self, ttc_ext, ext_log):
        if self.final is None or self.totals_1200 is None:
            raise ValueError('A34: the extension did not reach its end')
        t = self.totals_1200
        return dict(arrivals_end_s=self.arrivals_end, end_s=self.end, **deepcopy(self.final),
                    J_1200_veh_s=float(t['team_delay_vs']), totals_1200=deepcopy(t),
                    safety_1200=dict(collisions=int(t['collisions']), teleports=int(t['teleports']),
                                     hard_brakes=int(t['hard_brake'])),
                    ttc_events_ext=int(ttc_ext), future_trips=len(self.future), withdrawn=len(self.withdrawn),
                    removed_pending=self.removed_pending, removed_at_insertion=self.removed_at_insertion,
                    actual_sim_s=float(self.b.actual), ext_log=ext_log)


# ---------------------------------------------------------------- evaluation-only vehicle log, extended

def vehicle_log_class(base, horizon=None):
    """``base`` = nm_eval_runner.VehicleLog: [300, 1200) unchanged in its file; [1200, end) in EXT_LOG."""

    class ExtendedVehicleLog(base):
        def __init__(self, work):
            super().__init__(work)
            if horizon is not None:
                horizon.log = self
            self.ext_path = Path(work)/EXT_LOG
            self.ext_stream = gzip.open(self.ext_path, 'xt', encoding='utf-8')
            self.ext_last, self.ext_ticks = None, 0
            self.ext_info = None

        def tick(self, env):
            now = int(round(float(env.now)))
            if now < ARRIVALS_END_S:
                return super().tick(env)
            if now == self.ext_last or now >= END_S:
                return
            self.ext_last = now
            stream, self.stream = self.stream, self.ext_stream
            last, ticks = self.last, self.ticks
            try:
                self._row(env, now)
            finally:
                self.stream, self.last, self.ticks = stream, last, ticks
            self.ext_ticks += 1

        def _row(self, env, now):
            """The base row writer (VehicleLog.tick body, identical fields) for one second >= 1,200."""
            import numpy as np
            f, ids = env.frame, env.builder.vehicle_ids
            api = getattr(env, 'api', None)
            k, s, v, a, lane = [], [], [], [], []
            for i in np.flatnonzero(f.present):
                vid = ids[int(f.vid_index[i])]
                n = self.index.get(vid)
                if n is None:
                    n = self.index[vid] = len(self.index)
                    origin = delay = None
                    if api is not None:
                        try:
                            route = api.vehicle.getRoute(vid)
                            origin = str(route[0]) if route else None
                        except Exception:
                            origin = None
                        try:
                            delay = float(api.vehicle.getDepartDelay(vid))
                        except Exception:
                            delay = None
                    self.stream.write(json.dumps(dict(veh=n, id=str(vid), cav=bool(f.is_cav[i]), origin=origin,
                                                      depart_delay_s=delay, first_t=now))+'\n')
                k.append(n)
                s.append(round(float(f.s[i]), 2))
                v.append(round(float(f.v[i]), 3))
                a.append(round(float(f.a[i]), 3))
                lane.append(int(f.lane_carry[i] if f.junction[i] else f.lane[i]))
            self.stream.write(json.dumps(dict(t=now, k=k, s=s, v=v, a=a, lane=lane))+'\n')

        def close(self):
            out = super().close()
            if not self.ext_stream.closed:
                self.ext_stream.close()
            h = hashlib.sha256(self.ext_path.read_bytes()).hexdigest()
            self.ext_info = dict(file=EXT_LOG, sha256=h, ticks=self.ext_ticks, vehicles_total=len(self.index),
                                 first_t=ARRIVALS_END_S if self.ext_ticks else None,
                                 last_t=self.ext_last)
            return out
    return ExtendedVehicleLog


# ---------------------------------------------------------------- one extended episode (runner side)

def check_spec(spec, runner):
    h = spec.get('horizon_a34') or {}
    if spec['arm'] not in ARMS:
        raise ValueError(f'A34: arm {spec["arm"]!r} is not registered ({ARMS})')
    if (h.get('arrivals_end_s'), h.get('end_s')) != (ARRIVALS_END_S, END_S):
        raise ValueError('A34: horizon field differs from the registration')
    if h.get('a34_sha256') != a34_digest():
        raise ValueError('A34: a34_horizon differs from its registration (a34_sha256)')
    if int(spec['sim_s']) != END_S:
        raise ValueError('A34: precharge must be the full 2,400 s')
    # the sealed checks (map world, window {0, 1200}, noise seed = world seed, deterministic) on the sealed spec view
    return runner._check_spec(dict(spec, sim_s=runner.HORIZON_S))


@contextmanager
def extension(horizon, runner):
    """Patch (this worker only, for one episode): cf.tick (before/after hooks), the runner's VehicleLog class and
    accel_plant.episode (NetworkEmpty -> the normal result from the plant's own totals)."""
    from tools.coopfree import backend as cf
    from . import accel_plant
    from .accel_storage import write
    tick0, episode0, vlog0 = cf.tick, accel_plant.episode, runner.VehicleLog

    def tick(b, controller=None, trace=None):
        horizon.before_tick(b)
        out = tick0(b, controller, trace)
        horizon.after_tick(b)
        return out

    def episode(job, **kwargs):
        try:
            return episode0(job, **kwargs)
        except NetworkEmpty:
            b = horizon.b
            result = dict(id=job['id'], world=job['world'], arm=job['arm'], metrics={'900': dict(horizon.totals_1200)},
                          totals=dict(b.totals), actual_sim_s=b.actual, charged_sim_s=job['sim_s'],
                          a34_end=dict(t_end_s=horizon.final['t_end_s'], end_reason=horizon.final['end_reason']))
            write(Path(job['path'])/'RESULT.json', result)
            return result
    cf.tick, accel_plant.episode, runner.VehicleLog = tick, episode, vehicle_log_class(vlog0, horizon)
    try:
        yield
    finally:
        cf.tick, accel_plant.episode, runner.VehicleLog = tick0, episode0, vlog0


def episode(spec, cfg, work):
    """One registered A34 episode -> graded record: the sealed arm's record fields (J_veh_s = J over [300, end);
    safety collisions/teleports/hard_brakes over [300, end), ttc_events over the sealed [300, 1200) window) plus the
    ``a34`` receipt (J and safety at 1,200 s, end, withdrawals, never-entered trips, ext log)."""
    from . import nm_eval_runner as R
    from .replay_bank import ttc_events
    world = check_spec(spec, R)
    horizon = Horizon()
    with extension(horizon, R):
        record = R._policy_run(spec, cfg, work, world=world, stop=END_S, role=R.ROLE)
    t_end = int(round(horizon.final['t_end_s']))
    ttc_ext = ttc_events(Path(work)/'observable.jsonl.gz', ARRIVALS_END_S, t_end)
    record['a34'] = horizon.receipt(ttc_ext, horizon.log.ext_info)
    record['a34']['safety_1200']['ttc_events'] = int(record['safety']['ttc_events'])
    record['a34_sha256_loaded'] = a34_digest()
    return record


# ---------------------------------------------------------------- laptop reducer (zero physics)

def merged_report(work_dir, record, vehicle_log_report, *, warmup=WARMUP_S):
    """Per-vehicle delay over [300, end) for EVERY vehicle of the episode: the registered vehicle_log_report on the
    concatenation of the [300, 1200) log and the [1200, end) log (same convention: sum of max(0, 1 - v/limit) per
    logged second + SUMO depart delay if first seen after 300), plus every trip requested by 1,200 s that never
    entered, charged its source wait to the end (end_s - max(depart, 300)), origin undefined, listed separately."""
    import tempfile
    work_dir = Path(work_dir)
    main, ext = work_dir/'vehicles.jsonl.gz', work_dir/EXT_LOG
    a = record['a34']
    for path, entry in ((main, record['vehicle_log']), (ext, a['ext_log'])):
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'{path}: differs from its record')
    last = a['ext_log']['last_t']
    t1 = (int(last)+1) if last is not None else ARRIVALS_END_S
    with tempfile.TemporaryDirectory() as tmp:
        merged = Path(tmp)/'merged.jsonl.gz'
        merged.write_bytes(main.read_bytes()+ext.read_bytes())       # multi-member gzip
        rep = vehicle_log_report(merged, t0=warmup, t1=t1)
    per = dict(rep['per_vehicle'])
    charged = []
    for x in a['never_entered']:
        if x['id'] in per:
            raise ValueError(f'{x["id"]}: never-entered trip appears in the vehicle log')
        d = float(a['end_s'])-max(float(x['depart_s']), float(warmup))
        charged.append(d)
        per[x['id']] = [round(d, 3), None, bool(x['cav'])]
    return dict(ok=rep['ok'], problems=rep['problems'], ticks=rep['ticks'], t1=t1, per_vehicle=per,
                logged_vehicles=len(rep['per_vehicle']), never_entered=len(charged),
                never_entered_charged_s=float(sum(charged)),
                never_entered_ids=[x['id'] for x in a['never_entered']])
