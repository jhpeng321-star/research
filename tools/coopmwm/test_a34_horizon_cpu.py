"""Amendment 34 horizon extension (CPU only, no SUMO): fake-physics plant with the RequestLedger accounting.

The fake plant mimics what the extension touches: env.now / end_s / ledger / api (vehicle.remove of loaded-not-
departed trips, getIDCount), b.totals (team delay = speed deficit + source wait), the 1 s control tick made of two
0.5 s physics steps, and an entrance with limited insertion capacity (a queue at 1,200 s)."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from tools.coopmwm import a34_horizon as H


@dataclass(frozen=True)
class Trip:
    vehicle_id: str
    depart_s: float
    is_cav: bool = False


class Ledger:
    """traffic_shepherd.metrics.RequestLedger (frozen runtime), verbatim accounting."""

    def __init__(self, requests):
        self.requests = {r.vehicle_id: r for r in requests}
        self.times = sorted(r.depart_s for r in requests)
        self.inserted, self.arrived = set(), set()
        self.time = 0.0
        self.system_vehicle_seconds = self.source_vehicle_seconds = 0.0

    def due(self, now):
        return bisect_right(self.times, now)

    def advance(self, now, departed, arrived):
        for ids, seen in ((departed, self.inserted), (arrived, self.arrived)):
            if set(ids)-self.requests.keys() or set(ids) & seen:
                raise ValueError('unknown or duplicate lifecycle event')
        if set(arrived)-(self.inserted | set(departed)):
            raise ValueError('arrival before insertion')
        if any(self.requests[v].depart_s > now+1e-6 for v in departed):
            raise ValueError('insertion before requested departure')
        dt = now-self.time
        prior = self.due(self.time)
        area = prior*dt+sum(now-x for x in self.times[prior:self.due(now)])
        system, source = area-len(self.arrived)*dt, area-len(self.inserted)*dt
        self.system_vehicle_seconds += max(0., system)
        self.source_vehicle_seconds += max(0., source)
        self.inserted.update(departed)
        self.arrived.update(arrived)
        self.time = now
        return max(0., system)


class Sumo:
    """Loads trips LOOKAHEAD s before departure; inserts at most ``capacity`` per 0.5 s step; travel ``travel`` s."""
    LOOKAHEAD = 200.

    def __init__(self, trips, capacity=1, travel=100.):
        self.trips = {t.vehicle_id: t for t in trips}
        self.capacity, self.travel = capacity, travel
        self.now, self.loaded, self.removed, self.live = 0., set(), set(), {}
        self.vehicle = types.SimpleNamespace(remove=self.remove, getIDCount=lambda: len(self.live))

    def remove(self, vid):
        if vid not in self.loaded or vid in self.live or vid in self.removed:
            raise KeyError(f'Vehicle {vid!r} is not known')
        self.loaded.discard(vid)
        self.removed.add(vid)

    def step(self):
        self.now += .5
        self.loaded |= {v for v, t in self.trips.items() if t.depart_s <= self.now+self.LOOKAHEAD
                        and v not in self.removed and v not in self.live and v not in self.done}
        arrived = [v for v, t0 in self.live.items() if self.now-t0 >= self.travel]
        for v in arrived:
            del self.live[v]
            self.done.add(v)
        pending = sorted((self.trips[v].depart_s, v) for v in self.loaded if self.trips[v].depart_s <= self.now+1e-9)
        departed = [v for _, v in pending[:self.capacity]]
        for v in departed:
            self.loaded.discard(v)
            self.live[v] = self.now
        return departed, arrived

    done: set


class Env:
    def __init__(self, trips, **kw):
        self.api = Sumo(trips, **kw)
        self.api.done = set()
        self.ledger = Ledger(trips)
        self.now, self.end_s = 0., H.ORIGINAL_END_S
        while self.now < 300:
            self.physics()

    def physics(self):
        out = dict(road=0., source=0.)
        for _ in range(2):
            departed, arrived = self.api.step()
            before = self.ledger.source_vehicle_seconds
            self.ledger.advance(self.api.now, departed, arrived)
            out['source'] += self.ledger.source_vehicle_seconds-before
            out['road'] += .5*.2*len(self.api.live)          # constant 20 % speed deficit on the road
        self.now = self.api.now
        return out


class Plant:
    def __init__(self, trips, **kw):
        self.env = Env(trips, **kw)
        self.totals = dict.fromkeys(('team_delay_vs', 'road_delay_vs', 'source_wait_vs', 'collisions', 'teleports',
                                     'ttc_below_3_vs', 'hard_brake'), 0.)
        self.actual = 300.


def tick(b, controller=None, trace=None):
    x = b.env.physics()
    b.actual += 1.
    b.totals['road_delay_vs'] += x['road']
    b.totals['source_wait_vs'] += x['source']
    b.totals['team_delay_vs'] += x['road']+x['source']
    if b.env.now >= b.env.end_s-1e-9:
        raise ValueError('early termination/safety/slot/lane anomaly')


def trips(n=6000, rate=1.2, seed=0):
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.exponential(1/rate, n))
    return [Trip(f'v{i}', float(round(x*2)/2), bool(i % 5 == 0)) for i, x in enumerate(t) if x < 1800]


def run(plant, horizon=None, stop=1200):
    """accel_plant.episode loop shape: one tick per control second from 300; NetworkEmpty ends the episode."""
    try:
        for _ in range(300, stop):
            if horizon is not None:
                horizon.before_tick(plant)
            tick(plant)
            if horizon is not None:
                horizon.after_tick(plant)
    except H.NetworkEmpty:
        pass
    return plant


def test_identity_before_1200_and_no_arrivals_after():
    ts = trips()
    base = run(Plant(ts))                                      # the sealed 1,200 s episode
    hz = H.Horizon()
    ext = run(Plant(ts), hz, stop=H.END_S)
    assert hz.totals_1200 == base.totals                        # plant identity over [300, 1200)
    future = {t.vehicle_id for t in ts if t.depart_s > 1200}
    assert future and set(hz.future) == future
    assert not future & set(ext.env.api.done) and not future & set(ext.env.api.live)
    assert hz.final['end_reason'] == 'network_empty' and hz.final['on_road_at_end'] == 0
    assert hz.final['never_entered'] == [] and hz.final['arrived'] == hz.final['requested_by_1200']
    assert hz.withdrawn == future & ext.env.api.removed
    assert ext.totals['team_delay_vs'] > base.totals['team_delay_vs']
    assert ext.env.end_s == H.END_S+1


def test_entrance_queue_at_1200_enters_and_accrues_source_wait():
    ts = trips(rate=2.5)                                        # > capacity 2/s: a queue builds
    hz = H.Horizon()
    ext = run(Plant(ts, capacity=1), hz, stop=H.END_S)
    led = ext.env.ledger
    queued = [t for t in ts if t.depart_s <= 1200]
    assert len(queued) == len(led.requests)
    after = hz.final['totals_end']['source_wait_vs']-hz.totals_1200['source_wait_vs']
    assert after > 0                                            # the 1,200 s backlog keeps waiting
    assert hz.final['end_reason'] == 'network_empty' and hz.final['inserted'] == len(queued)


def test_not_empty_by_2400_reports_never_entered_and_on_road():
    ts = trips(rate=2.5)
    hz = H.Horizon()
    ext = run(Plant(ts, capacity=1, travel=1500.), hz, stop=H.END_S)
    assert hz.final['end_reason'] == 'end_s' and hz.final['t_end_s'] == H.END_S
    assert hz.final['on_road_at_end'] > 0
    rec = hz.receipt(0, dict(file=H.EXT_LOG, sha256='0'*64, ticks=1, vehicles_total=1, first_t=1200, last_t=2399))
    assert set(H.RECEIPT_KEYS) <= set(rec)
    assert rec['J_1200_veh_s'] == hz.totals_1200['team_delay_vs']
    assert ext.actual <= H.END_S


def test_withdraw_refuses_when_a_future_trip_cannot_be_held():
    ts = trips()
    plant = run(Plant(ts))
    hz = H.Horizon()
    hz.b = plant
    plant.env.end_s = H.END_S+1
    hz.totals_1200 = dict(plant.totals)
    hz._trim(plant.env)
    vid = min(hz.future, key=hz.future.get)
    plant.env.api.loaded.discard(vid)                          # not removable and due now
    plant.env.api.removed.add(vid)
    hz.future[vid] = 1200.5
    with pytest.raises(ValueError, match='cannot withdraw'):
        hz._withdraw(plant.env, 1200.)


def test_refuses_unexpected_plant_end():
    plant = Plant(trips())
    plant.env.end_s = 1500.
    with pytest.raises(ValueError, match='end_s'):
        H.Horizon().before_tick(plant)


def test_extension_context_routes_network_empty_to_a_result(tmp_path, monkeypatch):
    """extension(): cf.tick / accel_plant.episode / VehicleLog patched for one episode, then restored."""
    from tools.coopfree import backend as cf
    from tools.coopmwm import accel_plant, nm_eval_runner as R
    plant = Plant(trips())

    def fake_episode(job, controller_hook=None, policy_loader=None):
        for _ in range(300, job['stop']):
            cf.tick(plant)
        return dict(totals=dict(plant.totals))
    monkeypatch.setattr(cf, 'tick', tick)
    monkeypatch.setattr(accel_plant, 'episode', fake_episode)
    hz = H.Horizon()
    with H.extension(hz, R):
        result = accel_plant.episode(dict(id='x', world={}, arm='A', sim_s=2400, stop=2400, path=str(tmp_path)))
        assert R.VehicleLog.__name__ == 'ExtendedVehicleLog'
    assert cf.tick is tick and accel_plant.episode is fake_episode and R.VehicleLog.__name__ == 'VehicleLog'
    assert result['metrics']['900'] == hz.totals_1200 and result['totals'] == plant.totals
    assert result['a34_end']['end_reason'] == 'network_empty'
    assert (tmp_path/'RESULT.json').is_file()


class FrameEnv:
    """Minimal env for the vehicle log: frame arrays + builder ids + api getters."""

    def __init__(self):
        self.builder = types.SimpleNamespace(vehicle_ids=['h0', 'c0', 'h1'])
        self.api = types.SimpleNamespace(vehicle=types.SimpleNamespace(
            getRoute=lambda v: ['edgeA'], getDepartDelay=lambda v: 3.0))
        self.now = 0.

    def at(self, now, present):
        n = 3
        self.now = float(now)
        self.frame = types.SimpleNamespace(
            present=np.array([i in present for i in range(n)]), vid_index=np.arange(n),
            is_cav=np.array([False, True, False]), s=np.array([-500.+now, -480.+now, -520.+now]),
            v=np.array([20., 25., 10.+now % 3]), a=np.array([.1*(now % 3), -.2, .3]),
            lane_carry=np.zeros(n, int), junction=np.zeros(n, bool), lane=np.array([1, 1, 1]))
        return self


def test_extended_vehicle_log_splits_files_and_shares_the_index(tmp_path):
    from tools.coopmwm import nm_eval_runner as R
    cls = H.vehicle_log_class(R.VehicleLog)
    log = cls(tmp_path)
    env = FrameEnv()
    for t in range(300, 1203):
        log.tick(env.at(t, {0, 1} if t < 1201 else {0, 1, 2}))
    main = log.close()
    assert main['ticks'] == 900 and main['vehicles'] == 3
    assert log.ext_info['ticks'] == 3 and log.ext_info['last_t'] == 1202
    with gzip.open(tmp_path/'vehicles.jsonl.gz', 'rt') as f:
        rows = [json.loads(x) for x in f]
    assert [r['t'] for r in rows if 't' in r] == list(range(300, 1200))
    with gzip.open(tmp_path/H.EXT_LOG, 'rt') as f:
        ext = [json.loads(x) for x in f]
    assert [r['t'] for r in ext if 't' in r] == [1200, 1201, 1202]
    heads = [r for r in ext if 'veh' in r]
    assert heads == [dict(veh=2, id='h1', cav=False, origin='edgeA', depart_delay_s=3.0, first_t=1201)]


def test_merged_report_all_vehicles_and_never_entered_charge(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'ops'))
    import nm_finaleval as N
    from tools.coopmwm import nm_eval_runner as R
    log = H.vehicle_log_class(R.VehicleLog)(tmp_path)
    env = FrameEnv()
    for t in range(300, 1210):
        log.tick(env.at(t, {0, 1} if t < 1201 else {0, 1, 2}))
    main = log.close()
    rec = dict(vehicle_log=main, a34=dict(ext_log=log.ext_info, end_s=2400.,
                                          never_entered=[dict(id='q9', depart_s=1190., cav=False)]))
    rep = H.merged_report(tmp_path, rec, N.vehicle_log_report)
    assert rep['ok'], rep['problems']
    assert rep['t1'] == 1210 and set(rep['per_vehicle']) == {'h0', 'c0', 'h1', 'q9'}
    assert rep['per_vehicle']['q9'] == [1210.0, None, False] and rep['never_entered_charged_s'] == 1210.
    # h1 first seen at 1201 > 300: its depart delay (3 s) counts; deficit of v in [10, 12] over 9 s
    assert rep['per_vehicle']['h1'][0] > 3.0
    orig = N.vehicle_log_report(tmp_path/'vehicles.jsonl.gz')
    assert orig['per_vehicle']['h0'][0] < rep['per_vehicle']['h0'][0]
