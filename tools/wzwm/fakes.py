"""Planted fake world + fake SUMO core for CPU dry runs and tests (NO SUMO, no GPU, no real data).

PlantedWorld.effect(members) is the ground-truth J reduction (veh-s over 300 s vs base) of a set
of hold members. It has a planted optimum (v_w 18 m/s, hold 120 s, onset 10 s, dmax 1.5) and a
sweet spot of the member centroid; walls/pairs that include the closing lane earn a group bonus,
through-lane solos mostly hurt. The structured N3 variants therefore earn small positive effects,
random member sets mostly none - enough to exercise every funnel branch.

The fake core reads a nm2_core JOB.json and writes a RESULT.json with the same keys nm2_core
writes (J_road/J_src per second, safety counters, tracks, release_el, issued, saved, snapshots for
natives), so the vendored pg_plan / pg_analyze run unchanged on it:

    python -m tools.wzwm.fakes JOB.json
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

from . import geometry as G

OPT = dict(v_w=18.0, hold_s=120, start_s=10, dmax=1.5)
SWEET_X = G.X_B - 1100.0


def _rng(*parts):
    return random.Random(hashlib.sha256(':'.join(str(p) for p in parts).encode()).hexdigest())


class PlantedWorld:
    def __init__(self, group_gain=2400.0, closing_solo=120.0, through_cost=90.0, scale=1.0):
        self.group_gain = float(group_gain)
        self.closing_solo = float(closing_solo)
        self.through_cost = float(through_cost)
        self.scale = float(scale)

    def closeness(self, p):
        c = math.exp(-((float(p['v_w']) - OPT['v_w']) / 4.0) ** 2 - ((float(p['hold_s']) - OPT['hold_s']) / 60.0) ** 2
                     - ((float(p.get('start_s', 0)) - OPT['start_s']) / 8.0) ** 2)
        c *= 1.0 if float(p['dmax']) >= OPT['dmax'] else 0.8
        early = float(p.get('release_x', G.X_B)) < G.X_B or 'anchor_release_x' in p
        return c * (0.9 if early else 1.0)

    def effect(self, members):
        """Truth J reduction (veh-s / 300 s) of executing these hold members together."""
        if not members:
            return 0.0
        p = members[0]['params']
        close = self.closeness(p)
        xs = [float(m['x']) for m in members]
        pos = math.exp(-((sum(xs) / len(xs) - SWEET_X) / 350.0) ** 2)
        cut = max(0.0, G.SPEED_LIMIT - float(p['v_w'])) / 10.0 * float(p['hold_s']) / 120.0
        solo = 0.0
        for m in members:
            if int(m['lane']) == 0:
                solo += self.closing_solo * close * pos
            else:
                solo -= self.through_cost * cut
        lanes = {int(m['lane']) for m in members}
        bonus = 0.0
        if len(members) >= 2 and 0 in lanes and max(xs) - min(xs) <= 220.0:
            bonus = self.group_gain * close * pos * (len(lanes) / 3.0) ** 1.5
        return self.scale * (solo + bonus)


# ------------------------------------------------------------------------------------- fake natives
def fake_cavs(seed, penetration, t):
    """Deterministic CAV self-states (id, edge, lane, x, v) at time t of a fake world."""
    r = _rng('fake_cavs', seed, penetration)
    n = max(12, int(round(40 * float(penetration))))
    base = []
    for i in range(n):
        base.append((f'main.{100 + i}', r.uniform(150.0, 3900.0), r.choice((0, 1, 2)), r.uniform(12.0, 28.0)))
    # guarantee a dense member zone: a wall-capable cluster and a pair near the sweet spot
    for j, (lane, dx) in enumerate(((0, 0.0), (1, 30.0), (2, -40.0), (0, 150.0), (1, 120.0), (2, 90.0))):
        base.append((f'main.{90 + j}', SWEET_X + dx + r.uniform(-60.0, 60.0), lane, r.uniform(15.0, 25.0)))
    shift = _rng('fake_shift', seed, t).uniform(-40.0, 40.0)
    out = []
    for vid, x, lane, v in base:
        x = x + shift
        if x < G.X_B:
            out.append(dict(id=vid, edge='up', lane=lane, x=round(x, 2), v=round(v, 3)))
        elif x < 4000.0:
            out.append(dict(id=vid, edge='wz', lane=min(lane, 1), x=round(x, 2), v=round(v, 3)))
    return sorted(out, key=lambda c: c['id'])


def run_native(job):
    snaps = {}
    for t in job.get('save_times', []):
        cavs = fake_cavs(job['seed'], job['penetration'], int(t))
        snaps[f't{int(t)}'] = dict(cavs=cavs, lane_counts={}, approach_speed=12.0, approach_n=20, closing_lane_slow=3,
                                   inserted=100 + int(t))
    n = int(job['t_end'])
    return dict(kind='native', job=job, J_road=[1.0] * n, J_src=[0.0] * n, arrivals=[0] * n, J_total=float(n),
                collisions=0, collision_ids=[], teleports=0, hard_brake_vs=0.0, min_accel=-1.0, emergency_steps=0,
                snapshots=snaps, diag=[], sim_s=float(n), n_requests=0, n_cav=0, routes_sha256='fake')


def _native_cavs(job):
    p = Path(job['native_dir']) / 'RESULT.json'
    nat = json.loads(p.read_text())
    return {c['id']: c for c in nat['snapshots'][f"t{int(job['t0'])}"]['cavs']}


def run_branch(job, world=None):
    world = world or PlantedWorld()
    H = int(job['H'])
    members = list(job.get('members') or [])
    dither = list(job.get('dither') or [])
    sid, rep = job['sid'], int(job['rep'])
    ext = job['kind'] == 'extend'
    eff = world.effect(members)
    if members:                                            # near-chaotic replicate sensitivity
        chaos = _rng('chaos', sid, rep, ','.join(sorted(m['id'] for m in members))).gauss(0.0, 0.35)
        eff *= 1.0 + (chaos if dither else 0.0)
    if ext:
        eff = eff * 0.4 if 'g__' in job['arm'] else 0.0
    rate = eff / (120.0 if ext else 300.0)
    common = _rng('common', sid, rep, 'ext' if ext else 'main')
    arm = _rng('arm', sid, rep, job['arm'], 'ext' if ext else 'main')
    road = [round(max(0.0, 25.0 + common.gauss(0.0, 2.0) + arm.gauss(0.0, 0.6) - rate), 5) for _ in range(H)]
    cavs = _native_cavs(job)
    tracks, issued, release = {}, {}, {}
    track = list(dict.fromkeys(list(job.get('track_ids') or []) + [m['id'] for m in members + dither]))
    for vid in track:
        c = cavs.get(vid)
        if c is None:
            continue
        x, v = float(c['x']), float(c['v'])
        rows = []
        for el in range(min(H, int(job.get('track_s', 200))) + 1):
            edge = 'up' if x < G.X_B else 'wz'
            rows.append((el, edge, int(c['lane']) if edge == 'up' else min(int(c['lane']), 1), round(x, 2), round(v, 4)))
            x += v
        tracks[vid] = rows
    for m in members:
        p = m['params']
        rel = int(p.get('start_s', 0)) + int(p['hold_s'])
        release[m['id']] = min(rel, H)
        issued[m['id']] = [(el, round(max(-p['dmax'], min(p['amax'], p['v_w'] - m['v'])), 4))
                           for el in range(int(p.get('start_s', 0)), min(rel, H))]
    saved = {}
    for e in job.get('save_at', []):
        saved[f'e{int(e)}'] = dict(t=float(job['t_load']) + H, inserted=int(job['inserted_load']) + 7)
    return dict(J_road=road, J_src=[0.0] * H, arrivals=[0] * H, J_total=round(sum(road), 4), collisions=0,
                collision_ids=[], teleports=0, hard_brake_vs=1.0, min_accel=-2.0, emergency_steps=0,
                kind=job['kind'], job=job, sim_s=float(H), arm_events=[], issued=issued, release_el=release,
                tracks=tracks, saved=saved)


def run_job(job, world=None):
    res = run_native(job) if job['kind'] == 'native' else run_branch(job, world)
    res['status'] = 'ok'
    res['fake'] = True
    return res


def write_result(job, res):
    out = Path(job['out_dir'])
    out.mkdir(parents=True, exist_ok=True)
    if job['kind'] == 'native':
        (out / 'states').mkdir(exist_ok=True)
        (out / 'scenario').mkdir(exist_ok=True)
    (out / 'RESULT.json').write_text(json.dumps(res))


def main(path):
    job = json.loads(Path(path).read_text())
    t = time.time()
    try:
        res = run_job(job)
    except Exception as exc:  # failure receipt, never silently retried (nm2_core contract)
        import traceback
        res = dict(status='failed', job=job, error=repr(exc), tb=traceback.format_exc())
    res['elapsed_s'] = round(time.time() - t, 3)
    write_result(job, res)


if __name__ == '__main__':
    main(sys.argv[1])
