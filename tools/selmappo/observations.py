"""Pure sensor-receipt feature builder: no simulator dependency or truth channel."""
import math
from dataclasses import dataclass

from .protocol import LIBRARY

SHARED_DIM = 46
ACTOR_DIM = SHARED_DIM + len(LIBRARY) + 8


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('nonfinite observable')
    return value


def telemetry(row, now):
    stamp = finite(row['timestamp_s'])
    lane = row['lane']
    if stamp > now or type(lane) is not int or lane not in (0, 1, 2):
        raise ValueError('invalid sensor clock/lane')
    return (finite(row['s_m']) / 1000, finite(row['v_mps']) / 31.29,
            finite(row['a_mps2']) / 3, *(float(lane == i) for i in range(3)),
            (now-stamp) / 60)


def pooled(rows, now):
    # Sort numeric telemetry only: detector IDs/ordering cannot affect even rounding.
    rows = sorted(telemetry(row, now) for row in rows)
    if not rows:
        return (0.,) * 11
    n = len(rows)
    means = [math.fsum(r[i] for r in rows)/n for i in range(7)]
    stds = [math.sqrt(math.fsum((r[i]-means[i])**2 for r in rows)/n) for i in (0, 1)]
    return (math.log1p(n), means[0], stds[0], means[1], stds[1],
            means[2], *means[3:6], means[6], 1.)


@dataclass(frozen=True)
class Observation:
    keys: tuple
    actors: tuple
    shared: tuple


def build(payload, *, variant, incumbents):
    """Only allowlisted fields are accessed; extra hidden fields have no influence.

    CAV IDs are routing keys. Neighbour/RSU channels are separately pooled
    receipts, not an assertion of a globally unique census or true penetration.
    """
    if variant not in ('fleet', 'per_cav'):
        raise ValueError('unknown selector variant')
    history = payload['history']
    if not history:
        raise ValueError('observable history missing')
    row = history[-1]
    now = finite(row['now_s'])
    own = row['self_reports']
    ids = [r['vehicle_id'] for r in own]
    if len(set(ids)) != len(ids) or any(not isinstance(k, str) or not k for k in ids):
        raise ValueError('duplicate/invalid CAV self identity')
    shared = [(now-300)/600]
    for field in ('self_reports', 'detections', 'neighbours'):
        shared.extend(pooled(row.get(field, ()), now))
    coverage = [0.] * 9
    for item in row.get('coverage', ()):
        lane = item['lane']
        if type(lane) is not int or lane not in (0, 1, 2) or coverage[6+lane]:
            raise ValueError('invalid/duplicate coverage')
        start, end = finite(item['s0_m']), finite(item['s1_m'])
        stamp = finite(item['timestamp_s'])
        if end < start or stamp > now:
            raise ValueError('invalid coverage interval')
        coverage[lane], coverage[3+lane], coverage[6+lane] = (end-start)/1000, (now-stamp)/60, 1.
    shared.extend(coverage)
    sensed = [r['counters'] for r in history if r.get('counters') is not None]
    flow = [0., 0., 0.]
    if len(sensed) >= 2:
        first, last = sensed[0], sensed[-1]
        dt = finite(last['timestamp_s'])-finite(first['timestamp_s'])
        if dt <= 0 or finite(last['timestamp_s']) > now:
            raise ValueError('invalid observed entrance clock')
        delta = [finite(last[k])-finite(first[k]) for k in ('requested', 'entered')]
        if any(x < 0 for x in delta):
            raise ValueError('observed counter reset')
        flow = [delta[0]/dt, delta[1]/dt, 1.]
    shared.extend(flow)
    if len(shared) != SHARED_DIM:
        raise AssertionError('observation schema drift')
    by_id = {r['vehicle_id']: r for r in own}
    keys = ('fleet',) if variant == 'fleet' else tuple(sorted(ids))
    actors = []
    for key in keys:
        incumbent = incumbents.get(key, 'native')
        if incumbent not in LIBRARY:
            raise ValueError('unknown incumbent')
        ego = (*telemetry(by_id[key], now), 1.) if variant == 'per_cav' else (0.,)*8
        actors.append(tuple(shared) + tuple(float(incumbent == c) for c in LIBRARY) + ego)
    return Observation(keys, tuple(actors), tuple(shared))
