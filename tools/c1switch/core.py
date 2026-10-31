"""CPU-only selection, observable receipt boundary, and budget-saving branching."""
import copy
import math
import statistics

from .design import CONTROLLERS

EPS = 1e-6
OBS_KEYS = ('vehicle_id', 'timestamp_s', 's_m', 'lane', 'v_mps', 'a_mps2')


def choose(costs, incumbent='native'):
    if set(costs) != set(CONTROLLERS) or any(not math.isfinite(v) for v in costs.values()):
        raise ValueError('five finite controller costs required')
    tied = [c for c in CONTROLLERS if costs[c] <= min(costs.values()) + EPS]
    return incumbent if incumbent in tied else tied[0]


def medians(samples):
    if set(samples) != set(CONTROLLERS):
        raise ValueError('wrong library')
    if any(len(v) != 8 or any(not math.isfinite(x) or x < 0 for x in v) for v in samples.values()):
        raise ValueError('K8 finite nonnegative team costs required')
    return {c:statistics.median(samples[c]) for c in CONTROLLERS}


class Belief:
    """Receives ONLY sensor receipts; has no environment/global-state reference."""
    def __init__(self):
        self.history = []

    def observe(self, now, *, self_reports, detections, coverage, counters=None):
        def sanitize(row, source):
            out = {k:row[k] for k in OBS_KEYS}
            numeric = [out[k] for k in ('timestamp_s', 's_m', 'v_mps', 'a_mps2')]
            if any(not math.isfinite(x) for x in numeric) or out['timestamp_s'] > now or out['lane'] not in (0, 1, 2):
                raise ValueError('invalid observable receipt')
            out.update(age_s=now-out['timestamp_s'], source=source)
            return out
        own = sorted((sanitize(r, 'cav_self') for r in self_reports), key=lambda r:r['vehicle_id'])
        seen = sorted((sanitize(r, 'rsu') for r in detections), key=lambda r:r['vehicle_id'])
        cov = []
        for c in coverage:
            cov.append({k:c[k] for k in ('lane', 's0_m', 's1_m', 'timestamp_s')})
        count = None if counters is None else {k:counters[k] for k in ('timestamp_s', 'requested', 'entered')}
        row = dict(now_s=now, self_reports=own, detections=seen, coverage=cov, counters=count)
        self.history.append(row)
        self.history = [h for h in self.history if h['now_s'] >= now-60]
        return self.payload()

    def payload(self):
        return copy.deepcopy(dict(schema='c1switch_observable/1', history=self.history))


class Handover:
    def __init__(self, factory):
        self.factory, self.name, self.controller = factory, 'native', None
        self.switches = 0

    def select(self, name, env):
        if name not in CONTROLLERS:
            raise ValueError(name)
        if name != self.name:
            env.ownership.release_all(live_ids=env.api.vehicle.getIDList())
            env.ownership.controller = env._circles
            env.pi_session, env._direct_pi = None, False
            self.name = name
            self.controller = self.factory(name)
            self.switches += 1
        return self.controller


def branch_all(backend, *, horizon=90, repeat_native=False):
    """Backend snapshot covers env+controller+belief+episode metrics, never RNG-only.

    Branch evaluation is isolated; selected 60 s state becomes the next trunk for free.
    """
    base = backend.save('decision')
    results, continuations = {}, {}
    for name in CONTROLLERS:
        backend.load(base)
        backend.select(name)
        metric0 = backend.metrics()
        backend.advance(60)
        continuations[name] = backend.save(name+'_60')
        m60 = backend.metrics()
        if horizon == 90:
            backend.advance(30)
        elif horizon != 60:
            raise ValueError('only 60/90 clocks')
        m90 = backend.metrics()
        results[name] = {str(h): {k:m[k]-metric0[k] for k in m} for h, m in ((60, m60), (horizon, m90))}
    if repeat_native:
        backend.load(base)
        backend.select('native')
        metric0 = backend.metrics()
        backend.advance(horizon)
        results['native_repeat'] = {k:v-metric0[k] for k, v in backend.metrics().items()}
    return results, continuations
