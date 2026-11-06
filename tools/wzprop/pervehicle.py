"""Eval-only per-vehicle window delay logger and per-group fairness statistics for the S-WMSEARCH SUMO verification
(owner 2026-11-07: per-group fairness metrics with paired CIs in every verification output).

PerVehicle is copied VERBATIM from tools/wzprop/lssearch.py on c4/lssearch d587b5a9 (S-LOWSHARE-SEARCH, the
LSSEARCH / MULTICRIT per-vehicle logger): the env's _after_physics is wrapped on the instance (installed at the first
observe, no physics change, no driver parameters); per vehicle: class (request is_cav label), origin lane (SUMO lane
index at first sighting), window road delay (0.5 s x max(0, 1 - v / 31.29), the env's own term) and window source
wait. Sums reconcile with the episode totals.

Fairness (registered, REGISTRATION 04:03 / S-MULTICRIT P1-P4): groups = origin SUMO lane 0/1/2 and HDV / CAV;
group mean per-vehicle delay change = mean_g(arm) - mean_g(ref) per paired seed; ALLOWED = every group's change
<= 10 s/veh (point estimate; the paired 95% CI upper bound is reported too). Per-vehicle delay component 'road'
(S-MULTICRIT P4 convention, source waiting excluded) is primary; 'road+source' (S-LOWSHARE-SEARCH) is reported too.
CVaR90 = mean of the worst ceil(0.1 n) per-vehicle delays; p90 = worst-off 10% threshold; Gini.
Convention: diff = arm - ref (same seed); negative = lower delay (better).
"""
from __future__ import annotations

import math
import random
import sys

WARMUP_S, EPISODE_S = 300, 1200
SCHEMA = 'wmc.wmsearch_u0.v2'
FAIR_GROUPS = ('lane0', 'lane1', 'lane2', 'hdv', 'cav')
BOUND_S = 10.0
COMPONENTS = ('road', 'road+source')


# ------------------------------------------------------------------------------------------ verbatim (c4/lssearch d587b5a9)
class PerVehicle:
    """Eval-only per-vehicle window delay with origin lane and class (see module docstring)."""

    def __init__(self, a=WARMUP_S, b=EPISODE_S):
        self.a, self.b = float(a), float(b)
        self.origin, self.road, self.ins = {}, {}, {}
        self.n_ins = 0
        self.arrived_a = None
        self.requests = None
        self.limit = None
        self.installed_at = None
        self.substeps = 0
        self.window_substeps = 0
        self.error = None

    def install(self, env):
        if getattr(env, '_lss_pv', None) is self:
            return
        try:
            self.limit = float(sys.modules[type(env).__module__].SPEED_LIMIT_MPS)
            orig = env._after_physics
            pv = self

            def wrapped(after, *args, **kwargs):
                out = orig(after, *args, **kwargs)
                pv.substep(env, float(after))
                return out
            env._after_physics = wrapped
            env._lss_pv = self
            self.installed_at = float(env.now)
            self.requests = {vid: (float(r.depart_s), bool(r.is_cav)) for vid, r in env.ledger.requests.items()}
            self._sync(env, float(env.now))
        except Exception as exc:          # logging must never alter the episode
            self.error = repr(exc)

    def _sync(self, env, after):
        led = env.ledger
        if len(led.inserted) != self.n_ins:
            for v in led.inserted:
                if v not in self.ins:
                    self.ins[v] = float(led.time)
            self.n_ins = len(led.inserted)
        for vid, t in env._table.items():
            if vid not in self.origin:
                self.origin[vid] = (str(t.get('road_id')), int(t.get('lane_index')), after)
        if self.arrived_a is None and after >= self.a - 1e-9:
            self.arrived_a = set(led.arrived)

    def substep(self, env, after):
        if self.error is not None:
            return
        try:
            self.substeps += 1
            self._sync(env, after)
            if self.a + 1e-9 < after <= self.b + 1e-9:
                self.window_substeps += 1
                lim, road = self.limit, self.road
                for vid, t in env._table.items():
                    road[vid] = road.get(vid, 0.) + 0.5 * max(0.0, 1.0 - float(t['speed_mps']) / lim)
        except Exception as exc:
            self.error = repr(exc)

    def finish(self):
        rows, sum_road, sum_src = [], 0., 0.
        edges = {}
        for vid, (dep, cav) in sorted((self.requests or {}).items()):
            if dep >= self.b or (self.arrived_a is not None and vid in self.arrived_a):
                continue
            ins = self.ins.get(vid)
            src = max(0., min(self.b if ins is None else ins, self.b) - max(dep, self.a))
            road = self.road.get(vid, 0.)
            o = self.origin.get(vid)
            sum_road += road
            sum_src += src
            if o is not None:
                edges[o[0]] = edges.get(o[0], 0) + 1
            rows.append([vid, int(cav), -1 if o is None else o[1], round(road, 6), round(src, 6)])
        pop = {r[0] for r in rows}
        missing_road = sum(v for k, v in self.road.items() if k not in pop)
        return dict(schema=SCHEMA + '.perveh', window=[self.a, self.b], columns=['vid', 'cav', 'origin_lane',
                                                                                'road_vs', 'source_vs'],
                    rows=rows, sum_road_vs=sum_road, sum_source_vs=sum_src, road_outside_population_vs=missing_road,
                    origin_edges=edges, installed_at=self.installed_at, substeps=self.substeps,
                    window_substeps=self.window_substeps, error=self.error)



def install_on_observe(pv):
    """Install pv at the first Backend.observe (as LSHook._install_pv); returns False when the backend module is not
    imported yet (pv.error is set, the episode is unaffected)."""
    mod = sys.modules.get('tools.leverprobe.backend')
    if mod is None:
        pv.error = 'tools.leverprobe.backend not imported before the hook'
        return False
    original = mod.Backend.observe

    def observe(b, env):
        pv.install(env)
        return original(b, env)
    mod.Backend.observe = observe
    return True


# ------------------------------------------------------------------------------------------ fairness statistics
def cvar90(x):
    x = sorted((float(v) for v in x), reverse=True)
    return sum(x[:max(1, math.ceil(0.1 * len(x)))]) / max(1, math.ceil(0.1 * len(x))) if x else None


def p90(x):
    x = sorted(float(v) for v in x)
    if not x:
        return None
    pos = 0.9 * (len(x) - 1)
    lo = int(math.floor(pos))
    return x[lo] + (x[min(lo + 1, len(x) - 1)] - x[lo]) * (pos - lo)


def gini(x):
    x = sorted(float(v) for v in x)
    n, s = len(x), sum(x)
    return None if n == 0 or s <= 0 else sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(x)) / (n * s)


def fairness_of(pv, component='road'):
    """Per-episode group means and tails of one arm's PERVEH record (None when missing / errored)."""
    if pv is None or pv.get('error'):
        return None
    per, groups = [], {g: [] for g in FAIR_GROUPS + ('uninserted',)}
    for _vid, cav, lane, road, src in pv['rows']:
        d = road + (src if component == 'road+source' else 0.0)
        per.append(d)
        groups['cav' if cav else 'hdv'].append(d)
        groups[f'lane{lane}' if lane in (0, 1, 2) else 'uninserted'].append(d)
    return dict(groups={g: dict(n=len(v), total_vs=sum(v), mean_vs=(sum(v) / len(v) if v else None))
                        for g, v in groups.items()},
                cvar90=cvar90(per), p90=p90(per), gini=gini(per), n=len(per), sum_vs=sum(per))


def boot_ci(diffs, n=10_000, seed=20261107):
    rng = random.Random(seed)
    k = len(diffs)
    means = sorted(sum(diffs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return [means[int(.025 * n)], means[int(.975 * n) - 1]]


def paired_fairness(fair_by_seed, arm, ref, bound=BOUND_S, n_boot=10_000):
    """fair_by_seed: {seed: {arm: fairness_of(...)}} -> paired (arm - ref) per-group mean per-vehicle delay change
    with 95% CIs, the 10 s/veh ALLOWED flag (point; CI upper reported), CVaR90 / p90 / Gini changes."""
    seeds = sorted(s for s, d in fair_by_seed.items() if d.get(arm) and d.get(ref))
    if not seeds:
        return None
    out = dict(n=len(seeds), ref=ref, bound_s=bound, groups={})
    for g in FAIR_GROUPS:
        d = [fair_by_seed[s][arm]['groups'][g]['mean_vs'] - fair_by_seed[s][ref]['groups'][g]['mean_vs']
             for s in seeds if fair_by_seed[s][arm]['groups'][g]['n'] and fair_by_seed[s][ref]['groups'][g]['n']]
        if d:
            ci = boot_ci(d, n_boot)
            out['groups'][g] = dict(n=len(d), mean_per_vehicle_diff=sum(d) / len(d), ci95=ci,
                                    within_bound_point=sum(d) / len(d) <= bound, within_bound_ci_upper=ci[1] <= bound,
                                    worse_sig=ci[0] > 0)
    for k in ('cvar90', 'p90', 'gini'):
        d = [fair_by_seed[s][arm][k] - fair_by_seed[s][ref][k] for s in seeds
             if fair_by_seed[s][arm][k] is not None and fair_by_seed[s][ref][k] is not None]
        if d:
            out[k] = dict(mean_diff=sum(d) / len(d), ci95=boot_ci(d, n_boot))
    gm = {g: v['mean_per_vehicle_diff'] for g, v in out['groups'].items()}
    out['worst_group'] = max(gm, key=gm.get) if gm else None
    out['worst_group_diff'] = gm[out['worst_group']] if gm else None
    out['allowed_point'] = bool(gm) and all(v['within_bound_point'] for v in out['groups'].values())
    out['allowed_ci_upper'] = bool(gm) and all(v['within_bound_ci_upper'] for v in out['groups'].values())
    return out
