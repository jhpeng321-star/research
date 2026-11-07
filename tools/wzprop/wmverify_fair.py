"""S-WMVERIFY per-group fairness = N6's registered convention, ported EXACTLY (MAIN 2026-11-07, review REV-WMV3
blocker 1: training, eval and verification use one definition).

Sources (c4/nm-finaleval 4397d777):
* VehicleLog / logged_tick  <- tools/coopmwm/nm_eval_runner.py (evaluation-only ground-truth log of the plant's
  pre-step frame every whole second t in [300, 1200): per vehicle header {veh, id, cav, origin, depart_delay_s,
  first_t}, per tick {t, k, s, v, a, lane}; lane = WM lane (lane_carry on junction lanes)); read-only TraCI getters.
* per_vehicle              <- ops/nm_finaleval.py vehicle_log_report (per-vehicle delay = sum over logged seconds of
  max(0, 1 - v / 31.29) + depart delay if inserted after t0 = 300; origin = WM lane of the vehicle's FIRST logged
  row mapped ORIG {2: L0, 1: L1, 0: L2} if that row is upstream of the drop (s < 75.44), else 'D' (undefined)).
* _labels / _group_means / co_primary_groups <- ops/nm_finaleval.py (amendment 13): labels from the REFERENCE arm
  (u0) on the same world, applied by vehicle id to both compared arms; groups origin:L0/L1/L2, class:hdv, class:cav
  (co-primary) + origin:undefined (reported only); per group the per-world mean per-vehicle delay of B minus A must
  average <= 10 s (point estimate; 95% CI reported, paired whole-world percentile bootstrap = clean3_finaleval
  .paired_stats via planner_eval._paired_ci, BOOTSTRAP of nm_finaleval); a group needs >= 2 worlds, ALL five
  co-primary groups must be present and pass; vehicles of an arm absent from the reference and reference vehicles
  absent from the arm are counted; own-arm labels are a sensitivity only.
Only the B/A arm names are generalised (B = the verified arm, A = u0 or the incumbent).
"""
from __future__ import annotations

import gzip
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

WARMUP_S = 300
HORIZON_S = 1200
SPEED_LIMIT_MPS = 31.29                    # traffic_shepherd.c4.env.SPEED_LIMIT_MPS (team-delay term)
DROP_S_M = 75.44                           # N1 descriptor lane drop (s)
GROUP_DELAY_MAX_S = 10.0
BOOTSTRAP = dict(reps=10000, level=.95, seed=20261104, method='paired_percentile')   # nm_finaleval.BOOTSTRAP
ORIG = {2: 'L0', 1: 'L1', 0: 'L2'}
CO_PRIMARY_GROUPS = ('origin:L0', 'origin:L1', 'origin:L2', 'class:hdv', 'class:cav')
REPORTED_GROUPS = CO_PRIMARY_GROUPS + ('origin:undefined',)
VEHICLE_LOG = 'VEHICLES.jsonl.gz'
SOURCE = 'c4/nm-finaleval 4397d777 (nm_eval_runner.VehicleLog, nm_finaleval.vehicle_log_report / co_primary_groups)'


# ------------------------------------------------------------------------------------------ the log (verbatim port)
class VehicleLog:
    """EVALUATION-ONLY ground truth, read-only: never returned to any policy/controller/observation path."""

    def __init__(self, work):
        self.path = Path(work) / VEHICLE_LOG
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = gzip.open(self.path, 'xt', encoding='utf-8')
        self.index, self.last, self.ticks = {}, None, 0

    def tick(self, env):
        now = int(round(float(env.now)))
        if now == self.last or now < WARMUP_S or now >= HORIZON_S:
            return
        self.last = now
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
                    except Exception:      # read-only getter; absence is logged as None
                        origin = None
                    try:
                        delay = float(api.vehicle.getDepartDelay(vid))
                    except Exception:
                        delay = None
                self.stream.write(json.dumps(dict(veh=n, id=str(vid), cav=bool(f.is_cav[i]), origin=origin,
                                                  depart_delay_s=delay, first_t=now)) + '\n')
            k.append(n)
            s.append(round(float(f.s[i]), 2))
            v.append(round(float(f.v[i]), 3))
            a.append(round(float(f.a[i]), 3))
            lane.append(int(f.lane_carry[i] if f.junction[i] else f.lane[i]))
        self.stream.write(json.dumps(dict(t=now, k=k, s=s, v=v, a=a, lane=lane)) + '\n')
        self.ticks += 1

    def close(self):
        if not self.stream.closed:
            self.stream.close()
        import hashlib
        return dict(file=VEHICLE_LOG, sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(), ticks=self.ticks,
                    vehicles=len(self.index))


@contextmanager
def logged_tick(log):
    """Wrap tools.coopfree.backend.tick (the plant's 1 s step) to log the pre-step ground truth (read-only)."""
    from tools.coopfree import backend as cf
    original = cf.tick

    def tick(b, controller=None, trace=None):
        log.tick(b.env)
        return original(b, controller, trace)
    cf.tick = tick
    try:
        yield log
    finally:
        cf.tick = original


# ------------------------------------------------------------------------------------------ per-vehicle (port)
def per_vehicle(path, *, t0=None, t1=None):
    """{vehicle id: [delay_s, origin label | 'D' | None, cav]} of one vehicle log + problems (vehicle_log_report's
    per-vehicle part, same arithmetic; default window [WARMUP_S, HORIZON_S))."""
    t0 = WARMUP_S if t0 is None else t0
    t1 = HORIZON_S if t1 is None else t1
    heads, ticks, problems = {}, [], []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            x = json.loads(line)
            if 'veh' in x:
                heads[x['veh']] = x
            else:
                ticks.append(x)
    if [x.get('t') for x in ticks] != list(range(t0, t1)):
        problems.append(f'ticks {len(ticks)} != {t1-t0} consecutive seconds from {t0}')
    if not heads:
        problems.append('no vehicles')
    cols = {c: [] for c in ('t', 'k', 's', 'v', 'lane')}
    for x in ticks:
        n = len(x.get('k', []))
        cols['t'].append(np.full(n, x.get('t', -1), np.int64))
        for c in ('k', 's', 'v', 'lane'):
            cols[c].append(np.asarray(x[c], dtype=np.float64 if c in ('s', 'v') else np.int64)
                           if n else np.zeros(0, np.float64 if c in ('s', 'v') else np.int64))
    T, K, S, V, L = (np.concatenate(cols[c]) if cols[c] else np.zeros(0) for c in ('t', 'k', 's', 'v', 'lane'))
    K, L, T = K.astype(np.int64), L.astype(np.int64), T.astype(np.int64)
    nveh = (max(heads) + 1) if heads else 0
    deficit = np.maximum(0., 1. - V / SPEED_LIMIT_MPS)
    seen = np.bincount(K, minlength=nveh) if K.size else np.zeros(nveh, np.int64)
    delay_v = np.bincount(K, weights=deficit, minlength=nveh) if K.size else np.zeros(nveh)
    for k, h in heads.items():
        if seen[k] and h.get('first_t', -1) > t0 and isinstance(h.get('depart_delay_s'), (int, float)):
            delay_v[k] += float(h['depart_delay_s'])
    present = np.flatnonzero(seen > 0)
    first_lane, first_s = {}, {}
    if K.size:                                   # the vehicle's FIRST logged row (training convention)
        o = np.lexsort((T, K))
        kk, ll, ss = K[o], L[o], S[o]
        first = np.r_[True, kk[1:] != kk[:-1]]
        first_lane = dict(zip(kk[first].tolist(), ll[first].tolist()))
        first_s = dict(zip(kk[first].tolist(), ss[first].tolist()))

    def origin_of(k):                            # c4/clean5b 1caf5c1c fair_vehicle_log.per_vehicle, verbatim rule
        return (ORIG.get(first_lane[k]) if first_s[k] < DROP_S_M else 'D') if k in first_lane else None
    with_delay = sum(1 for h in heads.values() if isinstance(h.get('depart_delay_s'), (int, float)))
    if heads and with_delay < .95 * len(heads):
        problems.append(f'depart delay missing for {len(heads)-with_delay}/{len(heads)} vehicles')
    pv = {str(heads[k]['id']): [round(float(delay_v[k]), 3), origin_of(k), bool(heads[k].get('cav'))]
          for k in present.tolist() if k in heads}
    return pv, problems


# ------------------------------------------------------------------------------------------ groups (port)
def _group_means(per_vehicle, labels):
    """{group: mean per-vehicle delay} over the vehicles of ``per_vehicle`` that carry a label."""
    sums = {}
    for vid, (delay, _, _) in per_vehicle.items():
        for g in labels.get(vid, ()):
            n, total = sums.get(g, (0, 0.))
            sums[g] = (n + 1, total + delay)
    return {g: total / n for g, (n, total) in sums.items()}


def _labels(per_vehicle):
    """Training convention (c4/clean5b 1caf5c1c fair_vehicle_log.groups): every vehicle is in its class group;
    it is in origin:L0/L1/L2 only if its origin is defined (first logged row upstream of the drop); otherwise it
    goes to the separately reported origin:undefined group (never part of the criterion; still in total delay)."""
    out = {}
    for vid, (_, origin, cav) in per_vehicle.items():
        out[vid] = ('class:cav' if cav else 'class:hdv',
                    f'origin:{origin}' if origin in ORIG.values() else 'origin:undefined')
    return out


def paired_stats(treat, comp, bootstrap):
    """clean3_finaleval.paired_stats(..., pct=False) (registered paired whole-world percentile bootstrap)."""
    from tools.coopmwm import planner_eval as pe
    t, c = np.asarray(treat, float), np.asarray(comp, float)
    alpha = 1 - bootstrap['level']
    ci = pe._paired_ci(t - c, bootstrap['reps'], bootstrap['seed'], alpha)
    return dict(n=ci['n_worlds'], mean_treatment=float(t.mean()), mean_comparator=float(c.mean()),
                diff=ci['mean_delta_vs'], ci95=ci['ci95'], p_two_sided=ci['p_two_sided'],
                worlds_lower=ci['worlds_improved'], worlds_higher=int(np.sum(t - c > 0)))


def co_primary_groups(reports, worlds, arm_b, arm_a, reference='u0', bootstrap=BOOTSTRAP):
    """nm_finaleval.co_primary_groups with B = ``arm_b``, A = ``arm_a``: reports[arm][world] = per_vehicle dict."""
    if not {arm_b, arm_a, reference} <= set(reports):
        return dict(status='unavailable', reason=f'needs {arm_b}, {arm_a} and {reference} vehicle logs')
    worlds = [w for w in worlds if all(w in reports[x] for x in (arm_b, arm_a, reference))]

    def compare(label_of):
        groups = {}
        missing = {arm: dict(not_in_reference=0, reference_not_in_arm=0) for arm in (arm_b, arm_a)}
        per = {g: [] for g in REPORTED_GROUPS}
        for w in worlds:
            means = {}
            ref = reports[reference][w]
            for arm in (arm_b, arm_a):
                pv = reports[arm][w]
                labels = label_of(arm, w)
                missing[arm]['not_in_reference'] += sum(1 for vid in pv if vid not in ref)
                missing[arm]['reference_not_in_arm'] += sum(1 for vid in ref if vid not in pv)
                means[arm] = _group_means(pv, labels)
            for g in REPORTED_GROUPS:
                if g in means[arm_b] and g in means[arm_a]:
                    per[g].append((means[arm_b][g], means[arm_a][g]))
        for g, pairs in per.items():
            if len(pairs) < 2:
                groups[g] = dict(status='unavailable', n_worlds=len(pairs))
                continue
            b, a = (np.array(x, float) for x in zip(*pairs))
            stats = paired_stats(b, a, bootstrap)
            groups[g] = dict(n_worlds=len(pairs), increase_mean_s=stats['diff'], ci95=stats['ci95'],
                             passes=bool(stats['diff'] <= GROUP_DELAY_MAX_S), registered=g in CO_PRIMARY_GROUPS)
        ok = [groups[g] for g in CO_PRIMARY_GROUPS if 'passes' in groups[g]]
        return dict(groups=groups, missing=missing,
                    all_pass=len(ok) == len(CO_PRIMARY_GROUPS) and all(v['passes'] for v in ok))
    registered = compare(lambda arm, w: _labels(reports[reference][w]))
    own = compare(lambda arm, w: _labels(reports[arm][w]))
    return dict(rule=(f'every registered group (origin:L0/L1/L2, class:hdv, class:cav; labels from {reference} on the '
                      'same world by vehicle id; origin = WM lane of the first logged row if upstream of the drop) '
                      f'mean per-vehicle delay increase ({arm_b} - {arm_a}) <= {GROUP_DELAY_MAX_S} s (point estimate; '
                      '95% CI reported); origin:undefined reported only'), reference=reference, b=arm_b, a=arm_a,
                n_worlds=len(worlds), source=SOURCE,
                groups=registered['groups'], all_pass=registered['all_pass'],
                vehicles_missing=registered['missing'],
                sensitivity_own_arm_labels=dict(groups=own['groups'], all_pass=own['all_pass']))
