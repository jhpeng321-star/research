"""clean5b-F per-vehicle ground-truth logger for TRAINING episodes (MAIN 2026-11-07; eval-only, never a policy input).

PORTED VERBATIM from N6 (c4/nm-finaleval f565c5a7): ``_sha``, ``VehicleLog`` and ``logged_tick``
(tools/coopmwm/nm_eval_runner.py: the logger fed the pre-step ground truth by a wrapper of the plant's 1 s
``tools.coopfree.backend.tick``) and ``vehicle_log_report`` (ops/nm_finaleval.py: registered per-vehicle delay =
sum max(0, 1 - v/SPEED_LIMIT) over the logged seconds [300, 1200) + depart delay if first logged after t0).
Only the per-vehicle pairing below the port is new:

Group-label convention (MAIN, registered 04:2x): a vehicle's group = its ORIGIN SUMO lane and class IN THE A ARM on
the same world, paired by vehicle id; origin = the WM lane at the vehicle's first logged second if it is upstream of
the drop (WM 2 = SUMO lane 0 = L0 closing, WM 1 = L1, WM 0 = L2 left; MULTICRIT ``analyze.ORIG``), else 'D' (first
seen past the drop; not a registered fairness group). Vehicles absent from A are counted and reported separately
(never silently dropped). Groups: class:cav, class:hdv, origin:L0, origin:L1, origin:L2 ({n, mean_s}), the
fullpol_fair.summary_from_report shape.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np

VEHICLE_LOG = 'vehicles.jsonl.gz'
WARMUP_S = 300
HORIZON_S = 1200
SPEED_LIMIT_MPS = 31.29                                  # traffic_shepherd.c4.env.SPEED_LIMIT_MPS (team-delay term)
DROP_S_M = 75.44                                         # N1 descriptor lane drop (s)
NOMINAL_LENGTH_M = 5.0
ORIG = {2: 'L0', 1: 'L1', 0: 'L2'}                       # WM lane -> origin SUMO lane label (MULTICRIT analyze.ORIG)
REGISTERED_GROUPS = ('class:cav', 'class:hdv', 'origin:L0', 'origin:L1', 'origin:L2')


# ---------------------------------------------------------------- N6 port (verbatim)

def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class VehicleLog:
    """EVALUATION-ONLY ground truth, read-only: never returned to any policy/controller/observation path."""

    def __init__(self, work):
        import gzip
        self.path = Path(work)/VEHICLE_LOG
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = gzip.open(self.path, 'xt', encoding='utf-8')
        self.index, self.last, self.ticks = {}, None, 0

    def tick(self, env):
        import numpy as np
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
                                                  depart_delay_s=delay, first_t=now))+'\n')
            k.append(n)
            s.append(round(float(f.s[i]), 2))
            v.append(round(float(f.v[i]), 3))
            a.append(round(float(f.a[i]), 3))
            lane.append(int(f.lane_carry[i] if f.junction[i] else f.lane[i]))
        self.stream.write(json.dumps(dict(t=now, k=k, s=s, v=v, a=a, lane=lane))+'\n')
        self.ticks += 1

    def close(self):
        if not self.stream.closed:
            self.stream.close()
        return dict(file=VEHICLE_LOG, sha256=_sha(self.path), ticks=self.ticks, vehicles=len(self.index))


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


def vehicle_log_report(path, *, t0=WARMUP_S, t1=HORIZON_S):
    """Zero physics: field-by-field check + the registered secondary quantities of one ground-truth vehicle log
    (vectorised: per-vehicle delay = sum max(0, 1 - v/SPEED_LIMIT) + depart delay if inserted after t0; same-lane
    consecutive TTC with the nominal 5 m length; stop-and-go v>=5 -> v<1; crossings of the drop s)."""
    import gzip
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
    if any(set(h) != {'veh', 'id', 'cav', 'origin', 'depart_delay_s', 'first_t'} for h in heads.values()):
        problems.append('vehicle header fields differ from the registered set')
    cols = {c: [] for c in ('t', 'k', 's', 'v', 'a', 'lane')}
    for x in ticks:
        n = len(x.get('k', []))
        if any(len(x.get(c, [])) != n for c in ('s', 'v', 'a', 'lane')):
            problems.append(f't={x.get("t")}: column lengths differ')
            break
        cols['t'].append(np.full(n, x.get('t', -1), np.int64))
        for c in ('k', 's', 'v', 'a', 'lane'):
            cols[c].append(np.asarray(x[c], dtype=np.float64 if c in ('s', 'v', 'a') else np.int64)
                           if n else np.zeros(0, np.float64 if c in ('s', 'v', 'a') else np.int64))
    T, K, S, V, A, L = (np.concatenate(cols[c]) if cols[c] else np.zeros(0) for c in ('t', 'k', 's', 'v', 'a', 'lane'))
    K, L, T = K.astype(np.int64), L.astype(np.int64), T.astype(np.int64)
    if K.size and (K.min() < 0 or not set(np.unique(K).tolist()) <= set(heads)):
        problems.append('unknown vehicle index')
    if not (np.all(np.isfinite(S)) and np.all(np.isfinite(V)) and np.all(np.isfinite(A)) and np.all(V >= 0)
            and np.all(L >= 0)):
        problems.append('non-finite or invalid kinematics/lane')
    nveh = (max(heads)+1) if heads else 0
    deficit = np.maximum(0., 1.-V/SPEED_LIMIT_MPS)
    seen = np.bincount(K, minlength=nveh) if K.size else np.zeros(nveh, np.int64)
    delay_v = np.bincount(K, weights=deficit, minlength=nveh) if K.size else np.zeros(nveh)
    for k, h in heads.items():
        if seen[k] and h.get('first_t', -1) > t0 and isinstance(h.get('depart_delay_s'), (int, float)):
            delay_v[k] += float(h['depart_delay_s'])
    present = np.flatnonzero(seen > 0)
    d = np.sort(delay_v[present])
    # per-vehicle accel moments; stop-and-go and drop crossings on consecutive seconds of the same vehicle
    n_ = np.bincount(K, minlength=nveh).astype(float) if K.size else np.zeros(nveh)
    s1 = np.bincount(K, weights=A, minlength=nveh) if K.size else np.zeros(nveh)
    s2 = np.bincount(K, weights=A*A, minlength=nveh) if K.size else np.zeros(nveh)
    ok2 = n_ >= 2
    per_vehicle_var = (s2[ok2]/n_[ok2]-(s1[ok2]/n_[ok2])**2) if ok2.any() else np.zeros(0)
    order = np.lexsort((T, K))
    k_, t_, v_, s_ = K[order], T[order], V[order], S[order]
    same = (k_[1:] == k_[:-1]) & (t_[1:] == t_[:-1]+1)
    stops = int(np.sum(same & (v_[:-1] >= 5.0) & (v_[1:] < 1.0)))
    crossings = len(set(k_[1:][same & (s_[:-1] < DROP_S_M) & (s_[1:] >= DROP_S_M)].tolist()))
    # same-lane consecutive pairs per second (follower = lower s)
    order = np.lexsort((S, L, T))
    t2, l2, s2_, v2 = T[order], L[order], S[order], V[order]
    pair = (t2[1:] == t2[:-1]) & (l2[1:] == l2[:-1])
    gap = s2_[1:]-s2_[:-1]-NOMINAL_LENGTH_M
    closing = v2[:-1]-v2[1:]
    valid = pair & (closing > 0) & (gap > 0)
    ttc = gap[valid]/closing[valid]
    pairs = int(pair.sum())
    if heads and not pairs:
        problems.append('no same-lane vehicle pair (TTC not computable)')
    cav = sum(1 for h in heads.values() if h.get('cav') is True)
    hdv = sum(1 for h in heads.values() if h.get('cav') is False)
    origins = {}
    for h in heads.values():
        origins[h.get('origin')] = origins.get(h.get('origin'), 0)+1
    with_origin = sum(v for o, v in origins.items() if isinstance(o, str) and o)
    with_delay = sum(1 for h in heads.values() if isinstance(h.get('depart_delay_s'), (int, float)))
    if heads and (not cav or not hdv):
        problems.append(f'both fleet classes required (CAV {cav}, HDV {hdv})')
    if heads and with_origin < .95*len(heads):
        problems.append(f'route origin missing for {len(heads)-with_origin}/{len(heads)} vehicles')
    if heads and with_delay < .95*len(heads):
        problems.append(f'depart delay missing for {len(heads)-with_delay}/{len(heads)} vehicles')
    if A.size and float(A.var()) == 0.:
        problems.append('acceleration identically constant')
    gini = float((2*np.arange(1, d.size+1)-d.size-1).dot(d)/(d.size*d.sum())) if d.size and d.sum() > 0 else None
    jain = float(d.sum()**2/(d.size*(d**2).sum())) if d.size and (d**2).sum() > 0 else None
    top = d[int(np.floor(.9*d.size)):] if d.size else d

    def mean_of(keys):
        vals = [delay_v[k] for k in keys if k < nveh and seen[k]]
        return float(np.mean(vals)) if vals else None
    cav_mean = mean_of([k for k, h in heads.items() if h.get('cav') is True])
    hdv_mean = mean_of([k for k, h in heads.items() if h.get('cav') is False])
    by_origin = {str(o): mean_of([k for k, h in heads.items() if h.get('origin') == o]) for o in origins}
    return dict(ok=not problems, problems=problems, ticks=len(ticks), vehicles=len(heads), cav=cav, hdv=hdv,
                delay_by_class=dict(cav_mean_s=cav_mean, hdv_mean_s=hdv_mean,
                                    hdv_cav_ratio=hdv_mean/cav_mean if cav_mean and hdv_mean is not None else None),
                delay_by_origin_s=by_origin,
                accel_var_per_vehicle_mean=float(per_vehicle_var.mean()) if per_vehicle_var.size else None,
                origins={str(k): v for k, v in sorted(origins.items(), key=lambda kv: str(kv[0]))},
                origin_coverage=round(with_origin/len(heads), 4) if heads else None,
                depart_delay_coverage=round(with_delay/len(heads), 4) if heads else None,
                delay=dict(mean_s=float(d.mean()) if d.size else None, max_s=float(d.max()) if d.size else None,
                           gini=gini, jain=jain, worst10_mean_s=float(top.mean()) if top.size else None),
                ttc=dict(pairs=pairs, min_s=float(ttc.min()) if ttc.size else None, below_3_vs=int(np.sum(ttc < 3.0))),
                accel=dict(n=int(A.size), var=float(A.var()) if A.size else None, hard_brake_vs=int(np.sum(A < -3.0)),
                           emergency_vs=int(np.sum(A < -4.5))),
                stop_and_go=stops, crossings_drop=crossings)


# ---------------------------------------------------------------- per-vehicle table and A-labelled groups (new)

def per_vehicle(path, *, t0=WARMUP_S, t1=HORIZON_S):
    """{vehicle id: [cav, origin, delay_s]} with EXACTLY vehicle_log_report's per-vehicle delay (deficit sum over
    logged seconds + depart delay if first logged after t0); origin from the first logged row (see module doc)."""
    import gzip
    heads, first, delay = {}, {}, {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            x = json.loads(line)
            if 'veh' in x:
                heads[x['veh']] = x
                continue
            if not t0 <= int(x['t']) < t1:
                continue
            for k, s, v, lane in zip(x['k'], x['s'], x['v'], x['lane']):
                delay[k] = delay.get(k, 0.)+max(0., 1.-float(v)/SPEED_LIMIT_MPS)
                if k not in first:
                    first[k] = (float(s), int(lane))
    out = {}
    for k, h in heads.items():
        if k not in delay:
            continue                       # header without a logged row in [t0, t1)
        d = delay[k]
        if h.get('first_t', -1) > t0 and isinstance(h.get('depart_delay_s'), (int, float)):
            d += float(h['depart_delay_s'])
        s, lane = first[k]
        out[str(h['id'])] = [bool(h['cav']), ORIG.get(lane) if s < DROP_S_M else 'D', float(d)]
    return out


def labels(table):
    """A-arm labels {vehicle id: (cav, origin)} from A's own per-vehicle table."""
    return {vid: (row[0], row[1]) for vid, row in table.items()}


def groups(table, a_labels):
    """(vehicle_groups, unpaired) for one episode: every vehicle grouped by its A label (same world, by id);
    vehicles absent from A are counted separately by their own class. Only the registered groups are returned."""
    acc, unpaired = {}, dict(n=0, cav=0, hdv=0, delay_s=0.)
    for vid, (cav, _, d) in table.items():
        label = a_labels.get(vid)
        if label is None:
            unpaired['n'] += 1
            unpaired['cav' if cav else 'hdv'] += 1
            unpaired['delay_s'] += float(d)
            continue
        a_cav, origin = label
        for g in ('class:cav' if a_cav else 'class:hdv', f'origin:{origin}' if origin in ORIG.values() else None):
            if g is not None:
                n, total = acc.get(g, (0, 0.))
                acc[g] = (n+1, total+float(d))
    return {g: dict(n=n, mean_s=total/n) for g, (n, total) in sorted(acc.items()) if n}, unpaired


def a_groups_path(run_root, update):
    return Path(run_root)/'fair'/f'A_GROUPS_u{int(update):03}.json'


def a_labels_path(run_root, update):
    return Path(run_root)/'fair'/f'A_LABELS_u{int(update):03}.json'


def _table(episode):
    table = episode.get('vehicle_table')
    log = episode.get('vehicle_log') or {}
    if not isinstance(table, dict) or not table or log.get('ticks') != HORIZON_S-WARMUP_S:
        raise ValueError(f'fairness logger: no complete per-vehicle log for {episode.get("id")} (fail closed)')
    return table


def write_a(run_root, update, episodes):
    """Frozen A arm: per-world group summaries (A's own labels) + the A labels, written BEFORE B/R train."""
    from .accel_storage import write
    summaries, label_map = {}, {}
    for e in episodes:
        if e.get('practice'):
            continue
        seed = str(e['world']['seed'])
        table = _table(e)
        lab = labels(table)
        summaries[seed], _ = groups(table, lab)
        label_map[seed] = {vid: list(v) for vid, v in lab.items()}
    if not summaries:
        raise ValueError('fairness logger: A produced no ordinary episode summaries (fail closed)')
    write(a_labels_path(run_root, update), label_map)
    write(a_groups_path(run_root, update), summaries)
    return dict(worlds=sorted(summaries, key=int), path=str(a_groups_path(run_root, update)))


def attach(run_root, update, episodes):
    """B/R: vehicle_groups (+ vehicle_groups_unpaired) on every ordinary episode from A's labels (fail closed)."""
    path = a_labels_path(run_root, update)
    if not path.exists():
        raise ValueError(f'fairness logger: A labels {path} missing before B/R train (fail closed)')
    label_map = json.loads(path.read_text(encoding='utf-8'))
    out = dict(episodes=0, unpaired=0, practice_skipped=0)
    for e in episodes:
        if e.get('practice'):
            out['practice_skipped'] += 1
            continue
        seed = str(e['world']['seed'])
        if seed not in label_map:
            raise ValueError(f'fairness logger: no A labels for world {seed} (fail closed)')
        lab = {vid: tuple(v) for vid, v in label_map[seed].items()}
        e['vehicle_groups'], e['vehicle_groups_unpaired'] = groups(_table(e), lab)
        out['episodes'] += 1
        out['unpaired'] += e['vehicle_groups_unpaired']['n']
    return out


def enabled(cfg):
    from .fullpol_fair import settings
    return bool((settings(cfg) or {}).get('reward'))


def update_step(cfg, arm, update, episodes):
    """launch_training hook (fair reward ON only): A writes its summaries/labels, B/R attach groups."""
    if not enabled(cfg):
        return None
    if arm == 'A':
        return dict(arm='A', **write_a(cfg['run_root'], update, episodes))
    return dict(arm=arm, **attach(cfg['run_root'], update, episodes))
