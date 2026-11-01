"""Policy-centred, discovery-pure joint acceleration-plan search (MPPI/CEM).

Owner alignment 2026-11-01 22:39/22:43 (arm B): groups of 2..6 CAVs anywhere on
the road; the world model (WM) first scores the CURRENT POLICY's own continuation
and plain MAPPO, then searches JOINT changes around the policy's own imagined
acceleration trajectory using GENERIC correlated noise (smooth in time,
correlated across members). No hand-designed cooperative pattern is used as a
seed: the inherited O1-O4 families survive only as post-hoc labels
(``posthoc_label``). Search preferences are learned from SUMO-positive plans
only (``CovarianceModel``, ``PlanLibrary``); neither uses a hand-set threshold.

The WM is a black box::

    evaluate(record, actions, K, H) -> {action_id: [cost_future_0, ..., cost_future_K-1]}

(lower is better; each entry may also be a dict carrying ``team_delay``; futures
are paired/common across all actions of one call). Comparators ``policy``
(policy continues) and ``MAPPO`` are scored in every call on the same futures.

Everything is deterministic given ``seed`` and JSON-serialisable.

Registered rules (all exposed in ``REGISTERED``):

* Group: 1-D single-linkage chaining of CAVs on the shared road coordinate
  ``s_m`` (all lanes, ramp included; lane is ignored for chaining) with gap
  <= ``chain_m``. Components of size < 2 are dropped. A component larger than
  ``max_size`` is reduced by choosing, among its CONTIGUOUS windows of
  ``max_size`` members in road order (contiguity keeps the chosen group itself
  chained), the window spanning (1) the most distinct lanes, then (2) the
  smallest spread max(s)-min(s), then (3) the most downstream window. The chosen
  members are removed, the remainder is re-chained and the rule repeats, so
  groups are disjoint. Members are ordered leader first: (-s_m, lane, vid).
* Durations: {30,60,90,120,150,180,240,300} s up to ``max_duration_s``
  (default 300 s, owner decision 2026-11-01).
* Knots: per member n(D) = min(10, max(min(8, 1 + D//10), 1 + ceil(D/30)))
  at linspace(0, D, n(D)); spacing scales with D (D=60 -> 7 knots/10 s;
  D=180 -> 8/25.7 s; D=240 -> 9/30 s; D=300 -> 10/33.3 s).
* Noise: sampled in a canonical space of n(D_max) knots/member (10 for 300 s)
  spanning [0, D_max]
  seconds from the decision (``frame='absolute'``, default; the plan of
  duration D uses the curve on [0, D) and hands back to the policy after, so
  all durations share one time-anchored perturbation and a learned direction
  means e.g. "member 2 brakes for the first 40 s"; ``frame='normalised'``
  stretches the canonical knots over [0, D] instead and searched worse on the CPU
  benchmark), z ~ N(mu, Sigma) with Sigma_0 = sigma^2 * kron(I_members, K_time), K_time a
  squared-exponential kernel with length scale one canonical knot spacing and
  sigma = 0.5 m/s^2, the policy's own registered exploration std (design item
  11; capped by the uniform-over-bounds std (hi-lo)/sqrt(12)). On the CPU
  benchmark the inherited absolute-knot std (hi-lo)/sqrt(12)=1.3 failed to
  improve 6-member groups in 3 iterations; 0.5 did. Member correlation
  starts at identity (isotropic) and is learned only from SUMO-positive plans.
* Scaling: dim d = members * n(D_max); population N = max(32, 2*d) rounded up
  to a multiple of 4 (WM batch width); elites E = ceil(N/4); finalists
  F = min(2*top_k, pool). MPPI temperature is solved so the Kish effective
  sample size equals E (no hand-set temperature); CEM uses the top E.
* Iteration update (mean mu' = sum w z): covariance by the minimal proper
  inverse-Wishart prior (prior weight ONE pseudo-sample),
  Sigma' = (Sigma + n_eff*S_w)/(1 + n_eff); duration probabilities by the same
  one-pseudo-sample rule, p' = (p + n_eff*w_D)/(1 + n_eff), so no duration or
  direction collapses to exactly zero within a search (no tuned floor).
* Learned covariance (cross-moment): Sigma_r = (d*Sigma_0 + n_eff*M_w)/(d + n_eff)
  with M_w the weight-normalised second moment of SUMO-positive perturbations
  (relative to the policy trajectory) and n_eff their Kish count (scale-free).
* Horizons: H_screen = min(600, max(120, D_max)) (300 s by default); final
  re-score H_final = min(600, max(300, D + 120)) with D the longest finalist
  duration, so the release after a long hold is scored (``final_horizon``).
  All finalists and both comparators share that H_final and its futures.
* Top-k: candidates with mean predicted saving vs MAPPO > 0 on the K_final
  paired futures, best first, at most top_k. Thompson: bootstrap the K_final
  future indices (seeded); P(best) from ``thompson_bootstraps`` resamples, the
  executed choice is the winner of one further independent resample.
"""
import hashlib
import json
import math
import numpy as np

REGISTERED = dict(
    chain_m=100., min_size=2, max_size=6, max_knots=8, knot_base_s=10,
    long_spacing_s=30, long_max_knots=10,
    durations_s=(30, 60, 90, 120, 150, 180, 240, 300), max_duration_s=300,
    iterations=3, pop_min=32, pop_per_dim=2, pop_multiple=4, elite_frac=.25,
    top_k=4, finalists_per_k=2, screen_K=4, screen_H_min=120, final_K=8,
    final_H_min=300, release_s=120, final_H_max=600,
    thompson_bootstraps=4000, library_seeds=4, method='mppi', length_knots=1., frame='absolute', sigma=.5)
ORIGINS = ('centre', 'mappo', 'library', 'sampled')
PLAN_TYPE = 'MPPI'


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                     default=str).encode()).hexdigest()


def _f(x):
    """Plain JSON floats (rounded for stable hashing across platforms)."""
    return [[round(float(v), 9) for v in row] for row in np.asarray(x, float)]


# --------------------------------------------------------------------------- groups

def _chain(cavs, chain_m):
    ordered = sorted(cavs, key=lambda r: (r['s_m'], r['lane'], r['vid']))
    out, cur = [], []
    for r in ordered:
        if cur and r['s_m'] - cur[-1]['s_m'] > chain_m:
            out.append(cur); cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    return out


def _best_window(comp, size):
    def key(i):
        w = comp[i:i+size]
        return (-len({r['lane'] for r in w}), w[-1]['s_m']-w[0]['s_m'], -i)
    return min(range(len(comp)-size+1), key=key)


def group_components(rows, chain_m=REGISTERED['chain_m'], max_size=REGISTERED['max_size'],
                     min_size=REGISTERED['min_size'], exclude=()):
    """Disjoint CAV groups anywhere on the road (all lanes, ramp included).

    rows: observable rows with ``vid``, ``s_m`` (shared road coordinate),
    ``v_mps``, ``lane`` and ``is_cav``. ``exclude``: vids already committed to a
    plan. Returns a list of JSON dicts, most downstream group first, members
    leader first. See module docstring for the >max_size rule.
    """
    cavs = [dict(r) for r in rows if r['is_cav'] and r['vid'] not in set(exclude)]
    pending, groups = _chain(cavs, chain_m), []
    while pending:
        comp = pending.pop(0)
        if len(comp) < min_size:
            continue
        if len(comp) <= max_size:
            groups.append(comp); continue
        i = _best_window(comp, max_size)
        groups.append(comp[i:i+max_size])
        pending = _chain(comp[:i]+comp[i+max_size:], chain_m)+pending
    out = []
    for g in groups:
        g = sorted(g, key=lambda r: (-r['s_m'], r['lane'], r['vid']))
        members = [r['vid'] for r in g]
        out.append(dict(id='G'+_digest(members)[:12], members=members, size=len(g),
                        lanes=sorted({int(r['lane']) for r in g}),
                        spread_m=float(g[0]['s_m']-g[-1]['s_m']), leader_s_m=float(g[0]['s_m']),
                        rows=[{k: r[k] for k in ('vid', 's_m', 'v_mps', 'lane', 'is_cav')} for r in g],
                        rule=dict(chain_m=chain_m, max_size=max_size, min_size=min_size)))
    return sorted(out, key=lambda g: (-g['leader_s_m'], g['id']))


def describe(group, regime):
    """Observable descriptor: regime, size, lanes, positions/speeds relative to leader/mean."""
    rows = group['rows']; s0 = rows[0]['s_m']; vbar = float(np.mean([r['v_mps'] for r in rows]))
    return dict(regime=regime, size=len(rows), lanes=[int(r['lane']) for r in rows],
                rel_s=[float(r['s_m']-s0) for r in rows], rel_v=[float(r['v_mps']-vbar) for r in rows],
                leader_s=float(s0), mean_v=vbar)


def _descriptor_vector(d):
    return np.array(d['rel_s']+d['rel_v']+[float(x) for x in d['lanes']]+[d['leader_s'], d['mean_v']])


# --------------------------------------------------------------------------- plan space

def knot_count(duration_s, max_knots=REGISTERED['max_knots'], base_s=REGISTERED['knot_base_s'],
               long_spacing_s=REGISTERED['long_spacing_s'], long_max_knots=REGISTERED['long_max_knots']):
    """n(D) = min(10, max(min(8, 1 + D//10), 1 + ceil(D/30))).

    10 s spacing up to 8 knots (D <= 180 s unchanged); longer holds keep the
    spacing <= 30 s, capped at 10 knots (D=240 -> 9, D>=270 -> 10).
    """
    d = int(duration_s)
    return int(min(long_max_knots, max(min(max_knots, 1+d//base_s), 1+math.ceil(d/long_spacing_s))))


def final_horizon(duration_s, cfg=REGISTERED):
    """Final re-score horizon H = min(600, max(300, D + 120)): the release after a hold is scored."""
    return int(min(cfg['final_H_max'], max(cfg['final_H_min'], int(duration_s)+cfg['release_s'])))


def knot_times(duration_s, **kw):
    return np.linspace(0., float(duration_s), knot_count(duration_s, **kw))


def allowed_durations(durations=REGISTERED['durations_s'], max_duration_s=REGISTERED['max_duration_s']):
    out = sorted({int(d) for d in durations if 0 < int(d) <= max_duration_s})
    if not out:
        raise ValueError('no allowed plan duration under the evidence-based maximum')
    return out


def scaling(members, n_knots, top_k=REGISTERED['top_k'], cfg=REGISTERED):
    """Registered population/elite scaling with plan dimension members x knots."""
    d = int(members)*int(n_knots)
    n = max(cfg['pop_min'], cfg['pop_per_dim']*d)
    n = int(math.ceil(n/cfg['pop_multiple'])*cfg['pop_multiple'])
    return dict(dim=d, population=n, elites=int(math.ceil(cfg['elite_frac']*n)),
                finalists=int(cfg['finalists_per_k']*top_k))


def _interp_matrix(src, dst):
    """Rows: dst points as linear combinations of src knots (clamped at ends)."""
    A = np.zeros((len(dst), len(src)))
    for i, t in enumerate(dst):
        e = np.zeros(len(src))
        for j in range(len(src)):
            e[:] = 0; e[j] = 1; A[i, j] = np.interp(t, src, e)
    return A


def default_sigma(bounds):
    """Registered perturbation std: the policy's own exploration sigma (0.5 m/s^2,
    COOPMWM_DESIGN item 11), never more than the uniform-over-bounds std."""
    return float(min(REGISTERED['sigma'], (bounds[1]-bounds[0])/math.sqrt(12.)))


def canonical_knots(max_duration_s=REGISTERED['max_duration_s']):
    """Canonical knots per member = n(D_max) (10 for the registered 300 s maximum)."""
    return knot_count(max_duration_s)


def time_kernel(n=None, length_knots=REGISTERED['length_knots']):
    i = np.arange(canonical_knots() if n is None else n, dtype=float)
    return np.exp(-.5*np.square((i[:, None]-i[None, :])/length_knots))


def isotropic_covariance(members, sigma, n=None, length_knots=REGISTERED['length_knots']):
    return sigma**2*np.kron(np.eye(members), time_kernel(n, length_knots))


def _sampler(cov):
    w, V = np.linalg.eigh((cov+cov.T)/2)
    return V*np.sqrt(np.clip(w, 0., None))


def _as_traj(traj, members):
    if isinstance(traj, dict):
        traj = [traj[v] for v in members]
    a = np.asarray(traj, float)
    if a.ndim != 2 or a.shape[0] != len(members) or not np.isfinite(a).all():
        raise ValueError('trajectory must be finite members x seconds')
    return a


def _per_second(z, duration, span):
    """Canonical (m x n_canon knots over [0, span]) -> plan knots and per-second (m x D)."""
    kt = knot_times(duration); canon = np.linspace(0., float(span), z.shape[1])
    knots = z@_interp_matrix(canon, kt).T
    return knots, knots@_interp_matrix(kt, np.arange(duration, dtype=float)).T, kt


def _to_canonical(per_second, duration, span, n):
    """Effective per-second perturbation -> canonical knots; zero after the plan ends."""
    t = np.arange(duration, dtype=float); canon = np.linspace(0., float(span), n)
    return np.stack([np.interp(canon, t, row, right=0. if span > duration else None) for row in per_second])


def build_plan(group, policy_traj, delta, duration, bounds, origin, regime, tag='',
               absolute=None, span_s=None, n_canon=None):
    """Plan = policy trajectory + smooth perturbation, clipped to the plant bounds.

    ``delta`` is a canonical m x n_canon perturbation (or None when ``absolute`` gives
    an exact per-second m x D profile, used for the MAPPO seed). ``span_s``: the
    canonical knots cover [0, span_s] seconds from plan start (absolute frame,
    default REGISTERED max duration); ``span_s=duration`` gives a duration-
    normalised frame.
    """
    span = int(REGISTERED['max_duration_s'] if span_s is None else span_s)
    n_canon = canonical_knots() if n_canon is None else int(n_canon)
    if delta is not None and np.shape(delta)[1] != n_canon:
        raise ValueError('perturbation does not match the canonical knot count')
    if span < duration:
        raise ValueError('canonical span shorter than plan duration')
    members = group['members']; P = _as_traj(policy_traj, members)
    if P.shape[1] < duration:
        raise ValueError('policy trajectory shorter than plan duration')
    if absolute is None:
        knots, dsec, kt = _per_second(np.asarray(delta, float), duration, span)
        raw = P[:, :duration]+dsec
    else:
        raw = _as_traj(absolute, members)[:, :duration]; kt = knot_times(duration)
        knots = (raw-P[:, :duration])@_interp_matrix(np.arange(duration, dtype=float), kt).T
    profile = np.clip(raw, bounds[0], bounds[1])
    effective = profile-P[:, :duration]
    p = dict(members=list(members), duration_s=int(duration), profile=_f(profile),
             knot_times_s=[round(float(t), 9) for t in kt], perturbation_knots=_f(knots),
             perturbation_profile=_f(effective),
             perturbation_canonical=_f(_to_canonical(effective, duration, span, n_canon)), canonical_span_s=span,
             origin=origin, regime=regime, family=PLAN_TYPE, group_id=group.get('id'))
    p['id'] = tag+'M'+_digest(p)[:16]
    return p


def acceleration_at(plan, elapsed):
    """Absolute per-member acceleration of a profile plan; {} outside [0, D)."""
    if not 0 <= elapsed < plan['duration_s']:
        return {}
    t = int(math.floor(elapsed))
    return {v: float(row[t]) for v, row in zip(plan['members'], plan['profile'])}


def validate_plan(plan, bounds, durations=None):
    m = plan['members']
    if not REGISTERED['min_size'] <= len(m) <= REGISTERED['max_size'] or len(set(m)) != len(m):
        raise ValueError('group of 2..6 distinct CAVs required')
    if durations is not None and plan['duration_s'] not in durations:
        raise ValueError('unregistered plan duration')
    a = np.asarray(plan['profile'], float)
    if a.shape != (len(m), plan['duration_s']) or not np.isfinite(a).all():
        raise ValueError('invalid acceleration profile')
    if (a < bounds[0]-1e-9).any() or (a > bounds[1]+1e-9).any():
        raise ValueError('plan outside plant acceleration limits')
    if len(plan['knot_times_s']) > REGISTERED['long_max_knots']:
        raise ValueError('more than the registered knots per member')
    if plan['origin'] not in ORIGINS:
        raise ValueError('unregistered plan origin (hand-designed families are labels only)')
    return plan


def comparators(record, window_s=300):
    """'policy continues' and plain 'MAPPO' in the inherited baseline_actions format."""
    return [dict(id='policy', window=window_s, residuals=dict(record.get('policy_residuals', {}))),
            dict(id='MAPPO', window=0, residuals={})]


# --------------------------------------------------------------------------- learned preferences

class CovarianceModel:
    """Per-(regime, group size) joint perturbation covariance in canonical space.

    Starts isotropic-smooth; ``update`` adds SUMO-positive perturbations (relative
    to the policy trajectory, canonical members x n_canon). Prior weight = dimension d
    pseudo-samples; the data weight is the Kish effective count of the update
    weights, so the rule is invariant to the reward unit. No thresholds.
    """

    def __init__(self, sigma, n_canon=None, length_knots=REGISTERED['length_knots']):
        self.sigma, self.length = float(sigma), float(length_knots)
        self.n = canonical_knots() if n_canon is None else int(n_canon)
        self.stats = {}

    def prior(self, members):
        return isotropic_covariance(members, self.sigma, self.n, self.length)

    def update(self, regime, perturbation, weight):
        z = np.asarray(perturbation, float)
        if z.ndim != 2 or z.shape[1] != self.n or not np.isfinite(z).all():
            raise ValueError('canonical members x knots perturbation required')
        if not weight > 0:
            raise ValueError('only SUMO-positive plans update the search covariance')
        key = f'{regime}|{z.shape[0]}'; v = z.reshape(-1)
        s = self.stats.setdefault(key, dict(M=np.zeros((v.size, v.size)), W=0., W2=0., n=0))
        s['M'] = s['M']+weight*np.outer(v, v); s['W'] += weight; s['W2'] += weight**2; s['n'] += 1

    def get(self, regime, members):
        prior = self.prior(members); s = self.stats.get(f'{regime}|{members}')
        if not s:
            return prior
        d = prior.shape[0]; neff = s['W']**2/s['W2']
        return (d*prior+neff*s['M']/s['W'])/(d+neff)

    def favoured_directions(self, regime, members, k=3):
        """Top-k eigen-directions of the learned covariance relative to the prior."""
        P = self.prior(members); C = self.get(regime, members)
        L = np.linalg.cholesky(P+1e-12*np.eye(len(P))); Li = np.linalg.inv(L)
        w, V = np.linalg.eigh(Li@C@Li.T)
        idx = np.argsort(-w)[:k]
        return [dict(variance_ratio=float(w[i]), direction=_f((L@V[:, i]).reshape(members, self.n)))
                for i in idx]

    def to_json(self):
        return dict(sigma=self.sigma, n_canon=self.n, length_knots=self.length,
                    stats={k: dict(M=_f(s['M']), W=s['W'], W2=s['W2'], n=s['n']) for k, s in self.stats.items()})

    @classmethod
    def from_json(cls, data):
        c = cls(data['sigma'], data['n_canon'], data['length_knots'])
        c.stats = {k: dict(s, M=np.asarray(s['M'], float)) for k, s in data['stats'].items()}
        return c


class PlanLibrary:
    """Self-discovered, SUMO-verified plans keyed by (regime, group size).

    ``nearest`` ranks entries by Euclidean distance between observable
    descriptors standardised by the per-feature std of that key's entries
    (std 0 or a single entry -> unit scale). No distance cutoff: the n nearest
    are returned (deterministic ties by entry id).
    """

    def __init__(self):
        self.entries = []

    def add(self, descriptor, plan, advantage):
        if not advantage > 0:
            raise ValueError('library holds SUMO-positive plans only')
        e = dict(descriptor=descriptor, duration_s=int(plan['duration_s']),
                 perturbation_canonical=plan['perturbation_canonical'], advantage=float(advantage),
                 plan_id=plan['id'], canonical_span_s=plan.get('canonical_span_s'))
        e['id'] = 'L'+_digest(e)[:12]
        self.entries.append(e)
        return e['id']

    def nearest(self, descriptor, n):
        pool = [e for e in self.entries if e['descriptor']['regime'] == descriptor['regime']
                and e['descriptor']['size'] == descriptor['size']]
        if not pool or n <= 0:
            return []
        X = np.stack([_descriptor_vector(e['descriptor']) for e in pool])
        sd = X.std(0); sd[sd == 0] = 1.
        dist = np.sqrt((((X-_descriptor_vector(descriptor))/sd)**2).sum(1))
        order = sorted(range(len(pool)), key=lambda i: (float(dist[i]), pool[i]['id']))[:n]
        return [dict(pool[i], distance=float(dist[i])) for i in order]

    def to_json(self):
        return dict(entries=self.entries)

    @classmethod
    def from_json(cls, data):
        lib = cls(); lib.entries = list(data['entries']); return lib


def learn(plan, advantage, descriptor, covariance=None, library=None):
    """Record a SUMO-positive executed plan into the covariance model and library."""
    if not advantage > 0:
        return dict(learned=False, reason='non-positive SUMO advantage')
    if covariance is not None:
        covariance.update(plan['regime'], plan['perturbation_canonical'], advantage)
    entry = library.add(descriptor, plan, advantage) if library is not None else None
    return dict(learned=True, library_entry=entry)


# --------------------------------------------------------------------------- scoring

def _costs(scores, ids, K):
    out = {}
    for a in ids:
        v = np.array([x['team_delay'] if isinstance(x, dict) else x for x in scores[a]], float)
        if v.shape != (K,) or not np.isfinite(v).all():
            raise ValueError(f'complete finite K{K} score required for {a}')
        out[a] = v
    return out


def _ess_weights(scores, target):
    s = np.asarray(scores, float); s = s-s.min(); n = len(s)
    if target >= n or s.max() == 0:
        return np.full(n, 1./n), float(n), None
    def w(lam):
        x = np.exp(-s/lam); return x/x.sum()
    def ess(lam):
        x = w(lam); return 1./np.square(x).sum()
    lo, hi = math.log(s.max()*1e-9), math.log(s.max()*1e9)
    if ess(math.exp(lo)) >= target:
        return w(math.exp(lo)), ess(math.exp(lo)), math.exp(lo)
    for _ in range(200):
        mid = (lo+hi)/2
        lo, hi = (mid, hi) if ess(math.exp(mid)) < target else (lo, mid)
    lam = math.exp(hi)
    return w(lam), ess(lam), lam


def thompson_topk(plans, costs, top_k, rng, bootstraps=REGISTERED['thompson_bootstraps']):
    """Top-k plans with mean saving vs MAPPO > 0, P(best) and one Thompson pick.

    costs: {id: K paired future costs} including 'policy' and 'MAPPO'.
    """
    K = len(costs['MAPPO'])
    rows = []
    for p in plans:
        c = costs[p['id']]
        rows.append(dict(id=p['id'], saving_vs_mappo=float((costs['MAPPO']-c).mean()),
                         saving_vs_policy=float((costs['policy']-c).mean()), mean_cost=float(c.mean())))
    eligible = sorted([r for r in rows if r['saving_vs_mappo'] > 0], key=lambda r: (r['mean_cost'], r['id']))[:top_k]
    if not eligible:
        return dict(accepted=False, reason='no candidate with positive predicted saving vs MAPPO',
                    candidates=sorted(rows, key=lambda r: (r['mean_cost'], r['id'])))
    order = sorted(r['id'] for r in eligible)
    C = np.stack([costs[i] for i in order])
    idx = rng.integers(0, K, (bootstraps, K))
    winners = np.argmin(C[:, idx].mean(-1), axis=0)
    counts = np.bincount(winners, minlength=len(order))
    prob = {i: float(c/bootstraps) for i, c in zip(order, counts)}
    pick = rng.integers(0, K, K)
    sampled = {a: float(v[pick].mean()) for a, v in costs.items()}
    chosen = min(order, key=lambda i: (sampled[i], i))
    for r in eligible:
        r['p_best'] = prob[r['id']]
    return dict(accepted=True, chosen=chosen, candidates=eligible, probabilities=prob,
                bootstrap_indices=pick.tolist(), bootstraps=bootstraps,
                sampled_saving_vs_mappo=sampled['MAPPO']-sampled[chosen],
                sampled_saving_vs_policy=sampled['policy']-sampled[chosen])


# --------------------------------------------------------------------------- search

def _seed_from(record, group):
    return int(_digest(dict(members=group['members'], rows=group['rows'],
                            time_s=record.get('time_s'), state=record.get('state_id')))[:15], 16)


def _distribution(cfg, members, durations, covariance, regime, sigma):
    nc = canonical_knots(durations[-1])
    if covariance is not None and covariance.n != nc:
        raise ValueError('covariance model canonical knots differ from n(D_max)')
    cov = covariance.get(regime, members) if covariance is not None else isotropic_covariance(
        members, sigma, nc, cfg['length_knots'])
    return np.zeros(members*nc), cov, np.full(len(durations), 1./len(durations))


def _regrid(z, span_old, span_new, n_new):
    """Library perturbation re-expressed on the current canonical grid (time-anchored)."""
    z = np.asarray(z, float); old = np.linspace(0., float(span_old), z.shape[1])
    new = np.linspace(0., float(span_new), n_new)
    return np.stack([np.interp(new, old, row, right=0.) for row in z])


def _span(cfg, durations, D):
    if cfg['frame'] not in ('absolute', 'normalised'):
        raise ValueError('unregistered canonical time frame')
    return durations[-1] if cfg['frame'] == 'absolute' else D


def sample_unscored(group, policy_traj, bounds, regime, seed, covariance=None, config=None, duration_s=None):
    """Arm R: one generic correlated perturbation of the policy, NOT scored by the WM."""
    cfg = dict(REGISTERED, **(config or {})); durations = allowed_durations(cfg['durations_s'], cfg['max_duration_s'])
    m = len(group['members']); rng = np.random.default_rng(seed)
    mu, cov, pd = _distribution(cfg, m, durations, covariance, regime, default_sigma(bounds))
    z = mu+_sampler(cov)@rng.standard_normal(mu.size)
    D = int(durations[rng.choice(len(durations), p=pd)])
    if duration_s is not None:
        if duration_s not in durations:raise ValueError('unregistered matched duration')
        D=int(duration_s)
    nc = canonical_knots(durations[-1])
    return build_plan(group, policy_traj, z.reshape(m, nc), D, bounds, 'sampled', regime, 'R',
                      span_s=_span(cfg, durations, D), n_canon=nc)


def search(record, group, policy_traj, bounds, evaluate, regime, mappo_traj=None,
           covariance=None, library=None, seed=None, config=None):
    """Policy-centred MPPI/CEM joint-plan search with WM screening and K_final rescoring.

    record: request passed unchanged to ``evaluate`` (needs ``policy_residuals``
    for the 'policy continues' comparator). group: one ``group_components`` item.
    policy_traj / mappo_traj: imagined absolute accelerations, members (group
    order) x seconds (array or {vid: list}), length >= max duration.
    Returns a JSON dict: accepted, plan (Thompson choice), top-k candidates with
    savings vs MAPPO / policy and P(best), comparator means, search diagnostics.
    """
    cfg = dict(REGISTERED, **(config or {}))
    durations = allowed_durations(cfg['durations_s'], cfg['max_duration_s'])
    m = len(group['members'])
    if not cfg['min_size'] <= m <= cfg['max_size']:
        raise ValueError('group of 2..6 CAVs required')
    P = _as_traj(policy_traj, group['members'])
    if P.shape[1] < durations[-1]:
        raise ValueError('policy trajectory shorter than the maximum plan duration')
    seed = _seed_from(record, group) if seed is None else int(seed)
    search_rng, choice_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    nc = canonical_knots(durations[-1])
    sc = scaling(m, nc, cfg['top_k'], cfg)
    H_screen = int(min(cfg['final_H_max'], max(cfg['screen_H_min'], durations[-1])))
    sigma = default_sigma(bounds)
    mu, cov, pd = _distribution(cfg, m, durations, covariance, regime, sigma)
    desc = describe(group, regime)
    base = comparators(record, cfg['final_H_max'])
    seeds = [(np.zeros((m, nc)), durations[-1], 'centre', None)]
    if mappo_traj is not None:
        seeds.append((None, durations[-1], 'mappo', _as_traj(mappo_traj, group['members'])))
    lib_used = library.nearest(desc, cfg['library_seeds']) if library is not None else []
    for e in lib_used:
        D = min(durations, key=lambda d: (abs(d-e['duration_s']), d))
        old_span = e.get('canonical_span_s') or durations[-1]
        z = _regrid(e['perturbation_canonical'], old_span, _span(cfg, durations, D), nc)
        seeds.append((z, D, 'library', None))
    pool, zs, log, futures = {}, {}, [], 0
    for it in range(cfg['iterations']):
        cands = []
        if it == 0:
            for j, (z, D, origin, absolute) in enumerate(seeds):
                p = build_plan(group, P, z, D, bounds, origin, regime, f'i{it}s{j}-', absolute,
                               _span(cfg, durations, D), nc)
                cands.append(p); zs[p['id']] = np.asarray(p['perturbation_canonical'], float).reshape(-1)
        L = _sampler(cov)
        while len(cands) < sc['population']:
            z = mu+L@search_rng.standard_normal(mu.size)
            D = int(durations[search_rng.choice(len(durations), p=pd)])
            p = build_plan(group, P, z.reshape(m, nc), D, bounds, 'sampled', regime,
                           f'i{it}s{len(cands)}-', span_s=_span(cfg, durations, D), n_canon=nc)
            cands.append(p); zs[p['id']] = z
        scores = evaluate(record, base+cands, cfg['screen_K'], H_screen)
        futures += (len(cands)+len(base))*cfg['screen_K']
        c = _costs(scores, ['policy', 'MAPPO']+[p['id'] for p in cands], cfg['screen_K'])
        rel = np.array([float((c[p['id']]-c['policy']).mean()) for p in cands])
        for p, r in zip(cands, rel):
            pool[p['id']] = (float(r), p)
        if it == 0:
            # Open-loop policy trajectory minus closed-loop 'policy continues' on
            # the same futures: a check of the imagined-trajectory extraction.
            centre_gap = float(rel[0])
        if cfg['method'] == 'cem':
            top = np.argsort(rel, kind='stable')[:sc['elites']]
            w = np.zeros(len(cands)); w[top] = 1./len(top); neff, lam = float(len(top)), None
        else:
            w, neff, lam = _ess_weights(rel, sc['elites'])
        Z = np.stack([zs[p['id']] for p in cands])
        mu = w@Z; X = Z-mu; S = (X*w[:, None]).T@X
        cov = (cov+neff*S)/(1.+neff)
        wd = np.array([w[[p['duration_s'] == d for p in cands]].sum() for d in durations])
        pd = (pd+neff*wd/wd.sum())/(1.+neff)
        log.append(dict(iteration=it, candidates=len(cands), best_rel_cost=float(rel.min()),
                        mean_rel_cost=float(rel.mean()), ess=float(neff), temperature=lam,
                        duration_probs=dict(zip(map(str, durations), map(float, pd)))))
    ranked = sorted(pool.values(), key=lambda x: (x[0], x[1]['id']))
    finalists = [p for _, p in ranked[:min(sc['finalists'], len(ranked))]]
    H_final = final_horizon(max(p['duration_s'] for p in finalists), cfg)
    scores = evaluate(record, base+finalists, cfg['final_K'], H_final)
    futures += (len(finalists)+len(base))*cfg['final_K']
    costs = _costs(scores, ['policy', 'MAPPO']+[p['id'] for p in finalists], cfg['final_K'])
    t = thompson_topk(finalists, costs, cfg['top_k'], choice_rng, cfg['thompson_bootstraps'])
    byid = {p['id']: p for p in finalists}
    out = dict(accepted=t['accepted'], regime=regime, group=dict((k, group[k]) for k in ('id', 'members', 'lanes', 'spread_m', 'leader_s_m')),
               descriptor=desc, seed=seed, config={k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
               scaling=sc, durations_s=durations, screen=dict(K=cfg['screen_K'], H=H_screen),
               final=dict(K=cfg['final_K'], H=H_final),
               comparator_means=dict(policy=float(costs['policy'].mean()), MAPPO=float(costs['MAPPO'].mean())),
               iterations=log, centre_open_loop_gap=centre_gap, candidates_searched=len(pool), imagined_futures=futures,
               library_seeds=[e['id'] for e in lib_used], origins_searched=sorted({p['origin'] for _, p in pool.values()}),
               finalists=[dict(id=p['id'], origin=p['origin'], duration_s=p['duration_s'],
                               saving_vs_mappo=float((costs['MAPPO']-costs[p['id']]).mean()),
                               saving_vs_policy=float((costs['policy']-costs[p['id']]).mean())) for p in finalists])
    if not t['accepted']:
        out.update(reason=t['reason'], plan=None, option=None, expected_saving=None)
        return out
    plans = [dict(byid[r['id']], saving_vs_mappo=r['saving_vs_mappo'], saving_vs_policy=r['saving_vs_policy'],
                  p_best=r['p_best']) for r in t['candidates']]
    chosen = next(p for p in plans if p['id'] == t['chosen'])
    out.update(plan=chosen, option=PLAN_TYPE, topk=plans, probabilities=t['probabilities'],
               thompson=dict(bootstrap_indices=t['bootstrap_indices'], bootstraps=t['bootstraps'],
                             sampled_saving_vs_mappo=t['sampled_saving_vs_mappo'],
                             sampled_saving_vs_policy=t['sampled_saving_vs_policy']),
               expected_saving=chosen['saving_vs_mappo'], expected_saving_vs_policy=chosen['saving_vs_policy'],
               known_family=False)
    return out


# --------------------------------------------------------------------------- post-hoc labels

def posthoc_label(plan, prototypes, policy_traj):
    """Nearest known-move label for ANALYSIS only (never a search seed).

    prototypes: e.g. ``plans.family_seeds`` output (members, knots at 0,10,..s,
    family). Compared on the shared members at the prototype's 10 s grid within
    the plan duration, against the policy centre as the null prototype: if no
    family is nearer than the unchanged policy the label is 'new'. Threshold-free.
    """
    members = plan['members']; prof = np.asarray(plan['profile'], float)
    P = _as_traj(policy_traj, members)
    dists = {}
    for s in prototypes:
        if not set(s['members']) <= set(members):
            continue
        rows = [members.index(v) for v in s['members']]
        kn = np.asarray(s['knots'], float)
        grid = [j for j in range(kn.shape[1]) if 10*j < plan['duration_s']]
        if not grid:
            continue
        a = prof[np.ix_(rows, [10*j for j in grid])]
        dists[s.get('id', s['family'])] = (float(np.square(a-kn[:, grid]).mean()),
                                            float(np.square(a-P[np.ix_(rows, [10*j for j in grid])]).mean()),
                                            s['family'])
    if not dists:
        return dict(label='new', reason='no prototype on these members')
    best = min(dists, key=lambda k: (dists[k][0], k))
    d, null, family = dists[best]
    return dict(label=family if d < null else 'new', nearest=best, distance=d, policy_distance=null)
