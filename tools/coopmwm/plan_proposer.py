"""Learned plan proposer: an AlphaZero-style prior for the policy-centred MPPI search.

From the OBSERVABLE situation of a CAV group (member + neighbour rows that the
plan service already receives: ``vid, s_m, v_mps, lane, is_cav[, length_m]``;
regime label; group size 2-6) a small permutation-equivariant set network
predicts a mixture of M Gaussian joint-perturbation proposals in the SAME
canonical knot space as ``mppi_search`` (members x n_canon knots spanning
[0, 300] s from the decision, relative to the policy's own imagined trajectory)
plus a per-component categorical over the registered plan durations.

No hidden SUMO truth, driver parameters, world/seed/demand/share labels, clock
time or history enter the features (``world_id`` is used only to group the
train/validation split). The proposer never scores plans: its samples are
extra iteration-0 seeds; the WM scores everything as before.

Receipt schema (``SCHEMA``; one JSON object per completed search)::

    {"schema": "coopmwm.plan_proposer.receipt/1",
     "search_id": str,                 # record sha / state_id
     "world_id": any,                  # split grouping ONLY (never a feature)
     "arm": "B",                       # only arm-B receipts are training data
     "regime": "merge-dense"|"ordinary"|...,
     "rows": [{vid, s_m, v_mps, lane, is_cav, length_m?}, ...],   # observable rows
     "members": [vid, ...],            # leader first (mppi_search order)
     "canonical": {"n_knots": 10, "span_s": 300},
     "durations_s": [30, ..., 300],
     "initial": {"sigma": 0.5, "length_knots": 1.0,
                 "covariance": [[...]] | null},   # search's own initial N(0, Sigma)
     "candidates": [{"id", "origin", "duration_s",
                     "perturbation_canonical": m x n,     # effective (post-clip)
                     "stage": "screen"|"final",
                     "saving_vs_policy": float,           # WM predicted, > 0 = better
                     "saving_vs_mappo": float|null}],
     "topk": [ids],                    # the search's own top-k (final K8 scores)
     "executed": {"plan_id": str, "sumo_realized_advantage": float} | null}

Adapters: ``receipt_from_search`` (result of ``mppi_search.search`` plus an
optional ``RecordingEvaluator`` that captures the whole screened pool),
``receipt_from_service`` (plan-service request + response/forecast JSON;
tolerates missing pool, missing covariance, rejected searches), and
``attach_sumo`` (``search_learning.update`` entries -> ``executed``).
Missing fields are tolerated; unusable receipts are skipped with a reason.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import mppi_search as ms

SCHEMA = 'coopmwm.plan_proposer.receipt/1'
ROW_FIELDS = ('vid', 's_m', 'v_mps', 'lane', 'is_cav', 'length_m')
MERGE_ZONE = (-100., 75.44)          # same observable RSU zone as coopnative.encode
SENSE_M = 100.                       # same-lane/any-lane neighbour sensing radius
REGIMES = ('merge-dense', 'ordinary')
Z_SCALE = float(ms.REGISTERED['sigma'])   # perturbations are modelled in units of 0.5 m/s^2

REGISTERED = dict(
    components=4, d_model=64, layers=2, heads=4, logstd_min=-1.5, logstd_max=1.5, defensive=.1,
    steps=1500, batch=32, lr=3e-3, weight_decay=1e-4, seed=20261102, val_frac=.25,
    sumo_bonus=1., elites_frac=.25, eval_samples=16, bootstraps=2000,
    min_heldout_worlds=8, min_heldout_searches=40, threads=1, inner_frac=.2, inner_min_worlds=5, eval_every=100,
    min_pool=16, generic_share=.2, novelty_share=.1, novelty_oversample=4,
    entropy_floor=.5*math.log(4), entropy_weight=1.)

FEATURES = 41


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


# --------------------------------------------------------------------------- canonical space

def canonical_times(n=None, span=None):
    n = ms.canonical_knots() if n is None else int(n)
    span = ms.REGISTERED['max_duration_s'] if span is None else span
    return np.linspace(0., float(span), n)


def knot_mask(duration_s, n=None, span=None):
    """Canonical knots carrying information for a plan of duration D.

    ``mppi_search.build_plan`` stores the effective perturbation on the canonical
    grid with zeros after the plan ends (t > D-1) unless D equals the span.
    """
    t = canonical_times(n, span); span = t[-1]
    return np.ones(len(t), bool) if duration_s >= span else t <= duration_s-1


def durations_registered():
    return ms.allowed_durations()


def duration_index(D, durations=None):
    durations = durations_registered() if durations is None else durations
    return int(min(range(len(durations)), key=lambda i: (abs(durations[i]-int(D)), durations[i])))


# --------------------------------------------------------------------------- observable features

def _row(r):
    for k in ('vid', 's_m', 'v_mps', 'lane', 'is_cav'):
        if k not in r:
            raise ValueError(f'observable row missing {k}')
    return {k: r[k] for k in ROW_FIELDS if k in r}


def member_features(rows, members, regime):
    """Per-member observable tokens (members in the given order) -> m x FEATURES.

    Per member: rel position to leader/mean, road position, merge-zone flag,
    speed and rel speed, lane one-hot, nearest vehicle ahead/behind in each of
    the 3 lanes within 100 m (rel s, rel v, is_cav, present), HDV/CAV counts
    within 100 m. Group context appended to every token: regime one-hot,
    size/6, mean speed, merge-zone occupancy. Only ``ROW_FIELDS`` are read.
    """
    rows = [_row(r) for r in rows]
    by = {r['vid']: r for r in rows}
    g = [by[v] for v in members]
    s = np.array([float(r['s_m']) for r in g]); v = np.array([float(r['v_mps']) for r in g])
    lead = float(s.max()); smean, vmean = float(s.mean()), float(v.mean())
    zone = [r for r in rows if MERGE_ZONE[0] <= r['s_m'] <= MERGE_ZONE[1]]
    ctx = [float(regime == REGIMES[0]), float(regime == REGIMES[1]), float(regime not in REGIMES),
           len(members)/6., vmean/30., len(zone)/10.]
    out = []
    for own in g:
        so, vo = float(own['s_m']), float(own['v_mps'])
        lane = int(own['lane'])
        f = [(so-lead)/100., (so-smean)/100., so/1000., float(MERGE_ZONE[0] <= so <= MERGE_ZONE[1]),
             vo/30., (vo-vmean)/10., *[float(min(max(lane, 0), 2) == ln) for ln in range(3)]]
        sensed = [r for r in rows if r['vid'] != own['vid'] and abs(float(r['s_m'])-so) <= SENSE_M]
        for ln in range(3):
            for direction in (1, -1):
                c = [r for r in sensed if int(r['lane']) == ln and direction*(float(r['s_m'])-so) > 0]
                r = min(c, key=lambda r: (abs(float(r['s_m'])-so), str(r['vid']))) if c else None
                f.extend([(float(r['s_m'])-so)/100., (float(r['v_mps'])-vo)/10., float(bool(r['is_cav'])), 1.]
                         if r else [0., 0., 0., 0.])
        f.extend([sum(not r['is_cav'] for r in sensed)/10., sum(bool(r['is_cav']) for r in sensed)/10.])
        out.append(f+ctx)
    x = np.asarray(out, np.float32)
    if x.shape != (len(members), FEATURES) or not np.isfinite(x).all():
        raise ValueError('invalid observable features')
    return x


# --------------------------------------------------------------------------- receipts

class RecordingEvaluator:
    """Wrap an ``mppi_search`` evaluate callback and keep every scored candidate.

    The search is unchanged (same calls, same returned scores); the wrapper only
    reads the candidate plans it is asked to score and the paired comparator
    costs of the same call, so the receipt carries the complete screened pool.
    """

    def __init__(self, evaluate):
        self.evaluate, self.calls = evaluate, []

    def __call__(self, record, actions, K, H):
        scores = self.evaluate(record, actions, K, H)
        def mean(a):
            return float(np.mean([x['team_delay'] if isinstance(x, dict) else x for x in scores[a]]))
        pol, mappo = mean('policy'), mean('MAPPO')
        cands = [dict(id=a['id'], origin=a.get('origin'), duration_s=int(a['duration_s']),
                      perturbation_canonical=a['perturbation_canonical'],
                      saving_vs_policy=pol-mean(a['id']), saving_vs_mappo=mappo-mean(a['id']))
                 for a in actions if 'perturbation_canonical' in a]
        self.calls.append(dict(K=int(K), H=int(H), candidates=cands))
        return scores

    def candidates(self):
        out = []
        for i, c in enumerate(self.calls):
            stage = 'final' if i == len(self.calls)-1 and len(self.calls) > 1 else 'screen'
            out.extend(dict(x, stage=stage) for x in c['candidates'])
        return out


def receipt_from_search(record, result, recorder=None, arm='B', initial_covariance=None, sigma=None):
    """Canonical receipt from ``mppi_search.search`` output (+ optional recorder)."""
    members = list(result.get('group', {}).get('members') or record.get('group') or [])
    cfg = result.get('config', {})
    cands = recorder.candidates() if recorder is not None else []
    for p in result.get('topk') or []:
        cands.append(dict(id=p['id'], origin=p.get('origin'), duration_s=int(p['duration_s']),
                          perturbation_canonical=p['perturbation_canonical'], stage='final',
                          saving_vs_policy=p.get('saving_vs_policy'), saving_vs_mappo=p.get('saving_vs_mappo')))
    return dict(schema=SCHEMA, search_id=result.get('record_sha256') or record.get('state_id') or _digest(record),
                world_id=record.get('world_id'), arm=arm, regime=result.get('regime') or record.get('regime'),
                rows=record.get('rows'), members=members,
                canonical=dict(n_knots=ms.canonical_knots(int(cfg.get('max_duration_s', 300))),
                               span_s=int(cfg.get('max_duration_s', 300))),
                durations_s=result.get('durations_s') or durations_registered(),
                initial=dict(sigma=float(sigma if sigma is not None else cfg.get('sigma', Z_SCALE)),
                             length_knots=float(cfg.get('length_knots', 1.)),
                             covariance=None if initial_covariance is None else np.asarray(initial_covariance).tolist()),
                candidates=cands, topk=[p['id'] for p in result.get('topk') or []], executed=None)


def receipt_from_service(request, response, arm='B'):
    """Plan-service request + response (``{'key','result'}`` payload or bare result).

    Returns (receipt, None) or (None, reason). Arm-R random-match responses and
    rejected/unaccepted searches without any perturbation are skipped.
    """
    result = response.get('result', response) if isinstance(response, dict) else None
    if not isinstance(result, dict) or not isinstance(request, dict):
        return None, 'unreadable request/response'
    if result.get('random_match') or 'centre_contract' in result:
        return None, 'arm R / unscored plan (never proposer data)'
    if not result.get('topk'):
        return None, 'search without scored perturbations (rejected or unavailable)'
    if request.get('rows') is None:
        return None, 'request without observable rows'
    cov = None
    learning = request.get('search_learning')
    regime = result.get('regime') or request.get('regime')
    members = (result.get('group') or {}).get('members') or request.get('group')
    if learning and learning.get('covariance') and regime and members:
        try:
            cov = ms.CovarianceModel.from_json(learning['covariance']).get(regime, len(members))
        except (KeyError, ValueError, TypeError):
            cov = None
    rec = receipt_from_search(dict(request, regime=regime), result, None, arm, cov)
    if not rec['members']:
        return None, 'no group members'
    return rec, None


def attach_sumo(receipts, entries):
    """Join ``search_learning.update`` entries (plan id -> SUMO-realized advantage)."""
    adv = {e['plan']: float(e['sumo_realized_advantage']) for e in entries
           if e.get('plan') is not None and e.get('sumo_realized_advantage') is not None}
    for r in receipts:
        ids = {c['id'] for c in r.get('candidates') or []}
        hit = sorted(ids & set(adv))
        if hit:
            r['executed'] = dict(plan_id=hit[0], sumo_realized_advantage=adv[hit[0]])
    return receipts


def load_receipts(paths):
    """Load canonical receipts (*.json with ``schema``) or plan-service queue roots
    (``requests/`` + ``responses/`` or ``forecasts/``). Returns (receipts, skipped)."""
    out, skipped = [], []
    for p in map(Path, paths):
        if p.is_dir() and (p/'requests').is_dir():
            for req in sorted((p/'requests').glob('*.json')):
                resp = next((d/req.name for d in (p/'responses', p/'forecasts') if (d/req.name).exists()), None)
                if resp is None:
                    skipped.append((str(req), 'no response')); continue
                rec, why = receipt_from_service(json.loads(req.read_text()), json.loads(resp.read_text()))
                (out.append(rec) if rec else skipped.append((str(req), why)))
        else:
            files = sorted(p.glob('*.json')) if p.is_dir() else [p]
            for f in files:
                data = json.loads(f.read_text())
                for d in (data if isinstance(data, list) else [data]):
                    (out.append(d) if isinstance(d, dict) and d.get('schema') == SCHEMA
                     else skipped.append((str(f), 'not a proposer receipt')))
    return out, skipped


# --------------------------------------------------------------------------- examples

def initial_covariance(receipt, m):
    """Search's own initial covariance in Z_SCALE units (isotropic-smooth if absent)."""
    ini = receipt.get('initial') or {}
    n = int((receipt.get('canonical') or {}).get('n_knots', ms.canonical_knots()))
    if ini.get('covariance') is not None:
        c = np.asarray(ini['covariance'], float)
        if c.shape == (m*n, m*n):
            return c/Z_SCALE**2
    sigma = float(ini.get('sigma', Z_SCALE))
    return ms.isotropic_covariance(m, sigma, n, float(ini.get('length_knots', 1.)))/Z_SCALE**2


def initial_logpdf(z, D, cov0, cache=None):
    """Log-density (Z_SCALE units) of (z, D) under the search's initial distribution:
    N(0, Sigma_0) on the informative knots, duration uniform over the registered set."""
    m = z.shape[0]; key = int(D)
    if cache is None or key not in cache:
        keep = np.tile(knot_mask(D), m)
        L = np.linalg.cholesky(cov0[np.ix_(keep, keep)]+1e-9*np.eye(int(keep.sum())))
        val = (keep, L)
        if cache is not None:
            cache[key] = val
    keep, L = cache[key] if cache is not None else val
    a = np.linalg.solve(L, np.asarray(z, float).reshape(-1)[keep])
    return float(-.5*a@a-np.log(np.diag(L)).sum()-.5*len(a)*math.log(2*math.pi)-math.log(len(durations_registered())))


def build_search(receipt, cfg=REGISTERED):
    """Canonical receipt -> training/eval unit, or (None, reason). Tolerates gaps."""
    try:
        if receipt.get('arm', 'B') != 'B':
            return None, 'not arm B'
        members = list(receipt.get('members') or [])
        if not ms.REGISTERED['min_size'] <= len(members) <= ms.REGISTERED['max_size']:
            return None, 'group size outside 2..6'
        n = int((receipt.get('canonical') or {}).get('n_knots', ms.canonical_knots()))
        span = int((receipt.get('canonical') or {}).get('span_s', ms.REGISTERED['max_duration_s']))
        if n != ms.canonical_knots() or span != ms.REGISTERED['max_duration_s']:
            return None, 'canonical grid differs from the registered one'
        regime = receipt.get('regime') or 'unknown'
        x = member_features(receipt['rows'], members, regime)
    except (KeyError, ValueError, TypeError) as e:
        return None, f'unusable receipt: {e}'
    m = len(members); durations = durations_registered()
    cands = {}
    for c in receipt.get('candidates') or []:
        try:
            z = np.asarray(c['perturbation_canonical'], float)
            s = c.get('saving_vs_policy')
            if z.shape != (m, n) or not np.isfinite(z).all() or s is None or not math.isfinite(float(s)):
                continue
        except (KeyError, ValueError, TypeError):
            continue
        sm = c.get('saving_vs_mappo')
        sm = float(sm) if sm is not None and math.isfinite(float(sm)) else None
        # utility vs the search's own comparators: the better of 'policy continues' and MAPPO
        u = float(s) if sm is None else min(float(s), sm)
        stage = 'final' if c.get('stage') == 'final' else 'screen'
        e = cands.setdefault(c['id'], dict(id=c['id'], z=z/Z_SCALE, D=int(c['duration_s']), origin=c.get('origin')))
        # a finalist appears twice (screen K4 and final K8): keep both scores, never mix them
        e.setdefault(stage, dict(saving=float(s), u=u))
    ex = receipt.get('executed') or {}
    sumo = ex.get('sumo_realized_advantage')
    lost = ex.get('plan_id') if sumo is not None and sumo <= 0 else None
    screen = [dict(c, **c['screen']) for c in cands.values() if 'screen' in c]
    full_pool = len(screen) >= cfg['min_pool']
    if full_pool:
        # FULL screened pool (any B search, accepted or not): the search's own
        # elites by utility vs its comparators; weight = utility above the elite
        # threshold (the next-ranked candidate), so weights are >= 0 and rank-
        # informative even when no candidate beat the comparators.
        ranked = sorted(screen, key=lambda c: (-c['u'], c['id']))
        E = max(1, int(math.ceil(cfg['elites_frac']*len(screen)/ms.REGISTERED['iterations'])))
        thr = ranked[E]['u'] if len(ranked) > E else ranked[-1]['u']
        good = [dict(c, gain=c['u']-thr) for c in ranked[:E] if c['id'] != lost]
    else:
        # top-k-only receipt: candidates with positive predicted saving vs the policy
        scored = [dict(c, **c.get('final', c.get('screen'))) for c in cands.values()]
        good = [dict(c, gain=c['saving']) for c in sorted(scored, key=lambda c: (-c['saving'], c['id']))
                if c['saving'] > 0 and c['id'] != lost]         # plans that lost in SUMO are never imitated
    cov0 = initial_covariance(receipt, m); cache = {}
    def item(c, **kw):
        return dict(z=c['z'], d=duration_index(c['D'], durations), D=c['D'],
                    lp0=initial_logpdf(c['z'], c['D'], cov0, cache), **kw)
    targets = []
    if good:
        # each search contributes unit WM weight in total (scale-free)
        w = np.array([max(0., c['gain']) for c in good])
        w = w/w.sum() if w.sum() > 0 else np.full(len(good), 1/len(good))
        targets = [item(c, w=float(wi), source='wm') for c, wi in zip(good, w)]
    if sumo is not None and sumo > 0 and ex.get('plan_id') in cands:
        targets.append(item(cands[ex['plan_id']], w=float(cfg['sumo_bonus']), source='sumo'))
    topk = set(receipt.get('topk') or [])
    elites = good if full_pool else [c for c in good if c['id'] in topk and 'final' in c] or good
    return dict(id=str(receipt.get('search_id')), world=str(receipt.get('world_id')), regime=regime, m=m,
                full_pool=full_pool, accepted=bool(any(c[k]['u'] > 0 for c in cands.values() for k in ('screen', 'final')
                                                       if k in c)),
                x=x, targets=targets, elites=[item(c) for c in elites],
                cov0=cov0, n_candidates=len(cands)), None


def build_dataset(receipts, cfg=REGISTERED):
    searches, skipped = [], []
    for r in receipts:
        s, why = build_search(r, cfg)
        (searches.append(s) if s else skipped.append(why))
    return searches, skipped


def world_split(searches, val_frac=REGISTERED['val_frac'], seed=REGISTERED['seed']):
    """Deterministic world-grouped split (hash of world id + seed); no world in both."""
    worlds = sorted({s['world'] for s in searches}, key=lambda w: (_digest([seed, w]), w))
    n_val = max(1, int(round(val_frac*len(worlds)))) if len(worlds) > 1 else 0
    val = set(worlds[:n_val])
    return [s for s in searches if s['world'] not in val], [s for s in searches if s['world'] in val], sorted(val)


# --------------------------------------------------------------------------- model

class ProposerNet(nn.Module):
    """Permutation-equivariant set transformer over 2..6 member tokens.

    Outputs per member and component a mean/log-std over canonical knots, and per
    group mixture logits and per-component duration logits. Member order only
    determines output order (no positional encoding), so the likelihood of a
    joint plan is invariant to a consistent permutation of members.
    """

    def __init__(self, cfg=REGISTERED, n_knots=None, n_durations=None):
        super().__init__()
        self.cfg = dict(cfg); d = cfg['d_model']; M = cfg['components']
        self.n = ms.canonical_knots() if n_knots is None else n_knots
        self.nd = len(durations_registered()) if n_durations is None else n_durations
        self.embed = nn.Sequential(nn.Linear(FEATURES, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, cfg['heads'], 2*d, dropout=0., batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg['layers'], enable_nested_tensor=False)
        self.pool = nn.Sequential(nn.Linear(2*d, d), nn.GELU())
        self.member = nn.Linear(2*d, M*2*self.n)
        self.mix = nn.Linear(d, M)
        self.dur = nn.Linear(d, M*self.nd)
        nn.init.zeros_(self.member.weight); nn.init.zeros_(self.member.bias)   # start at the policy centre

    def forward(self, x, mask):
        """x: B x 6 x F, mask: B x 6 (True = real member)."""
        B, P, _ = x.shape; M = self.cfg['components']
        h = self.encoder(self.embed(x), src_key_padding_mask=~mask)
        mf = mask.unsqueeze(-1).float()
        mean = (h*mf).sum(1)/mf.sum(1).clamp_min(1.)
        mx = h.masked_fill(~mask.unsqueeze(-1), -1e9).max(1).values
        ctx = self.pool(torch.cat([mean, mx], -1))
        out = self.member(torch.cat([h, ctx.unsqueeze(1).expand(B, P, -1)], -1)).view(B, P, M, 2, self.n)
        lo, hi = self.cfg['logstd_min'], self.cfg['logstd_max']
        mu = out[..., 0, :]
        # zero-initialised head -> log-std 0, i.e. the search's own 0.5 m/s^2 scale
        logstd = lo+(hi-lo)*torch.sigmoid(out[..., 1, :]+math.log(-lo/hi))
        logpi = torch.log_softmax(self.mix(ctx), -1)
        logdur = torch.log_softmax(self.dur(ctx).view(B, M, self.nd), -1)
        return mu, logstd, logpi, logdur


def _batch(searches):
    P = ms.REGISTERED['max_size']
    x = np.zeros((len(searches), P, FEATURES), np.float32); mask = np.zeros((len(searches), P), bool)
    for i, s in enumerate(searches):
        x[i, :s['m']] = s['x']; mask[i, :s['m']] = True
    return torch.from_numpy(x), torch.from_numpy(mask)


def _targets(searches, key):
    P = ms.REGISTERED['max_size']; n = ms.canonical_knots()
    sid, Z, d, km, w, lp0 = [], [], [], [], [], []
    for i, s in enumerate(searches):
        for t in s[key]:
            z = np.zeros((P, n)); z[:s['m']] = t['z']
            sid.append(i); Z.append(z); d.append(t['d']); km.append(knot_mask(t['D'])); w.append(t.get('w', 1.))
            lp0.append(t['lp0'])
    if not sid:
        return None
    return (torch.tensor(sid), torch.tensor(np.stack(Z), dtype=torch.float32), torch.tensor(d),
            torch.tensor(np.stack(km)), torch.tensor(w, dtype=torch.float32), torch.tensor(lp0, dtype=torch.float32))


def log_likelihood(model, searches, key='targets', defensive=True, return_logpi=False):
    """Per-target joint log-density (Z_SCALE units) of (perturbation, duration).

    The proposer is a defensive mixture: (1 - eps) x learned M-component mixture
    + eps x the search's own initial distribution (eps = ``defensive``), so its
    density never falls more than log(1/eps) below the initial distribution.
    ``defensive=False``: learned mixture only (novelty ranking).
    """
    t = _targets(searches, key)
    if t is None:
        return (None, None, None) if return_logpi else (None, None)
    sid, Z, d, km, w, lp0 = t
    eps = float(model.cfg.get('defensive', 0.)) if defensive else 0.
    x, mask = _batch(searches)
    mu, logstd, logpi, logdur = model(x, mask)
    mu, logstd = mu[sid], logstd[sid]                                   # E x P x M x n
    z = Z.unsqueeze(2)
    lp = -.5*((z-mu)/logstd.exp())**2-logstd-.5*math.log(2*math.pi)
    keep = (mask[sid].unsqueeze(-1) & km.unsqueeze(1)).unsqueeze(2).float()   # E x P x 1 x n
    comp = (lp*keep).sum((1, 3))+logpi[sid]+logdur[sid].gather(-1, d.view(-1, 1, 1).expand(-1, logpi.shape[1], 1)).squeeze(-1)
    if eps > 0:
        comp = torch.cat([comp+math.log1p(-eps), (lp0+math.log(eps)).unsqueeze(-1)], -1)
    if return_logpi:
        return torch.logsumexp(comp, -1), w, logpi
    return torch.logsumexp(comp, -1), w


def _set_determinism(cfg):
    torch.manual_seed(cfg['seed'])
    prev = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    torch.use_deterministic_algorithms(True); torch.set_num_threads(cfg['threads'])
    return prev


def _weighted_nll(model, searches, entropy=False):
    """Weighted NLL; with ``entropy`` adds the registered mixture-entropy floor
    penalty  weight * mean_s relu(floor - H(pi_s))  so the proposer cannot
    collapse onto a single plan type (floor = 0.5 log M)."""
    ll, w, logpi = log_likelihood(model, searches, return_logpi=True)
    loss = -(w*ll).sum()/w.sum()
    if entropy and model.cfg.get('entropy_weight', 0) > 0:
        H = -(logpi.exp()*logpi).sum(-1)
        loss = loss+model.cfg['entropy_weight']*torch.relu(model.cfg['entropy_floor']-H).mean()
    return loss


def train(searches, cfg=REGISTERED):
    """Deterministic weighted-likelihood training (fixed seed, at most ``steps``).

    Early stopping uses an INNER world-grouped split of the training worlds only
    (``inner_frac``; needs >= ``inner_min_worlds`` worlds, else the last step is
    kept): every ``eval_every`` steps the inner weighted NLL is computed and the
    best state is restored. Held-out (validation) searches are never consulted.
    """
    cfg = dict(REGISTERED, **cfg)
    usable = [s for s in searches if s['targets']]
    if not usable:
        raise ValueError('no search with a positive-saving target')
    fit, inner = usable, []
    if len({s['world'] for s in usable}) >= cfg['inner_min_worlds']:
        fit, inner, _ = world_split(usable, cfg['inner_frac'], cfg['seed']+1)
    prev = _set_determinism(cfg)
    try:
        model = ProposerNet(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        g = torch.Generator().manual_seed(cfg['seed'])
        log, best = [], (math.inf, -1, None)
        for step in range(cfg['steps']):
            model.train()
            idx = torch.randint(len(fit), (min(cfg['batch'], len(fit)),), generator=g).tolist()
            loss = _weighted_nll(model, [fit[i] for i in idx], entropy=True)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
            if (step+1) % cfg['eval_every'] == 0 or step == cfg['steps']-1:
                row = dict(step=step+1, loss=float(loss.detach()))
                if inner:
                    model.eval()
                    with torch.no_grad():
                        row['inner_nll'] = float(_weighted_nll(model, inner))
                    if row['inner_nll'] < best[0]:
                        best = (row['inner_nll'], step+1, {k: v.clone() for k, v in model.state_dict().items()})
                log.append(row)
        if best[2] is not None:
            model.load_state_dict(best[2])
        model.eval()
        return model, dict(cfg=cfg, train_searches=len(usable), fit_searches=len(fit), inner_searches=len(inner),
                           selected_step=best[1] if best[2] is not None else cfg['steps'], log=log)
    finally:
        torch.set_num_threads(prev[0]); torch.use_deterministic_algorithms(prev[1])


def state_sha(model):
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in sorted(model.state_dict().items())}, buf)
    return _sha_bytes(buf.getvalue())


def save_checkpoint(model, meta, path):
    """Write <path>.pt (state dict) and <path>.json (config/meta + pt SHA256 + own SHA)."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO(); torch.save(model.state_dict(), buf); raw = buf.getvalue()
    pt = path.with_suffix('.pt'); pt.write_bytes(raw)
    body = dict(meta, schema='coopmwm.plan_proposer.checkpoint/1', cfg=model.cfg, features=FEATURES,
                z_scale=Z_SCALE, n_knots=model.n, durations_s=durations_registered(),
                pt_file=pt.name, pt_sha256=_sha_bytes(raw))
    body['json_sha256'] = _digest(body)
    path.with_suffix('.json').write_text(json.dumps(body, indent=1, sort_keys=True, default=str))
    return body


def load_checkpoint(path):
    path = Path(path); meta = json.loads(path.with_suffix('.json').read_text())
    body = {k: v for k, v in meta.items() if k != 'json_sha256'}
    if _digest(body) != meta['json_sha256']:
        raise ValueError('checkpoint JSON SHA mismatch')
    raw = (path.parent/meta['pt_file']).read_bytes()
    if _sha_bytes(raw) != meta['pt_sha256']:
        raise ValueError('checkpoint .pt SHA mismatch')
    model = ProposerNet(meta['cfg']); model.load_state_dict(torch.load(io.BytesIO(raw), weights_only=True))
    model.eval()
    return model, meta


# --------------------------------------------------------------------------- sampling / seeds

@torch.no_grad()
def sample(model, x, n, rng, modes=True, cov0=None):
    """n joint proposals [(z m x n_knots in m/s^2, D)] for one group's tokens x (m x F).

    The first min(n, M) are the component means (by mixture weight, modal
    duration); the rest are defensive-mixture samples (with probability eps from
    the search's initial N(0, cov0), isotropic-smooth if ``cov0`` is None).
    Knots after the plan end are zero.
    """
    m = x.shape[0]; durations = durations_registered()
    xb, mask = _batch([dict(x=x, m=m)])
    mu, logstd, logpi, logdur = model(xb, mask)
    mu, sd = mu[0, :m].numpy(), logstd[0, :m].exp().numpy()     # m x M x n
    pi, pdur = logpi[0].exp().numpy().astype(float), logdur[0].exp().numpy().astype(float)
    out = []
    order = sorted(range(len(pi)), key=lambda k: (-pi[k], k))
    for k in (order[:n] if modes else []):
        D = durations[int(np.argmax(pdur[k]))]
        out.append((mu[:, k]*knot_mask(D)*Z_SCALE, D))
    eps = float(model.cfg.get('defensive', 0.)); L0 = None
    while len(out) < n:
        if eps > 0 and rng.random() < eps:
            if L0 is None:
                c0 = ms.isotropic_covariance(m, Z_SCALE)/Z_SCALE**2 if cov0 is None else np.asarray(cov0, float)
                L0 = np.linalg.cholesky(c0+1e-9*np.eye(len(c0)))
            D = durations[int(rng.integers(len(durations)))]
            out.append(((L0@rng.standard_normal(L0.shape[0])).reshape(m, -1)*knot_mask(D)*Z_SCALE, D))
            continue
        k = int(rng.choice(len(pi), p=pi/pi.sum())); D = durations[int(rng.choice(len(durations), p=pdur[k]/pdur[k].sum()))]
        z = mu[:, k]+sd[:, k]*rng.standard_normal(mu[:, k].shape)
        out.append((z*knot_mask(D)*Z_SCALE, D))
    return out


def proposal_seeds(record, group, policy_traj, n, proposer, bounds, regime=None, seed=None):
    """Integration hook: ``n`` proposer seed plans for ``mppi_search.search(extra_seeds=...)``.

    record: plan-service request (observable ``rows``); group: a
    ``group_components`` item; policy_traj: members x seconds absolute m/s^2
    (same object passed to ``search``). Plans are built by ``build_plan`` (clipped
    to ``bounds``, registered knots, origin 'proposer'); the search still scores
    every seed with the WM. Deterministic given ``seed`` (default: record digest).
    """
    if n <= 0:
        return []
    model = proposer[0] if isinstance(proposer, tuple) else proposer
    regime = regime or record.get('regime') or 'unknown'
    members = group['members']
    rows = record.get('rows') or group['rows']
    x = member_features(rows, members, regime)
    seed = int(_digest(dict(record=record.get('state_id') or _digest(record), members=members))[:15], 16) \
        if seed is None else int(seed)
    rng = np.random.default_rng(seed)
    out = []
    for j, (z, D) in enumerate(sample(model, x, n, rng)):
        p = ms.build_plan(group, policy_traj, z, D, bounds, 'proposer', regime, f'P{j}-')
        out.append(ms.validate_plan(p, bounds, durations_registered()))
    return out


@torch.no_grad()
def mixture_moments(model, x, cov0=None):
    """Moment-matched single Gaussian of the defensive proposer mixture for tokens x.

    Returns (mean m*n, cov (m*n)^2) in canonical m/s^2 and duration probabilities.
    mean = sum_k w_k mu_k; cov = sum_k w_k (diag s_k^2 + mu_k mu_k^T) - mean mean^T,
    with w = (1-eps) pi and the eps defensive component N(0, cov0) (isotropic-smooth
    prior if None), uniform over durations.
    """
    m = x.shape[0]; n = ms.canonical_knots(); nd = len(durations_registered())
    xb, mask = _batch([dict(x=x, m=m)])
    mu, logstd, logpi, logdur = model(xb, mask)
    mu = mu[0, :m].permute(1, 0, 2).reshape(mu.shape[2], -1).double().numpy()          # M x (m*n)
    var = (2*logstd[0, :m]).exp().permute(1, 0, 2).reshape(mu.shape[0], -1).double().numpy()
    eps = float(model.cfg.get('defensive', 0.))
    w = (1-eps)*logpi[0].exp().double().numpy()
    c0 = ms.isotropic_covariance(m, Z_SCALE, n)/Z_SCALE**2 if cov0 is None else np.asarray(cov0, float)
    mean = w@mu
    cov = sum(w[k]*(np.diag(var[k])+np.outer(mu[k], mu[k])) for k in range(len(w)))+eps*c0-np.outer(mean, mean)
    cov = (cov+cov.T)/2+1e-6*np.eye(len(mean))
    pd = (w@logdur[0].exp().double().numpy())+eps/nd
    return mean*Z_SCALE, cov*Z_SCALE**2, pd/pd.sum()


@torch.no_grad()
def novelty_scorer(model, x, rank='ratio'):
    """Scores for the warm-start NOVELTY slice (lower = more novel; rank-based use only).

    ``ratio`` (registered): learned-mixture log-likelihood minus the generic
    isotropic-smooth prior log-density, so generic draws the proposer considers
    unlikely RELATIVE to the generic distribution rank first (not merely generic
    tails). ``likelihood``: learned-mixture log-likelihood alone.
    """
    m = x.shape[0]; c0 = ms.isotropic_covariance(m, Z_SCALE)/Z_SCALE**2

    @torch.no_grad()
    def score(Zs, Ds):
        cache = {}
        tg = [dict(z=np.asarray(z, float)/Z_SCALE, d=duration_index(D), D=int(D)) for z, D in zip(Zs, Ds)]
        for t in tg:
            t['lp0'] = initial_logpdf(t['z'], t['D'], c0, cache)
        ll, _ = log_likelihood(model, [dict(x=x, m=m, novel=tg)], 'novel', defensive=False)
        ll = ll.double().numpy()
        return ll-np.array([t['lp0'] for t in tg]) if rank == 'ratio' else ll
    return score


def warm_start(record, group, proposer, regime=None, cov0=None):
    """Learned-search hook: ``mppi_search.search(..., warm_start=warm_start(...))``.

    The proposer's (moment-matched, defensive) distribution replaces the
    policy-centred initial N(0, Sigma_0) and uniform durations; the search itself
    (population, iterations, WM scoring, top-k, Thompson) is unchanged unless the
    caller also passes a smaller ``config``. Flag-only: never used by default.
    ``generic_share`` (registered 0.2) of every iteration's sampled population
    stays on the generic policy-centred N(0, Sigma_0)/uniform-duration
    distribution so discovery is never lost, and a NOVELTY slice (registered 0.1)
    keeps, from ``novelty_oversample`` x as many generic draws, those the proposer
    ranks least likely (``novelty_scorer``; rank-based, no threshold).
    """
    model = proposer[0] if isinstance(proposer, tuple) else proposer
    regime = regime or record.get('regime') or 'unknown'
    x = member_features(record.get('rows') or group['rows'], group['members'], regime)
    mean, cov, pd = mixture_moments(model, x, None if cov0 is None else np.asarray(cov0, float)/Z_SCALE**2)
    cfg = dict(REGISTERED, **model.cfg)
    return dict(mean=mean.tolist(), cov=cov.tolist(), duration_probs=pd.tolist(), source='plan_proposer',
                generic_share=float(cfg['generic_share']), novelty_share=float(cfg['novelty_share']),
                novelty_oversample=int(cfg['novelty_oversample']), novelty_score=novelty_scorer(model, x))


@torch.no_grad()
def plan_novelty(plan, proposer, rows, regime, library=None, posthoc=None):
    """Novelty of one executed plan (report per update; descriptive only).

    Distances are RMS over the canonical m x 10 perturbation (m/s^2): to the
    nearest proposer component mean (modal duration, zero after the plan end) for
    this group's observable tokens, and to the nearest same-regime/size library
    entry (``mppi_search.PlanLibrary``). ``posthoc``: ``mppi_search.posthoc_label``
    output (family or 'new').
    """
    z = np.asarray(plan['perturbation_canonical'], float); members = plan['members']
    out = dict(plan_id=plan.get('id'), origin=plan.get('origin'), size=len(members), duration_s=plan.get('duration_s'))
    if proposer is not None:
        model = proposer[0] if isinstance(proposer, tuple) else proposer
        x = member_features(rows, members, regime)
        modes = sample(model, x, int(model.cfg['components']), np.random.default_rng(0))
        out['proposer_mode_distance'] = float(min(np.sqrt(((z-mz)**2).mean()) for mz, _ in modes))
    if library is not None:
        pool = [e for e in library.entries if e['descriptor']['regime'] == regime and e['descriptor']['size'] == len(members)]
        if pool:
            out['library_distance'] = float(min(np.sqrt(((z-np.asarray(e['perturbation_canonical'], float))**2).mean())
                                                for e in pool))
    if posthoc is not None:
        out['posthoc_label'] = posthoc.get('label')
    return out


def novelty_report(items):
    """Per-update summary of ``plan_novelty`` rows (descriptive)."""
    def stats(k):
        v = [i[k] for i in items if i.get(k) is not None]
        return dict(n=len(v), mean=float(np.mean(v)), median=float(np.median(v))) if v else dict(n=0)
    labels = [i.get('posthoc_label') for i in items if i.get('posthoc_label') is not None]
    return dict(plans=len(items), proposer_mode_distance=stats('proposer_mode_distance'),
                library_distance=stats('library_distance'),
                label_counts={k: labels.count(k) for k in sorted(set(labels))},
                new_fraction=float(np.mean([lab == 'new' for lab in labels])) if labels else None,
                by_origin={o: sum(i.get('origin') == o for i in items) for o in sorted({str(i.get('origin')) for i in items})})


# --------------------------------------------------------------------------- offline evaluation

def initial_elite_loglik(s):
    """Elite log-density under the search's own initial distribution: z ~ N(0, Sigma_0)
    on the informative knots, duration uniform over the registered set."""
    return [e['lp0'] for e in s['elites']]


def _initial_samples(s, n_samples, rng):
    durations = durations_registered(); m = s['m']; n = ms.canonical_knots()
    L = np.linalg.cholesky(s['cov0']+1e-9*np.eye(len(s['cov0'])))
    out = []
    for _ in range(n_samples):
        D = durations[int(rng.integers(len(durations)))]
        out.append(((L@rng.standard_normal(m*n)).reshape(m, n)*knot_mask(D)*Z_SCALE, D))
    return out


def _nearest(samples, elites):
    S = np.stack([z for z, _ in samples])
    return [float(np.sqrt(((S-e['z']*Z_SCALE)**2).mean((1, 2))).min()) for e in elites]


def _bootstrap_ci(values, worlds, B, seed):
    worlds = np.asarray(worlds); u = sorted(set(worlds.tolist()))
    vals = np.asarray(values, float)
    idx = {w: np.where(worlds == w)[0] for w in u}
    rng = np.random.default_rng(seed); means = []
    for _ in range(B):
        pick = rng.integers(0, len(u), len(u))
        sel = np.concatenate([idx[u[i]] for i in pick])
        means.append(vals[sel].mean())
    return float(vals.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


@torch.no_grad()
def offline_check(model, searches, cfg=REGISTERED):
    """Held-out comparison vs each search's own initial distribution.

    Per held-out search with >= 1 elite: mean elite log-likelihood difference
    (proposer - initial; nats, Z_SCALE units, duration included) and mean
    nearest-elite RMS distance difference (m/s^2) using ``eval_samples`` draws
    from each, and the same distance vs the search's existing zero 'centre' seed.
    World-cluster bootstrap 95% CIs. PASS iff enough held-out data, log-lik CI
    lower > 0 and both distance-change CI uppers < 0.
    """
    cfg = dict(REGISTERED, **cfg)
    ev = [s for s in searches if s['elites']]
    if not ev:
        return dict(passed=False, reason='no held-out search with elites', searches=0, worlds=0)
    ll, _ = log_likelihood(model, ev, 'elites')
    ll = ll.numpy(); i = 0; dll, ddist, dcent, worlds, rows = [], [], [], [], []
    for s in ev:
        k = len(s['elites']); prop = ll[i:i+k]; i += k
        base = initial_elite_loglik(s)
        rng = np.random.default_rng(int(_digest([cfg['seed'], s['id']])[:12], 16))
        dp = _nearest(sample(model, s['x'], cfg['eval_samples'], rng, cov0=s['cov0']), s['elites'])
        di = _nearest(_initial_samples(s, cfg['eval_samples'], rng), s['elites'])
        dc = [float(np.sqrt(((e['z']*Z_SCALE)**2).mean())) for e in s['elites']]   # 'centre' seed = zero
        dll.append(float(np.mean(prop)-np.mean(base))); ddist.append(float(np.mean(dp)-np.mean(di)))
        dcent.append(float(np.mean(dp)-np.mean(dc)))
        worlds.append(s['world'])
        rows.append(dict(id=s['id'], world=s['world'], m=s['m'], elites=k, ll_proposer=float(np.mean(prop)),
                         ll_initial=float(np.mean(base)), dist_proposer=float(np.mean(dp)),
                         dist_initial=float(np.mean(di)), dist_centre=float(np.mean(dc))))
    L = _bootstrap_ci(dll, worlds, cfg['bootstraps'], cfg['seed'])
    Dd = _bootstrap_ci(ddist, worlds, cfg['bootstraps'], cfg['seed']+1)
    Dc = _bootstrap_ci(dcent, worlds, cfg['bootstraps'], cfg['seed']+2)
    nw = len(set(worlds))
    enough = nw >= cfg['min_heldout_worlds'] and len(ev) >= cfg['min_heldout_searches']
    passed = bool(enough and L[1] > 0 and Dd[2] < 0 and Dc[2] < 0)
    return dict(passed=passed, enough_data=enough, searches=len(ev), worlds=nw,
                elites=int(sum(len(s['elites']) for s in ev)),
                loglik_gain=dict(mean=L[0], ci95=[L[1], L[2]]),
                nearest_elite_distance_change=dict(mean=Dd[0], ci95=[Dd[1], Dd[2]]),
                distance_change_vs_centre=dict(mean=Dc[0], ci95=[Dc[1], Dc[2]]),
                mean_ll_proposer=float(np.mean([r['ll_proposer'] for r in rows])),
                mean_ll_initial=float(np.mean([r['ll_initial'] for r in rows])),
                mean_dist_proposer=float(np.mean([r['dist_proposer'] for r in rows])),
                mean_dist_initial=float(np.mean([r['dist_initial'] for r in rows])),
                mean_dist_centre=float(np.mean([r['dist_centre'] for r in rows])),
                win_rate_loglik=float(np.mean([d > 0 for d in dll])),
                win_rate_distance=float(np.mean([d < 0 for d in ddist])), per_search=rows)


# --------------------------------------------------------------------------- synthetic receipts (tests / data-need curve)

def planted_direction(x, regime):
    """Synthetic ground truth from observable tokens: the fastest-relative member
    brakes for the first ~100 s and its follower eases; merge-dense holds longer."""
    m = x.shape[0]; n = ms.canonical_knots()
    z = np.zeros((m, n)); rel_v = x[:, 5]
    i = int(np.argmax(rel_v)); z[i, :4] = -1.
    j = int(np.argmin(np.where(np.arange(m) == i, np.inf, np.abs(x[:, 0]-x[i, 0]))))
    if j != i:
        z[j, 1:4] = -.5
    D = 180 if regime == 'merge-dense' else 90
    return z*knot_mask(D), D


def synthetic_receipts(n_searches, n_worlds, seed=0, sizes=(2, 3, 4, 5, 6), topk_only=False,
                       population=40, iterations=3, noise=.05, signal=1., null_frac=0.):
    """Fast numpy stand-in for MPPI searches with a planted, observable-dependent optimum.

    cost(z, D) = mean((z_eff - signal*z*)^2) + noise; the policy (z=0) is the
    reference. Search mimics policy-centred CEM from N(0, Sigma_0), uniform D.
    """
    rng = np.random.default_rng(seed); durations = durations_registered(); n = ms.canonical_knots()
    out = []
    for k in range(n_searches):
        w = k % n_worlds; m = int(sizes[k % len(sizes)])
        lead = float(rng.uniform(-300, 1400))
        rows, members = [], []
        s = lead
        for i in range(m):
            vid = f'w{w}c{k}_{i}'
            rows.append(dict(vid=vid, s_m=s, v_mps=float(rng.uniform(8, 30)), lane=int(rng.integers(0, 3)),
                             is_cav=True, length_m=5.)); members.append(vid)
            s -= float(rng.uniform(8, 40))
        for i in range(int(rng.integers(2, 10))):
            rows.append(dict(vid=f'w{w}h{k}_{i}', s_m=float(rng.uniform(s-60, lead+60)),
                             v_mps=float(rng.uniform(5, 30)), lane=int(rng.integers(0, 3)), is_cav=False, length_m=5.))
        regime = 'merge-dense' if MERGE_ZONE[0]-150 <= lead <= MERGE_ZONE[1]+150 else 'ordinary'
        x = member_features(rows, members, regime)
        zstar, Dstar = planted_direction(x, regime); zstar = zstar*signal
        if null_frac and rng.random() < null_frac:     # policy already optimal: an unaccepted search
            zstar = zstar*0
        cov = ms.isotropic_covariance(m, Z_SCALE, n)/Z_SCALE**2
        Lc = np.linalg.cholesky(cov+1e-9*np.eye(len(cov)))
        def cost(z, D):
            dur_pen = .02*abs(durations.index(D)-durations.index(Dstar))
            return float(((z*knot_mask(D)-zstar)**2).mean()+dur_pen+noise*rng.standard_normal())
        base = float((zstar**2).mean())
        mu = np.zeros(m*n); C = cov.copy(); pd = np.full(len(durations), 1/len(durations)); cands = []
        for it in range(iterations):
            Lc = np.linalg.cholesky(C+1e-9*np.eye(len(C))); batch = []
            for j in range(population):
                D = durations[int(rng.choice(len(durations), p=pd))]
                z = ((mu+Lc@rng.standard_normal(m*n)).reshape(m, n))*knot_mask(D)
                batch.append(dict(id=f'S{k}i{it}j{j}', origin='sampled', duration_s=D,
                                  perturbation_canonical=(z*Z_SCALE).tolist(), stage='screen',
                                  saving_vs_policy=(base-cost(z, D))*100., saving_vs_mappo=None, _z=z))
            cands += batch
            top = sorted(batch, key=lambda c: -c['saving_vs_policy'])[:population//4]
            Zt = np.stack([c['_z'].reshape(-1) for c in top]); mu = Zt.mean(0)
            C = (C+len(top)*np.cov(Zt.T, bias=True))/(1+len(top))
            wd = np.array([sum(c['duration_s'] == d for c in top) for d in durations], float)
            pd = (pd+len(top)*wd/wd.sum())/(1+len(top))
        fin = sorted(cands, key=lambda c: (-c['saving_vs_policy'], c['id']))[:8]
        final = [dict({k2: v for k2, v in c.items() if k2 != '_z'}, stage='final',
                      saving_vs_policy=(base-cost(c['_z'], c['duration_s']))*100.) for c in fin]
        topk = [c['id'] for c in sorted(final, key=lambda c: -c['saving_vs_policy'])[:4] if c['saving_vs_policy'] > 0]
        pool = [] if topk_only else [{k2: v for k2, v in c.items() if k2 != '_z'} for c in cands]
        cand_list = pool+[c for c in final if (not topk_only) or c['id'] in topk]
        executed = None
        if topk:
            ex = next(c for c in final if c['id'] == topk[0])
            executed = dict(plan_id=ex['id'], sumo_realized_advantage=float(ex['saving_vs_policy']/100.
                                                                             + .3*rng.standard_normal()))
        out.append(dict(schema=SCHEMA, search_id=f'syn{seed}-{k}', world_id=f'W{w}', arm='B', regime=regime,
                        rows=rows, members=members, canonical=dict(n_knots=n, span_s=300),
                        durations_s=durations, initial=dict(sigma=Z_SCALE, length_knots=1., covariance=None),
                        candidates=cand_list, topk=topk, executed=executed))
    return out


def data_need_curve(train_sizes=(24, 48, 96, 192, 384), heldout=96, worlds_per_search=.1, cfg=None, seed=7,
                    topk_only=False):
    """Synthetic learning curve: held-out offline check vs number of training searches."""
    cfg = dict(REGISTERED, **(cfg or {}))
    test = synthetic_receipts(heldout, 12, seed=seed+1000)
    for r in test:
        r['world_id'] = 'T'+str(r['world_id'])
    ev, _ = build_dataset(test, cfg)
    rows = []
    for n in train_sizes:
        tr, _ = build_dataset(synthetic_receipts(n, max(2, int(n*worlds_per_search)), seed=seed,
                                                 topk_only=topk_only), cfg)
        model, _ = train(tr, cfg)
        res = offline_check(model, ev, cfg)
        rows.append(dict(train_searches=n, passed=res['passed'], loglik_gain=res['loglik_gain'],
                         nearest_elite_distance_change=res['nearest_elite_distance_change'],
                         distance_change_vs_centre=res['distance_change_vs_centre'],
                         dist_proposer=res['mean_dist_proposer'], dist_initial=res['mean_dist_initial'],
                         dist_centre=res['mean_dist_centre']))
    return rows


# --------------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train', help='train on receipts (world-grouped split) and run the offline check')
    t.add_argument('inputs', nargs='+'); t.add_argument('--out', required=True)
    t.add_argument('--learning-entries', nargs='*', default=[])
    c = sub.add_parser('curve', help='synthetic data-need curve (CPU)')
    c.add_argument('--out'); c.add_argument('--topk-only', action='store_true')
    a = ap.parse_args(argv)
    if a.cmd == 'curve':
        rows = data_need_curve((24, 48, 96, 192, 384, 768), topk_only=a.topk_only)
        text = json.dumps(rows, indent=1)
        if a.out:
            Path(a.out).write_text(text)
        print(text); return 0
    receipts, skipped = load_receipts(a.inputs)
    entries = [e for f in a.learning_entries for e in json.loads(Path(f).read_text()).get('entries', [])]
    attach_sumo(receipts, entries)
    searches, why = build_dataset(receipts)
    train_s, val_s, val_worlds = world_split(searches)
    model, info = train(train_s)
    check = offline_check(model, val_s)
    meta = dict(receipts=len(receipts), skipped_files=len(skipped), skipped_receipts=len(why),
                dataset_sha256=_digest(sorted(r.get('search_id') for r in receipts)),
                validation_worlds=val_worlds, train_info=info,
                offline_check={k: v for k, v in check.items() if k != 'per_search'}, state_sha256=state_sha(model))
    body = save_checkpoint(model, meta, a.out)
    print(json.dumps({k: body[k] for k in ('pt_sha256', 'json_sha256', 'offline_check')}, indent=1, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
