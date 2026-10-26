"""Observable, native-registered breakdown events and calibrated cumulative risk.

No driver labels, prospective actions, actual entry backlog or hidden traffic
types enter the risk model. A head is frozen for a policy phase; refitting and
calibration use separate TRAIN worlds at a boundary, never actor gradients.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
from torch import nn

from .contract import MERGE_S0_M, MERGE_S1_M
from .rsu_field import CELLS, FIELD_SCALE, build_rsu_field

HORIZONS = (60, 120, 300)


def merge_speed(state):
    """Mean speed observed within existing merge geometry; empty sensor = NaN."""
    mask = (state['present'].bool() & (state['lane'] >= 0)
            & (state['s_m'] >= MERGE_S0_M) & (state['s_m'] < MERGE_S1_M))
    count = mask.sum(-1)
    value = torch.where(mask, state['v_mps'], 0.).sum(-1) / count.clamp_min(1)
    return torch.where(count > 0, value, torch.full_like(value, float('nan')))


def fit_native_rule(speeds, source_sha256):
    """Two Gaussian regimes in log1p(speed); constants determined ONLY by native fit.

    Quartile initialization, 100 EM iterations, numeric variance floor only.
    Refuse a two-regime interpretation unless BIC improves over one Gaussian.
    The posterior-equality boundary must lie between the fitted regime means.
    """
    x = np.asarray(speeds, dtype=float).ravel()
    x = np.log1p(x[np.isfinite(x) & (x >= 0)])
    if len(x) < 8:
        raise ValueError('insufficient native observations for two regimes')
    means = np.quantile(x, [.25, .75])
    variances = np.full(2, max(float(x.var()), np.finfo(float).eps))
    weights = np.full(2, .5)
    for _ in range(100):
        logp = (np.log(weights) - .5 * np.log(2 * np.pi * variances)
                - .5 * (x[:, None] - means) ** 2 / variances)
        r = np.exp(logp - logp.max(1, keepdims=True))
        r /= r.sum(1, keepdims=True)
        mass = r.sum(0).clip(np.finfo(float).eps)
        weights = mass / mass.sum()
        means = (r * x[:, None]).sum(0) / mass
        variances = ((r * (x[:, None] - means) ** 2).sum(0) / mass).clip(np.finfo(float).eps)
    order = np.argsort(means)
    means, variances, weights = means[order], variances[order], weights[order]
    logp = (np.log(weights) - .5 * np.log(2 * np.pi * variances)
            - .5 * (x[:, None] - means) ** 2 / variances)
    peak = logp.max(1)
    ll2 = (peak + np.log(np.exp(logp - peak[:, None]).sum(1))).sum()
    variance = max(float(x.var()), np.finfo(float).eps)
    ll1 = (-.5 * np.log(2 * np.pi * variance) - .5 * (x - x.mean()) ** 2 / variance).sum()
    bic1, bic2 = -2 * ll1 + 2 * np.log(len(x)), -2 * ll2 + 5 * np.log(len(x))
    def odds(v):
        p = np.log(weights) - .5 * np.log(variances) - .5 * (v - means) ** 2 / variances
        return p[0] - p[1]
    lo, hi = means
    if bic2 >= bic1 or not odds(lo) > 0 > odds(hi):
        raise ValueError('native data do not identify two ordered speed regimes')
    for _ in range(80):
        mid = (lo + hi) / 2
        if odds(mid) > 0:
            lo = mid
        else:
            hi = mid
    return {'schema': 'paircrit_breakdown_rule/1', 'source_sha256': source_sha256,
            'definition': 'first entry into native lower log1p merge-speed regime',
            'means': means.tolist(), 'variances': variances.tolist(), 'weights': weights.tolist(),
            'boundary_mps': float(np.expm1((lo + hi) / 2)), 'n': len(x),
            'bic_one': float(bic1), 'bic_two': float(bic2)}


def fit_otsu_rule(speeds, source_sha256):
    """Registered native-only fallback, with no assumed two-Gaussian regime split."""
    v = np.asarray(speeds, float).ravel()
    x = np.sort(np.log1p(v[np.isfinite(v) & (v >= 0)]))
    if len(x) < 8 or x[0] == x[-1]:
        raise ValueError('native speed split unidentified')
    n = len(x)
    count = np.arange(1, n)
    total = np.cumsum(x)
    score = count * (n - count) / n ** 2 * (
        total[:-1] / count - (total[-1] - total[:-1]) / (n - count)) ** 2
    score[x[:-1] == x[1:]] = -np.inf
    index = int(np.argmax(score))
    boundary = (x[index] + x[index + 1]) / 2
    return {'schema': 'paircrit_breakdown_rule/1', 'method': 'native_log1p_speed_otsu',
        'definition': 'first entry into native lower-speed Otsu class; operational congestion proxy',
        'source_sha256': source_sha256, 'boundary_mps': float(np.expm1(boundary)),
        'between_class_variance': float(score[index]), 'n': n,
        'lower_class_fraction': float((index + 1) / n), 'physical_bistability_identified': False}


def event_labels(speeds, times_s, rule):
    """[time,...] -> cumulative outcomes [...,3], mask, and already-broken flag.

    Initially broken or empty starts are excluded from incident-risk training.
    Event observations remain valid under later censoring; non-events require
    complete finite sensor coverage through T. Empty cells are never breakdowns.
    """
    v, times = np.asarray(speeds, float), np.asarray(times_s, float)
    if v.shape[0] != len(times) or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError('event trace requires increasing relative times from zero')
    step = np.diff(times).min() if len(times) > 1 else float('inf')
    low = v < float(rule['boundary_mps'])
    initial = low[0] | ~np.isfinite(v[0])
    outcomes, valid = [], []
    for h in HORIZONS:
        use = (times > 0) & (times <= h)
        event = low[use].any(0)
        prefix = times[times <= h]
        coverage = (bool((times == h).any()) and bool((np.diff(prefix) == step).all()))
        complete = coverage & np.isfinite(v[use]).all(0)
        outcomes.append(event.astype(float))
        valid.append(~initial & (event | complete))
    return np.stack(outcomes, -1), np.stack(valid, -1), initial


def observable_features(state):
    field = build_rsu_field(state)
    # Omit the pre-existing queue-speed indicator: no hand-set behavioural cutoff.
    use = [0, 1, 2, 4, 5]
    return (field / field.new_tensor(FIELD_SCALE))[..., use].flatten(1)


class BreakdownHead(nn.Module):
    """[B,arm(native,policy),T] cumulative hazards; temperatures fit on other worlds."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(len(CELLS) * 5, 32), nn.SiLU(), nn.Linear(32, 6))
        self.register_buffer('temperature', torch.ones(2, 3))
        self.register_buffer('feature_mean', torch.zeros(len(CELLS) * 5))
        self.register_buffer('feature_scale', torch.ones(len(CELLS) * 5))
        self.register_buffer('boundary_mps', torch.tensor(float('nan')))

    def normalize_from(self, states):
        with torch.no_grad():
            x = observable_features(states)
            self.feature_mean.copy_(x.mean(0))
            sd = x.std(0, unbiased=False)
            self.feature_scale.copy_(torch.where(sd > 0, sd, torch.ones_like(sd)))

    def logits(self, state):
        x = (observable_features(state) - self.feature_mean) / self.feature_scale
        raw = self.net(x).reshape(-1, 2, 3)
        hazard = torch.nn.functional.softplus(raw).cumsum(-1)
        # logit(1-exp(-cumulative hazard)), numerically stable for large hazards.
        return hazard + torch.log(-torch.expm1(-hazard))

    def forward(self, state):
        probability = (self.logits(state) / self.temperature).sigmoid().cummax(-1).values
        speed = merge_speed(state)
        # Incident head is trained on initially unbroken starts. Already-low
        # observable states have probability one for the inclusive within-T query.
        probability = torch.where((speed < self.boundary_mps)[:, None, None], 1., probability)
        return torch.where(torch.isfinite(speed)[:, None, None], probability,
                           torch.full_like(probability, float('nan')))


def risk_loss(head, states, targets, mask, weight=None):
    """Real labels or K-sample frequencies; zero unreliable WM horizons have no mass."""
    logits = head.logits(states)
    weights = mask.to(logits)
    if weight is not None:
        weights = weights * torch.as_tensor(weight, device=logits.device, dtype=logits.dtype)
    if not bool(torch.isfinite(targets).all()) or bool(((targets < 0) | (targets > 1)).any()):
        raise ValueError('risk targets must be finite event frequencies in [0,1]')
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.to(logits), reduction='none')
    return (weights * loss).sum() / weights.sum().clamp_min(1)


def calibrate(head, states, outcomes, mask):
    """Fit six positive temperatures on calibration worlds, keeping predictor fixed."""
    with torch.no_grad():
        logits = head.logits(states).detach()
    log_temperature = nn.Parameter(torch.zeros_like(head.temperature))
    opt = torch.optim.LBFGS([log_temperature], max_iter=50, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        temp = log_temperature.clamp(-5, 5).exp()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits / temp, outcomes.to(logits), reduction='none')
        loss = (loss * mask).sum() / mask.sum().clamp_min(1)
        loss.backward()
        return loss
    opt.step(closure)
    head.temperature.copy_(log_temperature.detach().clamp(-5, 5).exp())


def reliability_diagram(probability, outcome, mask, bins=10):
    p, y, m = np.asarray(probability), np.asarray(outcome), np.asarray(mask, bool)
    rows = []
    for i in range(bins):
        use = m & (p >= i / bins) & ((p < (i + 1) / bins) if i < bins - 1 else (p <= 1))
        rows.append({'lo': i / bins, 'hi': (i + 1) / bins, 'n': int(use.sum()),
                     'predicted': float(p[use].mean()) if use.any() else None,
                     'observed': float(y[use].mean()) if use.any() else None})
    return {'brier': float(((p[m] - y[m]) ** 2).mean()) if m.any() else None, 'bins': rows}


def event_reliability(doc, wm_sha256s, rule):
    """Event-specific bootstrap Brier skill lower bound, never delay reliability."""
    if (doc.get('schema') != 'paircrit_event_reliability/1'
            or set(wm_sha256s) != {doc.get('wm_sha256')}
            or doc.get('rule_sha256') != rule_sha256(rule) or doc.get('split') != 'TRAIN'):
        return [0.] * 3
    out = []
    for h in HORIZONS:
        row = doc.get('horizons', {}).get(str(h), {})
        try:
            lo = float(row.get('brier_skill_lower95', 0.))
            n = int(row.get('n_worlds', 0))
        except (TypeError, ValueError):
            lo, n = 0., 0
        out.append(max(0., min(1., lo)) if np.isfinite(lo) and n >= 3 else 0.)
    return out


def fit_head(head, states, outcomes, mask, *, updates=400, wm_batches=()):
    """Boundary-only supervised fit; WM batches contain K>=2 paired frequencies.

    Each WM batch is (states, frequencies, valid_sample_counts, measured_weights).
    Counts retain censoring; horizons with no event-specific reliability add no
    gradient. Predictive model is frozen again before actor use.
    """
    head.requires_grad_(True)
    head.normalize_from(states)
    optimizer = torch.optim.Adam(head.parameters(), lr=.003)
    for _ in range(updates):
        loss = risk_loss(head, states, outcomes, mask)
        for wm_states, frequency, counts, weights in wm_batches:
            if bool((counts < 0).any()):
                raise ValueError('negative WM sample counts')
            loss = loss + risk_loss(head, wm_states, frequency, counts >= 2, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return head.eval().requires_grad_(False)


def qualify_calibration(probability, outcome, mask, worlds, fit_prevalence):
    """Independent-world Brier noninferiority versus fitting-world prevalence.

    Simultaneous upper95 bootstrap of max(mean Brier difference across all six
    outputs), zero margin. No behavioral/probability cutoff is introduced.
    Calibration and qualification worlds must be separate at the caller.
    """
    p, y, m = np.asarray(probability), np.asarray(outcome), np.asarray(mask, bool)
    worlds = np.asarray(worlds)
    unique = sorted(set(worlds.tolist()))
    if not np.isfinite(p[m]).all():
        return {'qualified': False, 'reason': 'nonfinite probability on an observed outcome',
                'n_worlds': len(unique)}
    if len(unique) < 3 or any(len(set(worlds[m[:, a, h]].tolist())) < 3
                              for a in range(2) for h in range(3)):
        return {'qualified': False, 'reason': 'fewer than three independent scoring worlds per output',
                'n_worlds': len(unique)}
    delta = np.where(m, (p - y) ** 2 - (np.asarray(fit_prevalence) - y) ** 2, 0.)
    # Sum/count reduction preserves unequal per-world numbers of eligible starts.
    sums = np.stack([(delta[worlds == w] * m[worlds == w]).sum(0) for w in unique])
    counts = np.stack([m[worlds == w].sum(0) for w in unique])
    rng = np.random.default_rng(20261026)
    indices = rng.integers(len(unique), size=(2000, len(unique)))
    ratios = sums[indices].sum(1) / counts[indices].sum(1).clip(1)
    upper = float(np.quantile(ratios.max((1, 2)), .95))
    return {'qualified': bool(upper <= 0), 'upper95_max_brier_difference': upper,
            'n_worlds': len(unique), 'margin': 0., 'bootstrap_seed': 20261026, 'draws': 2000}


def rule_sha256(rule):
    return hashlib.sha256(json.dumps(rule, sort_keys=True).encode()).hexdigest()
