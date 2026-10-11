"""Observable-history driver-type identifier (WM-S-R4-CHK-TYPEID-20261011).

Primary method, no learning, no truth input: per-vehicle posterior over the 3
campaign vTypes (aggressive/normal/cautious IDM).  Evidence channels (chosen
from label-free residual diagnostics; see scripts/exp_noise.py for the study):

* car-following ticks (same-lane leader within 100 m): Student-t(df=3) fit of
  the candidate type's IDM acceleration (own speed, gap, leader speed; v0 =
  speedFactor x lane limit; gap floor 0.3 m; clamp [-decel, accel] then [-9, 3])
  to the observed 0.5 s acceleration,
* free-flow ticks (no leader within 100 m) with speed >= 0.85 x lane limit
  (near saturation): same accel fit,
* all free-flow ticks: overspeed evidence -0.5*max(0, v - v0 - 1.0)^2 (a speed
  sustained above a type's desired speed is evidence against it; underspeed is
  not - the vehicle may still be accelerating),
* population prior 70/21/10 (normal/cautious/aggressive, normalised).

Free-flow acceleration-phase ticks (v < 0.85 x limit) carry NO accel evidence:
their residuals are strongly biased for every type (measured means -1.7/-0.9/
-0.15 m/s^2 for aggressive/normal/cautious) and would dominate long windows.

The single noise scale is estimated label-free per state set: the pooled std of
residuals around each vehicle's pilot best-fit type (pilot = the naive
sigma-scaled Gaussian likelihood).  No truth types enter anywhere.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln

TYPE_NAMES = ("aggressive_car", "normal_car", "cautious_car")
SIGMA = {"aggressive_car": 0.6, "normal_car": 0.4, "cautious_car": 0.2}
SPEED_FACTOR = {"aggressive_car": 1.08, "normal_car": 1.00, "cautious_car": 0.92}
#: (tau, minGap, accel, decel) - the byte-identical campaign vTypes (DIAG-ORACLE).
IDM5 = {
    "aggressive_car": (0.95, 1.5, 3.0, 4.5),
    "normal_car": (1.10, 2.5, 2.6, 4.5),
    "cautious_car": (1.50, 3.5, 2.0, 3.0),
}
MAIN_LIMIT, RAMP_LIMIT = 31.29, 22.22
LEN_M = 5.0
DT = 0.5
PRIOR = {"normal_car": 70.0, "cautious_car": 21.0, "aggressive_car": 10.0}
_p = sum(PRIOR.values())
PRIOR = {k: v / _p for k, v in PRIOR.items()}
LPRIOR = {t: float(np.log(PRIOR[t])) for t in TYPE_NAMES}
FREEFLOW_GAP_M = 100.0
NEARSAT_FRAC = 0.85
OVERSPEED_MARGIN_MPS = 1.0
STUDENT_NU = 3.0

_SF3 = np.array([SPEED_FACTOR[t] for t in TYPE_NAMES])
_LPRIOR3 = np.array([LPRIOR[t] for t in TYPE_NAMES])


def idm_accel(v, leader_v, gap, v0, tau, s0, a, b, delta=4):
    """Numpy port of wms.physics.idm.idm_accel, clamped to [-b, a] (then [-9, 3])."""
    closing = v - leader_v
    desired = s0 + np.maximum(0.0, v * tau + v * closing / (2.0 * np.sqrt(a * b)))
    gap = np.maximum(gap, 0.3)
    accel = a * (1.0 - (v / v0) ** delta - (desired / gap) ** 2)
    return np.clip(accel, -b, a).clip(-9.0, 3.0)


def prep(history: dict) -> dict:
    """Per-record transition-level arrays [T-1, N] (+ [T-1, N, 3] type cubes)."""
    kin = np.asarray(history["kin"], dtype=np.float64)
    mask = np.asarray(history["mask"], dtype=bool)
    ids = [str(x) for x in history["ids"]]
    t, n = kin.shape[:2]
    lane = kin[:, :, 1:4].argmax(-1)
    s_, v = kin[:, :, 0], kin[:, :, 4]
    gap = np.full((t, n), 1.0e3)
    lead_v = v.copy()
    lead_idx = np.full((t, n), -1, dtype=np.int64)
    for ti in range(t):
        ok = mask[ti]
        if not ok.any():
            continue
        ds = s_[ti][None, :] - s_[ti][:, None]
        same = ok[None, :] & (lane[ti][None, :] == lane[ti][:, None])
        bumper = np.where(same & (ds > 0), ds - LEN_M, np.inf)
        g = bumper.min(-1)
        idx = bumper.argmin(-1)
        has = np.isfinite(g)
        gap[ti, has] = g[has]
        lead_v[ti, has] = v[ti][idx[has]]
        lead_idx[ti, has] = idx[has]
    ramp = (lane == 2) & (s_ < 0)
    vlim = np.where(ramp, RAMP_LIMIT, MAIN_LIMIT)
    a_obs = (v[1:] - v[:-1]) / DT
    tok = mask[1:] & mask[:-1] & (lane[1:] == lane[:-1])
    ap = np.stack([idm_accel(v[:-1], lead_v[:-1], gap[:-1],
                             np.broadcast_to(vlim[:-1] * SPEED_FACTOR[name],
                                             v[:-1].shape), *IDM5[name])
                   for name in TYPE_NAMES], -1)
    return dict(ids=ids, tok=tok, a_obs=a_obs, ap=ap, gap=gap, v=v, vlim=vlim,
                lane=lane, mask=mask, lead_idx=lead_idx)


def _students(res, scale):
    nu = STUDENT_NU
    half = (nu + 1.0) / 2.0
    c = gammaln(half) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi * scale * scale)
    return c - half * np.log1p(res * res / (nu * scale * scale))


def _ll_cube(d: dict, scale: float) -> np.ndarray:
    """Primary log-likelihood cube [T-1, N, 3] (zero where no evidence)."""
    res = d["a_obs"][..., None] - d["ap"]
    ff = d["tok"] & (d["gap"][:-1] >= FREEFLOW_GAP_M)
    cf = d["tok"] & ~ff
    nearsat = d["v"][:-1] >= NEARSAT_FRAC * d["vlim"][:-1]
    accel_ev = cf | (ff & nearsat)
    v0c = d["vlim"][:-1][..., None] * _SF3
    over = np.maximum(d["v"][:-1][..., None] - v0c - OVERSPEED_MARGIN_MPS, 0.0)
    return (np.where(accel_ev[..., None], _students(res, scale), 0.0)
            + np.where(ff[..., None], -0.5 * over ** 2, 0.0))


def pilot_scale(preps: list[dict]) -> float:
    """Label-free noise scale: residual std around each vehicle's naive best fit."""
    allr = []
    for d in preps:
        res = d["a_obs"][..., None] - d["ap"]
        sig3 = np.array([SIGMA[t] for t in TYPE_NAMES])[None, None]
        lln = np.where(d["tok"][..., None], -0.5 * (res / sig3) ** 2, 0.0).sum(0)
        am = lln.argmax(-1)
        r = res[:, np.arange(res.shape[1]), am]
        allr.extend(r[d["tok"]].tolist())
    return float(np.std(allr))


def posteriors(d: dict, lengths_s, scale: float):
    """Per-vehicle posterior/argmax for each truncation length.

    Returns per_length[L] = list over history columns of
    dict(posterior={type: p}, argmax=str, n_ticks=int), plus full-history
    condition stats (leader share, mean speed).
    """
    ll = _ll_cube(d, scale)
    tlen = ll.shape[0]
    n = ll.shape[1]
    per_length = {}
    for length in lengths_s:
        use = int(round(length / DT))
        lo = max(0, tlen - use)
        tot = ll[lo:].sum(0) + _LPRIOR3
        xe = np.exp(tot - tot.max(-1, keepdims=True))
        post = xe / xe.sum(-1, keepdims=True)
        am = post.argmax(-1)
        n_ev = (np.abs(ll[lo:]) > 0).any(-1).sum(0) + (d["tok"][lo:].sum(0) == 0) * 0
        out = []
        for j in range(n):
            p = {t: float(post[j, i]) for i, t in enumerate(TYPE_NAMES)}
            out.append({"posterior": p, "argmax": TYPE_NAMES[int(am[j])],
                        "n_ticks": int(n_ev[j])})
        per_length[length] = out
    cf_any = d["gap"][:-1] < FREEFLOW_GAP_M
    share, mspeed, nval = [], [], []
    for j in range(n):
        ticks = np.flatnonzero(d["tok"][:, j])
        nval.append(int(ticks.size))
        if ticks.size:
            share.append(float(np.mean(cf_any[ticks, j])))
            mspeed.append(float(np.mean(d["v"][ticks, j])))
        else:
            share.append(0.0)
            mspeed.append(float("nan"))
    stats = {"leader_share": share, "mean_v": mspeed, "n_transitions": nval}
    return per_length, stats


def identify_state(history: dict, lengths_s=(5, 10, 20, 30, 60), scale: float = 1.25):
    """Convenience wrapper: (ids, per_length, stats) for one record."""
    d = prep(history)
    per_length, stats = posteriors(d, lengths_s, scale)
    return d["ids"], per_length, stats
