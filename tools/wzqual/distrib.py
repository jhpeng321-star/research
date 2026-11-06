"""Distributional comparison WM vs SUMO at the lane drop (PRIMARY, WM_QUAL_GATE.md section 3).

Per start (world x t0, observed state shared by truth and WM) a summary of the future on
[t0, t0+H]: breakdown occurrence by 60/120/300 s and onset (frozen rule), recovery for starts that
are already broken, mean queue (all lanes / closing lane / extent), discharge throughput, zone-speed
10 s bins, team delay.  The WM contributes K samples per start (expected values = sample means);
truth contributes the one recorded future.  Metrics compare the DISTRIBUTIONS over starts per cell
and pooled (breakdown rate, onset W1, queue/throughput relative error and normalised W1, CRPS);
per-start skill (Brier, AUC, reliability) is reported as secondary.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np

from . import breakdown as BD
from . import frame as F
from . import traffic as TR
from .traj import Traj

HORIZONS = (60.0, 120.0, 300.0)
BOOT_B = 2000
BOOT_SEED = 20261106


def summarize_future(traj: Traj, t0: float, horizon_s: float, rule: BD.BreakdownRule, *,
                     broken_at_start: bool, cohort: Optional[np.ndarray] = None) -> dict:
    t1 = t0 + horizon_s
    zone = TR.zone_speed_bins(traj, t0, t1)
    ev = BD.event(zone, rule, broken_at_start=broken_at_start)
    q = TR.queue_bins(traj, t0, t1)
    out = dict(zone=[None if not math.isfinite(z) else float(z) for z in zone],
               onset_s=ev.onset_s, occ={str(int(h)): ev.occurred(h) for h in HORIZONS if h <= horizon_s},
               recovered={str(int(h)): BD.recovered(zone, rule, h) for h in HORIZONS if h <= horizon_s},
               queue_mean=float(np.mean(q["count"])), closing_queue_mean=float(np.mean(q["closing"])),
               extent_mean_m=float(np.mean(q["extent_m"])),
               throughput_vph=TR.throughput_vph(traj, t0, t1))
    if cohort is not None:
        out["J_cohort"] = TR.team_delay(traj, t0, t1, cohort=cohort, include_backlog=False)
    out["J_total"] = (TR.team_delay(traj, t0, t1, include_backlog=True)
                      if traj.backlog is not None else None)
    return out


# ------------------------------------------------------------------ statistics helpers
def weighted_w1(x, wx, y, wy) -> float:
    """Wasserstein-1 between two weighted 1-D empirical distributions."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    wx, wy = np.asarray(wx, float), np.asarray(wy, float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    wx, wy = wx / wx.sum(), wy / wy.sum()
    grid = np.sort(np.concatenate([x, y]))
    cx = np.array([wx[x <= g].sum() for g in grid])
    cy = np.array([wy[y <= g].sum() for g in grid])
    return float(np.sum(np.abs(cx - cy)[:-1] * np.diff(grid)))


def crps_ensemble(samples, y) -> float:
    xs = np.asarray(samples, float)
    if xs.size == 0:
        return float("nan")
    return float(np.mean(np.abs(xs - y)) - 0.5 * np.mean(np.abs(xs[:, None] - xs[None, :])))


def auc(p, y) -> Optional[float]:
    p, y = np.asarray(p, float), np.asarray(y, bool)
    if y.all() or (~y).all():
        return None
    pos, neg = p[y], p[~y]
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (pos.size * neg.size))


def cluster_bootstrap(records: Sequence[dict], stat: Callable[[Sequence[dict]], float], *,
                      key: str = "seed", b: int = BOOT_B, seed: int = BOOT_SEED) -> tuple:
    """Percentile 95% CI of ``stat`` resampling whole worlds (all cells/starts of a seed)."""
    groups: dict = {}
    for r in records:
        groups.setdefault(r[key], []).append(r)
    keys = sorted(groups)
    if len(keys) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        pick = rng.integers(0, len(keys), len(keys))
        rs = [r for i in pick for r in groups[keys[i]]]
        v = stat(rs)
        if v is not None and math.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# ------------------------------------------------------------------ metrics
def _p(r, h, src="wm"):
    smp = r[src]
    return float(np.mean([s["occ"][h] for s in smp])) if smp else float("nan")


def breakdown_rate(records, h="300", src="wm"):
    rs = [r for r in records if not r["broken_at_start"]]
    if not rs:
        return None
    return float(np.mean([_p(r, h, src) for r in rs]))


def truth_rate(records, h="300"):
    rs = [r for r in records if not r["broken_at_start"]]
    if not rs:
        return None
    return float(np.mean([float(r["truth"]["occ"][h]) for r in rs]))


def rate_diff(records, h="300", src="wm"):
    a, b = breakdown_rate(records, h, src), truth_rate(records, h)
    return None if a is None or b is None else a - b


def _onsets(records, src):
    xs, ws = [], []
    for r in records:
        if r["broken_at_start"]:
            continue
        smp = r[src] if src != "truth" else [r["truth"]]
        k = len(smp)
        for s in smp:
            if s["occ"]["300"]:
                xs.append(s["onset_s"])
                ws.append(1.0 / k)
    return np.asarray(xs), np.asarray(ws)


def onset_w1(records, src="wm"):
    x, wx = _onsets(records, src)
    y, wy = _onsets(records, "truth")
    if len(y) < 10 or len(x) == 0:
        return None
    return weighted_w1(x, wx, y, wy)


def _pooled(records, field, src):
    xs, ws, ys = [], [], []
    for r in records:
        smp = r[src]
        for s in smp:
            xs.append(s[field])
            ws.append(1.0 / len(smp))
        ys.append(r["truth"][field])
    return np.asarray(xs, float), np.asarray(ws, float), np.asarray(ys, float)


def rel_mean_err(records, field, src="wm"):
    x, w, y = _pooled(records, field, src)
    if y.size == 0 or abs(y.mean()) < 1e-9:
        return None
    return float((np.sum(w * x) / np.sum(w) - y.mean()) / y.mean())


def norm_w1(records, field, src="wm"):
    x, w, y = _pooled(records, field, src)
    sd = float(np.std(y))
    if y.size < 2 or sd < 1e-9:
        return None
    return weighted_w1(x, w, y, np.ones_like(y)) / sd


def zone_crps(records, horizon_s, src="wm"):
    """Mean ensemble CRPS of zone-speed 10 s bins within the horizon (empty zone = speed limit)."""
    nb = int(round(horizon_s / BD.BIN_S))
    vals = []
    for r in records:
        y = [F.SPEED_LIMIT if z is None else z for z in r["truth"]["zone"][:nb]]
        smp = [[F.SPEED_LIMIT if z is None else z for z in s["zone"][:nb]] for s in r[src]]
        if not smp:
            continue
        for b in range(min(nb, len(y))):
            vals.append(crps_ensemble([s[b] for s in smp if b < len(s)], y[b]))
    return float(np.nanmean(vals)) if vals else None


def brier(records, h="300", src="wm"):
    rs = [r for r in records if not r["broken_at_start"]]
    if not rs:
        return None
    return float(np.mean([(_p(r, h, src) - float(r["truth"]["occ"][h])) ** 2 for r in rs]))


def reliability(records, h="300", src="wm", bins=10):
    rs = [r for r in records if not r["broken_at_start"]]
    rows = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        sel = [r for r in rs if (lo <= _p(r, h, src) < hi) or (i == bins - 1 and _p(r, h, src) == 1.0)]
        rows.append(dict(bin=[lo, hi], n=len(sel),
                         mean_p=float(np.mean([_p(r, h, src) for r in sel])) if sel else None,
                         freq=float(np.mean([r["truth"]["occ"][h] for r in sel])) if sel else None))
    return rows


def recovery_diff(records, h="300", src="wm"):
    rs = [r for r in records if r["broken_at_start"]]
    if not rs:
        return None
    wm = np.mean([np.mean([s["recovered"][h] for s in r[src]]) for r in rs])
    tr = np.mean([float(r["truth"]["recovered"][h]) for r in rs])
    return dict(n=len(rs), wm=float(wm), truth=float(tr), diff=float(wm - tr))
