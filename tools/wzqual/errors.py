"""Car-level and traffic-level multi-step prediction errors (secondary metrics, WM_QUAL_GATE.md 5).

Car level, per horizon h (one physics step 0.5 s, 60, 120, 300 s): vehicles present at t0 in both
truth and prediction; |s_pred - s_true| and |v_pred - v_true| for vehicles present in both at
t0+h, lane agreement, presence agreement (exited vs still present); HDV and CAV separately.
Traffic level: per section 30 s bins mean speed / density absolute error; zone-speed 10 s bins;
t0-cohort team delay J over [t0, t0+h).
"""
from __future__ import annotations

import numpy as np

from . import traffic as TR
from .traj import Traj

HORIZONS_S = (0.5, 60.0, 120.0, 300.0)


def _frame_map(traj: Traj, t: float) -> dict:
    k = traj.step_at(t)
    fr = traj.frame(k)
    return {name: (fr["s"][j], fr["v"][j], fr["lane"][j]) for j, name in enumerate(fr["ids"])}


def car_errors(truth: Traj, pred: Traj, t0: float, horizons_s=HORIZONS_S) -> dict:
    """``{h: {"hdv"|"cav": {"ds": [...], "dv": [...], "lane_ok": [...], "presence_ok": [...]}}}``."""
    base = _frame_map(truth, t0)
    cav_of = {n: bool(truth.is_cav[truth.index_of(n)]) for n in base}
    out = {}
    for h in horizons_s:
        t = t0 + h
        try:
            tm = _frame_map(truth, t)
            pm = _frame_map(pred, t)
        except KeyError:
            continue
        rec = {g: dict(ds=[], dv=[], lane_ok=[], presence_ok=[]) for g in ("hdv", "cav")}
        for n in base:
            g = "cav" if cav_of[n] else "hdv"
            a, b = tm.get(n), pm.get(n)
            rec[g]["presence_ok"].append((a is None) == (b is None))
            if a is not None and b is not None:
                rec[g]["ds"].append(abs(float(b[0]) - float(a[0])))
                rec[g]["dv"].append(abs(float(b[1]) - float(a[1])))
                rec[g]["lane_ok"].append(int(a[2]) == int(b[2]))
        out[float(h)] = rec
    return out


def summarize_car(records: list, horizons_s=HORIZONS_S) -> dict:
    """Pool per-start car_errors records: median / p90 / mean |ds|, |dv|, lane and presence agreement."""
    out = {}
    for h in horizons_s:
        h = float(h)
        row = {}
        for g in ("hdv", "cav"):
            ds = np.concatenate([np.asarray(r[h][g]["ds"], float) for r in records if h in r] or [np.zeros(0)])
            dv = np.concatenate([np.asarray(r[h][g]["dv"], float) for r in records if h in r] or [np.zeros(0)])
            lo = np.concatenate([np.asarray(r[h][g]["lane_ok"], float) for r in records if h in r] or [np.zeros(0)])
            po = np.concatenate([np.asarray(r[h][g]["presence_ok"], float) for r in records if h in r] or [np.zeros(0)])
            row[g] = dict(n=int(ds.size),
                          ds_median=float(np.median(ds)) if ds.size else None,
                          ds_p90=float(np.percentile(ds, 90)) if ds.size else None,
                          ds_mean=float(ds.mean()) if ds.size else None,
                          dv_median=float(np.median(dv)) if dv.size else None,
                          dv_p90=float(np.percentile(dv, 90)) if dv.size else None,
                          lane_agree=float(lo.mean()) if lo.size else None,
                          presence_agree=float(po.mean()) if po.size else None)
        out[str(h)] = row
    return out


def traffic_errors(truth: Traj, pred: Traj, t0: float, horizon_s: float = 300.0) -> dict:
    """Section speed/density MAE (30 s bins), zone-speed MAE (10 s bins), cohort J error at 60/120/300."""
    t1 = t0 + horizon_s
    st, sp = TR.section_bins(truth, t0, t1), TR.section_bins(pred, t0, t1)
    out = {}
    for name in st:
        for q in ("speed", "density"):
            a, b = st[name][q], sp[name][q]
            m = np.isfinite(a) & np.isfinite(b)
            out[f"{name}_{q}_mae"] = float(np.abs(a[m] - b[m]).mean()) if m.any() else None
    za, zb = TR.zone_speed_bins(truth, t0, t1), TR.zone_speed_bins(pred, t0, t1)
    m = np.isfinite(za) & np.isfinite(zb)
    out["zone_speed_mae"] = float(np.abs(za[m] - zb[m]).mean()) if m.any() else None
    cohort_names = truth.frame(truth.step_at(t0))["ids"]
    ct = np.asarray([truth.index_of(n) for n in cohort_names])
    cp = np.asarray([pred.index_of(n) for n in cohort_names if n in pred._index])
    for h in (60.0, 120.0, 300.0):
        if h <= horizon_s:
            jt = TR.team_delay(truth, t0, t0 + h, cohort=ct, include_backlog=False)
            jp = TR.team_delay(pred, t0, t0 + h, cohort=cp, include_backlog=False)
            out[f"J_cohort_{int(h)}_err"] = jp - jt
            out[f"J_cohort_{int(h)}_true"] = jt
    return out
