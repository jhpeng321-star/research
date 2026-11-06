"""Action-response metrics: does the WM rank cooperation moves the way SUMO does?

Inputs are per-candidate rows ``{cid, moment, seed, family, arm, T_true, S_true, S_lb_true,
admitted, pred_T, pred_S}`` (pred_* = WM expectation over samples; T = J(base) - J(group),
S = min_i J(solo_i) - J(group), veh-s, positive = better).  Comparable merge-map reference
(WM_ERROR_CHECK_20261105.md): SUMO wins among WM-positive predictions 40.7 %, Spearman 0.103.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .distrib import cluster_bootstrap

MERGE_REFERENCE = dict(precision_positive=0.407, spearman=0.103,
                       source="runs/active/C4-V7FT-20261017/WM_ERROR_CHECK_20261105.md")


def _ranks(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return r


def spearman(x, y) -> Optional[float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return None
    rx, ry = _ranks(x[m]), _ranks(y[m])
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _ok(v):
    return v is not None and math.isfinite(v)


def rows_with(rows, *fields):
    return [r for r in rows if all(_ok(r.get(f)) for f in fields)]


def pooled_spearman(rows, pred="pred_T", true="T_true"):
    rs = rows_with(rows, pred, true)
    return spearman([r[pred] for r in rs], [r[true] for r in rs])


def within_moment_spearman(rows, pred="pred_T", true="T_true", min_n=3):
    by: dict = {}
    for r in rows_with(rows, pred, true):
        by.setdefault(r["moment"], []).append(r)
    vals = [spearman([r[pred] for r in g], [r[true] for r in g]) for g in by.values() if len(g) >= min_n]
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)) if vals else None), len(vals)


def precision_positive(rows, pred="pred_T", true="T_true"):
    rs = rows_with(rows, pred, true)
    pos = [r for r in rs if r[pred] > 0]
    base = float(np.mean([r[true] > 0 for r in rs])) if rs else None
    prec = float(np.mean([r[true] > 0 for r in pos])) if pos else None
    agree = float(np.mean([(r[pred] > 0) == (r[true] > 0) for r in rs])) if rs else None
    return dict(n=len(rs), n_pred_pos=len(pos), precision=prec, base_rate=base, agreement=agree)


def k_for(n: int) -> int:
    """Registered k: 10 when at least 30 labelled candidates, else ceil(n/3) (>= 1)."""
    return 10 if n >= 30 else max(1, math.ceil(n / 3))


def precision_at_k(rows, pred="pred_S", true="S_true", k: Optional[int] = None, strict_field=None):
    rs = rows_with(rows, pred, true)
    if not rs:
        return dict(n=0, k=0, precision=None, base_rate=None)
    k = k or k_for(len(rs))
    top = sorted(rs, key=lambda r: (-r[pred], r["cid"]))[:k]
    hit = (lambda r: _ok(r.get(strict_field)) and r[strict_field] > 0) if strict_field else (lambda r: r[true] > 0)
    return dict(n=len(rs), k=k, precision=float(np.mean([hit(r) for r in top])),
                base_rate=float(np.mean([hit(r) for r in rs])),
                admitted_in_top=int(sum(bool(r.get("admitted")) for r in top)),
                admitted_total=int(sum(bool(r.get("admitted")) for r in rs)))


def regret(rows, pred="pred_T", true="T_true"):
    """Per moment: WM choice = argmax pred_T if > 0 else none (gain 0); oracle = max(0, max T);
    random = uniform over the moment's candidates.  Returns per-moment records + means."""
    by: dict = {}
    for r in rows_with(rows, pred, true):
        by.setdefault(r["moment"], []).append(r)
    recs = []
    for mom, g in sorted(by.items()):
        best = max(0.0, max(r[true] for r in g))
        top = max(g, key=lambda r: (r[pred], r["cid"]))
        chosen = top[true] if top[pred] > 0 else 0.0
        recs.append(dict(moment=mom, seed=g[0]["seed"], n=len(g), best=best, wm_gain=chosen,
                         wm_regret=best - chosen, random_regret=best - float(np.mean([r[true] for r in g])),
                         none_regret=best, acted=bool(top[pred] > 0)))
    if not recs:
        return dict(n_moments=0, records=[])
    mean = lambda f: float(np.mean([x[f] for x in recs]))  # noqa: E731
    return dict(n_moments=len(recs), wm_regret=mean("wm_regret"), random_regret=mean("random_regret"),
                none_regret=mean("none_regret"), wm_gain=mean("wm_gain"),
                acted_frac=float(np.mean([x["acted"] for x in recs])), records=recs)


def action_response(rows: Sequence[dict], *, ks_rows: Optional[Sequence[dict]] = None) -> dict:
    """All action-response metrics with world-cluster bootstrap CIs (A1-A6 of the gate)."""
    rows = list(rows)
    wm_sp, n_mom = within_moment_spearman(rows)
    out = dict(
        n_candidates=len(rows_with(rows, "pred_T", "T_true")),
        within_moment_spearman_T=wm_sp, n_moments_ranked=n_mom,
        within_moment_spearman_T_ci=cluster_bootstrap(rows, lambda rs: within_moment_spearman(rs)[0]),
        pooled_spearman_T=pooled_spearman(rows),
        pooled_spearman_T_ci=cluster_bootstrap(rows, lambda rs: pooled_spearman(rs)),
        precision_positive_T=precision_positive(rows),
        pooled_spearman_S=pooled_spearman(rows, "pred_S", "S_true"),
        precision_at_k_S=precision_at_k(rows),
        precision_at_k_S_strict=precision_at_k(rows, strict_field="S_lb_true"),
        merge_reference=MERGE_REFERENCE)
    reg = regret(rows)
    out["regret"] = {k: v for k, v in reg.items() if k != "records"}
    if reg.get("records"):
        recs = reg["records"]
        out["regret"]["wm_gain_ci"] = cluster_bootstrap(recs, lambda rs: float(np.mean([x["wm_gain"] for x in rs])))
        out["regret"]["ratio_wm_to_random"] = (out["regret"]["wm_regret"] / out["regret"]["random_regret"]
                                               if out["regret"]["random_regret"] > 0 else None)
    fams = sorted({r["family"] for r in rows})
    out["by_family"] = {f: dict(n=len(rows_with([r for r in rows if r["family"] == f], "pred_T", "T_true")),
                                spearman_T=pooled_spearman([r for r in rows if r["family"] == f]),
                                precision_positive_T=precision_positive([r for r in rows if r["family"] == f])["precision"])
                        for f in fams}
    if ks_rows is not None:
        out["keep_speed"] = dict(pooled_spearman_T=pooled_spearman(ks_rows),
                                 within_moment_spearman_T=within_moment_spearman(ks_rows)[0],
                                 precision_positive_T=precision_positive(ks_rows),
                                 precision_at_k_S=precision_at_k(ks_rows),
                                 regret={k: v for k, v in regret(ks_rows).items() if k != "records"})
    return out
