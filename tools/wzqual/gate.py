"""Evaluate the pre-registered gate (gate_spec.json, WM_QUAL_GATE.md) on a metrics document.

``evaluate(metrics, spec)`` -> ``{"criteria": {name: {status, value, threshold}}, "verdict"}``
with status PASS / FAIL / UNIDENTIFIED per criterion and the registered verdict rule.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

SPEC_PATH = Path(__file__).resolve().parent / "gate_spec.json"


def load_spec(path: str | Path = SPEC_PATH) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _num(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _crit(status, value, threshold, note=""):
    return dict(status=status, value=value, threshold=threshold, note=note)


def evaluate(m: dict, spec: dict) -> dict:
    P = spec["primary"]
    mins = spec["minimum_samples"]
    c: dict = {}
    integ = m.get("integrity", {})
    bad = [k for k, v in integ.items() if v is not True]
    d = m.get("distributional", {})
    a = m.get("action_response", {})

    cov = m.get("role_coverage", {})
    frac, roles = cov.get("fraction_T_candidates_supported"), set(cov.get("supported_roles", []))
    need = set(P["C0_role_coverage"]["required_roles"])
    if not _num(frac):
        c["C0_role_coverage"] = _crit("UNIDENTIFIED", frac, P["C0_role_coverage"])
    else:
        ok = frac >= P["C0_role_coverage"]["min_fraction_T_candidates_supported"] and need <= roles
        c["C0_role_coverage"] = _crit("PASS" if ok else "FAIL", dict(fraction=frac, roles=sorted(roles)),
                                      P["C0_role_coverage"])

    # ---- distributional
    if not d.get("backend_models_entries", False):
        # D1-D4 need predicted entrants (the inflow drives the drop): without them UNIDENTIFIED
        d = dict(d, n_starts=0, n_starts_not_broken=0, n_truth_events=0, rule_degenerate=True)
    n_starts = d.get("n_starts_not_broken", 0)
    degenerate = bool(d.get("rule_degenerate", True))
    th = P["D1_breakdown_rate"]
    if degenerate or n_starts < mins["starts_total"] or not _num(d.get("rate_diff_pooled")):
        c["D1_breakdown_rate"] = _crit("UNIDENTIFIED", d.get("rate_diff_pooled"), th,
                                       "degenerate rule or too few starts")
    else:
        cells = {k: v for k, v in (d.get("rate_diff_by_cell") or {}).items()
                 if v.get("n", 0) >= mins["starts_per_cell_for_cell_check"] and _num(v.get("diff"))}
        ok = abs(d["rate_diff_pooled"]) <= th["pooled_abs_diff_max"] and all(
            abs(v["diff"]) <= th["per_cell_abs_diff_max"] for v in cells.values())
        c["D1_breakdown_rate"] = _crit("PASS" if ok else "FAIL",
                                       dict(pooled=d["rate_diff_pooled"],
                                            cells={k: v["diff"] for k, v in cells.items()}), th)
    w1 = d.get("onset_w1_s")
    if degenerate or d.get("n_truth_events", 0) < mins["truth_breakdown_events"] or not _num(w1):
        c["D2_onset"] = _crit("UNIDENTIFIED", w1, P["D2_onset_w1_max_s"], "too few real breakdowns")
    else:
        c["D2_onset"] = _crit("PASS" if w1 <= P["D2_onset_w1_max_s"] else "FAIL", w1, P["D2_onset_w1_max_s"])
    q = P["D3_queue"]
    qr, qw = d.get("queue_rel_mean_err"), d.get("queue_norm_w1")
    if d.get("n_starts", 0) < mins["starts_total"] or not (_num(qr) and _num(qw)):
        c["D3_queue"] = _crit("UNIDENTIFIED", dict(rel=qr, nw1=qw), q)
    else:
        ok = abs(qr) <= q["rel_mean_err_max"] and qw <= q["norm_w1_max"]
        c["D3_queue"] = _crit("PASS" if ok else "FAIL", dict(rel=qr, nw1=qw), q)
    tr = d.get("throughput_rel_err")
    if d.get("n_starts", 0) < mins["starts_total"] or not _num(tr):
        c["D4_throughput"] = _crit("UNIDENTIFIED", tr, P["D4_throughput_rel_err_max"])
    else:
        c["D4_throughput"] = _crit("PASS" if abs(tr) <= P["D4_throughput_rel_err_max"] else "FAIL", tr,
                                   P["D4_throughput_rel_err_max"])
    crps = d.get("zone_crps", {})
    hs = P["D5_zone_crps_not_worse_than_keep_speed"]["horizons_s"]
    vals = {str(h): (crps.get(str(h), {}).get("wm"), crps.get(str(h), {}).get("keep_speed")) for h in hs}
    if any(not (_num(w) and _num(k)) for w, k in vals.values()):
        c["D5_zone_crps"] = _crit("UNIDENTIFIED", vals, hs)
    else:
        c["D5_zone_crps"] = _crit("PASS" if all(w <= k for w, k in vals.values()) else "FAIL", vals, hs)

    # ---- action response
    n_mom = a.get("n_moments_ranked", 0)
    t1 = P["A1_within_moment_spearman_T"]
    v, ci = a.get("within_moment_spearman_T"), a.get("within_moment_spearman_T_ci") or [None, None]
    if n_mom < mins["moments_ranked"] or not _num(v):
        c["A1_within_moment_spearman_T"] = _crit("UNIDENTIFIED", dict(value=v, n=n_mom), t1)
    else:
        ok = v >= t1["min"] and _num(ci[0]) and ci[0] > t1["ci_low_min"]
        c["A1_within_moment_spearman_T"] = _crit("PASS" if ok else "FAIL", dict(value=v, ci=ci, n=n_mom), t1)
    t2 = P["A2_pooled_spearman_T"]
    v, ci = a.get("pooled_spearman_T"), a.get("pooled_spearman_T_ci") or [None, None]
    if n_mom < mins["moments_ranked"] or not _num(v):
        c["A2_pooled_spearman_T"] = _crit("UNIDENTIFIED", v, t2)
    else:
        ok = v >= t2["min"] and _num(ci[0]) and ci[0] > t2["ci_low_min"]
        c["A2_pooled_spearman_T"] = _crit("PASS" if ok else "FAIL", dict(value=v, ci=ci), t2)
    t3 = P["A3_precision_positive_T"]
    pp = a.get("precision_positive_T") or {}
    if n_mom < mins["moments_ranked"] or not (_num(pp.get("precision")) and _num(pp.get("base_rate"))):
        c["A3_precision_positive_T"] = _crit("UNIDENTIFIED", pp, t3)
    else:
        ok = pp["precision"] >= max(t3["min"], pp["base_rate"] + t3["min_over_base_rate"])
        c["A3_precision_positive_T"] = _crit("PASS" if ok else "FAIL", pp, t3)
    t4 = P["A4_precision_at_k_S"]
    pk = a.get("precision_at_k_S") or {}
    if pk.get("n", 0) < mins["S_candidates"] or not (_num(pk.get("precision")) and _num(pk.get("base_rate"))):
        c["A4_precision_at_k_S"] = _crit("UNIDENTIFIED", pk, t4)
    else:
        need = max(t4["min"], min(t4.get("lift_requirement_cap", 1.0),
                                  t4["min_lift_over_base_rate"] * pk["base_rate"]))
        ok = pk["precision"] >= need
        c["A4_precision_at_k_S"] = _crit("PASS" if ok else "FAIL", pk, t4)
    t5 = P["A5_regret"]
    rg = a.get("regret") or {}
    ratio, gci = rg.get("ratio_wm_to_random"), rg.get("wm_gain_ci") or [None, None]
    if n_mom < mins["moments_ranked"] or not _num(ratio):
        c["A5_regret"] = _crit("UNIDENTIFIED", rg, t5)
    else:
        ok = ratio <= t5["max_ratio_to_random"] and _num(gci[0]) and gci[0] > t5["wm_gain_ci_low_min"]
        c["A5_regret"] = _crit("PASS" if ok else "FAIL", dict(ratio=ratio, wm_gain_ci=gci,
                                                              wm_gain=rg.get("wm_gain")), t5)
    ks = (a.get("keep_speed") or {}).get("pooled_spearman_T")
    v = a.get("pooled_spearman_T")
    if not (_num(v) and _num(ks)):
        c["A6_beats_keep_speed"] = _crit("UNIDENTIFIED", dict(wm=v, keep_speed=ks), True)
    else:
        c["A6_beats_keep_speed"] = _crit("PASS" if v > ks else "FAIL", dict(wm=v, keep_speed=ks), True)

    statuses = [x["status"] for x in c.values()]
    if bad:
        verdict = "INVALID"
    elif "FAIL" in statuses:
        verdict = "FAIL"
    elif "UNIDENTIFIED" in statuses:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS"
    return dict(criteria=c, integrity_failures=bad, verdict=verdict)
