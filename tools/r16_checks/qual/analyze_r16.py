"""EXPLORATORY S-RAMPSPEED (from LOOP14C analyze.py): qualification-style sign units (pulse_p, pulse_m, gap_open) per CAV-side variant.

Uses the REGISTERED unit computations of qual_analyze.py (score_units, frozen TOLERANCES_v2.json) on scratch WM
tapes; the registered gate, its runs and its verdicts are unchanged.  Also reports the PULSEP diagnostics: target
dv sign agreement with SUMO at 2 s (pulse_p) and the gap_open ramp-car merge-time shift.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tools"))
import qual_analyze as Q  # noqa: E402

SP, SC = str(HERE / "data/pulse"), str(HERE / "data/coop")
tol = json.load(open(HERE / "tools/TOLERANCES_v2.json"))
sumo_p = Q.add_contrasts(Q.pulse_events(SP))
sumo_c = [r for r in Q.c9_events(SC) if r["item"] == "gap_open"]
UNITS = ["pulse_p/follower@5", "pulse_p/follower@10", "pulse_m/follower@5", "pulse_m/follower@10",
         "gap_open/follower@10", "gap_open/follower@30"]
variants = sys.argv[1].split(",") if len(sys.argv) > 1 else ["r14", "r16"]
out = {}


def target_agree(S, W, item, h):
    agree, n = 0, 0
    for k, s in S.items():
        if s["item"] != item or s["kind"] != "follower":
            continue
        w = W.get(k)
        xs = Q.at({"dv": s["dv_target"]}, h)
        if not np.isfinite(xs) or abs(xs) <= 0.05:
            continue
        n += 1
        xw = float("nan") if w is None else Q.at({"dv": w["dv_target"]}, h)
        agree += int(np.isfinite(xw) and np.sign(xw) == np.sign(xs))
    return f"{agree}/{n}"


for v in variants:
    pr, cr = str(HERE / "out" / v / "pulse"), str(HERE / "out" / v / "c9")
    res = {}
    for sname, ks in Q.SETS.items():
        def loader(k):
            rows = []
            if os.path.isdir(os.path.join(pr, "wm", f"k{k}")):
                rows += Q.pulse_events(SP, pr, k)
            var = "as_is" if k == 0 else f"as_is_k{k}"
            if os.path.isdir(os.path.join(cr, "wm", var)):
                rows += [r for r in Q.c9_events(SC, cr, var) if r["item"] == "gap_open"]
            for r in rows:      # keep the target dv for the diagnostic (wm_set_rows averages dv/dgap only)
                r.setdefault("dv_target_list", [r["dv_target"]])
            return Q.add_contrasts(rows) if rows else None
        W = Q.wm_set_rows(loader, ks)
        # mean target dv over the set's samples (diagnostic only)
        acc = {}
        for k in ks:
            for r in (loader(k) or []):
                acc.setdefault(Q.key(r), []).append(r["dv_target"])
        for kk, lst in acc.items():
            if kk in W:
                L = min(len(x) for x in lst)
                W[kk]["dv_target"] = np.nanmean(np.stack([x[:L] for x in lst]), 0)
        sc = Q.score_units(sumo_p + sumo_c, W, tol)
        S = {Q.key(r): r for r in sumo_p + sumo_c}
        dm = [r.get("dmerge_t") for r in W.values() if r["item"] == "gap_open" and r["kind"] == "ramp_car"]
        dm_s = [r.get("dmerge_t") for r in sumo_c if r["kind"] == "ramp_car"]
        res[sname] = {u: {k2: sc[u].get(k2) for k2 in ("n_clear", "worlds", "sign_rate", "sign_ci",
                                                         "majority_null_rate", "sign_verdict")} for u in UNITS}
        res[sname]["target_dv_sign_agree@2"] = {it: target_agree(S, W, it, 2) for it in ("pulse_p", "pulse_m")}
        res[sname]["gap_open_ramp_dmerge_wm_mean"] = float(np.nanmean([x for x in dm if x is not None])) if dm else None
        res[sname]["gap_open_ramp_dmerge_sumo_mean"] = float(np.nanmean([x for x in dm_s if x is not None])) if dm_s else None
        res[sname]["n_wm_events"] = len(W)
    out[v] = res
    print(v)
    for sname, r in res.items():
        print(" ", sname, {u: (r[u]["sign_verdict"], None if r[u]["sign_rate"] is None else round(r[u]["sign_rate"], 2),
                              r[u]["n_clear"]) for u in UNITS})
        print("   target agree@2", r["target_dv_sign_agree@2"], "dmerge wm/sumo",
              r["gap_open_ramp_dmerge_wm_mean"], r["gap_open_ramp_dmerge_sumo_mean"])
(HERE / "R16_QUALCHECK.json").write_text(json.dumps({"label": "EXPLORATORY (registered gate unchanged)",
                                                          "wm": "13E stage-B FINAL b2 wm_s.pt sha256 952ec059...84c8, R14 physics",
                                                          "results": out}, indent=1, default=float))
