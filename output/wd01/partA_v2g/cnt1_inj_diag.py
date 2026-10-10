"""WD-01-PA2G-CNT1 descriptive diagnostic: injected-vehicle s_RMSE vs
the MATCHED truth arrivals, by order of entry per entrance.

Registered metric (REGISTRATION_PA2G_CNT1.md section 4: "a separate
injected-vehicle s_RMSE vs the matched truth arrivals, reported
descriptively by order of entry per entrance").  The raw rows carry the
per-window aggregates only, not per-injected-vehicle trajectories, so
this diagnostic re-runs the deterministic v2g_cnt rollouts (identical
code and inputs -> identical trajectories; the engine has no RNG) on a
disclosed LABELLED SUBSET: the first 2 panel worlds (the preflight
worlds), all 18 windows, H = 120 (the longest registered horizon).

Matching: within a window, the i-th injected vehicle at entrance e is
matched to the i-th TRUE arrival at entrance e (true arrivals ordered
by first appearance in (w, w + H]); entrance from the first
appearance's model lane (0/1 = mainline, 2 = ramp).  s_RMSE on the
shared d axis (metrics.WindowSeries.rmse convention), pooled over
co-present steps.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WD01 = HERE.parent
for _p in (str(WD01), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import os
os.environ.setdefault("WD01_DATA_ROOT", str(Path("E:/rd-wt/wd01/output/wd01")))

from partA_v2g import arrivals as ARR  # noqa: E402
from partA_v2g import metrics as MX  # noqa: E402
from partA_v2g.engine import PartAV2GEngine, load_params  # noqa: E402
from partA_v2g import eval_a2 as A2  # noqa: E402
from partA_v2g import eval_a as EA  # noqa: E402
from partA_v2g import eval_cnt1 as C1  # noqa: E402

PANEL_FIRST2 = 2
H = 120


def true_arrivals_by_entrance(log, w, n_h):
    """True arrivals in (w, w+H] ordered by first appearance, split by
    entrance (first-appearance model lane)."""
    j0 = int(round((w - log.t0) / 0.5))
    seen = set(log.rows[j0])
    by_e = {"main": [], "ramp": []}
    for jj in range(1, n_h + 1):
        for vid, st in log.rows.get(j0 + jj, {}).items():
            if vid not in seen:
                if st[0] in (0, 1):
                    by_e["main"].append(vid)
                elif st[0] == 2:
                    by_e["ramp"].append(vid)
            seen.add(vid)
    return by_e


def main():
    EA.MAPPER = EA.ModelMapper()
    params = load_params()
    worlds = C1.load_panel_worlds()[:PANEL_FIRST2]
    offs = np.asarray(params["geometry"]["lane_offsets_m"])
    n_roll = int(round(C1.H_MAX / C1.DT))
    n_h = int(round(H / C1.DT))
    se = {"main": [], "ramp": []}
    n_pairs = {"main": 0, "ramp": 0}
    by_order = {}
    for entry in worlds:
        log = EA.NativeLog(entry)
        deps = ARR.read_departures(entry["native_run"])["departures"]
        for k in range(1, 19):
            w = log.t0 + 30.0 * k
            if w > log.t_end + 1e-9:
                break
            truth_w = log.truth_at(w)
            j0 = int(round((w - log.t0) / 0.5))
            vehicles, _ = EA.tracked_state(
                truth_w, params,
                inj_at=getattr(log, "inj", {}).get(j0))
            vehicles = A2.flag_cavs(vehicles, log.cav_ids)
            vids = sorted(vehicles)
            counts = ARR.counts_from_departures(deps, w, ARR.W_COUNTER_S)
            sched = ARR.insertion_schedule(counts, w, C1.H_MAX,
                                           ARR.W_COUNTER_S)
            eng = PartAV2GEngine(
                params, t0=w, vehicles=vehicles,
                control_hook=A2.native_control_hook(
                    w, params, set(vids) & log.cav_ids))
            inj = ARR.ArrivalInjector(eng, sched)
            inj.run(n_roll)
            snaps = MX.snaps_list(eng)
            truth_snaps = [log.rows.get(j0 + jj, {})
                           for jj in range(n_h + 1)]
            true_by_e = true_arrivals_by_entrance(log, w, n_h)
            for e in ("main", "ramp"):
                inj_e = [r for r in inj.inserted if r["entrance"] == e]
                for i, rec in enumerate(inj_e):
                    if i >= len(true_by_e[e]):
                        continue
                    tvid = true_by_e[e][i]
                    se_s = n = 0.0
                    order_key = (e, i)
                    se_o = no = 0.0
                    for jj in range(1, n_h + 1):
                        p = snaps[jj].get(rec["vid"])
                        t = truth_snaps[jj].get(tvid)
                        if p is None or t is None or p[0] < 0 or t[0] < 0:
                            continue
                        dd = (offs[p[0]] - p[1]) - (offs[t[0]] - t[1])
                        se_s += dd * dd
                        n += 1
                        se_o += dd * dd
                        no += 1
                    if n:
                        se[e].append(se_s)
                        n_pairs[e] += n
                        by_order.setdefault(order_key, [0.0, 0])
                        by_order[order_key][0] += se_o
                        by_order[order_key][1] += no
    out = {"subset_worlds": [w["world"] for w in worlds], "H": H,
           "by_entrance": {}, "by_order": {}}
    for e in ("main", "ramp"):
        if n_pairs[e]:
            out["by_entrance"][e] = dict(
                s_rmse=float(np.sqrt(sum(se[e]) / n_pairs[e])),
                n_pairs=int(n_pairs[e]))
    for (e, i), (s, n) in sorted(by_order.items()):
        if n:
            out["by_order"][f"{e}#{i}"] = dict(
                s_rmse=float(np.sqrt(s / n)), n_pairs=int(n))
    dest = HERE / "results" / "eval_cnt1" / "inj_diag.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print("DONE cnt1_inj_diag ->", dest)


if __name__ == "__main__":
    main()
