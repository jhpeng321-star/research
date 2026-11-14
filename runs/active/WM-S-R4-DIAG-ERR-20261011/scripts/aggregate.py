"""DIAG 1 step 2: decompose per-HDV |ds| at 60/120 s for C1 and R4 on wall_same.

Groups (defined on RECORDED data, model-independent unless stated):
  E_h  = t0 HDV whose recorded same-lane leader OR follower at any tick <= h is an
         entrant (id absent from the t0 modelled set, appearing later in the window);
  N_h  = not entrant-affected; split at horizon h by predicted-lane correctness
         (model-specific): wrong-lane vs right-lane.
Zones (t0 position): on_ramp (lane 2), merge_zone (lane 0/1, 0<=s<=75.44),
  upstream_mainline (lane 0/1, s<0), downstream_mainline (lane 0/1, s>75.44).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

OUT = HERE.parent / "out"
CACHE = ROOT / "runs/active/WM-S-R4-20261011/score/cache"
HORIZONS = {60: 120, 120: 240}
MERGE_LO, MERGE_HI = 0.0, 75.44


def zone_of(lane: int, s: float) -> str:
    if lane == 2:
        return "on_ramp"
    if lane in (0, 1):
        return "merge_zone" if MERGE_LO <= s <= MERGE_HI else (
            "upstream_mainline" if s < MERGE_LO else "downstream_mainline")
    return "junction"


def leader_follower_affected(state, t0_modelled: set[str], entrants: set[str]) -> tuple[set[str], set[str]]:
    """Ids whose same-lane recorded leader/follower at any tick is an entrant (all, and <=120 ticks)."""
    ids = state["rids"]
    rec = state["rec"]
    present, lane, s = rec["present"], rec["lane"], rec["s"]
    aff_all: set[str] = set()
    aff_120: set[str] = set()
    n = len(ids)
    for k in range(lane.shape[0]):
        rows = [j for j in range(n) if present[k, j] and lane[k, j] >= 0]
        if not rows:
            continue
        for cur_lane in (0, 1, 2):
            members = sorted((j for j in rows if lane[k, j] == cur_lane), key=lambda j: s[k, j])
            for a, b in zip(members, members[1:]):  # a behind, b ahead
                behind, ahead = ids[a], ids[b]      # b is a's leader; a is b's follower
                if ahead in entrants and behind in t0_modelled:
                    aff_all.add(behind)
                    if k <= 120:
                        aff_120.add(behind)
                if behind in entrants and ahead in t0_modelled:
                    aff_all.add(ahead)
                    if k <= 120:
                        aff_120.add(ahead)
    return aff_all, aff_120


def summarize(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    a = np.asarray(vals, float)
    return {"n": len(vals), "mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90))}


def main() -> None:
    with open(OUT / "rollouts.pkl", "rb") as fh:
        payloads = pickle.load(fh)
    states = payloads["C1"]
    n_states = len(states)

    per_state = []
    pooled: dict[str, dict] = {m: {str(h): {"rows": []} for h in HORIZONS} for m in payloads}
    for si, st in enumerate(states):
        rec_ids, rec = st["rids"], st["rec"]
        ridx = {x: i for i, x in enumerate(rec_ids)}
        t0_mod = [rec_ids[j] for j in range(len(rec_ids)) if rec["present"][0, j] and rec["lane"][0, j] >= 0]
        t0_set = set(t0_mod)
        hdv0 = [v for v in t0_mod if not bool(rec["is_cav"][0, ridx[v]])]
        # entrants: appear (present, lane>=0) at any tick 1..240, absent from t0 modelled set
        entrants, entrant_kind, first_tick = {}, {}, {}
        for j, vid in enumerate(rec_ids):
            if vid in t0_set:
                continue
            ticks = np.where(rec["present"][:, j] & (rec["lane"][:, j] >= 0))[0]
            ticks = ticks[ticks >= 1]
            if len(ticks):
                entrants[vid] = ticks
                first_tick[vid] = int(ticks[0])
                entrant_kind[vid] = "junction_at_t0" if rec["present"][0, j] else "new_entry"
        entrant_set = set(entrants)
        landed_mainline = sum(1 for v in entrant_set
                              if np.any(rec["lane"][:, ridx[v]] >= 0) and np.any(rec["lane"][:, ridx[v]] <= 1))
        aff_all, aff_120 = leader_follower_affected(st, t0_set, entrant_set)
        # exits among t0 HDVs
        exits = {}
        for v in hdv0:
            j = ridx[v]
            last = int(np.where(rec["present"][:, j])[0].max())
            exits[v] = last
        zone = {v: zone_of(int(rec["lane"][0, ridx[v]]), float(rec["s"][0, ridx[v]])) for v in hdv0}

        row = {"idx": si, "key": st["key"], "world": st["world"], "t0": st["t0"],
               "n_hdvs_t0": len(hdv0), "n_cavs_t0": len(t0_mod) - len(hdv0),
               "entrants_by_60": sum(1 for v in entrant_set if first_tick[v] <= 120),
               "entrants_by_120": sum(1 for v in entrant_set if first_tick[v] <= 240),
               "entrants_new": sum(1 for v in entrant_set if entrant_kind[v] == "new_entry"),
               "entrants_junction_t0": sum(1 for v in entrant_set if entrant_kind[v] == "junction_at_t0"),
               "entrants_landed_mainline": int(landed_mainline),
               "hdv_exits_before_60": sum(1 for v in hdv0 if exits[v] < 120),
               "hdv_exits_before_120": sum(1 for v in hdv0 if exits[v] < 240),
               "hdv_affected_by_60": sum(1 for v in hdv0 if v in aff_120),
               "hdv_affected_by_120": sum(1 for v in hdv0 if v in aff_all),
               "models": {}}
        for model, sts in payloads.items():
            pids, pred = sts[si]["pids"], sts[si]["pred"]
            pidx = {x: i for i, x in enumerate(pids)}
            mrow = {}
            for h, tick in HORIZONS.items():
                matched = [v for v in hdv0
                           if rec["present"][tick, ridx[v]] and pidx.get(v, -1) >= 0
                           and pred["present"][tick, pidx[v]]]
                wrong = [v for v in matched if rec["lane"][tick, ridx[v]] >= 0
                         and int(pred["lane"][tick, pidx[v]]) != int(rec["lane"][tick, ridx[v]])]
                mrow[str(h)] = {
                    "n_matched": len(matched),
                    "ds": {v: abs(float(pred["s"][tick, pidx[v]]) - float(rec["s"][tick, ridx[v]]))
                           for v in matched},
                    "signed": {v: float(pred["s"][tick, pidx[v]]) - float(rec["s"][tick, ridx[v]])
                               for v in matched},
                    "dv": {v: float(pred["v"][tick, pidx[v]]) - float(rec["v"][tick, ridx[v]])
                           for v in matched},
                    "wrong_lane": set(wrong),
                }
            # presence consistency for t0 HDVs at every 10 s tick
            mism = 0
            for tick in range(20, 241, 20):
                for v in hdv0:
                    rp = bool(rec["present"][tick, ridx[v]])
                    pp = bool(pidx.get(v, -1) >= 0 and pred["present"][tick, pidx[v]])
                    mism += int(rp != pp)
            mrow["presence_mismatch_per_10s_ticks"] = mism
            row["models"][model] = mrow
        per_state.append(row)
        # collect pooled per-HDV rows
        for model in payloads:
            for h in HORIZONS:
                block = row["models"][model][str(h)]
                for v, d in block["ds"].items():
                    pooled[model][str(h)]["rows"].append({
                        "state": si, "id": v, "ds": d, "zone": zone[v],
                        "e": v in aff_all if h == 120 else v in aff_120,
                        "wrong": v in block["wrong_lane"], "signed": block["signed"][v],
                        "dv": block["dv"][v]})

    # ---------------- aggregates ----------------
    def group_stats(rows, pred) -> dict:
        sel = [r for r in rows if pred(r)]
        return summarize([r["ds"] for r in sel])

    attribution = {}
    for model in payloads:
        attribution[model] = {}
        for h in HORIZONS:
            rows = pooled[model][str(h)]["rows"]
            total = len(rows)
            mean_all = float(np.mean([r["ds"] for r in rows])) if rows else None
            med_state = None  # state-median-of-medians for cross-check vs cache
            groups = {}
            for gname, pred in (
                ("E_affected", lambda r: r["e"]),
                ("N_wrong_lane", lambda r: (not r["e"]) and r["wrong"]),
                ("N_right_lane", lambda r: (not r["e"]) and not r["wrong"]),
            ):
                st = group_stats(rows, pred)
                st["share"] = st["n"] / total if total else None
                st["contrib_to_mean"] = (sum(r["ds"] for r in rows if pred(r)) / total) if total else None
                sel = [r for r in rows if pred(r)]
                st["signed_mean"] = float(np.mean([r["signed"] for r in sel])) if sel else None
                st["signed_median"] = float(np.median([r["signed"] for r in sel])) if sel else None
                st["dv_mean"] = float(np.mean([r["dv"] for r in sel])) if sel else None
                groups[gname] = st
            zones = {}
            for z in ("on_ramp", "merge_zone", "upstream_mainline", "downstream_mainline"):
                sel = [r for r in rows if r["zone"] == z]
                st = summarize([r["ds"] for r in sel])
                st["contrib_to_mean"] = (sum(r["ds"] for r in sel) / total) if total and sel else 0.0
                zones[z] = st
            attribution[model][str(h)] = {"n_pooled": total, "mean_abs_ds": mean_all,
                                          "pooled_median": float(np.median([r["ds"] for r in rows])) if rows else None,
                                          "groups": groups, "zones": zones}

    # paired gap R4-C1 on common matched ids
    paired = {}
    for h in HORIZONS:
        c1 = {(r["state"], r["id"]): r for r in pooled["C1"][str(h)]["rows"]}
        r4 = {(r["state"], r["id"]): r for r in pooled["R4"][str(h)]["rows"]}
        common = sorted(set(c1) & set(r4))
        gaps = []
        for k in common:
            gaps.append({"gap": r4[k]["ds"] - c1[k]["ds"], "e": c1[k]["e"],
                         "zone": c1[k]["zone"], "c1": c1[k]["ds"], "r4": r4[k]["ds"]})
        n = len(gaps)
        mean_gap = float(np.mean([g["gap"] for g in gaps])) if gaps else None
        by = {}
        for gname, pred in (("E_affected", lambda g: g["e"]), ("N", lambda g: not g["e"])):
            sel = [g for g in gaps if pred(g)]
            by[gname] = {"n": len(sel),
                         "mean_c1": float(np.mean([g["c1"] for g in sel])) if sel else None,
                         "mean_r4": float(np.mean([g["r4"] for g in sel])) if sel else None,
                         "mean_gap": float(np.mean([g["gap"] for g in gaps if pred(g)])) if sel else None,
                         "contrib_to_mean_gap": (sum(g["gap"] for g in sel) / n) if n and sel else None}
        by["mean_gap_all"] = mean_gap
        paired[str(h)] = by

    # time profile per 10 s for E120/N120 groups
    profile = {m: {} for m in payloads}
    for model in payloads:
        st_rows = states
        for tick in range(20, 241, 20):
            e_vals, n_vals, wrong_vals, all_vals = [], [], [], []
            for si, st in enumerate(st_rows):
                rec_ids, rec = st["rids"], st["rec"]
                ridx = {x: i for i, x in enumerate(rec_ids)}
                # recompute matched + affected sets quickly via per_state cache
                pass
            profile[model][str(tick // 2)] = None  # filled below

    # (fill profile from a second pass over per-state data already in pooled rows is not
    # possible at 10 s ticks; do it here directly)
    aff_cache = []
    for si, st in enumerate(states):
        rec_ids, rec = st["rids"], st["rec"]
        ridx = {x: i for i, x in enumerate(rec_ids)}
        t0_mod = [rec_ids[j] for j in range(len(rec_ids)) if rec["present"][0, j] and rec["lane"][0, j] >= 0]
        t0_set = set(t0_mod)
        entrant_set = set()
        for j, vid in enumerate(rec_ids):
            if vid in t0_set:
                continue
            ticks = np.where(rec["present"][:, j] & (rec["lane"][:, j] >= 0))[0]
            if len(ticks[ticks >= 1]):
                entrant_set.add(vid)
        aff_all, _ = leader_follower_affected(st, t0_set, entrant_set)
        aff_cache.append((ridx, t0_mod, aff_all))
    for model in payloads:
        pidx_all = {si: {x: i for i, x in enumerate(payloads[model][si]["pids"])} for si in range(n_states)}
        for tick in range(20, 241, 20):
            e_vals, n_vals, wrong_share, all_vals = [], [], [], []
            for si, st in enumerate(states):
                ridx, t0_mod, aff_all = aff_cache[si]
                rec = st["rec"]
                pred, pidx = payloads[model][si]["pred"], pidx_all[si]
                matched = [v for v in t0_mod if not bool(rec["is_cav"][0, ridx[v]])
                           and rec["present"][tick, ridx[v]] and pidx.get(v, -1) >= 0
                           and pred["present"][tick, pidx[v]]]
                for v in matched:
                    d = abs(float(pred["s"][tick, pidx[v]]) - float(rec["s"][tick, ridx[v]]))
                    (e_vals if v in aff_all else n_vals).append(d)
                    all_vals.append(d)
                    if int(pred["lane"][tick, pidx[v]]) != int(rec["lane"][tick, ridx[v]]) and rec["lane"][tick, ridx[v]] >= 0:
                        wrong_share.append(1)
                    else:
                        wrong_share.append(0)
            profile[model][str(tick // 2)] = {
                "E120_median": float(np.median(e_vals)) if e_vals else None,
                "N120_median": float(np.median(n_vals)) if n_vals else None,
                "all_median": float(np.median(all_vals)) if all_vals else None,
                "wrong_lane_share": float(np.mean(wrong_share)) if wrong_share else None}

    # verification vs the R4 score cache
    verify = {"max_abs_diff_ds_med": 0.0, "n_checked": 0, "missing_in_cache": 0}
    for model, cache_name in (("C1", "C1"), ("R4", "R4-last")):
        cache = json.loads((CACHE / f"{cache_name}__wall_same.json").read_text(encoding="utf-8"))["states"]
        for row, crow in zip(per_state, cache):
            arm = "wall"
            for h in HORIZONS:
                mine = row["models"][model][str(h)]
                ds = sorted(mine["ds"].values())
                mine_med = float(np.median(ds)) if ds else None
                theirs = (((crow.get("arms") or {}).get(arm) or {}).get("car", {})
                          .get(f"{float(h)}", {}).get("hdv", {}) or {}).get("ds_med")
                if theirs is None:
                    verify["missing_in_cache"] += 1
                    continue
                verify["n_checked"] += 1
                verify["max_abs_diff_ds_med"] = max(verify["max_abs_diff_ds_med"], abs(mine_med - float(theirs)))

    # per-state medians summary (the metric ROUND4_SCORE reports: median over states)
    state_medians = {m: {} for m in payloads}
    for model in payloads:
        for h in HORIZONS:
            vals = []
            for row in per_state:
                ds = sorted(row["models"][model][str(h)]["ds"].values())
                if ds:
                    vals.append(float(np.median(ds)))
            state_medians[model][str(h)] = summarize(vals)

    # blow-up rule + top states
    blowups = {m: 0 for m in payloads}
    for model in payloads:
        for row in per_state:
            d60 = row["models"][model]["60"]["ds"]
            d120 = row["models"][model]["120"]["ds"]
            m60 = float(np.median(sorted(d60.values()))) if d60 else None
            m120 = float(np.median(sorted(d120.values()))) if d120 else None
            if m60 is not None and m120 is not None and m60 > 50 and m120 > 100:
                blowups[model] += 1
    top_states = sorted(
        per_state, key=lambda r: -(np.median(sorted(r["models"]["C1"]["120"]["ds"].values()))
                                   if r["models"]["C1"]["120"]["ds"] else -1))[:5]

    results = {
        "schema": "wm_s_r4_diag_err/1",
        "set": "wall_same h60 (67 states; 1 truncated recorded arm < 120 s -> 66 at 120 s)",
        "models": {"C1": "data_cache/ckpt_c1fast/wm_s.pt",
                   "R4": "runs/active/WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s.pt (R4-last)"},
        "definitions": {
            "entrant": "id absent from the t0 modelled set (lane>=0 at t0) with a present lane>=0 row at tick>=1; "
                       "split new_entry (not in t0 frame at all) vs junction_at_t0 (on lane -1 at t0)",
            "affected": "recorded same-lane leader OR follower at any tick<=h is an entrant (lane -1 rows excluded)",
            "matched": "t0 HDV present in BOTH predicted and recorded frames at t0+h; HDV = recorded is_cav False",
            "zones": "t0 lane==2 -> on_ramp; lane 0/1: s in [0,75.44] merge_zone, s<0 upstream_mainline, "
                     "s>75.44 downstream_mainline (SECTIONS bounds from src/wms/eval/metrics.py)",
            "pooled_vs_reported": "attribution uses POOLED per-HDV values; the ROUND4_SCORE metric is the "
                                  "median over states of per-state medians (both reported)"},
        "verification_vs_round4_cache": verify,
        "state_median_metric": state_medians,
        "attribution": attribution,
        "paired_gap_r4_minus_c1": paired,
        "time_profile_10s": profile,
        "blowups_rule_50m60s_100m120s": blowups,
        "entrants_per_state": summarize([r["entrants_by_60"] for r in per_state]) | {"label": "by 60 s"} | summarize([r["entrants_by_120"] for r in per_state]),
        "per_state": per_state,
    }
    # cleaner entrant summaries
    results["entrants_per_state"] = {
        "by_60": summarize([r["entrants_by_60"] for r in per_state]),
        "by_120": summarize([r["entrants_by_120"] for r in per_state]),
        "new_entry_by_120": summarize([r["entrants_new"] for r in per_state]),
        "junction_at_t0_by_120": summarize([r["entrants_junction_t0"] for r in per_state]),
        "landed_mainline_by_120": summarize([r["entrants_landed_mainline"] for r in per_state]),
        "hdv_affected_by_60": summarize([r["hdv_affected_by_60"] for r in per_state]),
        "hdv_affected_by_120": summarize([r["hdv_affected_by_120"] for r in per_state]),
        "hdv_exits_before_60": summarize([r["hdv_exits_before_60"] for r in per_state]),
        "hdv_exits_before_120": summarize([r["hdv_exits_before_120"] for r in per_state]),
    }
    top = [{"key": r["key"], "c1_med120": (float(np.median(sorted(r["models"]["C1"]["120"]["ds"].values())))
                                          if r["models"]["C1"]["120"]["ds"] else None),
            "r4_med120": (float(np.median(sorted(r["models"]["R4"]["120"]["ds"].values())))
                          if r["models"]["R4"]["120"]["ds"] else None),
            "entrants_by_120": r["entrants_by_120"], "hdv_affected_by_120": r["hdv_affected_by_120"],
            "hdv_exits_before_120": r["hdv_exits_before_120"]} for r in top_states]
    results["top5_states_by_c1_120s"] = top

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items() if v is not None}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, (set, frozenset)):
            return len(obj)
        return obj

    # keep per-HDV dicts out of the JSON (too big / not needed); keep counts + stats
    slim = json.loads(json.dumps(clean(per_state), default=str))
    for row in slim:
        for model in row["models"]:
            for h in list(row["models"][model]):
                if h in ("60", "120"):
                    block = per_state[row["idx"]]["models"][model][h]
                    row["models"][model][h] = {
                        "n_matched": block["n_matched"], "n_wrong_lane": len(block["wrong_lane"]),
                        "ds_med": float(np.median(sorted(block["ds"].values()))) if block["ds"] else None,
                        "mean_signed": float(np.mean(list(block["signed"].values()))) if block["signed"] else None}
                elif h == "presence_mismatch_per_10s_ticks":
                    row["models"][model][h] = per_state[row["idx"]]["models"][model][h]
    results["per_state"] = slim
    path = HERE.parent / "results.json"
    path.write_text(json.dumps(clean(results), indent=1), encoding="utf-8")
    print("wrote", path)
    print(json.dumps(clean({"verify": verify, "state_medians": state_medians, "blowups": blowups}), indent=1))


if __name__ == "__main__":
    main()
