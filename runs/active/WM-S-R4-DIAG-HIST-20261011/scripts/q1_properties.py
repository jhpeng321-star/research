"""Q1: are real-history states intrinsically harder? State properties, paired by world.

Compares quick-20 of wallc1_states_h60_real vs wallc1_states_h60 (15 shared worlds,
no identical (world, t0) anywhere), plus the full-set context and the per-state C1/R4
errors (round-4 cache) on both quick sets. Read-only; writes out/q1_properties.json.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_lib2 import (DIAG, OUT, R4_SCORE, SETS, dump_json, ensure_env, iter_records,
                       load_json, q, quick20, true_lane_changes)

ensure_env()
import numpy as np  # noqa: E402


def state_properties(record: dict) -> dict:
    init = record.get("initial_state") or {}
    veh = [v for v in init.get("vehicles", []) if v.get("present", True)]
    hdv = [v for v in veh if not v.get("is_cav", False)]
    cav = [v for v in veh if v.get("is_cav", False)]
    speeds = np.array([float(v.get("v", 0.0)) for v in hdv], float)
    lanes = [int(v.get("lane", -1)) for v in hdv]
    svals = np.array([float(v.get("s", 0.0)) for v in veh], float)
    span = float(svals.max() - svals.min()) if svals.size else 0.0
    ramp = sum(1 for v in hdv if str(v.get("id", "")).startswith("ramp"))
    hist = record.get("history") or {}
    roster_span = [float(x) for x in (hist.get("roster_span_s") or {}).values()]

    from wms.eval.normalize import recorded_for
    rec = recorded_for(record, "wall")
    ids0 = {str(v.get("id")) for v in veh}
    seen, exited, ent_hdv, ent_cav = set(), set(), 0, 0
    last_n, slow_share, junction_share = [], [], []
    cav_speeds_0 = [float(v.get("v", 0.0)) for v in cav]
    lc = None
    ids_all = sorted(ids0 | {str(r["id"]) for fr in rec for r in fr.get("vehicles", [])})
    col = {v: i for i, v in enumerate(ids_all)}
    n = len(ids_all)
    lane = np.full((241, n), -9, np.int8)
    pres = np.zeros((241, n), bool)
    cavm = np.zeros((241, n), bool)
    vv = np.full((241, n), np.nan, np.float32)
    for fr in rec:
        k = int(round((float(fr["time_s"]) - float(record["t0_s"])) / 0.5))
        if not (0 <= k <= 240):
            continue
        for r in fr.get("vehicles", []):
            if not r.get("present", True):
                continue
            j = col.get(str(r["id"]))
            if j is None:
                continue
            lane[k, j] = int(r.get("lane", -1))
            pres[k, j] = True
            cavm[k, j] = bool(r.get("is_cav", False))
            vv[k, j] = float(r.get("v", 0.0))
            if k > 0 and str(r["id"]) not in ids0:
                seen.add(str(r["id"]))
    for vid in ids0:
        j = col[vid]
        if not pres[1:241, j].any():
            exited.add(vid)
    for vid in seen:
        j = col[vid]
        if cavm[:, j].any():
            ent_cav += 1
        else:
            ent_hdv += 1
    lc = true_lane_changes(lane, pres, cavm, ids_all)
    for k in (60, 120, 240):
        m = pres[k] & ~cavm[k]
        last_n.append(int(m.sum()))
        slow_share.append(float((vv[k][m] < 5.0).mean()) if m.any() else None)
        junction_share.append(float((lane[k][m] < 0).mean()) if m.any() else None)
    m0 = pres[0] & ~cavm[0]
    return {
        "world": str(record.get("world")), "t0_s": float(record.get("t0_s", 0.0)),
        "key": str(record.get("key")),
        "t0_rel_origin_s": float(record.get("t0_rel_origin_s", 0.0) or 0.0),
        "t0_tick_rel_start": int(record.get("t0_tick_rel_start", 0) or 0),
        "assembled_rel_s": record.get("assembled_rel_s"),
        "wall_phase_at_t0": record.get("wall_phase_at_t0"),
        "wall_engaged": record.get("wall_engaged"),
        "n_hdv0": len(hdv), "n_cav0": len(cav),
        "hdv_speed_mean": float(speeds.mean()) if speeds.size else None,
        "hdv_speed_median": float(np.median(speeds)) if speeds.size else None,
        "hdv_speed_p10": float(np.percentile(speeds, 10)) if speeds.size else None,
        "share_hdv_below5": float((speeds < 5.0).mean()) if speeds.size else None,
        "share_hdv_below2": float((speeds < 2.0).mean()) if speeds.size else None,
        "share_lane012": [float(np.mean([l == i for l in lanes])) if lanes else None
                          for i in (0, 1, 2)],
        "share_junction_lane0": float(np.mean([l < 0 for l in lanes])) if lanes else None,
        "ramp_hdv0": ramp, "ramp_share": (ramp / len(hdv)) if hdv else 0.0,
        "road_span_m": span,
        "density_veh_per_km_span": (len(veh) / span * 1000.0) if span > 1 else None,
        "cav_speed_mean0": float(np.mean(cav_speeds_0)) if cav_speeds_0 else None,
        "hist_roster_full": hist.get("n_roster_full_history"),
        "hist_roster_mean_span_s": float(np.mean(roster_span)) if roster_span else None,
        "entrants_hdv": ent_hdv, "entrants_cav": ent_cav, "exits_t0": len(exited),
        "n_hdv_at": last_n, "share_hdv_below5_at": slow_share,
        "share_junction_at": junction_share,
        "rec_lane_changes": lc["changes"], "rec_merges_2to1": lc["merges_2to1"],
    }


def cached_state_value(cache: Path, world: str, t0: float, horizon: str = "60.0"):
    d = load_json(cache)
    for s in d["states"]:
        if str(s["world"]) == world and abs(float(s["t0_s"]) - t0) < 1e-6:
            block = ((s.get("arms") or {}).get("wall") or {})
            if block.get("error"):
                return None
            value = block["car"][horizon]["hdv"]["ds_med"]
            return float(value) if value is not None else None
    return None


def main() -> None:
    t_start = time.time()
    real = quick20(SETS["real_q20"])
    std = quick20(SETS["std_q20"])
    rows = {"real_q20": [state_properties(r) for r in real],
            "std_q20": [state_properties(r) for r in std]}
    # attach cached errors
    caches = {"C1": R4_SCORE / "cache" / "C1__wall_same_real.json",
              "R4": R4_SCORE / "cache" / "R4-last__wall_same_real.json"}
    for row in rows["real_q20"]:
        for name, path in caches.items():
            row[f"ds60_{name}"] = cached_state_value(path, row["world"], row["t0_s"])
        row["ds120_R4"] = cached_state_value(caches["R4"], row["world"], row["t0_s"], "120.0")
    caches_std = {"C1": R4_SCORE / "cache" / "C1__wall_same.json",
                  "R4": R4_SCORE / "cache" / "R4-last__wall_same.json"}
    for row in rows["std_q20"]:
        for name, path in caches_std.items():
            row[f"ds60_{name}"] = cached_state_value(path, row["world"], row["t0_s"])

    def summarize(rows: list[dict]) -> dict:
        out = {}
        for k in rows[0]:
            vals = [r.get(k) for r in rows]
            if not any(isinstance(v, (int, float)) for v in vals):
                out[k] = None
                continue
            if any(isinstance(v, list) for v in vals):
                for idx in range(3):
                    sub = [v[idx] if isinstance(v, list) and v[idx] is not None else None
                           for v in vals]
                    if any(s is not None for s in sub):
                        out[f"{k}_{(idx + 1) * 30}s"] = q([s for s in sub if s is not None])
            else:
                out[k] = q([v for v in vals if v is not None])
        return out

    summary = {name: summarize(rs) for name, rs in rows.items()}

    # paired by world (15 shared worlds): property real - std and C1 error real - std
    std_by_world = {r["world"]: r for r in rows["std_q20"]}
    pairs = []
    for r in rows["real_q20"]:
        s = std_by_world.get(r["world"])
        if s is None:
            continue
        entry = {"world": r["world"]}
        for k in ("t0_rel_origin_s", "n_hdv0", "hdv_speed_mean", "share_hdv_below5",
                  "ramp_hdv0", "entrants_hdv", "rec_lane_changes", "rec_merges_2to1",
                  "ds60_C1", "ds60_R4"):
            a, b = r.get(k), s.get(k)
            entry[k] = (a - b) if (isinstance(a, (int, float)) and isinstance(b, (int, float))
                                   and a is not None and b is not None) else None
        pairs.append(entry)
    paired_summary = {}
    for k in pairs[0]:
        if k == "world":
            continue
        vals = [p[k] for p in pairs if p.get(k) is not None]
        paired_summary[k] = {"n": len(vals), **q(vals)} if vals else {"n": 0}

    # full-set context (properties only; cheap)
    rows["real_full"] = [state_properties(r) for r in iter_records(SETS["real_full"])]
    rows["std_full"] = [state_properties(r) for r in iter_records(SETS["std_full"])]
    summary["real_full"] = summarize(rows["real_full"])
    summary["std_full"] = summarize(rows["std_full"])

    out = {"n_states": {k: len(v) for k, v in rows.items()},
           "shared_worlds_q20": len(pairs),
           "identical_world_t0_pairs": 0,
           "summary": summary,
           "paired_real_minus_std_by_world": paired_summary,
           "rows": rows}
    dump_json(out, OUT / "q1_properties.json")
    print(f"q1 done in {time.time() - t_start:.0f}s; shared worlds {len(pairs)}",
          flush=True)


if __name__ == "__main__":
    main()
