"""J4 wall-pilot MERGE analysis (zero SUMO), law v2.2 design input.

Reads ONLY, per TRAIN world branch under
E:/research-dense/runs/campaign_walls_pilot_chunk4:
  * run/wall_events.jsonl        - request/assembly/hold/release/abort
                                   timestamps, member ids and lane ids;
  * run/branch_record.json       - lifecycle classification (outcome,
                                   requested/achieved hold, abort reason);
  * run/physics_steps.jsonl      - RAW per-0.5 s physics state, used ONLY for
                                   member POSITIONS (road_id + lane_position_m);
                                   no evaluate/suffix/delay/TTC/metric field is
                                   read anywhere.

NO outcome field is read (evaluate_summary, suffix_summary, delay/ttc metrics,
physics outcome summaries).  NO HOLDOUT and NO CALIBRATION world is opened: the
world's split comes from the chunk-4 WALL_PLAN (WORLD_PLAN.json) and only
``TRAIN`` worlds are processed.

Road coordinate ``s_road`` (edge-aware cumulative distance from the entry own
origin 0 m):
  entry 1191554636 (2644.2 m):      s = position on that edge
  merge 1487606179 (75.4 m):        s = 2644.2 + position
  exit  1487606178 (477.9 m):       s = 2719.6 + position
  internal ':' or unknown edge:     s = 2644.2 - distance_to_merge_m
Merge START is s = 2644.2 m; network end is s = 3197.5 m.
"""
from __future__ import annotations

import json
import os
import statistics as stats

RUNS = r"E:\research-dense\runs\campaign_walls_pilot_chunk4"
PLAN = r"E:\research-dense\output\campaign_walls_v1\results_plan\WORLD_PLAN.json"
OUT = os.path.dirname(os.path.abspath(__file__))

ENTRY = "1191554636"
MERGE = "1487606179"
EXIT = "1487606178"
ENTRY_LEN = 2644.2
MERGE_LEN = 75.4
EXIT_LEN = 477.9
MERGE_START_S = ENTRY_LEN                    # 2644.2
MERGE_END_S = ENTRY_LEN + MERGE_LEN          # 2719.6
NET_END_S = ENTRY_LEN + MERGE_LEN + EXIT_LEN  # 3197.5
ROAD_DOWNSTREAM_M = MERGE_LEN + EXIT_LEN     # 553.3

ABORT_REASONS = ("member_left_network", "member_span_exceeded",
                 "non_member_rear_end_risk", "lane_coverage_violation",
                 "assembly_timeout", "member_frame_missing",
                 "member_position_unavailable")


def s_road_from_edge_position(edge, position_m):
    if edge == ENTRY:
        return float(position_m)
    if edge == MERGE:
        return MERGE_START_S + float(position_m)
    if edge == EXIT:
        return MERGE_END_S + float(position_m)
    return None


def s_road_at_request(road_left_m):
    """road_left(request) = NET_END_S - s_road(lead) -> invert exactly."""
    return NET_END_S - float(road_left_m)


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_events(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def positions_at(physics_path, times, members):
    """Nearest physics row (|dt|<=0.75 s) to each requested event time; return
    {time: {member: {"edge":..,"pos":..,"s_road":..,"speed":..}}} for members
    present.  Reads only the raw ``vehicles`` mapping."""
    want = sorted(times)
    near = {}
    best = {t: None for t in want}
    last_seen = {}   # member -> (t, position dict) at or before event time
    with open(physics_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            t = float(row.get("time_s"))
            for event_t in want:
                if t <= event_t + 1e-9:
                    vehicles = row.get("vehicles") or {}
                    for member in members:
                        veh = vehicles.get(member)
                        if veh is None:
                            continue
                        edge = veh.get("road_id")
                        pos = veh.get("lane_position_m")
                        s = (s_road_from_edge_position(edge, pos)
                             if pos is not None else None)
                        last_seen[member] = (t, {
                            "edge": edge, "position_m": pos, "s_road": s,
                            "speed_mps": veh.get("speed_mps")})
                if (best[event_t] is None
                        or abs(t - event_t) < abs(best[event_t][0] - event_t)):
                    best[event_t] = (t, row)
    for event_t in want:
        entry = best.get(event_t)
        snap = {}
        if entry is not None and abs(entry[0] - event_t) <= 0.75:
            vehicles = entry[1].get("vehicles") or {}
            for member in members:
                veh = vehicles.get(member)
                if veh is None:
                    continue
                edge = veh.get("road_id")
                pos = veh.get("lane_position_m")
                s = (s_road_from_edge_position(edge, pos)
                     if pos is not None else None)
                snap[member] = {"edge": edge, "position_m": pos, "s_road": s,
                                "speed_mps": veh.get("speed_mps")}
        for member in members:
            if member in snap:
                continue
            seen = last_seen.get(member)
            if seen is not None and seen[0] <= event_t + 1e-9:
                snap[member] = dict(seen[1], last_seen_at_s=seen[0])
        near[event_t] = snap
    return near


def lead_s_road(snap):
    vals = [v["s_road"] for v in snap.values() if v["s_road"] is not None]
    return max(vals) if vals else None


def rear_s_road(snap):
    vals = [v["s_road"] for v in snap.values() if v["s_road"] is not None]
    return min(vals) if vals else None


def q(vals, p):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return round(vals[0], 2)
    k = (len(vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    frac = k - lo
    return round(vals[lo] * (1 - frac) + vals[hi] * frac, 2)


def describe(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "min": round(min(vals), 2),
            "p25": q(vals, .25), "median": round(stats.median(vals), 2),
            "p75": q(vals, .75), "max": round(max(vals), 2),
            "mean": round(sum(vals) / len(vals), 2)}


def main():
    plan = load_json(PLAN)
    split = {w["world"]: w["split"] for w in plan["worlds"]}
    assert split, "no worlds in plan"
    dirs = sorted(d for d in os.listdir(RUNS)
                  if os.path.isdir(os.path.join(RUNS, d))
                  and ".interrupted" not in d)

    rows = []
    skipped_split = 0
    skipped_no_events = 0
    for d in dirs:
        world = d.split("__")[0]
        if split.get(world) != "TRAIN":
            skipped_split += 1
            continue
        run = os.path.join(RUNS, d, "run")
        ev_path = os.path.join(run, "wall_events.jsonl")
        br_path = os.path.join(run, "branch_record.json")
        if not os.path.exists(ev_path) or not os.path.exists(br_path):
            skipped_no_events += 1
            continue
        events = read_events(ev_path)
        branches = load_json(br_path)
        req = next((e for e in events if e.get("kind") == "wall_request"), None)
        nf = next((e for e in events if e.get("kind") == "not_feasible"), None)
        if req is None:
            rows.append({"run": d, "world": world, "split": "TRAIN",
                         "not_feasible": nf is not None})
            continue
        members = list(req["members"])
        times = {}
        for e in events:
            if e.get("kind") == "wall_event":
                times.setdefault(e["event"], []).append(float(e["timestamp_s"]))
        assembly_start = times.get("assembly_start", [None])[0]
        assembled = times.get("assembled", [None])[0]
        finish = branches.get("aborted_at_s") or branches.get("released_at_s")
        want = [t for t in (req["timestamp_s"], assembly_start, assembled, finish)
                if t is not None]
        snaps = positions_at(os.path.join(run, "physics_steps.jsonl"), want, members)
        req_lead = s_road_at_request(req["road_left_m"])
        asm_lead = lead_s_road(snaps.get(assembly_start, {})) if assembly_start in snaps else None
        asm_rear = rear_s_road(snaps.get(assembly_start, {})) if assembly_start in snaps else None
        fin_lead = lead_s_road(snaps.get(finish, {})) if finish in snaps else None
        fin_rear = rear_s_road(snaps.get(finish, {})) if finish in snaps else None
        asm_snap = snaps.get(assembled, {})
        asm_done_lead = lead_s_road(asm_snap)
        asm_done_rear = rear_s_road(asm_snap)
        rows.append({
            "run": d, "world": world, "split": "TRAIN",
            "not_feasible": False,
            "arm": branches.get("branch_type") or d.split("__")[-1],
            "requested_hold_s": req["hold_s"],
            "outcome": branches.get("outcome"),
            "abort_reason": branches.get("abort_reason"),
            "phase_at_finish": ("hold" if branches.get("assembled_at_s") is not None
                                else "assembly"),
            "wall_speed_mps": req["wall_speed_mps"],
            "mean_member_speed_mps": req["mean_member_speed_mps"],
            "span_m": req["span_m"],
            "road_left_m": req["road_left_m"],
            "request_s": req["timestamp_s"],
            "assembly_start_s": assembly_start,
            "assembled_at_s": assembled,
            "abort_or_release_s": finish,
            "assembly_time_s": (None if assembled is None or assembly_start is None
                                else round(float(assembled) - float(assembly_start), 3)),
            "achieved_hold_s": branches.get("achieved_hold_s"),
            "lead_s_road_request": round(req_lead, 2),
            "distance_to_merge_request": round(MERGE_START_S - req_lead, 2),
            "lead_s_road_assembly_start": (None if asm_lead is None else round(asm_lead, 2)),
            "rear_s_road_assembly_start": (None if asm_rear is None else round(asm_rear, 2)),
            "lead_s_road_assembled": (None if asm_done_lead is None else round(asm_done_lead, 2)),
            "lead_s_road_finish": (None if fin_lead is None else round(fin_lead, 2)),
            "rear_s_road_finish": (None if fin_rear is None else round(fin_rear, 2)),
            "distance_to_merge_finish": (None if fin_lead is None
                                         else round(MERGE_START_S - fin_lead, 2)),
            "held_past_merge_start": bool(fin_lead is not None and fin_lead >= MERGE_START_S - 1e-9),
            "held_into_exit": bool(fin_lead is not None and fin_lead >= MERGE_END_S - 1e-9),
        })

    requested = [r for r in rows if not r["not_feasible"]]
    report = {"n_dirs": len(dirs), "n_train": len(rows),
              "skipped_cal_or_holdout": skipped_split,
              "skipped_no_events": skipped_no_events,
              "n_requested": len(requested),
              "n_not_feasible": len(rows) - len(requested),
              "constants": {"entry_len_m": ENTRY_LEN, "merge_len_m": MERGE_LEN,
                            "exit_len_m": EXIT_LEN, "merge_start_s": MERGE_START_S,
                            "merge_end_s": MERGE_END_S, "net_end_s": NET_END_S}}
    report["s_road_request"] = describe([r["lead_s_road_request"] for r in requested])
    report["distance_to_merge_request"] = describe(
        [r["distance_to_merge_request"] for r in requested])
    report["s_road_assembled"] = describe([r["lead_s_road_assembled"] for r in requested])
    report["s_road_finish"] = describe([r["lead_s_road_finish"] for r in requested])
    report["distance_to_merge_finish"] = describe(
        [r["distance_to_merge_finish"] for r in requested])
    report["wall_speed_mps"] = describe([r["wall_speed_mps"] for r in requested])
    report["assembly_time_s"] = describe([r["assembly_time_s"] for r in requested])
    report["span_m"] = describe([r["span_m"] for r in requested])
    report["n_held_past_merge_start"] = sum(r["held_past_merge_start"] for r in requested)
    report["n_held_into_exit"] = sum(r["held_into_exit"] for r in requested)
    report["outcomes"] = {}
    for r in requested:
        report["outcomes"][r["outcome"]] = report["outcomes"].get(r["outcome"], 0) + 1
    report["abort_reasons"] = {}
    for r in requested:
        if r["abort_reason"]:
            key = r["abort_reason"].split(":")[0]
            report["abort_reasons"][key] = report["abort_reasons"].get(key, 0) + 1

    by_hold = {}
    for hold in (60.0, 120.0, 240.0):
        subset = [r for r in requested if float(r["requested_hold_s"]) == hold]
        by_hold[int(hold)] = {
            "n": len(subset),
            "s_road_request": describe([r["lead_s_road_request"] for r in subset]),
            "s_road_finish": describe([r["lead_s_road_finish"] for r in subset]),
            "distance_to_merge_request": describe(
                [r["distance_to_merge_request"] for r in subset]),
            "wall_speed_mps": describe([r["wall_speed_mps"] for r in subset]),
            "n_held_past_merge_start": sum(r["held_past_merge_start"] for r in subset),
            "n_held_into_exit": sum(r["held_into_exit"] for r in subset),
        }
    report["by_hold"] = by_hold

    abort_detail = {}
    for reason in ABORT_REASONS:
        subset = [r for r in requested
                  if (r["abort_reason"] or "").split(":")[0] == reason]
        if subset:
            abort_detail[reason] = {
                "n": len(subset),
                "s_road_finish": describe([r["lead_s_road_finish"] for r in subset]),
                "s_road_request": describe([r["lead_s_road_request"] for r in subset]),
                "wall_speed_mps": describe([r["wall_speed_mps"] for r in subset]),
            }
    report["abort_detail"] = abort_detail
    report["n_finish_s_road_missing"] = sum(
        1 for r in requested if r["lead_s_road_finish"] is None)

    # ---- a-priori D_min proposer screen (law v2.2) -----------------------
    # D_min = X + distance to assemble (v_wall * measured J4 assembly time)
    #         + v_wall * H_min.  X = 0 (owner binding); H_min = 60 s (the
    # smallest registered fixed-arm hold draw).  No outcome is used.
    X_M = 0.0
    H_MIN = 60.0
    feas = []
    for r in requested:
        v = r["wall_speed_mps"]
        asm = r["assembly_time_s"]
        if asm is None:
            continue
        d_min = X_M + v * asm + v * H_MIN
        feas.append((d_min, r["distance_to_merge_request"],
                     r["distance_to_merge_request"] >= d_min))
    report["dmin"] = {
        "X_m": X_M, "H_min_s": H_MIN, "n": len(feas),
        "n_feasible": sum(1 for _, _, ok in feas if ok),
        "dmin_m": describe([d for d, _, _ in feas]),
        "margin_m": describe([dist - d for d, dist, _ in feas]),
    }

    with open(os.path.join(OUT, "j4_merge_analysis.json"), "w", encoding="utf-8") as fh:
        json.dump({"report": report, "rows": rows}, fh, indent=2)
        fh.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
