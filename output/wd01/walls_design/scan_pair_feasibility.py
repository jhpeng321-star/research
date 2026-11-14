"""WD-01 wall design scan (zero SUMO, TRAIN only): member-pair feasibility
under the current choose_pair rule vs the D1 upstream/min-span rule, from
chunk-4 TRAIN <world>__native physics rows (true native, same saved state the
wall branches restore). Descriptive only; informs the owner O1/O2 decision."""
import json, sys
from pathlib import Path
PLAN = Path("E:/research-dense/output/campaign_stage1_v3/results_plan_chunk4/WORLD_PLAN.json")
RUNS = Path("E:/research-dense/runs/campaign_stage1_chunk4")
UP, MERGE, EXIT = "1191554636", "1487606179", "1487606178"
L_UP, L_MERGE, L_EXIT = 2644.2, 75.4, 477.9
MERGE_ENTRY_D = L_UP  # downstream coordinate of merge-edge start
def s_down(road, pos):
    if road == UP: return pos
    if road == EXIT: return L_UP + L_MERGE + pos
    return None
def road_left(sd): return (L_UP + L_MERGE + L_EXIT) - sd
def scan_tick(veh, cavs, allowed_edges, rule):
    by_lane = {0: [], 1: []}
    for v in cavs:
        st = veh.get(v)
        if not st or st["road_id"] not in allowed_edges: continue
        ln = int(st["lane_id"].rsplit("_", 1)[1])
        if ln not in by_lane: continue
        by_lane[ln].append((s_down(st["road_id"], st["lane_position_m"]), st["speed_mps"], st["road_id"]))
    best = None
    for a in by_lane[0]:
        for b in by_lane[1]:
            if a[2] != b[2]: continue          # same edge (package rule)
            span = abs(a[0] - b[0])
            if span > 75.0: continue
            if rule == "current":              # nearest the merge = most downstream mean
                key = (-(a[0] + b[0]) / 2, span)
            else:                              # D1: minimal span, then most road left
                key = (span, -road_left(max(a[0], b[0])))
            if best is None or key < best[0]:
                best = (key, span, a, b)
    return best
def main():
    plan = json.loads(PLAN.read_text())
    out = []
    for w in plan["worlds"]:
        if w["split"] != "TRAIN" or float(w["penetration"]) not in (0.1, 0.2, 0.4): continue
        t0 = float(w["t0_s"]); run = RUNS / f"{w['world']}__native" / "run" / "physics_steps.jsonl"
        for timing, off in (("early", 0.0), ("late", 60.0)):
            tgt = t0 + off; res = {}
            with open(run, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line); t = r["time_s"]
                    if t < tgt or abs(t - round(t)) > 1e-9: continue
                    if t > tgt + 120: break
                    cavs = set(r["cav_ids"]); veh = r["vehicles"]
                    for rule, edges in (("current", {UP, EXIT}), ("d1", {UP})):
                        if rule in res: continue
                        b = scan_tick(veh, cavs, edges, rule)
                        if b:
                            front = max(b[2][0], b[3][0])
                            vmean = (b[2][1] + b[3][1]) / 2
                            res[rule] = dict(t_found=t - tgt, span=round(b[1], 1), edge=b[2][2],
                                             road_left=round(road_left(front), 1), vmean=round(vmean, 2))
                    if len(res) == 2: break
            out.append(dict(world=w["world"], pen=w["penetration"], timing=timing, **{k: res.get(k) for k in ("current", "d1")}))
    Path(sys.argv[1]).write_text("\n".join(json.dumps(o) for o in out), encoding="utf-8")
    print("rows", len(out))
if __name__ == "__main__":
    main()
