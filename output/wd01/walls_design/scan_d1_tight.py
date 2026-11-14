"""Companion to scan_pair_feasibility.py: under D1 (upstream edge, min span),
the first tick within the 120 s request window at which a pair with span
<= 6 m (assembly tolerance) AND road_left >= X exists. TRAIN only, zero SUMO."""
import json, sys
from pathlib import Path
import scan_pair_feasibility as S
def main(X):
    plan = json.loads(S.PLAN.read_text()); out = []
    for w in plan["worlds"]:
        if w["split"] != "TRAIN" or float(w["penetration"]) not in (0.1, 0.2, 0.4): continue
        t0 = float(w["t0_s"]); run = S.RUNS / f"{w['world']}__native" / "run" / "physics_steps.jsonl"
        for timing, off in (("early", 0.0), ("late", 60.0)):
            tgt = t0 + off; hit = None
            with open(run, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line); t = r["time_s"]
                    if t < tgt or abs(t - round(t)) > 1e-9: continue
                    if t > tgt + 120: break
                    veh = r["vehicles"]; cav = set(r["cav_ids"])
                    lanes = {0: [], 1: []}
                    for v in cav:
                        st = veh.get(v)
                        if st and st["road_id"] == S.UP:
                            ln = int(st["lane_id"].rsplit("_", 1)[1])
                            if ln in lanes: lanes[ln].append(st["lane_position_m"])
                    ok = [(abs(a - b), S.road_left(max(a, b))) for a in lanes[0] for b in lanes[1]
                          if abs(a - b) <= 6.0 and S.road_left(max(a, b)) >= X]
                    if ok: hit = t - tgt; break
            out.append(dict(world=w["world"], pen=w["penetration"], timing=timing, t_hit=hit))
    n = len(out); h = [o for o in out if o["t_hit"] is not None]
    print(f"X={X}: tight pair (<=6 m, road_left>=X) within 120 s: {len(h)}/{n}")
    for pen in (0.1, 0.2, 0.4):
        g = [o for o in out if o["pen"] == pen]; print("  pen", pen, sum(o["t_hit"] is not None for o in g), "/", len(g))
for X in [float(x) for x in sys.argv[1:]]:
    main(X)
