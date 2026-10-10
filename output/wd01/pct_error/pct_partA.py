"""Part A error as % of the true state (zero-sim, read-only; PA2G-A2 TRAIN raw rows).
Delay: MAE / signed mean as % of mean true tracked delay (exact, from rows).
Speed/distance: truth mean speed over vehicles present at window start w, averaged over
0.5 s ticks in (w, w+H] while still present (approximation of the tracked cohort);
mean distance travelled over H = that mean speed x H."""
import json, collections, statistics as st
from pathlib import Path
RAW = Path(__file__).resolve().parent.parent / "partA_v2g/results/eval_a2/raw_rows.jsonl"
RUNS = Path("E:/research-dense/runs/campaign_stage1_chunk4")
rows = [json.loads(l) for l in open(RAW, encoding="utf-8")]
rows = [r for r in rows if r["model"] in ("v2g", "const_speed", "v1_p3o_replica")]
need = collections.defaultdict(set)
for r in rows: need[r["world"]].add((r["w"], r["H"]))
vbar = {}
for world, keys in need.items():
    ws = sorted({w for w, _ in keys}); Hmax = max(H for _, H in keys)
    series = {}
    with open(RUNS / f"{world}__native_base/run/physics_steps.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line); t = r["time_s"]
            if t < ws[0] or t > ws[-1] + Hmax: continue
            series[round(t, 1)] = {k: v["speed_mps"] for k, v in r["vehicles"].items()}
    for (w, H) in keys:
        at_w = set(series.get(round(w, 1), {}))
        sp = []
        k = 1
        while True:
            t = round(w + 0.5 * k, 1)
            if t > w + H + 1e-9: break
            sp += [v for vid, v in series.get(t, {}).items() if vid in at_w]
            k += 1
        vbar[(world, w, H)] = st.mean(sp) if sp else float("nan")
out = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    key = (r["model"], r["H"]); v = vbar[(r["world"], r["w"], r["H"])]
    o = out[key]
    o["s"].append(r["s_rmse"]); o["v"].append(r["v_rmse"]); o["vbar"].append(v)
    o["dist"].append(v * r["H"]); o["err"].append(r["delay_err"]); o["dtrue"].append(r["delay_true_tracked"])
print("model|H  n  s_RMSE/meanDist%  v_RMSE/meanV%  delayMAE%  delayBias%  (meanV m/s, meanDist m, meanTrueDelay veh-s)")
for (m, H) in sorted(out, key=lambda k: (k[0], k[1])):
    o = out[(m, H)]; md = st.mean(o["dist"]); mv = st.mean(o["vbar"]); dt = st.mean(o["dtrue"])
    mae = st.mean(abs(e) for e in o["err"]); bias = st.mean(o["err"])
    print(f"{m}|H{H} {len(o['s'])} {100*st.mean(o['s'])/md:.1f} {100*st.mean(o['v'])/mv:.1f} {100*mae/dt:.1f} {100*bias/dt:+.1f}  ({mv:.2f}, {md:.0f}, {dt:.0f})")
