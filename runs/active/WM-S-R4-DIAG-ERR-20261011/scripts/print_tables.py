import json

d = json.load(open(r"E:/research-dense/runs/active/WM-S-R4-DIAG-ERR-20261011/results.json", encoding="utf-8"))
for m in ("C1", "R4"):
    for h in ("60", "120"):
        a = d["attribution"][m][h]
        print(f"== {m} {h}s: pooled n={a['n_pooled']} mean|ds|={a['mean_abs_ds']:.2f} pooled med={a['pooled_median']:.2f}")
        for g, st in a["groups"].items():
            def fmt(key, width, scale=1.0):
                v = st.get(key)
                return " " * width if not isinstance(v, (int, float)) else f"{v*scale:{width}.2f}"
            print(f"  {g:14s} n={st['n']:5d} share={st.get('share', 0)*100:5.1f}% med|ds|={fmt('median', 7)} "
                  f"mean|ds|={fmt('mean', 8)} contrib_mean={fmt('contrib_to_mean', 7)} "
                  f"signed_mean={fmt('signed_mean', 7)} dv_mean={fmt('dv_mean', 6)}")
print()
for m in ("C1", "R4"):
    for h in ("60", "120"):
        z = d["attribution"][m][h]["zones"]
        row = {k: (v["n"], None if v.get("median") is None else round(v["median"], 1),
                   round(v["contrib_to_mean"], 1)) for k, v in z.items()}
        print(f"-- zones {m} {h}s (n, med|ds|, contrib_to_mean): {row}")
print()
print("paired gap R4-C1:")
print(json.dumps(d["paired_gap_r4_minus_c1"], indent=1))
print()
print("entrants/exits per state:")
print(json.dumps(d["entrants_per_state"], indent=1))
print()
print("time profile:")
for m in ("C1", "R4"):
    print(m)
    for t in [str(x) for x in range(10, 121, 10)]:
        p = d["time_profile_10s"][m].get(t) or {}
        def g(key):
            v = p.get(key)
            return f"{v:8.2f}" if isinstance(v, float) else "    -   "
        wl = p.get("wrong_lane_share")
        wl = f"{wl*100:5.1f}%" if isinstance(wl, float) else "  -  "
        print(f"  {t:>4s}s E120med={g('E120_median')} N120med={g('N120_median')} "
              f"allmed={g('all_median')} wronglane={wl}")
print()
print("top5:", json.dumps(d["top5_states_by_c1_120s"], indent=1))
print("presence mismatches:", [(r["key"], {m: r["models"][m]["presence_mismatch_per_10s_ticks"] for m in r["models"]})
                               for r in d["per_state"] if any(r["models"][m]["presence_mismatch_per_10s_ticks"] for m in r["models"])][:5])
print("n states with any mismatch:", sum(1 for r in d["per_state"]
      if any(r["models"][m]["presence_mismatch_per_10s_ticks"] for m in r["models"])))
