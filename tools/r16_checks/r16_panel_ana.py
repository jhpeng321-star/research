"""S-RAMPSPEED offline analysis of roll_r16.py output (panel, tag ref): D2 merge timing + R14 metrics.

D2 (``tools/r14_checks/d2_merge_timing.py`` event definitions, unchanged): HDV and CAV ramp merges
lane 2 -> 0/1 inside the zone; per arm the pooled statistic with a 1,000-draw STATE (cluster)
bootstrap 95 % CI, and the paired R16 - R14 difference of the pooled statistic under the same
resampled states (common random numbers: same states, seeds and lane draws in both arms):
  dist_med         median distance before the lane end at the merge (m)
  late_share       share of merges in the last 15 m (``share_dist_lt15``)
  standstill_share share of merges at v < 0.5 m/s
  v_med            median merge speed;  t_seen_med  first seen on the ramp -> merge (s)
  t_zone_med       zone entry -> merge (s);  merges   merges per state
Delay / safety (``ana_r14.metrics``: dly_ramp, dly_merge, dly_up_near, merges_pm, spill, ttc, overlap,
hb_share, stuck, lc10 ...) with ana_r14's paired cluster bootstrap.

  python r16_panel_ana.py DIR --horizon 120 [--arms R14,R16] [--out DIR/R16_PANEL.json]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r14_checks"))
import ana_r14  # noqa: E402
import d2_merge_timing as D2  # noqa: E402

STATS = ("dist_med", "late_share", "standstill_share", "v_med", "t_seen_med", "t_zone_med", "merges")


def pooled(events, n_states):
    if not events:
        return {k: np.nan for k in STATS}
    a = {k: np.array([e[k] for e in events], float) for k in ("dist", "v", "t_seen", "t_zone")}

    def med(x):
        x = x[np.isfinite(x)]
        return float(np.median(x)) if len(x) else np.nan
    return {"dist_med": med(a["dist"]), "late_share": float((a["dist"] < 15).mean()),
            "standstill_share": float((a["v"] < 0.5).mean()), "v_med": med(a["v"]),
            "t_seen_med": med(a["t_seen"]), "t_zone_med": med(a["t_zone"]),
            "merges": len(events) / max(1, n_states)}


def d2_analysis(d, arms, rng, n=1000):
    files = ana_r14.collect(d, arms, ["ref"])
    ev = {}
    for (p, _t, arm), f in files.items():
        e, _rows = D2.analyse(*D2.load(f))
        ev[(p, arm)] = e
    states = sorted({p for p, _ in ev if all((p, a) in ev for a in ["sumo", *arms])})
    out = {"states": len(states), "absolute": {}, "paired": {}}
    draws = [rng.choice(states, len(states)) for _ in range(n)]
    for arm in ["sumo", *arms]:
        point = pooled([x for p in states for x in ev[(p, arm)]], len(states))
        boots = [pooled([x for p in s for x in ev[(p, arm)]], len(s)) for s in draws]
        out["absolute"][arm] = {k: [point[k], *np.nanpercentile([b[k] for b in boots], [2.5, 97.5]).tolist()]
                                for k in STATS}
    ref = arms[0]
    for arm in arms[1:]:
        pa = pooled([x for p in states for x in ev[(p, arm)]], len(states))
        pr = pooled([x for p in states for x in ev[(p, ref)]], len(states))
        diffs = []
        for s in draws:
            ba = pooled([x for p in s for x in ev[(p, arm)]], len(s))
            br = pooled([x for p in s for x in ev[(p, ref)]], len(s))
            diffs.append({k: ba[k] - br[k] for k in STATS})
        out["paired"][f"{arm}-{ref}"] = {k: [pa[k] - pr[k], *np.nanpercentile([x[k] for x in diffs],
                                                                              [2.5, 97.5]).tolist()]
                                         for k in STATS}
    return out


def main():
    d = sys.argv[1]
    H = float(sys.argv[sys.argv.index("--horizon") + 1]) if "--horizon" in sys.argv else 120.0
    arms = sys.argv[sys.argv.index("--arms") + 1].split(",") if "--arms" in sys.argv else ["R14", "R16"]
    out_path = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(d, "R16_PANEL.json")
    rng = np.random.default_rng(16)
    doc = {"d2": d2_analysis(d, arms, rng),
           "r14_metrics": ana_r14.analyze(d, H, arms, ["ref"], arms[0])}
    doc["r14_metrics"].pop("per_run", None)
    doc = ana_r14.clean(doc)
    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=1)
    fmt = lambda v, s=False: "n/a" if v[0] is None else (  # noqa: E731
        ("{:+.3g} [{:+.3g},{:+.3g}]" if s else "{:.3g} [{:.3g},{:.3g}]").format(*v[:3]))
    cols = ["sumo", *arms]
    pk = list(doc["d2"]["paired"])
    print(f"D2 ({doc['d2']['states']} states)\n| metric | " + " | ".join(cols + pk) + " |")
    for k in STATS:
        print(f"| {k} | " + " | ".join([fmt(doc["d2"]["absolute"][a][k]) for a in cols]
                                       + [fmt(doc["d2"]["paired"][c][k], True) for c in pk]) + " |")
    m = doc["r14_metrics"]
    print("\n| metric | " + " | ".join(cols + list(m["paired"])) + " |")
    for k in m["absolute"]["sumo"]:
        print(f"| {k} | " + " | ".join([fmt(m["absolute"][a][k]) for a in cols]
                                       + [fmt(m["paired"][c][k], True) for c in m["paired"]]) + " |")


if __name__ == "__main__":
    main()
