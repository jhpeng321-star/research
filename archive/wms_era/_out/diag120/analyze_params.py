"""Task 4: OLDHIST recovered IDM params of wave starters vs non-starters + stability.

Wave starter = a vehicle whose HDV |ds| crosses 10 m without a (recorded-lane)
leader crossing first; propagator = crosses after its leader; non-starter = never
crosses.  IDM low-speed string-stability condition (Treiber & Kesting 2013, as
reviewed in arXiv:2506.05909 Eq. 4): a >= s0 / T^2  (a = max accel, s0 = min gap,
T = desired time headway).  This is the low-speed sufficient condition.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent

PN = ("v0", "time_headway", "min_gap", "accel", "decel", "politeness",
      "mobil_threshold", "merge_gap", "merge_urgency")


def load_jsonl(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> int:
    params = {}
    for fn in ("params_worst.json", "params_nonblow.json"):
        params.update(json.loads((D / fn).read_text()))
    rows = {r["world"]: r for r in load_jsonl(D / "roll_oldhist_u_same.jsonl")
            if "error" not in r}

    classes: dict[str, list[dict]] = {"starter": [], "propagator": [], "non": []}
    for world, blob in params.items():
        if "error" in blob:
            continue
        rr = rows.get(world)
        if rr is None:
            continue
        veh = {v["id"]: v for v in rr["vehicles"]}
        for k, vid in enumerate(blob["ids"]):
            onset = veh.get(vid, {}).get("onset_s")
            lo = veh.get(vid, {}).get("leader_onset_s")
            if onset is None:
                cls = "non"
            elif lo is None or onset <= lo:
                cls = "starter"
            else:
                cls = "propagator"
            vals = {name: blob["params"][name][k] for name in PN}
            T = max(vals["time_headway"], 1e-3)
            margin = vals["accel"] - vals["min_gap"] / (T * T)
            classes[cls].append(vals | {"margin": margin, "stable": margin >= 0.0})

    def stats(name, key):
        v = np.array([x[key] for x in classes[name]])
        if v.size == 0:
            return None
        return {"n": int(v.size), "median": float(np.median(v)),
                "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}

    out = {"condition": "a >= s0 / T^2 (IDM low-speed string stability)",
           "n_by_class": {c: len(classes[c]) for c in classes},
           "params": {}, "stability": {}}
    for c in classes:
        out["params"][c] = {k: stats(c, k) for k in
                            ("time_headway", "accel", "decel", "min_gap", "v0")}
        arr = classes[c]
        st = np.array([x["stable"] for x in arr]) if arr else np.array([])
        mg = np.array([x["margin"] for x in arr]) if arr else np.array([])
        out["stability"][c] = {"n": len(arr),
                               "frac_stable": float(st.mean()) if st.size else None,
                               "median_margin": float(np.median(mg)) if mg.size else None}
    (D / "params_analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
