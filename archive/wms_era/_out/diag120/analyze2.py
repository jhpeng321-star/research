"""Refined Task 1/2/3 analysis + Task 5 summary -> analysis2.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
import diaglib as DL  # noqa: E402


def load_jsonl(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def series(state, key):
    return np.array([np.nan if v is None else v for v in state[key]], dtype=float)


def sec_series(state, which, sect, metric):
    key = "sec_pred" if which == "pred" else "sec_rec"
    out = np.full(len(state["t"]), np.nan)
    for i in range(len(state["t"])):
        row = state[key].get(str(i), {}).get(sect)
        if row and row.get(metric) is not None:
            out[i] = row[metric]
    return out


def mechanism(state):
    """Signed congestion mechanism for one state."""
    n = len(state["t"])
    sp_stop = sec_series(state, "pred", "merge", "stopped")
    sr_stop = sec_series(state, "rec", "merge", "stopped")
    sp_spd = sec_series(state, "pred", "merge", "mean_speed")
    sr_spd = sec_series(state, "rec", "merge", "mean_speed")
    up_sp_stop = sec_series(state, "pred", "up", "stopped")
    up_sr_stop = sec_series(state, "rec", "up", "stopped")
    m = ~(np.isnan(sp_spd) | np.isnan(sr_spd))
    speed_bias = float(np.nanmean((sp_spd - sr_spd)[m])) if m.any() else 0.0
    missed = float(np.nanmax(sr_stop - sp_stop)) if n else 0.0
    phantom = float(np.nanmax(sp_stop - sr_stop)) if n else 0.0
    missed_up = float(np.nanmax(up_sr_stop - up_sp_stop)) if n else 0.0
    phantom_up = float(np.nanmax(up_sp_stop - up_sr_stop)) if n else 0.0
    missed = max(missed, missed_up)
    phantom = max(phantom, phantom_up)
    if phantom >= 5 and phantom > missed + 2:
        cls = "phantom"
    elif missed >= 3 and missed >= phantom and speed_bias > 0.3:
        cls = "missed"
    elif speed_bias > 0.5:
        cls = "fast-drift"
    else:
        cls = "drift"
    return {"speed_bias": speed_bias, "missed_peak": missed, "phantom_peak": phantom,
            "cls": cls}


def jump_moment(state):
    ds = series(state, "ds_med")
    ds = np.where(np.isnan(ds), 0.0, ds)
    n = len(ds)
    hi = min(n, 241)
    rise = ds[10:hi] - ds[:hi - 10]
    if rise.size == 0:
        return None
    j = int(np.argmax(rise)) + 5
    # recorded braking signal around j
    sr_stop = sec_series(state, "rec", "merge", "stopped")
    sr_spd = sec_series(state, "rec", "merge", "mean_speed")
    lo, humid = max(0, j - 3), min(n, j + 8)
    stop_up = np.nanmax(sr_stop[lo:humid]) - np.nanmin(sr_stop[lo:humid]) if humid > lo else 0
    spd_drop = np.nanmax(sr_spd[lo:humid]) - np.nanmin(sr_spd[lo:humid]) if humid > lo else 0
    sp_stop = sec_series(state, "pred", "merge", "stopped")
    missed_here = float(np.nanmax(sr_stop[lo:humid]) - np.nanmax(sp_stop[lo:humid]))
    return {"jump_t_s": float(j * 0.5), "rise_5s": float(rise[j - 5]),
            "rec_stop_increase": float(stop_up) if np.isfinite(stop_up) else 0.0,
            "rec_speed_drop": float(spd_drop) if np.isfinite(spd_drop) else 0.0,
            "missed_here": missed_here,
            "coincides_missed_braking": bool((stop_up >= 3) or (spd_drop >= 2.0))}


def main():
    out = {"task1": {}, "task2": {}, "task5": {}}
    for model, fn in (("c1_struct", "roll_struct_u_same.jsonl"),
                      ("oldhist", "roll_oldhist_u_same.jsonl")):
        rows = [r for r in load_jsonl(D / fn) if "error" not in r]
        # Task1
        t1 = {}
        for sel, lab in ((True, "blow"), (False, "nonblow")):
            sub = [r for r in rows if r["blow"] == sel]
            mechs = [mechanism(r) for r in sub]
            from collections import Counter
            cnt = Counter(m["cls"] for m in mechs)
            ds120 = [r["ds120"] for r in sub if r["ds120"] is not None]
            t1[lab] = {"n": len(sub), "counts": dict(cnt),
                       "mean_speed_bias": float(np.mean([m["speed_bias"] for m in mechs])),
                       "mean_missed_peak": float(np.mean([m["missed_peak"] for m in mechs])),
                       "mean_phantom_peak": float(np.mean([m["phantom_peak"] for m in mechs])),
                       "mean_ds120": float(np.mean(ds120)) if ds120 else None}
        # correlation speed_bias vs ds120 across blow states
        b = [r for r in rows if r["blow"] and r["ds120"] is not None]
        sb = np.array([mechanism(r)["speed_bias"] for r in b])
        d120 = np.array([r["ds120"] for r in b])
        t1["corr_bias_ds120_blow"] = float(np.corrcoef(sb, d120)[0, 1]) if len(b) > 2 else None
        out["task1"][model] = t1
        # Task2 jump timing
        jm = [jump_moment(r) for r in rows if r["blow"]]
        jm = [j for j in jm if j]
        out["task2"][model] = {
            "n": len(jm),
            "median_jump_t": float(np.median([j["jump_t_s"] for j in jm])) if jm else None,
            "median_rise_5s": float(np.median([j["rise_5s"] for j in jm])) if jm else None,
            "coincides_missed_braking_frac": float(np.mean(
                [j["coincides_missed_braking"] for j in jm])) if jm else None,
            "median_rec_speed_drop": float(np.median([j["rec_speed_drop"] for j in jm])) if jm else None,
            "median_missed_here": float(np.median([j["missed_here"] for j in jm])) if jm else None,
        }
    # Task 5
    from pathlib import Path as P
    OFF = P(r"E:/rd-wt/ws-eval2/_out/run3_c1_entries_off")
    ON = P(r"E:/rd-wt/ws-eval2/_out/run3_c1_entries_on")
    for model in ("run3", "run3_s120"):
        for name in ("u_same", "u_new", "seen"):
            off = DL.load_states_file(OFF / f"states__{model}__{name}.json")
            on = DL.load_states_file(ON / f"states__{model}__{name}.json")
            arm = "pi" if name in ("u_same", "u_new") else "native"
            onw = {s["world"]: s for s in on}
            comm = [s for s in off if s["world"] in onw]
            ob = sum(DL.blowup(s, arm) for s in comm)
            nb = sum(DL.blowup(onw[s["world"]], arm) for s in comm)

            def mm(states):
                v = [((s["arms"].get(arm, {}).get("car", {}).get("120.0", {}).get("hdv", {}) or {})
                      .get("ds_med")) for s in states]
                v = [x for x in v if x is not None]
                return float(np.mean(v)) if v else None
            out["task5"][f"{model}/{name}"] = {
                "common": len(comm), "blow_off": ob, "blow_on": nb,
                "mean_ds120_off": mm(comm),
                "mean_ds120_on": mm([onw[s["world"]] for s in comm])}
    (D / "analysis2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
