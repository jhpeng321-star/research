"""Full DIAG120 analysis over the two rollout files + stored sheets.

Produces analysis.json with:
  * blow-up integrity check vs stored scores;
  * Task 1 phantom/missed/drift classification from signed section series;
  * Task 2 error-growth shape (crossing times, late jump, acceleration);
  * Task 3 per-vehicle attribution + leader-onset propagation (20 worst);
  * entrant / lane-change context.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
import diaglib as DL  # noqa: E402


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def arr(state, key):
    return np.array([np.nan if v is None else v for v in state[key]], dtype=float)


def jam_stats(state: dict, upto: int) -> dict:
    """Time fractions and peaks of predicted-vs-recorded congestion (merge+up)."""
    res = {}
    for sect in ("merge", "up"):
        sp = np.zeros(upto + 1)
        sr = np.zeros(upto + 1)
        for i in range(upto + 1):
            p = state["sec_pred"].get(str(i), {}).get(sect)
            r = state["sec_rec"].get(str(i), {}).get(sect)
            sp[i] = (p["stopped"] if p else 0.0)
            sr[i] = (r["stopped"] if r else 0.0)
        res[sect] = {
            "phantom_frac": float(np.mean(sp > sr + 3)),
            "missed_frac": float(np.mean(sr > sp + 3)),
            "both_frac": float(np.mean((sp > 3) & (sr > 3))),
            "stop_pred_mean": float(sp.mean()), "stop_rec_mean": float(sr.mean()),
            "stop_pred_peak": float(sp.max()), "stop_rec_peak": float(sr.max()),
        }
    return res


def classify(state: dict) -> str:
    """One label per state from the merge/upstream congestion comparison."""
    j = jam_stats(state, len(state["t"]) - 1)
    ph = max(j["merge"]["phantom_frac"], j["up"]["phantom_frac"])
    ms = max(j["merge"]["missed_frac"], j["up"]["missed_frac"])
    if ph >= 0.20 and ph > ms:
        return "phantom"
    if ms >= 0.20 and ms > ph:
        return "missed"
    if ph >= 0.10 and ms >= 0.10:
        return "mixed"
    return "drift"


def growth(state: dict) -> dict:
    ds = arr(state, "ds_med")
    n = len(ds)
    idx = np.arange(n) * 0.5

    def cross(th):
        hit = np.flatnonzero(~np.isnan(ds) & (ds > th))
        return float(idx[hit[0]]) if hit.size else None

    out = {"t_cross20": cross(20.0), "t_cross50": cross(50.0),
           "t_cross100": cross(100.0)}
    # largest 5 s (10-step) rise inside 0..60 s and 60..120 s
    def max_rise(a, b):
        lo, hi = int(a / 0.5), min(n, int(b / 0.5))
        seg = ds[lo:hi]
        seg = np.where(np.isnan(seg), 0.0, seg)
        if seg.size < 11:
            return 0.0
        return float(np.max(seg[10:] - seg[:-10]))
    out["rise_0_60"] = max_rise(0, 60)
    out["rise_60_120"] = max_rise(60, 120)
    # growth acceleration: slope 30-60 vs 60-90
    def slope(a, b):
        lo, hi = int(a / 0.5), min(n, int(b / 0.5))
        seg = ds[lo:hi]
        m = ~np.isnan(seg)
        if m.sum() < 5:
            return None
        return float(np.polyfit(idx[lo:hi][m], seg[m], 1)[0])
    out["slope_30_60"] = slope(30, 60)
    out["slope_60_90"] = slope(60, 90)
    return out


def attribution(state: dict, n_worst_rank: int | None) -> dict:
    """Per-vehicle error carriers and leader-onset propagation."""
    vs = [v for v in state["vehicles"] if np.isfinite(v.get("ds_peak", np.nan))]
    vs.sort(key=lambda v: -v["ds_peak"])
    top = vs[:10]
    # leader onset propagation: pairs where both have onset and same lane family
    prop = []
    for v in state["vehicles"]:
        lo = v.get("leader_onset_s")
        o = v.get("onset_s")
        if o is not None and lo is not None:
            prop.append(o - lo)
    # count vehicles near merge at t0 that blow up
    merge_veh = [v for v in state["vehicles"] if v["in_merge0"]]
    merge_onset = [v for v in merge_veh if v.get("onset_s") is not None]
    after = [v for v in state["vehicles"] if v.get("entrant")]
    lc = [v for v in state["vehicles"] if v["rec_lane_changes"] > 0]
    return {
        "top10": [{"id": v["id"], "ds_peak": v["ds_peak"], "onset_s": v.get("onset_s"),
                   "s0": v["s0"], "lane0": v["lane0"], "in_merge0": v["in_merge0"],
                   "leader": v.get("leader"), "leader_onset_s": v.get("leader_onset_s"),
                   "rec_lane_changes": v["rec_lane_changes"]} for v in top],
        "n_vehicles": len(vs),
        "n_onset": sum(1 for v in state["vehicles"] if v.get("onset_s") is not None),
        "n_merge_t0": len(merge_veh), "n_merge_onset": len(merge_onset),
        "n_after_entrant": len(after), "n_lane_changers": len(lc),
        "prop_delays": prop,
    }


def main() -> int:
    stored = json.loads((D / "stored_summary.json").read_text())
    out: dict = {"integrity": {}, "task1": {}, "task2": {}, "task3": {},
                 "context": {}}
    for model, fn, skey in (("c1_struct", "roll_struct_u_same.jsonl", "c1_struct"),
                            ("oldhist", "roll_oldhist_u_same.jsonl", "oldhist")):
        rows = [r for r in load_jsonl(D / fn) if "error" not in r]
        st = {s["world"]: s for s in DL.load_states_file(
            DL.SHIFT / skey / f"states__{skey}__u_same.json")}
        disagree = [r["world"] for r in rows
                    if DL.blowup(st.get(r["world"], {}), "pi") != r["blow"]]
        out["integrity"][model] = {"n": len(rows), "n_blow": sum(1 for r in rows if r["blow"]),
                                   "stored_blow": stored[skey]["u_same"]["n_blow"],
                                   "disagreements": disagree}

        # Task1
        t1 = {"blow": {}, "nonblow": {}}
        labels = {}
        for sel, key in ((True, "blow"), (False, "nonblow")):
            sub = [r for r in rows if r["blow"] == sel]
            cats = [classify(r) for r in sub]
            t1[key] = {
                "n": len(sub),
                "phantom": sum(c == "phantom" for c in cats),
                "missed": sum(c == "missed" for c in cats),
                "mixed": sum(c == "mixed" for c in cats),
                "drift": sum(c == "drift" for c in cats),
                "mean_merge_phantom_frac": float(np.mean(
                    [jam_stats(r, len(r["t"]) - 1)["merge"]["phantom_frac"] for r in sub])),
                "mean_merge_missed_frac": float(np.mean(
                    [jam_stats(r, len(r["t"]) - 1)["merge"]["missed_frac"] for r in sub])),
            }
        for r in rows:
            labels[r["world"]] = classify(r)
        out["task1"][model] = t1
        out["task1"][model + "_labels"] = labels

        # Task2
        g = [growth(r) for r in rows if r["blow"]]
        def stat(k):
            vals = [x[k] for x in g if x.get(k) is not None]
            return {"median": float(np.median(vals)) if vals else None,
                    "n": len(vals), "p90": float(np.percentile(vals, 90)) if vals else None}
        out["task2"][model] = {
            "t_cross20": stat("t_cross20"), "t_cross50": stat("t_cross50"),
            "t_cross100": stat("t_cross100"),
            "rise_0_60": stat("rise_0_60"), "rise_60_120": stat("rise_60_120"),
            "slope_30_60": stat("slope_30_60"), "slope_60_90": stat("slope_60_90"),
        }

        # Task3 on 20 worst by ds120 (fallback ds60)
        worst = sorted(rows, key=lambda r: -(r["ds120"] if r["ds120"] else (r["ds60"] or 0)))[:20]
        att = [attribution(r, i) for i, r in enumerate(worst)]
        prop_all = [p for a in att for p in a["prop_delays"]]
        out["task3"][model] = {
            "n_merge_t0_mean": float(np.mean([a["n_merge_t0"] for a in att])),
            "n_onset_mean": float(np.mean([a["n_onset"] for a in att])),
            "n_veh_mean": float(np.mean([a["n_vehicles"] for a in att])),
            "n_lane_changers_mean": float(np.mean([a["n_lane_changers"] for a in att])),
            "n_after_entrant_total": int(sum(a["n_after_entrant"] for a in att)),
            "prop_n": len(prop_all),
            "prop_median_delay": float(np.median(prop_all)) if prop_all else None,
            "prop_frac_positive": float(np.mean([p > 0 for p in prop_all])) if prop_all else None,
            "worst_worlds": [r["world"] for r in worst],
            "detail_top": att,
        }
        # context
        out["context"][model] = {
            "entrants_total": int(sum(math_sum(r) for r in rows)),
            "blow_with_entrants": int(sum(1 for r in rows if r["blow"] and math_sum(r))),
        }
    (D / "analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote analysis.json")
    print(json.dumps({m: out["integrity"][m] for m in out["integrity"]}, indent=1))
    for m in ("c1_struct", "oldhist"):
        print("TASK1", m, json.dumps(out["task1"][m]))
        print("TASK2", m, json.dumps(out["task2"][m]))
        print("TASK3", m, {k: v for k, v in out["task3"][m].items() if k != "detail_top"})
    return 0


def math_sum(r):
    return int(r.get("n_entrants", 0) > 0)


if __name__ == "__main__":
    raise SystemExit(main())
