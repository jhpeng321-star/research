"""Part 3h aggregation: read raw_metrics.jsonl, compute per-(variant, H)
summaries, gate v2 criterion (c) coverage + CRPS, and gate verdicts."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "results" / "raw" / "raw_metrics.jsonl"
OUT = HERE / "results"


def coverage_10_90(vals, truth):
    """10th-90th percentile band (numpy linear) covers truth?"""
    p10 = np.percentile(vals, 10)
    p90 = np.percentile(vals, 90)
    return p10 <= truth <= p90


def coverage_minmax(vals, truth):
    return min(vals) <= truth <= max(vals)


def crps_ensemble(vals, truth):
    """Empirical CRPS for an ensemble: mean|x_i - y| - 0.5*mean|x_i-x_j|."""
    vals = np.asarray(vals, dtype=np.float64)
    if len(vals) < 2:
        return abs(vals[0] - truth) if len(vals) else np.nan
    term1 = np.mean(np.abs(vals - truth))
    diff = np.abs(vals[:, None] - vals[None, :])
    term2 = 0.5 * np.mean(diff)
    return float(term1 - term2)


def load_raw():
    rows = []
    for line in RAW.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main():
    rows = load_raw()
    print(f"loaded {len(rows)} raw rows")
    by = defaultdict(list)
    for r in rows:
        by[(r["variant"], int(r["member"]), r["root"], int(r["t0"]),
            int(r["H"]))].append(r)
    # deduplicate: keep last occurrence
    dedup = {}
    for (v, m, r_, t, h), rr in by.items():
        dedup[(v, m, r_, t, h)] = rr[-1]

    # group by (variant, H): list of per-(root,t0,member) metrics
    grouped = defaultdict(lambda: defaultdict(list))
    for (v, m, r_, t, h), row in dedup.items():
        grouped.setdefault((v, int(h)), []).append(row)

    # aggregate per (variant, H)
    per_vh = {}
    for (variant, H), rr in sorted(grouped.items()):
        d = dict(n_rolls=len(rr), n_members=len(set(r["member"] for r in rr)))
        for k in ("pred_vehicle_steps", "matched_vehicle_steps",
                  "pred_collisions", "pred_lc", "true_lc",
                  "pred_ttc3_exposure", "true_ttc3_exposure",
                  "pred_hard_brake_s", "true_hard_brake_s",
                  "delay_pred", "delay_true"):
            vals = [r[k] for r in rr if k in r]
            if vals:
                d[k] = round(float(np.mean(vals)), 3)
        sr = [r["speed_rmse"] for r in rr if "speed_rmse" in r]
        if sr:
            d["speed_rmse"] = round(float(np.sqrt(np.mean(
                np.square(sr)))), 4)
        ps = [r["pos_rmse"] for r in rr if "pos_rmse" in r]
        if ps:
            d["pos_rmse"] = round(float(np.sqrt(np.mean(
                np.square(ps)))), 2)
        lf = [r["lane_frac"] for r in rr if "lane_frac" in r]
        if lf:
            d["lane_frac"] = round(float(np.mean(lf)), 4)
        cp = [r["collisions_per_1000"] for r in rr if "collisions_per_1000" in r]
        if cp:
            d["collisions_per_1000"] = round(float(np.mean(cp)), 3)
        for f in ("ttc3", "hb", "ttc15"):
            t = [bool(r[f"any_{f}_true"]) for r in rr if f"any_{f}_true" in r]
            pr = [bool(r[f"any_{f}_pred"]) for r in rr if f"any_{f}_pred" in r]
            if t and pr:
                tp = sum(1 for x, y in zip(t, pr) if x and y)
                d[f"any_{f}_recall"] = round(tp / sum(t), 4) if sum(t) else None
                d[f"any_{f}_precision"] = (round(tp / sum(pr), 4)
                                           if sum(pr) else None)
                d[f"any_{f}_base_rate"] = round(sum(t) / len(t), 4)
        if "delay_pred" in d and "delay_true" in d:
            d["delay_bias"] = round(d["delay_pred"] - d["delay_true"], 2)
        per_vh[f"{variant}@H{H}"] = d

    # ---- gate v2 criterion (c): 10-90% band coverage + CRPS at H=60
    per_member_at_h60 = defaultdict(lambda: defaultdict(dict))
    for (v, m, r_, t, h), row in dedup.items():
        if h != 60:
            continue
        per_member_at_h60[(v, r_, t)][m] = row
    coverage_results = {}
    for v in ("L1", "L3"):
        t3_cov = t3_cov_mm = 0
        delay_cov = delay_cov_mm = 0
        crps_vals = []
        ties = 0
        n_windows = 0
        for (v_, r_, t_), members in sorted(per_member_at_h60.items()):
            if v_ != v or len(members) < 2:
                continue
            t3_vals = [m["pred_ttc3_exposure"] for m in members.values()]
            delay_vals = [m["delay_pred"] for m in members.values()]
            truth_t3 = members[next(iter(members))].get(
                "true_ttc3_exposure", None)
            truth_d = members[next(iter(members))].get("delay_true", None)
            n_windows += 1
            if truth_t3 is not None:
                p10 = np.percentile(t3_vals, 10)
                p90 = np.percentile(t3_vals, 90)
                if p10 <= truth_t3 <= p90:
                    t3_cov += 1
                if all(v_ == 0 for v_ in t3_vals) and truth_t3 == 0:
                    ties += 1
                    t3_cov += 1
            if truth_d is not None:
                p10 = np.percentile(delay_vals, 10)
                p90 = np.percentile(delay_vals, 90)
                if p10 <= truth_d <= p90:
                    delay_cov += 1
        coverage_results[v] = dict(
            n_windows=n_windows, t3_cov_count=t3_cov, t3_cov_frac=(
                round(t3_cov / n_windows, 4) if n_windows else None),
            delay_cov_count=delay_cov, delay_cov_frac=(
                round(delay_cov / n_windows, 4) if n_windows else None),
            degenerate_ties=ties)
    # L0 deterministic MAE of delay at H=60
    l0_rows = [r for (v, m, r_, t, h), row in dedup.items()
               if v == "L0" and h == 60]
    l0_mae = float(np.mean([abs(r["delay_pred"] - r["delay_true"])
                            for r in l0_rows])) if l0_rows else None
    coverage_results["L0_deterministic_delay_MAE"] = l0_mae

    # ensemble CRPS of delay for L3
    crps_vals = []
    for (v_, r_, t_), members in sorted(per_member_at_h60.items()):
        if v_ != "L3" or len(members) < 2:
            continue
        delay_vals = np.array([m["delay_pred"] for m in members.values()])
        truth_d = members[next(iter(members))].get("delay_true", None)
        if truth_d is None:
            continue
        d = np.abs(delay_vals - truth_d)
        diff = np.abs(delay_vals[:, None] - delay_vals[None, :])
        crps = float(np.mean(d) - 0.5 * np.mean(diff))
        crps_vals.append(crps)
    coverage_results["L3_ens_delay_CRPS_mean"] = (round(
        float(np.mean(crps_vals)), 2) if crps_vals else None)
    coverage_results["L3_ens_delay_CRPS_n"] = len(crps_vals)
    coverage_results["L0_delay_MAE_for_comparison"] = l0_mae

    # gate v2 verdict at H=60
    gate_v2 = {}
    old_gate = {}
    for rung in ("L0", "L1", "L3"):
        key = f"{rung}@H60"
        if key not in per_vh:
            continue
        a = per_vh[key]
        coll = a.get("collisions_per_1000", 99)
        sr = a.get("speed_rmse", 99)
        lc_p = a.get("pred_lc", 0)
        lc_t = a.get("true_lc", 1)
        # gate v2 (a)
        a_ok = coll <= 1.0
        # gate v2 (b)
        b_ok = abs(lc_p - lc_t) / max(lc_t, 1) <= 0.20
        # gate v2 (c): coverage + CRPS
        cov = coverage_results.get(rung, {})
        c_t3 = (cov.get("t3_cov_frac", 0) or 0) >= 0.80
        c_d = (cov.get("delay_cov_frac", 0) or 0) >= 0.80
        crps_ok = True
        if rung == "L3" and coverage_results.get(
                "L3_ens_delay_CRPS_mean") is not None and l0_mae:
            crps_ok = coverage_results["L3_ens_delay_CRPS_mean"] < l0_mae
        c_ok = c_t3 and c_d and crps_ok
        # gate v2 (d)
        d_ok = sr <= 1.05 * 4.97
        gate_v2[rung] = dict(
            collisions_per_1000=coll, a_collisions_pass=a_ok,
            lc_pred=lc_p, lc_true=lc_t,
            lc_within_20pct=abs(lc_p - lc_t) / max(lc_t, 1) <= 0.20,
            b_lc_pass=b_ok,
            ttc3_cov_frac=cov.get("t3_cov_frac", 0), c_ttc3_pass=c_t3,
            delay_cov_frac=cov.get("delay_cov_frac", 0), c_delay_pass=c_d,
            crps_ok=crps_ok, c_coverage_crps_pass=c_ok,
            speed_rmse=sr, d_speed_pass=d_ok,
            d_speed_le_1p05xB2=d_ok,
            v2_PASS=a_ok and b_ok and c_ok and d_ok,
            v2_failing=[c for c, ok in
                        (("a_collisions", a_ok), ("b_lc", b_ok),
                         ("c_coverage_crps", c_ok), ("d_speed", d_ok))
                        if not ok])
        # old gate
        old_gate[rung] = dict(
            speed_rmse=sr, speed_lt_B2=sr < 4.97,
            collisions_per_1000=coll, collisions_ok=coll <= 1.0,
            lc_pred=lc_p, lc_true=lc_t,
            old_pass=sr < 4.97 and coll <= 1.0)
    result = dict(gate_v2=gate_v2, old_gate=old_gate,
                  coverage_results=coverage_results, per_vh=per_vh)
    (OUT / "gate_v2.json").write_text(json.dumps(result, indent=1,
                                                  default=str),
                                      encoding="utf-8")
    print(json.dumps(dict(gate_v2=gate_v2, old_gate=old_gate,
                          coverage=coverage_results),
                     indent=1, default=str))


if __name__ == "__main__":
    main()
