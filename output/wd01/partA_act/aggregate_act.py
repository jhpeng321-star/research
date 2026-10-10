"""A2 validation aggregation: per-arm tables and PAIRED deltas from the
raw validation rows (validate_act.py).

The planner quantity is the PAIRED delta over the same world:
  d_pred = delay_phys_pred(arm roll) - delay_phys_pred(native roll)
  d_true = delay_phys_true(arm)     - delay_phys_true(native)
reported per arm for the action-conditioned engine ('act') and for the
CAVs-as-native naive baseline ('naive', rolled per arm on the branch's
own arrival realisation).  The same pairing is reported for TTC<3 s
exposure.  Windows are tagged PRE-FIX when penetration == 0.40 or the
(world, arm) run appears in the chunk-1 COLLISION_TALLY_FINAL.txt;
collision statistics are reported separately for PRE-FIX and POST-FIX.

Pure aggregation over the raw jsonl; no data access beyond the tally
file.  ZERO simulation.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import act_lib as L  # noqa: E402

RAW = HERE / "results" / "raw" / "raw_metrics_act.jsonl"
TALLY = L.refload.CHUNK1_OUT.parent / "campaign_stage1_chunk1_launch" / \
    "COLLISION_TALLY_FINAL.txt"
ARMS = L.BRANCH_ARMS


def tally_runs():
    """{run_name} from COLLISION_TALLY_FINAL.txt (best effort; the
    aggregate degrades to penetration-only tagging if unavailable)."""
    out = set()
    try:
        for line in TALLY.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.add(line.split("|")[0].strip())
    except OSError:
        pass
    return out


def load_rows():
    rows = [json.loads(l) for l in RAW.read_text(encoding="utf-8")
            .splitlines() if l.strip()]
    tally = tally_runs()
    for r in rows:
        r["prefix"] = bool(r.get("penetration") == 0.40
                           or f"{r['world']}__{r['arm']}" in tally)
    return rows


def pooled_rmse(vals):
    return float(np.sqrt(np.mean(np.asarray(vals) ** 2))) if vals else None


def agg_group(rows):
    """Aggregate one (variant, arm[, prefix]) group of raw rows."""
    if not rows:
        return None
    dp = np.array([r["delay_phys_pred"] for r in rows])
    dt_ = np.array([r["delay_phys_true"] for r in rows])
    return dict(
        n_windows=len(rows),
        speed_rmse_cav=pooled_rmse([r["speed_rmse_cav"] for r in rows]),
        speed_rmse_human=pooled_rmse([r["speed_rmse_human"]
                                      for r in rows]),
        delay_phys_pred_mean=float(dp.mean()),
        delay_phys_true_mean=float(dt_.mean()),
        delay_bias=float((dp - dt_).mean()),
        pred_ttc3_exposure_mean=float(np.mean(
            [r["pred_ttc3_exposure"] for r in rows])),
        true_ttc3_exposure_mean=float(np.mean(
            [r["true_ttc3_exposure"] for r in rows])),
        ttc3_bias=float(np.mean([r["pred_ttc3_exposure"]
                                 - r["true_ttc3_exposure"]
                                 for r in rows])),
        collisions_per_1000_mean=float(np.mean(
            [r["collisions_per_1000"] for r in rows])),
        windows_with_pred_collision=int(sum(1 for r in rows
                                            if r["pred_collisions"] > 0)),
        truth_collisions_total=int(sum(r["truth_collisions_in_window"]
                                       for r in rows)),
        pred_vehicle_steps=int(sum(r["pred_vehicle_steps"]
                                   for r in rows)),
        n_commands_mean=float(np.mean([r["n_commands"] for r in rows])),
    )


def paired_delta(rows, arm, value_pred, value_true, min_abs=0.0):
    """(act/naive) paired arm-minus-native deltas over shared worlds."""
    by = defaultdict(dict)
    for r in rows:
        by[r["world"]][r["arm"]] = r
    dp, dt_ = [], []
    for world, arms in by.items():
        if arm not in arms or "native" not in arms:
            continue
        dp.append(arms[arm][value_pred] - arms["native"][value_pred])
        dt_.append(arms[arm][value_true] - arms["native"][value_true])
    dp = np.asarray(dp)
    dt_ = np.asarray(dt_)
    if len(dp) == 0:
        return None
    err = dp - dt_
    both_nonzero = (np.abs(dp) > min_abs) & (np.abs(dt_) > min_abs)
    sign = float(np.mean(np.sign(dp[both_nonzero])
                         == np.sign(dt_[both_nonzero]))) \
        if both_nonzero.any() else None
    return dict(
        n_pairs=int(len(dp)),
        d_pred_mean=float(dp.mean()), d_true_mean=float(dt_.mean()),
        delta_bias=float(err.mean()), delta_mae=float(np.abs(err).mean()),
        delta_rmse=float(np.sqrt(np.mean(err ** 2))),
        sign_agreement=sign,
        n_sign_evaluable=int(both_nonzero.sum()),
        pearson=L.corr(dt_, dp, "pearson"),
        spearman=L.corr(dt_, dp, "spearman"))


def main():
    rows = load_rows()
    print(f"=== A2 validation aggregation: {len(rows)} raw rows "
          f"({len(set(r['world'] for r in rows))} worlds) ===")
    result = dict(
        experiment="WD-01 partA_act A2 validation aggregation",
        raw_rows=len(rows),
        worlds=len(set(r["world"] for r in rows)),
        prefix_tagging="penetration == 0.40 or (world, arm) in "
                       "COLLISION_TALLY_FINAL.txt",
    )
    print("\n--- per arm x variant (all windows) ---")
    print(f"{'arm':>11s} {'var':>5s} {'n':>4s} "
          f"{'rmseCAV':>8s} {'rmseHum':>8s} {'dlyBias':>8s} "
          f"{'ttc3Bias':>9s} {'coll/1k':>8s} {'cmds':>6s}")
    table = {}
    for arm in ARMS:
        for variant in ("act", "naive"):
            sub = [r for r in rows if r["arm"] == arm
                   and r["variant"] == variant]
            a = agg_group(sub)
            table[f"{arm}|{variant}"] = a
            if a:
                print(f"{arm:>11s} {variant:>5s} {a['n_windows']:4d} "
                      f"{a['speed_rmse_cav']:8.3f} "
                      f"{a['speed_rmse_human']:8.3f} "
                      f"{a['delay_bias']:+8.1f} {a['ttc3_bias']:+9.2f} "
                      f"{a['collisions_per_1000_mean']:8.3f} "
                      f"{a['n_commands_mean']:6.1f}")
    result["per_arm"] = table

    print("\n--- paired delta (arm - native) per arm: predicted vs true, "
          "act vs naive ---")
    print(f"{'arm':>11s} {'var':>5s} {'n':>4s} {'dPred':>8s} {'dTrue':>8s} "
          f"{'bias':>8s} {'MAE':>7s} {'sign':>6s} {'pear':>6s} {'spear':>6s}")
    deltas = {}
    for arm in ARMS:
        if arm == "native":
            continue
        for variant in ("act", "naive"):
            sub = [r for r in rows if r["variant"] == variant]
            d = paired_delta(sub, arm, "delay_phys_pred",
                             "delay_phys_true")
            t = paired_delta(sub, arm, "pred_ttc3_exposure",
                             "true_ttc3_exposure")
            deltas[f"{arm}|{variant}"] = dict(delay=d, ttc3=t)
            if d:
                print(f"{arm:>11s} {variant:>5s} {d['n_pairs']:4d} "
                      f"{d['d_pred_mean']:+8.2f} {d['d_true_mean']:+8.2f} "
                      f"{d['delta_bias']:+8.2f} {d['delta_mae']:7.2f} "
                      f"{(d['sign_agreement'] if d['sign_agreement'] is not None else float('nan')):6.2f} "
                      f"{(d['pearson'] if d['pearson'] is not None else float('nan')):6.2f} "
                      f"{(d['spearman'] if d['spearman'] is not None else float('nan')):6.2f}")
    result["paired_delta"] = deltas

    print("\n--- paired TTC<3 delta per arm (act | naive) ---")
    for arm in ARMS:
        if arm == "native":
            continue
        line = f"{arm:>11s}: "
        for variant in ("act", "naive"):
            t = deltas[f"{arm}|{variant}"]["ttc3"]
            if t:
                line += (f"{variant} dPred {t['d_pred_mean']:+7.2f} "
                         f"dTrue {t['d_true_mean']:+7.2f} "
                         f"MAE {t['delta_mae']:6.2f} "
                         f"pear {(t['pearson'] if t['pearson'] is not None else float('nan')):+5.2f}   ")
        print(line)

    # where the action signal lives: commanded-CAV speed tracking by
    # penetration (act vs naive)
    print("\n--- CAV speed RMSE by penetration (act | naive) ---")
    rmse_pen = {}
    for pen in sorted({r["penetration"] for r in rows}):
        line = f"pen {pen:.2f}: "
        for variant in ("act", "naive"):
            sub = [r for r in rows if r["variant"] == variant
                   and r["penetration"] == pen]
            v = pooled_rmse([r["speed_rmse_cav"] for r in sub])
            rmse_pen[f"{pen}|{variant}"] = dict(
                n=len(sub), speed_rmse_cav=v)
            line += f"{variant} {v:5.2f} (n={len(sub):3d})   "
        print(line)
    result["cav_rmse_by_penetration"] = rmse_pen

    print("\n--- collisions: PRE-FIX vs POST-FIX windows ---")
    coll = {}
    for tag in (False, True):
        name = "PRE-FIX" if tag else "POST-FIX"
        sub = [r for r in rows if r["prefix"] == tag]
        by_var = {}
        for variant in ("act", "naive"):
            s2 = [r for r in sub if r["variant"] == variant]
            by_var[variant] = agg_group(s2)
        coll[name] = {k: v for k, v in by_var.items() if v}
        for variant, a in by_var.items():
            if a:
                print(f"{name:>8s} {variant:>5s}: n {a['n_windows']:4d}  "
                      f"coll/1000 {a['collisions_per_1000_mean']:6.3f}  "
                      f"windows_with_pred_collision "
                      f"{a['windows_with_pred_collision']:3d}  "
                      f"truth_collisions {a['truth_collisions_total']:2d}")
    result["collisions_prefix_split"] = coll

    # the naive-baseline signal test: delay-delta error vs the true
    # delta spread
    tds = []
    by = defaultdict(dict)
    for r in rows:
        if r["variant"] == "act":
            by[r["world"]][r["arm"]] = r
    for world, arms in by.items():
        if "native" in arms:
            for arm in ARMS:
                if arm != "native" and arm in arms:
                    tds.append(arms[arm]["delay_phys_true"]
                               - arms["native"]["delay_phys_true"])
    tds = np.asarray(tds)
    if len(tds):
        result["true_delta_spread"] = dict(
            n=int(len(tds)), mean=float(tds.mean()),
            sd=float(tds.std()), p95_abs=float(np.quantile(
                np.abs(tds), 0.95)))
        print(f"\ntrue (arm - native) delay deltas: n {len(tds)}  "
              f"mean {tds.mean():+.2f}  sd {tds.std():.2f}  "
              f"p95|d| {np.quantile(np.abs(tds), 0.95):.2f} veh-s")

    out = HERE / "results" / "validation_act.json"
    out.write_text(json.dumps(result, indent=1, default=str),
                   encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
