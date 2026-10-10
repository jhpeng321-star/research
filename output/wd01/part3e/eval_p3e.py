"""Part 3e final validation evaluation (complete)."""
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "partA_data"))
sys.path.insert(0, str(HERE.parent / "part3"))

import engine as E  # noqa: E402
import p3lib as P  # noqa: E402

T0 = time.time()
DEV = P.device()
RUN3 = HERE.parent / "part3" / "weights" / "run3"
VAL_T0S = tuple(range(120, 481, 30))
HORIZONS = (30, 60, 90, 120)
LC_SCALE = 0.004
MEMBERS = 5


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def agg(rows):
    n = len(rows)
    out = dict(n_windows=n)
    scalars = ("pred_vehicle_steps", "matched_vehicle_steps",
               "pred_collisions", "pred_lc", "true_lc",
               "pred_ttc3_exposure", "true_ttc3_exposure",
               "pred_ttc15_exposure", "true_ttc15_exposure",
               "pred_hard_brake_s", "true_hard_brake_s",
               "delay_pred", "delay_true")
    for k in scalars:
        out[k] = float(np.mean([r[k] for r in rows]))
    out["collisions_per_1000"] = float(np.mean(
        [r["collisions_per_1000"] for r in rows]))
    out["speed_rmse"] = float(np.sqrt(np.mean(
        [r["speed_rmse"] ** 2 for r in rows])))
    out["pos_rmse"] = float(np.sqrt(np.mean(
        [r["pos_rmse"] ** 2 for r in rows])))
    out["lane_frac"] = float(np.mean([r["lane_frac"] for r in rows]))
    for f in ("ttc15", "ttc3", "hb"):
        t = [r[f"any_{f}_true"] for r in rows]
        pr = [r[f"any_{f}_pred"] for r in rows]
        tp = sum(1 for x, y in zip(t, pr) if x and y)
        out[f"any_{f}_recall"] = round(tp / sum(t), 4) if sum(t) else None
        out[f"any_{f}_precision"] = round(tp / sum(pr), 4) if sum(pr) \
            else None
        out[f"any_{f}_base_rate"] = round(sum(t) / n, 4)
    for k in ("ttc3_exposure", "ttc15_exposure", "hard_brake_s"):
        t = [r[f"true_{k}"] for r in rows]
        pr = [r[f"pred_{k}"] for r in rows]
        out[f"{k}_bias"] = round(float(np.mean(np.asarray(pr)
                                               - np.asarray(t))), 2)
    out["delay_bias"] = round(out["delay_pred"] - out["delay_true"], 2)
    return out


def coverage(per_member, key, truth_key):
    c1 = c2 = mm = 0
    for vals, t in per_member:
        if not vals:
            continue
        mm += 1
        if min(vals) <= t <= max(vals):
            c1 += 1
        mu, sd = np.mean(vals), np.std(vals)
        if mu - 2 * sd <= t <= mu + 2 * sd:
            c2 += 1
    return (round(c1 / max(mm, 1), 4), round(c2 / max(mm, 1), 4))


def main():
    splits = P.roots_by_split()
    val_roots = splits["val"]
    log(f"validation roots {val_roots}")
    structured = [f"structured_member{i}" for i in range(5)]
    variants = (structured + ["B1_persistence", "B2_idm_normal",
                              "B3_idm_oracle", "B4_v3_1s", "p3c_run3"])
    rows = defaultdict(list)
    member_metrics = defaultdict(dict)  # (root,t0,H,mi) -> dict
    fwd = dict(steps=0, t=0.0)

    # ---------------- structured members (M=5) -------------------------
    for mi in range(5):
        variant = structured[mi]
        for root in val_roots:
            for t0 in VAL_T0S:
                t = time.time()
                r = E.Rollout(root, float(t0), kind="idm_sampled",
                              member_seed=11 + 17 * mi,
                              lc_scale=LC_SCALE, dev=DEV).run()
                fwd["steps"] += 240 * len(r.snapshots[0]["vid"])
                fwd["t"] += time.time() - t
                ra = r.ra
                kk = r.k0
                for H in HORIZONS:
                    m = E.window_metrics(r.snapshots, ra, kk, H)
                    rows[(variant, H)].append(m)
                    member_metrics[(root, t0, H, mi)] = m
        log(f"{variant}: done ({time.time() - T0:.0f}s)")

    # ---------------- baselines + p3c run3 ------------------------------
    for variant in ("B1_persistence", "B2_idm_normal", "B3_idm_oracle",
                    "B4_v3_1s", "p3c_run3"):
        for root in val_roots:
            for t0 in VAL_T0S:
                if variant == "B4_v3_1s":
                    raw = E.V3Baseline(root, float(t0), dev=DEV).run()\
                        .snapshots
                    snaps = {j: raw[min(j // 2, 120)]
                             for j in range(0, 241)}
                    kk = 2 * int(round((t0 - 60.0) / 1.0))
                    dt, stride = 1.0, 2
                    r = None
                elif variant == "p3c_run3":
                    t = time.time()
                    r = E.Rollout(root, float(t0), kind="model",
                                  weights=RUN3 / "member0.pt",
                                  dev=DEV).run()
                    fwd["steps"] += 240 * len(r.snapshots[0]["vid"])
                    fwd["t"] += time.time() - t
                    snaps, kk, dt, stride = r.snapshots, r.k0, 0.5, 1
                else:
                    kind = {"B1_persistence": "persistence",
                            "B2_idm_normal": "idm_normal",
                            "B3_idm_oracle": "idm_oracle"}[variant]
                    r = E.Rollout(root, float(t0), kind=kind,
                                  dev=DEV).run()
                    snaps, kk, dt, stride = r.snapshots, r.k0, 0.5, 1
                ra = P.root_arrays(root)
                for H in HORIZONS:
                    rows[(variant, H)].append(
                        E.window_metrics(snaps, ra, kk, H, dt=dt,
                                         stride=stride))
        log(f"{variant}: done ({time.time() - T0:.0f}s)")

    # ---------------- aggregate + gate ----------------------------------
    per_vh = {f"{v}@H{H}": E.agg_window_rows(rows[(v, H)])
              for v in variants for H in HORIZONS}
    # structured ensemble: compute ens speed RMSE / collisions as the
    # mean over members of their per-window metrics (already in per_vh)
    ens_speed_h60 = float(np.sqrt(np.mean([
        per_vh[f"structured_member{mi}@H60"]["speed_rmse"] ** 2
        for mi in range(5)])))
    ens_coll = float(np.mean([per_vh[f"structured_member{mi}@H60"]
                              ["collisions_per_1000"]
                              for mi in range(5)]))
    lc_pred = float(np.mean([per_vh[f"structured_member{mi}@H60"]
                             ["pred_lc"] for mi in range(5)]))
    lc_true = per_vh["structured_member0@H60"]["true_lc"] /         max(per_vh["structured_member0@H60"]["n_windows"], 1)

    # coverage of the structured ensemble (per window member spread)
    for H in HORIZONS:
        t3_pairs, d_pairs = [], []
        for root in val_roots:
            for t0 in VAL_T0S:
                mm = [member_metrics.get((root, t0, H, mi)) for mi in range(5)]
                mm = [m for m in mm if m]
                if len(mm) < 2:
                    continue
                t3_pairs.append(([m["pred_ttc3_exposure"] for m in mm],
                                 mm[0]["true_ttc3_exposure"]))
                d_pairs.append(([m["delay_pred"] for m in mm],
                                mm[0]["delay_true"]))
        c1, c2 = coverage(t3_pairs, None, None)
        per_vh[f"structured_ens@H{H}"] = dict(
            cov_ttc3_minmax=c1, cov_ttc3_pm2sd=c2,
            cov_delay_minmax=d1, cov_delay_pm2sd=d2)

    ens_speed_h60 = float(np.sqrt(np.mean([
        per_vh[f"structured_member{mi}@H60"]["speed_rmse"] ** 2
        for mi in range(5)])))
    ens_coll_h60 = float(np.mean([
        per_vh[f"structured_member{mi}@H60"]["collisions_per_1000"]
        for mi in range(5)]))
    lc_pred = float(np.mean([per_vh[f"structured_member{mi}@H60"]["pred_lc"]
                             for mi in range(5)]))
    lc_true = float(per_vh["structured_member0@H60"]["true_lc"])
    b1 = per_vh["B1_persistence@H60"]["speed_rmse"]
    b2 = per_vh["B2_idm_normal@H60"]["speed_rmse"]
    gate = dict(
        lc_scale=LC_SCALE,
        ens_speed_rmse_h60=round(ens_speed_h60, 3),
        beats_B1=ens_speed_h60 < b1,
        beats_B2=ens_speed_h60 < b2,
        B1_speed_rmse_h60=b1, B2_speed_rmse_h60=b2,
        ens_collisions_per_1000=round(ens_coll_h60, 3),
        collisions_ok=ens_coll_h60 <= 1.0,
        lc_pred=round(lc_pred, 1), lc_true=round(lc_true, 1),
        lc_within_20pct=abs(lc_pred - lc_true) / lc_true <= 0.2,
        delay_cov_h60=per_vh["structured_ens@H60"]["cov_delay_minmax"],
        ttc3_cov_h60=per_vh["structured_ens@H60"]["cov_ttc3_minmax"])
    gate["PASS"] = (gate["beats_B1"] and gate["beats_B2"]
                    and gate["collisions_ok"] and gate["lc_within_20pct"])

    result = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        lc_scale=LC_SCALE, members=MEMBERS,
        validation_roots=val_roots, t0s=list(VAL_T0S),
        horizons=list(HORIZONS),
        throughput_vehicle_steps_per_s=(fwd["steps"] / fwd["t"]
                                        if fwd["t"] else None),
        gate=gate, per_variant_H=per_vh)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    P.dump_json(out / "val_p3e_results.json", result)
    P.dump_json(out / "gate.json", gate)

    import csv
    with (out / "val_per_H.csv").open("w", newline="", encoding="utf-8") \
            as fh:
        wr = csv.writer(fh)
        head = ["variant", "H", "speed_rmse", "pos_rmse", "lane_frac",
                "collisions_per_1000", "lc_pred", "lc_true",
                "ttc3_recall", "ttc3_precision", "ttc3_base",
                "ttc15_recall", "ttc15_base",
                "hb_recall", "hb_base",
                "ttc3_exp_bias", "delay_bias",
                "cov_ttc3_minmax", "cov_delay_minmax"]
        wr.writerow(head)
        for v in variants:
            for H in HORIZONS:
                a = per_vh[f"{v}@H{H}"]
                extra = dict(cov_ttc3_minmax=a.get("cov_ttc3_minmax", ""),
                             cov_delay_minmax=a.get("cov_delay_minmax", ""))
                wr.writerow([v, H] + [fmt(a.get(k), 4) for k in head[2:]])

    log(f"GATE: {json.dumps(gate)}")
    log(f"EVAL DONE in {time.time() - T0:.0f}s; throughput "
        f"{result['throughput_vehicle_steps_per_s']:.0f} veh-steps/s")
    return 0


def fmt(x, nd=4):
    if x is None or x == "":
        return ""
    if isinstance(x, float):
        return round(x, nd)
    return x


if __name__ == "__main__":
    sys.exit(main())
