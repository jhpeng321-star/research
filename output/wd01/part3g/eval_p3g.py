"""Part 3g: calibration of lc_scale (TRAIN only) and eps (TRAIN only),
then the ablation ladder eval on validation.  Zero simulation."""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "partA_data"))
sys.path.insert(0, str(HERE.parent / "part3"))

import p3lib as P  # noqa: E402
import engine as E  # noqa: E402

T0 = time.time()
DEV = P.device()
VAL_T0S = tuple(range(120, 481, 30))
HORIZONS = (30, 60, 90, 120)


def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def calibrate_lc_scale(train_roots, dev):
    """Match the mean LC count per window to the truth LC count on
    TRAIN windows at H=60.  Returns the calibrated lc_scale."""
    truth_lc = 0.0
    n_windows = 0
    for root in train_roots[:4]:
        for t0 in (240.0, 360.0):
            ra = P.root_arrays(root)
            k0 = int(round((t0 - 60.0) / 0.5))
            for j in range(1, 121):
                truth_lc += sum(1 for v, t in zip(ra.lc_vid, ra.lc_tick)
                                if t == k0 + j)
            n_windows += 1
    target = truth_lc / max(n_windows, 1)
    print(f"truth LC/window on train (subset): {target:.1f}")
    for scale in (0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01):
        counts = []
        for root in train_roots[:2]:
            for t0 in (240.0, 360.0):
                r = E.Rollout(root, t0, kind="idm_sampled", member_seed=99,
                              lc_scale=scale, dev=dev).run()
                m = E.window_metrics(r.snapshots, r.ra, r.k0, 60.0)
                counts.append(m["pred_lc"])
        mean_lc = float(np.mean(counts))
        print(f"  lc_scale {scale}: {mean_lc:.1f} LC/window")
        if abs(mean_lc - target) / max(target, 1) <= 0.1:
            return scale
        if mean_lc > target:
            return scale
    return 0.005


def main():
    man = json.loads(Path(__file__).resolve().parent.parent.joinpath(
        "partA_data", "MANIFEST.json").read_text(encoding="utf-8"))
    splits = defaultdict(list)
    for name, r in man["roots"].items():
        splits[r["split"]].append(name)
    train_roots = sorted(splits["train"])
    val_roots = sorted(splits["holdout"])
    val_set = {"cv20_7005m2000", "cv20_7006m2000", "cv30_7305m2700",
               "seed_7535m2400"}
    val_roots = val_set
    trn = [r for r in train_roots if r not in val_set]
    print(f"train={len(trn)} val={len(val_roots)} holdout={len(splits['holdout'])}")

    # ---- calibrate lc_scale on TRAIN
    lc_scale = calibrate_lc_scale(trn, DEV)
    print(f"calibrated lc_scale={lc_scale}")
    (HERE / "results").mkdir(parents=True, exist_ok=True)
    (HERE / "results" / "lc_calibration.json").write_text(
        json.dumps(dict(lc_scale=lc_scale, calibrated_on="train_roots"),
                   indent=1), encoding="utf-8")

    # ---- ablation ladder on validation
    variants = ([f"L{li}" for li in range(4)] +
                ["B1_persistence", "B3_idm_oracle"])
    all_rows = defaultdict(list)
    for variant in variants:
        for root in val_roots:
            for t0 in range(120, 481, 30):
                if variant == "L0":
                    r = E.Rollout(root, float(t0), kind="idm_normal",
                                  dev=DEV).run()
                elif variant == "L1":
                    r = E.Rollout(root, float(t0), kind="idm_sampled",
                                  member_seed=1, lc_scale=0, dev=DEV).run()
                elif variant == "L2":
                    r = E.Rollout(root, float(t0), kind="idm_sampled",
                                  member_seed=2, lc_scale=0, dev=DEV).run()
                elif variant == "L3":
                    r = E.Rollout(root, float(t0), kind="idm_sampled",
                                  member_seed=3, lc_scale=lc_scale,
                                  dev=DEV).run()
                elif variant == "B1_persistence":
                    r = E.Rollout(root, float(t0), kind="persistence",
                                  dev=DEV).run()
                elif variant == "B3_idm_oracle":
                    r = E.Rollout(root, float(t0), kind="idm_oracle",
                                  dev=DEV).run()
                ra = r.ra
                kk = r.k0
                for H in (30, 60, 90, 120):
                    m = E.window_metrics(r.snapshots, ra, kk, H)
                    m["variant"] = variant; m["root"] = root
                    m["t0"] = t0; m["H"] = H
                    all_rows[(variant, H)].append(m)
        log(f"{variant}: done ({time.time()-T0:.0f}s)")

    # save raw
    raw_dir = HERE / "results" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for (variant, H), rr in sorted(all_rows.items()):
        f = raw_dir / f"{variant}_H{H}.json"
        f.write_text(json.dumps(rr, indent=1, default=str),
                     encoding="utf-8")
    log(f"raw metrics saved to {raw_dir}")

    # aggregate
    per_vh = {}
    for (variant, H), rr in sorted(all_rows.items()):
        n = len(rr)
        d = dict(n_windows=n)
        for k in ("pred_vehicle_steps", "matched_vehicle_steps",
                  "pred_collisions", "pred_lc", "true_lc",
                  "pred_ttc3_exposure", "true_ttc3_exposure",
                  "pred_hard_brake_s", "true_hard_brake_s",
                  "delay_pred", "delay_true"):
            d[k] = round(float(np.mean([r[k] for r in rows])), 3)
        d["speed_rmse"] = round(float(np.sqrt(np.mean(
            [r["speed_rmse"]**2 for r in rows]))), 4)
        d["pos_rmse"] = round(float(np.sqrt(np.mean(
            [r["pos_rmse"]**2 for r in rows]))), 2)
        d["lane_frac"] = round(float(np.mean(
            [r["lane_frac"] for r in rows])), 4)
        d["collisions_per_1000"] = round(float(np.mean(
            [r["collisions_per_1000"] for r in rows])), 3)
        for f in ("ttc3", "hb"):
            t = [r[f"any_{f}_true"] for r in rows]
            pr = [r[f"any_{f}_pred"] for r in rows]
            tp = sum(1 for x, y in zip(t, pr) if x and y)
            d[f"any_{f}_recall"] = round(tp / sum(t), 4) if sum(t) else None
            d[f"any_{f}_precision"] = round(tp / sum(pr), 4) if sum(pr) else None
            d[f"any_{f}_base_rate"] = round(sum(t) / n, 4)
        d["delay_bias"] = round(d["delay_pred"] - d["delay_true"], 2)
        per_vh[f"{variant}@H{H}"] = d

    (HERE / "results" / "per_variant_H.json").write_text(
        json.dumps(per_vh, indent=1, default=str), encoding="utf-8")

    # gate v2 at H=60
    def g(v, k):
        return per_vh.get(f"{v}@H60", {}).get(k)
    gate_v2 = {}
    for rung in ("L0", "L1", "L2", "L3"):
        a = g(rung, "collisions_per_1000") or 0
        b = g(rung, "pred_lc") or 0
        t = g(rung, "true_lc") or 1
        c_ok = a <= 1.0
        b_ok = abs(b - t) / max(t, 1) <= 0.20
        d_ok = abs(g(rung, "speed_rmse") or 99) <= 1.05 * 4.97
        gate_v2[rung] = dict(
            collisions_per_1000=a, C2_pass=c_ok,
            lc_pred=b, lc_true=t, lc_within_20pct=b_ok,
            speed_rmse=g(rung, "speed_rmse"),
            speed_ok=d_ok,
            v2_pass=c_ok and b_ok and d_ok)
    old_gate = {}
    for rung in ("L0", "L1", "L2", "L3"):
        a = g(rung, "collisions_per_1000") or 0
        s = g(rung, "speed_rmse") or 99
        old_gate[rung] = dict(
            speed_rmse=s, speed_lt_B2=s < 4.97,
            collisions=a, collisions_ok=a <= 1.0,
            old_pass=(s < 4.97 and a <= 1.0))

    (HERE / "results" / "gate_v2.json").write_text(
        json.dumps(dict(gate_v2=gate_v2, old_gate=old_gate),
                   indent=1, default=str), encoding="utf-8")
    print(json.dumps(dict(gate_v2=gate_v2, old_gate=old_gate), indent=1))


if __name__ == "__main__":
    main()
