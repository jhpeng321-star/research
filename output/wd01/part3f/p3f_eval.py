"""Part 3f: structured emulator validation eval (dawdle + LC + vType)."""
import json, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent / "partA_data"), str(HERE.parent / "part3"),
          str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import engine as E
import p3lib as P

DEV = P.device()
LC_SCALE = 0.002
T0 = time.time()

def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)

def main():
    P.guards()
    splits = P.roots_by_split()
    val_roots = splits["val"]
    log(f"validation: {val_roots}")
    variants = ([f"structured_member{i}" for i in range(3)] +
                ["B1_persistence", "B2_idm_normal", "B3_idm_oracle"])
    all_rows = defaultdict(list)
    for variant in variants:
        for root in val_roots:
            for t0 in (120, 240, 360, 480):
                if variant.startswith("structured"):
                    mi = int(variant.replace("structured_member", ""))
                    r = E.Rollout(root, float(t0), kind="idm_sampled",
                                  member_seed=mi*17+1, lc_scale=0.002,
                                  dev=DEV).run()
                else:
                    kind = {"B1_persistence": "persistence",
                            "B2_idm_normal": "idm_normal",
                            "B3_idm_oracle": "idm_oracle"}[variant]
                    r = E.Rollout(root, float(t0), kind=kind, dev=DEV).run()
                ra = r.ra
                kk = r.k0
                for H in (30, 60, 90, 120):
                    m = E.window_metrics(r.snapshots, ra, kk, H)
                    m["variant"] = variant; m["root"] = root
                    m["t0"] = t0; m["H"] = H
                    all_rows[(variant, H)].append(m)
        log(f"{variant}: done")

    # aggregate
    results = {}
    for (variant, H), rr in sorted(all_rows.items()):
        results[f"{variant}@H{H}"] = dict(
            n=len(rr),
            speed_rmse=round(float(np.sqrt(np.mean(
                [r["speed_rmse"]**2 for r in rr]))), 4),
            pos_rmse=round(float(np.sqrt(np.mean(
                [r["pos_rmse"]**2 for r in rr]))), 2),
            collisions_per_1000=round(float(np.mean(
                [r["collisions_per_1000"] for r in rr])), 3),
            pred_lc=round(float(np.mean([r["pred_lc"] for r in rr])), 1),
            true_lc=round(float(np.mean([r["true_lc"] for r in rr])), 1),
            ttc3_recall=round(float(np.mean(
                [r["any_ttc3_pred"] for r in rr])), 4),
            hb_recall=round(float(np.mean(
                [r["any_hb_pred"] for r in rr])), 4),
            delay_bias=round(float(np.mean(
                [r["delay_pred"] - r["delay_true"] for r in rr])), 2))

    # gate at H=60
    def m_(v): return results.get(f"{v}@H60", {})
    ens_speed = float(np.sqrt(np.mean([
        results.get(f"structured_member{mi}@H60", {}).get("speed_rmse", 99)**2
        for mi in range(3)])))
    gate = dict(
        ens_speed_rmse_h60=round(ens_speed, 3),
        B1_speed_rmse_h60=m_("B1_persistence").get("speed_rmse"),
        B2_speed_rmse_h60=m_("B2_idm_normal").get("speed_rmse"),
        beats_B1=ens_speed < m_("B1_persistence").get("speed_rmse", 99),
        beats_B2=ens_speed < m_("B2_idm_normal").get("speed_rmse", 99),
        ens_collisions_per_1000=round(float(np.mean([
            results.get(f"structured_member{mi}@H60", {}).get(
                "collisions_per_1000", 99) for mi in range(3)])), 3),
        collisions_ok=all(
            results.get(f"structured_member{mi}@H60", {}).get(
                "collisions_per_1000", 99) <= 1.0 for mi in range(3)))
    gate["PASS"] = gate["beats_B1"] and gate["collisions_ok"]

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "val_results.json").write_text(
        json.dumps(results, indent=1, default=str), encoding="utf-8")
    (HERE / "results" / "gate.json").write_text(
        json.dumps(gate, indent=1), encoding="utf-8")
    log(f"GATE: {json.dumps(gate)}")
    log(f"DONE in {time.time()-T0:.0f}s")
    return 0

if __name__ == "__main__":
    sys.exit(main())
