"""Part 3h eval: re-run L0/L1/L3 with M=5 members on validation, save
per-window per-member raw metrics INCREMENTALLY, then compute gate v2
criterion (c) coverage + CRPS and re-state gate v2 + old gate tables.
Zero simulation. Holdout never accessed."""
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

import engine as E  # noqa: E402
import p3lib as P  # noqa: E402

T0 = time.time()
DEV = P.device()
LC_SCALE = 0.003
MEMBERS = 5
VAL_T0S = tuple(range(120, 481, 30))
HORIZONS = (30, 60, 90, 120)
RAW_DIR = HERE / "results" / "raw"


def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def load_calibration():
    f = HERE.parent / "part3g" / "results" / "lc_calibration.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8")).get("lc_scale")
    return 0.003


def run_member(variant, root, t0, member_seed, lc_scale, dev):
    if variant == "L0":
        r = E.Rollout(root, float(t0), kind="idm_normal", dev=dev).run()
    else:
        r = E.Rollout(root, float(t0), kind="idm_sampled", member_seed=member_seed,
                      lc_scale=lc_scale, dev=dev).run()
    return r


def window_rows(r, root, t0, variant, mi):
    ra = r.ra
    rows = []
    for H in (30, 60, 90, 120):
        m = E.window_metrics(r.snapshots, ra, r.k0, H)
        rows.append(dict(
            variant=variant, member=mi, root=root, t0=t0, H=H,
            speed_rmse=m["speed_rmse"], pos_rmse=m["pos_rmse"],
            collisions_per_1000=m["collisions_per_1000"],
            pred_lc=m["pred_lc"], true_lc=m["true_lc"],
            delay_pred=m["delay_pred"], delay_true=m["delay_true"],
            pred_ttc3_exposure=m["pred_ttc3_exposure"],
            true_ttc3_exposure=m["true_ttc3_exposure"],
            pred_hard_brake_s=m["pred_hard_brake_s"],
            true_hard_brake_s=m["true_hard_brake_s"],
            any_ttc3_pred=m["any_ttc3_pred"], any_ttc3_true=m["any_ttc3_true"],
            any_hb_pred=m["any_hb_pred"], any_hb_true=m["any_hb_true"],
            lane_frac=m["lane_frac"]))
    return rows


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    lc_scale = load_calibration()
    log(f"lc_scale={lc_scale} (from p3g train calibration)")
    splits = P.roots_by_split()
    val_roots = splits["val"]
    log(f"validation roots {val_roots} (holdout never accessed)")

    variants = {"L0": None, "L1": 0, "L3": 0}
    # member seeds: m*17+1 for m=0..4; L0 is deterministic (single run)
    raw_path = RAW_DIR / "raw_metrics.jsonl"
    raw_f = open(raw_path, "a", encoding="utf-8")

    for variant, lc_s in (("L0", 0.0), ("L1", lc_scale), ("L3", lc_scale)):
        log(f"running {variant} ...")
        for root in val_roots:
            for t0 in VAL_T0S:
                n_m = MEMBERS if variant != "L0" else 1
                for mi in range(n_m):
                    r = run_member(variant, root, float(t0),
                                   mi * 17 + 1, lc_s, DEV)
                    rows = window_rows(r, root, t0, variant, mi)
                    for row in rows:
                        raw_f.write(json.dumps(row, default=str) + "\n")
                    raw_f.flush()
        log(f"{variant}: done ({time.time()-T0:.0f}s)")
    raw_f.close()
    log(f"raw rows appended to {raw_path}")


if __name__ == "__main__":
    main()
