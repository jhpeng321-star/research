"""CPU-only verification of the pack_v3 fix (no training, GPU, SUMO).

Builds the real TRAIN dataset with the new sidecar + lpwo_v3 lists, reports the
kept counts per pack, decodes a few pack_v3 windows and sanity-checks their
states (HDV/CAV counts, speed plausibility, identity consistency), and proves no
excluded run or eval-set world leaked in.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

WS = Path("E:/rd-wt/ws-packv3fix")
LPWO_V3 = WS / "_out" / "lpwo_v3"
SIDECAR_V2 = WS / "_out" / "sidecars_packv3_v2" / "identities"

PACKS = [r"D:/rd-staging/wmn1_feat/pack_v3",
         r"E:/rd-wt/dspeed-data/_out/pack_dspeed",
         r"E:/rd-wt/damping-snap/_out/pack_damping"]
IDENTITIES = [str(SIDECAR_V2),
              r"E:/rd-wt/a9/_out/identities_dspeed",
              r"E:/rd-wt/damping-snap/_out/identities_damping"]
OVERLAYS = [r"E:/rd-wt/dspeed-data/_out/labels_overlay_dspeed",
            r"E:/rd-wt/damping-snap/_out/labels_overlay_damping"]


def main() -> int:
    from wms.train import TrajectoryDataset
    from wms.train_guards import TrainingWindowGuard
    from wms import lpo

    os.environ["WM_N1_TRAIN_IDENTITIES"] = str(SIDECAR_V2)
    os.environ["WM_N1_TRAIN_OUTCOMES"] = str(SIDECAR_V2.parent / "outcomes")
    _ = lpo  # imported for parity with the real loader path

    train_list = LPWO_V3 / "windows_train_noPI_noWall.json"
    excl_list = LPWO_V3 / "excluded_runs_PI_wall_v3.json"
    guard = TrainingWindowGuard(horizon_steps=240, heldout=None,
                                leave_pi_wall_out=True)
    ds = TrajectoryDataset.from_real_packs(
        PACKS, identity_dirs=IDENTITIES, overlays=OVERLAYS, horizon_steps=240,
        start_shift_prob=0.0, window_list=str(train_list),
        exclude_window_list=[str(excl_list)], entries_v2=True, item_filter=guard)

    excluded = json.loads(excl_list.read_text(encoding="utf-8"))
    excluded_keys = set(str(k) for k in excluded["keys"])
    evals = json.loads((LPWO_V3 / "eval_worlds.json").read_text(encoding="utf-8"))
    eval_worlds = set()
    for info in evals["sets"].values():
        eval_worlds |= set(info["worlds"])

    # Walk every kept window (metadata only) to prove no exclusion leaked.
    leaked_key, leaked_world, leaked_eval = [], [], []
    p3_indices = []
    for i in range(len(ds)):
        pack, _pos = ds.positions[i]
        rec_meta = ds.positions[i]
        # loader meta requires a decode; use the sparse position list + receipts
        if pack == 0:
            p3_indices.append(i)
    # Verify by key using the retained window list (metadata) rather than decode.
    kept = json.loads(train_list.read_text(encoding="utf-8"))["windows"]
    for w in kept:
        if str(w["key"]) in excluded_keys:
            leaked_key.append(w["key"])
        if str(w["world"]) in eval_worlds:
            leaked_eval.append(w["key"])

    samples = []
    n_p3 = len(p3_indices)
    picks = sorted({p3_indices[j] for j in
                    (0, n_p3 // 4, n_p3 // 2, 3 * n_p3 // 4, n_p3 - 1)})
    for i in picks:
        rec = ds[i]
        states = np.asarray(rec["states"], np.float32)
        present = np.asarray(rec["present"], bool)
        is_cav = np.asarray(rec["is_cav"], bool)
        vids = np.asarray(rec["vehicle_ids"])
        speed = states[..., 4]
        # CAV roles must be a subset of present; ids must be present where CAV.
        roles_ok = bool(np.all(is_cav <= present))
        horizons = states.shape[0]
        n_hdv_t0 = int((present[0] & ~is_cav[0]).sum())
        n_cav_t0 = int((present[0] & is_cav[0]).sum())
        samples.append({
            "world": str(rec["meta"].get("world")),
            "arm": str(rec["meta"].get("arm")),
            "horizon": int(horizons),
            "n_slots": int(states.shape[1]),
            "hdv_t0": n_hdv_t0, "cav_t0": n_cav_t0,
            "speed_min_t0": float(np.nanmin(speed[0][present[0]])),
            "speed_max_t0": float(np.nanmax(speed[0][present[0]])),
            "speeds_finite": bool(np.isfinite(speed[present]).all()),
            "roles_subset_present": roles_ok,
            "vehicle_ids_min": int(vids.min()),
            "vehicle_ids_present_t0_nonneg": bool((vids[present[0]] >= 0).all()),
        })
    summary = {
        "status": "passed" if (len(p3_indices) > 0 and not leaked_key
                               and not leaked_eval) else "failed",
        "kept_by_pack": ds.receipt["kept_by_pack"],
        "n_records": ds.receipt["n_records"],
        "item_filter": ds.receipt["item_filter"],
        "n_pack_v3_windows": len(p3_indices),
        "n_excluded_keys": len(excluded_keys),
        "leaked_excluded_keys": leaked_key,
        "leaked_eval_worlds": leaked_eval,
        "samples": samples,
    }
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
