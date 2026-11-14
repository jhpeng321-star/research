"""A2 validation: 30 s commit-window rollouts of engine kind
'idm_action' against chunk-1 TRAIN branch truth.

For every TRAIN world (complete receipts per arm) and every arm in
{native, pi, fixed1, fixed2, switching1..3}: roll the engine from the
branch restore state for 60 x 0.5 s steps with the branch's dispatcher
action schedule replayed through the partA_ctrl emulators (free rollout,
no teacher forcing), plus the CAVs-as-native NAIVE baseline of the same
branch world (force_native; per arm, because the branch arrival
realisation is part of the branch world).  Raw per-(world, arm) rows are
appended incrementally to results/raw/raw_metrics_act.jsonl.

Metrics per row: id-matched CAV / human speed RMSE, contract physics
delay (pred/true), TTC<3 exposure (pred / matched-truth, frozen
window_metrics conventions), predicted collisions per 1000
vehicle-steps, truth collisions in the window.  Aggregation
(aggregate_act.py) builds the per-arm tables and the PAIRED deltas
(arm - native, predicted vs true, sign agreement / correlation / error)
against the naive baseline.

ZERO simulation; TRAIN only (ActWorld refuses HOLDOUT/CALIBRATION);
single process, below-normal priority (box shared with chunk-2 SUMO).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("WD_DEVICE", "cpu")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import act_lib as L  # noqa: E402
import act_engine as AE  # noqa: E402

RAW_DIR = HERE / "results" / "raw"
T_START = time.time()


def log(m):
    print(f"[{time.time()-T_START:6.1f}s] {m}", flush=True)


def below_normal_priority():
    if sys.platform == "win32":
        try:
            import ctypes
            h = ctypes.windll.kernel32.GetCurrentProcess()
            return bool(ctypes.windll.kernel32.SetPriorityClass(h, 0x4000))
        except Exception:
            return False
    return False


def births_name_map(births):
    """{j: {name: (lane, s)}} for the physics-delay pred side."""
    out = {}
    for j, rows in births.items():
        out[j] = {x[0]: (x[1], x[2]) for x in rows}
    return out


def truth_collisions_in_window(coll):
    """Number of truth collision records in the window (each record is
    (t, colliding_ids); count the ids per record)."""
    return sum(len(ids) for _t, ids in coll)


def main():
    print("priority below-normal:", below_normal_priority())
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    worlds = L.train_worlds()
    names = sorted(worlds)
    print("=== WD-01 partA_act A2 validation: 30 s commit windows on "
          "chunk-1 TRAIN ===")
    print("ZERO simulation; TRAIN only; HOLDOUT/CALIBRATION never opened.")
    print(f"device={torch.device('cpu')} torch_threads={torch.get_num_threads()}")
    print(f"{len(names)} TRAIN worlds x 7 arms (complete receipts only); "
          f"H=30 s from the branch t0; engine kind 'idm_action' "
          f"(exec rule {AE.EXEC_RULE}, p3m selected member knobs) + "
          f"per-arm CAVs-as-native naive baseline")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "raw_metrics_act.jsonl"
    n_rows = 0
    skipped = {}
    with open(raw_path, "w", encoding="utf-8") as raw_f:
        for wi, world in enumerate(names):
            t0w = float(worlds[world]["t0_s"])
            try:
                w = L.ActWorld(world, t0w)
            except Exception as e:                  # noqa: BLE001
                skipped[world] = str(e)
                continue
            pen = L.penetration_of(world)
            for arm in L.BRANCH_ARMS:
                if arm not in w.arms:
                    skipped[f"{world}__{arm}"] = "incomplete_receipt"
                    continue
                truth = w.truth_rows(arm)
                births = w.births(arm)
                bmap = births_name_map(births)
                cav_names = {n for n in w.cav_ids}
                truth_coll = truth_collisions_in_window(
                    w.truth_collisions(arm))
                for mode, r in (
                    ("act", AE.ActionRollout(w, arm, member_seed=0)),
                    ("naive", AE.ActionRollout(w, arm, member_seed=0,
                                               force_native=True)),
                ):
                    r.run(L.H30_TICKS)
                    snaps = r.snap_dicts()
                    dp, dt_, np_, nt_ = L.phys_delay(snaps, bmap, truth,
                                                     L.H30_TICKS)
                    m = L.window_metrics_act(snaps, truth, L.H30_TICKS,
                                             cav_names)
                    row = dict(
                        world=world, arm=arm, t0=t0w, H=30.0,
                        variant=mode, penetration=pen,
                        exec_rule=AE.EXEC_RULE,
                        speed_rmse_cav=m["speed_rmse_cav"],
                        speed_rmse_human=m["speed_rmse_human"],
                        n_matched_cav=m["n_matched_cav"],
                        n_matched_human=m["n_matched_human"],
                        delay_phys_pred=dp, delay_phys_true=dt_,
                        phys_steps_pred=np_, phys_steps_true=nt_,
                        pred_ttc3_exposure=m["pred_ttc3_exposure"],
                        true_ttc3_exposure=m["true_ttc3_exposure"],
                        pred_min_ttc=m["pred_min_ttc"],
                        true_min_ttc=m["true_min_ttc"],
                        pred_collisions=r.collisions,
                        collisions_per_1000=r.collisions
                        / max(m["pred_vehicle_steps"], 1) * 1000.0,
                        truth_collisions_in_window=truth_coll,
                        pred_lc=r.lc_count, n_birthed=r.n_birthed,
                        n_exited=r.n_exited, n_commands=r.n_commands,
                        pred_vehicle_steps=m["pred_vehicle_steps"],
                        anomalies=sum(r.anomalies.values()),
                    )
                    raw_f.write(json.dumps(row, default=str) + "\n")
                    raw_f.flush()
                    n_rows += 1
            if (wi + 1) % 5 == 0:
                log(f"[{wi+1}/{len(names)} worlds] {n_rows} rows")
    log(f"done: {n_rows} raw rows -> {raw_path}")
    if skipped:
        sk_path = HERE / "results" / "skipped.json"
        sk_path.write_text(json.dumps(skipped, indent=1),
                           encoding="utf-8")
        log(f"{len(skipped)} skipped (world, arm) entries -> {sk_path}")


if __name__ == "__main__":
    main()
