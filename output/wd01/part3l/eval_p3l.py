"""WD-01 part 3l TASK D: gate v2 ladder evaluation on VALIDATION.

Variants:
  L0  kind 'idm_normal', member 0 (deterministic);
  L4  kind 'idm_merge' with the p3k fit member PARAMETERS -- NOTE: the
      part 3l TASK A follower-slot fix is engine-global, so this rung
      is "p3k params on the FIXED engine", NOT the p3k published
      behaviour; differences vs the p3k gate table are attributable to
      the slot fix (its over-merging inflated LC counts);
  L5  kind 'idm_merge' with the part 3l fit members (fit_p3l.json) --
      fixed nearest-behind follower slot, extended grids, LC refitted
      at H=30, wider-draw members incl. dawdle widths;
  L5c is DERIVED at aggregation time (L5 rows + the TASK C spread
      calibration from calibration_p3l.json); no extra rollouts.

Grid: the 4 validation roots x t0 every 30 s (52 windows) x H in
{30, 60, 90, 120} s; raw per-(variant, member, root, t0, H) metric
rows are appended INCREMENTALLY to results/raw/raw_metrics.jsonl
BEFORE any aggregation.

ZERO simulation; holdout blocked in root_arrays (never passed).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("WD_DEVICE", "cpu")    # single process, shared box

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "part3k"))
import decompose_p3k as D  # noqa: E402
import engine as E  # noqa: E402
import p3l_lib as L  # noqa: E402
import p3lib as P3  # noqa: E402

HORIZONS = (30, 60, 90, 120)
VAL_T0S = tuple(range(120, 481, 30))
RAW_DIR = HERE / "results" / "raw"
P3K_FIT = HERE.parent / "part3k" / "results" / "fit_p3k.json"
LC_SCALE_L3 = D.LC_SCALE_L3
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


def member_configs():
    """[(variant, member, seed, lc_scale, merge_params, dawdle_scale)]."""
    p3k = L.load_fit(P3K_FIT)
    p3l = L.load_fit(HERE / "results" / "fit_p3l.json")
    out = [("L0", 0, 1, 0.0, None, 1.0)]
    for mi, mem in enumerate(p3k["members"]):
        out.append(("L4", mi, int(mem["seed"]), float(mem["lc_scale"]),
                    dict(mem["merge_params"]), 1.0))
    for mi, mem in enumerate(p3l["members"]):
        out.append(("L5", mi, int(mem["seed"]), float(mem["lc_scale"]),
                    dict(mem["merge_params"]),
                    float(mem.get("dawdle_scale", 1.0))))
    return out


def run_ladder():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "raw_metrics.jsonl"
    configs = member_configs()
    dev = P3.device()
    splits = L.splits_3l()
    val_roots = splits["val"]
    print("priority below-normal:", below_normal_priority())
    print("=== WD-01 part 3l TASK D: gate v2 ladder (VALIDATION only) ===")
    print("ZERO simulation; holdout blocked in root_arrays (never passed).")
    import torch
    print(f"device={dev} torch_threads={torch.get_num_threads()}")
    print(f"validation roots {val_roots}; t0 every 30 s {VAL_T0S} "
          f"({len(val_roots) * len(VAL_T0S)} windows); H {HORIZONS}")
    print("member configs (L4 = p3k reference incl. its follower-slot "
          "defect; L5 = part 3l fit, fixed slot):")
    for c in configs:
        print(f"  {c[0]} member {c[1]}: seed={c[2]} lc_scale={c[3]} "
              f"merge_params={c[4]} dawdle_scale={c[5]}")

    with open(raw_path, "w", encoding="utf-8") as raw_f:
        n_rows = 0
        for variant, mi, seed, lc_scale, mp, dw in configs:
            log(f"rolling {variant} member {mi} (seed {seed}, "
                f"dawdle {dw}) ...")
            for root in val_roots:
                ra = P3.root_arrays(root)
                for t0 in VAL_T0S:
                    if variant == "L0":
                        r = E.Rollout(root, float(t0), kind="idm_normal",
                                      dev=dev).run()
                    else:
                        r = E.Rollout(root, float(t0), kind="idm_merge",
                                      member_seed=seed, lc_scale=lc_scale,
                                      merge_params=mp, dawdle_scale=dw,
                                      dev=dev).run()
                    ra_, k0 = r.ra, r.k0
                    snaps = D.snaps_to_dicts(r, P3.ROLL_TICKS)
                    births = D.births_from_engine(r, P3.ROLL_TICKS)
                    rows = D.truth_rows(ra_, k0, P3.ROLL_TICKS)
                    lc_t = D.lc_ticks_from_ra(ra_)
                    bstates = D.birth_states_from_ra(ra_, k0,
                                                     P3.ROLL_TICKS)
                    phys = {}
                    for H in HORIZONS:
                        kh = int(round(H / 0.5))
                        dp, dt_, np_, nt_ = D.phys_delay_sums(
                            snaps, births, rows, lc_t, bstates, k0, kh)
                        phys[H] = (dp, dt_, np_, nt_)
                    for H in HORIZONS:
                        m = E.window_metrics(r.snapshots, ra_, k0, H)
                        dp, dt_, np_, nt_ = phys[H]
                        row = dict(
                            variant=variant, member=mi, root=root,
                            t0=int(t0), H=H,
                            speed_rmse=m["speed_rmse"],
                            pos_rmse=m["pos_rmse"],
                            collisions_per_1000=m["collisions_per_1000"],
                            pred_lc=m["pred_lc"], true_lc=m["true_lc"],
                            delay_pred=m["delay_pred"],
                            delay_true=m["delay_true"],
                            pred_vehicle_steps=m["pred_vehicle_steps"],
                            matched_vehicle_steps=m[
                                "matched_vehicle_steps"],
                            delay_phys_pred=dp, delay_phys_true=dt_,
                            phys_steps_pred=np_, phys_steps_true=nt_,
                            pred_ttc3_exposure=m["pred_ttc3_exposure"],
                            true_ttc3_exposure=m["true_ttc3_exposure"],
                            pred_hard_brake_s=m["pred_hard_brake_s"],
                            true_hard_brake_s=m["true_hard_brake_s"],
                            any_ttc3_pred=m["any_ttc3_pred"],
                            any_ttc3_true=m["any_ttc3_true"],
                            any_hb_pred=m["any_hb_pred"],
                            any_hb_true=m["any_hb_true"],
                            lane_frac=m["lane_frac"],
                            dawdle_scale=dw,
                            engine="part3l follower-slot fix, "
                            + ("p3l fit params" if variant == "L5"
                               else "p3k params"))
                        raw_f.write(json.dumps(row, default=str) + "\n")
                        raw_f.flush()
                        n_rows += 1
            log(f"{variant} member {mi}: done ({n_rows} rows total)")
    log(f"raw rows written to {raw_path} (total this run: {n_rows})")


if __name__ == "__main__":
    try:
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    sys.stdout = D.Tee(HERE / "results" / "stdout_eval.txt")
    try:
        run_ladder()
    finally:
        sys.stdout.file.close()
        sys.stdout = sys.stdout.stdout
