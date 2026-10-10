"""WD-01 part 3k: gate v2 ladder evaluation on VALIDATION.

Variants: L0 = kind 'idm_normal' (deterministic, 1 member);
L3 = kind 'idm_sampled' (M=5 members, seeds m*17+1, lc_scale=0.003 --
the p3h setup); L4 = kind 'idm_merge' (M=5 members, seeds m*17+1 with
bootstrap-refitted merge parameters from results/fit_p3k.json, member 0
= the main TRAIN fit).

Grid: the 4 validation roots x t0 every 30 s (52 windows) x H in
{30, 60, 90, 120} s.  Raw per-(variant, member, root, t0, H) metric rows
are appended INCREMENTALLY to results/raw/raw_metrics.jsonl BEFORE any
aggregation (aggregation is aggregate_p3k.py; raw rows are saved first).

Per row, gate criterion (c) uses the CONTRACT PHYSICS DELAY
(dt - ds/limit, the p3j/p3k decompose model, computed at that H from the
same rollout snapshots) in fields delay_phys_pred / delay_phys_true;
the exposure-based delay fields (delay_pred / delay_true, 0.5 s x
vehicle count, the p3h metric) are also saved for continuity.

ZERO simulation; holdout blocked in root_arrays (never passed).
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("WD_DEVICE", "cpu")    # single process, shared box

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import decompose_p3k as D  # noqa: E402
import engine as E  # noqa: E402
import p3lib as P3  # noqa: E402

HORIZONS = (30, 60, 90, 120)
VAL_T0S = tuple(range(120, 481, 30))
N_MEMBERS = 5
RAW_DIR = HERE / "results" / "raw"
LC_SCALE_L3 = D.LC_SCALE_L3
T_START = time.time()


def log(m):
    print(f"[{time.time()-T_START:6.1f}s] {m}", flush=True)


def below_normal_priority():
    if sys.platform == "win32":
        try:
            h = ctypes.windll.kernel32.GetCurrentProcess()
            return bool(ctypes.windll.kernel32.SetPriorityClass(h, 0x4000))
        except Exception:
            return False
    return False


def member_configs():
    """[(variant, member_index, seed, lc_scale, merge_params)] for the
    ladder: L0 member 0 deterministic; L3/L4 members 0..4."""
    fit = json.loads((HERE / "results" / "fit_p3k.json").read_text(
        encoding="utf-8"))
    members = fit["members"]
    assert len(members) == N_MEMBERS, len(members)
    out = [("L0", 0, 1, 0.0, None)]
    for mi, mem in enumerate(members):
        out.append(("L3", mi, int(mem["seed"]), LC_SCALE_L3, None))
        out.append(("L4", mi, int(mem["seed"]), float(mem["lc_scale"]),
                    dict(mem["merge_params"])))
    return out


def run_ladder():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "raw_metrics.jsonl"
    configs = member_configs()
    dev = P3.device()
    splits = P3.roots_by_split()
    val_roots = sorted(splits["val"])
    print("=== WD-01 part 3k: gate v2 ladder (VALIDATION only) ===")
    print("ZERO simulation; holdout blocked in root_arrays (never passed).")
    print(f"device={dev} below-normal priority={below_normal_priority()}")
    print(f"validation roots {val_roots}; t0 every 30 s {VAL_T0S} "
          f"({len(val_roots) * len(VAL_T0S)} windows); "
          f"H {HORIZONS}")
    print("member configs:")
    for c in configs:
        print(f"  {c[0]} member {c[1]}: seed={c[2]} lc_scale={c[3]} "
              f"merge_params={c[4]}")

    with open(raw_path, "a", encoding="utf-8") as raw_f:
        n_rows = 0
        for variant, mi, seed, lc_scale, mp in configs:
            log(f"rolling {variant} member {mi} (seed {seed}) ...")
            for root in val_roots:
                for t0 in VAL_T0S:
                    if variant == "L0":
                        r = E.Rollout(root, float(t0), kind="idm_normal",
                                      dev=dev).run()
                    elif variant == "L3":
                        r = E.Rollout(root, float(t0), kind="idm_sampled",
                                      member_seed=seed, lc_scale=lc_scale,
                                      dev=dev).run()
                    else:
                        r = E.Rollout(root, float(t0), kind="idm_merge",
                                      member_seed=seed, lc_scale=lc_scale,
                                      merge_params=mp, dev=dev).run()
                    ra, k0 = r.ra, r.k0
                    # contract physics delay per H from the same rollout
                    snaps = D.snaps_to_dicts(r, P3.ROLL_TICKS)
                    births = D.births_from_engine(r, P3.ROLL_TICKS)
                    rows = D.truth_rows(ra, k0, P3.ROLL_TICKS)
                    lc_t = D.lc_ticks_from_ra(ra)
                    bstates = D.birth_states_from_ra(ra, k0,
                                                     P3.ROLL_TICKS)
                    phys = {}
                    for H in HORIZONS:
                        K = int(round(H / 0.5))
                        dp, dt_, np_, nt_ = D.phys_delay_sums(
                            snaps, births, rows, lc_t, bstates, k0, K)
                        phys[H] = (dp, dt_, np_, nt_)
                    for H in HORIZONS:
                        m = E.window_metrics(r.snapshots, ra, k0, H)
                        dp, dt_, np_, nt_ = phys[H]
                        row = dict(
                            variant=variant, member=mi, root=root, t0=t0,
                            H=H,
                            speed_rmse=m["speed_rmse"],
                            pos_rmse=m["pos_rmse"],
                            collisions_per_1000=m["collisions_per_1000"],
                            pred_lc=m["pred_lc"], true_lc=m["true_lc"],
                            # exposure-based delay (p3h metric, continuity)
                            delay_pred=m["delay_pred"],
                            delay_true=m["delay_true"],
                            pred_vehicle_steps=m["pred_vehicle_steps"],
                            matched_vehicle_steps=m[
                                "matched_vehicle_steps"],
                            # contract physics delay (gate c, p3j model)
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
                            lane_frac=m["lane_frac"])
                        raw_f.write(json.dumps(row, default=str) + "\n")
                        raw_f.flush()
                        n_rows += 1
            log(f"{variant} member {mi}: done ({n_rows} rows total)")
    log(f"raw rows appended to {raw_path} (total this run: {n_rows})")


if __name__ == "__main__":
    try:
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    run_ladder()
