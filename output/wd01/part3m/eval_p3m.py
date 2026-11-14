"""WD-01 part 3m TASK D: L6 ladder rows on VALIDATION.

Rolls the five L6 members (fit_p3m.json, the part 3m refit with the
TTC<3 target and the fix knobs) over the 4 validation roots x t0 every
30 s (52 windows) x H in {30, 60}; raw per-(variant, member, root, t0,
H) metric rows are appended INCREMENTALLY to
results/raw/raw_metrics_p3m.jsonl BEFORE any aggregation.  L0/L4/L5 and
the derived L5c rung REUSE the part 3l raw rows (L0 is deterministic;
L4/L5 member rows are copied verbatim) -- no re-roll.

FREEZE GATE: refuses to run unless results/calibration_p3m.json matches
its FROZEN sha256 sidecar, i.e. the spread factors were frozen BEFORE
any validation row is produced (brief requirement; the calibration
split is TRAIN-internal and never overlapped with validation).

ZERO simulation; holdout blocked in root_arrays (never passed).
"""
from __future__ import annotations

import hashlib
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
import p3m_lib as M  # noqa: E402
import p3lib as P3  # noqa: E402

HORIZONS = (30, 60)
VAL_T0S = tuple(range(120, 481, 30))
K_TICKS = 120                     # 60 s rollout covers H <= 60 exactly
RAW_DIR = HERE / "results" / "raw"
CAL_JSON = HERE / "results" / "calibration_p3m.json"
CAL_FROZEN = HERE / "results" / "calibration_p3m.FROZEN.sha256"
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


def check_frozen():
    """Freeze gate: the calibration file must exist AND match its FROZEN
    sha256 sidecar (frozen before any validation rollout)."""
    if not CAL_JSON.exists() or not CAL_FROZEN.exists():
        raise SystemExit(f"FREEZE GATE: {CAL_JSON} / {CAL_FROZEN} missing "
                         f"-- run calibrate_p3m.py first; the spread "
                         f"factors must be frozen BEFORE validation")
    cal_text = CAL_JSON.read_text(encoding="utf-8")
    sha = hashlib.sha256(cal_text.encode("utf-8")).hexdigest()
    frozen = json.loads(CAL_FROZEN.read_text(encoding="utf-8"))
    if sha != frozen.get("sha256"):
        raise SystemExit(f"FREEZE GATE: calibration_p3m.json sha256 "
                         f"{sha} != FROZEN {frozen.get('sha256')} -- the "
                         f"calibration was modified after freezing")
    return frozen


def member_configs():
    fit = M.L3.load_fit(HERE / "results" / "fit_p3m.json")
    out = []
    for mi, mem in enumerate(fit["members"]):
        out.append((int(mem["seed"]), float(mem["lc_scale"]),
                    dict(mem["merge_params"]),
                    float(mem.get("dawdle_scale", 1.0))))
    return out


def run_l6():
    frozen = check_frozen()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "raw_metrics_p3m.jsonl"
    configs = member_configs()
    dev = P3.device()
    sp = M.L3.splits_3l()
    val_roots = sp["val"]
    print("priority below-normal:", below_normal_priority())
    print("=== WD-01 part 3m TASK D: L6 rows on VALIDATION ===")
    print("ZERO simulation; holdout blocked in root_arrays (never passed).")
    import torch
    print(f"device={dev} torch_threads={torch.get_num_threads()}")
    print(f"FREEZE GATE ok: calibration_p3m.json sha256 "
          f"{frozen['sha256'][:16]}... frozen_utc={frozen['frozen_utc']}")
    print(f"validation roots {val_roots}; t0 every 30 s {VAL_T0S} "
          f"({len(val_roots) * len(VAL_T0S)} windows); H {HORIZONS}; "
          f"L0/L4/L5 rows reused from part 3l (no re-roll)")
    print("L6 member configs (from fit_p3m.json):")
    for mi, (seed, lc, mp, dw) in enumerate(configs):
        print(f"  L6 member {mi}: seed={seed} lc_scale={lc} "
              f"merge_params={mp} dawdle_scale={dw}")

    n_rows = 0
    with open(raw_path, "w", encoding="utf-8") as raw_f:
        for mi, (seed, lc, mp, dw) in enumerate(configs):
            log(f"rolling L6 member {mi} (seed {seed}, dawdle {dw}) ...")
            for root in val_roots:
                ra = P3.root_arrays(root)
                for t0 in VAL_T0S:
                    r = E.Rollout(root, float(t0), kind="idm_merge",
                                  member_seed=seed, lc_scale=lc,
                                  merge_params=dict(mp), dawdle_scale=dw,
                                  dev=dev).run(n_ticks=K_TICKS)
                    ra_, k0 = r.ra, r.k0
                    snaps = D.snaps_to_dicts(r, K_TICKS)
                    births = D.births_from_engine(r, K_TICKS)
                    rows = D.truth_rows(ra_, k0, K_TICKS)
                    lc_t = D.lc_ticks_from_ra(ra_)
                    bstates = D.birth_states_from_ra(ra_, k0, K_TICKS)
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
                            variant="L6", member=mi, root=root,
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
                            engine="part3m knobs (end_force_s "
                                   f"{mp.get('end_force_s', 0.0)}, "
                                   f"v_merge_blend "
                                   f"{mp.get('v_merge_blend', 0.0)})")
                        raw_f.write(json.dumps(row, default=str) + "\n")
                        raw_f.flush()
                        n_rows += 1
            log(f"L6 member {mi}: done ({n_rows} rows total)")
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
        run_l6()
    finally:
        sys.stdout.file.close()
        sys.stdout = sys.stdout.stdout
