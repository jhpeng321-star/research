"""WD-01 part 3n TASK A: gate-v2 (b) lane-change diagnosis on TRAIN,
BY TYPE (mandatory accel->main_right merge vs discretionary 0<->1) and
BY MEMBER, plus the raw TRAIN-internal rows the rest of the part
reuses.

The (b) residual of the part 3m refit: on VALIDATION L6 overshoots
lane changes +39.6 % (16.87 vs 12.08 per window, H=30) while the same
fit sat -11/-19 % LOW on the TRAIN fit windows.  This script measures
the by-type split ON TRAIN ONLY:

  - rolls the five part 3m L6 members over ALL 8 TRAIN roots (4 fit +
    4 calibration) x t0 every 60 s (56 windows), K=120; writes the
    standard raw metric rows (H in {30, 60}) to results/raw/
    raw_train_internal_p3n.jsonl -- the calibration-root subset of
    these rows IS the iteration-0 TRAIN-internal gate roll (no
    re-roll), and their truth sides feed the TRAIN climatology;
  - writes per-(root, t0, member, H) LC counts BY TYPE on both sides
    to results/raw/raw_lc_type_p3n.jsonl;
  - rolls L0 (idm_normal, deterministic) on the same windows: the
    (d) speed reference for the TRAIN-internal gate;
  - prints the by-type x member x root tables and writes
    results/diag_p3n.json.

The part 3m member configs come from part 3m's committed fit_p3m.json
(read-only).  ZERO simulation; holdout blocked in root_arrays (never
passed); validation never touched here.
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
import p3n_lib as N  # noqa: E402
import decompose_p3k as D  # noqa: E402  (Tee, below_normal_priority)
import p3lib as P3  # noqa: E402
import engine as E  # noqa: E402

RAW_DIR = HERE / "results" / "raw"
T_START = time.time()
FIT_ROOTS = None


def log(m):
    print(f"[{time.time()-T_START:6.1f}s] {m}", flush=True)


def l6_member_configs():
    """The five part 3m L6 members (the iteration-0 model, read-only)."""
    fit = N.load_json(HERE.parent / "part3m" / "results" / "fit_p3m.json")
    out = []
    for mi, mem in enumerate(fit["members"]):
        out.append((mi, int(mem["seed"]), float(mem["lc_scale"]),
                    dict(mem["merge_params"]),
                    float(mem.get("dawdle_scale", 1.0))))
    return out


def lc_type_rows(r, ra, mi, root, t0):
    """Per-(root, t0, member, H) LC-by-type counts on both sides."""
    pred = N.pred_lc_by_type(r.snapshots, N.K_TICKS)
    truth = N.truth_lc_by_type(ra, r.k0, N.K_TICKS)
    out = []
    for H in N.HORIZONS:
        out.append(dict(member=mi, root=root, t0=int(t0), H=H,
                        split="fit" if root in FIT_ROOTS else
                        "calibration",
                        **pred[H], **truth[H]))
    return out


def agg_lc(rows, sel):
    """Mean per-window LC by type over the selected lc-type rows;
    totals derived from the two types (rows carry no total column)."""
    sel = [r for r in rows if sel(r)]
    n = max(len(sel), 1)
    out = {k: float(np.mean([r[k] for r in sel]))
           for k in ("pred_merge", "pred_disc", "true_merge",
                     "true_disc")}
    out["pred_lc"] = round(out["pred_merge"] + out["pred_disc"], 3)
    out["true_lc"] = round(out["true_merge"] + out["true_disc"], 3)
    for k in ("pred_merge", "pred_disc", "true_merge", "true_disc"):
        out[k] = round(out[k], 3)
    out["n_windows"] = len(sel)
    return out


def show(tag, t):
    dev_ = (t["pred_lc"] - t["true_lc"]) / max(t["true_lc"], 1)
    print(f"   {tag:<22} predM {t['pred_merge']:5.2f} predD "
          f"{t['pred_disc']:5.2f} | trueM {t['true_merge']:5.2f} trueD "
          f"{t['true_disc']:5.2f} | {t['pred_lc']:5.2f}/"
          f"{t['true_lc']:5.2f} ({100 * dev_:+.1f}%)")


def main():
    import torch
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    global FIT_ROOTS
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = D.Tee(HERE / "results" / "stdout_diag.txt")
    sp = N.L3.splits_3l()
    FIT_ROOTS = tuple(sp["fit"])
    cal_roots = tuple(sp["calibration"])
    train_roots = tuple(sorted(FIT_ROOTS + cal_roots))
    configs = l6_member_configs()
    dev = P3.device()
    print("priority below-normal:", D.below_normal_priority())
    print("=== WD-01 part 3n TASK A: gate-v2 (b) LC diagnosis on TRAIN "
          "(by type x member) ===")
    print("ZERO simulation; holdout blocked in root_arrays (never "
          "passed).")
    print(f"device={dev} torch_threads={torch.get_num_threads()}")
    print(f"TRAIN roots: fit {FIT_ROOTS} + calibration {cal_roots}; "
          f"t0 {N.GATE_T0S} ({len(train_roots) * len(N.GATE_T0S)} "
          f"windows); K={N.K_TICKS} (H<=60); VALIDATION {sp['val']} "
          f"NOT touched here; holdout blocked")
    print("part 3m L6 member configs (fit_p3m.json, read-only):")
    for mi, seed, lc, mp, dw in configs:
        print(f"  member {mi}: seed={seed} lc_scale={lc} "
              f"merge_params={mp} dawdle_scale={dw}")

    raw_path = RAW_DIR / "raw_train_internal_p3n.jsonl"
    lc_path = RAW_DIR / "raw_lc_type_p3n.jsonl"
    n_rows = n_lc = 0
    with open(raw_path, "w", encoding="utf-8") as rf, \
            open(lc_path, "w", encoding="utf-8") as lf:
        for mi, seed, lc, mp, dw in configs:
            log(f"rolling L6(p3m) member {mi} (seed {seed}, dawdle "
                f"{dw}) over 8 TRAIN roots ...")
            for root in train_roots:
                ra = P3.root_arrays(root)
                for t0 in N.GATE_T0S:
                    r = N.P3_roll(root, float(t0), mp, lc, dev, dw,
                                  seed=seed)
                    note = (f"part3n iter0: engine UNCHANGED; p3m L6 "
                            f"member {mi} (end_force_s "
                            f"{mp.get('end_force_s', 0.0)}, blend "
                            f"{mp.get('v_merge_blend', 0.0)})")
                    for row in N.window_raw_rows(r, "L6", mi, root, t0,
                                                 dw, note):
                        rf.write(json.dumps(row, default=str) + "\n")
                        n_rows += 1
                    for row in lc_type_rows(r, ra, mi, root, t0):
                        lf.write(json.dumps(row, default=str) + "\n")
                        n_lc += 1
                    rf.flush()
                    lf.flush()
            log(f"  member {mi}: done ({n_rows} metric rows, {n_lc} "
                f"lc-type rows)")
        log("rolling L0 (idm_normal) on the same 8 TRAIN roots ...")
        for root in train_roots:
            for t0 in N.GATE_T0S:
                r = E.Rollout(root, float(t0), kind="idm_normal",
                              dev=dev).run(n_ticks=N.K_TICKS)
                for row in N.window_raw_rows(r, "L0", 0, root, t0, 1.0,
                                             "part3n: L0 idm_normal "
                                             "(deterministic; engine "
                                             "unchanged)"):
                    rf.write(json.dumps(row, default=str) + "\n")
                    n_rows += 1
        rf.flush()
    log(f"wrote {n_rows} raw metric rows -> {raw_path.name}; {n_lc} "
        f"lc-type rows -> {lc_path.name}")

    # ---- diagnosis tables (per type x group x member x root, H30/60)
    lc_rows = [json.loads(ln) for ln in
               lc_path.read_text(encoding="utf-8").splitlines() if
               ln.strip()]
    diag = dict(
        part="3n", task="A",
        data="8 TRAIN roots x 7 t0 x 5 p3m L6 members; TRAIN only",
        holdout_accessed=False, simulation=False, validation_used=False,
        by_group={}, by_member={}, by_root={})
    for H in N.HORIZONS:
        print(f"\n== LC per window BY TYPE and BY GROUP (H={H}) ==")
        for grp, roots in (("fit", FIT_ROOTS), ("calibration", cal_roots)):
            t = agg_lc(lc_rows, lambda r, g=roots, h=H:
                       r["root"] in g and r["H"] == h)
            diag["by_group"][f"{grp}@H{H}"] = t
            show(f"{grp}", t)
        for mi, *_rest in configs:
            for grp, roots in (("fit", FIT_ROOTS),
                               ("calibration", cal_roots)):
                t = agg_lc(lc_rows, lambda r, g=roots, m=mi, h=H:
                           r["root"] in g and r["member"] == m and
                           r["H"] == h)
                diag["by_member"][f"{grp}_m{mi}@H{H}"] = t
                if H == 30:
                    show(f"{grp} member {mi}", t)
    print("\n== per root, all members pooled (H=30 primary | H=60) ==")
    for root in train_roots:
        t30 = agg_lc(lc_rows, lambda r, g=root:
                     r["root"] == g and r["H"] == 30)
        t60 = agg_lc(lc_rows, lambda r, g=root:
                     r["root"] == g and r["H"] == 60)
        diag["by_root"][f"{root}@H30"] = t30
        diag["by_root"][f"{root}@H60"] = t60
        tag = "fit" if root in FIT_ROOTS else "CAL"
        dev30 = (t30["pred_lc"] - t30["true_lc"]) / \
            max(t30["true_lc"], 1)
        dev60 = (t60["pred_lc"] - t60["true_lc"]) / \
            max(t60["true_lc"], 1)
        print(f"   [{tag:<3}] {root:<16} H30 {t30['pred_lc']:5.2f}/"
              f"{t30['true_lc']:5.2f} ({100 * dev30:+.1f}%)  "
              f"[M {t30['pred_merge']:.1f}/{t30['true_merge']:.1f} "
              f"D {t30['pred_disc']:.1f}/{t30['true_disc']:.1f}]  "
              f"H60 {t60['pred_lc']:5.2f}/{t60['true_lc']:5.2f} "
              f"({100 * dev60:+.1f}%)")

    print("\nNOTE: the TRAIN-internal gate (2-fold) for THIS model is "
          "computed from the calibration-root subset of "
          f"{raw_path.name} by gate_p3n.py --iter 0 (no re-roll).")

    (HERE / "results" / "diag_p3n.json").write_text(
        json.dumps(diag, indent=1, default=str), encoding="utf-8")
    log("wrote results/diag_p3n.json")
    print("\nDONE part 3n TASK A diagnosis.")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
