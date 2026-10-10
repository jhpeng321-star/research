"""WD-01 part 3k: TRAIN-root fit of the idm_merge parameters.

ZERO simulation; TRAIN roots only (holdout never passed; validation is
used ONLY by decompose_p3k.py / eval_p3k.py downstream, never here).

Fitted parameters (kind 'idm_merge', part 3k):
  tau_accept   s    -- M2 leader-gap acceptance threshold scale
  b_accept    m/s^2  -- M2 max follower required deceleration
  p_coop_scale --    -- M3 multiplier on the per-vType lcCooperative
                        prior (0.3/0.6/1.0); 0 disables cooperation
  lc_scale     --    -- M4 discretionary 0<->1 hazard scale (stage 2)

Fit targets (truth from root_arrays; pred from idm_merge rollouts;
evaluated on the first 4 sorted TRAIN roots x t0 {120, 300} s, H=60 s --
the p3j train-check convention):
  T1 accel-lane merge-zone occupancy (vehicle-steps with s >= 378.66 m)
  T2 accel-lane speed profile by 50 m d-bin (bins -50..300, n >= 100)
  T3 merge-point d (mean of accel->main_right merge d)
  T4 main_right mean speed with d in [-100, 100] m
  T5 window contract physics delay (dt - ds/limit, H=60)
  T6 lane-change count per window (all lanes)

Stage 1 grid (acceptance): tau_accept x b_accept with p_coop_scale = 0,
lc_scale = 0.003 (p3g train calibration); loss on T1+T2+T3.
Stage 2 grid (cooperation x hazard): p_coop_scale x lc_scale at the
stage-1 winner; loss on T4+T5+T6.
Members 1..4 for the M=5 ladder: bootstrap resamples of the 4 train
roots (numpy default_rng(200+m)), coordinate-neighbourhood refit
(+/- one stage-grid step per parameter, centre included), loss
stage1+stage2, evaluated on the t0=120 windows.

Every number in REPORT_P3K.md is pasted from this script's stdout
(results/stdout_fit.txt).
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("WD_DEVICE", "cpu")    # single process, shared box

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import decompose_p3k as D  # noqa: E402  (delay model + stats helpers)
import p3lib as P3  # noqa: E402

K = D.K_STEPS
EVAL_T0S = (120.0, 300.0)
BOOT_T0S = (120.0,)                # bootstrap eval windows (cost)
N_TRAIN_ROOTS = 4                  # p3j train-check convention
N_MEMBERS = 5
PROF_BINS = tuple(range(-50, 350, 50))
PROF_MIN_N = 60                     # pooled vehicle-steps per d-bin
TAU_GRID = (0.7, 1.0, 1.4, 1.9)
B_GRID = (2.0, 3.0, 4.5, 6.5)
P_COOP_GRID = (0.0, 0.25, 0.5, 0.75, 1.1)
LC_GRID = (0.002, 0.003, 0.005)

OUT = HERE / "results"
N_TRUTHS = {}                      # root -> t0 -> truth stats


def log(m):
    print(f"[{time.time()-D.T_START:6.1f}s] {m}", flush=True)


def truth_stats(root, t0, ra):
    k0 = int(round((t0 - 60.0) / 0.5))
    st = D.window_truth_stats(ra, k0, K)
    st["lc_all"] = int(((ra.lc_tick > k0) & (ra.lc_tick <= k0 + K)).sum())
    rows = D.truth_rows(ra, k0, K)
    recs_t, _p, _f = D.truth_records(
        rows, D.lc_ticks_from_ra(ra), D.birth_states_from_ra(ra, k0, K),
        k0, K)
    st["delay"] = sum(r_["delay"] for r_ in recs_t)
    return st


def pred_stats(variant, root, t0, dev, l4p, l4lc):
    r = D.rollout_for(variant, root, t0, dev, l4_params=l4p, l4_lc=l4lc)
    st = D.window_pred_stats(r, K)
    st["lc_all"] = int(r.lc_count)
    snaps = D.snaps_to_dicts(r, K)
    births = D.births_from_engine(r, K)
    ra = r.ra
    dp, _dtp, np_, _nt = D.phys_delay_sums(
        snaps, births, D.truth_rows(ra, r.k0, K), D.lc_ticks_from_ra(ra),
        D.birth_states_from_ra(ra, r.k0, K), r.k0, K)
    st["delay"] = dp
    return st


def loss_stage1(preds, truths):
    """Acceptance loss: zone occupancy + accel speed profile (POOLED over
    the eval windows: per-window accel bins are too sparse) + merge d."""
    occ_l = mg_l = 0.0
    pool = {}                       # db -> [sum_vp, n_p, sum_vt, n_t]
    n_w = 0
    for key, p in preds.items():
        t = truths[key]
        n_w += 1
        occ_l += ((p["occ"] - t["occ"]) / max(t["occ"], 50.0)) ** 2
        if p["merges"] and t["merges"]:
            mg_l += ((float(np.mean(p["merges"])) -
                      float(np.mean(t["merges"]))) / 75.0) ** 2
        elif bool(p["merges"]) != bool(t["merges"]):
            mg_l += 4.0                     # one side merged, other not
        for db in PROF_BINS:
            pp, pt_ = p["prof"].get(db), t["prof"].get(db)
            if pp:
                c = pool.setdefault(db, [0.0, 0, 0.0, 0])
                c[0] += pp[0]
                c[1] += pp[1]
            if pt_:
                c = pool.setdefault(db, [0.0, 0, 0.0, 0])
                c[2] += pt_[0]
                c[3] += pt_[1]
    prof_terms = [((c[0] / c[1] - c[2] / c[3]) / 5.0) ** 2
                  for c in pool.values()
                  if c[1] >= PROF_MIN_N and c[3] >= PROF_MIN_N]
    return (occ_l + sum(prof_terms) / max(len(prof_terms), 1)
            + mg_l) / max(n_w, 1), dict(
                occ=float(occ_l / max(n_w, 1)),
                prof=float(sum(prof_terms) / max(len(prof_terms), 1)),
                merge=float(mg_l / max(n_w, 1)))


def loss_stage2(preds, truths):
    """Cooperation/hazard loss: main_right merge-window speed + physics
    delay + LC count."""
    mr_l = d_l = lc_l = 0.0
    n_w = 0
    for key, p in preds.items():
        t = truths[key]
        n_w += 1
        if p["mr"][1] and t["mr"][1]:
            mr_l += ((p["mr"][0] / p["mr"][1] -
                      t["mr"][0] / t["mr"][1]) / 5.0) ** 2
        d_l += ((p["delay"] - t["delay"]) / 200.0) ** 2
        lc_l += ((p["lc_all"] - t["lc_all"]) / max(t["lc_all"], 10)) ** 2
    return (mr_l + d_l + lc_l) / max(n_w, 1), dict(
        mr=float(mr_l / max(n_w, 1)), delay=float(d_l / max(n_w, 1)),
        lc=float(lc_l / max(n_w, 1)))


def prof_str(d):
    out = []
    for db in PROF_BINS:
        pp = d["prof"].get(db)
        out.append(f"{pp[0]/pp[1]:4.1f}" if pp and pp[1] else "  . ")
    return " ".join(out)


def stats_table(preds, truths, title):
    print(f"\n  -- {title} (per eval window) --")
    print("  root             t0   occ_p occ_t | accel v (m/s) by d-bin "
          "-50..300 pred | truth | merge_d p/t   mr_spd p/t "
          "   delay_p  delay_t  lc p/t")
    for key in sorted(preds, key=lambda k: (k[0], k[1])):
        p, t = preds[key], truths[key]
        mg_p = f"{np.mean(p['merges']):6.1f}" if p["merges"] else "   n/a"
        mg_t = f"{np.mean(t['merges']):6.1f}" if t["merges"] else "   n/a"
        mr_p = f"{p['mr'][0]/p['mr'][1]:5.1f}" if p["mr"][1] else "  n/a"
        mr_t = f"{t['mr'][0]/t['mr'][1]:5.1f}" if t["mr"][1] else "  n/a"
        print(f"  {key[0]:<16} {key[1]:3.0f}  {p['occ']:5d} {t['occ']:5d} "
              f"| {prof_str(p)} | {prof_str(t)} | {mg_p}/{mg_t}  "
              f"{mr_p}/{mr_t} {p['delay']:9.1f} {t['delay']:8.1f}  "
              f"{p['lc_all']:3d}/{t['lc_all']:3d}")


def eval_stage1(grid, truths, dev):
    out = []
    for i, mp in enumerate(grid):
        preds = {}
        for root in sorted({k[0] for k in truths}):
            for t0 in EVAL_T0S:
                preds[(root, t0)] = pred_stats("L4", root, t0, dev, mp,
                                               0.003)
        loss, detail = loss_stage1(preds, truths)
        out.append((loss, detail, dict(mp)))
        log(f"stage1 {i+1}/{len(grid)} {mp} -> loss={loss:.4f} {detail}")
    return sorted(out, key=lambda x: x[0])


def eval_stage2(best1, truths, dev):
    out = []
    combos = [(pv, lc) for pv in P_COOP_GRID for lc in LC_GRID]
    for i, (pv, lc) in enumerate(combos):
        mp = dict(tau_accept=best1["tau_accept"],
                  b_accept=best1["b_accept"], p_coop_scale=pv)
        preds = {}
        for root in sorted({k[0] for k in truths}):
            for t0 in EVAL_T0S:
                preds[(root, t0)] = pred_stats("L4", root, t0, dev, mp, lc)
        loss, detail = loss_stage2(preds, truths)
        out.append((loss, detail, dict(mp, lc_scale=lc)))
        log(f"stage2 {i+1}/{len(combos)} {mp} lc_scale={lc} -> "
            f"loss={loss:.4f} {detail}")
    return sorted(out, key=lambda x: x[0])


def bootstrap_members(dev, best1, best2_full):
    """Members 1..4: bootstrap resample of the 4 train roots
    (rng 200+m), coordinate-neighbourhood refit around the stage winners
    (one grid step either side per parameter, centre included), loss
    stage1+stage2 on the t0=120 windows."""
    roots4 = sorted(N_TRUTHS)
    members = [dict(seed=1, lc_scale=best2_full["lc_scale"],
                    merge_params=dict(tau_accept=best2_full["tau_accept"],
                                      b_accept=best2_full["b_accept"],
                                      p_coop_scale=best2_full[
                                          "p_coop_scale"]))]
    for m in range(1, N_MEMBERS):
        rng = np.random.default_rng(200 + m)
        resample = [roots4[int(i)] for i in
                    rng.integers(0, len(roots4), len(roots4))]
        keys = [(r, t0) for r in resample for t0 in BOOT_T0S]
        tm = {k: N_TRUTHS[k[0]][k[1]] for k in keys}
        cands = []
        neigh = []
        for tv in TAU_GRID:
            neigh.append(dict(tau_accept=tv, b_accept=best1["b_accept"],
                              p_coop_scale=best2_full["p_coop_scale"]))
        for bv in B_GRID:
            neigh.append(dict(tau_accept=best1["tau_accept"], b_accept=bv,
                              p_coop_scale=best2_full["p_coop_scale"]))
        for pv in P_COOP_GRID:
            neigh.append(dict(tau_accept=best1["tau_accept"],
                              b_accept=best1["b_accept"],
                              p_coop_scale=pv))
        seen = set()
        for mp in neigh:
            key = tuple(sorted(mp.items()))
            if key in seen:
                continue
            seen.add(key)
            preds = {}
            for k in keys:
                preds[k] = pred_stats("L4", k[0], k[1], dev, mp,
                                      best2_full["lc_scale"])
            l1, d1 = loss_stage1(preds, tm)
            l2, d2 = loss_stage2(preds, tm)
            cands.append((l1 + l2, l1, l2, dict(mp)))
        for lc in LC_GRID:
            if lc == best2_full["lc_scale"]:
                continue
            preds = {}
            for k in keys:
                preds[k] = pred_stats("L4", k[0], k[1], dev,
                                      dict(best1, p_coop_scale=best2_full[
                                          "p_coop_scale"]), lc)
            l1, d1 = loss_stage1(preds, tm)
            l2, d2 = loss_stage2(preds, tm)
            cands.append((l1 + l2, l1, l2,
                          dict(best1, p_coop_scale=best2_full[
                              "p_coop_scale"], lc_scale=lc)))
        ranked = sorted(cands, key=lambda x: x[0])
        win = ranked[0]
        log(f"bootstrap member {m}: resample={sorted(set(resample))} -> "
            f"loss={win[0]:.4f} (s1 {win[1]:.4f} s2 {win[2]:.4f}) "
            f"params={win[3]}")
        mp_win = {k: v for k, v in win[3].items() if k != "lc_scale"}
        members.append(dict(seed=m * 17 + 1,
                            lc_scale=win[3].get("lc_scale",
                                                best2_full["lc_scale"]),
                            merge_params=mp_win))
    return members


def main():
    import torch
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout = D.Tee(OUT / "stdout_fit.txt")

    print("priority below-normal:", D.below_normal_priority())
    dev = P3.device()
    splits = P3.roots_by_split()
    train4 = sorted(splits["train"])[:N_TRAIN_ROOTS]
    val_roots = sorted(splits["val"])
    print("=== WD-01 part 3k: idm_merge parameter fit (TRAIN only) ===")
    print("ZERO simulation; holdout blocked in root_arrays (never "
          "passed).")
    print(f"device={dev} torch_threads={torch.get_num_threads()}")
    print(f"train fit roots (first {N_TRAIN_ROOTS} sorted): {train4}")
    print(f"eval windows: t0 {EVAL_T0S} s, H=60 s (K={K} steps of 0.5 s); "
          f"validation roots {val_roots} NOT used here")
    print(f"fit targets: T1 zone occupancy, T2 accel speed profile "
          f"(50 m bins, n>={PROF_MIN_N}), T3 merge-point d, T4 main_right "
          f"speed in [-100,100] m, T5 physics delay, T6 LC count")
    print(f"grids: tau_accept {TAU_GRID} x b_accept {B_GRID}; "
          f"p_coop_scale {P_COOP_GRID} x lc_scale {LC_GRID}")

    # ---- truth stats for the fit windows
    for root in train4:
        ra = P3.root_arrays(root)
        N_TRUTHS[root] = {}
        for t0 in EVAL_T0S:
            N_TRUTHS[root][t0] = truth_stats(root, t0, ra)
    truths = {(r_, t0): N_TRUTHS[r_][t0] for r_ in train4
              for t0 in EVAL_T0S}

    # ---- BEFORE stats: the current ladder variant L3 on the fit windows
    log("BEFORE: rolling L3 (idm_sampled, member 0, lc_scale=0.003) ...")
    preds_before = {}
    for root in train4:
        for t0 in EVAL_T0S:
            preds_before[(root, t0)] = pred_stats("L3", root, t0, dev,
                                                  None, 0.003)
    l1_b, d1_b = loss_stage1(preds_before, truths)
    l2_b, d2_b = loss_stage2(preds_before, truths)
    print(f"\nBEFORE (L3 = idm_sampled, member 0, lc_scale=0.003): "
          f"stage1 loss={l1_b:.4f} {d1_b} | stage2 loss={l2_b:.4f} "
          f"{d2_b}")
    stats_table(preds_before, truths, "BEFORE L3 (idm_sampled)")

    # ---- stage 1: acceptance grid
    grid1 = [dict(tau_accept=tv, b_accept=bv, p_coop_scale=0.0)
             for tv, bv in itertools.product(TAU_GRID, B_GRID)]
    log(f"STAGE 1: {len(grid1)} acceptance combos ...")
    r1 = eval_stage1(grid1, truths, dev)
    best1 = r1[0][2]
    print(f"\nSTAGE 1 winner: {best1} loss={r1[0][0]:.4f} {r1[0][1]}")

    # ---- stage 2: cooperation x hazard grid at the stage-1 winner
    log(f"STAGE 2: {len(P_COOP_GRID) * len(LC_GRID)} combos ...")
    r2 = eval_stage2(best1, truths, dev)
    best_full = r2[0][2]
    best2 = {k: v for k, v in best_full.items() if k != "lc_scale"}
    best_lc = best_full["lc_scale"]
    print(f"\nSTAGE 2 winner: merge_params={best2} lc_scale={best_lc} "
          f"loss={r2[0][0]:.4f} {r2[0][1]}")

    # ---- AFTER stats: fitted L4 on the same windows
    preds_after = {}
    for root in train4:
        for t0 in EVAL_T0S:
            preds_after[(root, t0)] = pred_stats("L4", root, t0, dev,
                                                 best2, best_lc)
    l1_a, d1_a = loss_stage1(preds_after, truths)
    l2_a, d2_a = loss_stage2(preds_after, truths)
    print(f"\nAFTER (L4 = idm_merge fitted): stage1 loss={l1_a:.4f} "
          f"{d1_a} | stage2 loss={l2_a:.4f} {d2_a}")
    stats_table(preds_after, truths, "AFTER L4 (idm_merge fitted)")

    # ---- bootstrap members for the M=5 ladder
    log("BOOTSTRAP members 1..4 ...")
    members = bootstrap_members(dev, best1, best_full)
    print("\nselected member parameters (member 0 = main fit):")
    for mi, mem in enumerate(members):
        print(f"  member {mi}: seed={mem['seed']} lc_scale="
              f"{mem['lc_scale']} merge_params={mem['merge_params']}")

    result = dict(
        variant="L4 kind='idm_merge'",
        fitted_on=dict(roots=train4, t0s=list(EVAL_T0S), H=60.0,
                       note="TRAIN only; holdout never accessed"),
        selected=dict(merge_params=best2, lc_scale=best_lc,
                      stage1_loss=r1[0][0], stage2_loss=r2[0][0],
                      before=dict(l3_stage1=l1_b, l3_stage2=l2_b,
                                  detail1=d1_b, detail2=d2_b),
                      after=dict(stage1=l1_a, stage2=l2_a,
                                 detail1=d1_a, detail2=d2_a)),
        members=members,
        grids=dict(tau_accept=list(TAU_GRID), b_accept=list(B_GRID),
                   p_coop_scale=list(P_COOP_GRID),
                   lc_scale=list(LC_GRID)),
    )
    (OUT / "fit_p3k.json").write_text(
        json.dumps(result, indent=1, default=str), encoding="utf-8")
    log(f"wrote {OUT / 'fit_p3k.json'} and {OUT / 'stdout_fit.txt'}")
    print("\nDONE part 3k fit.")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
