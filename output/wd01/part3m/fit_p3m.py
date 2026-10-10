"""WD-01 part 3m TASK B: TRAIN refit of the idm_merge parameters with an
ADDED TTC<3 s EXPOSURE TARGET and the part 3m fix knobs, after TASK A
located the excess exposure.

Differences vs the part 3l fit (fit_p3l.py, same 8 TRAIN fit windows):
  - b_accept grid EXTENDED UP to {2.0, 3.0, 4.0, 5.0} (brief) -- the
    b=1.5 low end is dropped, p3l measured it an order of magnitude
    worse (stage-1 loss 8.1-8.8 vs 0.92-1.57 at b=3.0) and TASK A shows
    the excess-exposure fix direction needs the ramp to discharge MORE,
    not less;
  - NEW end_force_s grid {0.0, 150.0} (stage 2, with the blend knobs):
    near-lane-end acceptance relaxation so the ramp discharges with
    rolling merges instead of the stop-and-go queue at the lane-end
    obstacle (TASK A: the excess concentrates DOWNSTREAM, d < 0, on the
    accel lane queue and its late merges);
  - NEW v_merge_blend grid {0.0, 0.5, 1.0} (stage 2): post-merge speed
    relaxation candidate;
  - stage-2 loss ADDS the TTC<3 exposure target
    ((pred - truth)/max(truth, 5))^2 per window (truth =
    p3m_lib.truth_ttc3_exposure over ALL truth rows -- a fixed number
    per window, no matched-set dependence);
  - lc_scale and p_coop re-fitted at the stage-2 winner (stage 3), then
    the acceptance grid is re-swept jointly (stage 4);
  - members 1-4: the p3l wider-draw mechanism (bootstrap resample of
    the fit roots, per-coordinate softmax draws T=0.15 over the
    coordinate loss lines, independent dawdle draw), now over the
    EXTENDED coordinate grids incl. the two new knobs;
  - the M2 follower slot stays the part 3l FIXED nearest-behind.

Protocol is TRAIN-only; the calibration split and validation are never
rolled here; holdout blocked in root_arrays (never passed).

ZERO simulation; single process, below-normal priority, torch 2
threads; rollouts stop at the fitted horizon (run(n_ticks=K), metrics
at H <= K*DT identical to the full roll).
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
import p3m_lib as M  # noqa: E402
import p3lib as P3  # noqa: E402

K = D.K_STEPS                     # H = 60 s (K=120 steps of 0.5 s)
EVAL_T0S = (120.0, 300.0)
BOOT_T0S = (120.0,)
PROF_BINS = tuple(range(-50, 350, 50))
PROF_MIN_N = 60
LC_REF = 0.002                    # p3k fitted hazard (stage-1 reference)
TAU_GRID = (1.9, 2.4, 3.0, 3.6)
B_GRID_P3M = (2.0, 3.0, 4.0, 5.0)         # extended UP (brief); 1.5 dropped
FORCE_GRID = (0.0, 150.0)
BLEND_GRID = (0.0, 0.5, 1.0)
LC_GRID = M.L3.LC_GRID
P_COOP_GRID = M.L3.P_COOP_GRID
DAWDLE_GRID = M.L3.DAWDLE_GRID
N_MEMBERS = 5
SOFTMAX_T = 0.15
T3_NORM_FLOOR = 5.0               # veh s floor in the ttc3 loss term

T_START = time.time()
N_TRUTHS = {}


def log(m):
    print(f"[{time.time()-T_START:6.1f}s] {m}", flush=True)


# ---------------------------------------------------------------- truth --

def truth_stats(root, t0, ra):
    k0 = int(round((t0 - 60.0) / 0.5))
    st = D.window_truth_stats(ra, k0, K)
    st["lc60"] = M.L3.lc_truth_between(ra, k0, k0 + K)
    st["lc30"] = M.L3.lc_truth_between(ra, k0, k0 + K // 2)
    rows = D.truth_rows(ra, k0, K)
    recs_t, _p, _f = D.truth_records(
        rows, D.lc_ticks_from_ra(ra), D.birth_states_from_ra(ra, k0, K),
        k0, K)
    st["delay"] = sum(r_["delay"] for r_ in recs_t)
    # part 3m added target: TTC<3 exposure over ALL truth rows (fixed
    # per window; the gate metric's matched-set variant is only used at
    # gate time)
    st["ttc3"] = M.truth_ttc3_exposure(ra, k0, K)
    return st


def pred_stats(root, t0, mp, lc, dev, dw=1.0):
    r = E.Rollout(root, float(t0), kind="idm_merge", member_seed=1,
                  lc_scale=lc, merge_params=dict(mp), dawdle_scale=dw,
                  dev=dev).run(n_ticks=K)
    st = D.window_pred_stats(r, K)
    snaps = D.snaps_to_dicts(r, K)
    births = D.births_from_engine(r, K)
    ra = r.ra
    dp, _dt, _np_, _nt = D.phys_delay_sums(
        snaps, births, D.truth_rows(ra, r.k0, K), D.lc_ticks_from_ra(ra),
        D.birth_states_from_ra(ra, r.k0, K), r.k0, K)
    st["delay"] = dp
    st["lc60"] = int(r.lc_count)
    st["lc30"] = M.L3.lc_count_between(r.snapshots, 0, K // 2)
    st["ttc3"] = sum(len(M.ttc_pairs(r.snapshots[j]["lane"],
                                     r.snapshots[j]["s"],
                                     r.snapshots[j]["v"]))
                     for j in range(1, K + 1)) * M.DT
    return st


# ---------------------------------------------------------------- loss --

def loss_stage1(preds, truths):
    """Acceptance loss (p3l form): zone occupancy + pooled accel speed
    profile + merge-point d."""
    occ_l = mg_l = 0.0
    pool = {}
    n_w = 0
    for key, p in preds.items():
        t = truths[key]
        n_w += 1
        occ_l += ((p["occ"] - t["occ"]) / max(t["occ"], 50.0)) ** 2
        if p["merges"] and t["merges"]:
            mg_l += ((float(np.mean(p["merges"])) -
                      float(np.mean(t["merges"]))) / 75.0) ** 2
        elif bool(p["merges"]) != bool(t["merges"]):
            mg_l += 4.0
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
    """Hazard/delay loss with the ADDED TTC<3 target: main_right
    merge-window speed + physics delay (H=60) + LC count at H=30
    (primary) and H=60 (half weight) + TTC<3 exposure per window."""
    mr_l = d_l = lc30_l = lc60_l = t3_l = 0.0
    n_w = 0
    for key, p in preds.items():
        t = truths[key]
        n_w += 1
        if p["mr"][1] and t["mr"][1]:
            mr_l += ((p["mr"][0] / p["mr"][1] -
                      t["mr"][0] / t["mr"][1]) / 5.0) ** 2
        d_l += ((p["delay"] - t["delay"]) / 200.0) ** 2
        lc30_l += ((p["lc30"] - t["lc30"]) / max(t["lc30"], 10)) ** 2
        lc60_l += ((p["lc60"] - t["lc60"]) / max(t["lc60"], 10)) ** 2
        t3_l += ((p["ttc3"] - t["ttc3"]) /
                 max(t["ttc3"], T3_NORM_FLOOR)) ** 2
    return (mr_l + d_l + lc30_l + 0.5 * lc60_l + t3_l) / max(n_w, 1), \
        dict(mr=float(mr_l / max(n_w, 1)),
             delay=float(d_l / max(n_w, 1)),
             lc30=float(lc30_l / max(n_w, 1)),
             lc60=float(lc60_l / max(n_w, 1)),
             ttc3=float(t3_l / max(n_w, 1)))


def stats_table(preds, truths, title):
    print(f"\n  -- {title} (per eval window) --")
    print("  root             t0   occ_p occ_t | accel v by d-bin -50..300"
          " pred | truth | merge_d p/t   mr_spd p/t    delay_p  delay_t"
          "  lc30 p/t lc60 p/t ttc3_p/t")
    for key in sorted(preds, key=lambda k: (k[0], k[1])):
        p, t = preds[key], truths[key]
        mg_p = f"{np.mean(p['merges']):6.1f}" if p["merges"] else "   n/a"
        mg_t = f"{np.mean(t['merges']):6.1f}" if t["merges"] else "   n/a"
        mr_p = f"{p['mr'][0]/p['mr'][1]:5.1f}" if p["mr"][1] else "  n/a"
        mr_t = f"{t['mr'][0]/t['mr'][1]:5.1f}" if t["mr"][1] else "  n/a"
        print(f"  {key[0]:<16} {key[1]:3.0f}  {p['occ']:5d} {t['occ']:5d} "
              f"| {prof_str(p)} | {prof_str(t)} | {mg_p}/{mg_t}  "
              f"{mr_p}/{mr_t} {p['delay']:9.1f} {t['delay']:8.1f}  "
              f"{p['lc30']:3d}/{t['lc30']:3d} {p['lc60']:3d}/{t['lc60']:3d} "
              f"{p['ttc3']:5.1f}/{t['ttc3']:4.1f}")


def prof_str(d):
    out = []
    for db in PROF_BINS:
        pp = d["prof"].get(db)
        out.append(f"{pp[0]/pp[1]:4.1f}" if pp and pp[1] else "  . ")
    return " ".join(out)


def roll_set(keys, mp, lc, dev, dw=1.0):
    return {k: pred_stats(k[0], k[1], mp, lc, dev, dw) for k in keys}


def softmax_draw(losses, rng, t=SOFTMAX_T):
    """Index draw with p_i proportional to exp(-(l_i - min)/t)."""
    l = np.asarray(losses, dtype=np.float64)
    w = np.exp(-(l - l.min()) / t)
    w = w / w.sum()
    return int(rng.choice(len(l), p=w))


# ------------------------------------------------------------- members --

def bootstrap_member(m, dev, winner):
    """Wider-draw member (p3l mechanism, extended coordinate grids):
    bootstrap resample of the fit roots, coordinate loss lines on the
    resample, softmax draws per coordinate (NOT argmin), independent
    dawdle-width draw."""
    rng = np.random.default_rng(400 + m)
    roots4 = sorted(N_TRUTHS)
    resample = [roots4[int(i)] for i in
                rng.integers(0, len(roots4), len(roots4))]
    keys = [(r_, BOOT_T0S[0]) for r_ in resample]
    tm = {k: N_TRUTHS[k[0]][k[1]] for k in keys}
    wmp = dict(winner["merge_params"])
    lines = {}
    simple = ((TAU_GRID, "tau_accept"), (B_GRID_P3M, "b_accept"),
              (P_COOP_GRID, "p_coop_scale"), (FORCE_GRID, "end_force_s"),
              (BLEND_GRID, "v_merge_blend"))
    for grid, field in simple:
        losses = []
        for g in grid:
            mp = dict(wmp, **{field: g})
            pr = roll_set(keys, mp, winner["lc_scale"], dev)
            l1, _ = loss_stage1(pr, tm)
            l2, _ = loss_stage2(pr, tm)
            losses.append(l1 + l2)
        lines[field] = (grid, losses)
    lc_losses = []
    for lc in LC_GRID:
        pr = roll_set(keys, wmp, lc, dev)
        l1, _ = loss_stage1(pr, tm)
        l2, _ = loss_stage2(pr, tm)
        lc_losses.append(l1 + l2)
    lines["lc_scale"] = (LC_GRID, lc_losses)
    params = {}
    for field, (grid, losses) in lines.items():
        pick = softmax_draw(losses, rng)
        params[field] = grid[pick]
        log(f"  member {m} {field} line losses "
            f"{['%.3f' % x for x in losses]} -> draw {grid[pick]}")
    params["dawdle_scale"] = float(
        DAWDLE_GRID[int(rng.integers(0, len(DAWDLE_GRID)))])
    return dict(seed=m * 17 + 1,
                lc_scale=float(params["lc_scale"]),
                merge_params=dict(tau_accept=float(params["tau_accept"]),
                                  b_accept=float(params["b_accept"]),
                                  p_coop_scale=float(
                                      params["p_coop_scale"]),
                                  end_force_s=float(params["end_force_s"]),
                                  v_merge_blend=float(
                                      params["v_merge_blend"])),
                dawdle_scale=params["dawdle_scale"],
                resample=sorted(set(resample)))


def main():
    import torch
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    (HERE / "results").mkdir(parents=True, exist_ok=True)
    sys.stdout = D.Tee(HERE / "results" / "stdout_fit.txt")

    print("priority below-normal:", D.below_normal_priority())
    dev = P3.device()
    sp = M.L3.splits_3l()
    train4 = sp["fit"]
    print("=== WD-01 part 3m TASK B: idm_merge refit, TTC<3 target + "
          "fix knobs (TRAIN only) ===")
    print("ZERO simulation; holdout blocked in root_arrays (never "
          "passed).")
    print(f"device={dev} torch_threads={torch.get_num_threads()}")
    print(f"train fit roots: {train4}; eval windows t0 {EVAL_T0S}, H=60 s "
          f"(K={K} steps of 0.5 s); calibration roots {sp['calibration']} "
          f"NOT used here; validation roots {sp['val']} NOT used here")
    print(f"grids: tau_accept {TAU_GRID} x b_accept {B_GRID_P3M} "
          f"(extended up per brief; 1.5 dropped, p3l-measured cliff); "
          f"end_force_s {FORCE_GRID}; v_merge_blend {BLEND_GRID}; "
          f"lc_scale {LC_GRID}; p_coop {P_COOP_GRID}; dawdle "
          f"{DAWDLE_GRID}; member softmax T={SOFTMAX_T}")
    print(f"stage-2 loss adds the TTC<3 exposure target "
          f"((p-t)/max(t,{T3_NORM_FLOOR:.0f}))^2 per window")
    print("M2 follower slot: FIXED nearest-behind (part 3l TASK A)")

    for root in train4:
        ra = P3.root_arrays(root)
        N_TRUTHS[root] = {t0: truth_stats(root, t0, ra) for t0 in EVAL_T0S}
    truths = {(r_, t0): N_TRUTHS[r_][t0] for r_ in train4
              for t0 in EVAL_T0S}
    keys = sorted(truths)
    print("\nTRAIN truth TTC<3 exposure per eval window (the added "
          "target):")
    for k in keys:
        print(f"  {k[0]:<16} t0={k[1]:3.0f}: {truths[k]['ttc3']:.1f} veh s")

    # ---- BEFORE: the p3l reference point (member 0 config)
    log("BEFORE: rolling the p3l L5 member 0 config ...")
    p3l_fit = M.L3.load_fit(HERE.parent / "part3l" / "results" /
                            "fit_p3l.json")
    m0 = p3l_fit["members"][0]
    before_mp = dict(m0["merge_params"], end_force_s=0.0, v_merge_blend=0.0)
    before = roll_set(keys, before_mp, m0["lc_scale"], dev)
    l1b, d1b = loss_stage1(before, truths)
    l2b, d2b = loss_stage2(before, truths)
    print(f"\nBEFORE (p3l L5 member 0): stage1 loss={l1b:.4f} {d1b} | "
          f"stage2 loss={l2b:.4f} {d2b}")
    stats_table(before, truths, "BEFORE p3l L5 member 0")

    # ---- stage 1: acceptance grid (knobs at defaults)
    log(f"STAGE 1: {len(TAU_GRID) * len(B_GRID_P3M)} acceptance combos "
        f"(force=0 blend=0, lc at reference {LC_REF}) ...")
    s1_table = []
    for tv in TAU_GRID:
        for bv in B_GRID_P3M:
            mp = dict(tau_accept=tv, b_accept=bv, p_coop_scale=0.0,
                      end_force_s=0.0, v_merge_blend=0.0)
            pr = roll_set(keys, mp, LC_REF, dev)
            l1, d1 = loss_stage1(pr, truths)
            l2, d2 = loss_stage2(pr, truths)
            s1_table.append((l1 + l2, l1, l2, dict(mp)))
            log(f"stage1 {mp} -> s1={l1:.4f} s2={l2:.4f} "
                f"(ttc3 term {d2['ttc3']:.4f})")
    s1_table.sort(key=lambda x: x[0])
    win1 = s1_table[0][3]
    print(f"\nSTAGE 1 winner: {win1} joint={s1_table[0][0]:.4f} "
          f"(s1 {s1_table[0][1]:.4f} s2 {s1_table[0][2]:.4f})")

    # ---- stage 2: fix knobs (force x blend) at the stage-1 winner
    combos2 = [(fs, bl) for fs in FORCE_GRID for bl in BLEND_GRID]
    log(f"STAGE 2: {len(combos2)} force x blend combos at {win1} ...")
    s2_table = []
    for fs, bl in combos2:
        mp2 = dict(win1, end_force_s=fs, v_merge_blend=bl)
        pr = roll_set(keys, mp2, LC_REF, dev)
        l1, d1 = loss_stage1(pr, truths)
        l2, d2 = loss_stage2(pr, truths)
        s2_table.append((l1 + l2, l1, l2, dict(mp2)))
        log(f"stage2 force={fs} blend={bl} -> s1={l1:.4f} s2={l2:.4f} "
            f"(ttc3 term {d2['ttc3']:.4f})")
    s2_table.sort(key=lambda x: x[0])
    win2 = s2_table[0][3]
    print(f"\nSTAGE 2 winner: force={win2['end_force_s']} "
          f"blend={win2['v_merge_blend']} joint={s2_table[0][0]:.4f}")

    # ---- stage 3: lc x p_coop at the stage-2 winner
    log(f"STAGE 3: {len(LC_GRID) * len(P_COOP_GRID)} lc x p_coop combos "
        f"at force={win2['end_force_s']} blend={win2['v_merge_blend']} "
        f"...")
    s3_table = []
    for lc in LC_GRID:
        for pv in P_COOP_GRID:
            mp3 = dict(win2, p_coop_scale=pv)
            pr = roll_set(keys, mp3, lc, dev)
            l1, d1 = loss_stage1(pr, truths)
            l2, d2 = loss_stage2(pr, truths)
            s3_table.append((l1 + l2, l1, l2, dict(mp3, lc_scale=lc)))
            log(f"stage3 lc={lc} p_coop={pv} -> s1={l1:.4f} s2={l2:.4f} "
                f"(ttc3 term {d2['ttc3']:.4f})")
    s3_table.sort(key=lambda x: x[0])
    win3 = s3_table[0][3]
    lc_win = win3["lc_scale"]
    print(f"\nSTAGE 3 winner: lc_scale={lc_win} "
          f"p_coop={win3['p_coop_scale']} joint={s3_table[0][0]:.4f}")

    # ---- stage 4: joint acceptance re-sweep at the lc/p_coop winner
    log(f"STAGE 4: {len(TAU_GRID) * len(B_GRID_P3M)} acceptance combos at "
        f"lc={lc_win} p_coop={win3['p_coop_scale']} ...")
    s4_table = []
    for tv in TAU_GRID:
        for bv in B_GRID_P3M:
            mp4 = dict(win2, tau_accept=tv, b_accept=bv,
                       p_coop_scale=win3["p_coop_scale"])
            pr = roll_set(keys, mp4, lc_win, dev)
            l1, d1 = loss_stage1(pr, truths)
            l2, d2 = loss_stage2(pr, truths)
            s4_table.append((l1 + l2, l1, l2, dict(mp4)))
            log(f"stage4 {mp4} -> s1={l1:.4f} s2={l2:.4f} "
                f"(ttc3 term {d2['ttc3']:.4f})")
    s4_table.sort(key=lambda x: x[0])
    best = s4_table[0]
    sel_mp = best[3]
    print(f"\nSTAGE 4 winner: {sel_mp} lc_scale={lc_win} "
          f"joint={best[0]:.4f} (s1 {best[1]:.4f} s2 {best[2]:.4f})")

    # ---- AFTER table at the selected point
    after = roll_set(keys, sel_mp, lc_win, dev)
    l1a, d1a = loss_stage1(after, truths)
    l2a, d2a = loss_stage2(after, truths)
    print(f"\nAFTER (L6 = idm_merge refit, part 3m knobs): stage1 "
          f"loss={l1a:.4f} {d1a} | stage2 loss={l2a:.4f} {d2a}")
    stats_table(after, truths, "AFTER L6 (idm_merge refit)")

    # ---- wider-draw members 1..4
    log("MEMBERS 1..4: wider draws (bootstrap resample x softmax lines "
        "x dawdle width) ...")
    winner = dict(merge_params=dict(sel_mp), lc_scale=float(lc_win))
    members = [dict(seed=1, lc_scale=float(lc_win),
                    merge_params=dict(sel_mp), dawdle_scale=1.0)]
    for m in range(1, N_MEMBERS):
        mem = bootstrap_member(m, dev, winner)
        members.append(mem)
        log(f"member {m}: {mem}")
    print("\nselected member parameters (member 0 = main fit):")
    for mi, mem in enumerate(members):
        print(f"  member {mi}: seed={mem['seed']} "
              f"lc_scale={mem['lc_scale']} "
              f"merge_params={mem['merge_params']} "
              f"dawdle_scale={mem['dawdle_scale']}")

    result = dict(
        variant="L6 kind='idm_merge' (p3l fixed slot + part 3m knobs)",
        fitted_on=dict(roots=train4, t0s=list(EVAL_T0S), H=60.0,
                       note="TRAIN only; holdout never accessed; "
                            "calibration split never rolled here"),
        added_target=dict(name="ttc3_exposure_H60",
                          form="((pred-truth)/max(truth,5))^2 per window",
                          truth="ALL truth rows (p3m_lib."
                                "truth_ttc3_exposure)"),
        selected=dict(merge_params=sel_mp, lc_scale=float(lc_win),
                      joint_loss=best[0], stage1_loss=best[1],
                      stage2_loss=best[2],
                      before=dict(p3l_stage1=l1b, p3l_stage2=l2b,
                                  detail1=d1b, detail2=d2b),
                      after=dict(stage1=l1a, stage2=l2a,
                                 detail1=d1a, detail2=d2a)),
        members=members,
        grids=dict(tau_accept=list(TAU_GRID),
                   b_accept=list(B_GRID_P3M),
                   end_force_s=list(FORCE_GRID),
                   v_merge_blend=list(BLEND_GRID),
                   lc_scale=list(LC_GRID),
                   p_coop_scale=list(P_COOP_GRID),
                   dawdle_scale=list(DAWDLE_GRID),
                   softmax_T=SOFTMAX_T),
        stage_tables=dict(
            stage1=[dict(joint=l, s1=a, s2=b, params=p)
                    for l, a, b, p in s1_table],
            stage2=[dict(joint=l, s1=a, s2=b, params=p)
                    for l, a, b, p in s2_table],
            stage3=[dict(joint=l, s1=a, s2=b, params=p)
                    for l, a, b, p in s3_table],
            stage4=[dict(joint=l, s1=a, s2=b, params=p)
                    for l, a, b, p in s4_table]),
    )
    (HERE / "results" / "fit_p3m.json").write_text(
        json.dumps(result, indent=1, default=str), encoding="utf-8")
    log(f"wrote {HERE / 'results' / 'fit_p3m.json'} and "
        f"{HERE / 'results' / 'stdout_fit.txt'}")
    print("\nDONE part 3m TASK B fit.")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
