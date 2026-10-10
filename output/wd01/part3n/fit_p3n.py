"""WD-01 part 3n TASK B: joint TRAIN refit of lc_scale x end_force_s
(brief grids), with the stage-2 LC term split BY TYPE.

Fix target: the part 3m refit's gate-v2 (b) residual -- the TRAIN-only
fit selected lc_scale=0.002 + end_force_s=150 jointly, and the rung
overshoots merge/LC counts out of sample (validation +39.6 %; TASK A
locates the by-type split ON TRAIN).  Per the brief:

  - lc_scale grid (0.0005, 0.00075, 0.001, 0.0015, 0.002)  [..0.002]
  - end_force_s grid (60, 100, 150, 200)                    [no 0:
    p3m measured force-off ~20x worse; the brief grid starts at 60]
  - refit JOINTLY on the 8 TRAIN fit windows (4 fit roots x t0
    {120, 300}, K=120) with the loss terms: LC count per window BY
    TYPE (mandatory accel->main_right merge vs discretionary 0<->1,
    H=30 primary + H=60 half weight), TTC<3 exposure, physics delay,
    merge-zone occupancy (the p3m stage-1 occupancy/profile/merge-d
    guard is kept and the mr guard kept -- ONLY the LC terms changed
    vs part 3m);
  - tau_accept / b_accept / p_coop_scale / v_merge_blend stay at the
    part 3m member-0 values (tau 1.9, b 2.0, p_coop 0.0, blend 0.0);
  - members 1-4: the p3l/p3m wider-draw mechanism (bootstrap resample
    of the fit roots, per-coordinate softmax draws T=0.15, independent
    dawdle draw) over the part 3n coordinate grids (lc 5 points capped
    at 0.002, force 4 points from 60);

Protocol is TRAIN-only; the calibration roots are the TRAIN-internal
gate split (never rolled here); validation is never touched; holdout
blocked in root_arrays.

ZERO simulation; single process, below-normal priority, torch 2
threads.
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
import decompose_p3k as D  # noqa: E402
import p3lib as P3  # noqa: E402

K = N.K_TICKS
# iteration-2 protocol (TRAIN-only, declared): the fit windows EXPAND to
# 4 fit roots x 7 t0 = 28 windows.  Iteration 1 fit on t0 {120, 300}
# (the p3l/p3m convention) and its member coordinate lines for
# tau_accept/b_accept came out FLAT (the 2-fold TRAIN-internal gate then
# failed (b) by 1.9 points of lc mean); the iteration-2 probe (results
# in the report) showed the joint loss SEPARATES the acceptance
# coordinates strongly once the fit sees the richer t0 grid (winner
# 2.774 vs a b=5.0/tau=2.4 config 5.119 on the same 28 windows), so the
# expanded set is what makes the member draws identifiable.  The fit
# ROOTS are unchanged (TRAIN fit roots only; the calibration roots stay
# the held-out gate split).
EVAL_T0S = tuple(range(120, 481, 60))
BOOT_T0S = tuple(range(120, 481, 60))
TAU_GRID = (1.9, 2.4, 3.0, 3.6)       # part 3m grid (lines only, frozen
# at the p3m member-0 value for the main fit)
B_GRID = (2.0, 3.0, 4.0, 5.0)
BLEND_GRID = (0.0, 0.5, 1.0)
P_COOP_GRID = (0.0, 0.25)
DAWDLE_GRID = N.L3.DAWDLE_GRID
LC_GRID = N.LC_GRID_P3N
FORCE_GRID = N.FORCE_GRID_P3N
N_MEMBERS = 5
SOFTMAX_T = 0.15
T_START = time.time()
N_TRUTHS = {}


def log(m):
    print(f"[{time.time()-T_START:6.1f}s] {m}", flush=True)


def roll_stats(root, t0, mp, lc, dev, dw=1.0):
    return N.pred_fit_stats(root, t0, mp, lc, dev, dw, K)


def stats_table(preds, truths, title):
    print(f"\n  -- {title} (per eval window) --")
    print("  root             t0   occ_p occ_t | merge_d p/t   mr_spd "
          "p/t    delay_p  delay_t  lcM30 p/t lcD30 p/t lcM60 p/t "
          "lcD60 p/t ttc3_p/t")
    for key in sorted(preds, key=lambda k: (k[0], k[1])):
        p, t = preds[key], truths[key]
        mg_p = f"{np.mean(p['merges']):6.1f}" if p["merges"] else "   n/a"
        mg_t = f"{np.mean(t['merges']):6.1f}" if t["merges"] else "   n/a"
        mr_p = f"{p['mr'][0]/p['mr'][1]:5.1f}" if p["mr"][1] else "  n/a"
        mr_t = f"{t['mr'][0]/t['mr'][1]:5.1f}" if t["mr"][1] else "  n/a"
        lc = " ".join(
            f"{p[k]:3d}/{t[k]:<3d}" for k in
            ("lcM30", "lcD30", "lcM60", "lcD60"))
        print(f"  {key[0]:<16} {key[1]:3.0f}  {p['occ']:5d} {t['occ']:5d}"
              f" | {mg_p}/{mg_t}  {mr_p}/{mr_t} {p['delay']:9.1f} "
              f"{t['delay']:8.1f}  {lc} {p['ttc3']:5.1f}/{t['ttc3']:4.1f}")


def roll_set(keys, mp, lc, dev, dw=1.0):
    return {k: roll_stats(k[0], k[1], mp, lc, dev, dw) for k in keys}


def joint_loss(preds, truths):
    l1, d1 = N.loss_stage1(preds, truths)
    l2, d2 = N.loss_p3n(preds, truths)
    return l1 + l2, l1, l2, d1, d2


def bootstrap_member(m, dev, winner):
    """Wider-draw member (p3l mechanism, expanded line windows):
    bootstrap resample of the fit roots, per-coordinate loss lines on
    the resample x the 7-t0 grid (the expanded, identifiable set),
    softmax draws T=0.15 (NOT argmin), independent dawdle-width
    draw."""
    rng = np.random.default_rng(400 + m)
    roots4 = sorted(N_TRUTHS)
    resample = [roots4[int(i)] for i in
                rng.integers(0, len(roots4), len(roots4))]
    keys = [(r_, float(t0)) for r_ in sorted(set(resample))
            for t0 in BOOT_T0S]
    tm = {k: N_TRUTHS[k[0]][k[1]] for k in keys}
    wmp = dict(winner["merge_params"])
    lines = {}
    simple = ((TAU_GRID, "tau_accept"), (B_GRID, "b_accept"),
              (P_COOP_GRID, "p_coop_scale"), (BLEND_GRID, "v_merge_blend"),
              (FORCE_GRID, "end_force_s"))
    for grid, field in simple:
        losses = []
        for g in grid:
            mp = dict(wmp, **{field: g})
            pr = roll_set(keys, mp, winner["lc_scale"], dev)
            losses.append(joint_loss(pr, tm)[0])
        lines[field] = (grid, losses)
    lc_losses = []
    for lc in LC_GRID:
        pr = roll_set(keys, wmp, lc, dev)
        lc_losses.append(joint_loss(pr, tm)[0])
    lines["lc_scale"] = (LC_GRID, lc_losses)
    params = {}
    for field, (grid, losses) in lines.items():
        pick = N.softmax_draw(losses, rng, SOFTMAX_T)
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
                                  end_force_s=float(
                                      params["end_force_s"]),
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
    sp = N.L3.splits_3l()
    train4 = sp["fit"]
    cal4 = sp["calibration"]
    assert not (set(train4) & set(cal4))
    print("=== WD-01 part 3n TASK B: joint lc_scale x end_force_s refit "
          "(LC loss BY TYPE; iteration 2 = EXPANDED fit windows "
          "4 fit roots x 7 t0, still fit roots ONLY) ===")
    print("ZERO simulation; holdout blocked in root_arrays (never "
          "passed).")
    print(f"device={dev} torch_threads={torch.get_num_threads()}")
    print(f"train fit roots: {train4}; eval windows t0 {EVAL_T0S}, "
          f"H=60 s (K={K} steps of 0.5 s); calibration roots {cal4} = "
          f"the TRAIN-internal GATE split (2-fold), NOT used here; "
          f"validation {sp['val']} NOT used here")
    print(f"grids: lc_scale {LC_GRID} (..0.002 per brief) x end_force_s "
          f"{FORCE_GRID} (brief; 0.0 dropped -- p3m measured force-off "
          f"~20x worse); tau/b/p_coop/blend FROZEN at the p3m member-0 "
          f"values tau 1.9 b 2.0 p_coop 0.0 blend 0.0; dawdle "
          f"{DAWDLE_GRID}; member softmax T={SOFTMAX_T}")
    print("loss: p3m stage-1 (occ+prof+merge-d) + mr/delay/ttc3 terms "
          "+ LC BY TYPE (merge, discretionary) H30 primary + H60 half "
          f"weight, norm floor {N.LC_NORM_FLOOR}")

    for root in train4:
        ra = P3.root_arrays(root)
        N_TRUTHS[root] = {t0: N.truth_fit_stats(root, t0, ra, K)
                          for t0 in EVAL_T0S}
    truths = {(r_, t0): N_TRUTHS[r_][t0] for r_ in train4
              for t0 in EVAL_T0S}
    keys = sorted(truths)
    print("\nTRAIN truth LC BY TYPE per eval window (the refit target; "
          "t0=120: lcM/lcD@H30, t0=300 likewise):")
    for k in keys:
        t = truths[k]
        print(f"  {k[0]:<16} t0={k[1]:3.0f}: lcM30 {t['lcM30']:2d} "
              f"lcD30 {t['lcD30']:2d} | lcM60 {t['lcM60']:2d} lcD60 "
              f"{t['lcD60']:2d} | ttc3 {t['ttc3']:.1f} veh s")

    # ---- BEFORE: the p3m member-0 point
    log("BEFORE: rolling the p3m L6 member-0 config ...")
    p3m_fit = N.load_json(HERE.parent / "part3m" / "results" /
                          "fit_p3m.json")
    m0 = p3m_fit["members"][0]
    before = roll_set(keys, dict(m0["merge_params"]),
                      m0["lc_scale"], dev)
    lb, l1b, l2b, d1b, d2b = joint_loss(before, truths)
    print(f"\nBEFORE (p3m L6 member 0): joint={lb:.4f} stage1={l1b:.4f} "
          f"{d1b} | stage2={l2b:.4f} {d2b}")
    stats_table(before, truths, "BEFORE p3m L6 member 0")

    # ---- the joint lc x force grid (THE refit; other coords frozen)
    log(f"JOINT GRID: {len(LC_GRID)} lc x {len(FORCE_GRID)} force = "
        f"{len(LC_GRID) * len(FORCE_GRID)} combos ...")
    grid_table = []
    for lc in LC_GRID:
        for fs in FORCE_GRID:
            pr = roll_set(keys, dict(m0["merge_params"]), lc, dev)
            lj, l1, l2, d1, d2 = joint_loss(pr, truths)
            grid_table.append((lj, l1, l2, lc, fs, d2))
            log(f"  lc={lc:.5f} force={fs:5.1f} -> joint={lj:.4f} "
                f"(s1 {l1:.4f} s2 {l2:.4f}; lcM30 {d2['lcM30']:.3f} "
                f"lcD30 {d2['lcD30']:.3f} ttc3 {d2['ttc3']:.3f} "
                f"delay {d2['delay']:.3f})")
    grid_table.sort(key=lambda x: x[0])
    bj, b1, b2, lc_win, fs_win, d2w = grid_table[0]
    # the joint winner: the p3m member-0 coordinates with BOTH refit
    # coordinates adopted (stable sort -> first combo at the min loss)
    sel_mp = dict(m0["merge_params"], end_force_s=fs_win)
    print(f"\nGRID winner: lc_scale={lc_win} end_force_s={fs_win} "
          f"joint={bj:.4f} (s1 {b1:.4f} s2 {b2:.4f})")

    after = roll_set(keys, sel_mp, lc_win, dev)
    la, l1a, l2a, d1a, d2a = joint_loss(after, truths)
    print(f"\nAFTER (L6 p3n = p3m member-0 coords + lc {lc_win} + force "
          f"{fs_win}): joint={la:.4f} stage1={l1a:.4f} {d1a} | "
          f"stage2={l2a:.4f} {d2a}")
    stats_table(after, truths, "AFTER L6 (part 3n refit)")

    # ---- wider-draw members 1..4
    log("MEMBERS 1..4: wider draws (bootstrap resample x softmax "
        "coordinate lines x dawdle width) ...")
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
        variant="L6 kind='idm_merge' (p3n joint lc x force refit; "
                "engine UNCHANGED)",
        fitted_on=dict(roots=train4, t0s=list(EVAL_T0S), H=60.0,
                       note="TRAIN only; holdout never accessed; "
                            "calibration split (the TRAIN-internal "
                            "gate) never rolled here"),
        loss=dict(form="p3m stage1 + p3m stage2 with lc30/lc60 SPLIT "
                       "BY TYPE (merge, discretionary); H30 primary + "
                       "H60 half weight",
                  changed_vs_p3m="lc terms only; norm floor 5 (p3m: "
                                 "10, total lc)",
                  grids_fixed_at_p3m_m0=dict(tau_accept=1.9,
                                             b_accept=2.0,
                                             p_coop_scale=0.0,
                                             v_merge_blend=0.0)),
        selected=dict(merge_params=sel_mp, lc_scale=float(lc_win),
                      joint_loss=bj, stage1_loss=b1, stage2_loss=b2,
                      stage2_detail=d2w,
                      before=dict(p3m_joint=lb, p3m_stage1=l1b,
                                  p3m_stage2=l2b, detail1=d1b,
                                  detail2=d2b),
                      after=dict(joint=la, stage1=l1a, stage2=l2a,
                                 detail1=d1a, detail2=d2a)),
        members=members,
        grids=dict(lc_scale=list(LC_GRID), end_force_s=list(FORCE_GRID),
                   tau_accept_lines=list(TAU_GRID),
                   b_accept_lines=list(B_GRID),
                   p_coop_scale_lines=list(P_COOP_GRID),
                   v_merge_blend_lines=list(BLEND_GRID),
                   dawdle_scale=list(DAWDLE_GRID),
                   softmax_T=SOFTMAX_T),
        stage_tables=dict(
            joint_grid=[dict(joint=l, s1=a, s2=b, lc_scale=lc,
                             end_force_s=fs, stage2_detail=d)
                        for l, a, b, lc, fs, d in grid_table]),
    )
    (HERE / "results" / "fit_p3n.json").write_text(
        json.dumps(result, indent=1, default=str), encoding="utf-8")
    log(f"wrote {HERE / 'results' / 'fit_p3n.json'} and "
        f"{HERE / 'results' / 'stdout_fit.txt'}")
    print("\nDONE part 3n TASK B fit.")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
