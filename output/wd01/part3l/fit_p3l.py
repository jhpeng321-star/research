"""WD-01 part 3l TASK B: TRAIN fit of the idm_merge parameters with the
extended acceptance grids and the LC hazard refitted at H=30.

Differences vs the p3k fit (fit_p3k.py, same 8 TRAIN fit windows):
  - tau_accept grid extended to {1.9, 2.4, 3.0, 3.6} and b_accept to
    {1.5, 2.0, 3.0} (p3k: {0.7..1.9} x {2.0..6.5});
  - lc_scale is refitted JOINTLY on a finer grid by matching the LC
    count per window at H=30 (primary) and H=60 (checked), replacing
    p3k's coarse {0.002, 0.003, 0.005} at H=60 only;
  - protocol: stage 1 acceptance grid (lc at the p3k reference 0.002)
    -> stage 2 lc grid at the stage-1 winner -> stage 3 acceptance grid
    re-swept at the stage-2 lc winner, joint best by s1+s2;
  - members 1..4 (TASK C mechanism B, wider parameter uncertainty):
    bootstrap resamples of the 4 fit roots (rng 400+m), coordinate loss
    lines on the resample, then SOFTMAX DRAWS (T=0.15) instead of
    argmin, plus an independent dawdle-width draw in {0.8, 1.0, 1.25,
    1.5} (engine dawdle_scale; default 1.0 is draw-identical);
  - the M2 follower slot is the FIXED one (nearest main_right vehicle
    behind in d; part 3l TASK A found the p3k argmax bug).

Fit targets (p3k convention): T1 accel merge-zone occupancy, T2 accel
speed profile (50 m d-bins), T3 merge-point d; T4 main_right speed in
[-100, 100] m, T5 contract physics delay (H=60), T6a LC count at H=30,
T6b LC count at H=60.

ZERO simulation; holdout blocked in root_arrays (never passed); TRAIN
fit roots only.  Single process, below-normal priority, torch 2 threads.
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

K = D.K_STEPS
EVAL_T0S = (120.0, 300.0)
BOOT_T0S = (120.0,)
PROF_BINS = tuple(range(-50, 350, 50))
PROF_MIN_N = 60
LC_REF = 0.002                       # p3k fitted hazard (stage-1 reference)
LC_SCALE_L3 = D.LC_SCALE_L3
TAU_GRID = L.TAU_GRID
B_GRID = L.B_GRID
LC_GRID = L.LC_GRID
P_COOP_GRID = L.P_COOP_GRID
DAWDLE_GRID = L.DAWDLE_GRID
N_MEMBERS = 5
SOFTMAX_T = 0.15

T_START = time.time()
N_TRUTHS = {}


def log(m):
    print(f"[{time.time()-T_START:6.1f}s] {m}", flush=True)


# ---------------------------------------------------------------- truth --

def truth_stats(root, t0, ra):
    k0 = int(round((t0 - 60.0) / 0.5))
    st = D.window_truth_stats(ra, k0, K)
    st["lc60"] = L.lc_truth_between(ra, k0, k0 + K)
    st["lc30"] = L.lc_truth_between(ra, k0, k0 + K // 2)
    rows = D.truth_rows(ra, k0, K)
    recs_t, _p, _f = D.truth_records(
        rows, D.lc_ticks_from_ra(ra), D.birth_states_from_ra(ra, k0, K),
        k0, K)
    st["delay"] = sum(r_["delay"] for r_ in recs_t)
    return st


def pred_stats(root, t0, mp, lc, dev):
    r = E.Rollout(root, float(t0), kind="idm_merge", member_seed=1,
                  lc_scale=lc, merge_params=mp, dev=dev).run()
    st = D.window_pred_stats(r, K)
    snaps = D.snaps_to_dicts(r, K)
    births = D.births_from_engine(r, K)
    ra = r.ra
    dp, _dt, _np_, _nt = D.phys_delay_sums(
        snaps, births, D.truth_rows(ra, r.k0, K), D.lc_ticks_from_ra(ra),
        D.birth_states_from_ra(ra, r.k0, K), r.k0, K)
    st["delay"] = dp
    st["lc60"] = int(r.lc_count)
    st["lc30"] = L.lc_count_between(r.snapshots, 0, K // 2)
    return st


# ---------------------------------------------------------------- loss --

def loss_stage1(preds, truths):
    """Acceptance loss (p3k form): zone occupancy + pooled accel speed
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
    """Hazard/delay loss: main_right merge-window speed + physics delay
    (H=60) + LC count matched at H=30 (primary) and H=60 (half weight)."""
    mr_l = d_l = lc30_l = lc60_l = 0.0
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
    return (mr_l + d_l + lc30_l + 0.5 * lc60_l) / max(n_w, 1), dict(
        mr=float(mr_l / max(n_w, 1)), delay=float(d_l / max(n_w, 1)),
        lc30=float(lc30_l / max(n_w, 1)),
        lc60=float(lc60_l / max(n_w, 1)))


def prof_str(d):
    out = []
    for db in PROF_BINS:
        pp = d["prof"].get(db)
        out.append(f"{pp[0]/pp[1]:4.1f}" if pp and pp[1] else "  . ")
    return " ".join(out)


def stats_table(preds, truths, title):
    print(f"\n  -- {title} (per eval window) --")
    print("  root             t0   occ_p occ_t | accel v by d-bin -50..300"
          " pred | truth | merge_d p/t   mr_spd p/t    delay_p  delay_t"
          "  lc30 p/t lc60 p/t")
    for key in sorted(preds, key=lambda k: (k[0], k[1])):
        p, t = preds[key], truths[key]
        mg_p = f"{np.mean(p['merges']):6.1f}" if p["merges"] else "   n/a"
        mg_t = f"{np.mean(t['merges']):6.1f}" if t["merges"] else "   n/a"
        mr_p = f"{p['mr'][0]/p['mr'][1]:5.1f}" if p["mr"][1] else "  n/a"
        mr_t = f"{t['mr'][0]/t['mr'][1]:5.1f}" if t["mr"][1] else "  n/a"
        print(f"  {key[0]:<16} {key[1]:3.0f}  {p['occ']:5d} {t['occ']:5d} "
              f"| {prof_str(p)} | {prof_str(t)} | {mg_p}/{mg_t}  "
              f"{mr_p}/{mr_t} {p['delay']:9.1f} {t['delay']:8.1f}  "
              f"{p['lc30']:3d}/{t['lc30']:3d} {p['lc60']:3d}/{t['lc60']:3d}")


def roll_set(keys, mp, lc, dev):
    return {k: pred_stats(k[0], k[1], mp, lc, dev) for k in keys}


def softmax_draw(losses, rng, t=SOFTMAX_T):
    """Index draw with p_i proportional to exp(-(l_i - min)/t)."""
    l = np.asarray(losses, dtype=np.float64)
    w = np.exp(-(l - l.min()) / t)
    w = w / w.sum()
    return int(rng.choice(len(l), p=w))


# ------------------------------------------------------------- members --

def bootstrap_member(m, dev, winner):
    """Wider-draw member (TASK C mechanism B): bootstrap resample of the
    fit roots, coordinate loss lines on the resample, softmax draws per
    coordinate (NOT argmin), independent dawdle-width draw."""
    rng = np.random.default_rng(400 + m)
    roots4 = sorted(N_TRUTHS)
    resample = [roots4[int(i)] for i in
                rng.integers(0, len(roots4), len(roots4))]
    keys = [(r_, BOOT_T0S[0]) for r_ in resample]
    tm = {k: N_TRUTHS[k[0]][k[1]] for k in keys}
    lines = {}
    for grid, field in ((TAU_GRID, "tau_accept"), (B_GRID, "b_accept"),
                        (P_COOP_GRID, "p_coop_scale")):
        losses = []
        for g in grid:
            mp = dict(winner["merge_params"], **{field: g})
            pr = roll_set(keys, mp, winner["lc_scale"], dev)
            l1, _ = loss_stage1(pr, tm)
            l2, _ = loss_stage2(pr, tm)
            losses.append(l1 + l2)
        lines[field] = (grid, losses)
    lc_losses = []
    for lc in LC_GRID:
        pr = roll_set(keys, winner["merge_params"], lc, dev)
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
    return dict(seed=m * 17 + 1, lc_scale=float(params["lc_scale"]),
                merge_params=dict(tau_accept=float(params["tau_accept"]),
                                  b_accept=float(params["b_accept"]),
                                  p_coop_scale=float(
                                      params["p_coop_scale"])),
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
    sp = L.splits_3l()
    train4 = sp["fit"]
    print("=== WD-01 part 3l TASK B: idm_merge fit, extended grids + "
          "LC@H30 (TRAIN only) ===")
    print("ZERO simulation; holdout blocked in root_arrays (never passed).")
    print(f"device={dev} torch_threads={torch.get_num_threads()}")
    print(f"train fit roots: {train4}; eval windows t0 {EVAL_T0S}, H=60 s "
          f"(K={K} steps of 0.5 s); validation roots {sp['val']} NOT used "
          f"here; calibration roots {sp['calibration']} NOT used here")
    print(f"grids: tau_accept {TAU_GRID} x b_accept {B_GRID}; lc_scale "
          f"{LC_GRID} (matched at H=30 primary, H=60 half weight); "
          f"p_coop {P_COOP_GRID}; dawdle draws {DAWDLE_GRID}; "
          f"member softmax T={SOFTMAX_T}")
    print(f"M2 follower slot: FIXED nearest-behind (part 3l TASK A)")

    for root in train4:
        ra = P3.root_arrays(root)
        N_TRUTHS[root] = {t0: truth_stats(root, t0, ra) for t0 in EVAL_T0S}
    truths = {(r_, t0): N_TRUTHS[r_][t0] for r_ in train4
              for t0 in EVAL_T0S}
    keys = sorted(truths)

    log("BEFORE: rolling L3 (idm_sampled, member 1, lc_scale=0.003) ...")
    before = {}
    for k in keys:
        r = E.Rollout(k[0], float(k[1]), kind="idm_sampled", member_seed=1,
                      lc_scale=LC_SCALE_L3, dev=dev).run()
        st = D.window_pred_stats(r, K)
        s_ = D.snaps_to_dicts(r, K)
        dp, _dt, _n1, _n2 = D.phys_delay_sums(
            s_, D.births_from_engine(r, K), D.truth_rows(r.ra, r.k0, K),
            D.lc_ticks_from_ra(r.ra),
            D.birth_states_from_ra(r.ra, r.k0, K), r.k0, K)
        st["delay"] = dp
        st["lc60"] = int(r.lc_count)
        st["lc30"] = L.lc_count_between(r.snapshots, 0, K // 2)
        before[k] = st
    l1b, d1b = loss_stage1(before, truths)
    l2b, d2b = loss_stage2(before, truths)
    print(f"\nBEFORE (L3 = idm_sampled, member 1, lc_scale="
          f"{LC_SCALE_L3}): stage1 loss={l1b:.4f} {d1b} | stage2 "
          f"loss={l2b:.4f} {d2b}")
    stats_table(before, truths, "BEFORE L3 (idm_sampled)")

    # ---- stage 1: extended acceptance grid (lc at p3k reference)
    log(f"STAGE 1: {len(TAU_GRID) * len(B_GRID)} acceptance combos ...")
    s1_table = []
    for tv in TAU_GRID:
        for bv in B_GRID:
            mp = dict(tau_accept=tv, b_accept=bv, p_coop_scale=0.0)
            pr = roll_set(keys, mp, LC_REF, dev)
            l1, d1 = loss_stage1(pr, truths)
            s1_table.append((l1, d1, dict(mp)))
            log(f"stage1 {mp} -> loss={l1:.4f} {d1}")
    s1_table.sort(key=lambda x: x[0])
    win1 = s1_table[0][2]
    print(f"\nSTAGE 1 winner: {win1} loss={s1_table[0][0]:.4f} "
          f"{s1_table[0][1]}")

    # ---- stage 2: lc x p_coop grid at the stage-1 winner (LC matched
    # at H=30; p_coop matters now that the follower slot is fixed --
    # cooperation is what keeps the follower's b_req small in SUMO)
    combos2 = [(lc, pv) for lc in LC_GRID for pv in P_COOP_GRID]
    log(f"STAGE 2: {len(combos2)} lc x p_coop combos at {win1} ...")
    s2_table = []
    for lc, pv in combos2:
        mp2 = dict(win1, p_coop_scale=pv)
        pr = roll_set(keys, mp2, lc, dev)
        l2, d2 = loss_stage2(pr, truths)
        s2_table.append((l2, d2, dict(mp2, lc_scale=lc)))
        log(f"stage2 lc_scale={lc} p_coop={pv} -> loss={l2:.4f} {d2}")
    s2_table.sort(key=lambda x: x[0])
    lc_win = s2_table[0][2]["lc_scale"]
    pco_win = s2_table[0][2]["p_coop_scale"]
    print(f"\nSTAGE 2 winner: lc_scale={lc_win} p_coop={pco_win} "
          f"loss={s2_table[0][0]:.4f} {s2_table[0][1]}")

    # ---- stage 3: joint refinement (acceptance re-swept at lc_win +
    # pco_win; the p_coop winner from stage 2 is carried)
    log(f"STAGE 3: {len(TAU_GRID) * len(B_GRID)} acceptance combos at "
        f"lc_scale={lc_win} p_coop={pco_win} ...")
    s3_table = []
    for tv in TAU_GRID:
        for bv in B_GRID:
            mp = dict(tau_accept=tv, b_accept=bv, p_coop_scale=pco_win)
            pr = roll_set(keys, mp, lc_win, dev)
            l1, d1 = loss_stage1(pr, truths)
            l2, d2 = loss_stage2(pr, truths)
            s3_table.append((l1 + l2, l1, l2, dict(mp)))
            log(f"stage3 {mp} -> s1+s2={l1 + l2:.4f} (s1 {l1:.4f} "
                f"s2 {l2:.4f})")
    s3_table.sort(key=lambda x: x[0])
    best = s3_table[0]
    sel_mp = best[3]
    print(f"\nSTAGE 3 winner: {sel_mp} lc_scale={lc_win} "
          f"joint={best[0]:.4f} (s1 {best[1]:.4f} s2 {best[2]:.4f})")

    # ---- AFTER table at the selected point
    after = roll_set(keys, sel_mp, lc_win, dev)
    l1a, d1a = loss_stage1(after, truths)
    l2a, d2a = loss_stage2(after, truths)
    print(f"\nAFTER (L5 = idm_merge fitted, fixed follower slot): "
          f"stage1 loss={l1a:.4f} {d1a} | stage2 loss={l2a:.4f} {d2a}")
    stats_table(after, truths, "AFTER L5 (idm_merge fitted)")

    # ---- wider-draw members 1..4
    log("MEMBERS 1..4: wider draws (bootstrap resample x softmax lines "
        "x dawdle width) ...")
    winner = dict(merge_params=sel_mp, lc_scale=lc_win)
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

    # ---- b_accept flatness re-check on the FULL fit windows (post-fix)
    flat = {}
    for l1, d1, mp in s1_table:
        flat[mp["b_accept"]] = flat.get(mp["b_accept"], []) + \
            [round(l1, 6)]
    print("\nb_accept sensitivity on the full fit windows (stage-1 loss "
          "values per tau, post-fix):")
    for bv, ls in sorted(flat.items()):
        print(f"  b_accept={bv}: {ls}")

    result = dict(
        variant="L5 kind='idm_merge' (fixed follower slot)",
        fitted_on=dict(roots=train4, t0s=list(EVAL_T0S), H=60.0,
                       note="TRAIN only; holdout never accessed"),
        selected=dict(merge_params=sel_mp, lc_scale=float(lc_win),
                      joint_loss=best[0], stage1_loss=best[1],
                      stage2_loss=best[2],
                      before=dict(l3_stage1=l1b, l3_stage2=l2b,
                                  detail1=d1b, detail2=d2b),
                      after=dict(stage1=l1a, stage2=l2a,
                                 detail1=d1a, detail2=d2a)),
        members=members,
        grids=dict(tau_accept=list(TAU_GRID), b_accept=list(B_GRID),
                   lc_scale=list(LC_GRID),
                   p_coop_scale=list(P_COOP_GRID),
                   dawdle_scale=list(DAWDLE_GRID),
                   softmax_T=SOFTMAX_T),
        stage_tables=dict(
            stage1=[dict(loss=l, detail=d, params=p)
                    for l, d, p in s1_table],
            stage2=[dict(loss=l, detail=d, params=p)
                    for l, d, p in s2_table],
            stage3=[dict(joint=l, s1=a, s2=b, params=p)
                    for l, a, b, p in s3_table]),
    )
    (HERE / "results" / "fit_p3l.json").write_text(
        json.dumps(result, indent=1, default=str), encoding="utf-8")
    log(f"wrote {HERE / 'results' / 'fit_p3l.json'} and "
        f"{HERE / 'results' / 'stdout_fit.txt'}")
    print("\nDONE part 3l TASK B fit.")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
