"""WD-01 part 3q TASK B: the merge-lever refit on the k-fold TRAIN
windows (folds f1..f4 = 13 roots x 7 t0 = 91 windows; fold f0 is held
out of ALL parameter fitting), iteration-protocol driven and resumable
per line entry.

Fix target (brief): the systematic MERGE over-count (accel->main_right)
-- +40 % pooled on the p3o TRAIN-internal folds, +59 % at the val8 look
-- with per-type LC as a first-class metric, alongside delay, TTC<3
exposure, occupancy and speed (the p3m stage-1 + p3n stage-2 loss,
UNCHANGED: lcM30/lcM60 are explicit loss terms).

Levers (the brief's list): merge_accept_prob, tau_accept/b_accept,
end_force_s, merge-zone POSITION dependence of acceptance (new engine
flags merge_early_d/merge_early_prob), cooperation (p_coop_scale), plus
the new merge_cooldown_ticks (hysteresis that makes thinning
count-effective; the p3o negative result showed plain per-tick-retry
thinning only delays ~1/p ticks).

Iteration 1 (declared):
  base  = p3o iter2 member 0 (the frozen candidate);
  lines = merge_cooldown_ticks (4, 8, 16)
          merge_early_prob  (0.3, 0.15)   at merge_early_d = -15.44
          merge_accept_prob (0.4, 0.7)
          lc_scale_disc     (0.0014, 0.0018)
          p_coop_scale      (0.25)
  Base-value line entries (cooldown 0, early_prob 0.55, mprob 0.55,
  disc 0.0016, pcoop 0.0) are the base roll's loss marked copied=true;
  merge_early_d = -15.44 is the end-force boundary (s = 394.10 m), so
  the early zone is exactly the pre-force part of the merge zone.
  Members 1-4 = p3o iter2 members 1-4 bases + softmax draws (T=0.15,
  rng 900+m) over (merge_cooldown_ticks, merge_accept_prob,
  merge_early_prob); merge_early_d = -15.44 on every member.

Protocol: fit windows are the k-fold TRAIN folds f1..f4 ONLY; fold f0
never enters the fit; validation never touched; holdout blocked.  Every
line entry is appended to results/fit_lines_p3q_iter{k}.json as soon as
it is computed; a restart skips computed entries.

ZERO simulation; single process, 2 torch threads, below-normal
priority.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("WD_DEVICE", "cpu")    # single process, shared box

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p3q_lib as Q  # noqa: E402
import p3o_lib as O  # noqa: E402
import decompose_p3k as D  # noqa: E402

T_START = time.time()
EARLY_D_AT = -15.44                  # the end-force boundary in d
SOFTMAX_T = 0.15
DRAW_FIELDS = ("merge_cooldown_ticks", "merge_accept_prob",
               "merge_early_prob")
ITERATIONS = {
    1: dict(
        base_from="p3o_iter2_member0",
        early_d=EARLY_D_AT,
        lines=(("merge_cooldown_ticks", (4, 8, 16)),
               ("merge_early_prob", (0.3, 0.15)),
               ("merge_accept_prob", (0.4, 0.7)),
               ("lc_scale_disc", (0.0014, 0.0018)),
               ("p_coop_scale", (0.25,))),
        base_equiv={"merge_cooldown_ticks": 0,
                    "merge_early_prob": 0.55,
                    "merge_accept_prob": 0.55,
                    "lc_scale_disc": 0.0016,
                    "p_coop_scale": 0.0},
        member_bases="p3o_iter2_members_1_4",
        member_draw_fields=DRAW_FIELDS,
    ),
    # iteration 2 (declared AFTER the iter1 lines, still folds f1..f4
    # only): iter1's joint optimum took disc 0.0014 + early_prob 0.15
    # and REJECTED every merge-count lever (cooldown/tau/mprob all cost
    # TTC<3 exposure -- pred already OVER-estimates ttc3, bias -1.76 s
    # vs truth in the base gate).  Iter2 probes the trade frontier from
    # the iter1 winner: does the cooldown trade improve on the new base
    # (disc 0.0014 reshapes the queue), and do the gap-side levers
    # (tau_accept up, end_force_s down -- a smaller forced zone) cut the
    # merge count at acceptable joint?
    2: dict(
        base_from="p3q_iter1_member0",
        early_d=EARLY_D_AT,
        lines=(("merge_cooldown_ticks", (4, 8, 16)),
               ("tau_accept", (3.0, 3.6)),
               ("end_force_s", (45.0, 30.0)),
               ("lc_scale_disc", (0.0012,)),
               ("merge_accept_prob", (0.45,))),
        base_equiv={"tau_accept": 2.4,
                    "end_force_s": 60.0,
                    "lc_scale_disc": 0.0014,
                    "merge_accept_prob": 0.55,
                    "merge_cooldown_ticks": 0},
        member_bases="p3q_iter1_members_1_4",
        member_draw_fields=("merge_cooldown_ticks", "tau_accept",
                            "end_force_s"),
    ),
}


def log(m):
    print(f"[{time.time()-T_START:6.1f}s] {m}", flush=True)


def line_cfg(base, field, value, early_d):
    cfg = Q.set_field(base, field, value)
    if field == "merge_early_prob":
        cfg = Q.set_field(cfg, "merge_early_d", early_d)
    return cfg


def loss_entry(cfg, keys, truths, dev):
    preds = {k: Q.pred_fit_stats_q(k[0], k[1], cfg, dev) for k in keys}
    lj, l1, l2, d1, d2 = Q.joint_loss(preds, truths)
    return dict(joint=float(lj), stage1=float(l1), stage2=float(l2),
                detail1=d1, detail2=d2)


def main():
    import torch
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    args = ap.parse_args()
    k = args.iter
    if k not in ITERATIONS:
        raise SystemExit(f"no declared protocol for iteration {k}")
    decl = ITERATIONS[k]
    Q.RES.mkdir(parents=True, exist_ok=True)
    sys.stdout = D.Tee(Q.RES / f"stdout_fit_iter{k}.txt")
    print("priority below-normal:", D.below_normal_priority())
    print(f"=== WD-01 part 3q TASK B: merge-lever refit, iteration {k} "
          f"(declared: {decl}) ===")
    print("ZERO simulation; fit windows = k-fold TRAIN folds "
          f"{Q.FIT_FOLDS} (13 roots x 7 t0); fold {Q.DEV_FOLD} never "
          "enters the fit; validation never touched; holdout blocked.")
    print(f"device={Q.P3.device()} torch_threads={torch.get_num_threads()}")
    roots = Q.train17()
    assert set(Q.FIT_ROOTS) <= set(roots)
    assert not (set(Q.FIT_ROOTS) & set(Q.FOLDS[Q.DEV_FOLD]))
    assert not (set(Q.FIT_ROOTS) & set(Q.O.splits()["val"]))
    dev = Q.P3.device()

    if decl["base_from"] == "p3o_iter2_member0":
        cand = Q.N.load_json(Q.PART3O / "results" /
                             "candidate_p3o_iter2.json")
        base = dict(cand["members"][0])
    elif decl["base_from"].startswith("p3q_iter"):
        prev = int(decl["base_from"].split("iter")[1].split("_")[0])
        base = dict(Q.N.load_json(Q.RES / f"fit_p3q_iter{prev}.json")
                    ["members"][0])
    else:
        raise SystemExit(f"unknown base_from {decl['base_from']}")
    print(f"base: {Q.cfg_note(base)}")

    keys = [(r, float(t0)) for r in sorted(Q.FIT_ROOTS)
            for t0 in Q.GATE_T0S]
    log(f"computing truth targets for {len(keys)} fit windows ...")
    truths = Q.truth_fit_targets(Q.FIT_ROOTS, Q.GATE_T0S)
    print("TRAIN truth LC by type per fit window (pooled means): "
          f"lcM30 {np.mean([t['lcM30'] for t in truths.values()]):.2f} "
          f"lcD30 {np.mean([t['lcD30'] for t in truths.values()]):.2f} "
          f"| lcM60 {np.mean([t['lcM60'] for t in truths.values()]):.2f}"
          f" lcD60 {np.mean([t['lcD60'] for t in truths.values()]):.2f}")

    # ---- base roll (the reference every line compares against)
    lines_path = Q.RES / f"fit_lines_p3q_iter{k}.json"
    lines = {}
    if lines_path.exists():
        try:
            lines = Q.N.load_json(lines_path)
            print(f"resuming: {lines_path.name} present")
        except Exception as e:
            print(f"WARNING: {lines_path.name} unreadable ({e}) -- "
                  f"starting the line tables fresh (a truncated file, "
                  f"e.g. from an ENOSPC mid-write, must not block the "
                  f"resume)")
    if "_base" not in lines:
        log(f"rolling the BASE config on {len(keys)} fit windows ...")
        e = loss_entry(base, keys, truths, dev)
        lines["_base"] = e
        lines_path.write_text(json.dumps(lines, indent=1),
                              encoding="utf-8")
    base_loss = lines["_base"]
    print(f"BASE joint={base_loss['joint']:.4f} "
          f"(s1 {base_loss['stage1']:.4f} s2 {base_loss['stage2']:.4f}; "
          f"lcM30 {base_loss['detail2']['lcM30']:.3f} lcD30 "
          f"{base_loss['detail2']['lcD30']:.3f} delay "
          f"{base_loss['detail2']['delay']:.3f})")

    # ---- coordinate lines (resumable per entry; base values copied)
    for field, grid in decl["lines"]:
        lines.setdefault(field, [])
        have = {round(e["value"], 12) for e in lines[field]}
        bev = float(decl["base_equiv"][field])
        if not any(abs(v - bev) < 1e-12 for v in have):
            lines[field].append(dict(
                value=bev, joint=base_loss["joint"],
                stage1=base_loss["stage1"], stage2=base_loss["stage2"],
                detail1=base_loss["detail1"],
                detail2=base_loss["detail2"], copied=True,
                note="base-equivalent value; loss = the base roll"))
        for g in grid:
            if any(abs(e["value"] - float(g)) < 1e-12
                   for e in lines[field]):
                continue
            log(f"LINE {field}={g}: rolling {len(keys)} fit windows ...")
            cfg = line_cfg(base, field, float(g), decl["early_d"])
            e = loss_entry(cfg, keys, truths, dev)
            e["value"] = float(g)
            lines[field].append(e)
            lines_path.write_text(json.dumps(lines, indent=1),
                                  encoding="utf-8")
            log(f"LINE {field}={g}: joint={e['joint']:.4f} "
                f"(s1 {e['stage1']:.4f} s2 {e['stage2']:.4f}; lcM30 "
                f"{e['detail2']['lcM30']:.3f} lcD30 "
                f"{e['detail2']['lcD30']:.3f} delay "
                f"{e['detail2']['delay']:.3f})")
    print("\n== line tables (fit windows; includes the copied "
          "base-equivalent entries) ==")
    winners = {}
    for field, grid in decl["lines"]:
        tab = sorted(lines[field], key=lambda e: e["value"])
        best = min(tab, key=lambda e: e["joint"])
        winners[field] = best["value"]
        row = "  ".join(
            f"{e['value']}:{e['joint']:.4f}"
            f"{'*' if e.get('copied') else ''}" for e in tab)
        print(f"  {field:<22} {row}  -> winner {best['value']}")

    # ---- combined candidate (member 0) + verification roll
    cand0 = dict(base)
    for field, w in winners.items():
        cand0 = Q.set_field(cand0, field, w)
    cand0 = Q.set_field(cand0, "merge_early_d", decl["early_d"])
    if winners.get("merge_early_prob", None) is None:
        cand0.pop("merge_early_prob", None)
        cand0["merge_early_prob"] = None
    log(f"AFTER: rolling the combined candidate on {len(keys)} fit "
        f"windows ...")
    after = loss_entry(cand0, keys, truths, dev)
    print(f"\nAFTER (combined {winners}): joint={after['joint']:.4f} "
          f"(s1 {after['stage1']:.4f} s2 {after['stage2']:.4f}) "
          f"detail2={after['detail2']}")
    print(f"BEFORE (base): joint={base_loss['joint']:.4f}")

    # ---- members 1-4: p3o iter2 bases + softmax draws over the merge
    # fields; early_d = the declared value on EVERY member (their drawn
    # early_prob is never None: the draw pool carries the base-equivalent
    # 0.55 = thinning without position dependence)
    members = [dict(cand0, seed=int(cand0.get("seed", 1)),
                    lc_scale=float(cand0["lc_scale"]),
                    lc_scale_disc=float(cand0["lc_scale_disc"]),
                    merge_accept_prob=float(
                        cand0.get("merge_accept_prob", 1.0)),
                    merge_cooldown_ticks=int(
                        cand0.get("merge_cooldown_ticks") or 0),
                    merge_early_d=float(
                        cand0.get("merge_early_d") or 0.0),
                    merge_early_prob=cand0.get("merge_early_prob"),
                    dawdle_scale=float(cand0.get("dawdle_scale", 1.0)))]
    prev_fit = None
    if decl["member_bases"] == "p3o_iter2_members_1_4":
        prev_fit = Q.N.load_json(Q.PART3O / "results" /
                                 "fit_p3o_iter2.json")
    elif decl["member_bases"].startswith("p3q_iter"):
        prev = int(decl["member_bases"].split("iter")[1].split("_")[0])
        prev_fit = Q.N.load_json(Q.RES / f"fit_p3q_iter{prev}.json")
    else:
        raise SystemExit(f"unknown member_bases {decl['member_bases']}")
    for m in range(1, 5):
        mb = dict(prev_fit["members"][m])
        rng = np.random.default_rng(900 + m)
        mem = dict(mb)
        draws = {}
        for field in decl["member_draw_fields"]:
            tab = sorted(lines[field], key=lambda e: e["value"])
            pick = Q.softmax_draw([e["joint"] for e in tab], rng,
                                  SOFTMAX_T)
            mem = line_cfg(mem, field, tab[pick]["value"],
                           decl["early_d"])
            draws[field] = tab[pick]["value"]
        members.append(dict(
            mem, seed=int(mb["seed"]),
            lc_scale=float(mem["lc_scale"]),
            lc_scale_disc=float(mem.get("lc_scale_disc")
                                or mem["lc_scale"]),
            merge_accept_prob=float(mem.get("merge_accept_prob", 1.0)),
            merge_cooldown_ticks=int(
                mem.get("merge_cooldown_ticks") or 0),
            merge_early_d=float(mem.get("merge_early_d") or 0.0),
            merge_early_prob=mem.get("merge_early_prob"),
            dawdle_scale=float(mem.get("dawdle_scale", 1.0))))
        log(f"member {m}: base ({decl['member_bases']} #{m}) + draws "
            f"{draws} -> {Q.cfg_note(members[-1])}")

    result = dict(
        part="3q", iteration=k,
        variant="L6 kind='idm_merge' (p3q merge-hazard fix: cooldown "
                "hysteresis + zone-position-dependent acceptance on top "
                "of the p3o per-type levers)",
        fitted_on=dict(roots=list(Q.FIT_ROOTS),
                       folds={f: list(Q.FOLDS[f]) for f in
                              Q.FIT_FOLDS},
                       dev_fold=Q.DEV_FOLD,
                       t0s=list(Q.GATE_T0S), H=60.0, windows=len(keys),
                       note="k-fold TRAIN fit folds f1..f4 ONLY; fold "
                            "f0 never entered the fit; holdout never "
                            "accessed; validation never touched"),
        declared=decl,
        loss_form="p3m stage1 + p3n stage2 (LC by type H30 primary + "
                  "H60 half weight, TTC<3, physics delay, mr guard) -- "
                  "the per-type MERGE COUNT is an explicit loss term",
        base=dict(base_from=decl["base_from"], config=base),
        lines=lines, winners=winners,
        after=dict(member0=after),
        members=members,
        member_protocol=dict(
            bases="p3o_iter2_members_1_4",
            draws=f"softmax T={SOFTMAX_T} over the shared "
                  "iteration-level lines of "
                  f"{decl['member_draw_fields']}, rng 900+m; "
                  f"merge_early_d={decl['early_d']} on every member",
            rationale="the by-type merge count signal dominates the "
                      "joint loss (lines robust to base); prior bases "
                      "keep the ensemble spread at zero extra roll "
                      "cost"),
        grids=dict(declared_lines={f: list(g) for f, g
                                   in decl["lines"]},
                   softmax_T=SOFTMAX_T, early_d=decl["early_d"]),
    )
    (Q.RES / f"fit_p3q_iter{k}.json").write_text(
        json.dumps(result, indent=1, default=str), encoding="utf-8")
    log(f"wrote {Q.RES / f'fit_p3q_iter{k}.json'}")
    print(f"\nDONE part 3q fit iteration {k}.")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
