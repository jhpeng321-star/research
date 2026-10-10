"""WD-01 part 3o TASK B': the per-type LC-hazard refit (TRAIN fit roots
ONLY), iteration-protocol driven and resumable per line entry.

Fix target: the p3n iter1 gate-v2 (b) residual ON TRAIN -- merge LC
(accel->main_right) over-counts ~+40-45 % and discretionary (0<->1)
~+17-24 % pooled on the TRAIN-internal folds, total +21.9 % at H=30.

Declared levers (the brief): lc_scale PER TYPE (the new engine flags
lc_scale_disc + merge_accept_prob = separate discretionary and merge
hazards), end_force_s, tau_accept/b_accept.

Iteration 1 (declared):
  base        = p3n fit member 0 (lc 0.002, tau 1.9, b 2.0, p_coop 0,
                force 60, blend 0, dawdle 1, seed 1);
  lines       = merge_accept_prob (1.0, 0.85, 0.7, 0.55)
                lc_scale_disc  (0.002, 0.0016, 0.0012)
                tau_accept     (1.9, 2.4)
  on the 4 fit roots x 7 t0 = 28 windows (the p3n iteration-2 fit
  windows), joint loss = p3m stage1 + p3n stage2 (LC by type at H30
  primary + H60 half weight, TTC<3, physics delay, mr/occupancy
  guards);
  members 1-4 = the p3n iter1 members 1-4 bases UNCHANGED (coordinate
  diversity inherited) + per-member softmax draws (T=0.15, seeded
  900+m) over the SHARED iteration-level lines of the two NEW fields;
  rationale: the merge/discretionary count signal dominates the joint
  loss, so the lines are robust to base, and the p3n bases keep the
  ensemble spread the (c) coverage needs at zero extra roll cost.

Protocol: fit windows are the FIT roots only; the calibration roots
(the TRAIN-internal gate split) are never rolled here; validation
never touched; holdout blocked.  Every line entry is appended to
results/fit_lines_iter{k}.json as soon as it is computed; a restart
skips computed entries and complete member builds.

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
import p3o_lib as O  # noqa: E402
import p3n_lib as N  # noqa: E402
import decompose_p3k as D  # noqa: E402
import p3lib as P3  # noqa: E402

T_START = time.time()
EVAL_T0S = tuple(range(120, 481, 60))          # 4 fit roots x 7 = 28
SOFTMAX_T = 0.15
MERGE_PROB_GRID = (1.0, 0.85, 0.7, 0.55)
DISC_GRID = (0.002, 0.0016, 0.0012)
TAU_GRID = (1.9, 2.4)
MERGE_FIELDS = ("merge_accept_prob", "lc_scale_disc")
ITERATIONS = {
    1: dict(
        base_from="p3n_iter1_member0",
        lines=(("merge_accept_prob", MERGE_PROB_GRID),
               ("lc_scale_disc", DISC_GRID),
               ("tau_accept", TAU_GRID)),
        member_draw_fields=MERGE_FIELDS,
        member_bases="p3n_iter1_members_1_4",
    ),
    # iteration 2 (declared AFTER the iter1 gate, still TRAIN-internal
    # only): iter1 fixed the TOTAL LC ((b) +6.2 %/-1.8 %, (a)/(d) PASS,
    # CRPS better at H30) but the MERGE type still over-counts
    # (+39.8 %/==+23.5 % pooled): merge_accept_prob thinning only
    # DELAYS queue-front retries (arrival-limited), it does not prevent
    # merges.  Most queue merges accept inside the relaxed end-force
    # zone (tau<=0.5, b<=6.0 for end_force_s=60 -> zone [394, 454] m),
    # so the count-effective levers are the zone extent (end_force_s
    # down) and the outside-zone criterion (tau/b up/down).  Disc goes
    # back UP slightly (iter1 pooled disc -8.2 % @H30 / -12.5 % @H60,
    # fold_A -29.8 %).
    2: dict(
        base_from="p3o_iter1_member0",
        lines=(("end_force_s", (60.0, 45.0, 30.0, 15.0)),
               ("tau_accept", (2.4, 3.0, 3.6)),
               ("b_accept", (2.0, 1.5, 1.0)),
               ("merge_accept_prob", (0.55, 0.7, 0.85)),
               ("lc_scale_disc", (0.0016, 0.0018, 0.002))),
        member_draw_fields=("end_force_s", "merge_accept_prob"),
        member_bases="p3o_iter1_members_1_4",
    ),
}


def log(m):
    print(f"[{time.time()-T_START:6.1f}s] {m}", flush=True)


def set_field(cfg, field, value):
    cfg = json.loads(json.dumps(cfg))          # deep copy
    if field in ("tau_accept", "b_accept", "end_force_s",
                 "v_merge_blend", "p_coop_scale"):
        cfg["merge_params"] = dict(cfg["merge_params"], **{field: value})
    else:
        cfg[field] = value
    return cfg


def get_field(cfg, field):
    if field in ("tau_accept", "b_accept", "end_force_s",
                 "v_merge_blend", "p_coop_scale"):
        return float(cfg["merge_params"][field])
    v = cfg.get(field)
    if v is None:
        # the p3n bases carry no p3o keys: the engine's own fallbacks
        v = {"lc_scale_disc": cfg.get("lc_scale"),
             "merge_accept_prob": 1.0}.get(field)
    return v


def loss_entry(cfg, keys, truths, dev):
    preds = {k: O.pred_fit_stats_o(k[0], k[1], cfg, dev)
             for k in keys}
    lj, l1, l2, d1, d2 = O.joint_loss(preds, truths)
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
    O.RES.mkdir(parents=True, exist_ok=True)
    sys.stdout = D.Tee(HERE / "results" / f"stdout_fit_iter{k}.txt")
    print("priority below-normal:", D.below_normal_priority())
    print(f"=== WD-01 part 3o TASK B': per-type LC-hazard refit, "
          f"iteration {k} (declared: {decl}) ===")
    print("ZERO simulation; fit roots ONLY; calibration split not "
          "rolled here; validation never touched; holdout blocked.")
    print(f"device={P3.device()} torch_threads={torch.get_num_threads()}")
    sp = O.splits()
    fit_roots = tuple(sp["fit"])
    assert not (set(fit_roots) & set(sp["calibration"]))
    assert not (set(fit_roots) & set(sp["val"]))
    dev = P3.device()

    p3n_fit = N.load_json(O.PART3N / "results" / "fit_p3n.json")
    if decl["base_from"] == "p3n_iter1_member0":
        base = dict(p3n_fit["members"][0])
    elif decl["base_from"].startswith("p3o_iter"):
        prev = int(decl["base_from"].split("iter")[1].split("_")[0])
        base = dict(N.load_json(O.RES / f"fit_p3o_iter{prev}.json")
                    ["members"][0])
    else:
        raise SystemExit(f"unknown base_from {decl['base_from']}")
    print(f"base: {O.cfg_note(base)}")

    keys = [(r, float(t0)) for r in sorted(fit_roots)
            for t0 in EVAL_T0S]
    log(f"computing truth targets for {len(keys)} fit windows ...")
    truths = O.truth_fit_targets(fit_roots, EVAL_T0S)
    print("TRAIN truth LC by type per fit window (pooled means): "
          f"lcM30 {np.mean([t['lcM30'] for t in truths.values()]):.2f} "
          f"lcD30 {np.mean([t['lcD30'] for t in truths.values()]):.2f} "
          f"| lcM60 {np.mean([t['lcM60'] for t in truths.values()]):.2f}"
          f" lcD60 {np.mean([t['lcD60'] for t in truths.values()]):.2f}")

    # ---- coordinate lines (resumable per entry)
    lines_path = O.RES / f"fit_lines_iter{k}.json"
    lines = {}
    if lines_path.exists():
        lines = N.load_json(lines_path)
        print(f"resuming: {lines_path.name} present")
    for field, grid in decl["lines"]:
        lines.setdefault(field, [])
        for g in grid:
            if any(abs(e["value"] - float(g)) < 1e-12
                   for e in lines[field]):
                continue
            log(f"LINE {field}={g}: rolling {len(keys)} fit windows ...")
            cfg = set_field(base, field, float(g))
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
    print("\n== line tables (fit windows) ==")
    winners = {}
    for field, grid in decl["lines"]:
        tab = sorted(lines[field], key=lambda e: e["value"])
        best = min(tab, key=lambda e: e["joint"])
        winners[field] = best["value"]
        row = "  ".join(f"{e['value']}:{e['joint']:.4f}" for e in tab)
        print(f"  {field:<18} {row}  -> winner {best['value']}")
    base_losses = {f: next(e["joint"] for e in lines[f]
                           if abs(e["value"] -
                                  float(get_field(base, f))) < 1e-12)
                   for f, _g in decl["lines"]}
    print(f"  base loss (each line at its base value): {base_losses}")

    # ---- combined candidate (member 0) + verification roll
    cand = dict(base)
    for field, w in winners.items():
        cand = set_field(cand, field, w)
    log(f"AFTER: rolling the combined candidate on {len(keys)} fit "
        f"windows ...")
    after = loss_entry(cand, keys, truths, dev)
    print(f"\nAFTER (combined {winners}): joint={after['joint']:.4f} "
          f"(s1 {after['stage1']:.4f} s2 {after['stage2']:.4f}) "
          f"detail2={after['detail2']}")
    print(f"BEFORE (base): joint={base_losses[list(base_losses)[0]]:.4f}")

    # ---- members 1-4: p3n bases + softmax draws over the new fields
    members = [dict(cand, seed=int(cand.get("seed", 1)),
                    lc_scale=float(cand["lc_scale"]),
                    lc_scale_disc=float(cand["lc_scale_disc"]),
                    merge_accept_prob=float(
                        cand.get("merge_accept_prob", 1.0)),
                    dawdle_scale=float(cand.get("dawdle_scale", 1.0)))]
    for m in range(1, 5):
        if decl["member_bases"] == "p3n_iter1_members_1_4":
            mb = dict(p3n_fit["members"][m])
        elif decl["member_bases"].startswith("p3o_iter"):
            prev = int(decl["member_bases"].split("iter")[1].split("_")[0])
            mb = dict(N.load_json(O.RES / f"fit_p3o_iter{prev}.json")
                      ["members"][m])
        else:
            raise SystemExit(
                f"unknown member_bases {decl['member_bases']}")
        rng = np.random.default_rng(900 + m)
        mem = dict(mb)
        draws = {}
        for field in decl["member_draw_fields"]:
            tab = sorted(lines[field], key=lambda e: e["value"])
            pick = O.softmax_draw([e["joint"] for e in tab], rng,
                                  SOFTMAX_T)
            mem = set_field(mem, field, tab[pick]["value"])
            draws[field] = tab[pick]["value"]
        members.append(dict(
            mem, seed=int(mb["seed"]),
            lc_scale=float(mem["lc_scale"]),
            lc_scale_disc=float(mem.get("lc_scale_disc") or
                                mem["lc_scale"]),
            merge_accept_prob=float(mem.get("merge_accept_prob", 1.0)),
            dawdle_scale=float(mem.get("dawdle_scale", 1.0))))
        log(f"member {m}: base ({decl['member_bases']} #{m}) + draws "
            f"{draws} -> {O.cfg_note(members[-1])}")

    result = dict(
        part="3o", iteration=k,
        variant="L6 kind='idm_merge' (p3o per-type LC-hazard fix; "
                "engine flags lc_scale_disc + merge_accept_prob)",
        fitted_on=dict(roots=list(fit_roots), t0s=list(EVAL_T0S),
                       H=60.0,
                       note="TRAIN fit roots ONLY; the calibration "
                            "split (the TRAIN-internal gate) never "
                            "rolled here; holdout never accessed; "
                            "validation never touched"),
        declared=decl,
        loss_form="p3m stage1 + p3n stage2 (LC by type H30 primary + "
                  "H60 half weight, TTC<3, physics delay, mr guard)",
        base=dict(base_from=decl["base_from"], config=base,
                  line_losses_at_base=base_losses),
        lines=lines, winners=winners,
        after=dict(member0=after),
        members=members,
        member_protocol=dict(
            bases=decl["member_bases"],
            draws=f"softmax T={SOFTMAX_T} over the shared "
                  "iteration-level lines of "
                  f"{decl['member_draw_fields']}, rng 900+m",
            rationale="the by-type count signal dominates the joint "
                      "loss (lines robust to base); prior bases keep "
                      "the ensemble spread at zero extra roll cost"),
        grids=dict(declared_lines={f: list(g) for f, g
                                   in decl["lines"]},
                   softmax_T=SOFTMAX_T),
    )
    (O.RES / f"fit_p3o_iter{k}.json").write_text(
        json.dumps(result, indent=1, default=str), encoding="utf-8")
    log(f"wrote {O.RES / f'fit_p3o_iter{k}.json'}")
    print(f"\nDONE part 3o fit iteration {k}.")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
