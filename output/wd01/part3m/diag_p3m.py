"""WD-01 part 3m TASK A: WHERE does the excess TTC<3 s exposure come
from?  (brief: at H=30 the L5c member band misses LOW in 37/52 validation
windows -- the emulator over-predicts close-closing exposure, a BIAS not
a spread problem).

Method: roll the part-3 engine (kind 'idm_merge', the p3l L5 member
configs) over TRAIN fit roots and VALIDATION roots at the 60 s t0 grid,
H = 30 s, and account every same-lane consecutive follower->leader pair
with TTC < 3 s per 0.5 s tick on THREE views:
  pa  pred_all     all predicted vehicles     (the frozen metric)
  pm  pred_matched the matched vehicle set, PREDICTED states
  tr  truth        the matched vehicle set, TRUE states (frozen metric)
pa - tr is the metric bias; pa - pm isolates vehicle-SET composition and
pm - tr isolates BEHAVIOUR (same vehicles, same lanes, predicted vs true
states).  Pair ticks are attributed by lane, segment (p3j bins),
event class (post-merge / post-discretionary-LC / pure car-following
within 5 s, pair-level priority merge > lc > cf, with `who` = which side
merged), TTC sub-band, vType (pred: sampled; truth: supervision label)
and member dawdle_scale.  Per window, pa/tr are asserted IDENTICAL to
engine.window_metrics' pred_/true_ttc3_exposure (the definition-parity
check the brief asks for).

ZERO simulation; holdout blocked in root_arrays (never passed); the fix
itself will be CHOSEN ON TRAIN ONLY.
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
import decompose_p3k as D  # noqa: E402  (Tee, below_normal_priority)
import engine as E  # noqa: E402
import p3m_lib as M  # noqa: E402
import p3lib as P3  # noqa: E402

T_START = time.time()
K = 60                       # H = 30 s at 0.5 s
T0S = tuple(range(120, 481, 60))          # 60 s grid (brief: fine)
KINDS = ("ev", "ev_lane", "seg", "lane", "who", "ttc", "vt", "dawdle")
TOP = 8


def log(m):
    print(f"[{time.time()-T_START:6.1f}s] {m}", flush=True)


def member_configs():
    """The five p3l L5 member configs (the FAILING reference emulator)."""
    fit = M.L3.load_fit(HERE.parent / "part3l" / "results" /
                        "fit_p3l.json")
    out = []
    for mi, mem in enumerate(fit["members"]):
        out.append((mi, int(mem["seed"]), float(mem["lc_scale"]),
                    dict(mem["merge_params"]),
                    float(mem.get("dawdle_scale", 1.0))))
    return out


def roll_window(root, t0, cfg, dev):
    mi, seed, lc, mp, dw = cfg
    return E.Rollout(root, float(t0), kind="idm_merge", member_seed=seed,
                     lc_scale=lc, merge_params=dict(mp), dawdle_scale=dw,
                     dev=dev).run(n_ticks=K)


def print_table(title, cells):
    rows = M.excess_table(cells)
    print(f"\n  -- {title} (excess = pa - tr; veh s pooled) --")
    print("  key                          excess     pa     pm     tr  "
          "  n_pa  n_pm  n_tr")
    for exc, pa, pm, tr, na, nm, nt, key in rows[:TOP]:
        print(f"  {str(key):<28} {exc:7.1f} {pa:6.1f} {pm:6.1f} "
              f"{tr:6.1f} {na:6d} {nm:5d} {nt:5d}")


def summarise_tot(tots):
    pa = float(np.mean([t["pred_s"] for t in tots]))
    pm = float(np.mean([t["pred_matched_s"] for t in tots]))
    tr = float(np.mean([t["truth_s"] for t in tots]))
    return dict(n_windows=len(tots), pred_all=round(pa, 3),
                pred_matched=round(pm, 3), truth=round(tr, 3),
                bias=round(pa - tr, 3),
                composition=round(pa - pm, 3),
                behaviour=round(pm - tr, 3))


def run_split(split, roots, cfgs, dev):
    """Returns (pooled_cells, per_member_cells, per_window_tots)."""
    pooled = {k: {} for k in M.CELL_KINDS}
    per_member = {}
    wins = []
    n_win = 0
    for mi, seed, lc, mp, dw in cfgs:
        mcells = {k: {} for k in M.CELL_KINDS}
        for root in roots:
            ra = P3.root_arrays(root)
            for t0 in T0S:
                r = roll_window(root, t0, (mi, seed, lc, mp, dw), dev)
                cells, tot = M.window_cells(r, ra, K)
                wm = E.window_metrics(r.snapshots, ra, r.k0, H=K * 0.5)
                assert abs(tot["pred_s"] - wm["pred_ttc3_exposure"]) \
                    < 1e-9, (root, t0, mi, "pred parity")
                assert abs(tot["truth_s"] - wm["true_ttc3_exposure"]) \
                    < 1e-9, (root, t0, mi, "truth parity")
                M.merge_cells(pooled, cells)
                M.merge_cells(mcells, cells)
                wins.append(dict(split=split, member=mi, root=root,
                                 t0=int(t0), dawdle_scale=dw,
                                 lc_scale=lc, p_coop_scale=mp.get(
                                     "p_coop_scale"),
                                 b_accept=mp.get("b_accept"),
                                 tau_accept=mp.get("tau_accept"),
                                 pred_s=tot["pred_s"],
                                 pred_matched_s=tot["pred_matched_s"],
                                 truth_s=tot["truth_s"]))
                n_win += 1
        per_member[mi] = mcells
        log(f"  {split} member {mi}: {n_win} windows done "
            f"({summarise_tot([w for w in wins if w['member'] == mi])})")
    return pooled, per_member, wins


def main():
    import torch
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    (HERE / "results").mkdir(parents=True, exist_ok=True)
    sys.stdout = D.Tee(HERE / "results" / "stdout_diag.txt")

    print("priority below-normal:", D.below_normal_priority())
    dev = P3.device()
    sp = M.L3.splits_3l()
    cfgs = member_configs()
    print("=== WD-01 part 3m TASK A: TTC<3 exposure excess diagnosis "
          "(TRAIN fit roots + VALIDATION report-only) ===")
    print("ZERO simulation; holdout blocked in root_arrays (never passed).")
    print(f"device={dev} torch_threads={torch.get_num_threads()}")
    print(f"H=30 s (K={K} steps of 0.5 s), t0 grid every 60 s {T0S}; "
          f"train-diagnosis roots = FIT roots {sp['fit']}; validation "
          f"roots {sp['val']} (report only); holdout blocked")
    print("member configs (p3l L5 reference members):")
    for mi, seed, lc, mp, dw in cfgs:
        print(f"  member {mi}: seed={seed} lc_scale={lc} merge_params="
              f"{mp} dawdle_scale={dw}")

    out = dict(meta=dict(part="3m", task="A", K=K, t0s=list(T0S),
                         windows_per_split_per_member=len(sp["fit"])
                         * len(T0S), holdout_accessed=False,
                         simulation=False,
                         note=("pa = pred all (frozen metric), pm = "
                               "matched vehicles with predicted states, "
                               "tr = matched vehicles with true states; "
                               "fix chosen on TRAIN only")),
               per_window=[])
    for split, roots in (("train", sp["fit"]), ("validation", sp["val"])):
        log(f"== {split}: {len(roots)} roots x {len(T0S)} t0 x "
            f"{len(cfgs)} members ...")
        pooled, per_member, wins = run_split(split, roots, cfgs, dev)
        out["per_window"].extend(wins)
        out[split] = dict(totals=summarise_tot(wins),
                          cells=M.cells_to_json({k: pooled[k]
                                                 for k in KINDS}),
                          parity="pa/tr == window_metrics on every window")
        print(f"\n==== {split.upper()} pooled over {len(wins)} windows "
              f"({summarise_tot(wins)}) ====")
        for kind in KINDS:
            print_table(f"{kind} (all members)", pooled[kind])
        for mi, mcells in per_member.items():
            dw = cfgs[mi][4]
            lc = cfgs[mi][2]
            pcoop = cfgs[mi][3].get("p_coop_scale")
            print(f"\n  ==== member {mi} (dawdle={dw} lc_scale={lc} "
                  f"p_coop={pcoop}) -- event + lane tables ====")
            for kind in ("ev", "ev_lane"):
                rows = M.excess_table(mcells[kind])
                tot = [r for r in rows]
                print(f"   member {mi} {kind}: "
                      + "; ".join(f"{r[7]}={r[0]:+.1f}(pa {r[1]:.1f}/"
                                  f"tr {r[3]:.1f})" for r in tot[:4]))
            out[f"{split}_member{mi}"] = M.cells_to_json(
                {k: mcells[k] for k in ("ev", "ev_lane", "lane", "seg")})

    (HERE / "results" / "diag_p3m.json").write_text(
        json.dumps(out, indent=1, default=str), encoding="utf-8")
    log(f"wrote {HERE / 'results' / 'diag_p3m.json'}")
    print("\nDONE part 3m TASK A diagnosis.")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
