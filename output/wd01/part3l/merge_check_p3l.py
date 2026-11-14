"""WD-01 part 3l TASK A: merge_acceptance rejection-cause audit (TRAIN).

p3k found the stage-1 fit loss IDENTICAL across b_accept 2.0-6.5.  This
script answers why, on the 8 TRAIN fit windows (first 4 sorted train
roots x t0 {120, 300}, H=60 s rollouts, kind 'idm_merge'):

  1. how often each M2 rejection cause fires: the lead-gap term
     (gap_lead < minGap + v*tau_accept) vs the follower-deceleration
     term (b_req > b_accept) vs both;
  2. the distribution of the follower's required deceleration b_req
     over follower-present checks (the term can only bind above
     b_accept);
  3. whether the extended b_accept {1.5, 2.0, 3.0} and tau_accept
     {1.9, 2.4, 3.0, 3.6} grids move the rejection pattern at all.

The follower slot itself (nearest main_right vehicle BEHIND the
candidate in d, engine._gap_accept_merge) is pinned by the unit test
test_merge_check_p3l.py::test_follower_slot_is_nearest_behind.

ZERO simulation; holdout blocked in root_arrays (never passed).  TRAIN
fit roots only.
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
sys.path.insert(0, str(HERE.parent / "part3k"))   # decompose_p3l helpers
import decompose_p3k as D  # noqa: E402
import engine as E  # noqa: E402
import p3l_lib as L  # noqa: E402
import p3lib as P3  # noqa: E402

T_START = time.time()
K = D.K_STEPS                     # H = 60 s rollout, the fit convention
EVAL_T0S = (120.0, 300.0)
MAIN = dict(tau_accept=1.9, b_accept=2.0, p_coop_scale=0.0)
SWEEPS = [dict(tau_accept=2.4, b_accept=2.0, p_coop_scale=0.0),
          dict(tau_accept=3.0, b_accept=2.0, p_coop_scale=0.0),
          dict(tau_accept=3.6, b_accept=2.0, p_coop_scale=0.0),
          dict(tau_accept=1.9, b_accept=1.5, p_coop_scale=0.0),
          dict(tau_accept=1.9, b_accept=3.0, p_coop_scale=0.0)]
LC_REF = 0.002                    # p3k fitted lc_scale (reference hazard)


def log(m):
    print(f"[{time.time()-T_START:6.1f}s] {m}", flush=True)


def roll(root, t0, mp, dev):
    return E.Rollout(root, float(t0), kind="idm_merge", member_seed=1,
                     lc_scale=LC_REF, merge_params=dict(mp), dev=dev).run()


def agg_stats(rollouts):
    """Sum engine merge_stats across rollouts (keys via p3l_lib, the
    canonical spellings; breq histogram elementwise, rejection list
    concatenated), plus the accel->main_right merge count."""
    tot = {}
    for r in rollouts:
        st = getattr(r, "merge_stats", None) or {}
        for k, v in st.items():
            if k == L.MS_BREQ_HIST:
                acc = tot.setdefault(k, [0] * len(v))
                for i, c in enumerate(v):
                    acc[i] += int(c)
            elif k == L.MS_BREQ_FOLL_REJ:
                tot.setdefault(k, []).extend(float(x) for x in v)
            else:
                tot[k] = int(tot.get(k, 0)) + int(v)
        tot["merges"] = int(tot.get("merges", 0)) + len(r.merge_d)
    return tot


def report(tag, mp, tot):
    br = np.array(tot.get(L.MS_BREQ_FOLL_REJ, []), dtype=np.float64)
    if len(br):
        bqs = "q10/50/90/max = %.2f/%.2f/%.2f/%.2f" % (
            tuple(np.percentile(br, [10, 50, 90])) + (float(br.max()),))
    else:
        bqs = "n/a (no follower-only rejection)"
    print(f"\n  [{tag}] merge_params={mp}")
    print(f"    M2 checks={tot.get(L.MS_CHECKS, 0)} "
          f"accepted={tot.get(L.MS_ACCEPTED, 0)} "
          f"(rate {tot.get(L.MS_ACCEPTED, 0) / max(tot.get(L.MS_CHECKS, 1), 1):.3f}"
          f"); accel->main_right merges={tot.get('merges', 0)}")
    print(f"    rejections: lead-only={tot.get(L.MS_REJ_LEAD, 0)} "
          f"foll-only={tot.get(L.MS_REJ_FOLL, 0)} "
          f"both={tot.get(L.MS_REJ_BOTH, 0)}")
    print(f"    follower-present checks={tot.get(L.MS_WITH_FOLL, 0)} "
          f"({100.0 * tot.get(L.MS_WITH_FOLL, 0) / max(tot.get(L.MS_CHECKS, 1), 1):.1f} %)"
          f"; b_req histogram <=0 / (0,1] / (1,1.5] / (1.5,2] / (2,3] / >3:"
          f" {tot.get(L.MS_BREQ_HIST, [0]*6)}")
    print(f"    b_req at follower-only rejections: {bqs}")
    return tot


def main():
    import torch
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    (HERE / "results").mkdir(parents=True, exist_ok=True)
    sys.stdout = D.Tee(HERE / "results" / "stdout_merge_check.txt")

    print("priority below-normal:", D.below_normal_priority())
    dev = P3.device()
    sp = L.splits_3l()
    print("=== WD-01 part 3l TASK A: merge rejection causes (TRAIN) ===")
    print("ZERO simulation; holdout blocked in root_arrays (never passed).")
    print(f"device={dev} torch_threads={torch.get_num_threads()}")
    print(f"train fit roots: {sp['fit']}; t0 {EVAL_T0S}; H=60 s (K={K} "
          f"steps of 0.5 s); lc_scale={LC_REF} (p3k fit); member_seed=1")
    print(f"calibration roots (used only by calibrate_p3l.py, never "
          f"here): {sp['calibration']}")
    print(f"validation roots NOT used here: {sp['val']}; "
          f"{len(sp['holdout'])} holdout roots blocked")

    out = {}
    for mp, tag in ([(MAIN, "main (p3k fit)")]
                    + [(s, f"tau={s['tau_accept']} b={s['b_accept']}")
                       for s in SWEEPS]):
        log(f"rolling {tag} over {len(sp['fit']) * len(EVAL_T0S)} windows ...")
        rs = [roll(root, t0, mp, dev) for root in sp["fit"]
              for t0 in EVAL_T0S]
        tot = report(tag, mp, agg_stats(rs))
        out[tag] = dict(merge_params=dict(mp),
                        stats={k: v for k, v in tot.items()})

    # ---- TASK A verdict
    mt = out["main (p3k fit)"]["stats"]
    n_foll_rej = mt.get(L.MS_REJ_FOLL, 0) + mt.get(L.MS_REJ_BOTH, 0)
    hist = mt.get(L.MS_BREQ_HIST, [0] * 6)
    above_grid_min = sum(hist[3:])       # b_req > 1.5 (smallest grid b)
    verdict = dict(
        follower_slot=("nearest main_right vehicle BEHIND the candidate "
                       "in d (engine._gap_accept_merge); pinned by unit "
                       "test test_follower_slot_is_nearest_behind"),
        follower_rejections=n_foll_rej,
        follower_term_binds=bool(n_foll_rej),
        breq_checks_above_b_grid_min=above_grid_min,
        b_accept_flat_explained=above_grid_min == 0,
    )
    print("\n== TASK A verdict ==")
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    if not verdict["follower_term_binds"]:
        print("  verdict: the follower term NEVER binds on the TRAIN fit "
              "windows -- every M2 rejection is caused by the lead-gap "
              "term, which is why p3k's fit loss was identical across "
              "b_accept; the follower slot is the correct target-lane "
              "follower (unit-tested) and needs NO fix.  Extending "
              "b_accept to 1.5/2.0/3.0 cannot change the fit by more "
              "than zero rejections unless a check has b_req in (1.5, "
              "2.0]; the histogram bounds that count.")
    out["verdict"] = verdict
    out["meta"] = dict(part="3l", task="A", split="TRAIN fit roots only",
                       windows=len(sp["fit"]) * len(EVAL_T0S),
                       holdout_accessed=False, simulation=False)
    (HERE / "results" / "merge_check_p3l.json").write_text(
        json.dumps(out, indent=1, default=str), encoding="utf-8")
    log(f"wrote {HERE / 'results' / 'merge_check_p3l.json'}")
    print("\nDONE part 3l TASK A.")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
