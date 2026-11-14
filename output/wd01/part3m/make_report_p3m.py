"""WD-01 part 3m: assemble REPORT_P3M.md (narrative + verbatim stdout
pastes from results/stdout_*.txt, the part 3l convention)."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def paste(name, start=None, end=None):
    text = (RES / name).read_text(encoding="utf-8").splitlines()
    if start is not None:
        i = next(j for j, ln in enumerate(text) if start in ln)
        text = text[i:]
    if end is not None:
        j = next(j for j, ln in enumerate(text) if j and end in ln)
        text = text[:j]
    return "\n".join(text)


def main():
    diag = paste("stdout_diag.txt", start="H=30 s (K=60")
    fit = paste("stdout_fit.txt", start="priority below-normal")
    calib = paste("stdout_calibrate.txt", start="priority below-normal")
    evalp = paste("stdout_eval.txt", start="priority below-normal")
    agg = paste("stdout_aggregate.txt", start="=== WD-01 part 3m")

    report = f"""# WD-01 Part 3M — TTC<3 s exposure bias fix: per-side diagnosis, TRAIN refit with an added TTC<3 target (end-force + post-merge knobs, extended b grid), recalibrated spread frozen before validation, gate v2 re-run

****Holdout NOT accessed.**  No holdout root was loaded (the holdout block inside
`root_arrays` stays authoritative); no registration hash was created or passed
anywhere in part 3m.**

ZERO simulation: no SUMO/TraCI import anywhere; the only dynamics is the part-3
rollout engine (pure torch).  TRAIN fit roots for fitting, the TRAIN-internal
calibration split for spread, VALIDATION for reporting only.  Branch
zcode/wd01-p3m.  Data root read-only: WD01_DATA_ROOT=E:/rd-wt/wd01/output/wd01.

## Summary verdict

The last failing gate-v2 item for native Part A (L5c, criterion (c) TTC<3
exposure) is FIXED by the part 3m refit: at H=30 the L6c 10-90 % TTC<3 band
covers truth in **92.31 %** of the 52 validation windows (L5c: 59.62 %), with
the below-band bias GONE (0/52 windows below; L5c raw L5 band: 37/52 below,
exactly the brief's failure description), delay-phys coverage 80.77 %
(calibration-predicted 82.14 %) and CRPS 36.33 < L0 MAE 293.39 — criterion (c)
**PASSES for the first time**.  Gate v2 at H=30 still fails for L6c, now ONLY
on criterion (b): the refit rung overshoots lane changes (+39.6 % vs the 20 %
gate; TRAIN-only fitting chose lc_scale=0.002 and the end-force knob raises
the merge/LC count) — a residual generalisation gap for a follow-up part, not
a re-opened (c).  L6 (b) overshoot also costs the old gate at H=60 (speed
5.134 > 4.97); L5/L5c remain old-gate PASS (rows reused verbatim).

## TASK A — where the excess TTC<3 exposure comes from (diagnosis)

Instrumentation: `p3m_lib.py` extracts, per 0.5 s tick, every same-lane
CONSECUTIVE follower->leader pair with gap>0, closing>0, TTC<3 s — the exact
frozen `window_metrics` definition — on THREE views: **pa** = all predicted
vehicles (the metric's pred side), **pm** = the matched vehicle set with
PREDICTED states, **tr** = the matched vehicle set with TRUE states (the
metric's truth side).  pa-tr is the metric bias; pa-pm isolates vehicle-SET
composition, pm-tr isolates BEHAVIOUR.  Per window, pa/tr are asserted equal
to `window_metrics`' exposures (definition-parity check: PASSED on all 280
window-rolls, both sides, max error < 1e-9).  Leader selection, junction
handling and post-lane-change pair creation are therefore identical on both
sides by construction.  Event attribution is PER SIDE (pred-observed merge/LC
events vs truth-declared lc events) — a shared vid->events map would mislabel
pred vehicles whose truth twin already merged (that contamination was caught
and removed during the part).

Pooled over 4 roots x 7 t0 x 5 L5 members x H=30 (140 window-rolls per
split; veh s; full tables in the stdout below):

| quantity | TRAIN (fit roots) | VALIDATION |
|---|---|---|
| mean truth TTC<3 exposure/window | 2.73 | 7.32 |
| mean pred exposure/window | 8.49 | 14.25 |
| **bias (pa-tr)** | **+5.76** | **+6.93** |
| composition (pa-pm) | **0.00** | **0.00** |
| behaviour (pm-tr) | +5.76 | +6.93 |
| pure car-following excess | +660.0 (75 %) | +767.5 (84 %) |
| post-merge excess (pred / truth) | +122.5 / **0.0** | +134.0 / 7.5 |
| post-discretionary-LC excess | +23.5 | +68.5 |
| segment downstream (d<0) excess | +887.5 | +1284.0 |
| segment merge_edge / rsu_window | -53.0 / -28.5 | -233.5 / -84.0 |
| lane 2 / lane 1 excess | +418.5 / +370.5 | +440.5 / +480.5 |
| TTC band [2,3) / [1.5,2) excess | +563.0 / +202.5 | +558.5 / +324.5 |
| sampled vType 2 (cautious) pa vs tr | 387.0 vs 7.5 | 520.0 vs 55.0 |
| dawdle 1.5 vs 1.0 member excess | +342 vs +133.5 | +372.5 vs +189.0 |

**Verdict (TRAIN numbers; the fix was chosen on TRAIN only).**  The excess is
100 % BEHAVIOUR (composition exactly 0): (1) it concentrates DOWNSTREAM of
the merge point (d<0) where unmerged accel vehicles pile up at the lane-end
obstacle (M1) and stop-and-go in a queue — lane-2 pure-CF exposure is 3x the
truth's, while at the merge_edge/rsu_window approach the emulator
UNDER-predicts (the truth closes there, the emulator does not); (2) the
emulator's merges create transient closings the truth never shows (TRAIN:
122.5 pred vs 0.0 truth post-merge veh s) — a just-merged or just-discharging
vehicle and its new follower/leader at TTC 1.5-3 s; (3) nearly all flagged
pairs are MILD ([2,3) s; the truth has almost no pair-tick below 1.5 s while
the emulator has 468); (4) sampled cautious vTypes (tau 1.5, b 3.0) produce
10-50x the truth's cautious exposure, and dawdle_width-1.5 members ~2.6x the
width-1.0 member — spread-mechanism-B amplifiers, not the root cause.

## TASK B — TRAIN refit (chosen on TRAIN only)

Protocol = part 3l's staged grid fit on the same 8 TRAIN fit windows, with:
b_accept grid EXTENDED UP to {{2.0, 3.0, 4.0, 5.0}} (brief; the 1.5 low end
dropped — p3l measured it an order worse and the fix direction needs MORE
ramp discharge, not less); two new default-neutral engine knobs fitted:
`end_force_s` in {{0, 150}} (within 150 m of the accel-lane end, M2
acceptance relaxes to tau<=0.5 / b>=6.0 — SUMO strategic urgency: rolling
merges instead of the stop-and-go barrier queue) and `v_merge_blend` in
{{0, 0.5, 1}} (post-merge speed blend toward the target-lane local mean);
and the ADDED TARGET `((pred - truth)/max(truth, 5))^2` on per-window TTC<3
exposure in the stage-2 loss (truth over ALL truth rows, a fixed number per
window).  Stages: 1 acceptance (16 combos) -> 2 force x blend (6) -> 3
lc x p_coop (12) -> 4 acceptance re-sweep (16).

Selected (member 0): **tau_accept=1.9, b_accept=2.0, p_coop_scale=0.0,
end_force_s=150.0, v_merge_blend=0.0, lc_scale=0.002** (joint loss 1.8586).
The TTC<3 loss term collapsed from **6.019 (p3l BEFORE) to 0.335**; per
eval-window ttc3 pred/truth went from 26.0/4.0, 24.0/6.0, 18.0/2.0 ... to
8.0/4.0, 0.5/6.0, 0.0/2.0 ... (balanced, no systematic over).  The
end-force knob is decisive on TRAIN (stage-2 member lines: force off 25.9-30.1
vs on 1.38-1.55; all four wider-draw members drew force=150), and with the
knob active the b_accept sensitivity flattens (the relaxed near-end
acceptance discharges the queue the tight terms used to trap).  Members 1-4:
the p3l wider-draw mechanism (bootstrap resample x softmax T=0.15 coordinate
lines x independent dawdle draw) over the extended grids (draws: b 3-5,
tau 1.9-2.4, p_coop 0/0.25, blend 0/0.5, dawdle 1.0-1.5).

## TASK C — spread recalibration on the TRAIN-internal split, FROZEN

Same split and mean-centred mechanism as part 3l (x' = mean + s*(x-mean);
mean exactly invariant), on the L6 members, outputs delay_phys + ttc3 at the
gate horizons; factors FROZEN (file + sha256 sidecar + UTC timestamp) BEFORE
the validation run — `eval_p3m.py` refuses to run unless the sidecar hash
matches, and the gate re-checks it.

- calibration split: cv20_7002m2000, cv20_7003m2000, cv20_7004m2000,
  cv20_7007m2700 (28 windows x 5 members).
- frozen: delay_phys s_out H30 = 2.5198, H60 = 2.2974; ttc3 s_out H30 =
  7.6387, H60 = 14.5876 (the L6 members agree more closely than L5's, so the
  widening factor is larger; calibration coverage 0.8214 at all four
  output/horizon cells, mean shift exactly 0.0).
- **frozen_utc=2026-09-26T20:24:17.243056+00:00,
  sha256=968d20ba26d8552014807f0fbd3a2d9d2bf0b87c41d5aa49410d4ae804b97334**
  (validation rows written AFTER this timestamp; freeze gate log line in the
  eval stdout below).

## TASK D — gate v2 re-run (VALIDATION, 52 windows x M=5)

L0/L4/L5 rows reused verbatim from part 3l (no re-roll; L0 deterministic);
L5c derived with the UNCHANGED part 3l factors; L6 rolled fresh; L6c derived
with the frozen part 3m factors.

At H=30 (PRIMARY, (d) binding per-horizon form):

- **L6c: (a) PASS (0.03/1000) | (b) FAIL (+39.6 % LC) | (c) PASS (ttc3
  0.9231, delay-phys 0.8077, CRPS 36.33 < 293.39) | (d) PASS (4.3438 <
  4.3975, binding) ==> v2 FAIL: b_lc only.**
- L5c: unchanged verdict, FAIL on (c) (ttc3 0.5962).
- L0 FAIL (b), L4 FAIL (c), L6 FAIL (b,c).
- H=60 (secondary): L6c ttc3 0.8846 PASS but delay-phys 0.75 < 0.80 ->
  (c) FAIL, plus (b) +29.5 % -> v2 FAIL (b,c).
- Old gate: L5/L5c PASS at H=30 and H=60 (reused rows); L6/L6c FAIL (LC and
  the H=60 speed 5.134 > 4.97).

TTC<3 coverage breakdown (10-90 % band; the brief's "37 of 52 windows below"
is the RAW L5 band row below, 0.7115 x 52 = 37):

| rung@H30 | cov | below | above | below w/ truth=0 |
|---|---|---|---|---|
| L5 raw | 0.1923 | **0.7115 (37/52)** | 0.0962 | 5 |
| L5c (p3l factors) | 0.5962 | 0.3846 (20) | 0.0192 | 3 |
| L6 raw | 0.4038 | 0.3269 (17) | 0.2692 (14) | 2 |
| **L6c (p3m frozen)** | **0.9231** | **0.0 (0)** | 0.0769 (4) | **0** |

The residual bias flipped direction: after the fix the (widened) band never
sits above the truth; the 4 remaining misses are mild ABOVE-band windows
(the emulator now slightly UNDER-exposes on validation, within the widened
band for 48/52 windows).

## Residual gap (why gate v2 still fails) and NOT DONE

1. **(b) LC overshoot on the refit rung**: L6 pred_lc 16.87 vs truth 12.08
   (+39.6 %, gate +-20 %).  TRAIN fitting (which includes the LC targets)
   selected lc_scale=0.002 and end_force_s=150 jointly; the combination
   generalises to a higher validation merge/LC rate.  Re-tuning against the
   VALIDATION numbers is forbidden; a follow-up part should revisit the
   TRAIN loss (e.g. validate lc on the TRAIN-internal calibration split, or
   penalise the merge count explicitly).
2. **H=60 delay-phys coverage 0.75 < 0.80** for L6c (calibration predicted
   0.8214) — near-miss, same split-flip direction as the ttc3 item that this
   part fixed.
3. Sampled-vType posterior untouched (diagnosis flags the cautious draw as a
   10-50x exposure amplifier; the posterior is part 3d's).
4. `end_force_s` relaxation constants (tau 0.5 / b 6.0) are fixed knob
   constants, not fitted per-vType.
5. idm_sampled stale-llen defect (p3k item) recorded, not fixed.
6. Holdout untouched (blocked until the owner opens it; no registration
   hash created).

## Artifacts

- Code: `part3m/p3m_lib.py` (TTC pair extraction + 3-view accounting +
  event attribution), `part3m/diag_p3m.py`, `part3m/fit_p3m.py`,
  `part3m/calibrate_p3m.py`, `part3m/eval_p3m.py`,
  `part3m/aggregate_p3m.py`, `part3m/make_ledger_p3m.py`,
  `part3m/make_report_p3m.py`; engine changes in `part3/engine.py`
  (run(n_ticks), vtype_map instrumentation, end_force_s + v_merge_blend
  knobs — ALL default-neutral, pinned by tests against a committed part 3l
  raw row).
- Tests: `part3m/tests/` (23 tests) + pre-existing `part3/tests/` (25) and
  `part3l/tests/` (28) — all passing.
- Results: `part3m/results/` (diag_p3m.json, fit_p3m.json,
  calibration_p3m.json + calibration_p3m.FROZEN.sha256,
  gate_v2_p3m.json, stdout_*.txt, raw/raw_p3l_metrics.jsonl [verbatim part 3l
  copy], raw/raw_calibration_p3m.jsonl, raw/raw_metrics_p3m.jsonl) and
  `LEDGER_DRAFT_P3M.json`.

## TASK A/B diagnosis stdout (diag_p3m.py -- verbatim)

```text
{diag}
```

## TASK B fit stdout (fit_p3m.py -- verbatim)

```text
{fit}
```

## TASK C calibration stdout (calibrate_p3m.py -- verbatim)

```text
{calib}
```

## TASK D eval stdout (eval_p3m.py -- verbatim)

```text
{evalp}
```

## Aggregation + gate stdout (aggregate_p3m.py -- verbatim)

```text
{agg}
```

*(This report is generated by make_report_p3m.py; all numbers are pasted from
the tee'd stdout files and the results JSONs above.  Holdout not accessed;
ZERO simulation.)*
"""
    (HERE / "REPORT_P3M.md").write_text(report, encoding="utf-8")
    print(f"wrote {HERE / 'REPORT_P3M.md'} "
          f"({len(report.splitlines())} lines)")


if __name__ == "__main__":
    main()
