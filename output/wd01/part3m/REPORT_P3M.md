# WD-01 Part 3M — TTC<3 s exposure bias fix: per-side diagnosis, TRAIN refit with an added TTC<3 target (end-force + post-merge knobs, extended b grid), recalibrated spread frozen before validation, gate v2 re-run

****Holdout NOT accessed.**  No holdout root was loaded (the holdout block inside
`root_arrays` stays authoritative); no registration hash was created or passed
anywhere in part 3m.**

ZERO simulation: no SUMO/TraCI import anywhere; the only dynamics is the part-3
rollout engine (pure torch).  TRAIN fit roots for fitting, the TRAIN-internal
calibration split for spread, VALIDATION for reporting only.  Branch
zcode/wd01-p3m.  Data root read-only: WD01_DATA_ROOT=D:/rd-wt/wd01/output/wd01.

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
b_accept grid EXTENDED UP to {2.0, 3.0, 4.0, 5.0} (brief; the 1.5 low end
dropped — p3l measured it an order worse and the fix direction needs MORE
ramp discharge, not less); two new default-neutral engine knobs fitted:
`end_force_s` in {0, 150} (within 150 m of the accel-lane end, M2
acceptance relaxes to tau<=0.5 / b>=6.0 — SUMO strategic urgency: rolling
merges instead of the stop-and-go barrier queue) and `v_merge_blend` in
{0, 0.5, 1} (post-merge speed blend toward the target-lane local mean);
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
H=30 s (K=60 steps of 0.5 s), t0 grid every 60 s (120, 180, 240, 300, 360, 420, 480); train-diagnosis roots = FIT roots ['cv17', 'cv19_4107', 'cv19_4108', 'cv20_7001m2000']; validation roots ['cv20_7005m2000', 'cv20_7006m2000', 'cv30_7305m2700', 'seed_7535m2400'] (report only); holdout blocked
member configs (p3l L5 reference members):
  member 0: seed=1 lc_scale=0.00075 merge_params={'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.0} dawdle_scale=1.0
  member 1: seed=18 lc_scale=0.003 merge_params={'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.25} dawdle_scale=1.25
  member 2: seed=35 lc_scale=0.0005 merge_params={'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.25} dawdle_scale=1.5
  member 3: seed=52 lc_scale=0.0015 merge_params={'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.0} dawdle_scale=0.8
  member 4: seed=69 lc_scale=0.001 merge_params={'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.25} dawdle_scale=1.5
[   0.0s] == train: 4 roots x 7 t0 x 5 members ...
[  36.1s]   train member 0: 28 windows done ({'n_windows': 28, 'pred_all': 7.5, 'pred_matched': 7.5, 'truth': 2.732, 'bias': 4.768, 'composition': 0.0, 'behaviour': 4.768})
[  72.8s]   train member 1: 56 windows done ({'n_windows': 28, 'pred_all': 8.929, 'pred_matched': 8.929, 'truth': 2.732, 'bias': 6.196, 'composition': 0.0, 'behaviour': 6.196})
[ 106.5s]   train member 2: 84 windows done ({'n_windows': 28, 'pred_all': 9.446, 'pred_matched': 9.446, 'truth': 2.732, 'bias': 6.714, 'composition': 0.0, 'behaviour': 6.714})
[ 139.7s]   train member 3: 112 windows done ({'n_windows': 28, 'pred_all': 8.339, 'pred_matched': 8.339, 'truth': 2.732, 'bias': 5.607, 'composition': 0.0, 'behaviour': 5.607})
[ 173.2s]   train member 4: 140 windows done ({'n_windows': 28, 'pred_all': 8.232, 'pred_matched': 8.232, 'truth': 2.732, 'bias': 5.5, 'composition': 0.0, 'behaviour': 5.5})

==== TRAIN pooled over 140 windows ({'n_windows': 140, 'pred_all': 8.489, 'pred_matched': 8.489, 'truth': 2.732, 'bias': 5.757, 'composition': 0.0, 'behaviour': 5.757}) ====

  -- ev (all members) (excess = pa - tr; veh s pooled) --
  key                          excess     pa     pm     tr    n_pa  n_pm  n_tr
  ('pure_cf',)                   660.0 1042.5 1042.5  382.5   2085  2085   765
  ('post_merge',)                122.5  122.5  122.5    0.0    245   245     0
  ('post_lc',)                    23.5   23.5   23.5    0.0     47    47     0

  -- ev_lane (all members) (excess = pa - tr; veh s pooled) --
  key                          excess     pa     pm     tr    n_pa  n_pm  n_tr
  ('pure_cf', 2)                 418.5  628.5  628.5  210.0   1257  1257   420
  ('pure_cf', 1)                 240.5  413.0  413.0  172.5    826   826   345
  ('post_merge', 1)              122.0  122.0  122.0    0.0    244   244     0
  ('post_lc', 0)                  15.5   15.5   15.5    0.0     31    31     0
  ('post_lc', 1)                   8.0    8.0    8.0    0.0     16    16     0
  ('pure_cf', 0)                   1.0    1.0    1.0    0.0      2     2     0
  ('post_merge', 0)                0.5    0.5    0.5    0.0      1     1     0

  -- seg (all members) (excess = pa - tr; veh s pooled) --
  key                          excess     pa     pm     tr    n_pa  n_pm  n_tr
  ('downstream',)                887.5 1082.5 1082.5  195.0   2165  2165   390
  ('upstream',)                    0.0    2.5    2.5    2.5      5     5     5
  ('rsu_window',)                -28.5   29.0   29.0   57.5     58    58   115
  ('merge_edge',)                -53.0   74.5   74.5  127.5    149   149   255

  -- lane (all members) (excess = pa - tr; veh s pooled) --
  key                          excess     pa     pm     tr    n_pa  n_pm  n_tr
  (2,)                           418.5  628.5  628.5  210.0   1257  1257   420
  (1,)                           370.5  543.0  543.0  172.5   1086  1086   345
  (0,)                            17.0   17.0   17.0    0.0     34    34     0

  -- who (all members) (excess = pa - tr; veh s pooled) --
  key                          excess     pa     pm     tr    n_pa  n_pm  n_tr
  ('pure_cf', '-')               660.0 1042.5 1042.5  382.5   2085  2085   765
  ('post_merge', 'foll')          50.5   50.5   50.5    0.0    101   101     0
  ('post_merge', 'lead')          47.0   47.0   47.0    0.0     94    94     0
  ('post_merge', 'both')          25.0   25.0   25.0    0.0     50    50     0
  ('post_lc', '-')                23.5   23.5   23.5    0.0     47    47     0

  -- ttc (all members) (excess = pa - tr; veh s pooled) --
  key                          excess     pa     pm     tr    n_pa  n_pm  n_tr
  ('[2,3)',)                     563.0  943.0  943.0  380.0   1886  1886   760
  ('[1.5,2)',)                   202.5  202.5  202.5    0.0    405   405     0
  ('[1,1.5)',)                    31.5   31.5   31.5    0.0     63    63     0
  ('[0,1)',)                       9.0   11.5   11.5    2.5     23    23     5

  -- vt (all members) (excess = pa - tr; veh s pooled) --
  key                          excess     pa     pm     tr    n_pa  n_pm  n_tr
  (2,)                           379.5  387.0  387.0    7.5    774   774    15
  (1,)                           343.5  698.5  698.5  355.0   1397  1397   710
  (0,)                            83.0  103.0  103.0   20.0    206   206    40

  -- dawdle (all members) (excess = pa - tr; veh s pooled) --
  key                          excess     pa     pm     tr    n_pa  n_pm  n_tr
  (1.5,)                         342.0  495.0  495.0  153.0    990   990   306
  (1.25,)                        173.5  250.0  250.0   76.5    500   500   153
  (0.8,)                         157.0  233.5  233.5   76.5    467   467   153
  (1.0,)                         133.5  210.0  210.0   76.5    420   420   153

  ==== member 0 (dawdle=1.0 lc_scale=0.00075 p_coop=0.0) -- event + lane tables ====
   member 0 ev: ('pure_cf',)=+109.5(pa 186.0/tr 76.5); ('post_merge',)=+18.5(pa 18.5/tr 0.0); ('post_lc',)=+5.5(pa 5.5/tr 0.0)
   member 0 ev_lane: ('pure_cf', 2)=+115.0(pa 157.0/tr 42.0); ('post_merge', 1)=+18.5(pa 18.5/tr 0.0); ('post_lc', 0)=+3.5(pa 3.5/tr 0.0); ('post_lc', 1)=+2.0(pa 2.0/tr 0.0)

  ==== member 1 (dawdle=1.25 lc_scale=0.003 p_coop=0.25) -- event + lane tables ====
   member 1 ev: ('pure_cf',)=+129.0(pa 205.5/tr 76.5); ('post_merge',)=+36.0(pa 36.0/tr 0.0); ('post_lc',)=+8.5(pa 8.5/tr 0.0)
   member 1 ev_lane: ('pure_cf', 1)=+74.5(pa 109.0/tr 34.5); ('pure_cf', 2)=+53.5(pa 95.5/tr 42.0); ('post_merge', 1)=+35.5(pa 35.5/tr 0.0); ('post_lc', 0)=+6.5(pa 6.5/tr 0.0)

  ==== member 2 (dawdle=1.5 lc_scale=0.0005 p_coop=0.25) -- event + lane tables ====
   member 2 ev: ('pure_cf',)=+153.5(pa 230.0/tr 76.5); ('post_merge',)=+30.0(pa 30.0/tr 0.0); ('post_lc',)=+4.5(pa 4.5/tr 0.0)
   member 2 ev_lane: ('pure_cf', 1)=+103.0(pa 137.5/tr 34.5); ('pure_cf', 2)=+50.5(pa 92.5/tr 42.0); ('post_merge', 1)=+30.0(pa 30.0/tr 0.0); ('post_lc', 0)=+4.5(pa 4.5/tr 0.0)

  ==== member 3 (dawdle=0.8 lc_scale=0.0015 p_coop=0.0) -- event + lane tables ====
   member 3 ev: ('pure_cf',)=+145.5(pa 222.0/tr 76.5); ('post_merge',)=+9.5(pa 9.5/tr 0.0); ('post_lc',)=+2.0(pa 2.0/tr 0.0)
   member 3 ev_lane: ('pure_cf', 2)=+170.5(pa 212.5/tr 42.0); ('post_merge', 1)=+9.5(pa 9.5/tr 0.0); ('post_lc', 0)=+1.0(pa 1.0/tr 0.0); ('post_lc', 1)=+1.0(pa 1.0/tr 0.0)

  ==== member 4 (dawdle=1.5 lc_scale=0.001 p_coop=0.25) -- event + lane tables ====
   member 4 ev: ('pure_cf',)=+122.5(pa 199.0/tr 76.5); ('post_merge',)=+28.5(pa 28.5/tr 0.0); ('post_lc',)=+3.0(pa 3.0/tr 0.0)
   member 4 ev_lane: ('pure_cf', 1)=+93.5(pa 128.0/tr 34.5); ('pure_cf', 2)=+29.0(pa 71.0/tr 42.0); ('post_merge', 1)=+28.5(pa 28.5/tr 0.0); ('post_lc', 1)=+3.0(pa 3.0/tr 0.0)
[ 173.2s] == validation: 4 roots x 7 t0 x 5 members ...
[ 211.8s]   validation member 0: 28 windows done ({'n_windows': 28, 'pred_all': 14.054, 'pred_matched': 14.054, 'truth': 7.304, 'bias': 6.75, 'composition': 0.0, 'behaviour': 6.75})
[ 249.3s]   validation member 1: 56 windows done ({'n_windows': 28, 'pred_all': 14.964, 'pred_matched': 14.964, 'truth': 7.339, 'bias': 7.625, 'composition': 0.0, 'behaviour': 7.625})
[ 285.5s]   validation member 2: 84 windows done ({'n_windows': 28, 'pred_all': 14.786, 'pred_matched': 14.786, 'truth': 7.304, 'bias': 7.482, 'composition': 0.0, 'behaviour': 7.482})
[ 324.0s]   validation member 3: 112 windows done ({'n_windows': 28, 'pred_all': 14.286, 'pred_matched': 14.286, 'truth': 7.321, 'bias': 6.964, 'composition': 0.0, 'behaviour': 6.964})
[ 363.5s]   validation member 4: 140 windows done ({'n_windows': 28, 'pred_all': 13.143, 'pred_matched': 13.143, 'truth': 7.321, 'bias': 5.821, 'composition': 0.0, 'behaviour': 5.821})

==== VALIDATION pooled over 140 windows ({'n_windows': 140, 'pred_all': 14.246, 'pred_matched': 14.246, 'truth': 7.318, 'bias': 6.929, 'composition': 0.0, 'behaviour': 6.929}) ====

  -- ev (all members) (excess = pa - tr; veh s pooled) --
  key                          excess     pa     pm     tr    n_pa  n_pm  n_tr
  ('pure_cf',)                   767.5 1784.5 1784.5 1017.0   3569  3569  2034
  ('post_merge',)                134.0  141.5  141.5    7.5    283   283    15
  ('post_lc',)                    68.5   68.5   68.5    0.0    137   137     0

  -- ev_lane (all members) (excess = pa - tr; veh s pooled) --
  key                          excess     pa     pm     tr    n_pa  n_pm  n_tr
  ('pure_cf', 2)                 440.5  945.0  945.0  504.5   1890  1890  1009
  ('pure_cf', 1)                 324.5  832.0  832.0  507.5   1664  1664  1015
  ('post_merge', 1)              130.5  138.0  138.0    7.5    276   276    15
  ('post_lc', 0)                  43.0   43.0   43.0    0.0     86    86     0
  ('post_lc', 1)                  25.5   25.5   25.5    0.0     51    51     0
  ('post_merge', 0)                3.5    3.5    3.5    0.0      7     7     0
  ('pure_cf', 0)                   2.5    7.5    7.5    5.0     15    15    10

  -- seg (all members) (excess = pa - tr; veh s pooled) --
  key                          excess     pa     pm     tr    n_pa  n_pm  n_tr
  ('downstream',)               1284.0 1756.0 1756.0  472.0   3512  3512   944
  ('upstream',)                    3.5    3.5    3.5    0.0      7     7     0
  ('rsu_window',)                -84.0   83.5   83.5  167.5    167   167   335
  ('merge_edge',)               -233.5  151.5  151.5  385.0    303   303   770

  -- lane (all members) (excess = pa - tr; veh s pooled) --
  key                          excess     pa     pm     tr    n_pa  n_pm  n_tr
  (1,)                           480.5  995.5  995.5  515.0   1991  1991  1030
  (2,)                           440.5  945.0  945.0  504.5   1890  1890  1009
  (0,)                            49.0   54.0   54.0    5.0    108   108    10

  -- who (all members) (excess = pa - tr; veh s pooled) --
  key                          excess     pa     pm     tr    n_pa  n_pm  n_tr
  ('pure_cf', '-')               767.5 1784.5 1784.5 1017.0   3569  3569  2034
  ('post_lc', '-')                68.5   68.5   68.5    0.0    137   137     0
  ('post_merge', 'lead')          52.0   59.5   59.5    7.5    119   119    15
  ('post_merge', 'both')          43.5   43.5   43.5    0.0     87    87     0
  ('post_merge', 'foll')          38.5   38.5   38.5    0.0     77    77     0

  -- ttc (all members) (excess = pa - tr; veh s pooled) --
  key                          excess     pa     pm     tr    n_pa  n_pm  n_tr
  ('[2,3)',)                     558.5 1580.5 1580.5 1022.0   3161  3161  2044
  ('[1.5,2)',)                   324.5  327.0  327.0    2.5    654   654     5
  ('[1,1.5)',)                    59.0   59.0   59.0    0.0    118   118     0
  ('[0,1)',)                      28.0   28.0   28.0    0.0     56    56     0

  -- vt (all members) (excess = pa - tr; veh s pooled) --
  key                          excess     pa     pm     tr    n_pa  n_pm  n_tr
  (1,)                           503.5 1323.0 1323.0  819.5   2646  2646  1639
  (2,)                           465.0  520.0  520.0   55.0   1040  1040   110
  (0,)                             1.5  151.5  151.5  150.0    303   303   300

  -- dawdle (all members) (excess = pa - tr; veh s pooled) --
  key                          excess     pa     pm     tr    n_pa  n_pm  n_tr
  (1.5,)                         372.5  782.0  782.0  409.5   1564  1564   819
  (1.25,)                        213.5  419.0  419.0  205.5    838   838   411
  (0.8,)                         195.0  400.0  400.0  205.0    800   800   410
  (1.0,)                         189.0  393.5  393.5  204.5    787   787   409

  ==== member 0 (dawdle=1.0 lc_scale=0.00075 p_coop=0.0) -- event + lane tables ====
   member 0 ev: ('pure_cf',)=+174.5(pa 377.5/tr 203.0); ('post_merge',)=+7.5(pa 9.0/tr 1.5); ('post_lc',)=+7.0(pa 7.0/tr 0.0)
   member 0 ev_lane: ('pure_cf', 2)=+186.0(pa 286.5/tr 100.5); ('post_merge', 1)=+7.5(pa 9.0/tr 1.5); ('post_lc', 1)=+3.5(pa 3.5/tr 0.0); ('post_lc', 0)=+3.5(pa 3.5/tr 0.0)

  ==== member 1 (dawdle=1.25 lc_scale=0.003 p_coop=0.25) -- event + lane tables ====
   member 1 ev: ('pure_cf',)=+150.0(pa 354.0/tr 204.0); ('post_merge',)=+36.0(pa 37.5/tr 1.5); ('post_lc',)=+27.5(pa 27.5/tr 0.0)
   member 1 ev_lane: ('pure_cf', 1)=+132.5(pa 234.0/tr 101.5); ('post_merge', 1)=+34.0(pa 35.5/tr 1.5); ('post_lc', 1)=+14.5(pa 14.5/tr 0.0); ('pure_cf', 2)=+14.0(pa 115.5/tr 101.5)

  ==== member 2 (dawdle=1.5 lc_scale=0.0005 p_coop=0.25) -- event + lane tables ====
   member 2 ev: ('pure_cf',)=+172.5(pa 375.5/tr 203.0); ('post_merge',)=+32.0(pa 33.5/tr 1.5); ('post_lc',)=+5.0(pa 5.0/tr 0.0)
   member 2 ev_lane: ('pure_cf', 1)=+144.0(pa 245.5/tr 101.5); ('post_merge', 1)=+30.5(pa 32.0/tr 1.5); ('pure_cf', 2)=+27.5(pa 128.0/tr 100.5); ('post_lc', 0)=+4.0(pa 4.0/tr 0.0)

  ==== member 3 (dawdle=0.8 lc_scale=0.0015 p_coop=0.0) -- event + lane tables ====
   member 3 ev: ('pure_cf',)=+166.5(pa 370.0/tr 203.5); ('post_merge',)=+15.5(pa 17.0/tr 1.5); ('post_lc',)=+13.0(pa 13.0/tr 0.0)
   member 3 ev_lane: ('pure_cf', 2)=+220.5(pa 321.5/tr 101.0); ('post_merge', 1)=+15.5(pa 17.0/tr 1.5); ('post_lc', 0)=+8.0(pa 8.0/tr 0.0); ('post_lc', 1)=+5.0(pa 5.0/tr 0.0)

  ==== member 4 (dawdle=1.5 lc_scale=0.001 p_coop=0.25) -- event + lane tables ====
   member 4 ev: ('pure_cf',)=+104.0(pa 307.5/tr 203.5); ('post_merge',)=+43.0(pa 44.5/tr 1.5); ('post_lc',)=+16.0(pa 16.0/tr 0.0)
   member 4 ev_lane: ('pure_cf', 1)=+112.0(pa 213.5/tr 101.5); ('post_merge', 1)=+43.0(pa 44.5/tr 1.5); ('post_lc', 0)=+14.5(pa 14.5/tr 0.0); ('post_lc', 1)=+1.5(pa 1.5/tr 0.0)
[ 363.6s] wrote D:\rd-wt\wd01k\output\wd01\part3m\results\diag_p3m.json

DONE part 3m TASK A diagnosis.
```

## TASK B fit stdout (fit_p3m.py -- verbatim)

```text
priority below-normal: False
=== WD-01 part 3m TASK B: idm_merge refit, TTC<3 target + fix knobs (TRAIN only) ===
ZERO simulation; holdout blocked in root_arrays (never passed).
device=cpu torch_threads=2
train fit roots: ['cv17', 'cv19_4107', 'cv19_4108', 'cv20_7001m2000']; eval windows t0 (120.0, 300.0), H=60 s (K=120 steps of 0.5 s); calibration roots ['cv20_7002m2000', 'cv20_7003m2000', 'cv20_7004m2000', 'cv20_7007m2700'] NOT used here; validation roots ['cv20_7005m2000', 'cv20_7006m2000', 'cv30_7305m2700', 'seed_7535m2400'] NOT used here
grids: tau_accept (1.9, 2.4, 3.0, 3.6) x b_accept (2.0, 3.0, 4.0, 5.0) (extended up per brief; 1.5 dropped, p3l-measured cliff); end_force_s (0.0, 150.0); v_merge_blend (0.0, 0.5, 1.0); lc_scale (0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.003); p_coop (0.0, 0.25); dawdle (0.8, 1.0, 1.25, 1.5); member softmax T=0.15
stage-2 loss adds the TTC<3 exposure target ((p-t)/max(t,5))^2 per window
M2 follower slot: FIXED nearest-behind (part 3l TASK A)

TRAIN truth TTC<3 exposure per eval window (the added target):
  cv17             t0=120: 4.0 veh s
  cv17             t0=300: 25.0 veh s
  cv19_4107        t0=120: 3.0 veh s
  cv19_4107        t0=300: 6.0 veh s
  cv19_4108        t0=120: 2.5 veh s
  cv19_4108        t0=300: 3.5 veh s
  cv20_7001m2000   t0=120: 2.0 veh s
  cv20_7001m2000   t0=300: 5.0 veh s
[   3.0s] BEFORE: rolling the p3l L5 member 0 config ...

BEFORE (p3l L5 member 0): stage1 loss=1.1095 {'occ': 1.0299149542658987, 'prof': 0.288453904302041, 'merge': 0.043551570485467775} | stage2 loss=7.3618 {'mr': 0.27373282658194953, 'delay': 0.5927296270059751, 'lc30': 0.28534786788337924, 'lc60': 0.3824319662495947, 'ttc3': 6.018750000000001}

  -- BEFORE p3l L5 member 0 (per eval window) --
  root             t0   occ_p occ_t | accel v by d-bin -50..300 pred | truth | merge_d p/t   mr_spd p/t    delay_p  delay_t  lc30 p/t lc60 p/t ttc3_p/t
  cv17             120    456   142 | 13.0 18.5 19.2 19.4 19.5 19.4 19.0 18.3 |  7.1 17.8 19.0 19.4 19.4 19.0 18.5 17.1 |  -71.0/ -58.4   20.1/ 17.1     974.9    743.2    8/ 11  11/ 20  26.0/ 4.0
  cv17             300     92   225 | 14.1 17.8 19.2 19.9 20.1 20.1 20.2 20.8 |  3.2 17.9 20.4 21.2 21.6 21.7 21.9 21.6 |  -58.4/ -56.2    4.9/  2.7    1829.6   2078.1   11/ 12  13/ 21  12.5/25.0
  cv19_4107        120     78    86 | 14.3 18.0 19.1 19.5 19.6 19.6 19.0 17.7 | 12.8 18.3 19.8 20.3 20.5 20.3 19.8 17.1 |  -16.2/ -18.9   26.3/ 24.6     464.3    419.4    5/ 24   8/ 38   5.5/ 3.0
  cv19_4107        300    221   432 | 14.8 18.4 19.5 19.9 20.0 19.8 19.4 17.2 |  3.5 17.1 19.9 20.8 20.9 20.7 20.2 17.9 |  -58.9/ -40.9   18.6/ 21.0     848.4    746.7    6/  5  10/ 22  24.0/ 6.0
  cv19_4108        120    291   115 | 13.6 17.8 18.7 18.9 18.8 18.6 18.6 18.0 |  8.5 18.2 19.1 19.4 19.4 19.0 18.5 17.1 |  -72.5/ -50.2   23.7/ 19.4     556.9    454.5    3/ 18   3/ 39  15.5/ 2.5
  cv19_4108        300    125   100 | 16.0 18.6 19.6 20.1 20.6 20.7 20.9 20.9 | 15.4 19.6 20.8 21.3 21.6 21.8 21.9 21.6 |  -47.2/ -52.3   22.3/ 23.6     577.7    407.2    4/ 16  11/ 30   2.5/ 3.5
  cv20_7001m2000   120    256   161 | 14.2 17.8 19.1 19.6 20.0 20.0 20.4 20.4 | 10.2 18.3 19.8 20.5 20.8 20.8 21.1 21.2 |  -59.8/ -31.6   23.6/ 23.7     653.9    529.6    3/  6   7/ 16  18.0/ 2.0
  cv20_7001m2000   300    107   108 | 15.4 18.1 19.2 19.6 19.8 19.9 20.0 19.8 | 14.0 18.3 19.4 19.8 20.0 20.1 20.1 19.4 |  -39.6/ -51.6   12.0/  8.4     808.6    891.6    8/ 15  18/ 34  12.5/ 5.0
[  21.6s] STAGE 1: 16 acceptance combos (force=0 blend=0, lc at reference 0.002) ...
[  39.8s] stage1 {'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0, 'end_force_s': 0.0, 'v_merge_blend': 0.0} -> s1=4.3349 s2=18.2141 (ttc3 term 16.5425)
[  57.7s] stage1 {'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.0, 'end_force_s': 0.0, 'v_merge_blend': 0.0} -> s1=0.9386 s2=6.3246 (ttc3 term 5.2076)
[  77.0s] stage1 {'tau_accept': 1.9, 'b_accept': 4.0, 'p_coop_scale': 0.0, 'end_force_s': 0.0, 'v_merge_blend': 0.0} -> s1=0.2646 s2=4.5914 (ttc3 term 3.6375)
[  94.3s] stage1 {'tau_accept': 1.9, 'b_accept': 5.0, 'p_coop_scale': 0.0, 'end_force_s': 0.0, 'v_merge_blend': 0.0} -> s1=0.3093 s2=4.0559 (ttc3 term 3.2125)
[ 112.5s] stage1 {'tau_accept': 2.4, 'b_accept': 2.0, 'p_coop_scale': 0.0, 'end_force_s': 0.0, 'v_merge_blend': 0.0} -> s1=4.7864 s2=23.3317 (ttc3 term 21.5800)
[ 132.6s] stage1 {'tau_accept': 2.4, 'b_accept': 3.0, 'p_coop_scale': 0.0, 'end_force_s': 0.0, 'v_merge_blend': 0.0} -> s1=0.9087 s2=6.6826 (ttc3 term 5.5917)
[ 151.0s] stage1 {'tau_accept': 2.4, 'b_accept': 4.0, 'p_coop_scale': 0.0, 'end_force_s': 0.0, 'v_merge_blend': 0.0} -> s1=0.2262 s2=4.9525 (ttc3 term 4.0040)
[ 170.0s] stage1 {'tau_accept': 2.4, 'b_accept': 5.0, 'p_coop_scale': 0.0, 'end_force_s': 0.0, 'v_merge_blend': 0.0} -> s1=0.2214 s2=5.7551 (ttc3 term 4.8902)
[ 190.2s] stage1 {'tau_accept': 3.0, 'b_accept': 2.0, 'p_coop_scale': 0.0, 'end_force_s': 0.0, 'v_merge_blend': 0.0} -> s1=4.8166 s2=23.9196 (ttc3 term 22.1388)
[ 208.5s] stage1 {'tau_accept': 3.0, 'b_accept': 3.0, 'p_coop_scale': 0.0, 'end_force_s': 0.0, 'v_merge_blend': 0.0} -> s1=1.1938 s2=9.8342 (ttc3 term 8.6003)
[ 226.7s] stage1 {'tau_accept': 3.0, 'b_accept': 4.0, 'p_coop_scale': 0.0, 'end_force_s': 0.0, 'v_merge_blend': 0.0} -> s1=0.2648 s2=6.0030 (ttc3 term 4.9918)
[ 244.8s] stage1 {'tau_accept': 3.0, 'b_accept': 5.0, 'p_coop_scale': 0.0, 'end_force_s': 0.0, 'v_merge_blend': 0.0} -> s1=0.2301 s2=9.3364 (ttc3 term 8.4339)
[ 262.6s] stage1 {'tau_accept': 3.6, 'b_accept': 2.0, 'p_coop_scale': 0.0, 'end_force_s': 0.0, 'v_merge_blend': 0.0} -> s1=5.5242 s2=24.7627 (ttc3 term 22.8687)
[ 280.2s] stage1 {'tau_accept': 3.6, 'b_accept': 3.0, 'p_coop_scale': 0.0, 'end_force_s': 0.0, 'v_merge_blend': 0.0} -> s1=1.5654 s2=11.4765 (ttc3 term 10.2457)
[ 297.6s] stage1 {'tau_accept': 3.6, 'b_accept': 4.0, 'p_coop_scale': 0.0, 'end_force_s': 0.0, 'v_merge_blend': 0.0} -> s1=0.2800 s2=6.7698 (ttc3 term 5.7797)
[ 317.9s] stage1 {'tau_accept': 3.6, 'b_accept': 5.0, 'p_coop_scale': 0.0, 'end_force_s': 0.0, 'v_merge_blend': 0.0} -> s1=0.2254 s2=4.4297 (ttc3 term 3.4941)

STAGE 1 winner: {'tau_accept': 1.9, 'b_accept': 5.0, 'p_coop_scale': 0.0, 'end_force_s': 0.0, 'v_merge_blend': 0.0} joint=4.3653 (s1 0.3093 s2 4.0559)
[ 317.9s] STAGE 2: 6 force x blend combos at {'tau_accept': 1.9, 'b_accept': 5.0, 'p_coop_scale': 0.0, 'end_force_s': 0.0, 'v_merge_blend': 0.0} ...
[ 336.5s] stage2 force=0.0 blend=0.0 -> s1=0.3093 s2=4.0559 (ttc3 term 3.2125)
[ 355.1s] stage2 force=0.0 blend=0.5 -> s1=0.3000 s2=1.7453 (ttc3 term 0.4281)
[ 374.1s] stage2 force=0.0 blend=1.0 -> s1=0.3135 s2=3.4782 (ttc3 term 1.7760)
[ 392.2s] stage2 force=150.0 blend=0.0 -> s1=0.6272 s2=1.2314 (ttc3 term 0.3347)
[ 409.9s] stage2 force=150.0 blend=0.5 -> s1=0.5656 s2=1.4922 (ttc3 term 0.3232)
[ 428.5s] stage2 force=150.0 blend=1.0 -> s1=0.5485 s2=1.8282 (ttc3 term 0.3173)

STAGE 2 winner: force=150.0 blend=0.0 joint=1.8586
[ 428.5s] STAGE 3: 12 lc x p_coop combos at force=150.0 blend=0.0 ...
[ 446.5s] stage3 lc=0.0005 p_coop=0.0 -> s1=0.6929 s2=1.8300 (ttc3 term 0.8567)
[ 464.2s] stage3 lc=0.0005 p_coop=0.25 -> s1=0.7895 s2=1.5673 (ttc3 term 0.5277)
[ 482.9s] stage3 lc=0.00075 p_coop=0.0 -> s1=0.6782 s2=1.7925 (ttc3 term 0.8717)
[ 501.8s] stage3 lc=0.00075 p_coop=0.25 -> s1=0.7816 s2=1.5044 (ttc3 term 0.5315)
[ 518.3s] stage3 lc=0.001 p_coop=0.0 -> s1=0.6946 s2=1.7831 (ttc3 term 0.8717)
[ 535.6s] stage3 lc=0.001 p_coop=0.25 -> s1=0.7920 s2=1.4987 (ttc3 term 0.5325)
[ 552.6s] stage3 lc=0.0015 p_coop=0.0 -> s1=0.6748 s2=1.3345 (ttc3 term 0.4547)
[ 569.0s] stage3 lc=0.0015 p_coop=0.25 -> s1=0.7879 s2=1.4684 (ttc3 term 0.5253)
[ 587.3s] stage3 lc=0.002 p_coop=0.0 -> s1=0.6272 s2=1.2314 (ttc3 term 0.3347)
[ 606.4s] stage3 lc=0.002 p_coop=0.25 -> s1=0.7899 s2=1.2586 (ttc3 term 0.3491)
[ 622.8s] stage3 lc=0.003 p_coop=0.0 -> s1=0.7479 s2=1.6635 (ttc3 term 0.3263)
[ 638.3s] stage3 lc=0.003 p_coop=0.25 -> s1=0.7337 s2=1.6726 (ttc3 term 0.2755)

STAGE 3 winner: lc_scale=0.002 p_coop=0.0 joint=1.8586
[ 638.3s] STAGE 4: 16 acceptance combos at lc=0.002 p_coop=0.0 ...
[ 654.0s] stage4 {'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0, 'end_force_s': 150.0, 'v_merge_blend': 0.0} -> s1=0.6272 s2=1.2314 (ttc3 term 0.3347)
[ 671.4s] stage4 {'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.0, 'end_force_s': 150.0, 'v_merge_blend': 0.0} -> s1=0.6272 s2=1.2314 (ttc3 term 0.3347)
[ 689.6s] stage4 {'tau_accept': 1.9, 'b_accept': 4.0, 'p_coop_scale': 0.0, 'end_force_s': 150.0, 'v_merge_blend': 0.0} -> s1=0.6272 s2=1.2314 (ttc3 term 0.3347)
[ 707.7s] stage4 {'tau_accept': 1.9, 'b_accept': 5.0, 'p_coop_scale': 0.0, 'end_force_s': 150.0, 'v_merge_blend': 0.0} -> s1=0.6272 s2=1.2314 (ttc3 term 0.3347)
[ 726.5s] stage4 {'tau_accept': 2.4, 'b_accept': 2.0, 'p_coop_scale': 0.0, 'end_force_s': 150.0, 'v_merge_blend': 0.0} -> s1=0.6272 s2=1.2314 (ttc3 term 0.3347)
[ 745.8s] stage4 {'tau_accept': 2.4, 'b_accept': 3.0, 'p_coop_scale': 0.0, 'end_force_s': 150.0, 'v_merge_blend': 0.0} -> s1=0.6272 s2=1.2314 (ttc3 term 0.3347)
[ 764.5s] stage4 {'tau_accept': 2.4, 'b_accept': 4.0, 'p_coop_scale': 0.0, 'end_force_s': 150.0, 'v_merge_blend': 0.0} -> s1=0.6272 s2=1.2314 (ttc3 term 0.3347)
[ 784.0s] stage4 {'tau_accept': 2.4, 'b_accept': 5.0, 'p_coop_scale': 0.0, 'end_force_s': 150.0, 'v_merge_blend': 0.0} -> s1=0.6272 s2=1.2314 (ttc3 term 0.3347)
[ 803.6s] stage4 {'tau_accept': 3.0, 'b_accept': 2.0, 'p_coop_scale': 0.0, 'end_force_s': 150.0, 'v_merge_blend': 0.0} -> s1=0.6272 s2=1.2314 (ttc3 term 0.3347)
[ 823.0s] stage4 {'tau_accept': 3.0, 'b_accept': 3.0, 'p_coop_scale': 0.0, 'end_force_s': 150.0, 'v_merge_blend': 0.0} -> s1=0.6272 s2=1.2314 (ttc3 term 0.3347)
[ 843.4s] stage4 {'tau_accept': 3.0, 'b_accept': 4.0, 'p_coop_scale': 0.0, 'end_force_s': 150.0, 'v_merge_blend': 0.0} -> s1=0.6272 s2=1.2314 (ttc3 term 0.3347)
[ 864.7s] stage4 {'tau_accept': 3.0, 'b_accept': 5.0, 'p_coop_scale': 0.0, 'end_force_s': 150.0, 'v_merge_blend': 0.0} -> s1=0.6272 s2=1.2314 (ttc3 term 0.3347)
[ 885.7s] stage4 {'tau_accept': 3.6, 'b_accept': 2.0, 'p_coop_scale': 0.0, 'end_force_s': 150.0, 'v_merge_blend': 0.0} -> s1=0.6272 s2=1.2314 (ttc3 term 0.3347)
[ 906.8s] stage4 {'tau_accept': 3.6, 'b_accept': 3.0, 'p_coop_scale': 0.0, 'end_force_s': 150.0, 'v_merge_blend': 0.0} -> s1=0.6272 s2=1.2314 (ttc3 term 0.3347)
[ 923.3s] stage4 {'tau_accept': 3.6, 'b_accept': 4.0, 'p_coop_scale': 0.0, 'end_force_s': 150.0, 'v_merge_blend': 0.0} -> s1=0.6272 s2=1.2314 (ttc3 term 0.3347)
[ 938.8s] stage4 {'tau_accept': 3.6, 'b_accept': 5.0, 'p_coop_scale': 0.0, 'end_force_s': 150.0, 'v_merge_blend': 0.0} -> s1=0.6272 s2=1.2314 (ttc3 term 0.3347)

STAGE 4 winner: {'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0, 'end_force_s': 150.0, 'v_merge_blend': 0.0} lc_scale=0.002 joint=1.8586 (s1 0.6272 s2 1.2314)

AFTER (L6 = idm_merge refit, part 3m knobs): stage1 loss=0.6272 {'occ': 0.5475097807593584, 'prof': 0.38083060276754993, 'merge': 0.03204742926181291} | stage2 loss=1.2314 {'mr': 0.29760893582567616, 'delay': 0.30544799828119973, 'lc30': 0.22710390082963477, 'lc60': 0.13299361221575282, 'ttc3': 0.3347347222222222}

  -- AFTER L6 (idm_merge refit) (per eval window) --
  root             t0   occ_p occ_t | accel v by d-bin -50..300 pred | truth | merge_d p/t   mr_spd p/t    delay_p  delay_t  lc30 p/t lc60 p/t ttc3_p/t
  cv17             120     30   142 | 12.4 18.5 19.1 19.2 19.2 19.3 19.3 18.7 |  7.1 17.8 19.0 19.4 19.4 19.0 18.5 17.1 |  -36.1/ -58.4   19.0/ 17.1     861.1    743.2   12/ 11  26/ 20   8.0/ 4.0
  cv17             300     55   225 | 15.4 18.7 20.0 20.5 20.7 21.0 21.0 21.3 |  3.2 17.9 20.4 21.2 21.6 21.7 21.9 21.6 |  -57.5/ -56.2    5.0/  2.7    1865.8   2078.1   15/ 12  30/ 21  18.5/25.0
  cv19_4107        120     51    86 | 16.0 18.4 19.4 19.8 20.1 20.0 19.6 18.0 | 12.8 18.3 19.8 20.3 20.5 20.3 19.8 17.1 |  -15.3/ -18.9   25.5/ 24.6     450.8    419.4   10/ 24  15/ 38   0.5/ 3.0
  cv19_4107        300     39   432 | 15.2 17.9 18.9 19.5 19.6 19.5 19.0 17.0 |  3.5 17.1 19.9 20.8 20.9 20.7 20.2 17.9 |  -32.2/ -40.9   20.1/ 21.0     765.0    746.7   13/  5  21/ 22   0.5/ 6.0
  cv19_4108        120     40   115 | 15.9 18.6 19.3 19.7 19.5 19.4 19.4 18.6 |  8.5 18.2 19.1 19.4 19.4 19.0 18.5 17.1 |  -53.5/ -50.2   19.6/ 19.4     525.8    454.5   14/ 18  25/ 39   5.5/ 2.5
  cv19_4108        300     29   100 | 16.2 18.5 19.5 20.2 20.5 20.5 21.0 20.9 | 15.4 19.6 20.8 21.3 21.6 21.8 21.9 21.6 |  -38.7/ -52.3   21.4/ 23.6     559.2    407.2    7/ 16  17/ 30   3.5/ 3.5
  cv20_7001m2000   120     21   161 | 15.6 18.2 19.3 19.7 20.1 20.3 20.6 20.6 | 10.2 18.3 19.8 20.5 20.8 20.8 21.1 21.2 |  -22.2/ -31.6   25.6/ 23.7     572.3    529.6   12/  6  19/ 16   0.0/ 2.0
  cv20_7001m2000   300     31   108 | 15.9 18.2 19.3 19.6 19.7 19.9 20.0 19.9 | 14.0 18.3 19.4 19.8 20.0 20.1 20.1 19.4 |  -27.7/ -51.6   14.8/  8.4     805.3    891.6   12/ 15  25/ 34   2.0/ 5.0
[ 955.1s] MEMBERS 1..4: wider draws (bootstrap resample x softmax lines x dawdle width) ...
[1139.7s]   member 1 tau_accept line losses ['1.377', '1.377', '1.377', '1.377'] -> draw 1.9
[1139.7s]   member 1 b_accept line losses ['1.377', '1.377', '1.377', '1.377'] -> draw 5.0
[1139.7s]   member 1 p_coop_scale line losses ['1.377', '1.424'] -> draw 0.25
[1139.7s]   member 1 end_force_s line losses ['25.921', '1.377'] -> draw 150.0
[1139.7s]   member 1 v_merge_blend line losses ['1.377', '1.613', '2.157'] -> draw 0.5
[1139.7s]   member 1 lc_scale line losses ['3.126', '3.036', '3.031', '1.884', '1.377', '1.482'] -> draw 0.003
[1139.7s] member 1: {'seed': 18, 'lc_scale': 0.003, 'merge_params': {'tau_accept': 1.9, 'b_accept': 5.0, 'p_coop_scale': 0.25, 'end_force_s': 150.0, 'v_merge_blend': 0.5}, 'dawdle_scale': 1.0, 'resample': ['cv17', 'cv19_4107', 'cv19_4108']}
[1314.4s]   member 2 tau_accept line losses ['1.404', '1.404', '1.404', '1.404'] -> draw 1.9
[1314.4s]   member 2 b_accept line losses ['1.404', '1.404', '1.404', '1.404'] -> draw 3.0
[1314.4s]   member 2 p_coop_scale line losses ['1.404', '1.482'] -> draw 0.25
[1314.4s]   member 2 end_force_s line losses ['26.930', '1.404'] -> draw 150.0
[1314.4s]   member 2 v_merge_blend line losses ['1.404', '1.574', '2.041'] -> draw 0.0
[1314.4s]   member 2 lc_scale line losses ['2.656', '2.569', '2.604', '1.729', '1.404', '1.751'] -> draw 0.002
[1314.4s] member 2: {'seed': 35, 'lc_scale': 0.002, 'merge_params': {'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.25, 'end_force_s': 150.0, 'v_merge_blend': 0.0}, 'dawdle_scale': 1.5, 'resample': ['cv17', 'cv19_4107', 'cv19_4108', 'cv20_7001m2000']}
[1500.2s]   member 3 tau_accept line losses ['1.550', '1.550', '1.550', '1.550'] -> draw 2.4
[1500.2s]   member 3 b_accept line losses ['1.550', '1.550', '1.550', '1.550'] -> draw 5.0
[1500.2s]   member 3 p_coop_scale line losses ['1.550', '1.389'] -> draw 0.25
[1500.2s]   member 3 end_force_s line losses ['30.077', '1.550'] -> draw 150.0
[1500.2s]   member 3 v_merge_blend line losses ['1.550', '1.684', '2.398'] -> draw 0.5
[1500.2s]   member 3 lc_scale line losses ['2.907', '2.818', '2.772', '1.706', '1.550', '1.864'] -> draw 0.002
[1500.2s] member 3: {'seed': 52, 'lc_scale': 0.002, 'merge_params': {'tau_accept': 2.4, 'b_accept': 5.0, 'p_coop_scale': 0.25, 'end_force_s': 150.0, 'v_merge_blend': 0.5}, 'dawdle_scale': 1.25, 'resample': ['cv17', 'cv19_4108', 'cv20_7001m2000']}
[1648.8s]   member 4 tau_accept line losses ['1.228', '1.228', '1.228', '1.228'] -> draw 2.4
[1648.8s]   member 4 b_accept line losses ['1.228', '1.228', '1.228', '1.228'] -> draw 4.0
[1648.8s]   member 4 p_coop_scale line losses ['1.228', '1.317'] -> draw 0.0
[1648.8s]   member 4 end_force_s line losses ['27.542', '1.228'] -> draw 150.0
[1648.8s]   member 4 v_merge_blend line losses ['1.228', '1.562', '1.804'] -> draw 0.0
[1648.8s]   member 4 lc_scale line losses ['2.977', '2.851', '2.853', '1.733', '1.228', '1.639'] -> draw 0.002
[1648.8s] member 4: {'seed': 69, 'lc_scale': 0.002, 'merge_params': {'tau_accept': 2.4, 'b_accept': 4.0, 'p_coop_scale': 0.0, 'end_force_s': 150.0, 'v_merge_blend': 0.0}, 'dawdle_scale': 1.25, 'resample': ['cv19_4107', 'cv19_4108', 'cv20_7001m2000']}

selected member parameters (member 0 = main fit):
  member 0: seed=1 lc_scale=0.002 merge_params={'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0, 'end_force_s': 150.0, 'v_merge_blend': 0.0} dawdle_scale=1.0
  member 1: seed=18 lc_scale=0.003 merge_params={'tau_accept': 1.9, 'b_accept': 5.0, 'p_coop_scale': 0.25, 'end_force_s': 150.0, 'v_merge_blend': 0.5} dawdle_scale=1.0
  member 2: seed=35 lc_scale=0.002 merge_params={'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.25, 'end_force_s': 150.0, 'v_merge_blend': 0.0} dawdle_scale=1.5
  member 3: seed=52 lc_scale=0.002 merge_params={'tau_accept': 2.4, 'b_accept': 5.0, 'p_coop_scale': 0.25, 'end_force_s': 150.0, 'v_merge_blend': 0.5} dawdle_scale=1.25
  member 4: seed=69 lc_scale=0.002 merge_params={'tau_accept': 2.4, 'b_accept': 4.0, 'p_coop_scale': 0.0, 'end_force_s': 150.0, 'v_merge_blend': 0.0} dawdle_scale=1.25
[1648.8s] wrote D:\rd-wt\wd01k\output\wd01\part3m\results\fit_p3m.json and D:\rd-wt\wd01k\output\wd01\part3m\results\stdout_fit.txt

DONE part 3m TASK B fit.
```

## TASK C calibration stdout (calibrate_p3m.py -- verbatim)

```text
priority below-normal: False
=== WD-01 part 3m TASK C: ensemble spread recalibration (TRAIN-internal) ===
ZERO simulation; holdout blocked in root_arrays (never passed).
device=cpu torch_threads=2
calibration roots (TRAIN, held out of every fit): ['cv20_7002m2000', 'cv20_7003m2000', 'cv20_7004m2000', 'cv20_7007m2700']; t0 (120, 180, 240, 300, 360, 420, 480) -> 28 windows
fit roots (NOT rolled here): ['cv17', 'cv19_4107', 'cv19_4108', 'cv20_7001m2000']; validation ['cv20_7005m2000', 'cv20_7006m2000', 'cv30_7305m2700', 'seed_7535m2400'] NOT used here; holdout blocked
L6 member configs (from fit_p3m.json):
  member 0: seed=1 lc_scale=0.002 merge_params={'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0, 'end_force_s': 150.0, 'v_merge_blend': 0.0} dawdle_scale=1.0
  member 1: seed=18 lc_scale=0.003 merge_params={'tau_accept': 1.9, 'b_accept': 5.0, 'p_coop_scale': 0.25, 'end_force_s': 150.0, 'v_merge_blend': 0.5} dawdle_scale=1.0
  member 2: seed=35 lc_scale=0.002 merge_params={'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.25, 'end_force_s': 150.0, 'v_merge_blend': 0.0} dawdle_scale=1.5
  member 3: seed=52 lc_scale=0.002 merge_params={'tau_accept': 2.4, 'b_accept': 5.0, 'p_coop_scale': 0.25, 'end_force_s': 150.0, 'v_merge_blend': 0.5} dawdle_scale=1.25
  member 4: seed=69 lc_scale=0.002 merge_params={'tau_accept': 2.4, 'b_accept': 4.0, 'p_coop_scale': 0.0, 'end_force_s': 150.0, 'v_merge_blend': 0.0} dawdle_scale=1.25
[   0.0s] rolling L6 member 0 (seed 1, dawdle 1.0) ...
[  66.2s] rolling L6 member 1 (seed 18, dawdle 1.0) ...
[ 134.8s] rolling L6 member 2 (seed 35, dawdle 1.5) ...
[ 211.6s] rolling L6 member 3 (seed 52, dawdle 1.25) ...
[ 284.5s] rolling L6 member 4 (seed 69, dawdle 1.25) ...
[ 348.6s] wrote 280 calibration rows to D:\rd-wt\wd01k\output\wd01\part3m\results\raw\raw_calibration_p3m.jsonl

== spread calibration per (output, H) on the TRAIN calibration split ==
  delay_phys@H30: raw_cov=0.3929 s_out=2.5198 cov_mean=0.8214 crps 33.279->32.024 mean_shift=0.0
  delay_phys@H60: raw_cov=0.3929 s_out=2.2974 cov_mean=0.8214 crps 76.013->76.036 mean_shift=0.0
  ttc3@H30: raw_cov=0.5357 s_out=7.6387 cov_mean=0.8214 crps 2.74->6.138 mean_shift=0.0
  ttc3@H60: raw_cov=0.1786 s_out=14.5876 cov_mean=0.8214 crps 5.806->16.06 mean_shift=0.0
[ 349.6s] FROZEN D:\rd-wt\wd01k\output\wd01\part3m\results\calibration_p3m.json
[ 349.6s]   sha256=968d20ba26d8552014807f0fbd3a2d9d2bf0b87c41d5aa49410d4ae804b97334
[ 349.6s]   frozen_utc=2026-09-26T20:24:17.243056+00:00

DONE part 3m TASK C calibration (factors frozen before validation).
```

## TASK D eval stdout (eval_p3m.py -- verbatim)

```text
priority below-normal: False
=== WD-01 part 3m TASK D: L6 rows on VALIDATION ===
ZERO simulation; holdout blocked in root_arrays (never passed).
device=cpu torch_threads=2
FREEZE GATE ok: calibration_p3m.json sha256 968d20ba26d85520... frozen_utc=2026-09-26T20:24:17.243056+00:00
validation roots ['cv20_7005m2000', 'cv20_7006m2000', 'cv30_7305m2700', 'seed_7535m2400']; t0 every 30 s (120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450, 480) (52 windows); H (30, 60); L0/L4/L5 rows reused from part 3l (no re-roll)
L6 member configs (from fit_p3m.json):
  L6 member 0: seed=1 lc_scale=0.002 merge_params={'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0, 'end_force_s': 150.0, 'v_merge_blend': 0.0} dawdle_scale=1.0
  L6 member 1: seed=18 lc_scale=0.003 merge_params={'tau_accept': 1.9, 'b_accept': 5.0, 'p_coop_scale': 0.25, 'end_force_s': 150.0, 'v_merge_blend': 0.5} dawdle_scale=1.0
  L6 member 2: seed=35 lc_scale=0.002 merge_params={'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.25, 'end_force_s': 150.0, 'v_merge_blend': 0.0} dawdle_scale=1.5
  L6 member 3: seed=52 lc_scale=0.002 merge_params={'tau_accept': 2.4, 'b_accept': 5.0, 'p_coop_scale': 0.25, 'end_force_s': 150.0, 'v_merge_blend': 0.5} dawdle_scale=1.25
  L6 member 4: seed=69 lc_scale=0.002 merge_params={'tau_accept': 2.4, 'b_accept': 4.0, 'p_coop_scale': 0.0, 'end_force_s': 150.0, 'v_merge_blend': 0.0} dawdle_scale=1.25
[   0.0s] rolling L6 member 0 (seed 1, dawdle 1.0) ...
[ 130.2s] L6 member 0: done (104 rows total)
[ 130.2s] rolling L6 member 1 (seed 18, dawdle 1.0) ...
[ 269.4s] L6 member 1: done (208 rows total)
[ 269.4s] rolling L6 member 2 (seed 35, dawdle 1.5) ...
[ 415.0s] L6 member 2: done (312 rows total)
[ 415.0s] rolling L6 member 3 (seed 52, dawdle 1.25) ...
[ 554.0s] L6 member 3: done (416 rows total)
[ 554.0s] rolling L6 member 4 (seed 69, dawdle 1.25) ...
[ 706.9s] L6 member 4: done (520 rows total)
[ 706.9s] raw rows written to D:\rd-wt\wd01k\output\wd01\part3m\results\raw\raw_metrics_p3m.jsonl (total this run: 520)
```

## Aggregation + gate stdout (aggregate_p3m.py -- verbatim)

```text
=== WD-01 part 3m: ladder aggregation + gate verdicts ===
ZERO simulation; holdout blocked in root_arrays (never passed); validation rows only.
FREEZE GATE ok: calibration_p3m.json sha256 968d20ba26d85520... frozen_utc=2026-09-26T20:24:17.243056+00:00
loaded 2288 part 3l rows (L0/L4/L5, reused) + 520 part 3m rows (L6)
L5c spread map (part 3l calibration, unchanged): {'delay_phys': {30: 1.7411, 60: 1.9097}, 'ttc3': {30: 2.0946, 60: 2.2974}}
L6c spread map (part 3m recalibration, frozen 2026-09-26T20:24:17.243056+00:00): {'delay_phys': {30: 2.5198, 60: 2.2974}, 'ttc3': {30: 7.6387, 60: 14.5876}}
  L4@H30: n_windows=52 ttc3_10-90_cov=0.1923 (below 0.75, above 0.0577; truth=0 ties below 6, above 0) delay_phys_10-90_cov=0.0577 delay_exposure_cov=0.25 minmax=0.0962 CRPS_phys=47.26 (s_out dp/t3 = None/None)
  L5@H30: n_windows=52 ttc3_10-90_cov=0.1923 (below 0.7115, above 0.0962; truth=0 ties below 5, above 0) delay_phys_10-90_cov=0.5 delay_exposure_cov=0.3846 minmax=0.6154 CRPS_phys=42.6 (s_out dp/t3 = None/None)
  L5c@H30: n_windows=52 ttc3_10-90_cov=0.5962 (below 0.3846, above 0.0192; truth=0 ties below 3, above 0) delay_phys_10-90_cov=0.8462 delay_exposure_cov=0.3846 minmax=0.9038 CRPS_phys=46.72 (s_out dp/t3 = 1.7411/2.0946)
  L6@H30: n_windows=52 ttc3_10-90_cov=0.4038 (below 0.3269, above 0.2692; truth=0 ties below 2, above 0) delay_phys_10-90_cov=0.2885 delay_exposure_cov=0.2885 minmax=0.4423 CRPS_phys=36.4 (s_out dp/t3 = None/None)
  L6c@H30: n_windows=52 ttc3_10-90_cov=0.9231 (below 0.0, above 0.0769; truth=0 ties below 0, above 0) delay_phys_10-90_cov=0.8077 delay_exposure_cov=0.2885 minmax=0.8846 CRPS_phys=36.33 (s_out dp/t3 = 2.5198/7.6387)
  L0@H30: deterministic delay_phys MAE=293.39 over 52 windows
  L4@H60: n_windows=52 ttc3_10-90_cov=0.0769 (below 0.9231, above 0.0; truth=0 ties below 3, above 0) delay_phys_10-90_cov=0.25 delay_exposure_cov=0.1923 minmax=0.25 CRPS_phys=144.37 (s_out dp/t3 = None/None)
  L5@H60: n_windows=52 ttc3_10-90_cov=0.2692 (below 0.7115, above 0.0192; truth=0 ties below 3, above 0) delay_phys_10-90_cov=0.5385 delay_exposure_cov=0.5577 minmax=0.5577 CRPS_phys=106.94 (s_out dp/t3 = None/None)
  L5c@H60: n_windows=52 ttc3_10-90_cov=0.5962 (below 0.3846, above 0.0192; truth=0 ties below 2, above 0) delay_phys_10-90_cov=0.7308 delay_exposure_cov=0.5577 minmax=0.8654 CRPS_phys=113.18 (s_out dp/t3 = 1.9097/2.2974)
  L6@H60: n_windows=52 ttc3_10-90_cov=0.3462 (below 0.1538, above 0.5; truth=0 ties below 1, above 0) delay_phys_10-90_cov=0.4423 delay_exposure_cov=0.5192 minmax=0.5577 CRPS_phys=89.78 (s_out dp/t3 = None/None)
  L6c@H60: n_windows=52 ttc3_10-90_cov=0.8846 (below 0.0, above 0.1154; truth=0 ties below 0, above 0) delay_phys_10-90_cov=0.75 delay_exposure_cov=0.5192 minmax=0.9231 CRPS_phys=93.12 (s_out dp/t3 = 2.2974/14.5876)
  L0@H60: deterministic delay_phys MAE=738.77 over 52 windows

== GATE v2 at H=30 (PRIMARY) -- (d) BINDING per-horizon form ==
  L0: (a) coll/1000=0.0 PASS | (b) lc 0.0/12.077 (100.0%) FAIL | (c) n/a (deterministic) -> NA | (d) speed 4.1881 vs 1.05xL0@H=4.3975 (PASS, binding) / fixed 5.2185 (PASS) ==> v2 FAIL: b_lc
  L4: (a) coll/1000=0.024 PASS | (b) lc 11.669/12.077 (3.4%) PASS | (c) ttc3 0.1923 delay-phys 0.0577 crps 47.26 vs mae 293.39 -> FAIL | (d) speed 4.0141 vs 1.05xL0@H=4.3975 (PASS, binding) / fixed 5.2185 (PASS) ==> v2 FAIL: c_coverage_crps
  L5: (a) coll/1000=0.019 PASS | (b) lc 10.927/12.077 (9.5%) PASS | (c) ttc3 0.1923 delay-phys 0.5 crps 42.6 vs mae 293.39 -> FAIL | (d) speed 3.954 vs 1.05xL0@H=4.3975 (PASS, binding) / fixed 5.2185 (PASS) ==> v2 FAIL: c_coverage_crps
  L5c: (a) coll/1000=0.019 PASS | (b) lc 10.927/12.077 (9.5%) PASS | (c) ttc3 0.5962 delay-phys 0.8462 crps 46.72 vs mae 293.39 -> FAIL | (d) speed 3.954 vs 1.05xL0@H=4.3975 (PASS, binding) / fixed 5.2185 (PASS) ==> v2 FAIL: c_coverage_crps
  L6: (a) coll/1000=0.03 PASS | (b) lc 16.865/12.077 (39.6%) FAIL | (c) ttc3 0.4038 delay-phys 0.2885 crps 36.4 vs mae 293.39 -> FAIL | (d) speed 4.3438 vs 1.05xL0@H=4.3975 (PASS, binding) / fixed 5.2185 (PASS) ==> v2 FAIL: b_lc,c_coverage_crps
  L6c: (a) coll/1000=0.03 PASS | (b) lc 16.865/12.077 (39.6%) FAIL | (c) ttc3 0.9231 delay-phys 0.8077 crps 36.33 vs mae 293.39 -> PASS | (d) speed 4.3438 vs 1.05xL0@H=4.3975 (PASS, binding) / fixed 5.2185 (PASS) ==> v2 FAIL: b_lc

== OLD gate at H=30 ==
  L0: speed 4.1881 < 4.97 True | coll 0.0 True | lc 100.0% False | delay minmax None False ==> FAIL
  L4: speed 4.0141 < 4.97 True | coll 0.024 True | lc 3.4% True | delay minmax 0.0962 False ==> FAIL
  L5: speed 3.954 < 4.97 True | coll 0.019 True | lc 9.5% True | delay minmax 0.6154 True ==> PASS
  L5c: speed 3.954 < 4.97 True | coll 0.019 True | lc 9.5% True | delay minmax 0.9038 True ==> PASS
  L6: speed 4.3438 < 4.97 True | coll 0.03 True | lc 39.6% False | delay minmax 0.4423 False ==> FAIL
  L6c: speed 4.3438 < 4.97 True | coll 0.03 True | lc 39.6% False | delay minmax 0.8846 True ==> FAIL

== GATE v2 at H=60 (secondary) -- (d) BINDING per-horizon form ==
  L0: (a) coll/1000=0.0 PASS | (b) lc 0.0/23.942 (100.0%) FAIL | (c) n/a (deterministic) -> NA | (d) speed 4.9774 vs 1.05xL0@H=5.2263 (PASS, binding) / fixed 5.2185 (PASS) ==> v2 FAIL: b_lc
  L4: (a) coll/1000=0.019 PASS | (b) lc 23.85/23.942 (0.4%) PASS | (c) ttc3 0.0769 delay-phys 0.25 crps 144.37 vs mae 738.77 -> FAIL | (d) speed 5.1805 vs 1.05xL0@H=5.2263 (PASS, binding) / fixed 5.2185 (PASS) ==> v2 FAIL: c_coverage_crps
  L5: (a) coll/1000=0.015 PASS | (b) lc 21.385/23.942 (10.7%) PASS | (c) ttc3 0.2692 delay-phys 0.5385 crps 106.94 vs mae 738.77 -> FAIL | (d) speed 4.8965 vs 1.05xL0@H=5.2263 (PASS, binding) / fixed 5.2185 (PASS) ==> v2 FAIL: c_coverage_crps
  L5c: (a) coll/1000=0.015 PASS | (b) lc 21.385/23.942 (10.7%) PASS | (c) ttc3 0.5962 delay-phys 0.7308 crps 113.18 vs mae 738.77 -> FAIL | (d) speed 4.8965 vs 1.05xL0@H=5.2263 (PASS, binding) / fixed 5.2185 (PASS) ==> v2 FAIL: c_coverage_crps
  L6: (a) coll/1000=0.019 PASS | (b) lc 31.015/23.942 (29.5%) FAIL | (c) ttc3 0.3462 delay-phys 0.4423 crps 89.78 vs mae 738.77 -> FAIL | (d) speed 5.134 vs 1.05xL0@H=5.2263 (PASS, binding) / fixed 5.2185 (PASS) ==> v2 FAIL: b_lc,c_coverage_crps
  L6c: (a) coll/1000=0.019 PASS | (b) lc 31.015/23.942 (29.5%) FAIL | (c) ttc3 0.8846 delay-phys 0.75 crps 93.12 vs mae 738.77 -> FAIL | (d) speed 5.134 vs 1.05xL0@H=5.2263 (PASS, binding) / fixed 5.2185 (PASS) ==> v2 FAIL: b_lc,c_coverage_crps

== OLD gate at H=60 ==
  L0: speed 4.9774 < 4.97 False | coll 0.0 True | lc 100.0% False | delay minmax None False ==> FAIL
  L4: speed 5.1805 < 4.97 False | coll 0.019 True | lc 0.4% True | delay minmax 0.25 False ==> FAIL
  L5: speed 4.8965 < 4.97 True | coll 0.015 True | lc 10.7% True | delay minmax 0.5577 True ==> PASS
  L5c: speed 4.8965 < 4.97 True | coll 0.015 True | lc 10.7% True | delay minmax 0.8654 True ==> PASS
  L6: speed 5.134 < 4.97 False | coll 0.019 True | lc 29.5% False | delay minmax 0.5577 True ==> FAIL
  L6c: speed 5.134 < 4.97 False | coll 0.019 True | lc 29.5% False | delay minmax 0.9231 True ==> FAIL

wrote D:\rd-wt\wd01k\output\wd01\part3m\results\gate_v2_p3m.json
```

*(This report is generated by make_report_p3m.py; all numbers are pasted from
the tee'd stdout files and the results JSONs above.  Holdout not accessed;
ZERO simulation.)*
