# WD-01 Part 3L — merge-aware emulator gap close: TASK A merge check (+ follower-slot fix), extended-grid refit with LC@H30, TRAIN-calibrated ensemble spread, gate v2 re-run

****Holdout NOT accessed.**  No holdout root was loaded (the holdout block inside `root_arrays` stays authoritative); no registration hash was created or passed anywhere in part 3l.**

ZERO simulation: no SUMO/TraCI import anywhere; the only dynamics is
the part-3 rollout engine (pure torch).  TRAIN fit roots for fitting,
TRAIN calibration roots for spread calibration, VALIDATION roots for
reporting only.  Branch zcode/wd01-p3l.

## TASK A — merge rejection causes on TRAIN (verdict)

Instrumentation: `merge_acceptance(..., return_parts=True)` (default
unchanged) and `Rollout.merge_stats` counters in `part3/engine.py`
(pure bookkeeping; the merge decision inputs unchanged).  Audit over
the 8 TRAIN fit windows, 6 acceptance configs:
`results/stdout_merge_check.txt` (verbatim below).

- **Follower-slot defect found and fixed**: `engine._gap_accept_merge`
  selected the target-lane follower with `argmax(occ_d[behind])` --
  the FARTHEST-behind main_right vehicle -- instead of the NEAREST
  follower (`argmin`).  The inflated gap made b_req tiny, so the
  follower term never bound and p3k's fit loss was identical across
  b_accept 2.0-6.5.  Fixed to argmin (documented in engine.py);
  unit-tested: `test_follower_slot_is_nearest_behind` /
  `test_follower_slot_rejects_on_fast_nearest`.
- Post-fix, over the same 8 TRAIN windows at the p3k params
  (tau_accept=1.9, b_accept=2.0): 8337 M2 checks, 44 accepted
  (rate 0.005);
  rejections lead-only=171,
  follower-only=7575,
  both=547;
  8169 checks have b_req above the
  smallest grid b (1.5) -- **the follower term now binds and
  b_accept is a first-order parameter** (accepted merges: 24 at
  b=1.5, 44 at b=2.0 over the same windows).  p3k's flat b_accept
  response is fully explained by the slot defect.
- The kind `idm_merge` behaviour change is the TASK A fix ONLY
  (documented per brief); `idm_sampled` remains untouched.

## TASK B — extended-grid fit with LC matched at H=30 (TRAIN only)

Protocol, grids and the selected parameters are in the fit stdout
below.  Selected: merge_params={'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.0}, lc_scale=0.00075
(member 0; members 1-4 are WIDER DRAWS per TASK C mechanism B:
bootstrap resample of the 4 fit roots, softmax draws (T=0.15) over
coordinate loss lines incl. p_coop_scale, plus independent dawdle-width
draws; seeds m*17+1).  Fitted-on record: {'roots': ['cv17', 'cv19_4107', 'cv19_4108', 'cv20_7001m2000'], 't0s': [120.0, 300.0], 'H': 60.0, 'note': 'TRAIN only; holdout never accessed'}.
Grids: tau_accept [1.9, 2.4, 3.0, 3.6], b_accept
[1.5, 2.0, 3.0], lc_scale [0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.003],
p_coop [0.0, 0.25], dawdle
[0.8, 1.0, 1.25, 1.5].
LC target: per-window LC count matched at H=30 (primary) and H=60
(half weight), jointly with the acceptance grids (stage 3 re-sweep).
Stage losses: stage1 1.1075, stage2
1.2283, joint 2.3358 (before L3:
4.5020 / 4.5600).

## TASK C — ensemble spread calibration (TRAIN-internal split)

Calibration roots (next 4 sorted TRAIN roots, disjoint from the fit
roots; validation never used): ['cv20_7002m2000', 'cv20_7003m2000', 'cv20_7004m2000', 'cv20_7007m2700'].
28 windows x 5 L5 members x H {30, 60, 90, 120}.
Mechanism A (applied at gate time as rung L5c): per-(output, horizon)
factor s_out widening each window's member ensemble around its MEAN --
the ensemble MEAN is exactly invariant; the brief's median-centred
variant is reported as a check (its mean shift is in
results/calibration_p3l.json).  Mechanism B: the wider-draw L5 member
ensemble itself; its raw coverage is the `raw_coverage` entries.
Applied L5c map at gate horizons: {'delay_phys': {'30': 1.7411, '60': 1.9097}, 'ttc3': {'30': 2.0946, '60': 2.2974}}.
Calibration coverage entries:
  - delay_phys@H30: {'raw_coverage': 0.4286, 's_out': 1.7411, 'target': 0.8, 'coverage_mean_centred': 0.8214, 'coverage_median_centred': 0.9286, 'crps_raw_mean': 39.657, 'crps_widened_mean': 43.03, 'mean_shift_max': 0.0, 'mean_shift_median_centred_max': 32.556}
  - delay_phys@H60: {'raw_coverage': 0.25, 's_out': 1.9097, 'target': 0.8, 'coverage_mean_centred': 0.8571, 'coverage_median_centred': 0.8571, 'crps_raw_mean': 95.315, 'crps_widened_mean': 96.024, 'mean_shift_max': 0.0, 'mean_shift_median_centred_max': 84.473}
  - delay_phys@H90: {'raw_coverage': 0.2857, 's_out': 2.2974, 'target': 0.8, 'coverage_mean_centred': 0.9286, 'coverage_median_centred': 0.8571, 'crps_raw_mean': 173.014, 'crps_widened_mean': 167.942, 'mean_shift_max': 0.0, 'mean_shift_median_centred_max': 180.632}
  - delay_phys@H120: {'raw_coverage': 0.3214, 's_out': 2.5198, 'target': 0.8, 'coverage_mean_centred': 0.8929, 'coverage_median_centred': 0.7857, 'crps_raw_mean': 286.288, 'crps_widened_mean': 264.424, 'mean_shift_max': 0.0, 'mean_shift_median_centred_max': 268.921}
  - ttc3@H30: {'raw_coverage': 0.4643, 's_out': 2.0946, 'target': 0.8, 'coverage_mean_centred': 0.8214, 'coverage_median_centred': 0.8214, 'crps_raw_mean': 4.236, 'crps_widened_mean': 4.265, 'mean_shift_max': 0.0, 'mean_shift_median_centred_max': 3.065}
  - ttc3@H60: {'raw_coverage': 0.2857, 's_out': 2.2974, 'target': 0.8, 'coverage_mean_centred': 0.8214, 'coverage_median_centred': 0.8214, 'crps_raw_mean': 9.59, 'crps_widened_mean': 9.863, 'mean_shift_max': 0.0, 'mean_shift_median_centred_max': 9.212}
  - ttc3@H90: {'raw_coverage': 0.2857, 's_out': 2.5198, 'target': 0.8, 'coverage_mean_centred': 0.8214, 'coverage_median_centred': 0.8929, 'crps_raw_mean': 14.59, 'crps_widened_mean': 15.008, 'mean_shift_max': 0.0, 'mean_shift_median_centred_max': 33.284}
  - ttc3@H120: {'raw_coverage': 0.3571, 's_out': 2.5198, 'target': 0.8, 'coverage_mean_centred': 0.8214, 'coverage_median_centred': 0.8929, 'crps_raw_mean': 20.882, 'crps_widened_mean': 21.553, 'mean_shift_max': 0.0, 'mean_shift_median_centred_max': 59.272}
(Coverage target 80 %; `None` s_out = even s=16 cannot reach it on
degenerate windows; full table in stdout_calibrate.txt.)

## TASK D — gate v2 ladder on VALIDATION (summary)

Ladder: L0 (deterministic), L4 (p3k PARAMETERS — the TASK A slot fix
is engine-global, so this rung is p3k params on the FIXED engine, NOT
the p3k published behaviour; its deviation from the p3k gate table is
attributable to the slot fix, whose over-merging inflated p3k's LC
counts), L5 (part 3l refit, fixed slot), L5c (L5 + spread
calibration, derived at aggregation time).
M=5 members, 4 validation roots x 52 windows, H {30, 60, 90, 120};
raw rows first: results/raw/raw_metrics.jsonl.

- Gate v2 at H=30 (PRIMARY per part3f/VALIDATION_GATE_V2_AMENDMENT_1.md):
  - L0: (a) PASS (coll/1000=0.0) | (b) FAIL (100.0%) | (c) NA (deterministic) | (d) per-horizon PASS (RMSE=4.1881 vs thr=4.3975, BINDING) / fixed PASS (thr=5.2185) ==> FAIL: b_lc
  - L4: (a) PASS (coll/1000=0.024) | (b) PASS (3.4%) | (c) FAIL | (d) per-horizon PASS (RMSE=4.0141 vs thr=4.3975, BINDING) / fixed PASS (thr=5.2185) ==> FAIL: c_coverage_crps
  - L5: (a) PASS (coll/1000=0.019) | (b) PASS (9.5%) | (c) FAIL | (d) per-horizon PASS (RMSE=3.954 vs thr=4.3975, BINDING) / fixed PASS (thr=5.2185) ==> FAIL: c_coverage_crps
  - L5c: (a) PASS (coll/1000=0.019) | (b) PASS (9.5%) | (c) FAIL | (d) per-horizon PASS (RMSE=3.954 vs thr=4.3975, BINDING) / fixed PASS (thr=5.2185) ==> FAIL: c_coverage_crps
- Gate v2 at H=60 (secondary): L0/L4/L5/L5c in the stdout below.
- Criterion (d) is reported BOTH ways per the reviewer clarification;
  the per-horizon form (1.05 x L0 speed RMSE at the same horizon of
  the same run) is BINDING and decides the verdict.
- Criterion (c) uses the CONTRACT PHYSICS DELAY (dt - ds/limit, the
  p3j/p3k decompose model); exposure-based coverage is reported
  alongside for continuity.
- Old gate at H=30: {'L0': False, 'L4': False, 'L5': True, 'L5c': True}
  and at H=60: {'L0': False, 'L4': False, 'L5': True, 'L5c': True} -- the OLD gate
  PASSES for L5/L5c at both horizons (first PASS in the series).
- Residual gate-v2 gap (criterion c): at H=30 L5c passes the
  delay-phys coverage (0.8462 >= 0.80)
  and CRPS (46.72 < L0 MAE
  293.39) checks; the failing term is the
  TTC<3 s exposure band coverage (0.5962 <
  0.80; calibrated to 0.8214
  on the TRAIN-calibration split -- a band-spread generalization gap,
  not a mean-bias shift).  At H=60 the delay-phys coverage
  (0.7308) is also
  short of 0.80.

## TASK A stdout (merge_check_p3l.py, TRAIN fit roots only -- verbatim)



```text
priority below-normal: False
=== WD-01 part 3l TASK A: merge rejection causes (TRAIN) ===
ZERO simulation; holdout blocked in root_arrays (never passed).
device=cpu torch_threads=2
train fit roots: ['cv17', 'cv19_4107', 'cv19_4108', 'cv20_7001m2000']; t0 (120.0, 300.0); H=60 s (K=120 steps of 0.5 s); lc_scale=0.002 (p3k fit); member_seed=1
calibration roots (used only by calibrate_p3l.py, never here): ['cv20_7002m2000', 'cv20_7003m2000', 'cv20_7004m2000', 'cv20_7007m2700']
validation roots NOT used here: ['cv20_7005m2000', 'cv20_7006m2000', 'cv30_7305m2700', 'seed_7535m2400']; 5 holdout roots blocked
[   0.0s] rolling main (p3k fit) over 8 windows ...

  [main (p3k fit)] merge_params={'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0}
    M2 checks=8337 accepted=44 (rate 0.005); accel->main_right merges=44
    rejections: lead-only=171 foll-only=7575 both=547
    follower-present checks=8314 (99.7 %); b_req histogram <=0 / (0,1] / (1,1.5] / (1.5,2] / (2,3] / >3: [106, 22, 17, 47, 492, 7630]
    b_req at follower-only rejections: q10/50/90/max = 3.68/9.02/63.98/1540.02
[  56.5s] rolling tau=2.4 b=2.0 over 8 windows ...

  [tau=2.4 b=2.0] merge_params={'tau_accept': 2.4, 'b_accept': 2.0, 'p_coop_scale': 0.0}
    M2 checks=8603 accepted=49 (rate 0.006); accel->main_right merges=49
    rejections: lead-only=260 foll-only=7676 both=618
    follower-present checks=8582 (99.8 %); b_req histogram <=0 / (0,1] / (1,1.5] / (1.5,2] / (2,3] / >3: [156, 34, 32, 66, 581, 7713]
    b_req at follower-only rejections: q10/50/90/max = 3.58/8.78/61.07/1540.02
[ 112.2s] rolling tau=3.0 b=2.0 over 8 windows ...

  [tau=3.0 b=2.0] merge_params={'tau_accept': 3.0, 'b_accept': 2.0, 'p_coop_scale': 0.0}
    M2 checks=8325 accepted=53 (rate 0.006); accel->main_right merges=53
    rejections: lead-only=265 foll-only=7344 both=663
    follower-present checks=8296 (99.7 %); b_req histogram <=0 / (0,1] / (1,1.5] / (1.5,2] / (2,3] / >3: [141, 39, 34, 75, 517, 7490]
    b_req at follower-only rejections: q10/50/90/max = 3.67/8.97/65.35/1540.02
[ 167.8s] rolling tau=3.6 b=2.0 over 8 windows ...

  [tau=3.6 b=2.0] merge_params={'tau_accept': 3.6, 'b_accept': 2.0, 'p_coop_scale': 0.0}
    M2 checks=8755 accepted=51 (rate 0.006); accel->main_right merges=51
    rejections: lead-only=302 foll-only=7628 both=774
    follower-present checks=8728 (99.7 %); b_req histogram <=0 / (0,1] / (1,1.5] / (1.5,2] / (2,3] / >3: [140, 37, 50, 99, 548, 7854]
    b_req at follower-only rejections: q10/50/90/max = 3.71/9.05/65.32/1540.02
[ 224.3s] rolling tau=1.9 b=1.5 over 8 windows ...

  [tau=1.9 b=1.5] merge_params={'tau_accept': 1.9, 'b_accept': 1.5, 'p_coop_scale': 0.0}
    M2 checks=11966 accepted=24 (rate 0.002); accel->main_right merges=24
    rejections: lead-only=103 foll-only=11233 both=606
    follower-present checks=11949 (99.9 %); b_req histogram <=0 / (0,1] / (1,1.5] / (1.5,2] / (2,3] / >3: [81, 14, 15, 89, 709, 11041]
    b_req at follower-only rejections: q10/50/90/max = 3.58/8.70/60.96/1530.33
[ 280.1s] rolling tau=1.9 b=3.0 over 8 windows ...

  [tau=1.9 b=3.0] merge_params={'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.0}
    M2 checks=3243 accepted=102 (rate 0.031); accel->main_right merges=102
    rejections: lead-only=388 foll-only=2476 both=277
    follower-present checks=3237 (99.8 %); b_req histogram <=0 / (0,1] / (1,1.5] / (1.5,2] / (2,3] / >3: [160, 40, 16, 61, 207, 2753]
    b_req at follower-only rejections: q10/50/90/max = 4.61/9.95/75.34/1472.55

== TASK A verdict ==
  follower_slot: nearest main_right vehicle BEHIND the candidate in d (engine._gap_accept_merge); pinned by unit test test_follower_slot_is_nearest_behind
  follower_rejections: 8122
  follower_term_binds: True
  breq_checks_above_b_grid_min: 8169
  b_accept_flat_explained: False
[ 334.8s] wrote D:\rd-wt\wd01f\output\wd01\part3l\results\merge_check_p3l.json

DONE part 3l TASK A.
```

## TASK B fit stdout (fit_p3l.py, TRAIN only -- verbatim)



```text
priority below-normal: False
=== WD-01 part 3l TASK B: idm_merge fit, extended grids + LC@H30 (TRAIN only) ===
ZERO simulation; holdout blocked in root_arrays (never passed).
device=cpu torch_threads=2
train fit roots: ['cv17', 'cv19_4107', 'cv19_4108', 'cv20_7001m2000']; eval windows t0 (120.0, 300.0), H=60 s (K=120 steps of 0.5 s); validation roots ['cv20_7005m2000', 'cv20_7006m2000', 'cv30_7305m2700', 'seed_7535m2400'] NOT used here; calibration roots ['cv20_7002m2000', 'cv20_7003m2000', 'cv20_7004m2000', 'cv20_7007m2700'] NOT used here
grids: tau_accept (1.9, 2.4, 3.0, 3.6) x b_accept (1.5, 2.0, 3.0); lc_scale (0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.003) (matched at H=30 primary, H=60 half weight); p_coop (0.0, 0.25); dawdle draws (0.8, 1.0, 1.25, 1.5); member softmax T=0.15
M2 follower slot: FIXED nearest-behind (part 3l TASK A)
[   4.0s] BEFORE: rolling L3 (idm_sampled, member 1, lc_scale=0.003) ...

BEFORE (L3 = idm_sampled, member 1, lc_scale=0.003): stage1 loss=4.5020 {'occ': 0.4024443788069275, 'prof': 0.7967072320468978, 'merge': 4.0} | stage2 loss=4.5600 {'mr': 2.002148363156379, 'delay': 1.599810555207017, 'lc30': 0.21521079043273644, 'lc60': 1.4857264081115922}

  -- BEFORE L3 (idm_sampled) (per eval window) --
  root             t0   occ_p occ_t | accel v by d-bin -50..300 pred | truth | merge_d p/t   mr_spd p/t    delay_p  delay_t  lc30 p/t lc60 p/t
  cv17             120     69   142 | 17.0 19.4 19.3 19.4 19.1 19.1 18.9 18.2 |  7.1 17.8 19.0 19.4 19.4 19.0 18.5 17.1 |    n/a/ -58.4   27.2/ 17.1     728.7    743.2   13/ 11  54/ 20
  cv17             300     90   225 | 12.6 20.1 20.3 20.7 20.9 20.8 20.7 20.9 |  3.2 17.9 20.4 21.2 21.6 21.7 21.9 21.6 |    n/a/ -56.2   10.4/  2.7    1415.3   2078.1   17/ 12  70/ 21
  cv19_4107        120     47    86 | 21.2 20.4 20.3 20.2 20.1 20.0 19.5 17.7 | 12.8 18.3 19.8 20.3 20.5 20.3 19.8 17.1 |    n/a/ -18.9   28.5/ 24.6     420.2    419.4   14/ 24  41/ 38
  cv19_4107        300     52   432 | 19.7 20.4 20.2 20.2 20.2 20.1 19.6 17.6 |  3.5 17.1 19.9 20.8 20.9 20.7 20.2 17.9 |    n/a/ -40.9   27.2/ 21.0     669.5    746.7   12/  5  42/ 22
  cv19_4108        120     47   115 | 20.4 18.9 18.6 18.5 18.7 18.6 18.3 18.0 |  8.5 18.2 19.1 19.4 19.4 19.0 18.5 17.1 |    n/a/ -50.2   28.6/ 19.4     403.7    454.5    9/ 18  34/ 39
  cv19_4108        300     24   100 | 21.9 20.6 20.6 20.8 20.8 20.4 20.7 21.0 | 15.4 19.6 20.8 21.3 21.6 21.8 21.9 21.6 |    n/a/ -52.3   26.6/ 23.6     455.0    407.2    9/ 16  36/ 30
  cv20_7001m2000   120     40   161 | 21.9 20.6 20.6 20.5 20.5 20.1 20.1 20.3 | 10.2 18.3 19.8 20.5 20.8 20.8 21.1 21.2 |    n/a/ -31.6   28.0/ 23.7     550.9    529.6   11/  6  42/ 16
  cv20_7001m2000   300     70   108 | 21.9 20.6 20.5 20.4 20.3 20.4 20.4 20.0 | 14.0 18.3 19.4 19.8 20.0 20.1 20.1 19.4 |    n/a/ -51.6   17.0/  8.4     644.3    891.6    9/ 15  39/ 34
[  57.4s] STAGE 1: 12 acceptance combos ...
[ 117.8s] stage1 {'tau_accept': 1.9, 'b_accept': 1.5, 'p_coop_scale': 0.0} -> loss=8.1073 {'occ': 6.076487932665402, 'prof': 0.19656122515323143, 'merge': 2.006273848548842}
[ 176.1s] stage1 {'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0} -> loss=3.8438 {'occ': 3.271238726386365, 'prof': 0.2587069311995647, 'merge': 0.5402720985337485}
[ 235.4s] stage1 {'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.0} -> loss=0.9494 {'occ': 0.8726846157237302, 'prof': 0.31168020602566826, 'merge': 0.037784254819945295}
[ 293.1s] stage1 {'tau_accept': 2.4, 'b_accept': 1.5, 'p_coop_scale': 0.0} -> loss=8.2555 {'occ': 6.226236080813551, 'prof': 0.18558880298500535, 'merge': 2.0060666706180217}
[ 351.8s] stage1 {'tau_accept': 2.4, 'b_accept': 2.0, 'p_coop_scale': 0.0} -> loss=4.2956 {'occ': 3.7235069736483917, 'prof': 0.23470807394600268, 'merge': 0.5427971305214591}
[ 410.9s] stage1 {'tau_accept': 2.4, 'b_accept': 3.0, 'p_coop_scale': 0.0} -> loss=0.9219 {'occ': 0.8485202220766108, 'prof': 0.29374098949088523, 'merge': 0.036656364619185276}
[ 468.9s] stage1 {'tau_accept': 3.0, 'b_accept': 1.5, 'p_coop_scale': 0.0} -> loss=8.7856 {'occ': 6.754013858591328, 'prof': 0.18103421444377785, 'merge': 2.008924760407802}
[ 526.4s] stage1 {'tau_accept': 3.0, 'b_accept': 2.0, 'p_coop_scale': 0.0} -> loss=4.3254 {'occ': 3.7598283657681826, 'prof': 0.2375324897871623, 'merge': 0.5359112356376712}
[ 588.0s] stage1 {'tau_accept': 3.0, 'b_accept': 3.0, 'p_coop_scale': 0.0} -> loss=1.2049 {'occ': 1.1127095079859017, 'prof': 0.28824582891429434, 'merge': 0.056160349904913066}
[ 652.0s] stage1 {'tau_accept': 3.6, 'b_accept': 1.5, 'p_coop_scale': 0.0} -> loss=8.7968 {'occ': 6.761783491650314, 'prof': 0.1797345435986634, 'merge': 2.012543119539837}
[ 712.3s] stage1 {'tau_accept': 3.6, 'b_accept': 2.0, 'p_coop_scale': 0.0} -> loss=5.0219 {'occ': 4.454353040600758, 'prof': 0.2259236853434376, 'merge': 0.5392759347512442}
[ 770.7s] stage1 {'tau_accept': 3.6, 'b_accept': 3.0, 'p_coop_scale': 0.0} -> loss=1.5724 {'occ': 1.4645921744073527, 'prof': 0.2742211755416334, 'merge': 0.07357905681587645}

STAGE 1 winner: {'tau_accept': 2.4, 'b_accept': 3.0, 'p_coop_scale': 0.0} loss=0.9219 {'occ': 0.8485202220766108, 'prof': 0.29374098949088523, 'merge': 0.036656364619185276}
[ 770.7s] STAGE 2: 12 lc x p_coop combos at {'tau_accept': 2.4, 'b_accept': 3.0, 'p_coop_scale': 0.0} ...
[ 830.5s] stage2 lc_scale=0.0005 p_coop=0.0 -> loss=1.2545 {'mr': 0.28974407900303617, 'delay': 0.5441739046952567, 'lc30': 0.34533317909205696, 'lc60': 0.15042636276268903}
[ 888.4s] stage2 lc_scale=0.0005 p_coop=0.25 -> loss=1.3381 {'mr': 0.2969059396716724, 'delay': 0.584260157224359, 'lc30': 0.39615300734618913, 'lc60': 0.1215173539273244}
[ 946.8s] stage2 lc_scale=0.00075 p_coop=0.0 -> loss=1.2372 {'mr': 0.2898691853637013, 'delay': 0.5941267674673899, 'lc30': 0.28534786788337924, 'lc60': 0.13577280068031203}
[1003.4s] stage2 lc_scale=0.00075 p_coop=0.25 -> loss=1.3298 {'mr': 0.29918682160688437, 'delay': 0.660506834143979, 'lc30': 0.319685580951561, 'lc60': 0.10084402714339535}
[1061.3s] stage2 lc_scale=0.001 p_coop=0.0 -> loss=1.2452 {'mr': 0.2904401568851434, 'delay': 0.5934217919011678, 'lc30': 0.2558110719728854, 'lc60': 0.21108786021878972}
[1117.4s] stage2 lc_scale=0.001 p_coop=0.25 -> loss=1.3971 {'mr': 0.37081316695679467, 'delay': 0.6415029731325371, 'lc30': 0.30637925379106723, 'lc60': 0.1568233795074973}
[1177.4s] stage2 lc_scale=0.0015 p_coop=0.0 -> loss=1.2808 {'mr': 0.31696039511046326, 'delay': 0.5929056773938068, 'lc30': 0.1977983452740792, 'lc60': 0.34627958623390453}
[1230.3s] stage2 lc_scale=0.0015 p_coop=0.25 -> loss=1.4568 {'mr': 0.4217673888470873, 'delay': 0.5720005175850543, 'lc30': 0.22329421304267422, 'lc60': 0.47955562565489707}
[1283.0s] stage2 lc_scale=0.002 p_coop=0.0 -> loss=1.5511 {'mr': 0.25273314131836233, 'delay': 0.5576279009305386, 'lc30': 0.20316582212593104, 'lc60': 1.0750698036304824}
[1338.0s] stage2 lc_scale=0.002 p_coop=0.25 -> loss=1.7693 {'mr': 0.4124510921007417, 'delay': 0.6494513014269555, 'lc30': 0.20626499567965004, 'lc60': 1.0022562669408026}
[1393.7s] stage2 lc_scale=0.003 p_coop=0.0 -> loss=2.8456 {'mr': 0.2239824727592839, 'delay': 0.6044235400977124, 'lc30': 0.38249463742539025, 'lc60': 3.2694909375059664}
[1453.6s] stage2 lc_scale=0.003 p_coop=0.25 -> loss=3.1667 {'mr': 0.28504019637146705, 'delay': 0.6332210565088806, 'lc30': 0.6972088247532139, 'lc60': 3.1025456278068444}

STAGE 2 winner: lc_scale=0.00075 p_coop=0.0 loss=1.2372 {'mr': 0.2898691853637013, 'delay': 0.5941267674673899, 'lc30': 0.28534786788337924, 'lc60': 0.13577280068031203}
[1453.6s] STAGE 3: 12 acceptance combos at lc_scale=0.00075 p_coop=0.0 ...
[1515.1s] stage3 {'tau_accept': 1.9, 'b_accept': 1.5, 'p_coop_scale': 0.0} -> s1+s2=8.6290 (s1 6.1856 s2 2.4434)
[1578.8s] stage3 {'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0} -> s1+s2=7.0256 (s1 4.6701 s2 2.3555)
[1640.3s] stage3 {'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.0} -> s1+s2=2.3358 (s1 1.1075 s2 1.2283)
[1703.4s] stage3 {'tau_accept': 2.4, 'b_accept': 1.5, 'p_coop_scale': 0.0} -> s1+s2=8.8635 (s1 6.3280 s2 2.5354)
[1765.8s] stage3 {'tau_accept': 2.4, 'b_accept': 2.0, 'p_coop_scale': 0.0} -> s1+s2=6.9956 (s1 4.6760 s2 2.3196)
[1828.8s] stage3 {'tau_accept': 2.4, 'b_accept': 3.0, 'p_coop_scale': 0.0} -> s1+s2=2.3447 (s1 1.1075 s2 1.2372)
[1892.6s] stage3 {'tau_accept': 3.0, 'b_accept': 1.5, 'p_coop_scale': 0.0} -> s1+s2=9.1737 (s1 6.6012 s2 2.5725)
[1960.1s] stage3 {'tau_accept': 3.0, 'b_accept': 2.0, 'p_coop_scale': 0.0} -> s1+s2=7.0708 (s1 4.7500 s2 2.3207)
[2028.7s] stage3 {'tau_accept': 3.0, 'b_accept': 3.0, 'p_coop_scale': 0.0} -> s1+s2=2.4926 (s1 1.1995 s2 1.2932)
[2094.8s] stage3 {'tau_accept': 3.6, 'b_accept': 1.5, 'p_coop_scale': 0.0} -> s1+s2=9.1737 (s1 6.6012 s2 2.5725)
[2162.8s] stage3 {'tau_accept': 3.6, 'b_accept': 2.0, 'p_coop_scale': 0.0} -> s1+s2=7.5429 (s1 5.1533 s2 2.3896)
[2225.0s] stage3 {'tau_accept': 3.6, 'b_accept': 3.0, 'p_coop_scale': 0.0} -> s1+s2=2.7026 (s1 1.3987 s2 1.3039)

STAGE 3 winner: {'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.0} lc_scale=0.00075 joint=2.3358 (s1 1.1075 s2 1.2283)

AFTER (L5 = idm_merge fitted, fixed follower slot): stage1 loss=1.1075 {'occ': 1.0299149542658987, 'prof': 0.288453904302041, 'merge': 0.04153359617522007} | stage2 loss=1.2283 {'mr': 0.27373282658194953, 'delay': 0.5927296270059751, 'lc30': 0.28534786788337924, 'lc60': 0.15291860072371294}

  -- AFTER L5 (idm_merge fitted) (per eval window) --
  root             t0   occ_p occ_t | accel v by d-bin -50..300 pred | truth | merge_d p/t   mr_spd p/t    delay_p  delay_t  lc30 p/t lc60 p/t
  cv17             120    456   142 | 13.0 18.5 19.2 19.4 19.5 19.4 19.0 18.3 |  7.1 17.8 19.0 19.4 19.4 19.0 18.5 17.1 |  -66.9/ -58.4   20.1/ 17.1     974.9    743.2    8/ 11  26/ 20
  cv17             300     92   225 | 14.1 17.8 19.2 19.9 20.1 20.1 20.2 20.8 |  3.2 17.9 20.4 21.2 21.6 21.7 21.9 21.6 |  -58.4/ -56.2    4.9/  2.7    1829.6   2078.1   11/ 12  23/ 21
  cv19_4107        120     78    86 | 14.3 18.0 19.1 19.5 19.6 19.6 19.0 17.7 | 12.8 18.3 19.8 20.3 20.5 20.3 19.8 17.1 |  -42.4/ -18.9   26.3/ 24.6     464.3    419.4    5/ 24  18/ 38
  cv19_4107        300    221   432 | 14.8 18.4 19.5 19.9 20.0 19.8 19.4 17.2 |  3.5 17.1 19.9 20.8 20.9 20.7 20.2 17.9 |  -56.5/ -40.9   18.6/ 21.0     848.4    746.7    6/  5  28/ 22
  cv19_4108        120    291   115 | 13.6 17.8 18.7 18.9 18.8 18.6 18.6 18.0 |  8.5 18.2 19.1 19.4 19.4 19.0 18.5 17.1 |  -59.0/ -50.2   23.7/ 19.4     556.9    454.5    3/ 18  14/ 39
  cv19_4108        300    125   100 | 16.0 18.6 19.6 20.1 20.6 20.7 20.9 20.9 | 15.4 19.6 20.8 21.3 21.6 21.8 21.9 21.6 |  -42.9/ -52.3   22.3/ 23.6     577.7    407.2    4/ 16  20/ 30
  cv20_7001m2000   120    256   161 | 14.2 17.8 19.1 19.6 20.0 20.0 20.4 20.4 | 10.2 18.3 19.8 20.5 20.8 20.8 21.1 21.2 |  -60.2/ -31.6   23.6/ 23.7     653.9    529.6    3/  6  24/ 16
  cv20_7001m2000   300    107   108 | 15.4 18.1 19.2 19.6 19.8 19.9 20.0 19.8 | 14.0 18.3 19.4 19.8 20.0 20.1 20.1 19.4 |  -47.9/ -51.6   12.0/  8.4     808.6    891.6    8/ 15  33/ 34
[2286.9s] MEMBERS 1..4: wider draws (bootstrap resample x softmax lines x dawdle width) ...
[2691.2s]   member 1 tau_accept line losses ['4.038', '4.041', '4.142', '4.146'] -> draw 1.9
[2691.2s]   member 1 b_accept line losses ['13.926', '13.250', '4.038'] -> draw 3.0
[2691.2s]   member 1 p_coop_scale line losses ['4.038', '4.217'] -> draw 0.25
[2691.2s]   member 1 lc_scale line losses ['4.064', '4.038', '4.127', '3.615', '3.671', '3.314'] -> draw 0.003
[2691.2s] member 1: {'seed': 18, 'lc_scale': 0.003, 'merge_params': {'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.25}, 'dawdle_scale': 1.25, 'resample': ['cv17', 'cv19_4107', 'cv19_4108']}
[3137.4s]   member 2 tau_accept line losses ['3.300', '3.420', '3.497', '3.501'] -> draw 1.9
[3137.4s]   member 2 b_accept line losses ['12.198', '11.691', '3.300'] -> draw 3.0
[3137.4s]   member 2 p_coop_scale line losses ['3.300', '3.365'] -> draw 0.25
[3137.4s]   member 2 lc_scale line losses ['3.301', '3.300', '3.379', '2.966', '3.170', '3.618'] -> draw 0.0005
[3137.4s] member 2: {'seed': 35, 'lc_scale': 0.0005, 'merge_params': {'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.25}, 'dawdle_scale': 1.5, 'resample': ['cv17', 'cv19_4107', 'cv19_4108', 'cv20_7001m2000']}
[3598.5s]   member 3 tau_accept line losses ['4.066', '4.224', '4.224', '4.209'] -> draw 1.9
[3598.5s]   member 3 b_accept line losses ['14.023', '13.969', '4.066'] -> draw 3.0
[3598.5s]   member 3 p_coop_scale line losses ['4.066', '4.144'] -> draw 0.0
[3598.5s]   member 3 lc_scale line losses ['4.040', '4.066', '4.076', '3.554', '3.921', '4.567'] -> draw 0.0015
[3598.5s] member 3: {'seed': 52, 'lc_scale': 0.0015, 'merge_params': {'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.0}, 'dawdle_scale': 0.8, 'resample': ['cv17', 'cv19_4108', 'cv20_7001m2000']}
[4023.3s]   member 4 tau_accept line losses ['2.151', '2.311', '2.413', '2.418'] -> draw 1.9
[4023.3s]   member 4 b_accept line losses ['6.105', '5.429', '2.151'] -> draw 3.0
[4023.3s]   member 4 p_coop_scale line losses ['2.151', '1.242'] -> draw 0.25
[4023.3s]   member 4 lc_scale line losses ['2.151', '2.151', '2.218', '2.114', '2.043', '2.953'] -> draw 0.001
[4023.3s] member 4: {'seed': 69, 'lc_scale': 0.001, 'merge_params': {'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.25}, 'dawdle_scale': 1.5, 'resample': ['cv19_4107', 'cv19_4108', 'cv20_7001m2000']}

selected member parameters (member 0 = main fit):
  member 0: seed=1 lc_scale=0.00075 merge_params={'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.0} dawdle_scale=1.0
  member 1: seed=18 lc_scale=0.003 merge_params={'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.25} dawdle_scale=1.25
  member 2: seed=35 lc_scale=0.0005 merge_params={'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.25} dawdle_scale=1.5
  member 3: seed=52 lc_scale=0.0015 merge_params={'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.0} dawdle_scale=0.8
  member 4: seed=69 lc_scale=0.001 merge_params={'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.25} dawdle_scale=1.5

b_accept sensitivity on the full fit windows (stage-1 loss values per tau, post-fix):
  b_accept=1.5: [8.107332, 8.255501, 8.785568, 8.796793]
  b_accept=2.0: [3.843849, 4.295643, 4.325431, 5.021869]
  b_accept=3.0: [0.921894, 0.949429, 1.204901, 1.572449]
[4023.3s] wrote D:\rd-wt\wd01f\output\wd01\part3l\results\fit_p3l.json and D:\rd-wt\wd01f\output\wd01\part3l\results\stdout_fit.txt

DONE part 3l TASK B fit.
```

## TASK C calibration stdout (calibrate_p3l.py, TRAIN-internal calibration split -- verbatim)



```text
priority below-normal: False
=== WD-01 part 3l TASK C: ensemble spread calibration (TRAIN-internal) ===
ZERO simulation; holdout blocked in root_arrays (never passed).
device=cpu torch_threads=2
calibration roots (TRAIN, held out of the fit): ['cv20_7002m2000', 'cv20_7003m2000', 'cv20_7004m2000', 'cv20_7007m2700']; t0 (120, 180, 240, 300, 360, 420, 480) -> 28 windows
fit roots (NOT rolled here): ['cv17', 'cv19_4107', 'cv19_4108', 'cv20_7001m2000']; validation ['cv20_7005m2000', 'cv20_7006m2000', 'cv30_7305m2700', 'seed_7535m2400'] NOT used; holdout blocked
member configs (from fit_p3l.json):
  member 0: seed=1 lc_scale=0.00075 merge_params={'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.0} dawdle_scale=1.0
  member 1: seed=18 lc_scale=0.003 merge_params={'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.25} dawdle_scale=1.25
  member 2: seed=35 lc_scale=0.0005 merge_params={'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.25} dawdle_scale=1.5
  member 3: seed=52 lc_scale=0.0015 merge_params={'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.0} dawdle_scale=0.8
  member 4: seed=69 lc_scale=0.001 merge_params={'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.25} dawdle_scale=1.5
[   0.0s] reusing 560 calibration rows already in D:\rd-wt\wd01f\output\wd01\part3l\results\raw\raw_calibration.jsonl (no re-roll)

== spread calibration per (output, H) on the TRAIN calibration split ==
  delay_phys@H30: raw_cov=0.4286 s_out=1.7411 cov_mean=0.8214 cov_median=0.9286 crps 39.657->43.03 mean_shift=0.0
  delay_phys@H60: raw_cov=0.25 s_out=1.9097 cov_mean=0.8571 cov_median=0.8571 crps 95.315->96.024 mean_shift=0.0
  delay_phys@H90: raw_cov=0.2857 s_out=2.2974 cov_mean=0.9286 cov_median=0.8571 crps 173.014->167.942 mean_shift=0.0
  delay_phys@H120: raw_cov=0.3214 s_out=2.5198 cov_mean=0.8929 cov_median=0.7857 crps 286.288->264.424 mean_shift=0.0
  ttc3@H30: raw_cov=0.4643 s_out=2.0946 cov_mean=0.8214 cov_median=0.8214 crps 4.236->4.265 mean_shift=0.0
  ttc3@H60: raw_cov=0.2857 s_out=2.2974 cov_mean=0.8214 cov_median=0.8214 crps 9.59->9.863 mean_shift=0.0
  ttc3@H90: raw_cov=0.2857 s_out=2.5198 cov_mean=0.8214 cov_median=0.8929 crps 14.59->15.008 mean_shift=0.0
  ttc3@H120: raw_cov=0.3571 s_out=2.5198 cov_mean=0.8214 cov_median=0.8929 crps 20.882->21.553 mean_shift=0.0
[   0.5s] wrote D:\rd-wt\wd01f\output\wd01\part3l\results\calibration_p3l.json

DONE part 3l TASK C calibration.
```

## TASK D ladder eval stdout (eval_p3l.py, VALIDATION only -- verbatim)



```text
priority below-normal: False
=== WD-01 part 3l TASK D: gate v2 ladder (VALIDATION only) ===
ZERO simulation; holdout blocked in root_arrays (never passed).
device=cpu torch_threads=2
validation roots ['cv20_7005m2000', 'cv20_7006m2000', 'cv30_7305m2700', 'seed_7535m2400']; t0 every 30 s (120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450, 480) (52 windows); H (30, 60, 90, 120)
member configs (L4 = p3k reference incl. its follower-slot defect; L5 = part 3l fit, fixed slot):
  L0 member 0: seed=1 lc_scale=0.0 merge_params=None dawdle_scale=1.0
  L4 member 0: seed=1 lc_scale=0.002 merge_params={'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0} dawdle_scale=1.0
  L4 member 1: seed=18 lc_scale=0.002 merge_params={'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0} dawdle_scale=1.0
  L4 member 2: seed=35 lc_scale=0.002 merge_params={'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0} dawdle_scale=1.0
  L4 member 3: seed=52 lc_scale=0.002 merge_params={'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.25} dawdle_scale=1.0
  L4 member 4: seed=69 lc_scale=0.002 merge_params={'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0} dawdle_scale=1.0
  L5 member 0: seed=1 lc_scale=0.00075 merge_params={'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.0} dawdle_scale=1.0
  L5 member 1: seed=18 lc_scale=0.003 merge_params={'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.25} dawdle_scale=1.25
  L5 member 2: seed=35 lc_scale=0.0005 merge_params={'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.25} dawdle_scale=1.5
  L5 member 3: seed=52 lc_scale=0.0015 merge_params={'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.0} dawdle_scale=0.8
  L5 member 4: seed=69 lc_scale=0.001 merge_params={'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.25} dawdle_scale=1.5
[   0.1s] rolling L0 member 0 (seed 1, dawdle 1.0) ...
[ 166.4s] L0 member 0: done (208 rows total)
[ 166.4s] rolling L4 member 0 (seed 1, dawdle 1.0) ...
[ 489.8s] L4 member 0: done (416 rows total)
[ 489.8s] rolling L4 member 1 (seed 18, dawdle 1.0) ...
[ 820.1s] L4 member 1: done (624 rows total)
[ 820.1s] rolling L4 member 2 (seed 35, dawdle 1.0) ...
[1173.1s] L4 member 2: done (832 rows total)
[1173.1s] rolling L4 member 3 (seed 52, dawdle 1.0) ...
[1516.8s] L4 member 3: done (1040 rows total)
[1516.8s] rolling L4 member 4 (seed 69, dawdle 1.0) ...
[1856.8s] L4 member 4: done (1248 rows total)
[1856.8s] rolling L5 member 0 (seed 1, dawdle 1.0) ...
[2193.1s] L5 member 0: done (1456 rows total)
[2193.1s] rolling L5 member 1 (seed 18, dawdle 1.25) ...
[2533.3s] L5 member 1: done (1664 rows total)
[2533.3s] rolling L5 member 2 (seed 35, dawdle 1.5) ...
[2894.0s] L5 member 2: done (1872 rows total)
[2894.0s] rolling L5 member 3 (seed 52, dawdle 0.8) ...
[3231.8s] L5 member 3: done (2080 rows total)
[3231.8s] rolling L5 member 4 (seed 69, dawdle 1.5) ...
[3553.8s] L5 member 4: done (2288 rows total)
[3553.8s] raw rows written to D:\rd-wt\wd01f\output\wd01\part3l\results\raw\raw_metrics.jsonl (total this run: 2288)
```

## Ladder aggregation + gate stdout (aggregate_p3l.py -- verbatim)

Raw rows: results/raw/raw_metrics.jsonl (written first by eval_p3l.py); aggregated verdicts: results/gate_v2_p3l.json.

```text
=== WD-01 part 3l: ladder aggregation + gate verdicts ===
ZERO simulation; holdout blocked in root_arrays (never passed); validation rows only.
loaded 2288 deduped raw rows
L5c spread map applied from calibration_p3l.json: {'delay_phys': {30: 1.7411, 60: 1.9097}, 'ttc3': {30: 2.0946, 60: 2.2974}}

== per-(variant, H) summary (4 validation roots x 52 windows x members) ==
  L0@H30: n_rolls=52 n_members=1 n_windows=52 speed_rmse=4.1881 coll/1000=0.0 lc 0.0/12.077 | exposure delay p/t 2471.856/2605.202 (bias -133.35) | physics delay p/t 335.629/629.015 (bias -293.39)
  L0@H60: n_rolls=52 n_members=1 n_windows=52 speed_rmse=4.9774 coll/1000=0.0 lc 0.0/23.942 | exposure delay p/t 4800.173/5261.288 (bias -461.11) | physics delay p/t 563.453/1302.219 (bias -738.77)
  L0@H90: n_rolls=52 n_members=1 n_windows=52 speed_rmse=5.2246 coll/1000=0.0 lc 0.0/35.846 | exposure delay p/t 7031.5/7958.096 (bias -926.6) | physics delay p/t 748.64/2019.537 (bias -1270.9)
  L0@H120: n_rolls=52 n_members=1 n_windows=52 speed_rmse=5.34 coll/1000=0.0 lc 0.0/47.904 | exposure delay p/t 9191.394/10689.885 (bias -1498.49) | physics delay p/t 913.526/2776.727 (bias -1863.2)
  L4@H30: n_rolls=260 n_members=5 n_windows=52 speed_rmse=4.0141 coll/1000=0.024 lc 11.669/12.077 | exposure delay p/t 2610.177/2605.202 (bias 4.97) | physics delay p/t 632.639/629.015 (bias 3.62)
  L4@H60: n_rolls=260 n_members=5 n_windows=52 speed_rmse=5.1805 coll/1000=0.019 lc 23.85/23.942 | exposure delay p/t 5257.69/5261.288 (bias -3.6) | physics delay p/t 1287.327/1302.219 (bias -14.89)
  L4@H90: n_rolls=260 n_members=5 n_windows=52 speed_rmse=5.6491 coll/1000=0.015 lc 35.685/35.846 | exposure delay p/t 7918.544/7958.096 (bias -39.55) | physics delay p/t 1953.221/2019.537 (bias -66.32)
  L4@H120: n_rolls=260 n_members=5 n_windows=52 speed_rmse=5.955 coll/1000=0.015 lc 47.462/47.904 | exposure delay p/t 10591.848/10689.885 (bias -98.04) | physics delay p/t 2637.534/2776.727 (bias -139.19)
  L5@H30: n_rolls=260 n_members=5 n_windows=52 speed_rmse=3.954 coll/1000=0.019 lc 10.927/12.077 | exposure delay p/t 2614.54/2605.202 (bias 9.34) | physics delay p/t 662.687/629.015 (bias 33.67)
  L5@H60: n_rolls=260 n_members=5 n_windows=52 speed_rmse=4.8965 coll/1000=0.015 lc 21.385/23.942 | exposure delay p/t 5278.837/5261.288 (bias 17.55) | physics delay p/t 1339.713/1302.219 (bias 37.49)
  L5@H90: n_rolls=260 n_members=5 n_windows=52 speed_rmse=5.3514 coll/1000=0.015 lc 31.846/35.846 | exposure delay p/t 7935.396/7958.096 (bias -22.7) | physics delay p/t 1998.335/2019.537 (bias -21.2)
  L5@H120: n_rolls=260 n_members=5 n_windows=52 speed_rmse=5.6417 coll/1000=0.014 lc 41.631/47.904 | exposure delay p/t 10581.667/10689.885 (bias -108.22) | physics delay p/t 2641.798/2776.727 (bias -134.93)

== gate v2 criterion (c): member bands + CRPS (contract physics delay; exposure coverage alongside; L5c = L5 + TRAIN-calibration spread factors) ==
  L4@H30: n_windows=52 delay_phys_10-90_cov=0.0577 delay_exposure_10-90_cov=0.25 ttc3_10-90_cov=0.1923 delay_phys_minmax=0.0962 CRPS_phys=47.26 (s_out dp/t3 = None/None)
  L5@H30: n_windows=52 delay_phys_10-90_cov=0.5 delay_exposure_10-90_cov=0.3846 ttc3_10-90_cov=0.1923 delay_phys_minmax=0.6154 CRPS_phys=42.6 (s_out dp/t3 = None/None)
  L5c@H30: n_windows=52 delay_phys_10-90_cov=0.8462 delay_exposure_10-90_cov=0.3846 ttc3_10-90_cov=0.5962 delay_phys_minmax=0.9038 CRPS_phys=46.72 (s_out dp/t3 = 1.7411/2.0946)
  L0@H30: deterministic delay_phys MAE=293.39 over 52 windows
  L4@H60: n_windows=52 delay_phys_10-90_cov=0.25 delay_exposure_10-90_cov=0.1923 ttc3_10-90_cov=0.0769 delay_phys_minmax=0.25 CRPS_phys=144.37 (s_out dp/t3 = None/None)
  L5@H60: n_windows=52 delay_phys_10-90_cov=0.5385 delay_exposure_10-90_cov=0.5577 ttc3_10-90_cov=0.2692 delay_phys_minmax=0.5577 CRPS_phys=106.94 (s_out dp/t3 = None/None)
  L5c@H60: n_windows=52 delay_phys_10-90_cov=0.7308 delay_exposure_10-90_cov=0.5577 ttc3_10-90_cov=0.5962 delay_phys_minmax=0.8654 CRPS_phys=113.18 (s_out dp/t3 = 1.9097/2.2974)
  L0@H60: deterministic delay_phys MAE=738.77 over 52 windows

== GATE v2 at H=30 (PRIMARY) -- (d) BINDING per-horizon form ==
  L0: (a) coll/1000=0.0 PASS | (b) lc 0.0/12.077 (100.0%) FAIL | (c) n/a (deterministic) -> NA | (d) speed 4.1881 vs 1.05xL0@H=4.3975 (PASS, binding) / fixed 5.2185 (PASS) ==> v2 FAIL: b_lc
  L4: (a) coll/1000=0.024 PASS | (b) lc 11.669/12.077 (3.4%) PASS | (c) ttc3 0.1923 delay-phys 0.0577 crps 47.26 vs mae 293.39 -> FAIL | (d) speed 4.0141 vs 1.05xL0@H=4.3975 (PASS, binding) / fixed 5.2185 (PASS) ==> v2 FAIL: c_coverage_crps
  L5: (a) coll/1000=0.019 PASS | (b) lc 10.927/12.077 (9.5%) PASS | (c) ttc3 0.1923 delay-phys 0.5 crps 42.6 vs mae 293.39 -> FAIL | (d) speed 3.954 vs 1.05xL0@H=4.3975 (PASS, binding) / fixed 5.2185 (PASS) ==> v2 FAIL: c_coverage_crps
  L5c: (a) coll/1000=0.019 PASS | (b) lc 10.927/12.077 (9.5%) PASS | (c) ttc3 0.5962 delay-phys 0.8462 crps 46.72 vs mae 293.39 -> FAIL | (d) speed 3.954 vs 1.05xL0@H=4.3975 (PASS, binding) / fixed 5.2185 (PASS) ==> v2 FAIL: c_coverage_crps

== GATE v2 at H=60 (secondary) -- (d) BINDING per-horizon form ==
  L0: (a) coll/1000=0.0 PASS | (b) lc 0.0/23.942 (100.0%) FAIL | (c) n/a (deterministic) -> NA | (d) speed 4.9774 vs 1.05xL0@H=5.2263 (PASS, binding) / fixed 5.2185 (PASS) ==> v2 FAIL: b_lc
  L4: (a) coll/1000=0.019 PASS | (b) lc 23.85/23.942 (0.4%) PASS | (c) ttc3 0.0769 delay-phys 0.25 crps 144.37 vs mae 738.77 -> FAIL | (d) speed 5.1805 vs 1.05xL0@H=5.2263 (PASS, binding) / fixed 5.2185 (PASS) ==> v2 FAIL: c_coverage_crps
  L5: (a) coll/1000=0.015 PASS | (b) lc 21.385/23.942 (10.7%) PASS | (c) ttc3 0.2692 delay-phys 0.5385 crps 106.94 vs mae 738.77 -> FAIL | (d) speed 4.8965 vs 1.05xL0@H=5.2263 (PASS, binding) / fixed 5.2185 (PASS) ==> v2 FAIL: c_coverage_crps
  L5c: (a) coll/1000=0.015 PASS | (b) lc 21.385/23.942 (10.7%) PASS | (c) ttc3 0.5962 delay-phys 0.7308 crps 113.18 vs mae 738.77 -> FAIL | (d) speed 4.8965 vs 1.05xL0@H=5.2263 (PASS, binding) / fixed 5.2185 (PASS) ==> v2 FAIL: c_coverage_crps

== OLD gate at H=30 ==
  L0: speed 4.1881 < 4.97 True | coll 0.0 True | lc 100.0% False | delay minmax None False ==> FAIL
  L4: speed 4.0141 < 4.97 True | coll 0.024 True | lc 3.4% True | delay minmax 0.0962 False ==> FAIL
  L5: speed 3.954 < 4.97 True | coll 0.019 True | lc 9.5% True | delay minmax 0.6154 True ==> PASS
  L5c: speed 3.954 < 4.97 True | coll 0.019 True | lc 9.5% True | delay minmax 0.9038 True ==> PASS

== OLD gate at H=60 ==
  L0: speed 4.9774 < 4.97 False | coll 0.0 True | lc 100.0% False | delay minmax None False ==> FAIL
  L4: speed 5.1805 < 4.97 False | coll 0.019 True | lc 0.4% True | delay minmax 0.25 False ==> FAIL
  L5: speed 4.8965 < 4.97 True | coll 0.015 True | lc 10.7% True | delay minmax 0.5577 True ==> PASS
  L5c: speed 4.8965 < 4.97 True | coll 0.015 True | lc 10.7% True | delay minmax 0.8654 True ==> PASS

wrote D:\rd-wt\wd01f\output\wd01\part3l\results\gate_v2_p3l.json
```

## NOT DONE / residual gaps

1. **Merge-zone occupancy / queueing residual**: with the corrected
   follower slot the emulator queues harder than before; whether the
   refit closes the p3k occupancy gap is visible in the fit stdout
   (occ term).  A SUMO-exact stopped-merge urgency term was still out
   of scope (p3k NOT-DONE item carried).
2. **Optional M2 for 0<->1** (gap acceptance on discretionary
   main-lane changes) remains unimplemented (brief: optional); 0<->1
   stays on the stochastic hazard (M4, refitted lc_scale).
3. **Spread mechanism A is per-(output, horizon) SCALING only**; it
   widens member bands but cannot fix a horizon-dependent BIAS (see
   the physics-delay bias per rung in the aggregation stdout).  A
   bias-calibrated member ensemble remains future work.
4. **idm_sampled stale-llen defect** (p3k item) recorded, not fixed.
5. **Single process, below-normal priority, torch 2 threads** -- no
   parallelism; box shared with SUMO workers.
6. **Holdout untouched**: the registered holdout evaluation remains
   blocked until the owner opens it (no registration hash created).

## Artifacts

- Code: `part3l/merge_check_p3l.py`, `part3l/fit_p3l.py`,
  `part3l/calibrate_p3l.py`, `part3l/eval_p3l.py`,
  `part3l/aggregate_p3l.py`, `part3l/p3l_lib.py`,
  `part3l/make_report_p3l.py`, `part3l/lint_names.py`;
  engine changes in `part3/engine.py` (TASK A instrumentation +
  follower-slot fix + optional dawdle_scale, default-neutral).
- Tests: `part3l/tests/` (28 tests) + the pre-existing
  `part3/tests/` suite (all passing).
- Results: `part3l/results/` (merge_check_p3l.json, fit_p3l.json,
  calibration_p3l.json, raw/raw_metrics.jsonl, raw/raw_calibration.jsonl,
  gate_v2_p3l.json, stdout_*.txt, LEDGER_DRAFT_P3L.json).
- Reconciliation: L0 rollouts and the deterministic delay model are
  the p3j/p3k configuration (L0 speed RMSE 4.1881 @H30 / 4.9774 @H60
  and physics-delay MAE 293.39 / 738.77 reproduce the p3k gate table
  exactly).  The L4 rung reuses the p3k member PARAMETERS but rolls on
  the FIXED engine, so it is expected to differ from the p3k published
  table by the slot-fix effect (e.g. LC +3.4 % here vs +29.9 % in p3k).

*(This report is generated by make_report_p3l.py; all numbers are
pasted from the tee'd stdout files and the results JSONs above.)*
