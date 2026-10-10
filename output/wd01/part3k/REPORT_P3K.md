# WD-01 Part 3K — SUMO-like merge dynamics in the structured native Part A emulator (kind `idm_merge`), refit + re-gate

****Holdout NOT accessed.**  No holdout root was loaded (the holdout block inside `root_arrays` stays authoritative); no registration hash was created or passed anywhere in part 3k.**

ZERO simulation: no SUMO/TraCI import anywhere; the only dynamics is the
part-3 rollout engine (pure torch).  TRAIN roots for fitting,
VALIDATION roots for reporting.

## What was built

1. **Data-root override** (`WD01_DATA_ROOT`): untracked
   data/weights/results live in a populated copy of the output tree;
   `pdl.py` (partA_data) resolves `DATA`/`PART1`, `p3lib.py` resolves
   `RESULTS`/`WEIGHTS`, and `engine.py` resolves the part3d weights dir
   through it.  UNSET (default) keeps the in-repo layout byte-identical
   to pre-3k behaviour (tested: `partA_data/tests/test_data_root.py`,
   `part3/tests/test_p3_merge.py`).  All part-3k runs used
   `WD01_DATA_ROOT=D:/rd-wt/wd01/output/wd01` (read-only).
2. **Engine kind `idm_merge`** in `part3/engine.py` = `idm_sampled`
   (unchanged branch) plus SUMO-like merge dynamics:
   - **M1** accel-lane end as a virtual standing obstacle at
     s = 454.10 m (IDM gap to the lane end) so unmerged vehicles
     decelerate and queue;
   - **M2** mandatory gap acceptance accel->main_right inside the merge
     zone (s >= 378.66 m): merge iff gap-to-target-leader >= minGap +
     max(0, v*tau_accept) AND the target-lane follower's required
     deceleration <= b_accept; fires as soon as acceptable, otherwise
     the vehicle keeps waiting (the `idm_sampled` stochastic
     accel->main_right merge is removed);
   - **M3** cooperative slow-down: a main_right follower with d in
     [-100, 100] m whose leader-to-be (nearest accel vehicle ahead in d,
     within 100 m) is still waiting adapts its IDM leader to it with
     per-vType probability lcCooperative_prior x p_coop_scale
     (prior 0.3/0.6/1.0);
   - **M4** discretionary 0<->1 changes stay on the existing LC hazard
     (lc_scale).
   Unit tests: `part3/tests/test_p3_merge.py` (lane-end obstacle stops a
   vehicle; synthetic gaps rejected/accepted; cooperative slow-down
   triggers; merge keeps d and survives `_drop_exits`).
   NOTE: `idm_sampled` itself is untouched, but the review found it
   leaves a stale accel-lane `llen`/`vcap` after a merge (a merged
   vehicle is then dropped by `_drop_exits` as if it had exited);
   `idm_merge` refreshes all lane attributes after a lane change.  The
   stale-`llen` defect in `idm_sampled` is recorded as a deviation, NOT
   fixed (that would change published L3 numbers).

## Fit (TRAIN only)

Protocol, grids, targets and the selected parameters are in the fit
stdout below.  Selected: merge_params={'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0}, lc_scale=0.002
(member 0; members 1-4 are bootstrap refits over the 4 train roots,
seeds m*17+1).  Fitted-on record: {'roots': ['cv17', 'cv19_4107', 'cv19_4108', 'cv20_7001m2000'], 't0s': [120.0, 300.0], 'H': 60.0, 'note': 'TRAIN only; holdout never accessed'}.

## Gate verdicts (summary; full tables in the aggregation stdout)

- Gate v2 at H=30 (PRIMARY per
  `part3f/VALIDATION_GATE_V2_AMENDMENT_1.md`):
  - L0: (a) PASS (coll/1000=0.0) | (b) FAIL (100.0%) | (c) NA (deterministic) | (d) PASS (RMSE=4.1881) ==> FAIL: b_lc
  - L3: (a) PASS (coll/1000=0.013) | (b) FAIL (24.4%) | (c) FAIL | (d) PASS (RMSE=3.8008) ==> FAIL: b_lc,c_coverage_crps
  - L4: (a) PASS (coll/1000=0.071) | (b) FAIL (29.9%) | (c) FAIL | (d) PASS (RMSE=4.4044) ==> FAIL: b_lc,c_coverage_crps
- Gate v2 at H=60 (secondary): L0/L3/L4 verdicts in the stdout below.
- Old gate at H=30/H=60: reported alongside (stdout below).
- Criterion (c) uses the CONTRACT PHYSICS DELAY (dt - ds/limit, the
  p3j/p3k decompose model); exposure-based coverage is reported
  alongside for continuity.

## Fit stdout (fit_p3k.py, TRAIN only -- verbatim)



```text
priority below-normal: False
=== WD-01 part 3k: idm_merge parameter fit (TRAIN only) ===
ZERO simulation; holdout blocked in root_arrays (never passed).
device=cpu torch_threads=2
train fit roots (first 4 sorted): ['cv17', 'cv19_4107', 'cv19_4108', 'cv20_7001m2000']
eval windows: t0 (120.0, 300.0) s, H=60 s (K=120 steps of 0.5 s); validation roots ['cv20_7005m2000', 'cv20_7006m2000', 'cv30_7305m2700', 'seed_7535m2400'] NOT used here
fit targets: T1 zone occupancy, T2 accel speed profile (50 m bins, n>=60), T3 merge-point d, T4 main_right speed in [-100,100] m, T5 physics delay, T6 LC count
grids: tau_accept (0.7, 1.0, 1.4, 1.9) x b_accept (2.0, 3.0, 4.5, 6.5); p_coop_scale (0.0, 0.25, 0.5, 0.75, 1.1) x lc_scale (0.002, 0.003, 0.005)
[   4.7s] BEFORE: rolling L3 (idm_sampled, member 0, lc_scale=0.003) ...

BEFORE (L3 = idm_sampled, member 0, lc_scale=0.003): stage1 loss=4.5020 {'occ': 0.4024443788069275, 'prof': 0.7967072320468978, 'merge': 4.0} | stage2 loss=5.0877 {'mr': 2.002148363156379, 'delay': 1.599810555207017, 'lc': 1.4857264081115922}

  -- BEFORE L3 (idm_sampled) (per eval window) --
  root             t0   occ_p occ_t | accel v (m/s) by d-bin -50..300 pred | truth | merge_d p/t   mr_spd p/t    delay_p  delay_t  lc p/t
  cv17             120     69   142 | 17.0 19.4 19.3 19.4 19.1 19.1 18.9 18.2 |  7.1 17.8 19.0 19.4 19.4 19.0 18.5 17.1 |    n/a/ -58.4   27.2/ 17.1     728.7    743.2   54/ 20
  cv17             300     90   225 | 12.6 20.1 20.3 20.7 20.9 20.8 20.7 20.9 |  3.2 17.9 20.4 21.2 21.6 21.7 21.9 21.6 |    n/a/ -56.2   10.4/  2.7    1415.3   2078.1   70/ 21
  cv19_4107        120     47    86 | 21.2 20.4 20.3 20.2 20.1 20.0 19.5 17.7 | 12.8 18.3 19.8 20.3 20.5 20.3 19.8 17.1 |    n/a/ -18.9   28.5/ 24.6     420.2    419.4   41/ 38
  cv19_4107        300     52   432 | 19.7 20.4 20.2 20.2 20.2 20.1 19.6 17.6 |  3.5 17.1 19.9 20.8 20.9 20.7 20.2 17.9 |    n/a/ -40.9   27.2/ 21.0     669.5    746.7   42/ 22
  cv19_4108        120     47   115 | 20.4 18.9 18.6 18.5 18.7 18.6 18.3 18.0 |  8.5 18.2 19.1 19.4 19.4 19.0 18.5 17.1 |    n/a/ -50.2   28.6/ 19.4     403.7    454.5   34/ 39
  cv19_4108        300     24   100 | 21.9 20.6 20.6 20.8 20.8 20.4 20.7 21.0 | 15.4 19.6 20.8 21.3 21.6 21.8 21.9 21.6 |    n/a/ -52.3   26.6/ 23.6     455.0    407.2   36/ 30
  cv20_7001m2000   120     40   161 | 21.9 20.6 20.6 20.5 20.5 20.1 20.1 20.3 | 10.2 18.3 19.8 20.5 20.8 20.8 21.1 21.2 |    n/a/ -31.6   28.0/ 23.7     550.9    529.6   42/ 16
  cv20_7001m2000   300     70   108 | 21.9 20.6 20.5 20.4 20.3 20.4 20.4 20.0 | 14.0 18.3 19.4 19.8 20.0 20.1 20.1 19.4 |    n/a/ -51.6   17.0/  8.4     644.3    891.6   39/ 34
[  51.7s] STAGE 1: 16 acceptance combos ...
[ 103.7s] stage1 1/16 {'tau_accept': 0.7, 'b_accept': 2.0, 'p_coop_scale': 0.0} -> loss=0.9799 {'occ': 0.7954678136123109, 'prof': 0.4632015200692287, 'merge': 0.12649682006966165}
[ 157.4s] stage1 2/16 {'tau_accept': 0.7, 'b_accept': 3.0, 'p_coop_scale': 0.0} -> loss=0.9799 {'occ': 0.7954678136123109, 'prof': 0.4632015200692287, 'merge': 0.12649682006966165}
[ 211.5s] stage1 3/16 {'tau_accept': 0.7, 'b_accept': 4.5, 'p_coop_scale': 0.0} -> loss=0.9799 {'occ': 0.7954678136123109, 'prof': 0.4632015200692287, 'merge': 0.12649682006966165}
[ 265.3s] stage1 4/16 {'tau_accept': 0.7, 'b_accept': 6.5, 'p_coop_scale': 0.0} -> loss=0.9799 {'occ': 0.7954678136123109, 'prof': 0.4632015200692287, 'merge': 0.12649682006966165}
[ 319.8s] stage1 5/16 {'tau_accept': 1.0, 'b_accept': 2.0, 'p_coop_scale': 0.0} -> loss=0.9190 {'occ': 0.7490950402528771, 'prof': 0.44455123577090344, 'merge': 0.1143497583990156}
[ 373.5s] stage1 6/16 {'tau_accept': 1.0, 'b_accept': 3.0, 'p_coop_scale': 0.0} -> loss=0.9190 {'occ': 0.7490950402528771, 'prof': 0.44455123577090344, 'merge': 0.1143497583990156}
[ 426.7s] stage1 7/16 {'tau_accept': 1.0, 'b_accept': 4.5, 'p_coop_scale': 0.0} -> loss=0.9190 {'occ': 0.7490950402528771, 'prof': 0.44455123577090344, 'merge': 0.1143497583990156}
[ 479.2s] stage1 8/16 {'tau_accept': 1.0, 'b_accept': 6.5, 'p_coop_scale': 0.0} -> loss=0.9190 {'occ': 0.7490950402528771, 'prof': 0.44455123577090344, 'merge': 0.1143497583990156}
[ 530.9s] stage1 9/16 {'tau_accept': 1.4, 'b_accept': 2.0, 'p_coop_scale': 0.0} -> loss=0.8628 {'occ': 0.7210603113058797, 'prof': 0.4314496136173575, 'merge': 0.08779445258375992}
[ 585.0s] stage1 10/16 {'tau_accept': 1.4, 'b_accept': 3.0, 'p_coop_scale': 0.0} -> loss=0.8628 {'occ': 0.7210603113058797, 'prof': 0.4314496136173575, 'merge': 0.08779445258375992}
[ 638.2s] stage1 11/16 {'tau_accept': 1.4, 'b_accept': 4.5, 'p_coop_scale': 0.0} -> loss=0.8628 {'occ': 0.7210603113058797, 'prof': 0.4314496136173575, 'merge': 0.08779445258375992}
[ 692.9s] stage1 12/16 {'tau_accept': 1.4, 'b_accept': 6.5, 'p_coop_scale': 0.0} -> loss=0.8628 {'occ': 0.7210603113058797, 'prof': 0.4314496136173575, 'merge': 0.08779445258375992}
[ 745.6s] stage1 13/16 {'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0} -> loss=0.7575 {'occ': 0.6404588678940164, 'prof': 0.42595676875318145, 'merge': 0.06381026683490032}
[ 801.6s] stage1 14/16 {'tau_accept': 1.9, 'b_accept': 3.0, 'p_coop_scale': 0.0} -> loss=0.7575 {'occ': 0.6404588678940164, 'prof': 0.42595676875318145, 'merge': 0.06381026683490032}
[ 860.7s] stage1 15/16 {'tau_accept': 1.9, 'b_accept': 4.5, 'p_coop_scale': 0.0} -> loss=0.7575 {'occ': 0.6404588678940164, 'prof': 0.42595676875318145, 'merge': 0.06381026683490032}
[ 914.9s] stage1 16/16 {'tau_accept': 1.9, 'b_accept': 6.5, 'p_coop_scale': 0.0} -> loss=0.7575 {'occ': 0.6404588678940164, 'prof': 0.42595676875318145, 'merge': 0.06381026683490032}

STAGE 1 winner: {'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0} loss=0.7575 {'occ': 0.6404588678940164, 'prof': 0.42595676875318145, 'merge': 0.06381026683490032}
[ 914.9s] STAGE 2: 15 combos ...
[ 973.1s] stage2 1/15 {'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0} lc_scale=0.002 -> loss=1.4497 {'mr': 0.2358170234918528, 'delay': 0.27975242321859367, 'lc': 0.9340979153924461}
[1032.8s] stage2 2/15 {'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0} lc_scale=0.003 -> loss=3.6778 {'mr': 0.2321683904632429, 'delay': 0.2653334297926724, 'lc': 3.180294875097804}
[1087.7s] stage2 3/15 {'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0} lc_scale=0.005 -> loss=9.6413 {'mr': 0.2294145417016351, 'delay': 0.3403547828153073, 'lc': 9.07151550130281}
[1142.1s] stage2 4/15 {'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.25} lc_scale=0.002 -> loss=1.6287 {'mr': 0.2639859298360595, 'delay': 0.290350953247895, 'lc': 1.0743673957294304}
[1196.9s] stage2 5/15 {'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.25} lc_scale=0.003 -> loss=3.6138 {'mr': 0.2857013594558636, 'delay': 0.27979964886602743, 'lc': 3.048263643005204}
[1249.5s] stage2 6/15 {'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.25} lc_scale=0.005 -> loss=9.8602 {'mr': 0.25207890766080326, 'delay': 0.3312228990040728, 'lc': 9.27691809350403}
[1303.7s] stage2 7/15 {'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.5} lc_scale=0.002 -> loss=1.5722 {'mr': 0.28552532680646414, 'delay': 0.30421871314653554, 'lc': 0.9824361461826433}
[1356.7s] stage2 8/15 {'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.5} lc_scale=0.003 -> loss=3.4736 {'mr': 0.27643983319042875, 'delay': 0.28088543262478866, 'lc': 2.9162660559792872}
[1411.3s] stage2 9/15 {'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.5} lc_scale=0.005 -> loss=9.6823 {'mr': 0.31468669508418035, 'delay': 0.36995751294479956, 'lc': 8.9976491641767}
[1466.3s] stage2 10/15 {'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.75} lc_scale=0.002 -> loss=1.6659 {'mr': 0.2890474173860707, 'delay': 0.2862038892278884, 'lc': 1.0906813778742672}
[1522.0s] stage2 11/15 {'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.75} lc_scale=0.003 -> loss=3.9552 {'mr': 0.2756256979645583, 'delay': 0.27148463760877717, 'lc': 3.4081124670385483}
[1576.0s] stage2 12/15 {'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.75} lc_scale=0.005 -> loss=8.6694 {'mr': 0.2895515390665302, 'delay': 0.33907570754538346, 'lc': 8.04075072870379}
[1632.2s] stage2 13/15 {'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 1.1} lc_scale=0.002 -> loss=1.4840 {'mr': 0.2784893219024722, 'delay': 0.29094378647664687, 'lc': 0.9145950678838721}
[1687.0s] stage2 14/15 {'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 1.1} lc_scale=0.003 -> loss=3.8008 {'mr': 0.29060410034728495, 'delay': 0.2832911104418243, 'lc': 3.226940152788818}
[1742.4s] stage2 15/15 {'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 1.1} lc_scale=0.005 -> loss=9.6853 {'mr': 0.2999982169240667, 'delay': 0.38982157557615166, 'lc': 8.99552885001285}

STAGE 2 winner: merge_params={'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0} lc_scale=0.002 loss=1.4497 {'mr': 0.2358170234918528, 'delay': 0.27975242321859367, 'lc': 0.9340979153924461}

AFTER (L4 = idm_merge fitted): stage1 loss=0.7823 {'occ': 0.6826002087869991, 'prof': 0.4403275251858006, 'merge': 0.04468555114918211} | stage2 loss=1.4497 {'mr': 0.2358170234918528, 'delay': 0.27975242321859367, 'lc': 0.9340979153924461}

  -- AFTER L4 (idm_merge fitted) (per eval window) --
  root             t0   occ_p occ_t | accel v (m/s) by d-bin -50..300 pred | truth | merge_d p/t   mr_spd p/t    delay_p  delay_t  lc p/t
  cv17             120     14   142 | 17.2 18.7 19.5 19.6 19.5 19.7 19.4 18.7 |  7.1 17.8 19.0 19.4 19.4 19.0 18.5 17.1 |  -34.9/ -58.4   17.9/ 17.1     842.6    743.2   40/ 20
  cv17             300     67   225 | 15.3 18.9 20.1 20.6 20.8 21.1 21.1 21.4 |  3.2 17.9 20.4 21.2 21.6 21.7 21.9 21.6 |  -58.6/ -56.2    4.9/  2.7    1880.3   2078.1   56/ 21
  cv19_4107        120     12    86 | 17.0 18.4 19.4 19.8 20.1 20.0 19.6 18.0 | 12.8 18.3 19.8 20.3 20.5 20.3 19.8 17.1 |  -12.6/ -18.9   23.3/ 24.6     450.8    419.4   30/ 38
  cv19_4107        300     34   432 | 15.7 18.1 19.2 19.7 19.8 19.6 19.1 16.9 |  3.5 17.1 19.9 20.8 20.9 20.7 20.2 17.9 |  -24.5/ -40.9   19.8/ 21.0     766.7    746.7   51/ 22
  cv19_4108        120     40   115 | 16.0 18.6 19.3 19.7 19.5 19.4 19.4 18.6 |  8.5 18.2 19.1 19.4 19.4 19.0 18.5 17.1 |  -43.3/ -50.2   19.8/ 19.4     522.6    454.5   35/ 39
  cv19_4108        300     13   100 | 17.1 18.9 19.7 20.5 20.7 20.7 21.2 21.3 | 15.4 19.6 20.8 21.3 21.6 21.8 21.9 21.6 |  -33.7/ -52.3   19.9/ 23.6     576.9    407.2   35/ 30
  cv20_7001m2000   120      6   161 | 17.0 18.2 19.3 19.8 20.1 20.1 20.4 20.4 | 10.2 18.3 19.8 20.5 20.8 20.8 21.1 21.2 |  -22.0/ -31.6   24.8/ 23.7     569.5    529.6   36/ 16
  cv20_7001m2000   300     35   108 | 16.3 18.1 19.2 19.6 19.7 19.8 19.9 19.8 | 14.0 18.3 19.4 19.8 20.0 20.1 20.1 19.4 |  -25.9/ -51.6   13.3/  8.4     827.3    891.6   53/ 34
[1796.3s] BOOTSTRAP members 1..4 ...
[2132.0s] bootstrap member 1: resample=['cv17', 'cv19_4107', 'cv19_4108', 'cv20_7001m2000'] -> loss=1.5555 (s1 0.7599 s2 0.7956) params={'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0}
[2478.4s] bootstrap member 2: resample=['cv17', 'cv19_4107'] -> loss=1.5330 (s1 0.8338 s2 0.6992) params={'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0}
[2802.4s] bootstrap member 3: resample=['cv19_4107', 'cv20_7001m2000'] -> loss=1.7093 (s1 0.8317 s2 0.8776) params={'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.25}
[3142.0s] bootstrap member 4: resample=['cv19_4108', 'cv20_7001m2000'] -> loss=1.5811 (s1 0.6892 s2 0.8920) params={'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0}

selected member parameters (member 0 = main fit):
  member 0: seed=1 lc_scale=0.002 merge_params={'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0}
  member 1: seed=18 lc_scale=0.002 merge_params={'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0}
  member 2: seed=35 lc_scale=0.002 merge_params={'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0}
  member 3: seed=52 lc_scale=0.002 merge_params={'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.25}
  member 4: seed=69 lc_scale=0.002 merge_params={'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0}
[3142.0s] wrote D:\rd-wt\wd01e\output\wd01\part3k\results\fit_p3k.json and D:\rd-wt\wd01e\output\wd01\part3k\results\stdout_fit.txt

DONE part 3k fit.
```

## Validation decomposition stdout (decompose_p3k.py, L0/L3/L4, 60 s grid, H=60 -- verbatim)

Top cells before/after and the accel speed profile are in sections C/D/E.

```text
priority below-normal: False
=== WD-01 part 3k: delay decomposition L0/L3/L4 (validation) ===
ZERO simulation; holdout blocked in root_arrays (never passed).
device=cpu torch_threads=2
variants: L0 = kind 'idm_normal'; L3 = kind 'idm_sampled' member_seed=1 lc_scale=0.003; L4 = kind 'idm_merge' member_seed=1 lc_scale=0.002 merge_params={'tau_accept': 1.9, 'b_accept': 2.0, 'p_coop_scale': 0.0}
H=60 s (K=120 steps of 0.5 s); validation roots ['cv20_7005m2000', 'cv20_7006m2000', 'cv30_7305m2700', 'seed_7535m2400']; grid t0 every 60 s (120, 180, 240, 300, 360, 420, 480)
d = OFFSETS[lane] - s, OFFSETS=(2645.67, 2644.22, 378.66); segments [('downstream', -inf, 0.0), ('merge_edge', 0.0, 75.0), ('rsu_window', 75.0, 800.0), ('upstream', 800.0, inf)]
limits via engine.lane_speed_limits: e.g. limit(0,100)=31.29 limit(2,0)=22.22 limit(2,440)=31.29
[   6.0s] L0 cv20_7005m2000 t0=120: n_p=7988 n_t=8744 expo_def=-378.0 delay_def=-483.1
[  13.4s] L3 cv20_7005m2000 t0=120: n_p=8165 n_t=8744 expo_def=-289.5 delay_def=-171.6
[  22.5s] L4 cv20_7005m2000 t0=120: n_p=8841 n_t=8744 expo_def=48.5 delay_def=63.7
[  27.0s] L0 cv20_7005m2000 t0=180: n_p=8365 n_t=8869 expo_def=-252.0 delay_def=-459.0
[  35.6s] L3 cv20_7005m2000 t0=180: n_p=8538 n_t=8869 expo_def=-165.5 delay_def=-121.0
[  43.2s] L4 cv20_7005m2000 t0=180: n_p=8922 n_t=8869 expo_def=26.5 delay_def=-3.9
[  47.0s] L0 cv20_7005m2000 t0=240: n_p=8060 n_t=8816 expo_def=-378.0 delay_def=-529.2
[  54.0s] L3 cv20_7005m2000 t0=240: n_p=8316 n_t=8816 expo_def=-250.0 delay_def=-173.5
[  61.7s] L4 cv20_7005m2000 t0=240: n_p=8726 n_t=8816 expo_def=-45.0 delay_def=-59.6
[  65.7s] L0 cv20_7005m2000 t0=300: n_p=8731 n_t=9811 expo_def=-540.0 delay_def=-763.6
[  73.1s] L3 cv20_7005m2000 t0=300: n_p=8937 n_t=9811 expo_def=-437.0 delay_def=-399.5
[  84.6s] L4 cv20_7005m2000 t0=300: n_p=9670 n_t=9811 expo_def=-70.5 delay_def=-141.2
[  90.1s] L0 cv20_7005m2000 t0=360: n_p=10221 n_t=10955 expo_def=-367.0 delay_def=-524.8
[ 100.3s] L3 cv20_7005m2000 t0=360: n_p=10483 n_t=10955 expo_def=-236.0 delay_def=-11.2
[ 110.4s] L4 cv20_7005m2000 t0=360: n_p=11076 n_t=10955 expo_def=60.5 delay_def=167.6
[ 114.4s] L0 cv20_7005m2000 t0=420: n_p=8892 n_t=10091 expo_def=-599.5 delay_def=-790.6
[ 122.2s] L3 cv20_7005m2000 t0=420: n_p=9159 n_t=10091 expo_def=-466.0 delay_def=-434.4
[ 130.1s] L4 cv20_7005m2000 t0=420: n_p=10077 n_t=10091 expo_def=-7.0 delay_def=-91.4
[ 133.5s] L0 cv20_7005m2000 t0=480: n_p=7304 n_t=8799 expo_def=-747.5 delay_def=-789.9
[ 141.5s] L3 cv20_7005m2000 t0=480: n_p=7522 n_t=8799 expo_def=-638.5 delay_def=-492.5
[ 150.1s] L4 cv20_7005m2000 t0=480: n_p=8751 n_t=8799 expo_def=-24.0 delay_def=11.8
[ 154.5s] L0 cv20_7006m2000 t0=120: n_p=6685 n_t=7267 expo_def=-291.0 delay_def=-283.9
[ 160.5s] L3 cv20_7006m2000 t0=120: n_p=6871 n_t=7267 expo_def=-198.0 delay_def=-21.7
[ 167.8s] L4 cv20_7006m2000 t0=120: n_p=7350 n_t=7267 expo_def=41.5 delay_def=70.3
[ 171.2s] L0 cv20_7006m2000 t0=180: n_p=7265 n_t=7792 expo_def=-263.5 delay_def=-286.5
[ 177.8s] L3 cv20_7006m2000 t0=180: n_p=7385 n_t=7792 expo_def=-203.5 delay_def=-37.8
[ 185.3s] L4 cv20_7006m2000 t0=180: n_p=7890 n_t=7792 expo_def=49.0 delay_def=49.8
[ 188.9s] L0 cv20_7006m2000 t0=240: n_p=6542 n_t=7967 expo_def=-712.5 delay_def=-724.4
[ 196.1s] L3 cv20_7006m2000 t0=240: n_p=6698 n_t=7967 expo_def=-634.5 delay_def=-486.9
[ 205.5s] L4 cv20_7006m2000 t0=240: n_p=8100 n_t=7967 expo_def=66.5 delay_def=35.4
[ 209.6s] L0 cv20_7006m2000 t0=300: n_p=6988 n_t=7936 expo_def=-474.0 delay_def=-545.2
[ 217.4s] L3 cv20_7006m2000 t0=300: n_p=7189 n_t=7936 expo_def=-373.5 delay_def=-265.3
[ 225.5s] L4 cv20_7006m2000 t0=300: n_p=7921 n_t=7936 expo_def=-7.5 delay_def=-17.2
[ 229.1s] L0 cv20_7006m2000 t0=360: n_p=7416 n_t=7818 expo_def=-201.0 delay_def=-309.4
[ 235.5s] L3 cv20_7006m2000 t0=360: n_p=7599 n_t=7818 expo_def=-109.5 delay_def=-5.7
[ 242.5s] L4 cv20_7006m2000 t0=360: n_p=7848 n_t=7818 expo_def=15.0 delay_def=70.8
[ 245.7s] L0 cv20_7006m2000 t0=420: n_p=7342 n_t=8408 expo_def=-533.0 delay_def=-674.8
[ 251.9s] L3 cv20_7006m2000 t0=420: n_p=7602 n_t=8408 expo_def=-403.0 delay_def=-370.6
[ 260.1s] L4 cv20_7006m2000 t0=420: n_p=8167 n_t=8408 expo_def=-120.5 delay_def=-208.1
[ 264.5s] L0 cv20_7006m2000 t0=480: n_p=7643 n_t=8582 expo_def=-469.5 delay_def=-496.9
[ 271.8s] L3 cv20_7006m2000 t0=480: n_p=7845 n_t=8582 expo_def=-368.5 delay_def=-154.7
[ 280.1s] L4 cv20_7006m2000 t0=480: n_p=8314 n_t=8582 expo_def=-134.0 delay_def=-61.5
[ 286.5s] L0 cv30_7305m2700 t0=120: n_p=10267 n_t=10789 expo_def=-261.0 delay_def=-514.0
[ 294.9s] L3 cv30_7305m2700 t0=120: n_p=10453 n_t=10789 expo_def=-168.0 delay_def=-49.9
[ 303.9s] L4 cv30_7305m2700 t0=120: n_p=10715 n_t=10789 expo_def=-37.0 delay_def=-25.4
[ 308.3s] L0 cv30_7305m2700 t0=180: n_p=10435 n_t=11374 expo_def=-469.5 delay_def=-754.1
[ 317.9s] L3 cv30_7305m2700 t0=180: n_p=10674 n_t=11374 expo_def=-350.0 delay_def=-294.0
[ 329.3s] L4 cv30_7305m2700 t0=180: n_p=11387 n_t=11374 expo_def=6.5 delay_def=-40.2
[ 333.9s] L0 cv30_7305m2700 t0=240: n_p=10827 n_t=11903 expo_def=-538.0 delay_def=-975.4
[ 343.6s] L3 cv30_7305m2700 t0=240: n_p=11108 n_t=11903 expo_def=-397.5 delay_def=-444.7
[ 353.8s] L4 cv30_7305m2700 t0=240: n_p=11639 n_t=11903 expo_def=-132.0 delay_def=-269.0
[ 358.8s] L0 cv30_7305m2700 t0=300: n_p=11107 n_t=12347 expo_def=-620.0 delay_def=-1139.7
[ 368.0s] L3 cv30_7305m2700 t0=300: n_p=11574 n_t=12347 expo_def=-386.5 delay_def=-550.5
[ 379.3s] L4 cv30_7305m2700 t0=300: n_p=12334 n_t=12347 expo_def=-6.5 delay_def=-197.2
[ 385.9s] L0 cv30_7305m2700 t0=360: n_p=12272 n_t=13618 expo_def=-673.0 delay_def=-1434.3
[ 397.5s] L3 cv30_7305m2700 t0=360: n_p=12739 n_t=13618 expo_def=-439.5 delay_def=-800.8
[ 410.3s] L4 cv30_7305m2700 t0=360: n_p=13385 n_t=13618 expo_def=-116.5 delay_def=-435.8
[ 415.9s] L0 cv30_7305m2700 t0=420: n_p=13657 n_t=14743 expo_def=-543.0 delay_def=-1396.1
[ 426.4s] L3 cv30_7305m2700 t0=420: n_p=14058 n_t=14743 expo_def=-342.5 delay_def=-739.9
[ 440.0s] L4 cv30_7305m2700 t0=420: n_p=14607 n_t=14743 expo_def=-68.0 delay_def=-453.7
[ 447.4s] L0 cv30_7305m2700 t0=480: n_p=14126 n_t=15280 expo_def=-577.0 delay_def=-1444.4
[ 460.0s] L3 cv30_7305m2700 t0=480: n_p=14485 n_t=15280 expo_def=-397.5 delay_def=-765.7
[ 475.8s] L4 cv30_7305m2700 t0=480: n_p=15239 n_t=15280 expo_def=-20.5 delay_def=-373.5
[ 481.8s] L0 seed_7535m2400 t0=120: n_p=9533 n_t=9938 expo_def=-202.5 delay_def=-332.6
[ 490.6s] L3 seed_7535m2400 t0=120: n_p=9736 n_t=9938 expo_def=-101.0 delay_def=79.7
[ 500.8s] L4 seed_7535m2400 t0=120: n_p=9996 n_t=9938 expo_def=29.0 delay_def=113.9
[ 506.7s] L0 seed_7535m2400 t0=180: n_p=9741 n_t=10489 expo_def=-374.0 delay_def=-544.1
[ 516.3s] L3 seed_7535m2400 t0=180: n_p=9962 n_t=10489 expo_def=-263.5 delay_def=-76.8
[ 526.4s] L4 seed_7535m2400 t0=180: n_p=10581 n_t=10489 expo_def=46.0 delay_def=67.8
[ 531.7s] L0 seed_7535m2400 t0=240: n_p=10225 n_t=10730 expo_def=-252.5 delay_def=-528.5
[ 540.6s] L3 seed_7535m2400 t0=240: n_p=10460 n_t=10730 expo_def=-135.0 delay_def=-37.2
[ 549.2s] L4 seed_7535m2400 t0=240: n_p=10874 n_t=10730 expo_def=72.0 delay_def=133.6
[ 553.9s] L0 seed_7535m2400 t0=300: n_p=11237 n_t=12072 expo_def=-417.5 delay_def=-638.6
[ 564.6s] L3 seed_7535m2400 t0=300: n_p=11495 n_t=12072 expo_def=-288.5 delay_def=-110.8
[ 574.1s] L4 seed_7535m2400 t0=300: n_p=12072 n_t=12072 expo_def=0.0 delay_def=66.8
[ 578.2s] L0 seed_7535m2400 t0=360: n_p=11455 n_t=12362 expo_def=-453.5 delay_def=-819.6
[ 587.1s] L3 seed_7535m2400 t0=360: n_p=11857 n_t=12362 expo_def=-252.5 delay_def=-255.6
[ 598.2s] L4 seed_7535m2400 t0=360: n_p=12484 n_t=12362 expo_def=61.0 delay_def=76.6
[ 602.0s] L0 seed_7535m2400 t0=420: n_p=11447 n_t=12652 expo_def=-602.5 delay_def=-1177.2
[ 609.7s] L3 seed_7535m2400 t0=420: n_p=11799 n_t=12652 expo_def=-426.5 delay_def=-616.1
[ 620.6s] L4 seed_7535m2400 t0=420: n_p=12510 n_t=12652 expo_def=-71.0 delay_def=-253.0
[ 626.5s] L0 seed_7535m2400 t0=480: n_p=12959 n_t=14050 expo_def=-545.5 delay_def=-1119.5
[ 635.6s] L3 seed_7535m2400 t0=480: n_p=13418 n_t=14050 expo_def=-316.0 delay_def=-429.3
[ 644.8s] L4 seed_7535m2400 t0=480: n_p=13949 n_t=14050 expo_def=-50.5 delay_def=-162.4

== B. per-window totals (validation, t0 every 60 s) ==
  root             t0  variant  n_pred n_true expo_deficit delay_pred delay_true delay_deficit
  cv20_7005m2000   120       L0   7988   8744      -378.00     257.54     740.68       -483.14
  cv20_7005m2000   120       L3   8165   8744      -289.50     569.12     740.68       -171.57
  cv20_7005m2000   120       L4   8841   8744        48.50     804.41     740.68         63.73
  cv20_7005m2000   180       L0   8365   8869      -252.00     295.16     754.19       -459.03
  cv20_7005m2000   180       L3   8538   8869      -165.50     633.20     754.19       -120.99
  cv20_7005m2000   180       L4   8922   8869        26.50     750.29     754.19         -3.90
  cv20_7005m2000   240       L0   8060   8816      -378.00     210.21     739.41       -529.20
  cv20_7005m2000   240       L3   8316   8816      -250.00     565.90     739.41       -173.51
  cv20_7005m2000   240       L4   8726   8816       -45.00     679.84     739.41        -59.57
  cv20_7005m2000   300       L0   8731   9811      -540.00     345.85    1109.50       -763.65
  cv20_7005m2000   300       L3   8937   9811      -437.00     710.03    1109.50       -399.47
  cv20_7005m2000   300       L4   9670   9811       -70.50     968.28    1109.50       -141.23
  cv20_7005m2000   360       L0  10221  10955      -367.00     458.43     983.25       -524.83
  cv20_7005m2000   360       L3  10483  10955      -236.00     972.02     983.25        -11.23
  cv20_7005m2000   360       L4  11076  10955        60.50    1150.87     983.25        167.61
  cv20_7005m2000   420       L0   8892  10091      -599.50     332.47    1123.04       -790.57
  cv20_7005m2000   420       L3   9159  10091      -466.00     688.67    1123.04       -434.37
  cv20_7005m2000   420       L4  10077  10091        -7.00    1031.62    1123.04        -91.42
  cv20_7005m2000   480       L0   7304   8799      -747.50     238.89    1028.77       -789.88
  cv20_7005m2000   480       L3   7522   8799      -638.50     536.30    1028.77       -492.47
  cv20_7005m2000   480       L4   8751   8799       -24.00    1040.53    1028.77         11.75
  cv20_7006m2000   120       L0   6685   7267      -291.00     139.46     423.36       -283.90
  cv20_7006m2000   120       L3   6871   7267      -198.00     401.68     423.36        -21.68
  cv20_7006m2000   120       L4   7350   7267        41.50     493.63     423.36         70.27
  cv20_7006m2000   180       L0   7265   7792      -263.50     146.90     433.38       -286.48
  cv20_7006m2000   180       L3   7385   7792      -203.50     395.61     433.38        -37.77
  cv20_7006m2000   180       L4   7890   7792        49.00     483.17     433.38         49.79
  cv20_7006m2000   240       L0   6542   7967      -712.50     159.52     883.94       -724.42
  cv20_7006m2000   240       L3   6698   7967      -634.50     397.02     883.94       -486.92
  cv20_7006m2000   240       L4   8100   7967        66.50     919.36     883.94         35.42
  cv20_7006m2000   300       L0   6988   7936      -474.00     235.10     780.28       -545.18
  cv20_7006m2000   300       L3   7189   7936      -373.50     514.94     780.28       -265.34
  cv20_7006m2000   300       L4   7921   7936        -7.50     763.10     780.28        -17.18
  cv20_7006m2000   360       L0   7416   7818      -201.00     177.77     487.14       -309.36
  cv20_7006m2000   360       L3   7599   7818      -109.50     481.46     487.14         -5.68
  cv20_7006m2000   360       L4   7848   7818        15.00     557.96     487.14         70.83
  cv20_7006m2000   420       L0   7342   8408      -533.00     194.61     869.41       -674.80
  cv20_7006m2000   420       L3   7602   8408      -403.00     498.78     869.41       -370.63
  cv20_7006m2000   420       L4   8167   8408      -120.50     661.27     869.41       -208.14
  cv20_7006m2000   480       L0   7643   8582      -469.50     243.47     740.40       -496.93
  cv20_7006m2000   480       L3   7845   8582      -368.50     585.69     740.40       -154.71
  cv20_7006m2000   480       L4   8314   8582      -134.00     678.86     740.40        -61.54
  cv30_7305m2700   120       L0  10267  10789      -261.00     554.06    1068.03       -513.96
  cv30_7305m2700   120       L3  10453  10789      -168.00    1018.09    1068.03        -49.94
  cv30_7305m2700   120       L4  10715  10789       -37.00    1042.64    1068.03        -25.39
  cv30_7305m2700   180       L0  10435  11374      -469.50     587.01    1341.12       -754.11
  cv30_7305m2700   180       L3  10674  11374      -350.00    1047.08    1341.12       -294.04
  cv30_7305m2700   180       L4  11387  11374         6.50    1300.88    1341.12        -40.25
  cv30_7305m2700   240       L0  10827  11903      -538.00     619.06    1594.43       -975.37
  cv30_7305m2700   240       L3  11108  11903      -397.50    1149.71    1594.43       -444.71
  cv30_7305m2700   240       L4  11639  11903      -132.00    1325.47    1594.43       -268.96
  cv30_7305m2700   300       L0  11107  12347      -620.00     829.71    1969.44      -1139.74
  cv30_7305m2700   300       L3  11574  12347      -386.50    1418.94    1969.44       -550.50
  cv30_7305m2700   300       L4  12334  12347        -6.50    1772.24    1969.44       -197.21
  cv30_7305m2700   360       L0  12272  13618      -673.00    1133.20    2567.47      -1434.28
  cv30_7305m2700   360       L3  12739  13618      -439.50    1766.64    2567.47       -800.83
  cv30_7305m2700   360       L4  13385  13618      -116.50    2131.65    2567.47       -435.82
  cv30_7305m2700   420       L0  13657  14743      -543.00    1678.70    3074.85      -1396.15
  cv30_7305m2700   420       L3  14058  14743      -342.50    2334.90    3074.85       -739.95
  cv30_7305m2700   420       L4  14607  14743       -68.00    2621.15    3074.85       -453.70
  cv30_7305m2700   480       L0  14126  15280      -577.00    1872.73    3317.11      -1444.39
  cv30_7305m2700   480       L3  14485  15280      -397.50    2551.40    3317.11       -765.72
  cv30_7305m2700   480       L4  15239  15280       -20.50    2943.65    3317.11       -373.47
  seed_7535m2400   120       L0   9533   9938      -202.50     386.82     719.41       -332.59
  seed_7535m2400   120       L3   9736   9938      -101.00     799.08     719.41         79.67
  seed_7535m2400   120       L4   9996   9938        29.00     833.29     719.41        113.88
  seed_7535m2400   180       L0   9741  10489      -374.00     513.31    1057.39       -544.08
  seed_7535m2400   180       L3   9962  10489      -263.50     980.61    1057.39        -76.78
  seed_7535m2400   180       L4  10581  10489        46.00    1125.22    1057.39         67.83
  seed_7535m2400   240       L0  10225  10730      -252.50     593.05    1121.57       -528.52
  seed_7535m2400   240       L3  10460  10730      -135.00    1084.32    1121.57        -37.25
  seed_7535m2400   240       L4  10874  10730        72.00    1255.21    1121.57        133.65
  seed_7535m2400   300       L0  11237  12072      -417.50     653.66    1292.28       -638.62
  seed_7535m2400   300       L3  11495  12072      -288.50    1181.47    1292.28       -110.81
  seed_7535m2400   300       L4  12072  12072         0.00    1359.05    1292.28         66.77
  seed_7535m2400   360       L0  11455  12362      -453.50     793.86    1613.50       -819.64
  seed_7535m2400   360       L3  11857  12362      -252.50    1357.95    1613.50       -255.55
  seed_7535m2400   360       L4  12484  12362        61.00    1690.10    1613.50         76.60
  seed_7535m2400   420       L0  11447  12652      -602.50     939.66    2116.88      -1177.22
  seed_7535m2400   420       L3  11799  12652      -426.50    1500.77    2116.88       -616.11
  seed_7535m2400   420       L4  12510  12652       -71.00    1863.92    2116.88       -252.97
  seed_7535m2400   480       L0  12959  14050      -545.50    1351.11    2470.59      -1119.48
  seed_7535m2400   480       L3  13418  14050      -316.00    2041.29    2470.59       -429.29
  seed_7535m2400   480       L4  13949  14050       -50.50    2308.15    2470.59       -162.44
  L0 60s-grid means over 28 windows: expo_def=-454.86  delay_pred=569.33  delay_true=1300.74  delay_def=-731.41 veh s
  L3 60s-grid means over 28 windows: expo_def=-322.77  delay_pred=1006.52  delay_true=1300.74  delay_def=-294.22 veh s
  L4 60s-grid means over 28 windows: expo_def=-13.88  delay_pred=1234.14  delay_true=1300.74  delay_def=-66.61 veh s
  [p3j published means: delay_true=1300.74 delay_def L0=-731.41 L3=-294.22 -- L0/L3 here must reproduce them (identical deterministic rollouts)]

== C. deficit by attribute (validation, 60 s grid; units veh s pooled over 28 windows/variant; expo_deficit = 0.5*(n_pred - n_true); delay_deficit = delay_pred - delay_true (physics delay); ff_component = ff_true - ff_pred; identity: expo + ff = delay deficit) ==
  -- by lane (L0) --
  value             n_steps_pred n_steps_true expo_deficit ff_component delay_deficit
  lane=0                128048       136645     -4298.50       320.41      -3978.09
  lane=1                131892       141719     -4913.50     -7817.95     -12731.45
  lane=2                  8790        15838     -3524.00      -245.97      -3769.97
  -- by segment (L0) --
  value             n_steps_pred n_steps_true expo_deficit ff_component delay_deficit
  segment=downstream        53234        65335     -6050.50     -2148.68      -8199.18
  segment=merge_edge         9185        13055     -1935.00      -645.01      -2580.01
  segment=rsu_window        61708        68014     -3153.00     -1822.01      -4975.01
  segment=upstream         144603       147798     -1597.50     -3127.81      -4725.31
  -- by class (L0) --
  value             n_steps_pred n_steps_true expo_deficit ff_component delay_deficit
  class=born              68253        72766     -2256.50     -2437.57      -4694.07
  class=t0               200477       221436    -10479.50     -5305.94     -15785.44
  -- by lc_affected (L0) --
  value             n_steps_pred n_steps_true expo_deficit ff_component delay_deficit
  lc_affected=no               268730       287263     -9266.50    -10172.00     -19438.50
  lc_affected=yes                   0         6939     -3469.50      2428.49      -1041.01
  -- by lane (L3) --
  value             n_steps_pred n_steps_true expo_deficit ff_component delay_deficit
  lane=0                131114       136645     -2765.50      5067.17       2301.67
  lane=1                136044       141719     -2837.50     -4231.56      -7069.06
  lane=2                  8969        15838     -3434.50       -36.26      -3470.76
  -- by segment (L3) --
  value             n_steps_pred n_steps_true expo_deficit ff_component delay_deficit
  segment=downstream        53816        65335     -5759.50      -178.56      -5938.06
  segment=merge_edge         9876        13055     -1589.50      -361.26      -1950.76
  segment=rsu_window        62704        68014     -2655.00       485.34      -2169.66
  segment=upstream         149731       147798       966.50       853.84       1820.34
  -- by class (L3) --
  value             n_steps_pred n_steps_true expo_deficit ff_component delay_deficit
  class=born              68284        72766     -2241.00       759.10      -1481.90
  class=t0               207843       221436     -6796.50        40.25      -6756.25
  -- by lc_affected (L3) --
  value             n_steps_pred n_steps_true expo_deficit ff_component delay_deficit
  lc_affected=no               268468       287263     -9397.50      1303.86      -8093.64
  lc_affected=yes                7659         6939       360.00      -504.50       -144.50
  -- by lane (L4) --
  value             n_steps_pred n_steps_true expo_deficit ff_component delay_deficit
  lane=0                131882       136645     -2381.50      4527.96       2146.46
  lane=1                152339       141719      5310.00     -6029.13       -719.13
  lane=2                  9204        15838     -3317.00        24.64      -3292.36
  -- by segment (L4) --
  value             n_steps_pred n_steps_true expo_deficit ff_component delay_deficit
  segment=downstream        70277        65335      2471.00     -2122.69        348.31
  segment=merge_edge        10599        13055     -1228.00      -386.96      -1614.96
  segment=rsu_window        62940        68014     -2537.00       324.44      -2212.56
  segment=upstream         149609       147798       905.50       708.69       1614.19
  -- by class (L4) --
  value             n_steps_pred n_steps_true expo_deficit ff_component delay_deficit
  class=born              72487        72766      -139.50      -522.60       -662.10
  class=t0               220938       221436      -249.00      -953.92      -1202.92
  -- by lc_affected (L4) --
  value             n_steps_pred n_steps_true expo_deficit ff_component delay_deficit
  lc_affected=no               285782       287263      -740.50     -1612.45      -2352.95
  lc_affected=yes                7643         6939       352.00       135.92        487.92

== D. top-10 cells by |delay_deficit| (validation, 60 s grid, per variant) ==
  -- L0 --
  cell                                                           delay_deficit n_pred n_true
  lane=1 seg=rsu_window cls=t0   lc=no                              -4179.30  27169  32557
  lane=1 seg=downstream cls=t0   lc=no                              -2904.26  29197  28841
  lane=2 seg=downstream cls=t0   lc=no                              -2182.13    927   5149
  lane=1 seg=merge_edge cls=t0   lc=no                              -2044.82   4747   7655
  lane=0 seg=upstream   cls=born lc=no                              -1368.45  29715  32244
  lane=1 seg=upstream   cls=t0   lc=no                              -1273.41  38810  39904
  lane=1 seg=upstream   cls=born lc=no                              -1270.83  31969  28159
  lane=2 seg=downstream cls=born lc=no                              -1120.68    689   2718
  lane=0 seg=downstream cls=t0   lc=no                               -839.75  22421  24526
  lane=0 seg=upstream   cls=t0   lc=no                               -837.28  44109  44996
  -- L3 --
  cell                                                           delay_deficit n_pred n_true
  lane=1 seg=rsu_window cls=t0   lc=no                              -2989.42  27401  32557
  lane=1 seg=downstream cls=t0   lc=no                              -2222.79  27869  28841
  lane=2 seg=downstream cls=t0   lc=no                              -2123.02   1037   5149
  lane=1 seg=merge_edge cls=t0   lc=no                              -1695.37   5199   7655
  lane=2 seg=downstream cls=born lc=no                              -1104.70    655   2718
  lane=0 seg=upstream   cls=t0   lc=no                                750.74  43872  44996
  lane=0 seg=rsu_window cls=t0   lc=no                                723.43  27628  28284
  lane=1 seg=downstream cls=t0   lc=yes                              -533.71    594   1721
  lane=0 seg=downstream cls=t0   lc=no                                472.09  22871  24526
  lane=1 seg=upstream   cls=t0   lc=no                                414.43  41473  39904
  -- L4 --
  cell                                                           delay_deficit n_pred n_true
  lane=1 seg=rsu_window cls=t0   lc=no                              -2852.28  27799  32557
  lane=1 seg=downstream cls=t0   lc=no                               2590.83  39116  28841
  lane=2 seg=downstream cls=t0   lc=no                              -2182.06    764   5149
  lane=1 seg=merge_edge cls=t0   lc=no                              -1406.16   5716   7655
  lane=2 seg=downstream cls=born lc=no                               -983.99    667   2718
  lane=0 seg=upstream   cls=t0   lc=no                                924.79  45871  44996
  lane=0 seg=downstream cls=t0   lc=no                                594.01  23286  24526
  lane=0 seg=rsu_window cls=t0   lc=no                                587.94  27992  28284
  lane=1 seg=upstream   cls=t0   lc=no                                239.59  40518  39904
  lane=1 seg=upstream   cls=born lc=no                                224.94  31582  28159

== E. speed profile, bins with |dv| > 1 m/s (matched vehicle-steps pooled over validation 60 s windows, binned by TRUTH (lane, 50 m d-bin) at tick j; dv = v_pred_mean - v_true_mean m/s) ==
  variant lane  d_bin  v_pred_mean v_true_mean      n     dv
  L0    0   -500       29.388       28.211   1669  +1.177
  L0    0   -450       29.200       27.717   2073  +1.483
  L0    0   -400       29.070       27.531   2233  +1.539
  L0    0   -350       29.010       27.483   2297  +1.527
  L0    0   -300       28.979       27.454   2258  +1.525
  L0    0   -250       28.930       27.455   2308  +1.475
  L0    0   -200       28.825       27.413   2277  +1.411
  L0    0   -150       28.785       27.409   2268  +1.376
  L0    0   -100       28.619       27.200   2700  +1.419
  L0    0    -50       28.918       27.420   2313  +1.498
  L0    0      0       29.161       27.606   2377  +1.555
  L0    0     50       29.192       27.762   2150  +1.430
  L0    0    100       29.273       27.892   2099  +1.381
  L0    0    150       29.326       28.108   2011  +1.218
  L0    0    200       29.330       28.198   2011  +1.132
  L0    0    250       29.336       28.267   1989  +1.069
  L0    0    300       29.334       28.284   1966  +1.050
  L0    0    350       29.317       28.247   1965  +1.070
  L0    0    400       29.334       28.209   1983  +1.126
  L0    0    450       29.406       28.240   1945  +1.166
  L0    0    500       29.425       28.248   1974  +1.177
  L0    0    550       29.411       28.282   1974  +1.129
  L0    0    600       29.408       28.322   1980  +1.086
  L0    0    650       29.409       28.324   1994  +1.085
  L0    0    700       29.381       28.298   2004  +1.083
  L0    0    750       29.378       28.293   2011  +1.085
  L0    0    800       29.392       28.285   1987  +1.107
  L0    0    850       29.378       28.292   2011  +1.086
  L0    0    900       29.377       28.249   2022  +1.128
  L0    0    950       29.388       28.247   1999  +1.141
  L0    0   1000       29.419       28.251   2009  +1.168
  L0    0   1050       29.409       28.205   2030  +1.204
  L0    0   1100       29.419       28.230   2029  +1.189
  L0    0   1150       29.412       28.260   1967  +1.153
  L0    0   1200       29.398       28.245   2002  +1.153
  L0    0   1250       29.387       28.251   2019  +1.136
  L0    0   1300       29.394       28.285   1974  +1.110
  L0    0   1350       29.383       28.273   2010  +1.110
  L0    0   1400       29.377       28.254   1993  +1.122
  L0    0   1450       29.389       28.277   1995  +1.113
  L0    0   1500       29.374       28.258   1973  +1.116
  L0    0   1550       29.387       28.255   1969  +1.132
  L0    0   1600       29.394       28.251   2003  +1.143
  L0    0   1650       29.413       28.254   1979  +1.158
  L0    0   1700       29.414       28.233   1995  +1.181
  L0    0   1750       29.443       28.230   2001  +1.213
  L0    0   1800       29.444       28.203   2011  +1.241
  L0    0   1850       29.444       28.186   1999  +1.258
  L0    0   1900       29.460       28.178   2008  +1.281
  L0    0   1950       29.449       28.126   2001  +1.323
  L0    0   2000       29.462       28.111   2001  +1.351
  L0    0   2050       29.496       28.085   2001  +1.411
  L0    0   2100       29.504       27.963   2014  +1.542
  L0    0   2150       29.468       27.836   2029  +1.632
  L0    0   2200       29.462       27.700   2021  +1.763
  L0    0   2250       29.395       27.482   2055  +1.913
  L0    0   2300       29.288       27.224   2079  +2.065
  L0    0   2350       29.084       26.827   2107  +2.257
  L0    0   2400       28.760       26.110   2143  +2.650
  L0    0   2450       28.047       24.864   2250  +3.183
  L0    0   2500       26.603       22.997   2406  +3.606
  L0    0   2550       23.072       19.351   2780  +3.721
  L0    0   2600       12.147        9.481   4431  +2.666
  L0    1   -550       29.747       28.423    409  +1.323
  L0    1   -500       29.530       27.257    738  +2.272
  L0    1   -450       29.254       26.697    954  +2.557
  L0    1   -400       29.119       26.374   1113  +2.745
  L0    1   -350       28.881       26.026   1173  +2.855
  L0    1   -300       28.535       25.552   1222  +2.982
  L0    1   -250       28.084       24.738   1242  +3.346
  L0    1   -200       27.332       23.454   1251  +3.878
  L0    1   -150       26.041       21.341   1353  +4.700
  L0    1   -100       23.406       11.565   2782  +11.841
  L0    1    -50       21.845        6.142   5299  +15.703
  L0    1      0       20.918        7.617   4919  +13.301
  L0    1     50       20.616       10.131   3906  +10.484
  L0    1    100       20.833       12.459   3365  +8.373
  L0    1    150       21.815       14.218   3118  +7.597
  L0    1    200       22.782       16.410   2755  +6.372
  L0    1    250       22.894       17.986   2552  +4.908
  L0    1    300       23.603       19.147   2425  +4.456
  L0    1    350       24.897       20.281   2296  +4.616
  L0    1    400       25.611       21.602   2187  +4.009
  L0    1    450       28.116       23.688   2013  +4.428
  L0    1    500       29.660       26.812   1775  +2.848
  L0    1    550       29.672       27.471   1728  +2.201
  L0    1    600       29.683       27.611   1713  +2.072
  L0    1    650       29.685       27.668   1716  +2.017
  L0    1    700       29.692       27.673   1691  +2.019
  L0    1    750       29.653       27.657   1724  +1.996
  L0    1    800       29.651       27.663   1737  +1.988
  L0    1    850       29.654       27.654   1708  +2.000
  L0    1    900       29.671       27.672   1727  +1.999
  L0    1    950       29.664       27.660   1719  +2.004
  L0    1   1000       29.661       27.644   1710  +2.017
  L0    1   1050       29.679       27.654   1703  +2.025
  L0    1   1100       29.665       27.635   1707  +2.030
  L0    1   1150       29.665       27.644   1741  +2.021
  L0    1   1200       29.653       27.641   1699  +2.011
  L0    1   1250       29.641       27.664   1726  +1.977
  L0    1   1300       29.632       27.644   1738  +1.987
  L0    1   1350       29.609       27.658   1707  +1.950
  L0    1   1400       29.606       27.660   1754  +1.946
  L0    1   1450       29.589       27.649   1733  +1.940
  L0    1   1500       29.572       27.642   1726  +1.931
  L0    1   1550       29.575       27.654   1747  +1.922
  L0    1   1600       29.562       27.632   1713  +1.929
  L0    1   1650       29.559       27.628   1735  +1.931
  L0    1   1700       29.560       27.621   1731  +1.939
  L0    1   1750       29.571       27.628   1719  +1.943
  L0    1   1800       29.539       27.602   1724  +1.937
  L0    1   1850       29.534       27.610   1715  +1.924
  L0    1   1900       29.533       27.595   1736  +1.938
  L0    1   1950       29.521       27.566   1717  +1.955
  L0    1   2000       29.506       27.519   1731  +1.987
  L0    1   2050       29.516       27.491   1740  +2.025
  L0    1   2100       29.512       27.436   1712  +2.076
  L0    1   2150       29.488       27.359   1728  +2.128
  L0    1   2200       29.432       27.232   1744  +2.200
  L0    1   2250       29.391       27.064   1753  +2.328
  L0    1   2300       29.274       26.822   1780  +2.452
  L0    1   2350       29.084       26.394   1807  +2.690
  L0    1   2400       28.669       25.586   1819  +3.083
  L0    1   2450       27.909       24.257   1959  +3.653
  L0    1   2500       26.241       22.134   2201  +4.107
  L0    1   2550       22.664       18.746   2683  +3.919
  L0    1   2600       12.339        9.773   4423  +2.566
  L0    2   -100        6.340        1.958    152  +4.382
  L0    2    -50       17.540        9.758    791  +7.782
  L0    2      0       21.420       16.390   1289  +5.030
  L0    2     50       21.637       19.802    996  +1.835
  L0    2    100       21.633       20.555   1002  +1.078
  L3    0   -600       26.217       29.677    107  -3.459
  L3    0   -550       26.063       28.907   1928  -2.844
  L3    0   -500       25.907       27.948   2199  -2.041
  L3    0   -450       25.704       27.628   2256  -1.923
  L3    0   -400       25.515       27.511   2275  -1.996
  L3    0   -350       25.314       27.483   2299  -2.168
  L3    0   -300       25.125       27.454   2258  -2.329
  L3    0   -250       24.874       27.452   2310  -2.579
  L3    0   -200       24.730       27.397   2283  -2.667
  L3    0   -150       24.629       27.383   2274  -2.753
  L3    0   -100       24.644       27.186   2706  -2.542
  L3    0    -50       25.140       27.355   2324  -2.214
  L3    0      0       25.697       27.606   2377  -1.909
  L3    0     50       26.022       27.762   2150  -1.740
  L3    0    100       26.258       27.892   2099  -1.634
  L3    0    150       26.341       28.108   2011  -1.767
  L3    0    200       26.464       28.198   2011  -1.735
  L3    0    250       26.562       28.267   1989  -1.705
  L3    0    300       26.649       28.284   1966  -1.635
  L3    0    350       26.692       28.247   1965  -1.555
  L3    0    400       26.757       28.209   1983  -1.451
  L3    0    450       26.859       28.240   1945  -1.381
  L3    0    500       26.927       28.248   1974  -1.321
  L3    0    550       26.969       28.282   1974  -1.313
  L3    0    600       26.969       28.322   1980  -1.353
  L3    0    650       26.947       28.324   1994  -1.378
  L3    0    700       26.941       28.298   2004  -1.356
  L3    0    750       26.943       28.293   2011  -1.349
  L3    0    800       26.966       28.285   1987  -1.319
  L3    0    850       26.954       28.292   2011  -1.338
  L3    0    900       26.958       28.249   2022  -1.292
  L3    0    950       26.968       28.247   1999  -1.278
  L3    0   1000       26.989       28.251   2009  -1.262
  L3    0   1050       27.000       28.205   2030  -1.205
  L3    0   1100       27.032       28.230   2029  -1.198
  L3    0   1150       27.021       28.260   1967  -1.238
  L3    0   1200       26.997       28.245   2002  -1.248
  L3    0   1250       26.983       28.251   2019  -1.267
  L3    0   1300       26.967       28.285   1974  -1.318
  L3    0   1350       26.937       28.273   2010  -1.336
  L3    0   1400       26.924       28.254   1993  -1.330
  L3    0   1450       26.920       28.277   1995  -1.357
  L3    0   1500       26.910       28.258   1973  -1.348
  L3    0   1550       26.886       28.255   1969  -1.369
  L3    0   1600       26.873       28.251   2003  -1.378
  L3    0   1650       26.813       28.254   1979  -1.442
  L3    0   1700       26.819       28.233   1995  -1.414
  L3    0   1750       26.840       28.230   2001  -1.390
  L3    0   1800       26.883       28.203   2011  -1.320
  L3    0   1850       26.885       28.186   1999  -1.301
  L3    0   1900       26.924       28.178   2008  -1.254
  L3    0   1950       26.890       28.126   2001  -1.236
  L3    0   2000       26.925       28.111   2001  -1.186
  L3    0   2050       26.907       28.085   2001  -1.178
  L3    0   2100       26.897       27.963   2014  -1.066
  L3    0   2150       26.803       27.836   2029  -1.033
  L3    0   2200       26.688       27.700   2021  -1.011
  L3    1   -600       27.019       29.025     42  -2.006
  L3    1   -550       27.106       28.119    941  -1.013
  L3    1   -200       24.450       23.357   1294  +1.093
  L3    1   -150       23.334       20.860   1460  +2.474
  L3    1   -100       20.880       11.024   3099  +9.856
  L3    1    -50       19.773        5.849   5906  +13.925
  L3    1      0       18.935        7.285   5197  +11.650
  L3    1     50       18.225        9.886   4016  +8.339
  L3    1    100       18.371       12.549   3334  +5.822
  L3    1    150       20.046       14.365   3076  +5.682
  L3    1    200       21.374       16.665   2705  +4.709
  L3    1    250       21.255       18.051   2541  +3.204
  L3    1    300       22.317       19.147   2425  +3.170
  L3    1    350       23.118       20.262   2299  +2.856
  L3    1    400       23.494       21.602   2187  +1.892
  L3    1    450       25.522       23.688   2013  +1.834
  L3    2   -100        6.571        2.359    182  +4.212
  L3    2    -50       16.836        9.457   1136  +7.379
  L3    2      0       19.456       16.117   1220  +3.339
  L4    0   -600       25.774       29.666    107  -3.891
  L4    0   -550       25.449       28.918   1981  -3.469
  L4    0   -500       25.019       27.960   2217  -2.941
  L4    0   -450       24.636       27.633   2278  -2.997
  L4    0   -400       24.104       27.507   2305  -3.403
  L4    0   -350       23.739       27.476   2332  -3.737
  L4    0   -300       23.376       27.440   2291  -4.064
  L4    0   -250       23.141       27.422   2343  -4.281
  L4    0   -200       22.942       27.331   2320  -4.389
  L4    0   -150       23.091       27.265   2315  -4.174
  L4    0   -100       23.501       26.976   2755  -3.475
  L4    0    -50       24.558       27.266   2336  -2.708
  L4    0      0       25.632       27.606   2377  -1.973
  L4    0     50       26.182       27.762   2150  -1.580
  L4    0    100       26.498       27.892   2099  -1.394
  L4    0    150       26.584       28.108   2011  -1.524
  L4    0    200       26.647       28.198   2011  -1.552
  L4    0    250       26.722       28.267   1989  -1.545
  L4    0    300       26.799       28.284   1966  -1.485
  L4    0    350       26.853       28.247   1965  -1.394
  L4    0    400       26.909       28.209   1983  -1.300
  L4    0    450       27.018       28.240   1945  -1.222
  L4    0    500       27.082       28.248   1974  -1.166
  L4    0    550       27.088       28.282   1974  -1.195
  L4    0    600       27.094       28.322   1980  -1.228
  L4    0    650       27.077       28.324   1994  -1.248
  L4    0    700       27.068       28.298   2004  -1.229
  L4    0    750       27.034       28.293   2011  -1.258
  L4    0    800       27.049       28.285   1987  -1.236
  L4    0    850       27.069       28.292   2011  -1.223
  L4    0    900       27.023       28.249   2022  -1.226
  L4    0    950       26.996       28.247   1999  -1.251
  L4    0   1000       26.968       28.251   2009  -1.283
  L4    0   1050       26.944       28.205   2030  -1.261
  L4    0   1100       26.983       28.230   2029  -1.247
  L4    0   1150       26.976       28.260   1967  -1.284
  L4    0   1200       26.952       28.245   2002  -1.293
  L4    0   1250       26.942       28.251   2019  -1.309
  L4    0   1300       26.935       28.285   1974  -1.350
  L4    0   1350       26.915       28.273   2010  -1.358
  L4    0   1400       26.879       28.254   1993  -1.375
  L4    0   1450       26.900       28.277   1995  -1.377
  L4    0   1500       26.860       28.258   1973  -1.398
  L4    0   1550       26.890       28.255   1969  -1.364
  L4    0   1600       26.928       28.251   2003  -1.322
  L4    0   1650       26.915       28.254   1979  -1.340
  L4    0   1700       26.937       28.233   1995  -1.295
  L4    0   1750       26.918       28.230   2001  -1.312
  L4    0   1800       26.881       28.203   2011  -1.322
  L4    0   1850       26.891       28.186   1999  -1.295
  L4    0   1900       26.954       28.178   2008  -1.225
  L4    0   1950       26.946       28.126   2001  -1.181
  L4    0   2000       26.976       28.111   2001  -1.135
  L4    0   2050       26.992       28.085   2001  -1.093
  L4    1   -600       25.502       28.826     58  -3.325
  L4    1   -550       25.558       28.029   1325  -2.472
  L4    1   -500       25.139       27.063   1482  -1.925
  L4    1   -450       24.559       26.624   1556  -2.065
  L4    1   -400       23.997       26.311   1620  -2.314
  L4    1   -350       23.263       25.952   1654  -2.690
  L4    1   -300       22.422       25.490   1728  -3.068
  L4    1   -250       21.403       24.427   1798  -3.023
  L4    1   -200       20.344       22.510   1981  -2.166
  L4    1   -100       16.823        9.247   5509  +7.576
  L4    1    -50       13.388        5.130   7389  +8.258
  L4    1      0       12.697        7.055   5398  +5.642
  L4    1     50       14.345        9.858   4042  +4.487
  L4    1    100       15.243       12.482   3357  +2.762
  L4    1    150       17.558       14.280   3101  +3.278
  L4    1    200       19.939       16.547   2728  +3.392
  L4    1    250       20.151       17.986   2552  +2.165
  L4    1    300       21.066       19.147   2425  +1.919
  L4    1    350       22.775       20.262   2299  +2.513
  L4    1    400       23.157       21.602   2187  +1.555
  L4    1    450       25.557       23.688   2013  +1.868
  L4    2   -100       16.733        1.164   3548  +15.570
  L4    2    -50       13.598        5.037   3213  +8.561
  L4    2      0       17.115       15.508   1390  +1.607
  (286 bins with |dv| > 1 m/s)
  -- accel-lane (lane 2) profile, ALL bins --
  variant  d_bin  v_pred_mean v_true_mean      n     dv
  L0       -100        6.340        1.958    152  +4.382
  L0        -50       17.540        9.758    791  +7.782
  L0          0       21.420       16.390   1289  +5.030
  L0         50       21.637       19.802    996  +1.835
  L0        100       21.633       20.555   1002  +1.078
  L0        150       21.603       20.826    989  +0.777
  L0        200       21.518       20.799    909  +0.719
  L0        250       21.475       20.795    992  +0.680
  L0        300       21.126       20.519    902  +0.608
  L0        350       20.347       20.079    607  +0.268
  L3       -100        6.571        2.359    182  +4.212
  L3        -50       16.836        9.457   1136  +7.379
  L3          0       19.456       16.117   1220  +3.339
  L3         50       20.131       19.780    932  +0.351
  L3        100       20.205       20.519    951  -0.314
  L3        150       20.151       20.795    943  -0.645
  L3        200       20.076       20.793    884  -0.717
  L3        250       20.080       20.805    971  -0.726
  L3        300       19.908       20.515    896  -0.607
  L3        350       19.687       20.069    604  -0.382
  L4       -100       16.733        1.164   3548  +15.570
  L4        -50       13.598        5.037   3213  +8.561
  L4          0       17.115       15.508   1390  +1.607
  L4         50       19.290       19.802    996  -0.513
  L4        100       19.747       20.555   1002  -0.808
  L4        150       19.953       20.826    989  -0.873
  L4        200       20.005       20.799    909  -0.794
  L4        250       20.037       20.795    992  -0.758
  L4        300       19.883       20.519    902  -0.636
  L4        350       19.658       20.079    607  -0.421

== F. exit mismatch & entry waiting (validation, 60 s grid) ==
  L0: pred-exited-early: 1304 vehicles, truth remained 25634 steps, remaining truth delay=7399.80 s | truth-exited-early: 219 vehicles, pred remained 162 steps, remaining pred delay=26.86 s
  L0: entry waiting=0 steps over 1243 matched births (expected 0); pred-only births=0; truth-only births=0; t0 vehicles missing in pred=0 (expected 0); truth presence gaps=0
  L3: pred-exited-early: 713 vehicles, truth remained 20797 steps, remaining truth delay=6560.65 s | truth-exited-early: 736 vehicles, pred remained 2776 steps, remaining pred delay=317.58 s
  L3: entry waiting=0 steps over 1243 matched births (expected 0); pred-only births=0; truth-only births=0; t0 vehicles missing in pred=0 (expected 0); truth presence gaps=0
  L4: pred-exited-early: 421 vehicles, truth remained 5926 steps, remaining truth delay=1502.41 s | truth-exited-early: 873 vehicles, pred remained 5149 steps, remaining pred delay=621.53 s
  L4: entry waiting=0 steps over 1243 matched births (expected 0); pred-only births=0; truth-only births=0; t0 vehicles missing in pred=0 (expected 0); truth presence gaps=0
[ 644.9s] wrote D:\rd-wt\wd01e\output\wd01\part3k\results\decomposition_p3k.csv (2277 data rows), D:\rd-wt\wd01e\output\wd01\part3k\results\speed_profile_p3k.csv (420 data rows), D:\rd-wt\wd01e\output\wd01\part3k\results\stdout_decompose.txt

DONE part 3k decomposition.
```

## Ladder aggregation + gate stdout (aggregate_p3k.py -- verbatim)

Raw rows: results/raw/raw_metrics.jsonl (written first by eval_p3k.py); aggregated verdicts: results/gate_v2_p3k.json.

```text
=== WD-01 part 3k: ladder aggregation + gate verdicts ===
ZERO simulation; holdout blocked in root_arrays (never passed); validation only.
loaded 2288 deduped raw rows

== per-(variant, H) summary (4 validation roots x 52 windows x members) ==
  L0@H30: n_rolls=52 n_members=1 n_windows=52 speed_rmse=4.1881 coll/1000=0.0 lc 0.0/12.077 | exposure delay p/t 2471.856/2605.202 (bias -133.35) | physics delay p/t 335.629/629.015 (bias -293.39)
  L0@H60: n_rolls=52 n_members=1 n_windows=52 speed_rmse=4.9774 coll/1000=0.0 lc 0.0/23.942 | exposure delay p/t 4800.173/5261.288 (bias -461.11) | physics delay p/t 563.453/1302.219 (bias -738.77)
  L0@H90: n_rolls=52 n_members=1 n_windows=52 speed_rmse=5.2246 coll/1000=0.0 lc 0.0/35.846 | exposure delay p/t 7031.5/7958.096 (bias -926.6) | physics delay p/t 748.64/2019.537 (bias -1270.9)
  L0@H120: n_rolls=52 n_members=1 n_windows=52 speed_rmse=5.34 coll/1000=0.0 lc 0.0/47.904 | exposure delay p/t 9191.394/10689.885 (bias -1498.49) | physics delay p/t 913.526/2776.727 (bias -1863.2)
  L3@H30: n_rolls=260 n_members=5 n_windows=52 speed_rmse=3.8008 coll/1000=0.013 lc 15.027/12.077 | exposure delay p/t 2502.037/2605.202 (bias -103.17) | physics delay p/t 534.801/629.015 (bias -94.21)
  L3@H60: n_rolls=260 n_members=5 n_windows=52 speed_rmse=4.6886 coll/1000=0.008 lc 29.292/23.942 | exposure delay p/t 4933.237/5261.288 (bias -328.05) | physics delay p/t 1003.167/1302.219 (bias -299.05)
  L3@H90: n_rolls=260 n_members=5 n_windows=52 speed_rmse=5.0293 coll/1000=0.006 lc 42.977/35.846 | exposure delay p/t 7314.45/7958.096 (bias -643.65) | physics delay p/t 1432.393/2019.537 (bias -587.14)
  L3@H120: n_rolls=260 n_members=5 n_windows=52 speed_rmse=5.2619 coll/1000=0.004 lc 56.896/47.904 | exposure delay p/t 9673.777/10689.885 (bias -1016.11) | physics delay p/t 1845.392/2776.727 (bias -931.33)
  L4@H30: n_rolls=260 n_members=5 n_windows=52 speed_rmse=4.4044 coll/1000=0.071 lc 15.692/12.077 | exposure delay p/t 2614.65/2605.202 (bias 9.45) | physics delay p/t 628.997/629.015 (bias -0.02)
  L4@H60: n_rolls=260 n_members=5 n_windows=52 speed_rmse=5.1457 coll/1000=0.043 lc 29.431/23.942 | exposure delay p/t 5237.556/5261.288 (bias -23.73) | physics delay p/t 1222.75/1302.219 (bias -79.47)
  L4@H90: n_rolls=260 n_members=5 n_windows=52 speed_rmse=5.5655 coll/1000=0.032 lc 42.777/35.846 | exposure delay p/t 7803.644/7958.096 (bias -154.45) | physics delay p/t 1768.766/2019.537 (bias -250.77)
  L4@H120: n_rolls=260 n_members=5 n_windows=52 speed_rmse=5.8485 coll/1000=0.026 lc 56.054/47.904 | exposure delay p/t 10335.506/10689.885 (bias -354.38) | physics delay p/t 2285.492/2776.727 (bias -491.23)

== gate v2 criterion (c): member bands + CRPS (contract physics delay; exposure coverage alongside) ==
  L3@H30: n_windows=52 delay_phys_10-90_cov=0.0769 delay_exposure_10-90_cov=0.0 ttc3_10-90_cov=0.5385 delay_phys_minmax=0.1154 CRPS_phys=92.14 (L0 MAE_phys=293.39)
  L4@H30: n_windows=52 delay_phys_10-90_cov=0.1346 delay_exposure_10-90_cov=0.2308 ttc3_10-90_cov=0.3269 delay_phys_minmax=0.1731 CRPS_phys=38.76 (L0 MAE_phys=293.39)
  L0@H30: deterministic delay_phys MAE=293.39 over 52 windows
  L3@H60: n_windows=52 delay_phys_10-90_cov=0.0192 delay_exposure_10-90_cov=0.0 ttc3_10-90_cov=0.3846 delay_phys_minmax=0.0385 CRPS_phys=289.74 (L0 MAE_phys=738.77)
  L4@H60: n_windows=52 delay_phys_10-90_cov=0.1923 delay_exposure_10-90_cov=0.3654 ttc3_10-90_cov=0.3462 delay_phys_minmax=0.2885 CRPS_phys=108.3 (L0 MAE_phys=738.77)
  L0@H60: deterministic delay_phys MAE=738.77 over 52 windows

== GATE v2 at H=30 (PRIMARY) ==
  L0: (a) coll/1000=0.0 PASS | (b) lc 0.0/12.077 (100.0%) FAIL | (c) n/a (deterministic) -> NA | (d) speed 4.1881 PASS ==> v2 FAIL: b_lc
  L3: (a) coll/1000=0.013 PASS | (b) lc 15.027/12.077 (24.4%) FAIL | (c) ttc3 0.5385 delay-phys 0.0769 crps 92.14 vs mae 293.39 -> FAIL | (d) speed 3.8008 PASS ==> v2 FAIL: b_lc,c_coverage_crps
  L4: (a) coll/1000=0.071 PASS | (b) lc 15.692/12.077 (29.9%) FAIL | (c) ttc3 0.3269 delay-phys 0.1346 crps 38.76 vs mae 293.39 -> FAIL | (d) speed 4.4044 PASS ==> v2 FAIL: b_lc,c_coverage_crps

== GATE v2 at H=60 (secondary) ==
  L0: (a) coll/1000=0.0 PASS | (b) lc 0.0/23.942 (100.0%) FAIL | (c) n/a (deterministic) -> NA | (d) speed 4.9774 PASS ==> v2 FAIL: b_lc
  L3: (a) coll/1000=0.008 PASS | (b) lc 29.292/23.942 (22.4%) FAIL | (c) ttc3 0.3846 delay-phys 0.0192 crps 289.74 vs mae 738.77 -> FAIL | (d) speed 4.6886 PASS ==> v2 FAIL: b_lc,c_coverage_crps
  L4: (a) coll/1000=0.043 PASS | (b) lc 29.431/23.942 (22.9%) FAIL | (c) ttc3 0.3462 delay-phys 0.1923 crps 108.3 vs mae 738.77 -> FAIL | (d) speed 5.1457 PASS ==> v2 FAIL: b_lc,c_coverage_crps

== OLD gate at H=30 ==
  L0: speed 4.1881 < 4.97 True | coll 0.0 True | lc 100.0% False | delay minmax None False ==> FAIL
  L3: speed 3.8008 < 4.97 True | coll 0.013 True | lc 24.4% False | delay minmax 0.1154 False ==> FAIL
  L4: speed 4.4044 < 4.97 True | coll 0.071 True | lc 29.9% False | delay minmax 0.1731 False ==> FAIL

== OLD gate at H=60 ==
  L0: speed 4.9774 < 4.97 False | coll 0.0 True | lc 100.0% False | delay minmax None False ==> FAIL
  L3: speed 4.6886 < 4.97 True | coll 0.008 True | lc 22.4% False | delay minmax 0.0385 False ==> FAIL
  L4: speed 5.1457 < 4.97 False | coll 0.043 True | lc 22.9% False | delay minmax 0.2885 False ==> FAIL

wrote D:\rd-wt\wd01e\output\wd01\part3k\results\gate_v2_p3k.json
```

## NOT DONE / residual gaps

1. **Merge-zone occupancy still under-predicted**: even at the fitted
   acceptance, L4 accel-lane zone occupancy remains below truth (see
   the fit/decomposition stdout).  A stopped waiting vehicle accepts
   any gap >= its minGap (2.5 m for normal cars); reproducing SUMO's
   longer queues likely needs a stopped-merge urgency/penalty term
   (e.g. lcStrategic-style urgency or a d-dependent minGap) -- not
   fitted here.
2. **Optional M2 for 0<->1** (gap acceptance on discretionary main-lane
   changes) was NOT implemented (brief: optional); 0<->1 stays on the
   stochastic hazard (M4).
3. **Parameter bootstrap is coordinate-neighbourhood**, not a full
   refit per member (cost): members 1-4 refit tau_accept/b_accept/
   p_coop_scale one grid step around the stage winners on bootstrap
   resamples of the 4 train roots, lc_scale refined on the same grid.
4. **idm_sampled stale-llen defect** (merged vehicles dropped as
   exited) recorded, not fixed.
5. **Single process, below-normal priority, torch 2 threads** -- no
   parallelism; box shared with SUMO workers.
6. **Holdout untouched**: the registered holdout evaluation remains
   blocked until the owner opens it (no registration hash created).

## Artifacts

- Code: `part3k/decompose_p3k.py` (p3j copy + L4 + helpers),
  `part3k/fit_p3k.py`, `part3k/eval_p3k.py`, `part3k/aggregate_p3k.py`,
  `part3k/make_report_p3k.py`; engine changes in
  `part3/engine.py` (kind `idm_merge`); data-root override in
  `partA_data/pdl.py`, `part3/p3lib.py`, `part3/engine.py`.
- Tests: `part3/tests/test_p3_merge.py`,
  `partA_data/tests/test_data_root.py`, `part3k/tests/` (14 tests).
- Results: `part3k/results/` (fit_p3k.json, stdout_fit.txt,
  decomposition_p3k.csv, speed_profile_p3k.csv, stdout_decompose.txt,
  raw/raw_metrics.jsonl, gate_v2_p3k.json, stdout_aggregate.txt,
  LEDGER_DRAFT_P3K.json).
- Reconciliation: L0/L3 rollouts are the p3j configuration (same
  engine branch, same seeds); their 60 s-grid validation means must
  reproduce the p3j published values (delay_true 1300.74; delay_def
  L0 -731.41 / L3 -294.22) -- checked in the decomposition stdout
  section A/B footer.

*(This report is generated by make_report_p3k.py; all numbers are
pasted from the tee'd stdout files above.)*
