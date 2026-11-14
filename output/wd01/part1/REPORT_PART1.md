# WD-01 Part 1 - world-model identity (T1) and rollout drift (T2)

Generated 2026-09-25T16:55:35.826827+00:00. Zero simulation: no SUMO/TraCI process was started; every number replays logged data and cached states on CPU (torch 2.12.1+cpu).

## 1. Pinned inputs

| artifact | sha256 (start == end) |
| --- | --- |
| surviving vehicle_dynamics.pt | `ded83907b8d218574d3f5359c332408f17167da055574afbe29eaecb59d4a8f5` |
| pinned expectation | `ded83907b8d218574d3f5359c332408f17167da055574afbe29eaecb59d4a8f5` |
| dynamics_ensemble.pt (never written) | `d017d17e7d5ae6f722f2c78c702bc715f09b91631861749b45ce5e2d0802a934` |
| pinned expectation | `d017d17e7d5ae6f722f2c78c702bc715f09b91631861749b45ce5e2d0802a934` |

Both end-of-task checks re-run in section 9 and in the backup receipt.

## 2. Data provenance: tag -> regenerated-run map (88 tags)

- 88/88 tags resolved to regenerated runs; 8 retry1 replacements (the 8 owner-interrupted runs); 0 problems; 6 informational warnings (the v3 build manifest lacks loading for the 6 cv19 tags; the v2 MANIFEST values are recorded per tag).
- Per tag recorded in results/tag_run_map.json: physics_steps.jsonl and telemetry.jsonl sha256, dataset jsonl sha256 == original v2 MANIFEST output_sha256 == MIG-05 PLAN expected_output_sha256, and resolved run dir == PLAN regenerated_run_dir.

## 3. T1a - rebuilt tracks vs surviving tracks

- Exact-equal runs: **88/88**; 33,092,848 array elements compared array-by-array (v_self, v_leader, gap, lane_group, free, action, dv, t) plus meta; worst max-abs-diff 0.0.
- The regenerated 602 s prefixes reproduce the deleted originals exactly at the track level, so every later number is computed on data identical to the original build.

### 3.1 Per-run comparison (all 88)

| tag | split | exact | n_pairs | worst array | max abs diff |
| --- | --- | --- | --- | --- | --- |
| cv17_native | train | yes | 51117 | v_self | 0.0 |
| cv17_pi-external-u | train | yes | 51117 | v_self | 0.0 |
| cv17_retain | train | yes | 51117 | v_self | 0.0 |
| cv19_4107-native | train | yes | 35705 | v_self | 0.0 |
| cv19_4107-pi-external-u | train | yes | 35705 | v_self | 0.0 |
| cv19_4107-retain | train | yes | 35705 | v_self | 0.0 |
| cv19_4108-native | train | yes | 31442 | v_self | 0.0 |
| cv19_4108-pi-external-u | train | yes | 31442 | v_self | 0.0 |
| cv19_4108-retain | train | yes | 31442 | v_self | 0.0 |
| cv20_7001m2000-native | train | yes | 37430 | v_self | 0.0 |
| cv20_7001m2000-pi-external-u | train | yes | 37430 | v_self | 0.0 |
| cv20_7001m2000-retain | train | yes | 37430 | v_self | 0.0 |
| cv20_7002m2000-native | train | yes | 34085 | v_self | 0.0 |
| cv20_7002m2000-pi-external-u | train | yes | 34085 | v_self | 0.0 |
| cv20_7002m2000-retain | train | yes | 34085 | v_self | 0.0 |
| cv20_7003m2000-native | train | yes | 38462 | v_self | 0.0 |
| cv20_7003m2000-pi-external-u | train | yes | 38462 | v_self | 0.0 |
| cv20_7003m2000-retain | train | yes | 38462 | v_self | 0.0 |
| cv20_7004m2000-native | train | yes | 39118 | v_self | 0.0 |
| cv20_7004m2000-pi-external-u | train | yes | 39118 | v_self | 0.0 |
| cv20_7004m2000-retain | train | yes | 39118 | v_self | 0.0 |
| cv20_7005m2000-native | train | yes | 39052 | v_self | 0.0 |
| cv20_7005m2000-pi-external-u | train | yes | 39052 | v_self | 0.0 |
| cv20_7005m2000-retain | train | yes | 39052 | v_self | 0.0 |
| cv20_7006m2000-native | train | yes | 32781 | v_self | 0.0 |
| cv20_7006m2000-pi-external-u | train | yes | 32781 | v_self | 0.0 |
| cv20_7006m2000-retain | train | yes | 32781 | v_self | 0.0 |
| cv20_7007m2700-native | train | yes | 55764 | v_self | 0.0 |
| cv20_7007m2700-pi-external-u | train | yes | 55764 | v_self | 0.0 |
| cv20_7007m2700-retain | train | yes | 55764 | v_self | 0.0 |
| cv21_7008m2700-native | train | yes | 53600 | v_self | 0.0 |
| cv21_7008m2700-pi-external-u | train | yes | 53600 | v_self | 0.0 |
| cv21_7008m2700-retain | train | yes | 53600 | v_self | 0.0 |
| cv23_7010m2700-native | train | yes | 50181 | v_self | 0.0 |
| cv23_7010m2700-pi-external-u | train | yes | 50181 | v_self | 0.0 |
| cv23_7010m2700-retain | train | yes | 50181 | v_self | 0.0 |
| cv23_7012m2700-retain | holdout | yes | 60271 | v_self | 0.0 |
| cv24_7011m2700-retain | holdout | yes | 54904 | v_self | 0.0 |
| cv25_7011m2700-native | holdout | yes | 54904 | v_self | 0.0 |
| cv25_7011m2700-pi-external-u | holdout | yes | 54904 | v_self | 0.0 |
| cv25_7011m2700-retain | holdout | yes | 54904 | v_self | 0.0 |
| cv25_7012m2700-retain | holdout | yes | 60271 | v_self | 0.0 |
| cv26_7012m2700-wall | holdout | yes | 60271 | v_self | 0.0 |
| cv27_7012m2700-native | holdout | yes | 60271 | v_self | 0.0 |
| cv27_7012m2700-pi-external-u | holdout | yes | 60271 | v_self | 0.0 |
| cv27_7012m2700-retain | holdout | yes | 60271 | v_self | 0.0 |
| cv28_7101m2000-native | holdout | yes | 33026 | v_self | 0.0 |
| cv28_7101m2000-pi-external-u | holdout | yes | 33026 | v_self | 0.0 |
| cv28_7101m2000-retain | holdout | yes | 33026 | v_self | 0.0 |
| cv29_7301m2000-retain | train | yes | 34404 | v_self | 0.0 |
| cv30_7301m2000-native | train | yes | 34404 | v_self | 0.0 |
| cv30_7301m2000-pi-external-u | train | yes | 34404 | v_self | 0.0 |
| cv30_7301m2000-retain | train | yes | 34404 | v_self | 0.0 |
| cv30_7302m2000-native | train | yes | 39409 | v_self | 0.0 |
| cv30_7302m2000-pi-external-u | train | yes | 39409 | v_self | 0.0 |
| cv30_7302m2000-retain | train | yes | 39409 | v_self | 0.0 |
| cv30_7303m2000-native | train | yes | 31846 | v_self | 0.0 |
| cv30_7303m2000-pi-external-u | train | yes | 31846 | v_self | 0.0 |
| cv30_7303m2000-retain | train | yes | 31846 | v_self | 0.0 |
| cv30_7304m2700-native | train | yes | 54640 | v_self | 0.0 |
| cv30_7304m2700-pi-external-u | train | yes | 54640 | v_self | 0.0 |
| cv30_7304m2700-retain | train | yes | 54640 | v_self | 0.0 |
| cv30_7305m2700-native | train | yes | 55453 | v_self | 0.0 |
| cv30_7305m2700-pi-external-u | train | yes | 55453 | v_self | 0.0 |
| cv30_7305m2700-retain | train | yes | 55453 | v_self | 0.0 |
| cv30_7306m2700-native | train | yes | 53729 | v_self | 0.0 |
| cv30_7306m2700-pi-external-u | train | yes | 53729 | v_self | 0.0 |
| cv30_7306m2700-retain | train | yes | 53729 | v_self | 0.0 |
| cv31_7531m2700-native | train | yes | 57522 | v_self | 0.0 |
| cv31_7531m2700-pi-external-u | train | yes | 57522 | v_self | 0.0 |
| cv31_7531m2700-retain | train | yes | 57522 | v_self | 0.0 |
| cv31_7532m2700-native | train | yes | 58118 | v_self | 0.0 |
| cv31_7532m2700-pi-external-u | train | yes | 58118 | v_self | 0.0 |
| cv31_7532m2700-retain | train | yes | 58118 | v_self | 0.0 |
| cv31_7534m2700-native | holdout | yes | 63760 | v_self | 0.0 |
| cv31_7534m2700-pi-external-u | holdout | yes | 63760 | v_self | 0.0 |
| cv31_7534m2700-retain | holdout | yes | 63760 | v_self | 0.0 |
| cv31_7535m2400-native | train | yes | 50330 | v_self | 0.0 |
| cv31_7535m2400-pi-external-u | train | yes | 50330 | v_self | 0.0 |
| cv31_7535m2400-retain | train | yes | 50330 | v_self | 0.0 |
| cv31_7536m2400-native | holdout | yes | 49577 | v_self | 0.0 |
| cv31_7536m2400-pi-external-u | holdout | yes | 49577 | v_self | 0.0 |
| cv31_7536m2400-retain | holdout | yes | 49577 | v_self | 0.0 |
| cv32_7531m2700-wall | train | yes | 57522 | v_self | 0.0 |
| cv32_7532m2700-wall | train | yes | 58118 | v_self | 0.0 |
| cv32_7534m2700-wall | holdout | yes | 63760 | v_self | 0.0 |
| cv32_7535m2400-wall | train | yes | 50330 | v_self | 0.0 |
| cv32_7536m2400-wall | holdout | yes | 49577 | v_self | 0.0 |

## 4. T1b - surviving checkpoint re-evaluation

Protocol imported from the source verify module: 21 holdout runs x start ticks {120..480 step 60} = 147 evals; one-step fixed-mask 50 m x 4-group field RMSE (model raw / persistence / field-copy); per-vehicle |dv error| vs zero-dv; 60-step rollout on the logged congested run with leaders recomputed from predicted states.

| metric | logged | re-evaluated | abs dev | <= 1e-3 |
| --- | --- | --- | --- | --- |
| mean_model_rmse | 2.169141 | 2.169141 | 0.0e+00 | yes |
| mean_model_rmse_raw | 2.169568 | 2.169568 | 0.0e+00 | yes |
| pooled_model_rmse | 2.690901 | 2.690901 | 0.0e+00 | yes |
| pooled_model_rmse_raw | 2.691288 | 2.691288 | 0.0e+00 | yes |
| mean_persistence_rmse | 2.324604 | 2.324604 | 0.0e+00 | yes |
| pooled_persistence_rmse | 2.788778 | 2.788778 | 0.0e+00 | yes |
| mean_field_copy_rmse | 1.109665 | 1.109665 | 0.0e+00 | yes |
| dv_mae_model | 0.137061 | 0.137061 | 0.0e+00 | yes |
| dv_mae_zero | 0.329364 | 0.329364 | 0.0e+00 | yes |
| n_dv_pairs | 14583.000000 | 14583.000000 | 0.0e+00 | yes |
| n_evals | 147.000000 | 147.000000 | 0.0e+00 | yes |
| per-eval rows (6 fields x 147) | 882 | 882 | 0.0e+00 (max) | yes |
| rollout horizon rows (7 fields x 61) | 427 | 427 | 0.0e+00 (max) | yes |
| rollout max RMSE truth-occupied (m/s) | 22.345532 | 22.345532 | | |
| rollout max RMSE start-fixed (m/s) | 21.981702 | 21.981702 | | |

Every logged one-step number, all 147 per-eval rows and all 61 rollout horizon rows reproduce with zero deviation. The v3 logged metrics are fully accounted for by the surviving checkpoint on regenerated data.

## 5. T1c - retrain-and-compare identity check

- Retrained from byte-exact copies of the 67 surviving train npz with the byte-exact trainer: 29 s, 15 epochs, 499,954 pairs (67 runs x 7,462 cap; numpy seed 0, torch seed 0; train MSE 0.10324 vs zero-dv 0.39409, matching the logged 0.1032/0.3941).
- Weights: 6/6 tensors bit-identical, max abs diff 0.0; prediction agreement on the 14,583 holdout pairs: mean abs diff 0.0, Pearson 1.000000.

| quantity | surviving | retrained | registered tolerance | verdict |
| --- | --- | --- | --- | --- |
| one-step fixed-mask RMSE (m/s) | 2.169141 | 2.169141 | within 0.02 of logged and < persistence 2.324604 | pass |
| per-vehicle dv MAE (m/s) | 0.137061 | 0.137061 | within 0.005 of logged | pass |

**IDENTITY VERDICT: PASS.**

Modern retrained checkpoint: `D:\rd-wt\wd01\output\wd01\part1\identity\retrained\vehicle_dynamics_v3_retrained.pt` sha256 `b5836346d0d615f1c0433bcfa87d6b33f48d98e8d729c0117ca44e7669042db1` (85533 B); state_dict + plain metadata; torch.load(weights_only=True) verified true.

## 6. T2 - rollout drift (protocol: REGISTRATION.md)

Registration sha256 `484ee7857be11f6b4cb73f8344ff4233766481075dac502072de302005d665c5` recorded 2026-09-25T16:35:12Z, verified unchanged at scoring time. 21 holdout runs x 7 start ticks = 147 evals per horizon; vehicle set frozen at t0 (no births; lane group fixed); leaders recomputed each step within lane group from predicted states; RAW = original propagate (v >= 0 clamp only); CONSTRAINED = accel clip [-9, +3] m/s2, speed clip [0, 1.2 x train lane-group max], no-pass-through with 5 m behind-leader reset, overlaps counted not hidden.

Train lane-group max v (m/s): g0=33.79, g1=33.79, g2=33.79, g3=24.00; v_cap = 1.2x: g0=40.55, g1=40.55, g2=40.55, g3=28.80.

### 6.1 Verdicts per horizon x variant (means over 147 evals)

| variant | H | speed RMSE @H | persistence RMSE @H | TTC<3s rec | TTC<3s prec | brake rec | brake prec | capviol/1000 | overlap/1000 | max v | C1 | C2 | C3 | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RAW | 30 | 3.948 | 7.389 | 0.083 | 0.081 | 0.378 | 0.398 | 0.00 | 0.00 | 33.4 | F | P | P | **FAIL** |
| RAW | 60 | 5.799 | 10.347 | 0.053 | 0.064 | 0.215 | 0.257 | 0.00 | 0.00 | 34.5 | F | P | P | **FAIL** |
| RAW | 90 | 9.532 | 13.740 | 0.044 | 0.065 | 0.140 | 0.193 | 0.00 | 0.00 | 35.0 | F | P | P | **FAIL** |
| RAW | 120 | 13.333 | 16.990 | 0.036 | 0.065 | 0.104 | 0.175 | 0.00 | 0.00 | 35.3 | F | P | P | **FAIL** |
| CONSTRAINED | 30 | 4.710 | 7.389 | 0.138 | 0.146 | 0.432 | 0.354 | 0.00 | 15.76 | 33.4 | F | F | P | **FAIL** |
| CONSTRAINED | 60 | 5.764 | 10.347 | 0.079 | 0.097 | 0.267 | 0.233 | 0.00 | 12.19 | 34.5 | F | F | P | **FAIL** |
| CONSTRAINED | 90 | 9.211 | 13.740 | 0.058 | 0.083 | 0.189 | 0.191 | 0.00 | 10.46 | 35.0 | F | F | P | **FAIL** |
| CONSTRAINED | 120 | 13.721 | 16.990 | 0.045 | 0.080 | 0.141 | 0.182 | 0.00 | 8.61 | 35.5 | F | F | P | **FAIL** |

C1 = recall >= 0.80 for both event types; C2 = zero vehicle-steps above 1.2 x lane-group train max AND overlaps <= 1/1000 vehicle-steps; C3 = speed RMSE @H <= persistence RMSE @H.

Lost vehicle-steps (truth lane change or exit), fraction at H=120: RAW 0.486, CONSTRAINED 0.486. Per-run breakdown: results/t2_per_run.csv; metrics vs step: results/t2_per_step.csv; H x variant table: results/t2_per_horizon.csv.

### 6.2 Recommendation: NO registered horizon passes

First failing criteria by horizon (CONSTRAINED): {'30': ['C1', 'C2'], '60': ['C1', 'C2'], '90': ['C1', 'C2'], '120': ['C1', 'C2']}. Fallback bound: None. C1 (safety-event recall) is the binding failure at every horizon for both variants - predicted tracks are smoother than truth (the model's dv MAE is 0.137 m/s but TTC/hard-brake seconds are rare, noise-dominated events it does not reproduce co-timed). CONSTRAINED additionally fails C2: the no-pass-through reset fires 15.8/1000 vehicle-steps at H=30 (registered bound 1/1000) because 1 s steps with +-0.14 m/s dv noise routinely invert sub-metre orderings in dense congestion. RAW satisfies C2 (zero overlaps, zero cap violations, max speed 35.3 m/s < 40.6 cap) and C3 at every horizon - the native-dynamics rollout does not run away from truth states, in contrast to the WM-30 hybrid-sample failure (42 m/s), which came from reconstructor hybrid states, not truth states.

Practical reading (outside the registered rule): the drift-accuracy headroom (C3) extends to at least 120 s, but safety-event fidelity does not exist at any registered horizon, so the action-conditioned Part A must add event-level fidelity (multi-step loss, event heads, 0.5 s step) rather than a longer open-loop budget. Part A horizon for WD-01 should be chosen on WD-36 branch data, re-measuring all three criteria.

Prediction throughput: 149534 vehicle-steps/s of model forward on CPU (3,568,800 vehicle-steps in 23.9 s). Device resolution is one-time and CUDA-ready, but CUDA is untested (no GPU attached).

## 7. Limitations

- Action-blind window: in-window (60-600 s) physics is arm-independent, so this is a native-dynamics baseline; the horizon does not transfer to post-switch or wall conditions.
- No lane-change or birth model: rollout vehicles keep their t0 lane group forever; truth lane changes/exits censor vehicle-steps (fraction reported, not predicted); TTC leader sets ignore cross-lane interactions.
- 1 s step is coarse for TTC: events scored at the registered 3 s threshold; the contract's TTC<1.5 s gating needs the Part A 0.5 s step.
- Provenance: T1b/T1c/T2 run on regenerated 602 s prefixes; T1a proves array-exact equality with the surviving tracks and dataset bytes hash-match the original manifest, but the original raw run trees remain deleted (MIG-04).
- Bit-identity of the retrain holds for this torch/numpy build; the registered tolerances cover the non-bit-identical case regardless.

## 8. Historical note

The WM-12/v3-postswitch arm-aware result (0.1228 vs 0.1364) stands as historical evidence that action identity is informative; its checkpoint and training data were lost in MIG-04; reproduction waived by owner (Q1=b, 2026-09-25).

## 9. End-of-task integrity

- vehicle_dynamics.pt sha256 at end: `ded83907b8d218574d3f5359c332408f17167da055574afbe29eaecb59d4a8f5`
- dynamics_ensemble.pt sha256 at end: `d017d17e7d5ae6f722f2c78c702bc715f09b91631861749b45ce5e2d0802a934`
- REGISTRATION.md sha256 at end: `484ee7857be11f6b4cb73f8344ff4233766481075dac502072de302005d665c5`
- Backups + SHA256SUMS.txt: D:/research-dense-backups/wd01/part1/ (see MANIFEST_PART1.json).

## 10. Execution evidence (commands + wall-clock)

All steps: `C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe <script>` run from `output/wd01/part1/` (Git Bash on win32); no SUMO or TraCI process at any point.

| step | command | wall-clock |
| --- | --- | --- |
| tag->run map | `python build_tag_map.py` | 104 s |
| T1a rebuild+compare | `python rebuild_tracks.py` (8 workers) | 127 s total; 0 worker-seconds |
| T1b re-evaluation | `python identity_eval.py surviving` | 5 s |
| T1c retrain | `python retrain.py` (trains via byte-exact copy) | train 29 s of 29 s pipeline |
| T2 rollouts+scoring | `python t2_run.py` | 275 s total; model forward 23.9 s |
| pytest | `python -m pytest tests/ -q` | 2.1 s, 18 passed |
| backup | `python backup.py` | 33 files, SHA256SUMS verified |

## 11. Files

