# LANERULE_CHECK — physics-only lane-rule comparison on wall_same (67 states)

Owner decision 2026-10-12 (WM-S-LANECHECK-20261012: MOBIL matches only 1.5% of
recorded changes; 80% of recorded changes never proposed; overtake-category cars carry
75% of the 120 s wrong-lane error): drop discretionary MOBIL changes, keep only the
forced ramp merges. This check measures what that switch does to the physics-only
baseline and to the R5b best checkpoint.

## Command

```
cd E:/agent-work/c4-wm
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe tools/lanerule_check.py --workers 11
```

(one process pool per variant; per-state rows and `results.json` under
`runs/active/LANERULE-CHECK-20261012/`, summaries only)

## Setup

- Set: `E:/research-dense/data_cache/wallc1_states_h60/wall_same.json` — all 67 states,
  entries as in R5 scoring (entries OFF, CAVs teacher-forced, seed 0, dt 0.5 s, 240 steps).
- Physics-only variants: R4 checkpoint scaffold (`WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s.pt`,
  physics from its `physics.json`), residual gate forced to zero, **default driver
  settings** (`driver_params=None`), `lane_rule` switched on the physics object.
- Diagnostic variants: R5b best checkpoint
  (`WM-S-R5B-20261012/box/gpuA/r5b_main_a1/wm_s_best.pt`) with the learned
  residual/gate intact, `lane_rule` switched **at inference only**.
- Metrics as in R5 scoring: median over the 67 states of the per-state median HDV |ds|;
  blow-up = state HDV ds_med > 50 m at 60 s AND > 100 m at 120 s; wrong-lane share =
  pooled HDV slots (predicted present, recorded present, recorded non-CAV, recorded lane
  valid) with predicted lane != recorded lane; lane rate = predicted lane changes per
  HDV-step over 0-120 s.

## Results

| variant | 10 s | 30 s | 60 s | 120 s | blow-ups | wrong-lane 60 s | wrong-lane 120 s | lane changes/HDV-step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phys MOBIL (default) | 3.72 | 10.17 | 15.68 | 53.12 | 7/67 | 0.101 | 0.234 | 0.000796 |
| phys forced_merge_only | 3.72 | 9.84 | 15.68 | 59.28 | **3/67** | **0.085** | **0.208** | **0.000221** |
| phys recorded lanes (lane_teacher) | **3.35** | **8.44** | **10.13** | **10.70** | 3/67 | 0.000 | 0.000 | 0.001407 |
| R5b best (as trained, MOBIL) — diagnostic | 1.72 | 4.47 | 12.48 | 116.86 | 1/67 | 0.081 | 0.288 | 0.003653 |
| R5b best, forced_merge_only at inference — diagnostic | 1.70 | 4.23 | 11.75 | 115.68 | 1/67 | 0.084 | **0.208** | **0.000197** |

## Reading

- **Reproduction gate**: `phys MOBIL` is exactly the published physics-only reference
  (3.72/10.17/15.68/53.12 m, 7 blow-ups; CHK-TYPEID/DIAG-ORACLE), so the `lane_rule`
  default is bit-neutral on real data. `R5b best (MOBIL)` reproduces the R5 diagnosis
  base numbers (12.48/116.86 m), confirming the harness.
- **forced_merge_only (physics-only)**: 10/30/60 s medians essentially unchanged
  (short-horizon error is longitudinal, not lane, error); 120 s median slightly worse
  (53.1 -> 59.3 m) but blow-ups drop 7 -> 3, wrong-lane share drops at both horizons
  (0.101 -> 0.085 at 60 s, 0.234 -> 0.208 at 120 s) and the predicted lane-change rate
  falls to 0.000221/HDV-step (recorded TRUE rate ~0.0018/HDV-step on this set; MOBIL
  intents were 77% early among ramp changes with a candidate, per LANECHECK).
- **Recorded lanes (perfect lanes)** cut the 120 s median by 80% (53.1 -> 10.7 m) and
  eliminate wrong-lane error entirely: lane identity dominates long-horizon physics-only
  error, consistent with the LANECHECK attribution.
- **R5b diagnostic**: switching `lane_rule` at inference keeps the learned model's
  short-horizon accuracy (1.72/4.47 -> 1.70/4.23 m), improves 60 s (12.48 -> 11.75 m),
  trims 120 s (116.9 -> 115.7 m), cuts the wrong-lane share at 120 s from 0.288 to
  0.208 and the lane-change rate by ~18x (0.003653 -> 0.000197/HDV-step). Diagnostic
  only — C4 trains with the rule on, it does not patch it in at inference.

Provenance: code `tools/lanerule_check.py` (branch `c4/wm`); read-only inputs from
`E:/research-dense` (data_cache states, R4/R5b checkpoints). No GPU, no SUMO; CPU only.
