# GAPRULE_NOTES: speed-dependent gap acceptance for the physics lane-change mask (C4 WM)

Branch `c4/wmgap` (from c4/int 683e83f). Switch: `model.lane_gap_rule = "fixed" | "speed"`
(default `"fixed"`, bit-identical). Config `configs/c4_wm_v3.toml` = `c4_wm_v2.toml` + the
rule on (plus its own `runtime.out_dir`); nothing else changes. No SUMO was run.

## Problem
In `learned_discretionary` mode the hard feasibility mask (`wms.physics.mobil`) needed both
target-lane gaps >= 6 m (`lane_safe_gap_m`, or the driver `merge_gap`) and an IDM follower
response >= -4..-6 m/s2. On wall_same it vetoes 567/952 recorded SUMO changes (WMLANE_NOTES.md),
so jam overtakes and jam merges are impossible and imagined merge queues last too long.

## Design (`lane_gap_rule = "speed"`)
`speed_gap_safe` (src/wms/physics/mobil.py) replaces the `safe` test for discretionary
changes and the ramp merge 2 -> 1 alike, in the form of SUMO LC2013's secure gap:

    lead:   gap_l >= s0_lead + T_lead * v   + max(0, v^2  - v_l^2) / (2 b_lead)
    follow: gap_f >= s0_foll + T_foll * v_f + max(0, v_f^2 - v^2)  / (2 b_foll)
    and no target-lane car overlapping the ego footprint (length-aware, 0.5 m margin)

* The kinematic term is the follower-braking limit (replaces the IDM follower-braking check,
  which explodes at jam gaps: IDM wants < -15 m/s2 at 2 m / 2.4 m/s). b_foll = 6 m/s2.
* **Judged one step ahead** (`gap_anticipation_s = 0.5`, every car extrapolated at its current
  acceleration, `anticipated_kin`). Reason found in the data: SUMO moves all cars and THEN
  changes lanes, and the WM applies a lane change together with the same step's motion, so the
  gap that matters is the gap after the step. At the pre-step state 20 % of the recorded TRAIN
  changes have a target-lane car side by side (typically a 30 m/s car passing a slow ego
  that is 10 m ahead 0.5 s later); without anticipation no admissible rule exceeds 0.77 mean
  recall (see table), with it 0.95.
* Unchanged: non-adjacent targets, changes into the ramp lane, ramp merges outside the zone
  [0, 75.44] m stay masked; the forced end-of-lane merge window (hazard model, last 25 m) and
  the learned-residual / hazard re-mask are untouched; MOBIL incentive/politeness terms unchanged.
* In speed mode the driver `merge_gap` is not used by the mask (population rule; drivers stay
  learned from prediction error only, no labels added). `merge_cal` (off in C4) and the
  `history` lane head keep their own acceptance rules.
* Constants live on `PhysicsConfig.gap_*` (written into physics.json, so checkpoints reload
  with their rule); TOML only selects the rule. `fixed` mode never reads them.

## Constants: TRAIN only (tools/gaprule_fit.py)
`collect`: every TRAIN window of configs/c4_wm.toml (2041 windows: pack_v3 native + pack_dspeed
+ switching direct-mix TRAIN part; no wall_same / held-out / CAL list), every HDV step x legal
adjacent target; all recorded changes (36,770: merge 7,466 / overtake 18,870 / return 10,434),
non-changes subsampled 5 % (weight 20). **Junction-transit rows are dropped** (see "Looked
wrong"). `fit`: grid (s0 0.5-3 m, T 0-1 s, b 2-6 m/s2 per side, 14,400 rules); admissible =
admits <= 2 % of the DANGEROUS non-change states; chosen = the most conservative admissible rule
(smallest feasible-but-unused rate) within 1 point of the best category-averaged recall.

DANGEROUS (rule-independent, judged on the post-step configuration): overlap, or bumper gap
< 1 m, or > 4.5 m/s2 constant decel needed by follower/ego to avoid contact, or time headway
< 0.3 s above 5 m/s. It flags 5.3 % of SUMO's own recorded TRAIN changes (calibration check).

Chosen: **lead 1.0 m + 0.2 s * v + dv-term b 3.0 m/s2; follow 0.5 m + 0.6 s * v_f + b 6.0 m/s2**
(s0_follow and b_follow sit on the grid edge: 0.5 m is the physical floor above the 0.3 m gap
clamp; b_follow 6 equals the old urgent-merge braking limit).

TRAIN (docs/gaprule/train_fit.json, train_fit_timing_t.json):

| rule | recall merge | overtake | return | all | feasible non-change | P(feasible \| dangerous) |
|---|---|---|---|---|---|---|
| old fixed (default params) | 0.296 | 0.607 | 0.729 | 0.578 | 0.355 | 0.065 |
| speed, judged at t (best admissible) | 0.854 | 0.623 | 0.793 | 0.718 | 0.292 | 0.013 |
| **speed, anticipated (chosen)** | **0.975** | **0.921** | **0.939** | **0.937** | **0.334** | **0.017** |

## Diagnostic on wall_same (not used for tuning; tools/wmlane_mask_reasons.py, default params)
`docs/gaprule/mask_reasons_{fixed,speed}.json`. Masked = the change's target logit is masked
on the pre-change state t; "t and t+1" = also masked one step later (t+1 state, ego put back
into its origin lane).

| category | recorded | masked old | masked new | masked t&t+1 old | masked t&t+1 new |
|---|---|---|---|---|---|
| overtake 1->0 | 621 | 386 (62 %) | 49 (8 %) | 215 | 0 |
| return 0->1 | 171 | 29 (17 %) | 35 (20 %) | 18 | 20 |
| merge 2->1 | 160 | 152 (95 %) | 44 (28 %) | 144 | 9 |
| **all** | **952** | **567 (59.6 %)** | **128 (13.4 %)** | **377 (39.6 %)** | **29 (3.0 %)** |

New vetoes: overtakes lead gap < requirement (median lead gap 0.75 m); merges overlap at the
anticipated state (all 44 pass one step later except 9); returns: follower gap 1.25 m median at
28 m/s (25 follow, 10 lead) - SUMO does these (27/171 returns are DANGEROUS by the definition),
the rule deliberately does not.

False-feasible side (HDV steps whose car did NOT change, x legal adjacent targets, 675,932):

| | feasible old | feasible new | feasible & dangerous old | new | P(feasible \| dangerous) old | new |
|---|---|---|---|---|---|---|
| merge (7,658) | 3.2 % | 15.8 % | 0.34 % | 0.26 % | 0.43 % | 0.33 % |
| overtake (370,927) | 24.8 % | 27.8 % | 1.99 % | 1.40 % | 3.4 % | 2.4 % |
| return (297,347) | 45.5 % | 37.7 % | 4.83 % | 0.53 % | 10.9 % | 1.2 % |
| **all** | **33.7 %** | **32.0 %** | **3.22 %** | **1.00 %** | **6.1 %** | **1.9 %** |

So: most recorded changes become feasible (87 % at t, 97 % within one step), the overall
feasible-but-unused rate does not grow (33.7 -> 32.0 %; merges 3 -> 16 %, which is the intended
jam-merge opening), and dangerous cut-ins are admitted 3x less often than by the old mask.

## Looked wrong (not fixed here)
* **TRAIN packs encode junction transits as lane-0 rows with a frozen `s` and `loss_valid`
  True** (pack_v3/dspeed windows via `TrajectoryDataset`; s held at ~0 or ~75.4 m while v
  grows, lane one-hot [1,0,0]). They create fake 1->0->1 "changes" at the edge boundaries
  (~70 % of all raw recorded lane changes in the TRAIN windows: 121k raw vs 36.8k after
  dropping frozen-s rows) and fake lane-0 neighbours. wall_same (lane -1, forward-filled) is
  barely affected (31/952). This likely inflates the WM's lane loss / lane_event targets and
  the flip-rate statistics; worth a separate data fix.
* SUMO itself makes ~5 % changes that are physically tight in the WM frame (nominal 5 m
  lengths, 0.5 s sampling): those stay masked by design.

## Tests / smoke
OMP_NUM_THREADS=1: `tests/unit/test_wmlane.py test_lane_rule.py test_wmloss.py test_c4_core.py
test_import_graph.py test_gaprule.py` -> **283 passed**; ruff clean on the touched files.
CPU smoke `tools/c4_smoke.py E:/agent-work/tmp/gaprule_smoke_v3 configs/c4_wm_v3.toml` (data via
`wmlane_smoke.point_data(E:/research-dense)`): **passed**, 2 optimizer steps, 10 s unroll, batch 4,
finite, loss 14.95 -> 11.22, 2041 TRAIN / 261 val windows, physics lane_gap_rule = speed.

`tests/unit/test_gaprule.py` (22): fixed-mode mask
bit-identity vs golden numbers captured with the pre-GAPRULE mobil.py on a random dense scene
(3 lane rules x with/without driver params), one C4 training step bit-identical to the
pre-WMLANE golden with the switch absent and explicit "fixed" (physics built through the
config), jam merge 4 m / 0 m/s open (also with driver params), jam overtake open, overlap
(ds 0, +-1, +-4.9 m), non-adjacent, ramp entry, outside-zone, fast-follower and fast-into-
standing-leader vetoes, monotonicity in the gap, anticipation (side-by-side passer), config
plumbing (v3 = v2 + rule only, flat roundtrip, settings, build_physics, physics.json
roundtrip), a speed-mode training step finite with gradients.
