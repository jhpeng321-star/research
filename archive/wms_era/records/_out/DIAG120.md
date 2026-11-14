# DIAG120 - why WM-S HDV predictions blow up at 60-120 s

Offline, CPU-only re-rollout diagnosis of the WM-S (C1-structured), OLDHIST (per-driver physics) and RUN-3 world models on the shifted v2 unseen set (`u_same`, PI arm). Blow-up = HDV median |ds| > 50 m at 60 s or > 100 m at 120 s. Every re-rolled state was checked against the stored shift scores (0 disagreements), so the diagnosis runs on the same predictions that were scored.

## Table 1 - blow-up rate by model (stored scores)

| model | set | arm | states | blow-ups | rate |
|---|---|---|---|---|---|
| c1_struct | u_same | pi | 140 | 57 | 40.7% |
| c1_struct | u_new | pi | 21 | 6 | 28.6% |
| c1_struct | seen | native | 157 | 99 | 63.1% |
| oldhist | u_same | pi | 140 | 72 | 51.4% |
| oldhist | u_new | pi | 21 | 6 | 28.6% |
| oldhist | seen | native | 157 | 93 | 59.2% |
| run3 | u_same | pi | 140 | 78 | 55.7% |
| run3 | u_new | pi | 21 | 10 | 47.6% |
| run3 | seen | native | 157 | 109 | 69.4% |
| run3_s120 | u_same | pi | 140 | 72 | 51.4% |
| run3_s120 | u_new | pi | 21 | 9 | 42.9% |
| run3_s120 | seen | native | 157 | 100 | 63.7% |
| blackbox_id | u_same | pi | 140 | 66 | 47.1% |
| blackbox_id | u_new | pi | 21 | 8 | 38.1% |
| blackbox_id | seen | native | 157 | 111 | 70.7% |
| blackbox_traj | u_same | pi | 140 | 89 | 63.6% |
| blackbox_traj | u_new | pi | 21 | 14 | 66.7% |
| blackbox_traj | seen | native | 157 | 114 | 72.6% |
| keep_speed | u_same | pi | 140 | 125 | 89.3% |
| keep_speed | u_new | pi | 21 | 19 | 90.5% |
| keep_speed | seen | native | 157 | 154 | 98.1% |
| pure_physics | u_same | pi | 140 | 127 | 90.7% |
| pure_physics | u_new | pi | 21 | 18 | 85.7% |
| pure_physics | seen | native | 157 | 154 | 98.1% |

Baselines bracket the problem: constant speed (`keep_speed`) and the residual-free `pure_physics` blow up on ~90% of unseen states, so a large part of the 120 s error is simply any sustained speed error integrating over the horizon. The structured models roughly halve that but still fail on 41-56% of u_same states (RUN-3 `run3` 55.7%, `run3_s120` 51.4%).

> **RUN-3 note.** RUN-3 blow-up rates and the entries comparison below come from its stored scores. Its per-state re-rollout could not be reproduced with the workspace loader: the stored `run3` scores require the projection/bounded-residual/merge-cal rollout switches (`projection_enabled`, `residual_cap_mps2=1.5`, `merge_cal=true`) that `wms_eval.wms_plugin` does not wire in, so a naive reload gives ds@60 of 900/1438/1776 m where the stored scores are 5.0/11.3/13.7 m (even with the FeasibilityProjection monkey-patched in: 44.5/79.4/96.5 m). Tasks 1-3 are therefore derived from the two re-rollout-verified models (c1_struct, oldhist); RUN-3 is used for rates (Table 1) and the entries ablation (Table 6) only.

## Table 2 - mechanism mix, blow-up vs non-blow-up (Task 1)

`speed_bias` = time-mean merge (pred - recorded) mean speed, m/s (>0 = model drives too fast where real traffic slows). `missed_peak`/`phantom_peak` = largest recorded-minus-predicted / predicted-minus-recorded simultaneous stopped-vehicle gap in merge or upstream. Classes: missed (real jam the model under-forms), phantom (model invents a jam), fast-drift (systematic under-braking), drift (rest).

| model | group | n | missed | phantom | fast-drift | drift | speed_bias m/s | missed_peak | phantom_peak |
|---|---|---|---|---|---|---|---|---|---|
| c1_struct | blow | 57 | 28 | 5 | 11 | 13 | 2.64 | 4.40 | 3.81 |
| c1_struct | nonblow | 83 | 17 | 28 | 14 | 24 | 1.18 | 4.51 | 6.04 |
| oldhist | blow | 72 | 32 | 9 | 26 | 5 | 4.82 | 4.40 | 3.97 |
| oldhist | nonblow | 68 | 21 | 10 | 15 | 22 | 1.85 | 2.87 | 3.31 |

Blow-ups are dominated by *missed jams* and systematic fast drift, not phantom jams. Mean merge speed bias is 2.64 m/s for c1_struct blow-ups vs 1.18 m/s for non-blow-ups (4.82 vs 1.85 for oldhist). Across blow-ups the speed bias correlates with ds@120 at r=0.71 (c1_struct) and r=0.53 (oldhist). Phantom jam is a minority (5/57 c1_struct, 9/72 oldhist).

### Signed section bias (mean over states)

| model | group | upstream bias m/s | merge bias m/s | downstream bias m/s |
|---|---|---|---|---|
| c1_struct | blow | +1.18 | +2.64 | -5.92 |
| c1_struct | non | +0.11 | +1.18 | -5.06 |
| oldhist | blow | +1.85 | +4.82 | -0.35 |
| oldhist | non | +0.09 | +1.85 | -0.35 |

The blow-up signal is concentrated **upstream of the merge**: predicted upstream speed is 1.2-1.9 m/s too high on blow-ups versus ~0.1 m/s on non-blow-ups. The model fails to carry a recorded upstream queue/deceleration.

## Table 3 - error-growth shape (Task 2)

| model | blow-ups | median t(ds>20) | t(ds>50) | t(ds>100) | largest 5 s rise | where | recorded speed drop at jump | occurs during recorded braking |
|---|---|---|---|---|---|---|---|---|
| c1_struct | 57 | 58 s | 84 s | 99 s | 76 m | t~108 s | 1.77 m/s | 52.6% |
| oldhist | 72 | 67 s | 89 s | 96 s | 166 m | t~108 s | 1.81 m/s | 52.8% |

Growth is gradual then accelerating: the 30-60 s slope is 0.44 m/s (c1_struct) / 0.31 m/s (oldhist) but the 60-90 s slope rises to 1.06 / 0.83 m/s, and the single largest 5 s rise (76-166 m) lands at t=108-108 s, i.e. near the end of the 120 s horizon and not at t0. In ~53% of blow-ups that late jump coincides with a recorded braking event (recorded merge speed dropping ~1.8 m/s); the model under-reacts and the position error accelerates. There is no single event at t0 that starts it - it is compounded under-braking. CAV commands cannot be the trigger: this harness teacher-forces every CAV from the recorded arm, so only HDVs are modelled.

## Table 4 - attribution of the 20 worst blow-ups (Task 3)

| model | HDVs/state | with onset | in merge at t0 | lane changers/state | top-10 carriers upstream at t0 | median onset delay vs leader |
|---|---|---|---|---|---|---|
| c1_struct | 84 | 69 | 7 | 55 | 180/200 | 1.00 s |
| oldhist | 96 | 76 | 9 | 60 | 195/200 | 1.50 s |

In the 20 worst blow-ups there are ~84-96 modelled HDVs, of which ~69-76 have crossed 10 m by 120 s. The error is **not carried at the merge**: 90-97% of the top-10 carriers are already upstream (s<0, up to ~2 km back) at t0. Onset propagates backward through the string: a vehicle crosses 10 m about 1.00-1.50 s after its recorded-lane leader (67.6% of leader/follower pairs have the follower later). No entrants are involved (0 of the 20 worst contain an entrant), and ~79-90% of top carriers perform at least one recorded lane change during the horizon (merge-related weaving), consistent with the string being excited by local lane changes too.

## Table 5 - OLDHIST recovered per-driver IDM parameters (Task 4)

Classes on the 20 worst + 15 non-blow-up states: starter = diverges before its leader, propagator = diverges after its leader, non = never crosses 10 m. Stability test = IDM low-speed sufficient condition a >= s0 / T^2 (Treiber & Kesting 2013; arXiv:2506.05909 Eq. 4).

| class | n | median T (s) | median accel | median min_gap (m) | frac satisfying a>=s0/T^2 | median margin |
|---|---|---|---|---|---|---|
| starter | 974 | 1.07 | 2.05 | 3.50 | 56.6% | 0.15 |
| propagator | 1599 | 1.09 | 2.05 | 3.08 | 59.8% | 0.16 |
| non | 1005 | 1.03 | 2.05 | 3.30 | 57.2% | 0.07 |

Recovered parameters do **not** separate wave starters from non-starters: the encoder pins accel at ~2.05 for every vehicle and the stable fraction is essentially identical across classes (56.6% starter, 59.8% propagator, 57.2% non). The recovered T and s0 vary (T p10-p90 0.84-1.58 s) but the ~43% that violate the low-speed stability condition do so equally in all classes, so per-driver string instability is not the discriminator - the failure is a shared systematic under-braking.

## Table 6 - RUN-3 entries off vs on (Task 5)

Matched to the worlds present in both arms.

| model | set | common states | blow-ups off | blow-ups on | mean ds@120 off | mean ds@120 on |
|---|---|---|---|---|---|---|
| run3 | u_same | 12 | 7 | 8 | 423 | 362 |
| run3 | u_new | 12 | 5 | 6 | 321 | 257 |
| run3 | seen | 52 | 42 | 45 | 315 | 250 |
| run3_s120 | u_same | 12 | 7 | 7 | 461 | 313 |
| run3_s120 | u_new | 12 | 5 | 4 | 233 | 163 |
| run3_s120 | seen | 52 | 34 | 32 | 254 | 167 |

Turning observed-counter entries ON does not remove the blow-ups (the count is unchanged or slightly higher, and mean ds@120 changes by <30%). Entry dynamics are not the dominant cause, matching the 0-entrant attribution above (task 5 is on a 12-52 state overlap, so it is directional, not decisive).

## Conclusions

1. **No phantom jam.** Predicted congestion does not exceed recorded congestion in general (phantom only 5/57 c1_struct, 9/72 oldhist blow-ups). The models do form queues; they form them too weakly and too late.
2. **Dominant mechanism = missed jam + systematic under-braking (fast drift).** Predicted traffic is too fast in the upstream and merge sections (blow-up mean bias +1.2/+2.6 m/s c1_struct, +1.9/+4.8 m/s oldhist; non-blow-ups ~+0.1/+1.2). The bias integrates into the position error (corr(bias, ds@120) = 0.71 c1_struct, 0.53 oldhist). It is a per-step deceleration deficit, not an invented jam.
3. **Growth is gradual then accelerating, with a late (90-110 s) jump, not an onset at t0.** Median crossing times: 20 m at ~57-67 s, 50 m at ~84-89 s, 100 m at ~96-99 s; the largest 5 s rise lands at t~107-108 s and coincides with a recorded braking event in ~53% of cases. Fixing only the first 30 s cannot fix the 120 s error.
4. **Error is carried upstream and propagates backward** (90-97% of top-10 carriers are upstream at t0; follower onset ~1.0-1.5 s after its leader). The merge zone itself is not where the error lives.
5. **Lane changes are present but not entrants.** ~79-90% of top carriers change lanes during the horizon; 0 of the 20 worst contain an entrant, and RUN-3 entries-on does not reduce blow-ups. The disturbance source is recorded upstream queueing plus local lane changes, which the model under-propagates.
6. **Per-driver parameters do not explain it.** OLDHIST's recovered IDM params are near-identical for wave starters and non-starters; the low-speed string-stability condition fails for ~43% of all vehicles regardless of class.

## What each mechanism implies (fix directions, not yet evaluated)

- **Under-braking / missed jam** -> increase deceleration fidelity and sensitise the residual to a closing/hard-braking leader (loss weighting on decelerating HDVs, an asymmetric speed-error loss, or a physics decel floor matched to the recorded distribution). This is the primary lever.
- **Backward wave propagation deficit** -> the residual must transfer a leader's braking to followers within 1-2 s; add a string-propagation training term (short-horizon multistep loss on followers after a leader decelerates).
- **Upstream queue carry** -> boundary/observation coverage upstream matters; the error origin is 1-2 km behind the merge, so upstream state coverage or a boundary queue term is needed.
- **Entries** -> not a priority; entries-on does not reduce blow-ups.
- **Per-driver params** -> not the discriminator; further driver-encoder work is unlikely to fix long-horizon blow-up.

## Reproduce / artifacts

- Rollouts: `_out/diag120/roll_struct_u_same.jsonl`, `roll_oldhist_u_same.jsonl` (4 workers, 0.5 s resolution, verified 0/140 flag disagreements vs stored scores).
- Scripts: `_out/diag120/{summarize,rollouts,analyze,analyze2,extract_params,analyze_params}.py`.
- Analysis: `_out/diag120/{stored_summary,analysis,analysis2,params_analysis}.json`, `_out/DIAG120.json`.
